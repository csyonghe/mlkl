#include "conditioned-unet.h"

#include "kernels.h"

#include <vector>

ConditionedUNet::ConditionedUNet(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int outChannels,
    int tDim,
    int cDim,
    int baseCh,
    int nClasses)
    : context(ctx)
    , inputChannels(inChannels)
    , outputChannels(outChannels)
    , timeEmbedDim(tDim)
    , contextDim(cDim)
    , baseChannels(baseCh)
    , classCount(nClasses)
{
    // Standard Config: Multipliers [1, 2, 4] -> Channels [64, 128, 256]
    channelMultipliers = {1, 2, 4};

    // 1. Time and class Embedding
    timeEmbed = new TimeEmbedingKernel(ctx, timeEmbedDim);
    classEmbed = new GatherKernel(ctx, classCount, contextDim);

    // 2. Initial Conv
    initialConv =
        new Conv2DKernel(ctx, 16, 3, 1, inputChannels, baseChannels, kernelOutput(), "InitConv");

    // 3. Down Blocks
    // Block 0: 1 -> 2 (64 -> 128)
    // Block 1: 2 -> 4 (128 -> 256)
    for (int i = 0; i < channelMultipliers.getCount() - 1; i++)
    {
        int inMult = channelMultipliers[i];
        int outMult = channelMultipliers[i + 1];

        downBlocks.add(new UNetBlock(
            ctx,
            UNetBlockKind::Down,
            baseChannels * inMult,  // In Channels
            baseChannels * outMult, // Out Channels
            timeEmbedDim));
    }

    // 4. Mid Block (Bottleneck Attention)
    // Input to mid is the output of the last down block (e.g., 256 channels)
    int bottleneckChannels = baseChannels * channelMultipliers[channelMultipliers.getCount() - 1];
    midAttn = new CrossAttentionKernel(ctx, bottleneckChannels, contextDim, 64);

    // 5. Up Blocks
    // We traverse multipliers in reverse: [4, 2, 1]
    // The "input" to an Up block is the concat of (Current, Skip).
    // The "Skip" comes from the corresponding Down block.
    // Block 0: 4 -> 2 (256 -> 128). Input Concat: 256(skip) + 256(up) = 512.
    // Block 1: 2 -> 1 (128 -> 64).  Input Concat: 128(skip) + 128(up) = 256.
    for (int i = channelMultipliers.getCount() - 1; i > 0; i--)
    {
        int inMult = channelMultipliers[i];
        int outMult = channelMultipliers[i - 1];

        // Note: The 'inChannels' for UNetBlock::Up usually refers to the
        // *result* of the operation.
        // But physically, the first Conv2D in the block takes (inCh + skipCh).
        // In this symmetric architecture, inCh == skipCh.
        // So the UpBlock will see (base*inMult) + (base*inMult) channels at input.
        // We pass the "logical" channel sizes of the layer level.

        upBlocks.add(new UNetBlock(
            ctx,
            UNetBlockKind::Up,
            baseChannels * inMult,  // 256, then 128
            baseChannels * outMult, // 128, then 64
            timeEmbedDim));
    }

    // 6. Final Conv
    // Output of last Up block has `baseChannels` (64).
    finalConv =
        new Conv2DKernel(ctx, 16, 1, 1, baseChannels, outputChannels, kernelOutput(), "FinalConv");

    concat = new ConcatKernel(ctx, 2);
}

SlangResult ConditionedUNet::loadParams(TorchParamReader& reader)
{
    // 1. Time MLP
    SLANG_RETURN_ON_FAIL(timeEmbed->loadParams(reader));

    // 2. Class Embedding
    SLANG_RETURN_ON_FAIL(classEmbed->loadParams(reader));

    // 3. Init Conv
    SLANG_RETURN_ON_FAIL(initialConv->loadParams(reader, false)); // No bias usually? Check model.

    // 4. Down Blocks
    for (auto& block : downBlocks)
    {
        SLANG_RETURN_ON_FAIL(block->loadParams(reader));
    }

    // 5. Mid Attention
    SLANG_RETURN_ON_FAIL(midAttn->loadParams(reader));

    // 6. Up Blocks
    for (auto& block : upBlocks)
    {
        SLANG_RETURN_ON_FAIL(block->loadParams(reader));
    }

    // 7. Final Conv
    SLANG_RETURN_ON_FAIL(finalConv->loadParams(reader, false));

    return SLANG_OK;
}

void ConditionedUNet::queueExecute(
    InferencingTask& task,
    TensorView outputImage,
    TensorView inputImage,
    TensorView classLabels,
    int timeStep)
{
    context->pushAllocScope();
    SLANG_DEFER(context->popAllocScope());

    SLANG_ASSERT(inputImage.shape.getRank() == 4);

    int batchSize = inputImage.shape[0];
    int inputHeight = inputImage.shape[1];
    int inputWidth = inputImage.shape[2];

    // 1. Time Embedding
    auto tEmb = timeEmbed->allocateResultBuffer(outputImage.elementType, batchSize);
    timeEmbed->queueExecute(task, tEmb, timeStep);

    // 2. Context embedding
    auto contextEmbedding = classEmbed->allocateResultBuffer(outputImage.elementType, batchSize);
    classEmbed->queueExecute(task, contextEmbedding, classLabels);

    // 3. Init Conv
    auto x =
        initialConv
            ->allocateResultBuffer(outputImage.elementType, inputWidth, inputHeight, 1, batchSize);
    initialConv->queueExecute(task, x, inputImage, 1);

    // 4. Down Blocks
    ShortList<TensorView> skipConnections;
    int w = inputWidth;
    int h = inputHeight;

    for (auto& block : downBlocks)
    {
        // Down Block Output
        auto out = block->allocateResultBuffer(outputImage.elementType, w, h, batchSize);
        block->queueExecute(task, out, x, tEmb);

        skipConnections.add(out);
        x = out;

        // Block includes stride, so dimensions halve
        w /= 2;
        h /= 2;
    }

    // 5. Mid Attention (Bottleneck)
    // x is now [Batch, H/4, W/4, 256] in NHWC
    int seqQ = w * h;
    int channels = baseChannels * channelMultipliers[channelMultipliers.getCount() - 1];
    int seqKV = 1; // Context is [Batch, 1, contextDim]
    int heads = 4;

    // Alloc output
    auto attnOut =
        midAttn->allocateResultBuffer(outputImage.elementType, batchSize, seqQ, channels);

    midAttn->queueExecute(
        task,
        attnOut,
        x.reshape({batchSize * seqQ, channels}), // Input Latent
        contextEmbedding,                        // Context
        batchSize,
        seqQ,
        seqKV,
        heads);
    x = attnOut; // Shape is preserved.

    // 6. Up Blocks
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        auto& block = upBlocks[i];
        auto skip = skipConnections[skipConnections.getCount() - 1 - i];

        // Define Shapes for Concat (NHWC layout)
        Shape shape = {batchSize, h, w, block->inChannels};
        Shape shapes[] = {shape, shape};
        TensorView buffers[] = {x.reshape(shape), skip};

        // Concat along channel axis (axis 3 in NHWC)
        auto concated =
            concat->allocateResultBuffer(outputImage.elementType, makeArrayView(shapes), 3);
        concat->queueExecute(task, concated, makeArrayView(buffers), 3);

        auto res = block->allocateResultBuffer(outputImage.elementType, w, h, batchSize);
        block->queueExecute(task, res, concated, tEmb);
        x = res;

        w *= 2;
        h *= 2;
    }

    // 7. Final Conv
    finalConv->queueExecute(task, outputImage, x, 0);
}

UNetBlock::UNetBlock(
    RefPtr<InferencingContext> inferencingCtx,
    UNetBlockKind kind,
    int inChannels,
    int outChannels,
    int timeEmbedDim)
    : inferencingCtx(inferencingCtx), inChannels(inChannels), outChannels(outChannels)
{
    if (kind == UNetBlockKind::Down)
    {
        conv1 = new Conv2DKernel(
            inferencingCtx,
            16,
            3,
            1,
            inChannels,
            outChannels,
            relu(kernelOutput()),
            "conv1");
        downTransform = new Conv2DKernel(
            inferencingCtx,
            16,
            4,
            2,
            outChannels,
            outChannels,
            kernelOutput(),
            "transformDown");
    }
    else
    {
        conv1 = new Conv2DKernel(
            inferencingCtx,
            16,
            3,
            1,
            2 * inChannels,
            outChannels,
            relu(kernelOutput()),
            "conv1");
        upTransform = new TransposedConv2DKernel(
            inferencingCtx,
            16,
            4,
            2,
            outChannels,
            outChannels,
            kernelOutput(),
            "transformUp");
    }
    conv2 = new Conv2DKernel(
        inferencingCtx,
        16,
        3,
        1,
        outChannels,
        outChannels,
        relu(kernelOutput()),
        "conv2");
    timeEmbedTransform = new LinearKernel(
        inferencingCtx,
        buffer(),
        relu(kernelOutput()),
        bufferSink(),
        timeEmbedDim,
        outChannels);
    broadcastAdd = new BroadcastAddKernel(inferencingCtx);
}

SlangResult UNetBlock::loadParams(TorchParamReader& reader)
{
    SLANG_RETURN_ON_FAIL(timeEmbedTransform->loadParams(reader));
    SLANG_RETURN_ON_FAIL(conv1->loadParams(reader, true));
    if (downTransform)
        SLANG_RETURN_ON_FAIL(downTransform->loadParams(reader, false));
    if (upTransform)
        SLANG_RETURN_ON_FAIL(upTransform->loadParams(reader));
    SLANG_RETURN_ON_FAIL(conv2->loadParams(reader, true));
    return SLANG_OK;
}

void UNetBlock::writeResult(const char* name, BufferView buffer)
{
    ComPtr<ISlangBlob> blob;
    inferencingCtx->getDevice()
        ->readBuffer(buffer.buffer, buffer.offset, buffer.size, blob.writeRef());
    File::writeAllBytes(String(name) + ".bin", blob->getBufferPointer(), blob->getBufferSize());
}

TensorView UNetBlock::allocateResultBuffer(
    ElementType elementType,
    int inputWidth,
    int inputHeight,
    int batchSize)
{
    if (downTransform)
    {
        return downTransform
            ->allocateResultBuffer(elementType, inputWidth, inputHeight, 1, batchSize);
    }
    else
    {
        return upTransform
            ->allocateResultBuffer(elementType, inputWidth, inputHeight, 1, batchSize);
    }
}
void UNetBlock::queueExecute(
    InferencingTask& task,
    TensorView outputImage,
    TensorView inputImage,
    TensorView timeEmbedding)
{
    task.context->pushAllocScope();
    SLANG_DEFER(task.context->popAllocScope());

    SLANG_ASSERT(inputImage.shape.getRank() == 4);

    int batchSize = inputImage.shape[0];
    int inputHeight = inputImage.shape[1];
    int inputWidth = inputImage.shape[2];

    auto transformedTimeEmbedding =
        timeEmbedTransform->allocateResultBuffer(outputImage.elementType, batchSize);

    timeEmbedTransform->queueExecute(task, transformedTimeEmbedding, timeEmbedding);

    auto convResult =
        conv1->allocateResultBuffer(outputImage.elementType, inputWidth, inputHeight, 1, batchSize);
    conv1->queueExecute(task, convResult, inputImage, 1);

    int shapeA[] = {batchSize, inputHeight, inputWidth, outChannels};
    int shapeB[] = {batchSize, 1, 1, outChannels};
    auto added = broadcastAdd->allocateResultBuffer(
        outputImage.elementType,
        makeArrayView(shapeA),
        makeArrayView(shapeB));

    // Reshape time embedding from [B, C] to [B, 1, 1, C] for proper broadcasting
    // across spatial dimensions (H, W).
    auto reshapedTimeEmbedding =
        transformedTimeEmbedding.reshape(Shape(batchSize, 1, 1, outChannels));
    broadcastAdd->queueExecute(task, added, convResult, reshapedTimeEmbedding);
    conv2->queueExecute(task, convResult, added, 1);

    if (downTransform)
    {
        downTransform->queueExecute(task, outputImage, convResult, 1);
    }
    else
    {
        upTransform->queueExecute(task, outputImage, convResult, 1);
    }
}

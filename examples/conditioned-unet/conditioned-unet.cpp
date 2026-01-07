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
    // Standard Config: Multipliers [1, 2, 4, 8] -> Channels [64, 128, 256, 512]
    channelMultipliers = {1, 2, 4, 8};

    // 1. Time and class Embedding
    timeEmbed = new TimeEmbedingKernel(ctx, timeEmbedDim);
    classEmbed = new GatherKernel(ctx, classCount, contextDim);
    
    // Project class embedding to time embedding dimension for injection
    classToTimeEmbed = new LinearKernel(ctx, buffer(), kernelOutput(), bufferSink(), contextDim, timeEmbedDim);
    embedAdd = new BroadcastAddKernel(ctx);

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

    // 3. Class to Time projection
    SLANG_RETURN_ON_FAIL(classToTimeEmbed->loadParams(reader, true));

    // 4. Init Conv (no BatchNorm fusion)
    SLANG_RETURN_ON_FAIL(initialConv->loadParams(reader, false));

    // 5. Down Blocks
    for (auto& block : downBlocks)
    {
        SLANG_RETURN_ON_FAIL(block->loadParams(reader));
    }

    // 6. Mid Attention
    SLANG_RETURN_ON_FAIL(midAttn->loadParams(reader));

    // 7. Up Blocks
    for (auto& block : upBlocks)
    {
        SLANG_RETURN_ON_FAIL(block->loadParams(reader));
    }

    // 8. Final Conv
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

    // 3. Project class embedding to time dimension and add to time embedding
    auto classToTime = classToTimeEmbed->allocateResultBuffer(outputImage.elementType, batchSize);
    classToTimeEmbed->queueExecute(task, classToTime, contextEmbedding);
    
    // Add class-to-time projection to time embedding (t = t + class_to_time(class_emb))
    auto combinedEmb = embedAdd->allocateResultBuffer(
        outputImage.elementType,
        tEmb.shape,
        classToTime.shape);
    embedAdd->queueExecute(task, combinedEmb, tEmb, classToTime);
    tEmb = combinedEmb;  // Use combined embedding going forward

    // 4. Init Conv
    auto x =
        initialConv
            ->allocateResultBuffer(outputImage.elementType, inputWidth, inputHeight, 1, batchSize);
    initialConv->queueExecute(task, x, inputImage, 1);

    // 5. Down Blocks
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

    // 6. Mid Attention (Bottleneck)
    // x is now [Batch, H/8, W/8, 512] in NHWC (with new 4-level architecture)
    int seqQ = w * h;
    int channels = baseChannels * channelMultipliers[channelMultipliers.getCount() - 1];
    int seqKV = 1; // Context is [Batch, 1, contextDim]
    // numHeads must satisfy: channels = numHeads * headDim
    // CrossAttention was created with headDim=64, so numHeads = channels / 64
    int heads = channels / 64;

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

    // 7. Up Blocks
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

    // 8. Final Conv
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
    // GroupNorm configuration: 8 groups (matching Python model)
    const int numGroups = 8;

    if (kind == UNetBlockKind::Down)
    {
        // conv1 without fused ReLU (GroupNorm + ReLU applied separately)
        conv1 = new Conv2DKernel(
            inferencingCtx,
            16,
            3,
            1,
            inChannels,
            outChannels,
            kernelOutput(),  // No fused activation
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
        // conv1 without fused ReLU (GroupNorm + ReLU applied separately)
        conv1 = new Conv2DKernel(
            inferencingCtx,
            16,
            3,
            1,
            2 * inChannels,
            outChannels,
            kernelOutput(),  // No fused activation
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
    
    // conv2 without fused ReLU (GroupNorm + ReLU applied separately)
    conv2 = new Conv2DKernel(
        inferencingCtx,
        16,
        3,
        1,
        outChannels,
        outChannels,
        kernelOutput(),  // No fused activation
        "conv2");
    
    // GroupNorm layers
    gnorm1 = new GroupNormKernel(inferencingCtx, outChannels, numGroups);
    gnorm2 = new GroupNormKernel(inferencingCtx, outChannels, numGroups);
    
    // ReLU activation kernel (applied after GroupNorm)
    reluKernel = new ElementwiseKernel(inferencingCtx, relu(buffer()));
    
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
    // Load order matches Python model's named_modules traversal order:
    // 1. time_mlp (Linear)
    SLANG_RETURN_ON_FAIL(timeEmbedTransform->loadParams(reader));
    
    // 2. conv1 (Conv2d with bias, no BatchNorm fusion - GroupNorm is separate)
    SLANG_RETURN_ON_FAIL(conv1->loadParams(reader, false));
    
    // 3. gnorm1 (GroupNorm)
    SLANG_RETURN_ON_FAIL(gnorm1->loadParams(reader));
    
    // 4. transform (down: Conv2d with bias, up: ConvTranspose2d with bias)
    if (downTransform)
        SLANG_RETURN_ON_FAIL(downTransform->loadParams(reader, false));
    if (upTransform)
        SLANG_RETURN_ON_FAIL(upTransform->loadParams(reader));
    
    // 5. conv2 (Conv2d with bias, no BatchNorm fusion - GroupNorm is separate)
    SLANG_RETURN_ON_FAIL(conv2->loadParams(reader, false));
    
    // 6. gnorm2 (GroupNorm)
    SLANG_RETURN_ON_FAIL(gnorm2->loadParams(reader));
    
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

    // 1. Time embedding projection (with ReLU)
    auto transformedTimeEmbedding =
        timeEmbedTransform->allocateResultBuffer(outputImage.elementType, batchSize);
    timeEmbedTransform->queueExecute(task, transformedTimeEmbedding, timeEmbedding);

    // 2. First convolution (no activation)
    auto conv1Result =
        conv1->allocateResultBuffer(outputImage.elementType, inputWidth, inputHeight, 1, batchSize);
    conv1->queueExecute(task, conv1Result, inputImage, 1);

    // 3. GroupNorm1
    auto gnorm1Result = gnorm1->allocateResultBuffer(outputImage.elementType, batchSize, inputHeight, inputWidth);
    gnorm1->queueExecute(task, gnorm1Result, conv1Result);

    // 4. ReLU after GroupNorm1
    auto relu1Result = task.context->allocScratchTensor(
        outputImage.elementType,
        Shape{batchSize, inputHeight, inputWidth, outChannels},
        "relu1_out");
    reluKernel->queueExecute(task, relu1Result, gnorm1Result);

    // 5. Add time embedding
    int shapeA[] = {batchSize, inputHeight, inputWidth, outChannels};
    int shapeB[] = {batchSize, 1, 1, outChannels};
    auto added = broadcastAdd->allocateResultBuffer(
        outputImage.elementType,
        makeArrayView(shapeA),
        makeArrayView(shapeB));

    // Reshape time embedding from [B, C] to [B, 1, 1, C] for proper broadcasting
    auto reshapedTimeEmbedding =
        transformedTimeEmbedding.reshape(Shape(batchSize, 1, 1, outChannels));
    broadcastAdd->queueExecute(task, added, relu1Result, reshapedTimeEmbedding);

    // 6. Second convolution (no activation)
    auto conv2Result =
        conv2->allocateResultBuffer(outputImage.elementType, inputWidth, inputHeight, 1, batchSize);
    conv2->queueExecute(task, conv2Result, added, 1);

    // 7. GroupNorm2
    auto gnorm2Result = gnorm2->allocateResultBuffer(outputImage.elementType, batchSize, inputHeight, inputWidth);
    gnorm2->queueExecute(task, gnorm2Result, conv2Result);

    // 8. ReLU after GroupNorm2
    auto relu2Result = task.context->allocScratchTensor(
        outputImage.elementType,
        Shape{batchSize, inputHeight, inputWidth, outChannels},
        "relu2_out");
    reluKernel->queueExecute(task, relu2Result, gnorm2Result);

    // 9. Down/Up transform
    if (downTransform)
    {
        downTransform->queueExecute(task, outputImage, relu2Result, 1);
    }
    else
    {
        upTransform->queueExecute(task, outputImage, relu2Result, 1);
    }
}

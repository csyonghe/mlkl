#include "simple-unet.h"

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
            kernelOutput(),  // No ReLU - matches PyTorch's self.transform
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

UNetModel::UNetModel(
    RefPtr<InferencingContext> inferencingCtx,
    int inputChannels,
    int outputChannels)
    : inferencingCtx(inferencingCtx)
{
    static const int timeEmbedDim = 32;
    timeEmbedKernel = new TimeEmbedingKernel(inferencingCtx, timeEmbedDim);
    int channelSizes[] = {64, 128, 256, 512, 1024};
    for (Index i = 0; i < SLANG_COUNT_OF(channelSizes) - 1; i++)
    {
        downBlocks.add(new UNetBlock(
            inferencingCtx,
            UNetBlockKind::Down,
            channelSizes[i],
            channelSizes[i + 1],
            timeEmbedDim));
        upBlocks.add(new UNetBlock(
            inferencingCtx,
            UNetBlockKind::Up,
            channelSizes[SLANG_COUNT_OF(channelSizes) - 1 - i],
            channelSizes[SLANG_COUNT_OF(channelSizes) - 2 - i],
            timeEmbedDim));
    }
    initialConv = new Conv2DKernel(
        inferencingCtx,
        16,
        3,
        1,
        inputChannels,
        channelSizes[0],
        kernelOutput(),
        "initialConv");
    finalConv = new Conv2DKernel(
        inferencingCtx,
        16,
        1,
        1,
        channelSizes[0],
        outputChannels,
        kernelOutput(),
        "predictedNoiseConv");
    concat = new ConcatKernel(inferencingCtx, 2);
}

SlangResult UNetModel::loadParams(TorchParamReader& reader)
{
    SLANG_RETURN_ON_FAIL(timeEmbedKernel->loadParams(reader));
    SLANG_RETURN_ON_FAIL(initialConv->loadParams(reader, false));
    for (auto& block : downBlocks)
    {
        SLANG_RETURN_ON_FAIL(block->loadParams(reader));
    }
    for (auto& block : upBlocks)
    {
        SLANG_RETURN_ON_FAIL(block->loadParams(reader));
    }
    SLANG_RETURN_ON_FAIL(finalConv->loadParams(reader, false));
    return SLANG_OK;
}

void UNetModel::queueExecute(
    InferencingTask& task,
    TensorView outputImage,
    TensorView inputImage,
    int timeStep)
{
    task.context->pushAllocScope();
    SLANG_DEFER(task.context->popAllocScope());
    SLANG_ASSERT(inputImage.shape.getRank() == 4);

    int batchSize = outputImage.shape[0];
    int inputHeight = inputImage.shape[1];
    int inputWidth = inputImage.shape[2];

    auto timeEmbedding = timeEmbedKernel->allocateResultBuffer(ElementType::Float32, batchSize);
    timeEmbedKernel->queueExecute(task, timeEmbedding, timeStep);
    auto x =
        initialConv
            ->allocateResultBuffer(ElementType::Float32, inputWidth, inputHeight, 1, batchSize);
    initialConv->queueExecute(task, x, inputImage, 1);
    ShortList<TensorView> skipConnections;
    for (auto& block : downBlocks)
    {
        auto x1 = block->allocateResultBuffer(
            outputImage.elementType,
            inputWidth,
            inputHeight,
            batchSize);
        block->queueExecute(task, x1, x, timeEmbedding);
        skipConnections.add(x1);
        x = x1;
        inputWidth /= 2;
        inputHeight /= 2;
    }
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        auto& block = upBlocks[i];
        // Concat skip connection
        auto skipConnection = skipConnections[skipConnections.getCount() - 1 - i];
        Shape shape = {batchSize, inputHeight, inputWidth, block->inChannels};
        Shape shapes[] = {shape, shape};
        TensorView buffers[] = {x, skipConnection};
        auto concated =
            concat->allocateResultBuffer(outputImage.elementType, makeArrayView(shapes), 3);
        concat->queueExecute(task, concated, makeArrayView(buffers), 3);
        // Up block
        auto upsampled = block->allocateResultBuffer(
            outputImage.elementType,
            inputWidth,
            inputHeight,
            batchSize);
        block->queueExecute(task, upsampled, concated, timeEmbedding);
        x = upsampled;
        inputWidth *= 2;
        inputHeight *= 2;
    }
    finalConv->queueExecute(task, outputImage, x, 0);
}

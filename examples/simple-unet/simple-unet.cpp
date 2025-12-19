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
            ActivationFunction::ReLU,
            "conv1");
        downTransform = new Conv2DKernel(
            inferencingCtx,
            16,
            4,
            2,
            outChannels,
            outChannels,
            ActivationFunction::None,
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
            ActivationFunction::ReLU,
            "conv1");
        upTransform = new TransposedConv2DKernel(
            inferencingCtx,
            16,
            4,
            2,
            outChannels,
            outChannels,
            ActivationFunction::None,
            "transformUp");
    }
    conv2 = new Conv2DKernel(
        inferencingCtx,
        16,
        3,
        1,
        outChannels,
        outChannels,
        ActivationFunction::ReLU,
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

BufferView UNetBlock::allocateResultBuffer(int inputWidth, int inputHeight, int batchSize)
{
    if (downTransform)
    {
        return downTransform->allocateResultBuffer(inputWidth, inputHeight, 1, batchSize);
    }
    else
    {
        return upTransform->allocateResultBuffer(inputWidth, inputHeight, 1, batchSize);
    }
}
void UNetBlock::queueExecute(
    InferencingTask& task,
    BufferView outputImage,
    BufferView inputImage,
    int inputWidth,
    int inputHeight,
    int batchSize,
    BufferView timeEmbedding)
{
    task.context->pushAllocScope();
    SLANG_DEFER(task.context->popAllocScope());

    auto transformedTimeEmbedding = timeEmbedTransform->allocateResultBuffer(batchSize);

    timeEmbedTransform->queueExecute(task, transformedTimeEmbedding, timeEmbedding, batchSize);

    auto convResult = conv1->allocateResultBuffer(inputWidth, inputHeight, 1, batchSize);
    conv1->queueExecute(task, convResult, inputImage, inputWidth, inputHeight, 1, batchSize);

    int shapeA[] = {inputHeight, inputWidth, outChannels};
    int shapeB[] = {1, 1, outChannels};
    auto added =
        broadcastAdd->allocateResultBuffer(makeArrayView(shapeA), makeArrayView(shapeB), batchSize);

    broadcastAdd->queueExecute(
        task,
        added,
        convResult,
        makeArrayView(shapeA),
        transformedTimeEmbedding,
        makeArrayView(shapeB),
        batchSize);
    conv2->queueExecute(task, convResult, added, inputWidth, inputHeight, 1, batchSize);

    if (downTransform)
    {
        downTransform
            ->queueExecute(task, outputImage, convResult, inputWidth, inputHeight, 1, batchSize);
    }
    else
    {
        upTransform
            ->queueExecute(task, outputImage, convResult, inputWidth, inputHeight, 1, batchSize);
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
        ActivationFunction::None,
        "initialConv");
    finalConv = new Conv2DKernel(
        inferencingCtx,
        16,
        1,
        1,
        channelSizes[0],
        outputChannels,
        ActivationFunction::None,
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
    BufferView outputImage,
    BufferView inputImage,
    int inputWidth,
    int inputHeight,
    int timeStep,
    int batchSize)
{
    task.context->pushAllocScope();
    SLANG_DEFER(task.context->popAllocScope());

    auto timeEmbedding = timeEmbedKernel->allocateResultBuffer(batchSize);
    timeEmbedKernel->queueExecute(task, timeEmbedding, timeStep, batchSize);
    auto x = initialConv->allocateResultBuffer(inputWidth, inputHeight, 1, batchSize);
    initialConv->queueExecute(task, x, inputImage, inputWidth, inputHeight, 1, batchSize);
    List<BufferView> skipConnections;
    for (auto& block : downBlocks)
    {
        auto x1 = block->allocateResultBuffer(inputWidth, inputHeight, batchSize);
        block->queueExecute(task, x1, x, inputWidth, inputHeight, batchSize, timeEmbedding);
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
        BufferView buffers[] = {x, skipConnection};
        auto concated = concat->allocateResultBuffer(makeArrayView(shapes), 3);
        concat->queueExecute(task, concated, makeArrayView(buffers), makeArrayView(shapes), 3);
        // Up block
        auto upsampled = block->allocateResultBuffer(inputWidth, inputHeight, batchSize);
        block->queueExecute(
            task,
            upsampled,
            concated,
            inputWidth,
            inputHeight,
            batchSize,
            timeEmbedding);
        x = upsampled;
        inputWidth *= 2;
        inputHeight *= 2;
    }
    finalConv->queueExecute(task, outputImage, x, inputWidth, inputHeight, 0, batchSize);
}

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
    timeEmbedTransform =
        new LinearKernel(inferencingCtx, ActivationFunction::ReLU, 128, timeEmbedDim, outChannels);
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

void UNetBlock::writeResult(const char* name, rhi::IBuffer* buffer)
{
    ComPtr<ISlangBlob> blob;
    inferencingCtx->getDevice()->readBuffer(buffer, 0, buffer->getDesc().size, blob.writeRef());
    File::writeAllBytes(String(name) + ".bin", blob->getBufferPointer(), blob->getBufferSize());
}

ComPtr<rhi::IBuffer> UNetBlock::forward(
    InferencingTask& task,
    rhi::IBuffer* inputImage,
    int inputWidth,
    int inputHeight,
    rhi::IBuffer* timeEmbedding)
{
    auto transformedTimeEmbedding = timeEmbedTransform->queueExecute(task, timeEmbedding);
    writeResult("time_embed_out", transformedTimeEmbedding);
    auto conv1Result = conv1->queueExecute(task, inputImage, inputWidth, inputHeight, 1);
    writeResult("conv1_fused_out", conv1Result);

    int shapeA[] = {inputHeight, inputWidth, outChannels};
    int shapeB[] = {1, 1, outChannels};
    auto added = broadcastAdd->queueExecute(
        task,
        conv1Result,
        makeArrayView(shapeA),
        transformedTimeEmbedding,
        makeArrayView(shapeB));
    auto conv2Result = conv2->queueExecute(task, added, inputWidth, inputHeight, 1);
    writeResult("conv2_fused_out", conv2Result);

    rhi::IBuffer* finalResult = conv2Result;
    if (downTransform)
        finalResult = downTransform->queueExecute(task, conv2Result, inputWidth, inputHeight, 1);
    else
        finalResult = upTransform->queueExecute(task, conv2Result, inputWidth, inputHeight, 1);
    return ComPtr<rhi::IBuffer>(finalResult);
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
    concat = new ConcatKernel(inferencingCtx);
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

ComPtr<rhi::IBuffer> UNetModel::forward(
    InferencingTask& task,
    rhi::IBuffer* inputImage,
    int inputWidth,
    int inputHeight,
    int timeStep)
{
    auto timeEmbedding = timeEmbedKernel->queueExecute(task, timeStep);
    auto x = initialConv->queueExecute(task, inputImage, inputWidth, inputHeight, 1);
    List<ComPtr<rhi::IBuffer>> skipConnections;
    for (auto& block : downBlocks)
    {
        x = block->forward(task, x, inputWidth, inputHeight, timeEmbedding);
        skipConnections.add(x);
        inputWidth /= 2;
        inputHeight /= 2;
    }
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        auto& block = upBlocks[i];
        // Concat skip connection
        auto skipConnection = skipConnections[skipConnections.getCount() - 1 - i];
        int shape[] = {inputHeight, inputWidth, block->inChannels};
        auto shapeView = makeArrayView(shape);
        x = concat->queueExecute(task, x, shapeView, skipConnection, shapeView, 2);
        // Up block
        x = block->forward(task, x, inputWidth, inputHeight, timeEmbedding);
        inputWidth *= 2;
        inputHeight *= 2;
    }
    x = finalConv->queueExecute(task, x, inputWidth, inputHeight, 0);
    return ComPtr<rhi::IBuffer>(x);
}


#include "time-embed.h"

TimeEmbedingKernel::TimeEmbedingKernel(InferencingContext* context, int outputChannels)
    : context(context), outputChannels(outputChannels)
{
    pipeline = context->createComputePipeline(
        "computeTimeEmbedding",
        makeConstArrayViewSingle(String(outputChannels)));
}

SlangResult TimeEmbedingKernel::loadParams(TorchParamReader& reader)
{
    logInfo("Loading TimeEmbed Linear Layer: outputChannel %d\n", outputChannels);
    LinearLayerParams linearParams;
    reader.readLinearLayer(outputChannels, outputChannels, linearParams);
    biasesBuffer = context->createBuffer(linearParams.biases);
    if (!biasesBuffer)
        return SLANG_FAIL;
    weightsBuffer = context->createBuffer(linearParams.weights);
    if (!weightsBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

struct TimeEmbeddingKernelParams
{
    rhi::DeviceAddress output;  // Layout: [OutputDim]
    rhi::DeviceAddress weights; // Layout: [EmbeddingDim * OutputDim] (Row-Major: In x Out)
    rhi::DeviceAddress biases;  // [OutputDim]
    uint32_t timeStep;
    uint32_t embeddingDim; // The size of the sin/cos vector (InChannels)
    float maxPeriod;       // Default 10000.0
    uint32_t batchSize;
};

ComPtr<rhi::IBuffer> TimeEmbedingKernel::queueExecute(
    InferencingTask& task,
    uint32_t timeStep,
    int batchSize)
{
    // Allocate [Batch, OutputDim]
    auto outputBuffer =
        task.allocateBuffer("time-embed", batchSize * outputChannels * sizeof(float));

    TimeEmbeddingKernelParams params = {};
    params.output = outputBuffer->getDeviceAddress();
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();
    params.embeddingDim = outputChannels;
    params.maxPeriod = 10000.0f;
    params.timeStep = timeStep;
    params.batchSize = batchSize;

    // Dispatch: X=Channels, Y=Batch
    // Group size is 32 in X.
    uint32_t threadsX = (outputChannels + 31) / 32;
    task.dispatchKernel(pipeline, threadsX, batchSize, 1, params);

    return ComPtr<rhi::IBuffer>(outputBuffer);
}

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
    reader.readLinearLayer(outputChannels, outputChannels, true, linearParams);
    biasesBuffer = context->createPersistentBuffer(linearParams.biases);
    if (!biasesBuffer)
        return SLANG_FAIL;
    weightsBuffer = context->createPersistentBuffer(linearParams.weights);
    if (!weightsBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

BufferView TimeEmbedingKernel::allocResultBuffer(int batchSize)
{
    return context->allocScratchBuffer(
        batchSize * outputChannels * sizeof(float),
        "time_embed_output");
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

void TimeEmbedingKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    uint32_t timeStep,
    int batchSize)
{
    TimeEmbeddingKernelParams params = {};
    params.output = output.getDeviceAddress();
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
}
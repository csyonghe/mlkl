
#include "kernels.h"

using namespace Slang;

TimeEmbedingKernel::TimeEmbedingKernel(InferencingContext* context, int outputChannels)
    : context(context), outputChannels(outputChannels)
{
    pipeline = context->createComputePipeline("computeTimeEmbedding", makeConstArrayViewSingle(String(outputChannels)));
}

SlangResult TimeEmbedingKernel::loadParams(TorchParamReader& reader)
{
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
    rhi::DeviceAddress output; // Layout: [OutputDim]
    rhi::DeviceAddress weights; // Layout: [EmbeddingDim * OutputDim] (Row-Major: In x Out)
    rhi::DeviceAddress biases; // [OutputDim]
    uint32_t timeStep;
    uint32_t embeddingDim; // The size of the sin/cos vector (InChannels)
    float maxPeriod;   // Default 10000.0
};

ComPtr<rhi::IBuffer> TimeEmbedingKernel::queueExecute(InferencingTask& task, uint32_t timeStepsBuffer)
{
    auto outputBuffer = task.allocateBuffer(outputChannels * sizeof(float));

    TimeEmbeddingKernelParams params = {};
    params.output = outputBuffer->getDeviceAddress();
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();
    params.embeddingDim = outputChannels;
    params.maxPeriod = 10000.0f;
    task.dispatchKernel(pipeline, 1, 1, 1, params);

    return ComPtr<rhi::IBuffer>(outputBuffer);
}

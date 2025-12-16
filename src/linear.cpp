
#include "kernels.h"

using namespace Slang;

const char* getActivationFuncName(ActivationFunction func)
{
    switch (func)
    {
    case ActivationFunction::None:
        return "IdentityActivation";
    case ActivationFunction::ReLU:
        return "ReLUActivation";
    case ActivationFunction::SiLU:
        return "SiLUActivation";
    default:
        return "IdentityActivation";
    }
}

LinearKernel::LinearKernel(
    InferencingContext* context,
    ActivationFunction activation,
    int tileSize,
    int inputSize,
    int outputSize)
    : context(context), tileSize(tileSize), inputSize(inputSize), outputSize(outputSize)
{
    String specArgs[] = {String(tileSize), getActivationFuncName(activation)};
    pipeline = context->createComputePipeline("linearLayer", makeConstArrayView(specArgs));
}

SlangResult LinearKernel::loadParams(TorchParamReader& reader)
{
    logInfo("Loading Linear Layer: inputSize=%d, outputSize=%d\n", inputSize, outputSize);
    LinearLayerParams params;
    SLANG_RETURN_ON_FAIL(reader.readLinearLayer(inputSize, outputSize, params));
    weightsBuffer = context->createBuffer(params.weights);
    biasesBuffer = context->createBuffer(params.biases);
    return SLANG_OK;
}

ComPtr<rhi::IBuffer> LinearKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* inputVector,
    int batchSize)
{
    struct LinearLayerParamsData
    {
        rhi::DeviceAddress weights;
        rhi::DeviceAddress biases;
        rhi::DeviceAddress inputVector;
        rhi::DeviceAddress outputVector;
        int inputSize;
        int outputSize;
        int batchSize;
    } paramsData;

    // Allocate output for the full batch
    auto outputBuffer =
        task.allocateBuffer("linear_output", batchSize * outputSize * sizeof(float));

    paramsData.inputVector = inputVector->getDeviceAddress();
    paramsData.outputVector = outputBuffer->getDeviceAddress();
    paramsData.weights = weightsBuffer->getDeviceAddress();
    paramsData.biases = biasesBuffer->getDeviceAddress();
    paramsData.inputSize = inputSize;
    paramsData.outputSize = outputSize;
    paramsData.batchSize = batchSize;

    // Dispatch Grid:
    // X: Enough groups to cover outputSize
    // Y: One group per batch item (or more if needed, but 1 group per batch index usually maps to Y
    // dim)
    uint32_t threadGroupCountX = (uint32_t)((outputSize + tileSize - 1) / tileSize);
    uint32_t threadGroupCountY = (uint32_t)batchSize;

    task.dispatchKernel(pipeline, threadGroupCountX, threadGroupCountY, 1, paramsData);

    return ComPtr<rhi::IBuffer>(outputBuffer);
}
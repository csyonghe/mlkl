
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

SlangResult LinearKernel::loadParams(TorchParamReader& reader, bool loadBias)
{
    logInfo("Loading Linear Layer: inputSize=%d, outputSize=%d\n", inputSize, outputSize);
    LinearLayerParams params;
    SLANG_RETURN_ON_FAIL(reader.readLinearLayer(inputSize, outputSize, loadBias, params));
    weightsBuffer = context->createPersistentBuffer(params.weights);
    if (loadBias)
        biasesBuffer = context->createPersistentBuffer(params.biases);
    return SLANG_OK;
}


BufferView LinearKernel::allocateResultBuffer(int batchSize)
{
    auto outputBuffer =
        context->allocScratchBuffer(batchSize * outputSize * sizeof(float), "linear_output");
    return outputBuffer;
}

void LinearKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    BufferView inputVector,
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

    paramsData.inputVector = inputVector.getDeviceAddress();
    paramsData.outputVector = output.getDeviceAddress();
    paramsData.weights = weightsBuffer->getDeviceAddress();
    paramsData.biases = biasesBuffer ? biasesBuffer->getDeviceAddress() : 0;
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
}
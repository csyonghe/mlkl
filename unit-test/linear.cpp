
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

LinearKernel::LinearKernel(InferencingContext* context, ActivationFunction activation, int tileSize, int inputSize, int outputSize)
    : context(context)
    , tileSize(tileSize)
    , inputSize(inputSize)
    , outputSize(outputSize)
{
    String specArgs[] = {
        String(tileSize),
        getActivationFuncName(activation)
    };
    pipeline = context->createComputePipeline("linearLayer",
        makeConstArrayView(specArgs));
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

ComPtr<rhi::IBuffer> LinearKernel::queueExecute(InferencingTask& task, rhi::IBuffer* inputVector)
{
    struct LinearLayerParamsData
    {
        rhi::DeviceAddress weights;
        rhi::DeviceAddress biases;
        rhi::DeviceAddress inputVector;
        rhi::DeviceAddress outputVector;
        int inputSize;
        int outputSize;
    } paramsData;
    auto outputBuffer = context->createBuffer(nullptr, outputSize * sizeof(float));
    paramsData.inputVector = inputVector->getDeviceAddress();
    paramsData.outputVector = outputBuffer->getDeviceAddress();
    paramsData.weights = weightsBuffer->getDeviceAddress();
    paramsData.biases = biasesBuffer->getDeviceAddress();
    paramsData.inputSize = inputSize;
    paramsData.outputSize = outputSize;
    uint32_t threadGroupCountX = (uint32_t)((outputSize + tileSize - 1) / tileSize);
    task.dispatchKernel(
        pipeline,
        threadGroupCountX,
        1,
        1,
        paramsData);
    return outputBuffer;
}
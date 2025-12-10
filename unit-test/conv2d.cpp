
#include "kernels.h"

using namespace Slang;

Conv2DKernel::Conv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int inChannels, int outChannels)
    : context(context), tileSize(tileSize), kernelSize(kernelSize), inChannels(inChannels), outChannels(outChannels)
{
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(inChannels),
        String(outChannels)
    };
    pipeline = context->createComputePipeline("simpleConvolution", makeArrayView(specArgs));
}

SlangResult Conv2DKernel::loadParams(TorchParamReader& reader)
{
    Conv2DLayerParams convParams;
    SLANG_RETURN_ON_FAIL(reader.readConv2DLayer(inChannels, outChannels, kernelSize, convParams));
    biasesBuffer = context->createBuffer(convParams.biases);
    if (!biasesBuffer)
        return SLANG_FAIL;
    weightsBuffer = context->createBuffer(convParams.weights);
    if (!weightsBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

SlangResult Conv2DKernel::loadParams(int kernelSize, int outputChannelCount, float* weightsData, float* biasesData)
{
    weightsBuffer = context->createBuffer(weightsData, kernelSize * kernelSize * inChannels * outputChannelCount * sizeof(float));
    if (!weightsBuffer)
        return SLANG_FAIL;
    biasesBuffer = context->createBuffer(biasesData, outputChannelCount * sizeof(float));
    if (!biasesBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

struct Conv2DKernelParams
{
    rhi::DeviceAddress weights;
    rhi::DeviceAddress biases;
    rhi::DeviceAddress inputImage;
    rhi::DeviceAddress outputImage;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int stride;
    int padding;
};

ComPtr<rhi::IBuffer> Conv2DKernel::queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int stride, int padding)
{
    int outputWidth = (inputWidth + padding - 1) / stride;
    int outputHeight = (inputHeight + padding - 1) / stride;
    auto outputBuffer = task.allocateBuffer(outputWidth * outputHeight * outChannels * sizeof(float));

    Conv2DKernelParams params = {};
    params.inputImage = inputImage->getDeviceAddress();
    params.outputImage = outputBuffer->getDeviceAddress();
    params.inputImageWidth = inputWidth;
    params.inputImageHeight = inputHeight;
    params.outputImageWidth = outputWidth;
    params.stride = stride;
    params.padding = padding;
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();
    task.dispatchKernel(pipeline, (outputWidth+tileSize-1)/tileSize, (outputHeight+tileSize-1)/tileSize, 1, params);
    return ComPtr<rhi::IBuffer>(outputBuffer);
}

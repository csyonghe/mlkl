
#include "kernels.h"

using namespace Slang;

TransposedConv2DKernel::TransposedConv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int stride, int inChannels, int outChannels)
    : context(context), tileSize(tileSize), stride(stride), kernelSize(kernelSize), inChannels(inChannels), outChannels(outChannels)
{
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels)
    };
    pipeline = context->createComputePipeline("simpleTransposedConvolution", makeArrayView(specArgs));
}

SlangResult TransposedConv2DKernel::loadParams(TorchParamReader& reader)
{
    TransposedConv2DLayerParams convParams;
    SLANG_RETURN_ON_FAIL(reader.readTransposedConv2DLayer(inChannels, outChannels, kernelSize, convParams));
    biasesBuffer = context->createBuffer(convParams.biases);
    if (!biasesBuffer)
        return SLANG_FAIL;
    weightsBuffer = context->createBuffer(convParams.weights);
    if (!weightsBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

SlangResult TransposedConv2DKernel::loadParams(int kernelSize, int outputChannelCount, float* weightsData, float* biasesData)
{
    weightsBuffer = context->createBuffer(weightsData, kernelSize * kernelSize * inChannels * outputChannelCount * sizeof(float));
    if (!weightsBuffer)
        return SLANG_FAIL;
    biasesBuffer = context->createBuffer(biasesData, outputChannelCount * sizeof(float));
    if (!biasesBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

struct TransposedConv2DKernelParams
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

ComPtr<rhi::IBuffer> TransposedConv2DKernel::queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int padding)
{
    int outputWidth = (inputWidth - 1) * stride - 2 * padding + kernelSize;
    int outputHeight = (inputHeight - 1) * stride - 2 * padding + kernelSize;
    auto outputBuffer = task.allocateBuffer(outputWidth * outputHeight * outChannels * sizeof(float));

    TransposedConv2DKernelParams params = {};
    params.inputImage = inputImage->getDeviceAddress();
    params.outputImage = outputBuffer->getDeviceAddress();
    params.inputImageWidth = inputWidth;
    params.inputImageHeight = inputHeight;
    params.outputImageWidth = outputWidth;
    params.stride = stride;
    params.padding = padding;
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();
    task.dispatchKernel(pipeline, (outputWidth + tileSize - 1) / tileSize, (outputHeight + tileSize - 1) / tileSize, 1, params);
    return ComPtr<rhi::IBuffer>(outputBuffer);
}

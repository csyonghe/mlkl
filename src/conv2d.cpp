
#include "conv2d.h"

Conv2DKernel::Conv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int stride, int inChannels, int outChannels, ActivationFunction activation, String name)
    : context(context), tileSize(tileSize), kernelSize(kernelSize), stride(stride), inChannels(inChannels), outChannels(outChannels), activation(activation), name(name)
{
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        getActivationFuncName(activation)
    };
    pipeline = context->createComputePipeline("simpleConvolution", makeArrayView(specArgs));
}

SlangResult Conv2DKernel::loadParams(TorchParamReader& reader, bool loadAndFuseBNorm)
{
    logInfo("Loading Conv2D Layer: inChannels=%d, outChannels=%d, kernelSize=%d\n", inChannels, outChannels, kernelSize);
    Conv2DLayerParams convParams;
    SLANG_RETURN_ON_FAIL(reader.readConv2DLayer(inChannels, outChannels, kernelSize, convParams));
    if (loadAndFuseBNorm)
    {
        logInfo("Loading and fusing BatchNorm2D Layer: numFeatures=%d\n", outChannels);
        BatchNorm2DLayerParams bnParams;
        SLANG_RETURN_ON_FAIL(reader.readBatchNorm2DLayer(outChannels, bnParams));
        convParams.fuseBatchNorm(bnParams);
    }
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
    int outputImageHeight;
    int padding;
};

ComPtr<rhi::IBuffer> Conv2DKernel::queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int padding)
{
    int outputWidth = (inputWidth + padding * 2 - kernelSize) / stride + 1;
    int outputHeight = (inputHeight + padding * 2 - kernelSize) / stride + 1;
    String resultBufferName = name + "_" + String(outputWidth) + "x" + String(outputHeight) + "x" + String(outChannels);
    auto outputBuffer = task.allocateBuffer(resultBufferName.getBuffer(), outputWidth * outputHeight * outChannels * sizeof(float));
    auto expectedInputSize = inputWidth * inputHeight * inChannels * sizeof(float);
    SLANG_ASSERT(inputImage->getDesc().size == expectedInputSize);
    Conv2DKernelParams params = {};
    params.inputImage = inputImage->getDeviceAddress();
    params.outputImage = outputBuffer->getDeviceAddress();
    params.inputImageWidth = inputWidth;
    params.inputImageHeight = inputHeight;
    params.outputImageWidth = outputWidth;
    params.outputImageHeight = outputHeight;
    params.padding = padding;
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();
    static const int batchOutChannels = 32;
    int zBlocks = (outChannels + batchOutChannels - 1) / batchOutChannels;
    task.dispatchKernel(pipeline, (outputWidth+tileSize-1)/tileSize, (outputHeight+tileSize-1)/tileSize, zBlocks, params);
    return ComPtr<rhi::IBuffer>(outputBuffer);
}

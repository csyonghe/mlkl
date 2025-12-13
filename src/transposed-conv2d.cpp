
#include "transposed-conv2d.h"

TransposedConv2DKernel::TransposedConv2DKernel(
    InferencingContext* context,
    int tileSize,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    ActivationFunction activation,
    String name)
    : context(context)
    , tileSize(tileSize)
    , stride(stride)
    , kernelSize(kernelSize)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , name(name)
{
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        getActivationFuncName(activation)};
    pipeline =
        context->createComputePipeline("tiledTransposedConvolution", makeArrayView(specArgs));

    // Create Flat Pipeline
    String flatArgs[] = {
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        getActivationFuncName(activation)};
    // Note: tileSize is NOT needed for flat kernel generic
    flatPipeline =
        context->createComputePipeline("flatTransposedConvolution", makeArrayView(flatArgs));
}

SlangResult TransposedConv2DKernel::loadParams(TorchParamReader& reader)
{
    logInfo(
        "Loading TransposedConv2D Layer: inChannels=%d, outChannels=%d, kernelSize=%d\n",
        inChannels,
        outChannels,
        kernelSize);
    TransposedConv2DLayerParams convParams;
    SLANG_RETURN_ON_FAIL(
        reader.readTransposedConv2DLayer(inChannels, outChannels, kernelSize, convParams));
    biasesBuffer = context->createBuffer(convParams.biases);
    if (!biasesBuffer)
        return SLANG_FAIL;
    weightsBuffer = context->createBuffer(convParams.weights);
    if (!weightsBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

SlangResult TransposedConv2DKernel::loadParams(
    int kernelSize,
    int outputChannelCount,
    float* weightsData,
    float* biasesData)
{
    weightsBuffer = context->createBuffer(
        weightsData,
        kernelSize * kernelSize * inChannels * outputChannelCount * sizeof(float));
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
    int outputImageHeight;
    int stride;
    int padding;
};

ComPtr<rhi::IBuffer> TransposedConv2DKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* inputImage,
    int inputWidth,
    int inputHeight,
    int padding)
{
    int outputWidth = (inputWidth - 1) * stride - 2 * padding + kernelSize;
    int outputHeight = (inputHeight - 1) * stride - 2 * padding + kernelSize;
    String resultBufferName =
        name + "_" + String(outputWidth) + "x" + String(outputHeight) + "x" + String(outChannels);
    auto outputBuffer = task.allocateBuffer(
        resultBufferName.getBuffer(),
        outputWidth * outputHeight * outChannels * sizeof(float));

    TransposedConv2DKernelParams params = {};
    params.inputImage = inputImage->getDeviceAddress();
    params.outputImage = outputBuffer->getDeviceAddress();
    params.inputImageWidth = inputWidth;
    params.inputImageHeight = inputHeight;
    params.outputImageWidth = outputWidth;
    params.outputImageHeight = outputHeight;
    params.stride = stride;
    params.padding = padding;
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();

    if (outputWidth * outputHeight <= 1024)
    {
        // Dispatch 1D Grid
        int totalElements = outputWidth * outputHeight * outChannels;
        int groupSize = 256;
        int numGroups = (totalElements + groupSize - 1) / groupSize;

        task.dispatchKernel(flatPipeline, numGroups, 1, 1, params);
    }
    else
    {
        static const int batchOutChannels = 32;
        int zBlocks = (outChannels + batchOutChannels - 1) / batchOutChannels;
        task.dispatchKernel(
            pipeline,
            (outputWidth + tileSize - 1) / tileSize,
            (outputHeight + tileSize - 1) / tileSize,
            zBlocks,
            params);
    }
    return ComPtr<rhi::IBuffer>(outputBuffer);
}

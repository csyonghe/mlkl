
#include "conv2d.h"

static const int kTinyKernelMaxOutputPixels = 4;

Conv2DKernel::Conv2DKernel(
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
    , kernelSize(kernelSize)
    , stride(stride)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , activation(activation)
    , name(name)
{
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        getActivationFuncName(activation)};
    tilePipeline = context->createComputePipeline("tiledConvolution", makeArrayView(specArgs));

    // Create Flat Pipeline
    String flatArgs[] = {
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        getActivationFuncName(activation)};
    // Note: tileSize is NOT needed for flat kernel generic
    flatPipeline = context->createComputePipeline("flatConvolution", makeArrayView(flatArgs));
    flatWaveReducePipeline =
        context->createComputePipeline("flatConvolutionWaveReduce", makeArrayView(flatArgs));
}

SlangResult Conv2DKernel::loadParams(TorchParamReader& reader, bool loadAndFuseBNorm)
{
    logInfo(
        "Loading Conv2D Layer: inChannels=%d, outChannels=%d, kernelSize=%d\n",
        inChannels,
        outChannels,
        kernelSize);
    Conv2DLayerParams convParams;
    SLANG_RETURN_ON_FAIL(reader.readConv2DLayer(inChannels, outChannels, kernelSize, convParams));
    if (loadAndFuseBNorm)
    {
        logInfo("Loading and fusing BatchNorm2D Layer: numFeatures=%d\n", outChannels);
        BatchNorm2DLayerParams bnParams;
        SLANG_RETURN_ON_FAIL(reader.readBatchNorm2DLayer(outChannels, bnParams));
        convParams.fuseBatchNorm(bnParams);
    }

    SLANG_RETURN_ON_FAIL(loadParams(
        kernelSize,
        outChannels,
        convParams.weights.getBuffer(),
        convParams.biases.getBuffer()));
    return SLANG_OK;
}

SlangResult Conv2DKernel::loadParams(
    int kernelSize,
    int outputChannelCount,
    float* weightsData,
    float* biasesData)
{
    int weightsCount = kernelSize * kernelSize * inChannels * outputChannelCount;
    weightsBuffer = context->createBuffer(weightsData, weightsCount * sizeof(float));
    if (!weightsBuffer)
        return SLANG_FAIL;
    biasesBuffer = context->createBuffer(biasesData, outputChannelCount * sizeof(float));
    if (!biasesBuffer)
        return SLANG_FAIL;

    // Create Transposed Buffer (For Wave-Reduced Kernel)
    // Target Layout: [Out, K, K, In]
    // We do this AFTER fusion so the transposed weights include the BN scaling.
    {
        List<float> transposedWeights;
        transposedWeights.setCount(weightsCount);

        const float* src = weightsData;
        float* dst = transposedWeights.getBuffer();

        int K = kernelSize;
        int I = inChannels;
        int O = outChannels;

        // Iterate in Destination Order to keep writes sequential (CPU cache friendly)
        // Dest: o (outer), ky, kx, i (inner/contiguous)
        for (int o = 0; o < O; o++)
        {
            for (int ky = 0; ky < K; ky++)
            {
                for (int kx = 0; kx < K; kx++)
                {
                    for (int i = 0; i < I; i++)
                    {
                        // Source Index (Standard): [i, ky, kx, o]
                        // Stride Logic: i * (K*K*O) + ky * (K*O) + kx * O + o
                        int64_t srcIdx =
                            (int64_t)i * (K * K * O) + (int64_t)ky * (K * O) + (int64_t)kx * O + o;

                        // Dest Index (Transposed): [o, ky, kx, i]
                        // Stride Logic: o * (K*K*I) + ky * (K*I) + kx * I + i
                        int64_t dstIdx =
                            (int64_t)o * (K * K * I) + (int64_t)ky * (K * I) + (int64_t)kx * I + i;

                        dst[dstIdx] = src[srcIdx];
                    }
                }
            }
        }

        // 4. Create the Transposed Buffer
        weightsTransposedBuffer = context->createBuffer(transposedWeights);
        if (!weightsTransposedBuffer)
            return SLANG_FAIL;
    }
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

ComPtr<rhi::IBuffer> Conv2DKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* inputImage,
    int inputWidth,
    int inputHeight,
    int padding)
{
    int outputWidth = (inputWidth + padding * 2 - kernelSize) / stride + 1;
    int outputHeight = (inputHeight + padding * 2 - kernelSize) / stride + 1;
    String resultBufferName =
        name + "_" + String(outputWidth) + "x" + String(outputHeight) + "x" + String(outChannels);
    auto outputBuffer = task.allocateBuffer(
        resultBufferName.getBuffer(),
        outputWidth * outputHeight * outChannels * sizeof(float));
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

    int totalOutputValues = outputWidth * outputHeight * outChannels;

    // Depending on input/output shape, we will dispatch different kernels for better performance.

    if (outputWidth * outputHeight <= 16 && inChannels >= 32)
    {
        // 1. How many output values can one block handle?
        // Block Size (256) / Wave Size (32) = 8
        int valuesPerBlock = 256 / 32;

        // 2. Dispatch enough blocks to cover all values
        int numGroups = (totalOutputValues + valuesPerBlock - 1) / valuesPerBlock;

        params.weights = weightsTransposedBuffer->getDeviceAddress();
        task.dispatchKernel(flatWaveReducePipeline, numGroups, 1, 1, params);
    }
    else if (outputWidth * outputHeight <= 1024)
    {
        int numGroups = (totalOutputValues + 255) / 256;
        task.dispatchKernel(flatPipeline, numGroups, 1, 1, params);
    }
    else
    {
        static const int batchOutChannels = 32;
        int zBlocks = (outChannels + batchOutChannels - 1) / batchOutChannels;
        task.dispatchKernel(
            tilePipeline,
            (outputWidth + tileSize - 1) / tileSize,
            (outputHeight + tileSize - 1) / tileSize,
            zBlocks,
            params);
    }
    return ComPtr<rhi::IBuffer>(outputBuffer);
}

#include "conv2d.h"

static const int kTinyKernelMaxOutputPixels = 4;

Conv2DKernel::Conv2DKernel(
    InferencingContext* context,
    int tileSize,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    Expr inputExpr,
    Expr outputExpr,
    SinkExpr sinkExpr,
    String name)
    : context(context)
    , tileSize(tileSize)
    , kernelSize(kernelSize)
    , stride(stride)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , sinkExpr(sinkExpr)
    , name(name)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);
    outputProgram = compileExprToProgram(outputExpr, &globalRegCounter);
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        inputProgram.getSlangTypeName(),
        outputProgram.getSlangTypeName(),
        sinkExpr.node->getSlangTypeName()};

    tilePipeline = context->createComputePipeline("tiledConvolution", makeArrayView(specArgs));

    String flatArgs[] = {
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        inputProgram.getSlangTypeName(),
        outputProgram.getSlangTypeName(),
        sinkExpr.node->getSlangTypeName()};
    // Note: tileSize is NOT needed for flat kernel generic
    flatPipeline = context->createComputePipeline("flatConvolution", makeArrayView(flatArgs));
    flatWaveReducePipeline =
        context->createComputePipeline("flatConvolutionWaveReduce", makeArrayView(flatArgs));
}

Conv2DKernel::Conv2DKernel(
    InferencingContext* context,
    int tileSize,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    Expr outputExpr,
    String name)
    : Conv2DKernel(
          context,
          tileSize,
          kernelSize,
          stride,
          inChannels,
          outChannels,
          buffer(),
          outputExpr,
          bufferSink(),
          name)
{
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
    weightsBuffer = context->createPersistentBuffer(weightsData, weightsCount * sizeof(float));
    if (!weightsBuffer)
        return SLANG_FAIL;
    biasesBuffer = context->createPersistentBuffer(biasesData, outputChannelCount * sizeof(float));
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
        weightsTransposedBuffer = context->createPersistentBuffer(transposedWeights);
        if (!weightsTransposedBuffer)
            return SLANG_FAIL;
    }
    return SLANG_OK;
}

TensorView Conv2DKernel::allocateResultBuffer(
    ElementType elementType,
    int inputWidth,
    int inputHeight,
    int padding,
    int batchSize)
{
    int outputWidth = (inputWidth + padding * 2 - kernelSize) / stride + 1;
    int outputHeight = (inputHeight + padding * 2 - kernelSize) / stride + 1;

    // Naming for debug
    String resultBufferName = name + "_" + String(outputWidth) + "x" + String(outputHeight) + "x" +
                              String(outChannels) + "_B" + String(batchSize);

    // Allocation includes BatchSize
    auto outputBuffer = context->allocScratchTensor(
        elementType,
        Shape(batchSize, outputHeight, outputWidth, outChannels),
        resultBufferName.getBuffer());
    return outputBuffer;
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
    int batchSize;
};

void Conv2DKernel::queueExecute(
    InferencingTask& task,
    EvalContext& ctx,
    TensorView output,
    int padding)
{
    auto inputShape = inputProgram.resolveShape(ctx);
    if (inputShape.dims.getLast() != inChannels)
        throw std::runtime_error("Conv2DKernel: Input channel count mismatch.");
    if (inputShape.getRank() != 4)
    {
        throw std::runtime_error("Conv2DKernel: Input shape must be [B, H, W, C].");
    }
    int batchSize = inputShape[0];
    int inputHeight = inputShape[1];
    int inputWidth = inputShape[2];
    int outputWidth = (inputWidth + padding * 2 - kernelSize) / stride + 1;
    int outputHeight = (inputHeight + padding * 2 - kernelSize) / stride + 1;

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};
    inputProgram.pack(writer, ctx);
    outputProgram.pack(writer, ctx);

    SinkExprEvalContext sinkContext;
    sinkContext.outputBuffer = output;
    sinkContext.logicalShape = Shape(batchSize, outputHeight, outputWidth, outChannels);
    sinkExpr.node->pack(writer, sinkContext);
    writer.align(8);
    writer.write(weightsBuffer->getDeviceAddress());
    writer.write(biasesBuffer->getDeviceAddress());
    writer.write(weightsTransposedBuffer->getDeviceAddress());
    writer.write<int>(inputWidth);
    writer.write<int>(inputHeight);
    writer.write<int>(outputWidth);
    writer.write<int>(outputHeight);
    writer.write<int>(padding);
    writer.write<int>(batchSize); // Set Batch Size
    writer.finish();

    int totalOutputValues = outputWidth * outputHeight * outChannels;
    // For dispatch calculations, we multiply total work by batch size where linear indexing is
    // used.

    if (outputWidth * outputHeight <= 16 && inChannels >= 32)
    {
        // FlatWaveReduce
        // Thread Group (256) handles a contiguous chunk of (Batch * Pixel * Channel)
        // Global Linear Index covers everything.
        int totalWorkItems = totalOutputValues * batchSize;

        int valuesPerBlock = 256 / 32; // Each warp takes 1 item
        int numGroups = (totalWorkItems + valuesPerBlock - 1) / valuesPerBlock;

        task.dispatchKernel(flatWaveReducePipeline, numGroups, 1, 1, paramData);
    }
    else if (outputWidth * outputHeight <= 1024)
    {
        // Flat Kernel
        int totalWorkItems = totalOutputValues * batchSize;
        int numGroups = (totalWorkItems + 255) / 256;
        task.dispatchKernel(flatPipeline, numGroups, 1, 1, paramData);
    }
    else
    {
        // Tiled Kernel
        static const int batchOutChannels = 32;
        int zBlocksPerImage = (outChannels + batchOutChannels - 1) / batchOutChannels;

        // Z Dimension now handles (Channels * Batches)
        int totalZBlocks = zBlocksPerImage * batchSize;

        task.dispatchKernel(
            tilePipeline,
            (outputWidth + tileSize - 1) / tileSize,
            (outputHeight + tileSize - 1) / tileSize,
            totalZBlocks,
            paramData);
    }
}

void Conv2DKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView inputImage,
    int padding)
{
    EvalContext ctx;
    if (inputProgram.bufferNodes.getCount() > 1)
    {
        throw std::runtime_error("insufficient input buffers for Conv2D kernel.");
    }
    if (inputProgram.bufferNodes.getCount() < 1)
    {
        throw std::runtime_error("The Conv2D kernel does not take any input buffers.");
    }
    ctx.inputs.add(inputProgram.bufferNodes[0], inputImage);

    queueExecute(task, ctx, output, padding);
}

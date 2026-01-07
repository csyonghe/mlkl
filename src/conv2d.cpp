#include "conv2d.h"
#include "safetensors-reader.h"

static const int kTinyKernelMaxOutputPixels = 4;

Conv2DKernel::Conv2DKernel(
    InferencingContext* context,
    ElementType elementType,
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
    , elementType(elementType)
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

    String elemTypeName = getSlangElementTypeName(elementType);
    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        outputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};

    tilePipeline = context->createComputePipeline("tiledConvolution", makeArrayView(specArgs));

    String flatArgs[] = {
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        outputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    // Note: tileSize is NOT needed for flat kernel generic
    flatPipeline = context->createComputePipeline("flatConvolution", makeArrayView(flatArgs));
    flatWaveReducePipeline =
        context->createComputePipeline("flatConvolutionWaveReduce", makeArrayView(flatArgs));
}

void Conv2DKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("Conv2DKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
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
          ElementType::Float32,
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

// Permutation constants for Conv2D weight layouts
// PyTorch Conv2D: [OutCh, InCh, Ky, Kx] -> Engine: [InCh, Ky, Kx, OutCh]
static const int kConv2DWeightPermutation[] = {1, 2, 3, 0};
// Transposed layout for wave-reduced kernel: [OutCh, Ky, Kx, InCh]
static const int kConv2DWeightTransposedPermutation[] = {0, 2, 3, 1};

SlangResult Conv2DKernel::loadParams(
    SafeTensorsReader& reader,
    UnownedStringSlice weightName,
    UnownedStringSlice biasName)
{
    logInfo(
        "Loading Conv2D Layer from SafeTensors: inChannels=%d, outChannels=%d, kernelSize=%d\n",
        inChannels,
        outChannels,
        kernelSize);

    // Read and permute weights from [OutCh, InCh, K, K] to [InCh, K, K, OutCh]
    List<uint8_t> weightsData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(
        weightName,
        elementType,
        weightsData,
        makeConstArrayView(kConv2DWeightPermutation)));

    weightsBuffer = context->createPersistentBuffer(weightsData.getBuffer(), weightsData.getCount());
    if (!weightsBuffer)
        return SLANG_FAIL;

    // Read bias
    List<uint8_t> biasData;
    if (biasName.getLength() > 0 && reader.hasTensor(biasName))
    {
        SLANG_RETURN_ON_FAIL(reader.readTensor(biasName, elementType, biasData));
    }
    else
    {
        // No bias - initialize to zeros
        size_t biasSize = outChannels * getElementTypeSize(elementType);
        biasData.setCount(biasSize);
        memset(biasData.getBuffer(), 0, biasSize);
    }

    biasesBuffer = context->createPersistentBuffer(biasData.getBuffer(), biasData.getCount());
    if (!biasesBuffer)
        return SLANG_FAIL;

    // Create transposed weights buffer for wave-reduced kernel
    // [OutCh, InCh, K, K] -> [OutCh, K, K, InCh]
    List<uint8_t> transposedData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(
        weightName,
        elementType,
        transposedData,
        makeConstArrayView(kConv2DWeightTransposedPermutation)));

    weightsTransposedBuffer = context->createPersistentBuffer(
        transposedData.getBuffer(), transposedData.getCount());
    if (!weightsTransposedBuffer)
        return SLANG_FAIL;

    return SLANG_OK;
}

SlangResult Conv2DKernel::loadParams(
    int kernelSize,
    int outputChannelCount,
    float* weightsData,
    float* biasesData)
{
    int weightsCount = kernelSize * kernelSize * inChannels * outputChannelCount;
    
    // Convert weights and biases to the kernel's element type
    auto weightsConverted = convertFloatData(weightsData, weightsCount, elementType);
    weightsBuffer = context->createPersistentBuffer(weightsConverted.getBuffer(), weightsConverted.getCount());
    if (!weightsBuffer)
        return SLANG_FAIL;
    
    auto biasConverted = convertFloatData(biasesData, outputChannelCount, elementType);
    biasesBuffer = context->createPersistentBuffer(biasConverted.getBuffer(), biasConverted.getCount());
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

        // Convert transposed weights to element type
        auto transposedConverted = convertFloatData(transposedWeights, elementType);
        weightsTransposedBuffer = context->createPersistentBuffer(
            transposedConverted.getBuffer(), transposedConverted.getCount());
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
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto bufferNode : inputProgram.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input");
    }

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

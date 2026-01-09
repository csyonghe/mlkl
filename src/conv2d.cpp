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

    // ========================================================================
    // Create GEMM-style tiled convolution (for ConvolutionAlgorithm::Gemm)
    // ========================================================================
    // This kernel caches both weights AND input in shared memory for maximum reuse.
    // Uses the same ConvolutionParams struct as tiledConvolution/flatConvolution.
    // Note: tileSize is included for API consistency but not used by gemmConvolution.
    gemmPipeline = context->createComputePipeline("gemmConvolution", makeArrayView(specArgs));
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
    // Validate that outputExpr is well-formed.
    // It's a mistake if outputExpr contains buffer() but no kernelOutput().
    // This catches the common error of passing silu(buffer()) as outputExpr instead of inputExpr.
    if (!isValidOutputExpr(outputExpr))
    {
        throw std::runtime_error(
            "Conv2DKernel: outputExpr contains buffer() but no kernelOutput() - this is likely a mistake. "
            "buffer() is for INPUT expressions. For output transformations, use kernelOutput(). "
            "If you need a custom input expression (like silu(buffer())), use the full constructor:\n"
            "  Conv2DKernel(ctx, elemType, tileSize, kernelSize, stride, inCh, outCh, "
            "inputExpr, outputExpr, sinkExpr)");
    }
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
    int padding,
    ConvolutionAlgorithm algorithm)
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

    int totalOutputValues = outputWidth * outputHeight * outChannels;

    // Determine which algorithm to use
    ConvolutionAlgorithm effectiveAlgorithm = algorithm;
    if (algorithm == ConvolutionAlgorithm::Auto)
    {
        // Use GEMM-style tiled convolution for best performance
        // It caches both weights and input in shared memory
        effectiveAlgorithm = ConvolutionAlgorithm::Gemm;
    }

    // GEMM-style tiled convolution (caches both weights and input)
    if (effectiveAlgorithm == ConvolutionAlgorithm::Gemm)
    {
        executeGemmConv(task, ctx, output, padding);
        return;
    }

    // Standard tiled/flat kernels
    List<uint8_t> paramData;
    ParameterWriter writer{paramData};
    // Set kernelOutputShape for outputExpr to resolve kernelOutput() shape
    ctx.kernelOutputShape = Shape(batchSize, outputHeight, outputWidth, outChannels);
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
    writer.write<int>(batchSize);
    writer.finish();

    if (effectiveAlgorithm == ConvolutionAlgorithm::Flat)
    {
        if (outputWidth * outputHeight <= 16 && inChannels >= 32)
        {
            // FlatWaveReduce for very small spatial sizes
            int totalWorkItems = totalOutputValues * batchSize;
            int valuesPerBlock = 256 / 32;
            int numGroups = (totalWorkItems + valuesPerBlock - 1) / valuesPerBlock;
            task.dispatchKernel(flatWaveReducePipeline, numGroups, 1, 1, paramData);
        }
        else
        {
            // Standard Flat Kernel
            int totalWorkItems = totalOutputValues * batchSize;
            int numGroups = (totalWorkItems + 255) / 256;
            task.dispatchKernel(flatPipeline, numGroups, 1, 1, paramData);
        }
    }
    else // Tiled
    {
        static const int batchOutChannels = 32;
        int zBlocksPerImage = (outChannels + batchOutChannels - 1) / batchOutChannels;
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
    const std::initializer_list<InputInfo>& inputs,
    int padding,
    ConvolutionAlgorithm algorithm)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("Conv2DKernel: insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : outputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    queueExecute(task, ctx, output, padding, algorithm);
}

void Conv2DKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView inputImage,
    int padding,
    ConvolutionAlgorithm algorithm)
{
    EvalContext ctx;
    if (inputProgram.bufferNodes.getCount() > 1)
    {
        throw std::runtime_error("Conv2DKernel: kernel requires multiple input buffers, use initializer_list overload.");
    }
    if (inputProgram.bufferNodes.getCount() == 1)
    {
        ctx.inputs.add(inputProgram.bufferNodes[0], inputImage);
    }
    if (outputProgram.bufferNodes.getCount() > 0)
    {
        throw std::runtime_error("Conv2DKernel: kernel requires additional output buffers, use initializer_list overload.");
    }
    queueExecute(task, ctx, output, padding, algorithm);
}

// ============================================================================
// GEMM-style tiled convolution
// ============================================================================
// Uses shared memory for both weights and input for maximum data reuse.
// Tile sizes defined in convolution.slang:
//   GEMM_TILE_OH = 8, GEMM_TILE_OW = 8, GEMM_TILE_OC = 32, GEMM_TILE_IC = 8

void Conv2DKernel::executeGemmConv(
    InferencingTask& task,
    EvalContext& ctx,
    TensorView output,
    int padding)
{
    // Validate element types
    validateTensorElementType(output, "output");

    // Get input shape from the input expression
    Shape inputShape = inputProgram.resolveShape(ctx);
    if (inputShape.getRank() != 4)
        throw std::runtime_error("Conv2DKernel: Input shape must be [B, H, W, C].");

    int batchSize = inputShape[0];
    int inputHeight = inputShape[1];
    int inputWidth = inputShape[2];
    int inputChannels = inputShape[3];

    if (inputChannels != inChannels)
        throw std::runtime_error("Conv2DKernel: Input channel count mismatch.");

    // Compute output dimensions
    int outputHeight = (inputHeight + padding * 2 - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + padding * 2 - kernelSize) / stride + 1;

    // Pack parameters - same format as tiledConvolution (ConvolutionParams struct)
    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack input/output expressions and sink
    // Set kernelOutputShape for outputExpr to resolve kernelOutput() shape
    ctx.kernelOutputShape = Shape(batchSize, outputHeight, outputWidth, outChannels);
    inputProgram.pack(writer, ctx);
    outputProgram.pack(writer, ctx);

    SinkExprEvalContext sinkContext;
    sinkContext.outputBuffer = output;
    sinkContext.logicalShape = Shape(batchSize, outputHeight, outputWidth, outChannels);
    sinkExpr.node->pack(writer, sinkContext);

    // Pack weight/bias pointers (same layout as tiledConvolution)
    writer.align(8);
    writer.write(weightsBuffer->getDeviceAddress());           // weights (not used by gemmConv)
    writer.write(biasesBuffer->getDeviceAddress());            // bias
    writer.write(weightsTransposedBuffer->getDeviceAddress()); // weightsIOKK [OC, KH, KW, IC]

    // Pack dimension parameters
    writer.write<int>(inputWidth);
    writer.write<int>(inputHeight);
    writer.write<int>(outputWidth);
    writer.write<int>(outputHeight);
    writer.write<int>(padding);
    writer.write<int>(batchSize);
    writer.finish();

    // Dispatch: tile sizes from convolution.slang
    static const int GEMM_TILE_OH = 16;
    static const int GEMM_TILE_OW = 16;
    static const int GEMM_TILE_OC = 16;

    int numOCTiles = (outChannels + GEMM_TILE_OC - 1) / GEMM_TILE_OC;

    task.dispatchKernel(
        gemmPipeline,
        (outputWidth + GEMM_TILE_OW - 1) / GEMM_TILE_OW,
        (outputHeight + GEMM_TILE_OH - 1) / GEMM_TILE_OH,
        batchSize * numOCTiles,
        paramData);
}

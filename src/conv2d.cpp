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
    // GEMM Convolution Tile Configuration - Tuning Summary
    // ========================================================================
    // Extensive benchmarking (batchSize=2, SD 1.5 configs) explored:
    //   - TILE_OH/OW: 8, 16
    //   - TILE_OC: 2, 4, 8, 16, 32
    //   - TILE_IC: 4, 8, 16, 32
    //
    // Key findings:
    //   1. TILE_IC=8 is optimal for all configs. Larger values (16, 32) cause
    //      ~2x slowdown due to register pressure and reduced occupancy.
    //      Smaller (IC=4) also slower.
    //
    //   2. For outputs <= 8x8:
    //      - 8x8 spatial tiles are better than 16x16 (100% vs 25% thread utilization)
    //      - TILE_OC=8 is 12-24% faster than TILE_OC=16
    //      - Combined improvement: ~4% over baseline
    //
    //   3. For outputs > 8x8:
    //      - 16x16 spatial tiles are optimal (thread waste not an issue)
    //      - TILE_OC=16 is slightly better than TILE_OC=8
    //
    //   4. Further gains require algorithmic changes (double buffering, Winograd)
    //      or FP16/Tensor Cores.
    // ========================================================================

    // Default pipeline: 16x16 spatial, TILE_OC=16, TILE_IC=8
    gemmConfig = GemmTileConfig::defaultConfig();
    gemmPipeline = createGemmPipelineWithConfig(gemmConfig);

    // Small-spatial pipeline for outputs <= 8x8: 8x8 spatial, TILE_OC=8, TILE_IC=8
    gemmSmallSpatialConfig = GemmTileConfig::defaultConfig();
    gemmSmallSpatialConfig.tileOH = 8;
    gemmSmallSpatialConfig.tileOW = 8;
    gemmSmallSpatialConfig.tileOC = 8;
    gemmSmallSpatialPipeline = createGemmPipelineWithConfig(gemmSmallSpatialConfig);

    // Wave shuffle pipeline: same config but uses wave intrinsics instead of weight shared memory
    gemmWaveShufflePipeline = createGemmWaveShufflePipeline(gemmConfig);

    // Winograd pipeline for 3x3 stride=1 convolutions
    if (kernelSize == 3 && stride == 1)
    {
        createWinogradPipeline();
    }
}

ComPtr<rhi::IComputePipeline> Conv2DKernel::createGemmPipelineWithConfig(const GemmTileConfig& config)
{
    String elemTypeName = getSlangElementTypeName(elementType);

    // ========================================================================
    // Create GEMM-style tiled convolution pipeline
    // ========================================================================
    // This kernel caches both weights AND input in shared memory for maximum reuse.
    // Each thread computes THREAD_OH x THREAD_OW x TILE_OC outputs.
    //
    // Shared memory constraint for stride=2, kernelSize=3:
    //   INPUT_TILE_H = (TILE_OH-1)*stride + kernelSize
    //   s_input = TILE_IC * INPUT_TILE_H * INPUT_TILE_W * 4 bytes
    //   Must fit in ~48KB along with s_weight
    String gemmArgs[] = {
        String(config.tileOH),
        String(config.tileOW),
        String(config.tileOC),
        String(config.tileIC),
        String(config.threadOH),
        String(config.threadOW),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        outputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    return context->createComputePipeline("gemmConvolution", makeArrayView(gemmArgs));
}

ComPtr<rhi::IComputePipeline> Conv2DKernel::createGemmWaveShufflePipeline(const GemmTileConfig& config)
{
    String elemTypeName = getSlangElementTypeName(elementType);

    // Same arguments as gemmConvolution, but calls wave shuffle version
    String gemmArgs[] = {
        String(config.tileOH),
        String(config.tileOW),
        String(config.tileOC),
        String(config.tileIC),
        String(config.threadOH),
        String(config.threadOW),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        outputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    return context->createComputePipeline("gemmConvolutionWaveShuffle", makeArrayView(gemmArgs));
}

void Conv2DKernel::createWinogradPipeline()
{
    String elemTypeName = getSlangElementTypeName(elementType);

    // Winograd F(4x4, 3x3) pipeline
    // TILE_OC and TILE_IC are tunable parameters
    const int TILE_OC = 16;  // Output channels per block
    const int TILE_IC = 8;   // Input channels per K-iteration

    String winogradArgs[] = {
        String(TILE_OC),
        String(TILE_IC),
        String(inChannels),
        String(outChannels),
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        outputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    winogradPipeline = context->createComputePipeline("winogradConvolution", makeArrayView(winogradArgs));
}

void Conv2DKernel::transformWeightsToWinograd()
{
    // Transform 3x3 weights to Winograd domain using G transform
    // G matrix for F(4,3):
    // [ 1/4     0       0    ]
    // [-1/6   -1/6    -1/6   ]
    // [-1/6    1/6    -1/6   ]
    // [1/24   1/12    1/6    ]
    // [1/24  -1/12    1/6    ]
    // [ 0      0       1     ]
    //
    // For each filter: U = G * g * GT  (where g is 3x3 filter, U is 6x6 transformed)

    if (!weightsTransposedBuffer)
        return;

    // Read weights from GPU
    int weightsSize = outChannels * inChannels * 9;  // 3x3 = 9
    List<float> weights;
    weights.setCount(weightsSize);

    // weightsTransposedBuffer is a raw IBuffer, read directly
    context->getDevice()->readBuffer(
        weightsTransposedBuffer, 
        0,  // offset
        weightsSize * sizeof(float), 
        weights.getBuffer());

    // G matrix for F(4,3) - transforms 3x3 to 6x6
    const float G[6][3] = {
        { 1.0f/4,       0,       0},
        {-1.0f/6, -1.0f/6, -1.0f/6},
        {-1.0f/6,  1.0f/6, -1.0f/6},
        {1.0f/24, 1.0f/12,  1.0f/6},
        {1.0f/24,-1.0f/12,  1.0f/6},
        {      0,       0,       1}
    };

    // Allocate transformed weights: [outChannels, inChannels, 6, 6]
    int transformedSize = outChannels * inChannels * 36;
    List<float> transformedWeights;
    transformedWeights.setCount(transformedSize);

    // Transform each 3x3 filter to 6x6
    for (int oc = 0; oc < outChannels; oc++)
    {
        for (int ic = 0; ic < inChannels; ic++)
        {
            // Get 3x3 filter (layout: [OC, KH, KW, IC] for weightsTransposed)
            // Actually need to check the exact layout...
            // weightsTransposed is [OC, KH, KW, IC] based on the GEMM kernel
            float g[3][3];
            for (int ky = 0; ky < 3; ky++)
            {
                for (int kx = 0; kx < 3; kx++)
                {
                    int idx = oc * (3 * 3 * inChannels) + ky * (3 * inChannels) + kx * inChannels + ic;
                    g[ky][kx] = weights[idx];
                }
            }

            // Compute G * g (6x3 result)
            float temp[6][3];
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    temp[i][j] = G[i][0] * g[0][j] + G[i][1] * g[1][j] + G[i][2] * g[2][j];
                }
            }

            // Compute (G * g) * GT = temp * GT (6x6 result)
            float U[6][6];
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    U[i][j] = temp[i][0] * G[j][0] + temp[i][1] * G[j][1] + temp[i][2] * G[j][2];
                }
            }

            // Store transformed weights: [OC, IC, 6, 6]
            for (int ky = 0; ky < 6; ky++)
            {
                for (int kx = 0; kx < 6; kx++)
                {
                    int idx = oc * (inChannels * 36) + ic * 36 + ky * 6 + kx;
                    transformedWeights[idx] = U[ky][kx];
                }
            }
        }
    }

    // Upload to GPU
    winogradWeightsBuffer = context->createPersistentBuffer(transformedWeights, "winograd_weights");
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

Conv2DKernel::Conv2DKernel(
    InferencingContext* context,
    int tileSize,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    const GemmTileConfig& gemmTileConfig,
    String name)
    : context(context)
    , elementType(ElementType::Float32)
    , tileSize(tileSize)
    , kernelSize(kernelSize)
    , stride(stride)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , sinkExpr(bufferSink())
    , name(name)
{
    // Simplified constructor for benchmarking with custom tile config
    // Uses default buffer() input and kernelOutput() output expressions
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(buffer(), &globalRegCounter);
    outputProgram = compileExprToProgram(kernelOutput(), &globalRegCounter);

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
    flatPipeline = context->createComputePipeline("flatConvolution", makeArrayView(flatArgs));
    flatWaveReducePipeline =
        context->createComputePipeline("flatConvolutionWaveReduce", makeArrayView(flatArgs));

    // Create GEMM pipeline with custom configuration
    gemmConfig = gemmTileConfig;
    gemmPipeline = createGemmPipelineWithConfig(gemmTileConfig);
    // For custom config, use same config for small-spatial (no auto-selection)
    gemmSmallSpatialConfig = gemmTileConfig;
    gemmSmallSpatialPipeline = gemmPipeline;
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

        // Transform weights to Winograd domain if applicable (3x3 stride=1)
        if (winogradPipeline && kernelSize == 3 && stride == 1)
        {
            transformWeightsToWinograd();
        }
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

    // Winograd convolution for 3x3 stride=1
    if (effectiveAlgorithm == ConvolutionAlgorithm::Winograd)
    {
        if (!winogradPipeline || kernelSize != 3 || stride != 1)
        {
            throw std::runtime_error("Winograd algorithm requires 3x3 kernel with stride=1");
        }
        executeWinogradConv(task, ctx, output, padding);
        return;
    }

    // GEMM-style tiled convolution (caches both weights and input)
    if (effectiveAlgorithm == ConvolutionAlgorithm::Gemm)
    {
        executeGemmConv(task, ctx, output, padding);
        return;
    }

    // GEMM with wave shuffle (uses warp shuffle instead of weight shared memory)
    if (effectiveAlgorithm == ConvolutionAlgorithm::GemmWaveShuffle)
    {
        executeGemmWaveShuffleConv(task, ctx, output, padding);
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
// GEMM-style tiled convolution with register blocking
// ============================================================================
// Uses shared memory for both weights and input for maximum data reuse.
// Register blocking: each thread computes THREAD_OH x THREAD_OW x TILE_OC outputs.
// Tile parameters are stored in gemmConfig and used at dispatch time.

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

    // Use small-spatial pipeline for outputs <= 8x8
    // This uses TILE_OH=8, TILE_OW=8, TILE_OC=8 which:
    // 1. Avoids thread waste (16x16 tile with 8x8 output = 75% idle threads)
    // 2. Uses TILE_OC=8 which benchmark showed is 12-24% faster for small spatial
    bool useSmallSpatialPipeline = 
        (outputWidth <= 8 && outputHeight <= 8);

    const GemmTileConfig& config = useSmallSpatialPipeline ? gemmSmallSpatialConfig : gemmConfig;

    // Dispatch: tile sizes from selected config
    int numOCTiles = (outChannels + config.tileOC - 1) / config.tileOC;

    auto& pipeline = useSmallSpatialPipeline ? gemmSmallSpatialPipeline : gemmPipeline;

    task.dispatchKernel(
        pipeline,
        (outputWidth + config.tileOW - 1) / config.tileOW,
        (outputHeight + config.tileOH - 1) / config.tileOH,
        batchSize * numOCTiles,
        paramData);
}

void Conv2DKernel::executeGemmWaveShuffleConv(
    InferencingTask& task,
    EvalContext& ctx,
    TensorView output,
    int padding)
{
    // Same as executeGemmConv but uses wave shuffle pipeline
    auto inputShape = inputProgram.resolveShape(ctx);
    int batchSize = inputShape[0];
    int inputHeight = inputShape[1];
    int inputWidth = inputShape[2];

    if (inputShape[3] != inChannels)
        throw std::runtime_error("Conv2DKernel: Input channel count mismatch.");

    int outputHeight = (inputHeight + padding * 2 - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + padding * 2 - kernelSize) / stride + 1;

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

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

    const GemmTileConfig& config = gemmConfig;
    int numOCTiles = (outChannels + config.tileOC - 1) / config.tileOC;

    task.dispatchKernel(
        gemmWaveShufflePipeline,
        (outputWidth + config.tileOW - 1) / config.tileOW,
        (outputHeight + config.tileOH - 1) / config.tileOH,
        batchSize * numOCTiles,
        paramData);
}

// ============================================================================
// Winograd F(4x4, 3x3) Convolution
// ============================================================================
// Reduces multiplications from 9 to ~2.25 per output for 3x3 stride=1 convs.
// Requires pre-transformed weights (computed in transformWeightsToWinograd).

void Conv2DKernel::executeWinogradConv(
    InferencingTask& task,
    EvalContext& ctx,
    TensorView output,
    int padding)
{
    auto inputShape = inputProgram.resolveShape(ctx);
    int batchSize = inputShape[0];
    int inputHeight = inputShape[1];
    int inputWidth = inputShape[2];
    int outputWidth = (inputWidth + padding * 2 - 3) / 1 + 1;  // kernelSize=3, stride=1
    int outputHeight = (inputHeight + padding * 2 - 3) / 1 + 1;

    // Set kernelOutputShape for outputExpr
    ctx.kernelOutputShape = Shape(batchSize, outputHeight, outputWidth, outChannels);

    // Build parameter data for WinogradParams struct
    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    inputProgram.pack(writer, ctx);
    outputProgram.pack(writer, ctx);

    SinkExprEvalContext sinkContext;
    sinkContext.outputBuffer = output;
    sinkContext.logicalShape = Shape(batchSize, outputHeight, outputWidth, outChannels);
    sinkExpr.node->pack(writer, sinkContext);

    writer.align(8);
    writer.write(biasesBuffer->getDeviceAddress());
    writer.write(winogradWeightsBuffer->getDeviceAddress());
    writer.write<int>(inputWidth);
    writer.write<int>(inputHeight);
    writer.write<int>(outputWidth);
    writer.write<int>(outputHeight);
    writer.write<int>(padding);
    writer.write<int>(batchSize);
    writer.finish();

    // Dispatch: each tile produces 4x4 outputs
    const int TILE_SIZE = 4;
    const int TILE_OC = 16;  // Must match pipeline creation
    int numOCTiles = (outChannels + TILE_OC - 1) / TILE_OC;

    task.dispatchKernel(
        winogradPipeline,
        (outputWidth + TILE_SIZE - 1) / TILE_SIZE,
        (outputHeight + TILE_SIZE - 1) / TILE_SIZE,
        batchSize * numOCTiles,
        paramData);
}

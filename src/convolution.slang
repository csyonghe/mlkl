implementing mlkl;

static const int inputChannelBatchSize = 8;
static const int outputChannelBatchSize = 32;

// --- Shader Parameters ---
struct ConvolutionParams<T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>
{
    TInput inputImage; // Layout: [batch, height, width, channels]
    FOutput outputTransform; // Layout: [batch, height, width, channels]
    TSink outputImage;
    // Layout: [inChannels, kernelSize, kernelSize, outChannels]
    T* weights;         // Weights in element type T
    T* bias;            // Bias in element type T
    T* weightsIOKK;     // Transposed weights in element type T          
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int padding;
    int batchSize;
};

// [numthreads(tileSize, tileSize, 1)]
// Dispatch Z = (OutChannels/32) * BatchSize
[numthreads(tileSize, tileSize, 1)]
void tiledConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels, T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<ConvolutionParams<T, TInput, FOutput, TSink>, CDataLayout> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    // --- 1. Batch & Channel Decoding ---
    uint numChannelGroups = (outChannels + outputChannelBatchSize - 1) / outputChannelBatchSize;
    uint batchIdx = groupId.z / numChannelGroups;
    uint channelGroupIdx = groupId.z % numChannelGroups;

    if (batchIdx >= params.batchSize) return;

    // --- 2. Coordinates & Setup ---
    const int2 outPos = groupId.xy * int2(tileSize) + groupThreadId.xy;
    const int2 tileOrigin = groupId.xy * int2(tileSize) * stride - int2(params.padding);
    const int outChannelStart = channelGroupIdx * outputChannelBatchSize;

    // --- 3. Shared Memory Allocation ---
    static const int sharedDim = (tileSize - 1) * stride + kernelSize;
    static groupshared T.UnpackedType s_inputTile[inputChannelBatchSize][sharedDim * sharedDim];

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // --- 4. Output Accumulators (in T.UnpackedType precision) ---
    T.UnpackedType outputAccumulators[outputChannelBatchSize];
    for (int o = 0; o < outputChannelBatchSize; ++o)
    {
        int globalOutC = outChannelStart + o;
        if (globalOutC < outChannels)
            outputAccumulators[o] = params.bias[globalOutC].unpack();
        else
            outputAccumulators[o] = T.UnpackedType(0);
    }

    // --- 5. Main Loop ---
    for (int inCBase = 0; inCBase < inChannels; inCBase += inputChannelBatchSize)
    {
        // A. Load Input Tile
        const int numSharedPixels = sharedDim * sharedDim;
        const int numThreads = tileSize * tileSize;
        const int linearThreadIndex = groupThreadId.y * tileSize + groupThreadId.x;

        for (int i = linearThreadIndex; i < numSharedPixels; i += numThreads)
        {
            int sY = i / sharedDim;
            int sX = i % sharedDim;
            int globalInY = tileOrigin.y + sY;
            int globalInX = tileOrigin.x + sX;

            bool inBounds = globalInY >= 0 && globalInY < params.inputImageHeight &&
                            globalInX >= 0 && globalInX < params.inputImageWidth;

            for (int k = 0; k < inputChannelBatchSize; ++k)
            {
                int currentInC = inCBase + k;
                if (inBounds && currentInC < inChannels)
                {
                    Coord inCoord = Coord(batchIdx, globalInY, globalInX, currentInC);
                    s_inputTile[k][i] = params.inputImage.eval(inCoord, emptyInput);
                }
                else
                {
                    s_inputTile[k][i] = T.UnpackedType(0);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // B. Convolution Math
        if (outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight)
        {
            for (int ky = 0; ky < kernelSize; ++ky)
            {
                for (int kx = 0; kx < kernelSize; ++kx)
                {
                    int sY = groupThreadId.y * stride + ky;
                    int sX = groupThreadId.x * stride + kx;
                    int sharedIdx = sY * sharedDim + sX;

                    for (int inOffset = 0; inOffset < inputChannelBatchSize; ++inOffset)
                    {
                        int globalInC = inCBase + inOffset;
                        if (globalInC >= inChannels) break;

                        T.UnpackedType pixelVal = s_inputTile[inOffset][sharedIdx];

                        [unroll]
                        for (int outOffset = 0; outOffset < outputChannelBatchSize; ++outOffset)
                        {
                            int globalOutC = outChannelStart + outOffset;
                            if (globalOutC >= outChannels) continue;

                            int64_t wIdx = (int64_t)globalInC  * (kernelSize * kernelSize * outChannels) + 
                                           (int64_t)ky * (kernelSize * outChannels) + 
                                           (int64_t)kx * (outChannels) + 
                                           (int64_t)globalOutC;
                            outputAccumulators[outOffset] = outputAccumulators[outOffset] + pixelVal * params.weights[wIdx].unpack();
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // --- 6. Write Output ---
    if (outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight)
    {
        for (int o = 0; o < outputChannelBatchSize; ++o)
        {
            int globalOutC = outChannelStart + o;
            if (globalOutC < outChannels)
            {
                Coord outCoord = Coord(batchIdx, outPos.y, outPos.x, globalOutC);
                Input<T> outInput = {};
                outInput.value = outputAccumulators[o];
                params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
            }
        }
    }
}

// --- FLAT CONVOLUTION ---
[numthreads(256, 1, 1)]
void flatConvolution<int kernelSize, int stride, int inChannels, int outChannels, T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<ConvolutionParams<T, TInput, FOutput, TSink>> params,
    uint3 id : SV_DispatchThreadID)
{
    uint globalIdx = id.x;
    uint valuesPerImage = params.outputImageWidth * params.outputImageHeight * outChannels;
    uint totalOutputs = valuesPerImage * params.batchSize;
    
    if (globalIdx >= totalOutputs) return;

    // 1. Decode Batch
    uint batchIdx = globalIdx / valuesPerImage;
    uint localIdx = globalIdx % valuesPerImage;

    // 2. Decode Spatial
    uint outCh = localIdx % outChannels;
    uint spatialIdx = localIdx / outChannels;
    uint outX = spatialIdx % params.outputImageWidth;
    uint outY = spatialIdx / params.outputImageWidth;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // 3. Setup Accumulator (in T.UnpackedType precision)
    T.UnpackedType sum = T.UnpackedType(0);
    if (outCh < outChannels)
        sum = params.bias[outCh].unpack();

    int startX = (int)outX * stride - params.padding;
    int startY = (int)outY * stride - params.padding;

    for (int ky = 0; ky < kernelSize; ky++)
    {
        int inY = startY + ky;
        if (inY >= 0 && inY < params.inputImageHeight)
        {
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int inX = startX + kx;
                if (inX >= 0 && inX < params.inputImageWidth)
                {
                    int wBase = (ky * kernelSize * outChannels + kx * outChannels + outCh);
                    int wStride = kernelSize * kernelSize * outChannels;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        Coord inCoord = Coord(batchIdx, inY, inX, ic);
                        T.UnpackedType val = params.inputImage.eval(inCoord, emptyInput);
                        T.UnpackedType w = params.weights[ic * wStride + wBase].unpack();
                        sum = sum + val * w;
                    }
                }
            }
        }
    }

    Coord outCoord = Coord(batchIdx, outY, outX, outCh);
    Input<T> outInput = {};
    outInput.value = sum;
    params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
}

// --- TINY CONVOLUTION ---
[numthreads(256, 1, 1)]
void tinyConvolution<int kernelSize, int stride, int inChannels, int outChannels, int maxTotalPixels, T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<ConvolutionParams<T, TInput, FOutput, TSink>> params,
    uint3 id : SV_DispatchThreadID)
{
    uint outCh = id.x;
    uint batchIdx = id.y;

    if (outCh >= outChannels || batchIdx >= params.batchSize) return;

    int totalPixels = params.outputImageWidth * params.outputImageHeight;
    T.UnpackedType sums[maxTotalPixels];
    T.UnpackedType b = params.bias[outCh].unpack();
    
    [unroll]
    for (int i = 0; i < totalPixels; ++i) sums[i] = b;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    for (int ic = 0; ic < inChannels; ++ic)
    {
        int wBaseIC = ic * kernelSize * kernelSize * outChannels + outCh;
        for (int ky = 0; ky < kernelSize; ++ky)
        {
            for (int kx = 0; kx < kernelSize; ++kx)
            {
                int wOffset = ky * kernelSize * outChannels + kx * outChannels;
                T.UnpackedType w = params.weights[wBaseIC + wOffset].unpack();

                [unroll]
                for (int p = 0; p < totalPixels; ++p)
                {
                    int2 outCoord2 = int2(p % params.outputImageWidth, p / params.outputImageWidth);
                    int inY = outCoord2.y * stride - params.padding + ky;
                    int inX = outCoord2.x * stride - params.padding + kx;

                    if (inX >= 0 && inX < params.inputImageWidth && 
                        inY >= 0 && inY < params.inputImageHeight)
                    {
                        Coord inCoord = Coord(batchIdx, inY, inX, ic);
                        sums[p] = sums[p] + params.inputImage.eval(inCoord, emptyInput) * w;
                    }
                }
            }
        }
    }

    [unroll]
    for (int p = 0; p < totalPixels; ++p)
    {
        Coord outCoord = Coord(batchIdx, p / params.outputImageWidth, p % params.outputImageWidth, outCh);
        Input<T> outInput = {};
        outInput.value = sums[p];
        params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
    }
}

// --- WAVE CONVOLUTION ---
[numthreads(256, 1, 1)]
void flatConvolutionWaveReduce<int kernelSize, int stride, int inChannels, int outChannels, T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<ConvolutionParams<T, TInput, FOutput, TSink>> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    uint waveSize = WaveGetLaneCount();
    uint warpsPerBlock = 256 / waveSize;
    uint laneId = WaveGetLaneIndex();
    uint warpId = groupThreadId.x / waveSize;

    uint globalValueIdx = groupId.x * warpsPerBlock + warpId;
    
    uint valuesPerImage = params.outputImageWidth * params.outputImageHeight * outChannels;
    uint totalValues = valuesPerImage * params.batchSize;

    if (globalValueIdx >= totalValues) return;

    // Decode Batch
    uint batchIdx = globalValueIdx / valuesPerImage;
    uint localValueIdx = globalValueIdx % valuesPerImage;

    uint outCh = localValueIdx % outChannels;
    uint spatialIdx = localValueIdx / outChannels;
    uint outX = spatialIdx % params.outputImageWidth;
    uint outY = spatialIdx / params.outputImageWidth;

    T.UnpackedType partialSum = T.UnpackedType(0);
    int startX = (int)outX * stride - params.padding;
    int startY = (int)outY * stride - params.padding;

    int wStrideK = (int)inChannels;
    int wStrideY = (int)kernelSize * inChannels;
    int wStrideOut = (int)kernelSize * kernelSize * inChannels;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    for (int ky = 0; ky < kernelSize; ky++)
    {
        int inY = startY + ky;
        if (inY >= 0 && inY < params.inputImageHeight)
        {
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int inX = startX + kx;
                if (inX >= 0 && inX < params.inputImageWidth)
                {
                    int wBase = (int)outCh * wStrideOut + (int)ky * wStrideY + (int)kx * wStrideK;

                    for (int ic = laneId; ic < inChannels; ic += waveSize)
                    {
                        Coord inCoord = Coord(batchIdx, inY, inX, ic);
                        T.UnpackedType val = params.inputImage.eval(inCoord, emptyInput);
                        T.UnpackedType w = params.weightsIOKK[wBase + ic].unpack();
                        partialSum = partialSum + val * w;
                    }
                }
            }
        }
    }

    T.UnpackedType totalSum = partialSum.waveActiveSum();
    if (laneId == 0)
    {
        if (outCh < outChannels) totalSum = totalSum + params.bias[outCh].unpack();
        Coord outCoord = Coord(batchIdx, outY, outX, outCh);
        Input<T> outInput = {};
        outInput.value = totalSum;
        params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
    }
}

// --- TRANSPOSED CONVOLUTION ---

struct TransposedConvolutionParams<T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>
{
    TInput inputImage; // Layout: [batch, height, width, channels]
    FOutput outputTransform;
    TSink outputImage; // Layout: [batch, kernelSize, kernelSize, outChannels]
    T* weights;         // Weights in element type T
    T* bias;            // Bias in element type T
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int stride;
    int padding;
    int batchSize;
};

[numthreads(tileSize, tileSize, 1)]
void tiledTransposedConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels, T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<TransposedConvolutionParams<T, TInput, FOutput, TSink>> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    // --- 1. Batch & Channel Decoding ---
    uint numChannelGroups = (outChannels + outputChannelBatchSize - 1) / outputChannelBatchSize;
    uint batchIdx = groupId.z / numChannelGroups;
    uint channelGroupIdx = groupId.z % numChannelGroups;

    if (batchIdx >= params.batchSize) return;

    const int2 outPos = groupId.xy * int2(tileSize) + groupThreadId.xy;
    const bool validOutput = outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight;
    const int outChannelStart = channelGroupIdx * outputChannelBatchSize;

    // --- Input Tile Logic ---
    const int2 groupOriginOut = groupId.xy * int2(tileSize);
    const int2 tileOriginIn = (groupOriginOut + params.padding - (kernelSize - 1)) / params.stride;
    static const int inputTileDim = (tileSize + stride - 1) / stride + kernelSize;
    static groupshared T.UnpackedType s_inputTile[inputChannelBatchSize][inputTileDim * inputTileDim];

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // --- Accumulators (in T.UnpackedType precision) ---
    T.UnpackedType outputAccumulators[outputChannelBatchSize];
    for (int o = 0; o < outputChannelBatchSize; ++o)
    {
        int globalOutC = outChannelStart + o;
        if (globalOutC < outChannels)
            outputAccumulators[o] = params.bias[globalOutC].unpack();
        else
            outputAccumulators[o] = T.UnpackedType(0);
    }

    // --- Main Loop ---
    for (int inCBase = 0; inCBase < inChannels; inCBase += inputChannelBatchSize)
    {
        // Load Input
        const int numSharedPixels = inputTileDim * inputTileDim;
        const int numThreads = tileSize * tileSize;
        const int linearThreadIndex = groupThreadId.y * tileSize + groupThreadId.x;

        for (int i = linearThreadIndex; i < numSharedPixels; i += numThreads)
        {
            int sY = i / inputTileDim;
            int sX = i % inputTileDim;
            int globalInY = tileOriginIn.y + sY;
            int globalInX = tileOriginIn.x + sX;
            bool inBounds = globalInY >= 0 && globalInY < params.inputImageHeight &&
                            globalInX >= 0 && globalInX < params.inputImageWidth;

            for (int k = 0; k < inputChannelBatchSize; ++k)
            {
                int currentInC = inCBase + k;
                if (inBounds && currentInC < inChannels)
                {
                    Coord inCoord = Coord(batchIdx, globalInY, globalInX, currentInC);
                    s_inputTile[k][i] = params.inputImage.eval(inCoord, emptyInput);
                }
                else
                {
                    s_inputTile[k][i] = T.UnpackedType(0);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Math
        if (validOutput)
        {
            int startKy = (outPos.y + params.padding) % stride;
            int startKx = (outPos.x + params.padding) % stride;

            for (int ky = startKy; ky < kernelSize; ky += stride)
            {
                int numY = outPos.y + params.padding - ky;
                int globalInY = numY < 0 ? (numY - stride + 1) / stride : numY / stride;
                int sY = globalInY - tileOriginIn.y;
                if (sY < 0 || sY >= inputTileDim) continue;

                for (int kx = startKx; kx < kernelSize; kx += stride)
                {
                    int globalInX = (outPos.x + params.padding - kx) / stride;
                    int sX = globalInX - tileOriginIn.x;
                    if (sX < 0 || sX >= inputTileDim) continue;

                    int sharedIdx = sY * inputTileDim + sX;

                    for (int inOffset = 0; inOffset < inputChannelBatchSize; ++inOffset)
                    {
                        int globalInC = inCBase + inOffset;
                        if (globalInC >= inChannels) break;

                        T.UnpackedType pixelVal = s_inputTile[inOffset][sharedIdx];

                        [unroll]
                        for (int outOffset = 0; outOffset < outputChannelBatchSize; ++outOffset)
                        {
                            int globalOutC = outChannelStart + outOffset;
                            if (globalOutC >= outChannels) continue;

                            int64_t wIdx = (int64_t)globalInC  * (kernelSize * kernelSize * outChannels) +
                                           (int64_t)ky * (kernelSize * outChannels) + 
                                           (int64_t)kx * (outChannels) + 
                                           (int64_t)globalOutC;
                            outputAccumulators[outOffset] = outputAccumulators[outOffset] + pixelVal * params.weights[wIdx].unpack();
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // --- Write Output ---
    if (validOutput)
    {
        for (int o = 0; o < outputChannelBatchSize; ++o)
        {
            int globalOutC = outChannelStart + o;
            if (globalOutC < outChannels)
            {
                Coord outCoord = Coord(batchIdx, outPos.y, outPos.x, globalOutC);
                Input<T> outInput = {};
                outInput.value = outputAccumulators[o];
                params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
            }
        }
    }
}

// --- FLAT TRANSPOSED ---
[numthreads(256, 1, 1)]
void flatTransposedConvolution<int kernelSize, int stride, int inChannels, int outChannels, T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<TransposedConvolutionParams<T, TInput, FOutput, TSink>> params,
    uint3 id : SV_DispatchThreadID)
{
    uint globalIdx = id.x;
    uint valuesPerImage = params.outputImageWidth * params.outputImageHeight * outChannels;
    uint totalOutputs = valuesPerImage * params.batchSize;

    if (globalIdx >= totalOutputs) return;

    // Decode Batch
    uint batchIdx = globalIdx / valuesPerImage;
    uint localIdx = globalIdx % valuesPerImage;

    uint outCh = localIdx % outChannels;
    uint spatialIdx = localIdx / outChannels;
    uint outX = spatialIdx % params.outputImageWidth;
    uint outY = spatialIdx / params.outputImageWidth;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    T.UnpackedType sum = T.UnpackedType(0);
    if (outCh < outChannels) sum = params.bias[outCh].unpack();

    int startKy = (outY + params.padding) % stride;
    int startKx = (outX + params.padding) % stride;

    for (int ky = startKy; ky < kernelSize; ky += stride)
    {
        int numY = outY + params.padding - ky;
        int inY = numY < 0 ? (numY - stride + 1) / stride : numY / stride;

        if (inY >= 0 && inY < params.inputImageHeight)
        {
            for (int kx = startKx; kx < kernelSize; kx += stride)
            {
                int numX = outX + params.padding - kx;
                int inX = numX < 0 ? (numX - stride + 1) / stride : numX / stride;
                if (inX >= 0 && inX < params.inputImageWidth)
                {
                    int wBase = (ky * kernelSize * outChannels + kx * outChannels + outCh);
                    int wStride = kernelSize * kernelSize * outChannels;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        Coord inCoord = Coord(batchIdx, inY, inX, ic);
                        T.UnpackedType val = params.inputImage.eval(inCoord, emptyInput);
                        T.UnpackedType w = params.weights[ic * wStride + wBase].unpack();
                        sum = sum + val * w;
                    }
                }
            }
        }
    }
    Coord outCoord = Coord(batchIdx, outY, outX, outCh);
    Input<T> outInput = {};
    outInput.value = sum;
    params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
}

// ============================================================================
// GEMM-Style Optimized Convolution with Register Blocking
// ============================================================================
// Key optimizations:
// 1. Caches BOTH weights AND input in shared memory
// 2. Register blocking: each thread computes THREAD_OH x THREAD_OW spatial positions
// 3. Full weight reuse across spatial positions (loaded once, used for all positions)
// 4. Outer product formulation for maximum register reuse
//
// Generic parameters:
// - TILE_OH, TILE_OW: Output spatial tile dimensions (block computes this many outputs)
// - TILE_OC: Output channels per block
// - TILE_IC: Input channels per K-iteration
// - THREAD_OH, THREAD_OW: Spatial positions computed per thread (register blocking)
//
// Thread block: (TILE_OW/THREAD_OW, TILE_OH/THREAD_OH, 1)
// Each thread computes THREAD_OH x THREAD_OW x TILE_OC outputs
//
// Dispatch: x = (outputW + TILE_OW - 1) / TILE_OW
//           y = (outputH + TILE_OH - 1) / TILE_OH
//           z = batchSize * ((outChannels + TILE_OC - 1) / TILE_OC)

[numthreads(TILE_OW / THREAD_OW, TILE_OH / THREAD_OH, 1)]
void gemmConvolution<
    let TILE_OH : int,      // Output spatial tile height
    let TILE_OW : int,      // Output spatial tile width  
    let TILE_OC : int,      // Output channels per block
    let TILE_IC : int,      // Input channels per K-iteration
    let THREAD_OH : int,    // Spatial rows per thread
    let THREAD_OW : int,    // Spatial cols per thread
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    T : ITensorElement,
    TInput : IExpr<T>,
    FOutput : IExpr<T>,
    TSink : ISink<T>
>(
    ConstantBuffer<ConvolutionParams<T, TInput, FOutput, TSink>, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    // Thread block dimensions (computed from generic params)
    static const int BLOCK_X = TILE_OW / THREAD_OW;
    static const int BLOCK_Y = TILE_OH / THREAD_OH;
    static const int NUM_THREADS = BLOCK_X * BLOCK_Y;

    // Decode batch and output channel tile
    uint numOCTiles = (outChannels + TILE_OC - 1) / TILE_OC;
    uint batchIdx = groupId.z / numOCTiles;
    uint ocTileIdx = groupId.z % numOCTiles;

    if (batchIdx >= uint(params.batchSize)) return;

    // Output tile origin
    uint ohBase = groupId.y * TILE_OH;
    uint owBase = groupId.x * TILE_OW;
    uint ocBase = ocTileIdx * TILE_OC;

    // This thread's base output position
    uint ohThreadBase = ohBase + groupThreadId.y * THREAD_OH;
    uint owThreadBase = owBase + groupThreadId.x * THREAD_OW;

    // Shared memory for input tile
    // Size: (TILE_OH-1)*stride + kernelSize) x (TILE_OW-1)*stride + kernelSize) x TILE_IC
    static const int INPUT_TILE_H = (TILE_OH - 1) * stride + kernelSize;
    static const int INPUT_TILE_W = (TILE_OW - 1) * stride + kernelSize;
    static const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;
    static groupshared T.UnpackedType s_input[TILE_IC][INPUT_TILE_SIZE];

    // Shared memory for weight tile
    // Size: TILE_OC x (kernelSize x kernelSize x TILE_IC)
    static const int WEIGHT_K = kernelSize * kernelSize * TILE_IC;
    static groupshared T.UnpackedType s_weight[TILE_OC][WEIGHT_K];

    // Input tile origin in input image coordinates
    int ihBase = int(ohBase * stride) - params.padding;
    int iwBase = int(owBase * stride) - params.padding;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // Thread linear index for cooperative loading
    int threadLinear = int(groupThreadId.y) * BLOCK_X + int(groupThreadId.x);

    // Accumulators - each thread computes THREAD_OH x THREAD_OW x TILE_OC outputs
    T.UnpackedType accum[THREAD_OH][THREAD_OW][TILE_OC];

    // Initialize with bias
    [unroll]
    for (int th = 0; th < THREAD_OH; th++)
    {
        [unroll]
        for (int tw = 0; tw < THREAD_OW; tw++)
        {
            [unroll]
            for (int oc = 0; oc < TILE_OC; oc++)
            {
                int globalOC = int(ocBase) + oc;
                if (globalOC < outChannels)
                    accum[th][tw][oc] = params.bias[globalOC].unpack();
                else
                    accum[th][tw][oc] = T.UnpackedType(0);
            }
        }
    }

    // Main loop over input channels
    for (int icBase = 0; icBase < inChannels; icBase += TILE_IC)
    {
        // ================================================================
        // Cooperative load: Input tile
        // ================================================================
        for (int i = threadLinear; i < INPUT_TILE_SIZE; i += NUM_THREADS)
        {
            int sY = i / INPUT_TILE_W;
            int sX = i % INPUT_TILE_W;
            int ih = ihBase + sY;
            int iw = iwBase + sX;

            bool inBounds = ih >= 0 && ih < params.inputImageHeight &&
                            iw >= 0 && iw < params.inputImageWidth;

            for (int k = 0; k < TILE_IC; ++k)
            {
                int currentIC = icBase + k;
                if (inBounds && currentIC < inChannels)
                {
                    Coord inCoord = Coord(batchIdx, uint(ih), uint(iw), uint(currentIC));
                    s_input[k][i] = params.inputImage.eval(inCoord, emptyInput);
                }
                else
                {
                    s_input[k][i] = T.UnpackedType(0);
                }
            }
        }

        // ================================================================
        // Cooperative load: Weight tile
        // ================================================================
        // weightsIOKK layout: [OC, KH, KW, IC]
        int wStrideOC = kernelSize * kernelSize * inChannels;
        int wStrideKH = kernelSize * inChannels;
        int wStrideKW = inChannels;
        static const int NUM_WEIGHTS = TILE_OC * WEIGHT_K;
        for (int i = threadLinear; i < NUM_WEIGHTS; i += NUM_THREADS)
        {
            int oc_local = i / WEIGHT_K;
            int k = i % WEIGHT_K;

            // Decode k -> (kh, kw, ic_local)
            int ic_local = k % TILE_IC;
            int spatialK = k / TILE_IC;
            int kh = spatialK / kernelSize;
            int kw = spatialK % kernelSize;

            int globalOC = int(ocBase) + oc_local;
            int globalIC = icBase + ic_local;

            T.UnpackedType wval = T.UnpackedType(0);
            if (globalOC < outChannels && globalIC < inChannels)
            {
                int wIdx = globalOC * wStrideOC + kh * wStrideKH + kw * wStrideKW + globalIC;
                wval = params.weightsIOKK[wIdx].unpack();
            }
            s_weight[oc_local][k] = wval;
        }

        GroupMemoryBarrierWithGroupSync();

        // ================================================================
        // Compute: register-blocked iteration over kernel positions
        // ================================================================
        [loop]
        for (int kh = 0; kh < kernelSize; kh++)
        {
            [loop]
            for (int kw = 0; kw < kernelSize; kw++)
            {
                [loop]
                for (int ic_local = 0; ic_local < TILE_IC; ic_local++)
                {
                    int globalIC = icBase + ic_local;
                    if (globalIC >= inChannels) break;

                    // Load input values for all spatial positions this thread handles
                    // (loaded once, reused across all OC chunks)
                    T.UnpackedType regInput[THREAD_OH][THREAD_OW];
                    [unroll]
                    for (int th = 0; th < THREAD_OH; th++)
                    {
                        [unroll]
                        for (int tw = 0; tw < THREAD_OW; tw++)
                        {
                            int sY = int(groupThreadId.y * THREAD_OH + th) * stride + kh;
                            int sX = int(groupThreadId.x * THREAD_OW + tw) * stride + kw;
                            int sharedIdx = sY * INPUT_TILE_W + sX;
                            regInput[th][tw] = s_input[ic_local][sharedIdx];
                        }
                    }

                    int wK = (kh * kernelSize + kw) * TILE_IC + ic_local;
                    static const int OC_CHUNK = 4;
                    [loop]
                    for (int oc_chunk = 0; oc_chunk < TILE_OC; oc_chunk += OC_CHUNK)
                    {
                        // Load only 4 weights at a time
                        T.UnpackedType regW[OC_CHUNK];
                        [unroll]
                        for (int i = 0; i < OC_CHUNK; i++)
                        {
                            regW[i] = s_weight[oc_chunk + i][wK];
                        }

                        // Compute for these 4 OC
                        [unroll]
                        for (int th = 0; th < THREAD_OH; th++)
                        {
                            [unroll]
                            for (int tw = 0; tw < THREAD_OW; tw++)
                            {
                                T.UnpackedType inputVal = regInput[th][tw];
                                [unroll]
                                for (int i = 0; i < OC_CHUNK; i++)
                                {
                                    accum[th][tw][oc_chunk + i] =
                                        accum[th][tw][oc_chunk + i] + inputVal * regW[i];
                                }
                            }
                        }
                    }
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write output: each thread writes THREAD_OH x THREAD_OW x TILE_OC outputs
    [unroll]
    for (int th = 0; th < THREAD_OH; th++)
    {
        uint oh = ohThreadBase + th;
        if (oh >= uint(params.outputImageHeight)) continue;

        [unroll]
        for (int tw = 0; tw < THREAD_OW; tw++)
        {
            uint ow = owThreadBase + tw;
            if (ow >= uint(params.outputImageWidth)) continue;

            [unroll]
            for (int oc_local = 0; oc_local < TILE_OC; oc_local++)
            {
                int globalOC = int(ocBase) + oc_local;
                if (globalOC >= outChannels) continue;

                Coord outCoord = Coord(batchIdx, oh, ow, uint(globalOC));
                Input<T> outInput = {};
                outInput.value = accum[th][tw][oc_local];
                params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
            }
        }
    }
}

// ============================================================================
// GEMM-Style Convolution with Wave Shuffle (No Weight Shared Memory)
// ============================================================================
// Key optimization: Uses wave shuffles to broadcast weights instead of shared memory.
// - Threads within a warp computing different spatial positions need the same weights
// - Lane 0-15 each load one weight (for oc=0-15), then broadcast via WaveReadLaneAt
// - Saves ~4.6 KB shared memory per block, potentially improving occupancy
//
// Thread layout: 16x16 = 256 threads = 8 warps
// Each warp: 32 threads = 2 rows × 16 columns of spatial positions

[numthreads(TILE_OW / THREAD_OW, TILE_OH / THREAD_OH, 1)]
void gemmConvolutionWaveShuffle<
    let TILE_OH : int,
    let TILE_OW : int,
    let TILE_OC : int,
    let TILE_IC : int,
    let THREAD_OH : int,
    let THREAD_OW : int,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    T : ITensorElement,
    TInput : IExpr<T>,
    FOutput : IExpr<T>,
    TSink : ISink<T>
>(
    ConstantBuffer<ConvolutionParams<T, TInput, FOutput, TSink>, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    static const int BLOCK_X = TILE_OW / THREAD_OW;
    static const int BLOCK_Y = TILE_OH / THREAD_OH;
    static const int NUM_THREADS = BLOCK_X * BLOCK_Y;
    static const int WARP_SIZE = 32;

    uint numOCTiles = (outChannels + TILE_OC - 1) / TILE_OC;
    uint batchIdx = groupId.z / numOCTiles;
    uint ocTileIdx = groupId.z % numOCTiles;

    if (batchIdx >= uint(params.batchSize)) return;

    uint ohBase = groupId.y * TILE_OH;
    uint owBase = groupId.x * TILE_OW;
    uint ocBase = ocTileIdx * TILE_OC;

    uint ohThreadBase = ohBase + groupThreadId.y * THREAD_OH;
    uint owThreadBase = owBase + groupThreadId.x * THREAD_OW;

    // Shared memory for input tile ONLY (no weights!)
    static const int INPUT_TILE_H = (TILE_OH - 1) * stride + kernelSize;
    static const int INPUT_TILE_W = (TILE_OW - 1) * stride + kernelSize;
    static const int INPUT_TILE_SIZE = INPUT_TILE_H * INPUT_TILE_W;
    static groupshared T.UnpackedType s_input[TILE_IC][INPUT_TILE_SIZE];

    int ihBase = int(ohBase * stride) - params.padding;
    int iwBase = int(owBase * stride) - params.padding;

    // Accumulators
    T.UnpackedType accum[THREAD_OH][THREAD_OW][TILE_OC];

    // Initialize with bias
    [unroll]
    for (int th = 0; th < THREAD_OH; th++)
    {
        [unroll]
        for (int tw = 0; tw < THREAD_OW; tw++)
        {
            [unroll]
            for (int oc = 0; oc < TILE_OC; oc++)
            {
                int globalOC = int(ocBase) + oc;
                if (globalOC < outChannels)
                    accum[th][tw][oc] = params.bias[globalOC].unpack();
                else
                    accum[th][tw][oc] = T.UnpackedType(0);
            }
        }
    }

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    int threadLinear = int(groupThreadId.y) * BLOCK_X + int(groupThreadId.x);
    int laneId = threadLinear % WARP_SIZE;  // 0-31 within warp

    // Main loop over input channels
    for (int icBase = 0; icBase < inChannels; icBase += TILE_IC)
    {
        // Cooperative load: Input tile (same as before)
        for (int i = threadLinear; i < INPUT_TILE_SIZE; i += NUM_THREADS)
        {
            int sY = i / INPUT_TILE_W;
            int sX = i % INPUT_TILE_W;
            int ih = ihBase + sY;
            int iw = iwBase + sX;

            bool inBounds = ih >= 0 && ih < params.inputImageHeight &&
                            iw >= 0 && iw < params.inputImageWidth;

            for (int k = 0; k < TILE_IC; ++k)
            {
                int currentIC = icBase + k;
                if (inBounds && currentIC < inChannels)
                {
                    Coord inCoord = Coord(batchIdx, uint(ih), uint(iw), uint(currentIC));
                    s_input[k][i] = params.inputImage.eval(inCoord, emptyInput);
                }
                else
                {
                    s_input[k][i] = T.UnpackedType(0);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute using wave shuffle for weights
        for (int kh = 0; kh < kernelSize; kh++)
        {
            for (int kw = 0; kw < kernelSize; kw++)
            {
                for (int ic_local = 0; ic_local < TILE_IC; ic_local++)
                {
                    int globalIC = icBase + ic_local;
                    if (globalIC >= inChannels) break;

                    // Load input values
                    T.UnpackedType regInput[THREAD_OH][THREAD_OW];
                    [unroll]
                    for (int th = 0; th < THREAD_OH; th++)
                    {
                        [unroll]
                        for (int tw = 0; tw < THREAD_OW; tw++)
                        {
                            int sY = int(groupThreadId.y * THREAD_OH + th) * stride + kh;
                            int sX = int(groupThreadId.x * THREAD_OW + tw) * stride + kw;
                            int sharedIdx = sY * INPUT_TILE_W + sX;
                            regInput[th][tw] = s_input[ic_local][sharedIdx];
                        }
                    }

                    // Wave shuffle for weights: lanes 0-15 load weights, broadcast to all 32
                    // Each lane loads one weight if laneId < TILE_OC
                    T.UnpackedType myWeight = T.UnpackedType(0);
                    if (laneId < TILE_OC)
                    {
                        int myOC = int(ocBase) + laneId;
                        if (myOC < outChannels)
                        {
                            int64_t wIdx = int64_t(myOC) * (kernelSize * kernelSize * inChannels) +
                                           int64_t(kh) * (kernelSize * inChannels) +
                                           int64_t(kw) * inChannels +
                                           int64_t(globalIC);
                            myWeight = params.weightsIOKK[wIdx].unpack();
                        }
                    }

                    // Process OC in chunks, using waveReadLaneAt to broadcast weights
                    static const int OC_CHUNK = 4;
                    for (int oc_chunk = 0; oc_chunk < TILE_OC; oc_chunk += OC_CHUNK)
                    {
                        // Get weights via wave shuffle (lanes oc_chunk to oc_chunk+3)
                        T.UnpackedType regW[OC_CHUNK];
                        [unroll]
                        for (int i = 0; i < OC_CHUNK; i++)
                        {
                            regW[i] = myWeight.waveReadLaneAt(oc_chunk + i);
                        }

                        // Compute
                        [unroll]
                        for (int th = 0; th < THREAD_OH; th++)
                        {
                            [unroll]
                            for (int tw = 0; tw < THREAD_OW; tw++)
                            {
                                T.UnpackedType inputVal = regInput[th][tw];
                                [unroll]
                                for (int i = 0; i < OC_CHUNK; i++)
                                {
                                    accum[th][tw][oc_chunk + i] = accum[th][tw][oc_chunk + i] + inputVal * regW[i];
                                }
                            }
                        }
                    }
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Write output
    [unroll]
    for (int th = 0; th < THREAD_OH; th++)
    {
        uint oh = ohThreadBase + th;
        if (oh >= uint(params.outputImageHeight)) continue;

        [unroll]
        for (int tw = 0; tw < THREAD_OW; tw++)
        {
            uint ow = owThreadBase + tw;
            if (ow >= uint(params.outputImageWidth)) continue;

            [unroll]
            for (int oc_local = 0; oc_local < TILE_OC; oc_local++)
            {
                int globalOC = int(ocBase) + oc_local;
                if (globalOC >= outChannels) continue;

                Coord outCoord = Coord(batchIdx, oh, ow, uint(globalOC));
                Input<T> outInput = {};
                outInput.value = accum[th][tw][oc_local];
                params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
            }
        }
    }
}

// ============================================================================
// Winograd F(4x4, 3x3) Convolution
// ============================================================================
// Reduces 3x3 convolution from 9 multiplications to ~2.25 per output.
// 
// For each 4x4 output tile:
//   1. Load 6x6 input tile (4 output + 2 overlap for 3x3 kernel)
//   2. Apply input transform: V = BT * d * B (where d is input tile)
//   3. Element-wise multiply with pre-transformed weights: M = V ⊙ U
//   4. Apply output transform: Y = AT * M * A
//
// Generic parameters:
//   TILE_OC: Output channels per thread block
//   TILE_IC: Input channels per K-iteration
//
// Dispatch: 
//   x = (outputW + 3) / 4   (4x4 output tiles)
//   y = (outputH + 3) / 4
//   z = batchSize * ((outChannels + TILE_OC - 1) / TILE_OC)

// Winograd F(4x4, 3x3) transformation matrices
// BT: 6x6 input transform (BT * input * B)
// AT: 4x6 output transform (AT * output * A)

struct WinogradParams<T : ITensorElement, TInput : IExpr<T>, FOutput : IExpr<T>, TSink : ISink<T>>
{
    TInput inputImage;
    FOutput outputTransform;
    TSink outputImage;
    T* bias;
    // Winograd-transformed weights: [outChannels, inChannels, 6, 6]
    T* winogradWeights;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int padding;
    int batchSize;
};

// Helper: Apply Winograd input transform BT to a 6-element vector
// BT for F(4,3):
// [  1   0  -21/4   0   21/4    0  -1  0 ]  <- simplified to common form
// Actually using the simpler F(2,3) first, then can extend to F(4,3)
// For F(4,3), the standard BT matrix is:
//   [ 4  0 -5  0  1  0]
//   [ 0 -4 -4  1  1  0]
//   [ 0  4 -4 -1  1  0]
//   [ 0 -2 -1  2  1  0]
//   [ 0  2 -1 -2  1  0]
//   [ 0  4  0 -5  0  1]

// ============================================================================
// Winograd F(4x4, 3x3) Convolution
// ============================================================================
// Uses Winograd transform to reduce multiplications from 9 to 2.25 per output.
// Weight transform is done on CPU at load time.
//
// Thread block: 16x16 = 256 threads
// Thread mapping: 16 output channels × 16 output positions (4×4)
// Each thread computes one output value for one output channel.

[numthreads(16, 16, 1)]
void winogradConvolution<
    let TILE_OC : int,      // Output channels per block (e.g., 16)
    let TILE_IC : int,      // Input channels per K-iteration (e.g., 8)
    int inChannels,
    int outChannels,
    T : ITensorElement,
    TInput : IExpr<T>,
    FOutput : IExpr<T>,
    TSink : ISink<T>
>(
    ConstantBuffer<WinogradParams<T, TInput, FOutput, TSink>, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    static const int TILE_SIZE = 4;      // Output tile size (4x4)
    static const int INPUT_TILE = 6;     // Input tile size (6x6)
    static const int NUM_THREADS = 256;
    
    // Decode batch and output channel tile
    uint numOCTiles = (outChannels + TILE_OC - 1) / TILE_OC;
    uint batchIdx = groupId.z / numOCTiles;
    uint ocTileIdx = groupId.z % numOCTiles;
    
    if (batchIdx >= uint(params.batchSize)) return;
    
    // Output tile origin
    int tileY = int(groupId.y) * TILE_SIZE;
    int tileX = int(groupId.x) * TILE_SIZE;
    uint ocBase = ocTileIdx * TILE_OC;
    
    // Input tile origin (with padding)
    int inputTileY = tileY - params.padding;
    int inputTileX = tileX - params.padding;
    
    // Thread mapping: 256 = 16 OC × 16 positions
    int threadLinear = int(groupThreadId.y) * 16 + int(groupThreadId.x);
    int localOC = threadLinear / 16;      // 0-15 for output channels
    int localOutPos = threadLinear % 16;  // 0-15 for 4x4 positions
    int localY = localOutPos / 4;
    int localX = localOutPos % 4;
    int globalOC = int(ocBase) + localOC;
    
    if (localOC >= TILE_OC || globalOC >= outChannels) return;
    
    int outY = tileY + localY;
    int outX = tileX + localX;
    bool validOutput = outY < params.outputImageHeight && outX < params.outputImageWidth;
    
    // Shared memory for input tile (before transform)
    static groupshared T.UnpackedType s_inputTile[TILE_IC][INPUT_TILE][INPUT_TILE];
    // Shared memory for transformed input
    static groupshared T.UnpackedType s_transformedInput[TILE_IC][INPUT_TILE][INPUT_TILE];
    // Shared memory for weights [36][TILE_OC][TILE_IC]
    static groupshared T.UnpackedType s_weights[36][TILE_OC][TILE_IC];
    
    // Accumulator for M[6][6] for this thread's output channel
    T.UnpackedType M[36];
    for (int i = 0; i < 36; i++)
        M[i] = T.UnpackedType(0);
    
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    
    // Constants
    T.UnpackedType c2 = T.UnpackedType(2.0f);
    T.UnpackedType c4 = T.UnpackedType(4.0f);
    T.UnpackedType c5 = T.UnpackedType(5.0f);
    T.UnpackedType c8 = T.UnpackedType(8.0f);
    
    // Main loop over input channels
    for (int icBase = 0; icBase < inChannels; icBase += TILE_IC)
    {
        // ================================================================
        // Step 1: Cooperatively load 6×6 input tile
        // ================================================================
        for (int i = threadLinear; i < INPUT_TILE * INPUT_TILE * TILE_IC; i += NUM_THREADS)
        {
            int ic_local = i / 36;
            int pos = i % 36;
            int sy = pos / INPUT_TILE;
            int sx = pos % INPUT_TILE;
            
            int iy = inputTileY + sy;
            int ix = inputTileX + sx;
            int globalIC = icBase + ic_local;
            
            bool inBounds = iy >= 0 && iy < params.inputImageHeight &&
                            ix >= 0 && ix < params.inputImageWidth &&
                            globalIC < inChannels;
            
            if (inBounds)
            {
                Coord inCoord = Coord(batchIdx, uint(iy), uint(ix), uint(globalIC));
                s_inputTile[ic_local][sy][sx] = params.inputImage.eval(inCoord, emptyInput);
            }
            else
            {
                s_inputTile[ic_local][sy][sx] = T.UnpackedType(0);
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // ================================================================
        // Step 2: Apply BT transform to columns (BT * d)
        // ================================================================
        for (int i = threadLinear; i < TILE_IC * INPUT_TILE * INPUT_TILE; i += NUM_THREADS)
        {
            int ic_local = i / 36;
            int pos = i % 36;
            int row = pos / INPUT_TILE;
            int col = pos % INPUT_TILE;
            
            T.UnpackedType d0 = s_inputTile[ic_local][0][col];
            T.UnpackedType d1 = s_inputTile[ic_local][1][col];
            T.UnpackedType d2 = s_inputTile[ic_local][2][col];
            T.UnpackedType d3 = s_inputTile[ic_local][3][col];
            T.UnpackedType d4 = s_inputTile[ic_local][4][col];
            T.UnpackedType d5 = s_inputTile[ic_local][5][col];
            
            T.UnpackedType result;
            if (row == 0) result = d0 * c4 - d2 * c5 + d4;
            else if (row == 1) result = -d1 * c4 - d2 * c4 + d3 + d4;
            else if (row == 2) result = d1 * c4 - d2 * c4 - d3 + d4;
            else if (row == 3) result = -d1 * c2 - d2 + d3 * c2 + d4;
            else if (row == 4) result = d1 * c2 - d2 - d3 * c2 + d4;
            else result = d1 * c4 - d3 * c5 + d5;
            
            s_transformedInput[ic_local][row][col] = result;
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // ================================================================
        // Step 3: Apply B transform to rows (complete BT * d * B)
        // ================================================================
        for (int i = threadLinear; i < TILE_IC * INPUT_TILE * INPUT_TILE; i += NUM_THREADS)
        {
            int ic_local = i / 36;
            int pos = i % 36;
            int row = pos / INPUT_TILE;
            int col = pos % INPUT_TILE;
            
            T.UnpackedType r0 = s_transformedInput[ic_local][row][0];
            T.UnpackedType r1 = s_transformedInput[ic_local][row][1];
            T.UnpackedType r2 = s_transformedInput[ic_local][row][2];
            T.UnpackedType r3 = s_transformedInput[ic_local][row][3];
            T.UnpackedType r4 = s_transformedInput[ic_local][row][4];
            T.UnpackedType r5 = s_transformedInput[ic_local][row][5];
            
            T.UnpackedType result;
            if (col == 0) result = r0 * c4 - r2 * c5 + r4;
            else if (col == 1) result = -r1 * c4 - r2 * c4 + r3 + r4;
            else if (col == 2) result = r1 * c4 - r2 * c4 - r3 + r4;
            else if (col == 3) result = -r1 * c2 - r2 + r3 * c2 + r4;
            else if (col == 4) result = r1 * c2 - r2 - r3 * c2 + r4;
            else result = r1 * c4 - r3 * c5 + r5;
            
            s_inputTile[ic_local][row][col] = result;  // Reuse input tile
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // ================================================================
        // Step 4: Load weights into shared memory
        // ================================================================
        int weightsToLoad = 36 * TILE_OC * TILE_IC;
        for (int i = threadLinear; i < weightsToLoad; i += NUM_THREADS)
        {
            int pos = i / (TILE_OC * TILE_IC);
            int rem = i % (TILE_OC * TILE_IC);
            int oc_local = rem / TILE_IC;
            int ic_local = rem % TILE_IC;
            
            int globalOCLoad = int(ocBase) + oc_local;
            int globalICLoad = icBase + ic_local;
            
            T.UnpackedType w = T.UnpackedType(0);
            if (globalOCLoad < outChannels && globalICLoad < inChannels)
            {
                int64_t wIdx = int64_t(globalOCLoad) * (inChannels * 36) +
                               int64_t(globalICLoad) * 36 + int64_t(pos);
                w = params.winogradWeights[wIdx].unpack();
            }
            s_weights[pos][oc_local][ic_local] = w;
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // ================================================================
        // Step 5: Element-wise multiply and accumulate
        // ================================================================
        for (int pos = 0; pos < 36; pos++)
        {
            int py = pos / 6;
            int px = pos % 6;
            for (int ic_local = 0; ic_local < TILE_IC; ic_local++)
            {
                int globalIC = icBase + ic_local;
                if (globalIC >= inChannels) break;
                
                T.UnpackedType v = s_inputTile[ic_local][py][px];
                T.UnpackedType w = s_weights[pos][localOC][ic_local];
                M[pos] = M[pos] + v * w;
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
    }
    
    // ================================================================
    // Step 6: Apply output transform AT * M * A
    // ================================================================
    T.UnpackedType temp[4][6];
    for (int col = 0; col < 6; col++)
    {
        T.UnpackedType m0 = M[0 * 6 + col];
        T.UnpackedType m1 = M[1 * 6 + col];
        T.UnpackedType m2 = M[2 * 6 + col];
        T.UnpackedType m3 = M[3 * 6 + col];
        T.UnpackedType m4 = M[4 * 6 + col];
        T.UnpackedType m5 = M[5 * 6 + col];
        
        temp[0][col] = m0 + m1 + m2 + m3 + m4;
        temp[1][col] = m1 - m2 + m3 * c2 - m4 * c2;
        temp[2][col] = m1 + m2 + m3 * c4 + m4 * c4;
        temp[3][col] = m1 - m2 + m3 * c8 - m4 * c8 + m5;
    }
    
    T.UnpackedType output[4][4];
    for (int row = 0; row < 4; row++)
    {
        T.UnpackedType t0 = temp[row][0];
        T.UnpackedType t1 = temp[row][1];
        T.UnpackedType t2 = temp[row][2];
        T.UnpackedType t3 = temp[row][3];
        T.UnpackedType t4 = temp[row][4];
        T.UnpackedType t5 = temp[row][5];
        
        output[row][0] = t0 + t1 + t2 + t3 + t4;
        output[row][1] = t1 - t2 + t3 * c2 - t4 * c2;
        output[row][2] = t1 + t2 + t3 * c4 + t4 * c4;
        output[row][3] = t1 - t2 + t3 * c8 - t4 * c8 + t5;
    }
    
    // Add bias and write output
    T.UnpackedType result = output[localY][localX] + params.bias[globalOC].unpack();
    
    if (validOutput)
    {
        Coord outCoord = Coord(batchIdx, uint(outY), uint(outX), uint(globalOC));
        Input<T> outInput = {};
        outInput.value = result;
        params.outputImage.store(outCoord, params.outputTransform.eval(outCoord, outInput));
    }
}
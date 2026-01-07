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

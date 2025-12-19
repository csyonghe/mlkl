implementing mlkl;

static const int inputChannelBatchSize = 8;
static const int outputChannelBatchSize = 32;

// --- Shader Parameters ---
struct ConvolutionParams<TInput : IExpr, FOutput : IExpr, TSink : ISink>
{
    TInput inputImage; // Layout: [batch, height, width, channels]
    FOutput outputTransform; // Layout: [batch, height, width, channels]
    TSink outputImage;
    // Layout: [inChannels, kernelSize, kernelSize, outChannels]
    float* weights;
    float* bias;
    float* weightsIOKK; // Transposed weights, layout: [inChannels, outChannels, kernelSize, kernelSize]          
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int padding;
    int batchSize;
};

// Helper to calculate linear offsets for batching
int getBatchOffset(int batchIdx, int width, int height, int channels)
{
    return batchIdx * width * height * channels;
}

// [numthreads(tileSize, tileSize, 1)]
// Dispatch Z = (OutChannels/32) * BatchSize
[numthreads(tileSize, tileSize, 1)]
void tiledConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels, TInput : IExpr, FOutput : IExpr, TSink : ISink>(
    ConstantBuffer<ConvolutionParams<TInput, FOutput, TSink>, CDataLayout> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    // --- 1. Batch & Channel Decoding ---
    // We pack BatchID and ChannelGroupID into SV_GroupID.z
    uint numChannelGroups = (outChannels + outputChannelBatchSize - 1) / outputChannelBatchSize;
    uint batchIdx = groupId.z / numChannelGroups;
    uint channelGroupIdx = groupId.z % numChannelGroups;

    if (batchIdx >= params.batchSize) return;

    // --- 2. Coordinates & Setup ---
    const int2 outPos = groupId.xy * int2(tileSize) + groupThreadId.xy;
    const int2 tileOrigin = groupId.xy * int2(tileSize) * stride - int2(params.padding);
    const int outChannelStart = channelGroupIdx * outputChannelBatchSize;

    // Calculate Batch Offsets
    int inBatchOffset = getBatchOffset(batchIdx, params.inputImageWidth, params.inputImageHeight, inChannels);
    int outBatchOffset = getBatchOffset(batchIdx, params.outputImageWidth, params.outputImageHeight, outChannels);

    // --- 3. Shared Memory Allocation ---
    static const int sharedDim = (tileSize - 1) * stride + kernelSize;
    static groupshared float s_inputTile[inputChannelBatchSize][sharedDim * sharedDim];

    // --- 4. Output Accumulators ---
    float outputAccumulators[outputChannelBatchSize];
    for (int o = 0; o < outputChannelBatchSize; ++o)
    {
        int globalOutC = outChannelStart + o;
        if (globalOutC < outChannels)
            outputAccumulators[o] = params.bias[globalOutC];
        else
            outputAccumulators[o] = 0.0f;
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
                    int idx = (globalInY * params.inputImageWidth + globalInX) * inChannels + currentInC;
                    s_inputTile[k][i] = params.inputImage[inBatchOffset + idx]; // Apply Batch Offset
                }
                else
                {
                    s_inputTile[k][i] = 0.0f;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // B. Convolution Math (Unchanged logic, just using loaded tile)
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

                        float pixelVal = s_inputTile[inOffset][sharedIdx];

                        [unroll]
                        for (int outOffset = 0; outOffset < outputChannelBatchSize; ++outOffset)
                        {
                            int globalOutC = outChannelStart + outOffset;
                            if (globalOutC >= outChannels) continue;

                            int64_t wIdx = (int64_t)globalInC  * (kernelSize * kernelSize * outChannels) + 
                                           (int64_t)ky * (kernelSize * outChannels) + 
                                           (int64_t)kx * (outChannels) + 
                                           (int64_t)globalOutC;
                            outputAccumulators[outOffset] += pixelVal * params.weights[wIdx];
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
                int outIdx = (outPos.y * params.outputImageWidth + outPos.x) * outChannels + globalOutC;
                int logicalOutIdx = outBatchOffset + outIdx;
                params.outputImage.store(logicalOutIdx, params.outputTransform.eval(logicalOutIdx, Input(outputAccumulators[o])));
            }
        }
    }
}

// --- FLAT CONVOLUTION ---
// Handles Batching by extending the global linear index
[numthreads(256, 1, 1)]
void flatConvolution<int kernelSize, int stride, int inChannels, int outChannels, TInput : IExpr, FOutput : IExpr, TSink : ISink>(
    ConstantBuffer<ConvolutionParams<TInput, FOutput, TSink>> params,
    uint3 id : SV_DispatchThreadID)
{
    uint globalIdx = id.x;
    // Total items = Batch * H * W * C
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

    // Calculate Offsets
    int inBatchOffset = getBatchOffset(batchIdx, params.inputImageWidth, params.inputImageHeight, inChannels);
    // Note: output writing uses globalIdx directly, so explicit offset not strictly needed if mapped 1:1, 
    // but good for clarity if buffers separated. Here globalIdx is absolute index into params.outputImage.

    // 3. Setup Accumulator
    float sum = 0.0;
    if (outCh < outChannels)
        sum = params.bias[outCh];

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
                    int inPixelOffset = (inY * params.inputImageWidth + inX) * inChannels;
                    int wBase = (ky * kernelSize * outChannels + kx * outChannels + outCh);
                    int wStride = kernelSize * kernelSize * outChannels;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        // Add Batch Offset here
                        float val = params.inputImage[inBatchOffset + inPixelOffset + ic];
                        float w = params.weights[ic * wStride + wBase];
                        sum += val * w;
                    }
                }
            }
        }
    }

    params.outputImage.store(globalIdx, params.outputTransform.eval(globalIdx, Input(sum)));
}

// --- TINY CONVOLUTION ---
// 1 Thread = 1 Output Channel (Calculates ALL pixels for that channel in the image)
// Batching Strategy: Grid X = OutChannels, Grid Y = BatchSize
[numthreads(256, 1, 1)]
void tinyConvolution<int kernelSize, int stride, int inChannels, int outChannels, int maxTotalPixels, TInput : IExpr, FOutput : IExpr, TSink : ISink>(
    ConstantBuffer<ConvolutionParams<TInput, FOutput, TSink>> params,
    uint3 id : SV_DispatchThreadID)
{
    uint outCh = id.x;
    uint batchIdx = id.y; // Dispatch Y handles Batch

    if (outCh >= outChannels || batchIdx >= params.batchSize) return;

    int totalPixels = params.outputImageWidth * params.outputImageHeight;
    float sums[maxTotalPixels];
    float b = params.bias[outCh];
    
    [unroll]
    for (int i = 0; i < totalPixels; ++i) sums[i] = b;

    int inBatchOffset = getBatchOffset(batchIdx, params.inputImageWidth, params.inputImageHeight, inChannels);
    int outBatchOffset = getBatchOffset(batchIdx, params.outputImageWidth, params.outputImageHeight, outChannels);

    for (int ic = 0; ic < inChannels; ++ic)
    {
        int wBaseIC = ic * kernelSize * kernelSize * outChannels + outCh;
        for (int ky = 0; ky < kernelSize; ++ky)
        {
            for (int kx = 0; kx < kernelSize; ++kx)
            {
                int wOffset = ky * kernelSize * outChannels + kx * outChannels;
                float w = params.weights[wBaseIC + wOffset];

                [unroll]
                for (int p = 0; p < totalPixels; ++p)
                {
                    int2 outCoord = int2(p % params.outputImageWidth, p / params.outputImageWidth);
                    int inY = outCoord.y * stride - params.padding + ky;
                    int inX = outCoord.x * stride - params.padding + kx;

                    if (inX >= 0 && inX < params.inputImageWidth && 
                        inY >= 0 && inY < params.inputImageHeight)
                    {
                        int inIdx = (inY * params.inputImageWidth + inX) * inChannels + ic;
                        sums[p] += params.inputImage[inBatchOffset + inIdx] * w;
                    }
                }
            }
        }
    }

    [unroll]
    for (int p = 0; p < totalPixels; ++p)
    {
        // Output Layout: [Batch, H, W, C]
        int outIdx = outBatchOffset + p * outChannels + outCh;
        params.outputImage.store(outIdx, params.outputTransform.eval(outIdx, Input(sums[p])));
    }
}

// --- WAVE CONVOLUTION ---
// 1 Warp = 1 Pixel
[numthreads(256, 1, 1)]
void flatConvolutionWaveReduce<int kernelSize, int stride, int inChannels, int outChannels, TInput : IExpr, FOutput : IExpr, TSink : ISink>(
    ConstantBuffer<ConvolutionParams<TInput, FOutput, TSink>> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    uint waveSize = WaveGetLaneCount();
    uint warpsPerBlock = 256 / waveSize;
    uint laneId = WaveGetLaneIndex();
    uint warpId = groupThreadId.x / waveSize;

    // Linear index covers Batch * Pixels * Channels
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

    float partialSum = 0.0;
    int startX = (int)outX * stride - params.padding;
    int startY = (int)outY * stride - params.padding;

    int wStrideK = (int)inChannels;
    int wStrideY = (int)kernelSize * inChannels;
    int wStrideOut = (int)kernelSize * kernelSize * inChannels;
    
    // Batch Offset for Input
    int inBatchOffset = getBatchOffset(batchIdx, params.inputImageWidth, params.inputImageHeight, inChannels);

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
                    int inPixelOffset = ((int)inY * params.inputImageWidth + inX) * inChannels;
                    int wBase = (int)outCh * wStrideOut + (int)ky * wStrideY + (int)kx * wStrideK;

                    for (int ic = laneId; ic < inChannels; ic += waveSize)
                    {
                        float val = params.inputImage[inBatchOffset + inPixelOffset + ic];
                        float w = params.weights[wBase + ic];
                        partialSum += val * w;
                    }
                }
            }
        }
    }

    float totalSum = WaveActiveSum(partialSum);
    if (laneId == 0)
    {
        if (outCh < outChannels) totalSum += params.bias[outCh];
        params.outputImage.store(globalValueIdx, params.outputTransform.eval(globalValueIdx, Input(totalSum)));
    }
}

// --- TRANSPOSED CONVOLUTION ---

struct TransposedConvolutionParams<TInput : IExpr, FOutput : IExpr, TSink : ISink>
{
    TInput inputImage; // Layout: [batch, height, width, channels]
    FOutput outputTransform;
    TSink outputImage; // Layout: [batch, kernelSize, kernelSize, outChannels]
    float* weights; // Layout: [inChannels, kernelSize, kernelSize, outChannels]
    float* bias;            
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int stride;
    int padding;
    int batchSize;
};

[numthreads(tileSize, tileSize, 1)]
void tiledTransposedConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels, TInput : IExpr, FOutput : IExpr, TSink : ISink>(
    ConstantBuffer<TransposedConvolutionParams<TInput, FOutput, TSink>> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    // --- 1. Batch & Channel Decoding ---
    uint numChannelGroups = (outChannels + outputChannelBatchSize - 1) / outputChannelBatchSize;
    uint batchIdx = groupId.z / numChannelGroups;
    uint channelGroupIdx = groupId.z % numChannelGroups;

    if (batchIdx >= params.batchSize) return;

    // Calculate Offsets
    int inBatchOffset = getBatchOffset(batchIdx, params.inputImageWidth, params.inputImageHeight, inChannels);
    int outBatchOffset = getBatchOffset(batchIdx, params.outputImageWidth, params.outputImageHeight, outChannels);

    const int2 outPos = groupId.xy * int2(tileSize) + groupThreadId.xy;
    const bool validOutput = outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight;
    const int outChannelStart = channelGroupIdx * outputChannelBatchSize;

    // --- Input Tile Logic ---
    const int2 groupOriginOut = groupId.xy * int2(tileSize);
    const int2 tileOriginIn = (groupOriginOut + params.padding - (kernelSize - 1)) / params.stride;
    static const int inputTileDim = (tileSize + stride - 1) / stride + kernelSize;
    static groupshared float s_inputTile[inputChannelBatchSize][inputTileDim * inputTileDim];

    // --- Accumulators ---
    float outputAccumulators[outputChannelBatchSize];
    for (int o = 0; o < outputChannelBatchSize; ++o)
    {
        int globalOutC = outChannelStart + o;
        if (globalOutC < outChannels)
            outputAccumulators[o] = params.bias[globalOutC];
        else
            outputAccumulators[o] = 0.0f;
    }

    // --- Main Loop ---
    for (int inCBase = 0; inCBase < inChannels; inCBase += inputChannelBatchSize)
    {
        // Load Input (Apply Batch Offset)
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
                    int idx = (globalInY * params.inputImageWidth + globalInX) * inChannels + currentInC;
                    s_inputTile[k][i] = params.inputImage[inBatchOffset + idx];
                }
                else
                {
                    s_inputTile[k][i] = 0.0f;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Math (Unchanged)
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

                        float pixelVal = s_inputTile[inOffset][sharedIdx];

                        [unroll]
                        for (int outOffset = 0; outOffset < outputChannelBatchSize; ++outOffset)
                        {
                            int globalOutC = outChannelStart + outOffset;
                            if (globalOutC >= outChannels) continue;

                            int64_t wIdx = (int64_t)globalInC  * (kernelSize * kernelSize * outChannels) +
                                           (int64_t)ky * (kernelSize * outChannels) + 
                                           (int64_t)kx * (outChannels) + 
                                           (int64_t)globalOutC;
                            outputAccumulators[outOffset] += pixelVal * params.weights[wIdx];
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
                int outIdx = outBatchOffset + (outPos.y * params.outputImageWidth + outPos.x) * outChannels + globalOutC;
                params.outputImage.store(outIdx, params.outputTransform.eval(outIdx, Input(outputAccumulators[o])));
            }
        }
    }
}

// --- FLAT TRANSPOSED ---
[numthreads(256, 1, 1)]
void flatTransposedConvolution<int kernelSize, int stride, int inChannels, int outChannels, TInput : IExpr, FOutput : IExpr, TSink : ISink>(
    ConstantBuffer<TransposedConvolutionParams<TInput, FOutput, TSink>> params,
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

    // Calculate Offsets
    int inBatchOffset = getBatchOffset(batchIdx, params.inputImageWidth, params.inputImageHeight, inChannels);

    float sum = 0.0;
    if (outCh < outChannels) sum = params.bias[outCh];

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
                    int inIdx = (inY * params.inputImageWidth + inX) * inChannels;
                    int wBase = (ky * kernelSize * outChannels + kx * outChannels + outCh);
                    int wStride = kernelSize * kernelSize * outChannels;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        float val = params.inputImage[inBatchOffset + inIdx + ic];
                        float w = params.weights[ic * wStride + wBase];
                        sum += val * w;
                    }
                }
            }
        }
    }
    params.outputImage.store(globalIdx, params.outputTransform.eval(globalIdx, Input(sum)));
}
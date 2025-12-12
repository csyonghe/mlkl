implementing diffusion;

// --- Constants for Tiling and Batching ---
// These define the size of the work chunks. 
// 32 is chosen to align with typical GPU warp/wavefront sizes.
static const int inputChannelBatchSize = 8;
static const int outputChannelBatchSize = 32;

// --- Shader Parameters ---
struct SimpleConvolutionParams
{
    // CRITICAL CHANGE: Weight layout must be permuted on the host.
    // New Layout: [inChannels, kernelSize, kernelSize, outChannels]
    // This ensures stride-1 access in the innermost loop.
    float* weights;         
    
    float* bias;            // Layout: [outChannels]
    float* inputImage;      // Layout: [Height, Width, inChannels] (Standard NHWC)
    float* outputImage;     // Layout: [Height, Width, outChannels] (Standard NHWC)
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int padding;
};

// The shader entry point using generics for compile-time constants needed for array sizing.
[numthreads(tileSize, tileSize, 1)]
void simpleConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels, TActivation : IActivation>(
    ConstantBuffer<SimpleConvolutionParams> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    // --- 1. Coordinates & Setup ---
    
    // Global output pixel position for this thread
    const int2 outPos = groupId.xy * int2(tileSize) + groupThreadId.xy;
    
    // Top-left corner of the input area required for this thread group (the anchor)
    const int2 tileOrigin = groupId.xy * int2(tileSize) * stride - int2(params.padding);
    
    // Determine which batch of output channels this thread group is responsible for
    const int outChannelStart = groupId.z * outputChannelBatchSize;

    // --- 2. Shared Memory Allocation ---
    
    // The shared memory tile must hold the data for the threads plus the required "halo" (apron).
    static const int sharedDim = (tileSize - 1) * stride + kernelSize;
    
    // Layout: [ChannelBatch][Row][Col] used to minimize bank conflicts during loading/reading.
    static groupshared float s_inputTile[inputChannelBatchSize][sharedDim * sharedDim];

    // --- 3. Output Accumulators ---
    
    // Registers to hold partial sums for the batch of output channels being computed.
    float outputAccumulators[outputChannelBatchSize];
    
    // Initialize accumulators with bias values (if bias exists and index is valid)
    for (int o = 0; o < outputChannelBatchSize; ++o)
    {
        int globalOutC = outChannelStart + o;
        if (globalOutC < outChannels)
            outputAccumulators[o] = params.bias[globalOutC];
        else
            outputAccumulators[o] = 0.0f;
    }

    // --- 4. Main Convolution Loop (Input Channel Slicing) ---
    
    // We iterate over total input channels in chunks of 'inputChannelBatchSize'
    // to keep shared memory usage constant regardless of input depth.
    for (int inCBase = 0; inCBase < inChannels; inCBase += inputChannelBatchSize)
    {
        // ==============================================================================
        // A. Cooperative Load: Load Input Tile into Shared Memory
        // ==============================================================================
        // The shared tile region is larger than the number of threads in the block.
        // Threads loop to cooperatively load the entire required region.
        
        const int numSharedPixels = sharedDim * sharedDim;
        const int numThreads = tileSize * tileSize;
        const int linearThreadIndex = groupThreadId.y * tileSize + groupThreadId.x;

        for (int i = linearThreadIndex; i < numSharedPixels; i += numThreads)
        {
            // Map linear index to 2D shared coordinates
            int sY = i / sharedDim;
            int sX = i % sharedDim;

            // Calculate corresponding global input coordinates based on tile anchor
            int globalInY = tileOrigin.y + sY;
            int globalInX = tileOrigin.x + sX;

            // Check boundaries for zero-padding
            bool inBounds = globalInY >= 0 && globalInY < params.inputImageHeight &&
                            globalInX >= 0 && globalInX < params.inputImageWidth;

            // Load the batch of input channels for this specific pixel location
            for (int k = 0; k < inputChannelBatchSize; ++k)
            {
                int currentInC = inCBase + k;
                
                if (inBounds && currentInC < inChannels)
                {
                    // Input Layout: [H, W, C] (Channels Last)
                    // Contiguous read for the inner loop 'k'
                    int64_t idx = ((int64_t)globalInY * params.inputImageWidth + globalInX) * inChannels + currentInC;
                    s_inputTile[k][i] = params.inputImage[idx];
                }
                else
                {
                    // Handle padding or out-of-bounds channels
                    s_inputTile[k][i] = 0.0f;
                }
            }
        }

        // Ensure all threads have finished loading shared memory before proceeding
        GroupMemoryBarrierWithGroupSync();

        // ==============================================================================
        // B. Convolution Math
        // ==============================================================================
        // Only proceed if this thread maps to a valid output pixel
        if (outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight)
        {
            // Iterate over Kernel Spatial Dimensions
            for (int ky = 0; ky < kernelSize; ++ky)
            {
                for (int kx = 0; kx < kernelSize; ++kx)
                {
                    // Calculate the index in shared memory corresponding to the input pixel
                    // needed for this kernel weight location.
                    // Local pixel index = (threadY * stride + ky) * sharedDim + (threadX * stride + kx)
                    int sY = groupThreadId.y * stride + ky;
                    int sX = groupThreadId.x * stride + kx;
                    int sharedIdx = sY * sharedDim + sX;

                    // Iterate over the current batch of Input Channels sitting in LDS
                    for (int inOffset = 0; inOffset < inputChannelBatchSize; ++inOffset)
                    {
                        int globalInC = inCBase + inOffset;
                        if (globalInC >= inChannels) break;

                        // Read input value once from shared memory
                        float pixelVal = s_inputTile[inOffset][sharedIdx];

                        // Compute a batch of Output Channels in parallel registers.
                        // Unrolling is crucial here for instruction-level parallelism.
                        [unroll]
                        for (int outOffset = 0; outOffset < outputChannelBatchSize; ++outOffset)
                        {
                            int globalOutC = outChannelStart + outOffset;
                            if (globalOutC >= outChannels) continue;

                            // ---------------------------------------------------------
                            // OPTIMIZED WEIGHT INDEXING
                            // Layout Requirement: [InC][KH][KW][OutC]
                            // 
                            // Because 'globalOutC' is the varying variable in this innermost loop,
                            // having it as the last dimension ensures that we read contiguous
                            // memory addresses in a single coalesced transaction.
                            // ---------------------------------------------------------
                            int64_t wIdx = 
                                (int64_t)globalInC  * (kernelSize * kernelSize * outChannels) + 
                                (int64_t)ky         * (kernelSize * outChannels) + 
                                (int64_t)kx         * (outChannels) + 
                                (int64_t)globalOutC;

                            // Fused Multiply-Add
                            outputAccumulators[outOffset] += pixelVal * params.weights[wIdx];
                        }
                    }
                }
            }
        }
        
        // Sync before overwriting shared memory in the next input-channel-batch iteration
        GroupMemoryBarrierWithGroupSync();
    }

    // --- 5. Write Final Output ---
    
    if (outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight)
    {
        for (int o = 0; o < outputChannelBatchSize; ++o)
        {
            int globalOutC = outChannelStart + o;
            if (globalOutC < outChannels)
            {
                // Output Layout: [H, W, OutC] (Channels Last)
                int64_t outIdx = ((int64_t)outPos.y * params.outputImageWidth + outPos.x) * outChannels + globalOutC;
                params.outputImage[outIdx] = TActivation.apply(outputAccumulators[o]);
            }
        }
    }
}

struct SimpleTransposedConvolutionParams
{
    // CRITICAL: Weights must be permuted on Host (CPU) to this layout!
    // Standard PyTorch: [inChannels, outChannels, kernelSize, kernelSize]
    // Optimized Layout: [inChannels, kernelSize, kernelSize, outChannels]
    float* weights;         
    
    float* bias;            // Layout: [outChannels]
    float* inputImage;      // Layout: [Height, Width, inChannels] (NHWC)
    float* outputImage;     // Layout: [Height, Width, outChannels] (NHWC)
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int stride;
    int padding;
};

// Generics used for static array sizing
[numthreads(tileSize, tileSize, 1)]
void simpleTransposedConvolution<int tileSize, int kernelSize, int stride, int inChannels, int outChannels>(
    ConstantBuffer<SimpleTransposedConvolutionParams> params,
    int3 groupThreadId : SV_GroupThreadID,
    int3 groupId : SV_GroupID)
{
    // --- 1. Coordinate Setup ---

    // Global Output Position (The pixel this thread is responsible for)
    const int2 outPos = groupId.xy * int2(tileSize) + groupThreadId.xy;
    const bool validOutput = outPos.x < params.outputImageWidth && outPos.y < params.outputImageHeight;

    // Determine the Output Channel Batch (Z-dimension dispatch)
    const int outChannelStart = groupId.z * outputChannelBatchSize;

    // --- 2. Input Tile Calculation ---
    
    // We need to determine the range of Input pixels required to cover this Output tile.
    // Logic: input_idx = (output_idx + padding - kernel_idx) / stride
    
    // Calculate the top-left input coordinate (anchor) for this tile
    const int2 groupOriginOut = groupId.xy * int2(tileSize);
    // Integer division floor logic to find the top-most, left-most input pixel needed
    const int2 tileOriginIn = (groupOriginOut + params.padding - (kernelSize - 1)) / params.stride;

    // Calculate Shared Memory Size
    // Dimension must cover the span from Min to Max input requirements.
    static const int inputTileDim = (tileSize + stride - 1) / stride + kernelSize; 
    
    // Shared Memory Layout: [Batch][Row][Col]
    static groupshared float s_inputTile[inputChannelBatchSize][inputTileDim * inputTileDim];

    // --- 3. Accumulators ---

    float outputAccumulators[outputChannelBatchSize];
    
    // Initialize accumulators (Bias)
    for (int o = 0; o < outputChannelBatchSize; ++o)
    {
        int globalOutC = outChannelStart + o;
        if (globalOutC < outChannels)
            outputAccumulators[o] = params.bias[globalOutC];
        else
            outputAccumulators[o] = 0.0f;
    }

    // --- 4. Main Loop: Input Channel Slicing ---

    // Iterate over Input Channels in chunks (batching) to keep shared memory small
    for (int inCBase = 0; inCBase < inChannels; inCBase += inputChannelBatchSize)
    {
        // ==============================================================================
        // A. Load Input Tile into Shared Memory
        // ==============================================================================
        
        const int numSharedPixels = inputTileDim * inputTileDim;
        const int numThreads = tileSize * tileSize;
        const int linearThreadIndex = groupThreadId.y * tileSize + groupThreadId.x;

        // Cooperative loading loop
        for (int i = linearThreadIndex; i < numSharedPixels; i += numThreads)
        {
            int sY = i / inputTileDim;
            int sX = i % inputTileDim;
            int globalInY = tileOriginIn.y + sY;
            int globalInX = tileOriginIn.x + sX;

            bool inBounds = globalInY >= 0 && globalInY < params.inputImageHeight &&
                            globalInX >= 0 && globalInX < params.inputImageWidth;

            // Load batch of channels
            for (int k = 0; k < inputChannelBatchSize; ++k)
            {
                int currentInC = inCBase + k;
                if (inBounds && currentInC < inChannels)
                {
                    // Input Layout: [H, W, C]
                    int64_t idx = ((int64_t)globalInY * params.inputImageWidth + globalInX) * inChannels + currentInC;
                    s_inputTile[k][i] = params.inputImage[idx];
                }
                else
                {
                    s_inputTile[k][i] = 0.0f;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // ==============================================================================
        // B. Transposed Convolution Math
        // ==============================================================================
        
        if (validOutput)
        {
            // Optimization: "Strided Jump"
            // We only iterate kernel positions (ky, kx) that theoretically align with this output pixel.
            // Constraint: (outPos + padding - k) % stride == 0
            
            int startKy = (outPos.y + params.padding) % stride;
            int startKx = (outPos.x + params.padding) % stride;

            for (int ky = startKy; ky < kernelSize; ky += stride)
            {
                // Calculate which input pixel corresponds to this kernel offset
                // Formula: input = (output + pad - kernel) / stride
                int numY = outPos.y + params.padding - ky;
                int globalInY = numY < 0 ? (numY - stride + 1) / stride : numY / stride;
                int sY = globalInY - tileOriginIn.y;
                
                // Bounds check for shared memory
                if (sY < 0 || sY >= inputTileDim) continue;

                for (int kx = startKx; kx < kernelSize; kx += stride)
                {
                    int globalInX = (outPos.x + params.padding - kx) / stride;
                    int sX = globalInX - tileOriginIn.x;

                    if (sX < 0 || sX >= inputTileDim) continue;
                    
                    int sharedIdx = sY * inputTileDim + sX;

                    // Iterate over the batch of Input Channels currently in Shared Mem
                    for (int inOffset = 0; inOffset < inputChannelBatchSize; ++inOffset)
                    {
                        int globalInC = inCBase + inOffset;
                        if (globalInC >= inChannels) break;

                        float pixelVal = s_inputTile[inOffset][sharedIdx];

                        // Unroll Output Channel computation
                        [unroll]
                        for (int outOffset = 0; outOffset < outputChannelBatchSize; ++outOffset)
                        {
                            int globalOutC = outChannelStart + outOffset;
                            if (globalOutC >= outChannels) continue;

                            // ---------------------------------------------------------
                            // OPTIMIZED WEIGHT INDEXING
                            // Layout: [InC][KH][KW][OutC]
                            // 
                            // Since 'globalOutC' varies fastest in this loop (stride 1),
                            // we get coalesced memory reads.
                            // ---------------------------------------------------------
                            int64_t wIdx = 
                                (int64_t)globalInC  * (kernelSize * kernelSize * outChannels) +
                                (int64_t)ky         * (kernelSize * outChannels) + 
                                (int64_t)kx         * (outChannels) + 
                                (int64_t)globalOutC;

                            outputAccumulators[outOffset] += pixelVal * params.weights[wIdx];
                        }
                    }
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // --- 5. Write Output ---

    if (validOutput)
    {
        for (int o = 0; o < outputChannelBatchSize; ++o)
        {
            int globalOutC = outChannelStart + o;
            if (globalOutC < outChannels)
            {
                // Output Layout: [H, W, OutC]
                int64_t outIdx = ((int64_t)outPos.y * params.outputImageWidth + outPos.x) * outChannels + globalOutC;
                params.outputImage[outIdx] = outputAccumulators[o];
            }
        }
    }
}
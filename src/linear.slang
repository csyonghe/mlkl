implementing mlkl;

struct LinearParams<T : ITensorElement, TIn : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>> 
{
    TIn input;
    FOut outFunc;      // Value Transformer (Pull-based)
    TSink outputSink;  // Address/Storage Transformer (Push-based)
    T* weights;        // Weights in element type T
    T* bias;           // Bias in element type T
    uint M; // Batch * Seq
    uint K; // DimIn
    uint N; // DimOut
    uint has_bias;
};

// ============================================================================
// Optimized Tiled GEMM with Register Blocking
// ============================================================================
// Each thread computes THREAD_M x THREAD_N output elements (register blocking)
// This increases arithmetic intensity and reduces shared memory traffic per output
//
// Block dimensions: (TILE_N/THREAD_N, TILE_M/THREAD_M, 1)
// Default: TILE_M=64, TILE_N=64 with THREAD_M=4, THREAD_N=4 → 16x16 = 256 threads
//
// Key optimizations:
// 1. Register blocking: each thread computes 4x4=16 outputs instead of 1
// 2. Transposed B tile for coalesced shared memory access during compute
// 3. Outer product formulation for efficient register reuse

static const int THREAD_M = 4;  // Each thread computes 4 rows
static const int THREAD_N = 4;  // Each thread computes 4 cols

[numthreads(TILE_N / THREAD_N, TILE_M / THREAD_M, 1)]
void linearTiled<let TILE_M : int, let TILE_N : int, let TILE_K : int, T : ITensorElement, TIn : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<T, TIn, TSink, FOut>, CDataLayout> params)
{
    // Thread and block dimensions
    static const int THREADS_X = TILE_N / THREAD_N;
    static const int THREADS_Y = TILE_M / THREAD_M;
    static const int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;

    // Shared memory tiles - transposed B for coalesced access during compute
    static groupshared T.UnpackedType s_A[TILE_M][TILE_K + 1]; // +1 to avoid bank conflicts
    static groupshared T.UnpackedType s_B[TILE_K][TILE_N + 1]; // Transposed layout

    uint threadIdX = groupThreadID.x;
    uint threadIdY = groupThreadID.y;
    uint localThreadId = threadIdY * THREADS_X + threadIdX;

    // Base output position for this thread
    uint baseRow = groupID.y * TILE_M + threadIdY * THREAD_M;
    uint baseCol = groupID.x * TILE_N + threadIdX * THREAD_N;

    // Register file for accumulating results
    T.UnpackedType acc[THREAD_M][THREAD_N];
    [unroll]
    for (int m = 0; m < THREAD_M; m++)
    {
        [unroll]
        for (int n = 0; n < THREAD_N; n++)
            acc[m][n] = T.UnpackedType(0);
    }

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // Loop over K dimension in tiles
    uint numTiles = (params.K + TILE_K - 1) / TILE_K;
    for (uint t = 0; t < numTiles; t++)
    {
        uint kTileStart = t * TILE_K;

        // --- Collaborative Load Input A into s_A[TILE_M][TILE_K] ---
        for (uint i = localThreadId; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK)
        {
            uint localRow = i / TILE_K;
            uint localK = i % TILE_K;
            uint globalRow = groupID.y * TILE_M + localRow;
            uint globalK = kTileStart + localK;

            if (globalRow < params.M && globalK < params.K)
                s_A[localRow][localK] = params.input.eval(Coord(globalRow, globalK), emptyInput);
            else
                s_A[localRow][localK] = T.UnpackedType(0);
        }

        // --- Collaborative Load Weights B into s_B[TILE_K][TILE_N] (transposed) ---
        // Weight layout: weights[col * K + k] -> we want s_B[k][col]
        for (uint i = localThreadId; i < TILE_N * TILE_K; i += THREADS_PER_BLOCK)
        {
            uint localK = i / TILE_N;
            uint localCol = i % TILE_N;
            uint globalCol = groupID.x * TILE_N + localCol;
            uint globalK = kTileStart + localK;

            if (globalCol < params.N && globalK < params.K)
                s_B[localK][localCol] = params.weights[globalCol * params.K + globalK].unpack();
            else
                s_B[localK][localCol] = T.UnpackedType(0);
        }

        GroupMemoryBarrierWithGroupSync();

        // --- Compute: each thread computes THREAD_M x THREAD_N outputs ---
        [unroll]
        for (uint k = 0; k < TILE_K; k++)
        {
            // Load A values into registers (THREAD_M values)
            T.UnpackedType regA[THREAD_M];
            [unroll]
            for (int m = 0; m < THREAD_M; m++)
                regA[m] = s_A[threadIdY * THREAD_M + m][k];

            // Load B values into registers (THREAD_N values)
            T.UnpackedType regB[THREAD_N];
            [unroll]
            for (int n = 0; n < THREAD_N; n++)
                regB[n] = s_B[k][threadIdX * THREAD_N + n];

            // Outer product accumulation
            [unroll]
            for (int m = 0; m < THREAD_M; m++)
            {
                [unroll]
                for (int n = 0; n < THREAD_N; n++)
                    acc[m][n] = acc[m][n] + regA[m] * regB[n];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // --- Writeback: each thread writes THREAD_M x THREAD_N outputs ---
    [unroll]
    for (int m = 0; m < THREAD_M; m++)
    {
        uint row = baseRow + m;
        if (row >= params.M) continue;

        [unroll]
        for (int n = 0; n < THREAD_N; n++)
        {
            uint col = baseCol + n;
            if (col >= params.N) continue;

            T.UnpackedType result = acc[m][n];

            if (params.has_bias != 0)
                result = result + params.bias[col].unpack();

            Coord outCoord = Coord(row, col);
            Input<T> outInput = {};
            outInput.value = result;
            T.UnpackedType finalVal = params.outFunc.eval(outCoord, outInput);
            params.outputSink.store(outCoord, finalVal);
        }
    }
}

// We keep the signature identical to avoid changing the C++ side
[numthreads(32, 8, 1)]
void linearBruteforce<let TILE_M : int, let TILE_N : int, let TILE_K : int, T : ITensorElement, TIn : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<T, TIn, TSink, FOut>, CDataLayout> params)
{
    // 1. Calculate the logical Row (M) and Column (N) this thread is responsible for
    uint row = groupID.y * TILE_M + groupThreadID.y;
    uint col = groupID.x * TILE_N + groupThreadID.x;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // 2. Bounds check
    if (row < params.M && col < params.N)
    {
        T.UnpackedType acc = T.UnpackedType(0);

        // 3. Brute force dot product: Row of Input @ Column of Weights^T
        for (uint k = 0; k < params.K; k++)
        {
            // Pull input value: input[row, k]
            T.UnpackedType inVal = params.input.eval(Coord(row, k), emptyInput);
            
            // Read weight: weight[col, k]
            T.UnpackedType weightVal = params.weights[col * params.K + k].unpack();
            
            acc = acc + inVal * weightVal;
        }

        // 4. Add Bias
        if (params.has_bias != 0)
        {
            acc = acc + params.bias[col].unpack();
        }

        // 5. Apply Value Fusion (e.g., ReLU or Identity)
        Coord outCoord = Coord(row, col);
        Input<T> outInput = {};
        outInput.value = acc;
        T.UnpackedType finalVal = params.outFunc.eval(outCoord, outInput);

        // 6. Apply Storage Fusion (Push to Sink)
        params.outputSink.store(outCoord, finalVal);
    }
}

// ============================================================================
// Warp-Cooperative GEMV Kernel - Optimized for small batch sizes (1-8)
// ============================================================================
// Key optimizations:
// 1. Input vector cached in shared memory - loaded once, reused for all N
// 2. Warp-cooperative reduction: 32 threads split K, use WaveActiveSum
// 3. Each warp handles multiple output columns to maximize parallelism
//
// Thread organization:
// - Block has GEMV_WARPS_PER_BLOCK warps (e.g., 8 warps = 256 threads)
// - Each warp handles GEMV_COLS_PER_WARP output columns
// - Within each column, 32 threads split the K reduction
//
// Dispatch: x = ceil(N / (GEMV_WARPS_PER_BLOCK * GEMV_COLS_PER_WARP)), y = batchSize, z = 1

static const int GEMV_WARP_SIZE = 32;
static const int GEMV_WARPS_PER_BLOCK = 8;     // 8 warps = 256 threads
static const int GEMV_COLS_PER_WARP = 4;       // Each warp handles 4 output columns
static const int GEMV_BLOCK_SIZE = GEMV_WARP_SIZE * GEMV_WARPS_PER_BLOCK;  // 256
static const int GEMV_COLS_PER_BLOCK = GEMV_WARPS_PER_BLOCK * GEMV_COLS_PER_WARP;  // 32
static const int GEMV_K_TILE = 1024;           // How much K to cache in shared memory at once

[numthreads(GEMV_BLOCK_SIZE, 1, 1)]
void linearGemv<let TILE_M : int, let TILE_N : int, let TILE_K : int, T : ITensorElement, TIn : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<T, TIn, TSink, FOut>, CDataLayout> params)
{
    // Shared memory for input vector tile
    static groupshared T.UnpackedType s_input[GEMV_K_TILE];
    
    uint localThreadId = groupThreadID.x;
    uint batchIdx = groupID.y;  // Which batch item this block handles
    
    // Warp and lane indices
    uint warpId = localThreadId / GEMV_WARP_SIZE;
    uint laneId = localThreadId % GEMV_WARP_SIZE;
    
    // Which output columns this warp handles
    uint baseCol = groupID.x * GEMV_COLS_PER_BLOCK + warpId * GEMV_COLS_PER_WARP;
    
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    
    // Accumulators for GEMV_COLS_PER_WARP columns
    T.UnpackedType acc[GEMV_COLS_PER_WARP];
    for (int c = 0; c < GEMV_COLS_PER_WARP; c++)
        acc[c] = T.UnpackedType(0);
    
    // Process K in tiles to cache input in shared memory
    uint numKTiles = (params.K + GEMV_K_TILE - 1) / GEMV_K_TILE;
    
    for (uint kTile = 0; kTile < numKTiles; kTile++)
    {
        uint kTileStart = kTile * GEMV_K_TILE;
        uint kTileEnd = min(kTileStart + GEMV_K_TILE, params.K);
        uint kTileSize = kTileEnd - kTileStart;
        
        // Collaboratively load input vector tile into shared memory
        for (uint i = localThreadId; i < kTileSize; i += GEMV_BLOCK_SIZE)
        {
            uint globalK = kTileStart + i;
            s_input[i] = params.input.eval(Coord(batchIdx, globalK), emptyInput);
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // Each lane processes K elements strided by warp size
        // Lane 0: k=0,32,64,... Lane 1: k=1,33,65,...
        for (int c = 0; c < GEMV_COLS_PER_WARP; c++)
        {
            uint col = baseCol + c;
            if (col < params.N)
            {
                T.UnpackedType partialSum = T.UnpackedType(0);
                
                // Strided access over K within this tile
                for (uint k = laneId; k < kTileSize; k += GEMV_WARP_SIZE)
                {
                    uint globalK = kTileStart + k;
                    T.UnpackedType inVal = s_input[k];
                    T.UnpackedType w = params.weights[col * params.K + globalK].unpack();
                    partialSum = partialSum + inVal * w;
                }
                
                // Warp-level reduction using wave intrinsics
                acc[c] = acc[c] + partialSum.waveActiveSum();
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Only lane 0 of each warp writes the final results
    if (laneId == 0)
    {
        for (int c = 0; c < GEMV_COLS_PER_WARP; c++)
        {
            uint col = baseCol + c;
            if (col < params.N)
            {
                T.UnpackedType result = acc[c];
                
                // Add bias
                if (params.has_bias != 0)
                    result = result + params.bias[col].unpack();
                
                // Apply output transformation
                Coord outCoord = Coord(batchIdx, col);
                Input<T> outInput = {};
                outInput.value = result;
                T.UnpackedType finalVal = params.outFunc.eval(outCoord, outInput);
                params.outputSink.store(outCoord, finalVal);
            }
        }
    }
}
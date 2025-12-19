implementing mlkl;

struct LinearParams<TIn: IExpr, TSink: ISink, FOut: IExpr> {
    TIn input;
    FOut outFunc;      // Value Transformer (Pull-based)
    TSink outputSink;  // Address/Storage Transformer (Push-based)
    float* weights;
    float* bias;
    uint M; // Batch * Seq
    uint K; // DimIn
    uint N; // DimOut
    uint has_bias;
};

[numthreads(32, 8, 1)]
void linearTiled<let TILE_M: int, let TILE_N: int, let TILE_K: int, TIn: IExpr, TSink: ISink, FOut: IExpr>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<TIn, TSink, FOut>, CDataLayout> params)
{
    // Local SRAM for tiles
    static groupshared float s_A[TILE_M][TILE_K];
    static groupshared float s_B[TILE_N][TILE_K];

    uint row = groupID.y * TILE_M + groupThreadID.y;
    uint col = groupID.x * TILE_N + groupThreadID.x;

    float acc = 0.0f;

    uint numTiles = (params.K + TILE_K - 1) / TILE_K;
    for (uint t = 0; t < numTiles; t++)
    {
        // 1. Collaborative load Input [M, K]
        uint k_idx_A = t * TILE_K + groupThreadID.x % TILE_K;
        if (row < params.M && k_idx_A < params.K)
            s_A[groupThreadID.y][groupThreadID.x % TILE_K] = params.input.eval(row * params.K + k_idx_A, Input(0.0f));
        else
            s_A[groupThreadID.y][groupThreadID.x % TILE_K] = 0.0f;

        // 2. Collaborative load Weight [N, K]
        uint k_idx_B = t * TILE_K + groupThreadID.y % TILE_K;
        if (col < params.N && k_idx_B < params.K)
            s_B[groupThreadID.x][groupThreadID.y % TILE_K] = params.weights[col * params.K + k_idx_B];
        else
            s_B[groupThreadID.x][groupThreadID.y % TILE_K] = 0.0f;

        GroupMemoryBarrierWithGroupSync();

        // 3. Tile Compute
        [unroll]
        for (uint k = 0; k < TILE_K; k++)
        {
            acc += s_A[groupThreadID.y][k] * s_B[groupThreadID.x][k];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 4. Final Writeback
    if (row < params.M && col < params.N)
    {
        if (params.has_bias != 0)
            acc += params.bias[col];
        
        uint linearIdx = row * params.N + col;

        // --- DUAL TRANSFORM FUSION ---
        // 1. Value Transform (Pull): Apply ReLU, Add, etc.
        float finalVal = params.outFunc.eval(linearIdx, Input(acc));

        // 2. Storage Transform (Push): Handle permutations and memory write
        params.outputSink.store(linearIdx, finalVal);
    }
}

// We keep the signature identical to avoid changing the C++ side
[numthreads(32, 8, 1)]
void linearBruteforce<let TILE_M: int, let TILE_N: int, let TILE_K: int, TIn: IExpr, TSink: ISink, FOut: IExpr>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<TIn, TSink, FOut>, CDataLayout> params)
{
    // 1. Calculate the logical Row (M) and Column (N) this thread is responsible for
    // Based on the grid dispatch in C++: gridX = N / TILE_N, gridY = M / TILE_M
    uint row = groupID.y * TILE_M + groupThreadID.y;
    uint col = groupID.x * TILE_N + groupThreadID.x;

    // 2. Bounds check
    if (row < params.M && col < params.N)
    {
        float acc = 0.0f;

        // 3. Brute force dot product: Row of Input @ Column of Weights^T
        // Recall PyTorch layout [Out, In] means weights[col] is the start of a row of length K
        for (uint k = 0; k < params.K; k++)
        {
            // Pull input value: input[row, k]
            float inVal = params.input.eval(row * params.K + k, Input(0.0f));
            
            // Read weight: weight[col, k]
            float weightVal = params.weights[col * params.K + k];
            
            acc += inVal * weightVal;
        }

        // 4. Add Bias
        if (params.has_bias != 0)
        {
            acc += params.bias[col];
        }

        // 5. Apply Value Fusion (e.g., ReLU or Identity)
        uint linearIdx = row * params.N + col;
        float finalVal = params.outFunc.eval(linearIdx, Input(acc));

        // 6. Apply Storage Fusion (Push to Sink)
        params.outputSink.store(linearIdx, finalVal);
    }
}
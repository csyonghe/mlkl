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
// Use the specialization constants directly in the attribute
[numthreads(TILE_N, TILE_M, 1)]
void linearTiled<let TILE_M: int, let TILE_N: int, let TILE_K: int, TIn: IExpr, TSink: ISink, FOut: IExpr>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<TIn, TSink, FOut>, CDataLayout> params)
{
    // 1. Shared memory tiles
    // We use TILE_K as the shared dimension for both Input and Weights
    static groupshared float s_A[TILE_M][TILE_K+1]; // +1 to reduce bank conflict
    static groupshared float s_B[TILE_N][TILE_K+1];

    uint row = groupID.y * TILE_M + groupThreadID.y;
    uint col = groupID.x * TILE_N + groupThreadID.x;

    float acc = 0.0f;

    // 2. Loop over tiles along the K dimension
    uint numTiles = (params.K + TILE_K - 1) / TILE_K;
    for (uint t = 0; t < numTiles; t++)
    {
        // --- Collaborative Load Input (s_A) ---
        // Each thread in the TILE_N x TILE_M block helps load the TILE_M x TILE_K tile.
        // We linearize the threads to map them to the tile elements.
        uint localThreadId = groupThreadID.y * TILE_N + groupThreadID.x;
        uint threadsPerBlock = TILE_M * TILE_N;

        for (uint i = localThreadId; i < TILE_M * TILE_K; i += threadsPerBlock)
        {
            uint localRow = i / TILE_K;
            uint localK = i % TILE_K;
            uint globalRow = groupID.y * TILE_M + localRow;
            uint globalK = t * TILE_K + localK;

            if (globalRow < params.M && globalK < params.K)
                s_A[localRow][localK] = params.input.eval(globalRow * params.K + globalK, Input(0.0f));
            else
                s_A[localRow][localK] = 0.0f;
        }

        // --- Collaborative Load Weights (s_B) ---
        for (uint i = localThreadId; i < TILE_N * TILE_K; i += threadsPerBlock)
        {
            uint localCol = i / TILE_K;
            uint localK = i % TILE_K;
            uint globalCol = groupID.x * TILE_N + localCol;
            uint globalK = t * TILE_K + localK;

            if (globalCol < params.N && globalK < params.K)
                s_B[localCol][localK] = params.weights[globalCol * params.K + globalK];
            else
                s_B[localCol][localK] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // --- 3. Compute Dot Product for Tile ---
        [unroll]
        for (uint k = 0; k < TILE_K; k++)
        {
            acc += s_A[groupThreadID.y][k] * s_B[groupThreadID.x][k];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // --- 4. Writeback ---
    if (row < params.M && col < params.N)
    {
        if (params.has_bias != 0)
            acc += params.bias[col];
        
        uint linearIdx = row * params.N + col;
        float finalVal = params.outFunc.eval(linearIdx, Input(acc));
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
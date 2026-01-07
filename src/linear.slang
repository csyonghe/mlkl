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

// Use the specialization constants directly in the attribute
[numthreads(TILE_N, TILE_M, 1)]
void linearTiled<let TILE_M : int, let TILE_N : int, let TILE_K : int, T : ITensorElement, TIn : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<T, TIn, TSink, FOut>, CDataLayout> params)
{
    // 1. Shared memory tiles (in T.UnpackedType for computation)
    static groupshared T.UnpackedType s_A[TILE_M][TILE_K+1]; // +1 to reduce bank conflict
    static groupshared T.UnpackedType s_B[TILE_N][TILE_K+1];

    uint row = groupID.y * TILE_M + groupThreadID.y;
    uint col = groupID.x * TILE_N + groupThreadID.x;

    T.UnpackedType acc = T.UnpackedType(0);

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // 2. Loop over tiles along the K dimension
    uint numTiles = (params.K + TILE_K - 1) / TILE_K;
    for (uint t = 0; t < numTiles; t++)
    {
        // --- Collaborative Load Input (s_A) ---
        uint localThreadId = groupThreadID.y * TILE_N + groupThreadID.x;
        uint threadsPerBlock = TILE_M * TILE_N;

        for (uint i = localThreadId; i < TILE_M * TILE_K; i += threadsPerBlock)
        {
            uint localRow = i / TILE_K;
            uint localK = i % TILE_K;
            uint globalRow = groupID.y * TILE_M + localRow;
            uint globalK = t * TILE_K + localK;

            if (globalRow < params.M && globalK < params.K)
                s_A[localRow][localK] = params.input.eval(Coord(globalRow, globalK), emptyInput);
            else
                s_A[localRow][localK] = T.UnpackedType(0);
        }

        // --- Collaborative Load Weights (s_B) ---
        for (uint i = localThreadId; i < TILE_N * TILE_K; i += threadsPerBlock)
        {
            uint localCol = i / TILE_K;
            uint localK = i % TILE_K;
            uint globalCol = groupID.x * TILE_N + localCol;
            uint globalK = t * TILE_K + localK;

            if (globalCol < params.N && globalK < params.K)
                s_B[localCol][localK] = params.weights[globalCol * params.K + globalK].unpack();
            else
                s_B[localCol][localK] = T.UnpackedType(0);
        }

        GroupMemoryBarrierWithGroupSync();

        // --- 3. Compute Dot Product for Tile ---
        [unroll]
        for (uint k = 0; k < TILE_K; k++)
        {
            acc = acc + s_A[groupThreadID.y][k] * s_B[groupThreadID.x][k];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // --- 4. Writeback ---
    if (row < params.M && col < params.N)
    {
        if (params.has_bias != 0)
            acc = acc + params.bias[col].unpack();
        
        Coord outCoord = Coord(row, col);
        Input<T> outInput = {};
        outInput.value = acc;
        T.UnpackedType finalVal = params.outFunc.eval(outCoord, outInput);
        params.outputSink.store(outCoord, finalVal);
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

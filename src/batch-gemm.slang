implementing mlkl;

struct BatchGemmParams<T : ITensorElement, TA : IExpr<T>, TB : IExpr<T>, TC : IExpr<T>, TSink : ISink<T>, TOut : IExpr<T>>
{
    TA A;
    TB B;
    TC C;
    TSink Sink;
    TOut FOut;
    
    uint M; 
    uint N; 
    uint K; 
    
    float alpha;  // Keep as float for simplicity
    float beta;   // Keep as float for simplicity
};

// Dispatch: x=N/16, y=M/16, z=BatchSize
[numthreads(16, 16, 1)]
void batchGemm<T : ITensorElement, TA : IExpr<T>, TB : IExpr<T>, TC : IExpr<T>, TSink : ISink<T>, TOut : IExpr<T>>(
    ConstantBuffer<BatchGemmParams<T, TA, TB, TC, TSink, TOut>, CDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint col = id.x; // n
    uint row = id.y; // m
    uint batchIdx = id.z;

    if (row >= params.M || col >= params.N) return;

    // Strides for logical dimensions
    // A: [Batch, M, K] -> Stride K
    // B: [Batch, K, N] -> Stride N
    // C: [Batch, M, N] -> Stride N
    
    // Batch Offsets in Logical Space
    uint batchOffsetA = batchIdx * params.M * params.K;
    uint batchOffsetB = batchIdx * params.K * params.N;
    uint batchOffsetC = batchIdx * params.M * params.N;

    // Accumulate in T.UnpackedType precision
    T.UnpackedType sum = T.UnpackedType(0);
    Input<T> emptyInput = {}; 
    emptyInput.value = T.UnpackedType(0);

    for (uint k = 0; k < params.K; k++)
    {
        // A[batch, row, k]
        T.UnpackedType valA = params.A.eval(Coord(batchIdx, row, k), emptyInput);

        // B[batch, k, col]
        T.UnpackedType valB = params.B.eval(Coord(batchIdx, k, col), emptyInput);
        
        sum = sum + valA * valB;
    }

    // C[batch, row, col]
    Coord outCoord = Coord(batchIdx, row, col);
    T.UnpackedType valC = params.C.eval(outCoord, emptyInput);

    // alpha/beta are float, convert to T.UnpackedType for arithmetic
    T.UnpackedType res = T.UnpackedType(params.alpha) * sum + T.UnpackedType(params.beta) * valC;

    // Apply Output Epilogue (Activation, etc.)
    Input<T> outInput = {};
    outInput.value = res;
    T.UnpackedType finalVal = params.FOut.eval(outCoord, outInput);

    params.Sink.store(outCoord, finalVal);
}

implementing mlkl;

struct BatchGemmParams<TA:IExpr, TB:IExpr, TC:IExpr, TSink:ISink, TOut:IExpr>
{
    TA A;
    TB B;
    TC C;
    TSink Sink;
    TOut FOut;
    
    uint M; 
    uint N; 
    uint K; 
    
    float alpha;
    float beta;
};

// Dispatch: x=N/16, y=M/16, z=BatchSize
[numthreads(16, 16, 1)]
void batchGemm<TA:IExpr, TB:IExpr, TC:IExpr, TSink:ISink, TOut:IExpr>(
    ConstantBuffer<BatchGemmParams<TA, TB, TC, TSink, TOut>, CDataLayout> params,
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

    float sum = 0.0f;
    Input emptyInput = {}; 
    emptyInput.value = 0.0f;

    for (uint k = 0; k < params.K; k++)
    {
        // Resolve Linear Index for A [row, k]
        uint idxA = batchOffsetA + row * params.K + k;
        float valA = params.A.eval(idxA, emptyInput);

        // Resolve Linear Index for B [k, col]
        uint idxB = batchOffsetB + k * params.N + col;
        float valB = params.B.eval(idxB, emptyInput);
        
        sum += valA * valB;
    }

    uint idxC = batchOffsetC + row * params.N + col;
    
    // Evaluate C (Bias/Accumulator)
    float valC = params.C.eval(idxC, emptyInput);

    float res = params.alpha * sum + params.beta * valC;

    // Apply Output Epilogue (Activation, etc.)
    Input outInput = {};
    outInput.value = res;
    float finalVal = params.FOut.eval(idxC, outInput);

    params.Sink.store(idxC, finalVal);
}
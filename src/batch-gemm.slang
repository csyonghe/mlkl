implementing mlkl;

struct BatchGemmParams
{
    float* A;
    float* B;
    float* C;
    
    uint M; // Rows of Output C
    uint N; // Cols of Output C
    uint K; // Inner dimension
    
    float alpha;
    float beta;

    uint transposeA; // 1 if true
    uint transposeB; // 1 if true
};

// Dispatch: x=N, y=M, z=BatchSize
[numthreads(16, 16, 1)]
void batchGemm(
    ConstantBuffer<BatchGemmParams, CDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint col = id.x; // n
    uint row = id.y; // m
    uint batchIdx = id.z;

    if (row >= params.M || col >= params.N) return;

    // Calculate strides based on dimensions
    // A: [Batch, M, K] normally. If TransposeA: [Batch, K, M]
    // B: [Batch, K, N] normally. If TransposeB: [Batch, N, K]
    // C: [Batch, M, N]
    
    // Total elements per matrix in the batch
    uint sizeA = params.M * params.K; 
    uint sizeB = params.K * params.N;
    uint sizeC = params.M * params.N;

    // Pointer arithmetic
    float* matA = params.A + batchIdx * sizeA;
    float* matB = params.B + batchIdx * sizeB;
    float* matC = params.C + batchIdx * sizeC;

    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++)
    {
        // Resolve Index A
        // Normal: A[row, k] -> row * K + k
        // Transposed: A[k, row] -> k * M + row
        uint idxA = params.transposeA != 0 ? (k * params.M + row) : (row * params.K + k);
        
        // Resolve Index B
        // Normal: B[k, col] -> k * N + col
        // Transposed: B[col, k] -> col * K + k
        uint idxB = params.transposeB != 0 ? (col * params.K + k) : (k * params.N + col);
        
        sum += matA[idxA] * matB[idxB];
    }

    uint idxC = row * params.N + col;
    if (params.beta != 0.0f)
    {
        matC[idxC] = params.alpha * sum + params.beta * matC[idxC];
    }
    else
    {
        matC[idxC] = params.alpha * sum;
    }
}
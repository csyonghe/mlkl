#include "batch-gemm.h"


BatchGemmKernel::BatchGemmKernel(InferencingContext* ctx)
    : context(ctx)
{
    pipeline = context->createComputePipeline("batchGemm", {});
}

// Allocate output buffer C [BatchSize, M, N]
BufferView BatchGemmKernel::allocResultBuffer(int batchSize, int m, int n)
{
    size_t size = (size_t)batchSize * m * n * sizeof(float);
    return context->allocScratchBuffer(size, "BatchGemm_Result");
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    BufferView C, // Output
    BufferView A,
    BufferView B,
    int batchSize,
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    bool transposeA,
    bool transposeB)
{
    struct BatchGemmParams
    {
        rhi::DeviceAddress A, B, C;
        uint32_t M, N, K;
        float alpha, beta;
        uint32_t transposeA, transposeB;
    } params;

    params.A = A.getDeviceAddress();
    params.B = B.getDeviceAddress();
    params.C = C.getDeviceAddress();
    params.M = m;
    params.N = n;
    params.K = k;
    params.alpha = alpha;
    params.beta = beta;
    params.transposeA = transposeA ? 1 : 0;
    params.transposeB = transposeB ? 1 : 0;

    // Dispatch (N, M) groups
    uint32_t groupX = (n + 15) / 16;
    uint32_t groupY = (m + 15) / 16;
    task.dispatchKernel(pipeline, groupX, groupY, batchSize, params);
}
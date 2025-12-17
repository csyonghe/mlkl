#pragma once
#include "kernel-base.h"

class BatchGemmKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

public:
    BatchGemmKernel(InferencingContext* ctx);

    // Allocate output buffer C [BatchSize, M, N]
    BufferView allocResultBuffer(int batchSize, int m, int n);

    void queueExecute(
        InferencingTask& task,
        BufferView C, // Output
        BufferView A,
        BufferView B,
        int batchSize,
        int m,
        int n,
        int k,
        float alpha = 1.0f,
        float beta = 0.0f,
        bool transposeA = false,
        bool transposeB = false);
};
#pragma once
#include "kernel-base.h"

class SoftmaxKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

public:
    SoftmaxKernel(InferencingContext* ctx);

    BufferView allocateResultBuffer(int rows, int cols);

    // rows: Number of independent vectors (Batch * Heads * SeqQ)
    // cols: Length of each vector (SeqKV)
    void queueExecute(
        InferencingTask& task,
        BufferView output,
        BufferView input,
        int rows,
        int cols);
};
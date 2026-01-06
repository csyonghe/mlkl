#pragma once
#include "kernel-base.h"

// Given an input tensor of shape [Rows, Cols], compute the softmax along each row.
// Resulting a tensor of shape [Rows, Cols] where each row sums to 1.0.
class SoftmaxKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

public:
    SoftmaxKernel(InferencingContext* ctx);

    TensorView allocateResultBuffer(ElementType elementType, int rows, int cols);

    // rows: Number of independent vectors (Batch * Heads * SeqQ)
    // cols: Length of each vector (SeqKV)
    void queueExecute(InferencingTask& task, TensorView output, TensorView input);
};
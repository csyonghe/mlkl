#pragma once
#include "elementwise.h"
#include "kernel-base.h"

class BatchGemmKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

    ProgramNode programA;
    ProgramNode programB;
    ProgramNode programC;
    ProgramNode programOut;

public:
    BatchGemmKernel(InferencingContext* ctx, Expr A, Expr B, Expr C, Expr Out);

    BufferView allocResultBuffer(int batchSize, int m, int n);

    // Eval takes shape params explicitly, plus inputs for all expressions
    void queueExecute(
        InferencingTask& task,
        BufferView output,
        int M,
        int N,
        int K,
        int batchSize,
        float alpha,
        float beta,
        const Dictionary<Expr, InputInfo>& inputs);
};
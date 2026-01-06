#pragma once
#include "elementwise.h"
#include "kernel-base.h"

// Compute batched matrix multiply: alpha*A[i]*B[i] + beta*C[i], where i = 0..batchSize-1
// Input layout:
// - A: [BatchSize M K]
// - B: [BatchSize K N]
// - C: [BatchSize M N]
// Output: [BatchSize M N]
//
class BatchGemmKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

    ProgramNode programA;
    ProgramNode programB;
    ProgramNode programC;
    ProgramNode programOut;
    SinkExpr sinkExpr;

public:
    BatchGemmKernel(
        InferencingContext* ctx,
        Expr A,
        Expr B,
        Expr C,
        SinkExpr sinkExpr,
        Expr outExpr);

    BatchGemmKernel(InferencingContext* ctx, Expr outExpr = kernelOutput())
        : BatchGemmKernel(ctx, buffer(), buffer(), buffer(), bufferSink(), outExpr)
    {
    }

    TensorView allocateResultBuffer(ElementType elementType, int batchSize, int m, int n);

    // Eval takes shape params explicitly, plus inputs for all expressions
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        float alpha,
        float beta,
        EvalContext& ctx);
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        float alpha,
        float beta,
        const Dictionary<Expr, InputInfo>& inputs);
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        float alpha,
        float beta,
        ArrayView<InputInfo> inputs);
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        float alpha,
        float beta,
        const std::initializer_list<InputInfo>& inputs);
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        float alpha,
        float beta,
        TensorView A,
        TensorView B,
        TensorView C);
};
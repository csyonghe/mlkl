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
// USAGE PATTERNS:
//
// 1. Simple matrix multiply (no bias):
//    Use constant(0.0f) for C expression and beta=0:
//      BatchGemmKernel(ctx, bufA, bufB, constant(0.0f), bufferSink(), kernelOutput())
//      kernel.queueExecute(task, out, alpha, 0.0f, {inputA, inputB})
//
// 2. Fused transpose (e.g., for Q @ K^T):
//    Use transpose() in the B expression:
//      auto kExpr = buffer();
//      BatchGemmKernel(ctx, qExpr, transpose(kExpr, 1, 2), constant(0.0f), ...)
//
// 3. With initializer_list inputs (order matches buffer() order in expressions):
//    kernel.queueExecute(task, out, alpha, beta, {tensorA, tensorB})
//
// See testBatchGemm and testFusedBatchGemm for complete examples.
//
// COMMON MISTAKES:
// - Using buffer() for C but not binding it - use constant(0.0f) with beta=0 instead
// - Using permute() instead of transpose() - transpose(expr, dim1, dim2) is cleaner
class BatchGemmKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
    ElementType elementType;

    ProgramNode programA;
    ProgramNode programB;
    ProgramNode programC;
    ProgramNode programOut;
    SinkExpr sinkExpr;

public:
    BatchGemmKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr A,
        Expr B,
        Expr C,
        SinkExpr sinkExpr,
        Expr outExpr);

    // Convenience constructor defaulting to Float32.
    BatchGemmKernel(InferencingContext* ctx, Expr outExpr = kernelOutput())
        : BatchGemmKernel(
              ctx,
              ElementType::Float32,
              buffer(),
              buffer(),
              buffer(),
              bufferSink(),
              outExpr)
    {
    }
    BatchGemmKernel(
        InferencingContext* ctx,
        Expr A,
        Expr B,
        Expr C,
        SinkExpr sinkExpr,
        Expr outExpr)
        : BatchGemmKernel(ctx, ElementType::Float32, A, B, C, sinkExpr, outExpr)
    {
    }


    ElementType getElementType() const { return elementType; }

    TensorView allocateResultBuffer(ElementType elementType, int batchSize, int m, int n);

private:
    void validateTensorElementType(const TensorView& tv, const char* name) const;

public:
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
#pragma once
#include "elementwise.h"
#include "kernel-base.h"

// Given an input tensor of shape [Rows, Cols], compute the softmax along each row.
// Resulting a tensor of shape [Rows, Cols] where each row sums to 1.0.
class SoftmaxKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
    ElementType elementType;
    
    ProgramNode inputProgram;
    SinkExpr sinkExpr;

    void validateTensorElementType(const TensorView& tv, const char* name) const;

public:
    SoftmaxKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr inputExpr,
        SinkExpr sinkExpr);

    // Convenience constructor defaulting to Float32 with buffer input/output
    SoftmaxKernel(InferencingContext* ctx)
        : SoftmaxKernel(ctx, ElementType::Float32, buffer(), bufferSink())
    {
    }

    ElementType getElementType() const { return elementType; }

    TensorView allocateResultBuffer(ElementType elementType, int rows, int cols);

    // Execute with full EvalContext
    void queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx);

    // Execute with dictionary of inputs
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs)
    {
        return queueExecute(task, output, makeEvalContext(inputs));
    }

    // Execute with initializer list of inputs (for simple cases)
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs);

    // Simplified version: single input tensor
    void queueExecute(InferencingTask& task, TensorView output, TensorView input)
    {
        return queueExecute(task, output, std::initializer_list<InputInfo>{input});
    }
};

#pragma once

#include "elementwise.h"
#include "kernel-base.h"

class ConcatKernel : public RefObject
{
    InferencingContext* context;
    RefPtr<ElementwiseKernel> elementwiseKernel;
    Dictionary<int, Expr> mapOperandToExprNode;
    Expr axisExpr;
    int operandCount;

public:
    ConcatKernel(InferencingContext* ctx, int operandCount);

    // Concatenates N inputs along the specified axis.
    // inputs: List of buffers
    // inputShapes: List of shapes corresponding to inputs
    // axis: The dimension to concatenate along
    ComPtr<rhi::IBuffer> queueExecute(
        InferencingTask& task,
        ArrayView<rhi::IBuffer*> inputs,
        ArrayView<Shape> inputShapes,
        int axis);
};
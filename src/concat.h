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

    TensorView allocateResultBuffer(
        ElementType elementType,
        ArrayView<Shape> inputShapes,
        int axis);

    // Concatenates N inputs along the specified axis.
    // inputs: List of buffers
    // inputShapes: List of shapes corresponding to inputs
    // axis: The dimension to concatenate along
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        ArrayView<TensorView> inputs,
        int axis);
};
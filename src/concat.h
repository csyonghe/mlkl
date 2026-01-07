#pragma once

#include "elementwise.h"
#include "kernel-base.h"

// Concatenation kernel for joining tensors along an axis.
//
// FUSION OPPORTUNITY: Consider if concat can be avoided!
// - If concatenating then immediately processing, consider processing separately
// - For attention Q/K/V: compute separately rather than concat then split
// - Check if upstream kernels can write to different regions of same buffer
//
// This kernel is appropriate when:
// - Multiple tensors genuinely need to be joined for downstream processing
// - The concat result is used by a kernel that requires contiguous input
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
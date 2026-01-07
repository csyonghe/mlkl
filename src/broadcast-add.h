#pragma once

#include "elementwise.h"
#include "kernel-base.h"
#include "tensor.h"

// Standalone broadcast addition kernel.
//
// FUSION OPPORTUNITY: Consider fusing into adjacent kernels instead!
// - For residual connections: use (a + b) expression in ElementwiseKernel
// - Can combine with activation: silu(a + b), relu(a + b)
// - For bias addition after matmul: use kernel's built-in bias support
//
// Only use this standalone kernel when:
// - Broadcast semantics are specifically needed
// - No adjacent kernel supports the addition fusion
class BroadcastAddKernel : public RefObject
{
private:
    InferencingContext* context;
    RefPtr<ElementwiseKernel> kernel;

    // We store the generic expression nodes to bind inputs later
    Expr inputAExpr;
    Expr inputBExpr;

public:
    BroadcastAddKernel(InferencingContext* context);
    TensorView allocateResultBuffer(
        ElementType elementType,
        const Shape& shapeA,
        const Shape& shapeB);
    void queueExecute(
        InferencingTask& task,
        TensorView result,
        TensorView inputA,
        TensorView inputB);
};
#pragma once

#include "elementwise.h"
#include "kernel-base.h"
#include "tensor.h"

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
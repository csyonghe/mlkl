#pragma once

#include "elementwise.h"
#include "kernel-base.h"

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
    BufferView allocateResultBuffer(const Shape& shapeA, const Shape& shapeB, int batchSize = 1);
    void queueExecute(
        InferencingTask& task,
        BufferView result,
        BufferView inputA,
        const Shape& shapeA,
        BufferView inputB,
        const Shape& shapeB,
        int batchSize = 1);
};
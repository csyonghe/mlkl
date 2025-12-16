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

    ComPtr<rhi::IBuffer> queueExecute(
        InferencingTask& task,
        rhi::IBuffer* inputA,
        ArrayView<int> shapeA,
        rhi::IBuffer* inputB,
        ArrayView<int> shapeB,
        int batchSize = 1);
};
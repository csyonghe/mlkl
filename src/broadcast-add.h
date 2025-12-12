#include "kernel-base.h"

class BroadcastAddKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
public:
    BroadcastAddKernel(InferencingContext* context);
    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputA, ArrayView<int> shapeA, rhi::IBuffer* inputB, ArrayView<int> shapeB);
};

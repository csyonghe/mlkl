
#pragma once

#include "kernel-base.h"

class ConcatKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
public:
    ConcatKernel(InferencingContext* context);
    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputA, ArrayView<int> shapeA, rhi::IBuffer* inputB, ArrayView<int> shapeB, int axis);
};

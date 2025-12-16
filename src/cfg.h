#pragma once

#include "elementwise.h"
#include "kernel-base.h"

class ClassifierFreeGuidanceKernel : public RefObject
{
    InferencingContext* context;
    RefPtr<ElementwiseKernel> kernel;

    // We store the Expr nodes to bind inputs at runtime
    Expr uncond;
    Expr cond;
    Expr scale;

public:
    ClassifierFreeGuidanceKernel(InferencingContext* context);

    ComPtr<rhi::IBuffer> queueExecute(
        InferencingTask& task,
        rhi::IBuffer* batchedInput,
        int width,
        int height,
        int channels,
        float guidanceScale);
};
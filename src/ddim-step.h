#pragma once

#include "kernel-base.h"

class DDIMStepKernel : public RefObject
{
protected:
    RefPtr<InferencingContext> inferencingCtx;
    ComPtr<rhi::IComputePipeline> pipeline;

public:
    DDIMStepKernel(RefPtr<InferencingContext> inferencingCtx);
    void queueExecute(
        InferencingTask& task,
        BufferView currentImage,
        BufferView predictedNoise,
        BufferView outputImage,
        float alphaBar_t,    // Current cumulative alpha
        float alphaBar_prev, // Target cumulative alpha
        int width,
        int height,
        int channels,
        int batchSize = 1);
};
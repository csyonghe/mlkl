
#pragma once

#include "kernel-base.h"

class TimeEmbedingKernel : public RefObject
{
private:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

public:
    int outputChannels;
    TimeEmbedingKernel(InferencingContext* context, int outputChannels);

    SlangResult loadParams(TorchParamReader& reader);

    TensorView allocateResultBuffer(ElementType elementType, int batchSize);
    void queueExecute(InferencingTask& task, TensorView output, uint32_t timeStep);
};

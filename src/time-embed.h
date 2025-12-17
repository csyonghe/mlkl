
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

    BufferView allocResultBuffer(int batchSize);
    void queueExecute(
        InferencingTask& task,
        BufferView output,
        uint32_t timeStep,
        int batchSize = 1);
};

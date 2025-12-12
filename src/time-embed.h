
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
    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, uint32_t timeStepsBuffer);
};

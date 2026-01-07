#pragma once

#include "kernel-base.h"

class SafeTensorsReader;

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

    // Load from SafeTensors - time embedding MLP weights
    SlangResult loadParams(
        SafeTensorsReader& reader,
        UnownedStringSlice linear1WeightName,
        UnownedStringSlice linear1BiasName,
        UnownedStringSlice linear2WeightName,
        UnownedStringSlice linear2BiasName);

    TensorView allocateResultBuffer(ElementType elementType, int batchSize);
    void queueExecute(InferencingTask& task, TensorView output, uint32_t timeStep);
};

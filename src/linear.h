
#pragma once

#include "kernel-base.h"

class LinearKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
    int inputSize;
    int outputSize;
    int tileSize;

public:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;
    LinearKernel(
        InferencingContext* context,
        ActivationFunction activation,
        int tileSize,
        int inputSize,
        int outputSize);
    SlangResult loadParams(TorchParamReader& reader);
    BufferView allocateResultBuffer(int batchSize);
    void queueExecute(
        InferencingTask& task,
        BufferView output,
        BufferView inputVector,
        int batchSize = 1);
};

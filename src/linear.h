
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
    ComPtr<rhi::IBuffer> queueExecute(
        InferencingTask& task,
        rhi::IBuffer* inputVectorint,
        int batchSize = 1);
};

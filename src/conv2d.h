#pragma once

#include "kernel-base.h"

class Conv2DKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> tilePipeline;
    ComPtr<rhi::IComputePipeline> flatPipeline;
    ComPtr<rhi::IComputePipeline> flatWaveReducePipeline;

    InferencingContext* context;
public:
    int tileSize;
    int kernelSize;
    int inChannels;
    int outChannels;
    int stride;
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;
    ComPtr<rhi::IBuffer> weightsTransposedBuffer; // [outChannels, kernelSize, kernelSize, inChannels]
    ActivationFunction activation;
    String name;
    Conv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int stride, int inChannels, int outChannels, ActivationFunction activation = ActivationFunction::None, String name = "conv2d");

    SlangResult loadParams(TorchParamReader& reader, bool loadAndFuseBNorm);
    SlangResult loadParams(int kernelSize, int outputChannelCount, float* weightsData, float* biasesData);

    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int padding);
};
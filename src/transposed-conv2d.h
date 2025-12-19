#pragma once

#include "kernel-base.h"

// Transposed Conv2D Kernel
// Weights layout: [InChannels, KernelSize, KernelSize, OutChannels]
// Input/Output layout: [BatchSize, Height, Width, Channels]
class TransposedConv2DKernel : public RefObject
{
private:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;
    ComPtr<rhi::IComputePipeline> pipeline;
    ComPtr<rhi::IComputePipeline> flatPipeline;
    InferencingContext* context;

public:
    int tileSize;
    int kernelSize;
    int inChannels;
    int outChannels;
    int stride;
    String name;

    TransposedConv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        ActivationFunction activation = ActivationFunction::None,
        String name = "transConv2d");

    SlangResult loadParams(TorchParamReader& reader);
    SlangResult loadParams(
        int kernelSize,
        int outputChannelCount,
        float* weightsData,
        float* biasesData);

    BufferView allocateResultBuffer(int inputWidth, int inputHeight, int padding, int batchSize);

    void queueExecute(
        InferencingTask& task,
        BufferView outputImage,
        BufferView inputImage,
        int inputWidth,
        int inputHeight,
        int padding,
        int batchSize = 1);
};
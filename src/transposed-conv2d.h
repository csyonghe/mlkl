#pragma once

#include "kernel-base.h"

class TransposedConv2DKernel : public RefObject
{
private:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
public:
    int tileSize;
    int kernelSize;
    int inChannels;
    int outChannels;
    int stride;
    String name;

    TransposedConv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int stride, int inChannels, int outChannels, String name = "transConv2d");

    SlangResult loadParams(TorchParamReader& reader);
    SlangResult loadParams(int kernelSize, int outputChannelCount, float* weightsData, float* biasesData);

    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int padding);
};
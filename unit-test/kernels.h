#pragma once

#include "core/slang-basic.h"

#include "torch-reader.h"
#include "inference-context.h"

using namespace Slang;

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

enum class ActivationFunction
{
    None,
    ReLU,
    SiLU
};

const char* getActivationFuncName(ActivationFunction func);

class LinearKernel : public RefObject
{
private:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
    int inputSize;
    int outputSize;
    int tileSize;
public:
    LinearKernel(InferencingContext* context, ActivationFunction activation, int tileSize, int inputSize, int outputSize);
    SlangResult loadParams(TorchParamReader& reader);
    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputVector);
};

class BroadcastAddKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
public:
    BroadcastAddKernel(InferencingContext* context);
    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputA, ArrayView<int> shapeA, rhi::IBuffer* inputB, ArrayView<int> shapeB);
};

class ConcatKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
public:
    ConcatKernel(InferencingContext* context);
    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputA, ArrayView<int> shapeA, rhi::IBuffer* inputB, ArrayView<int> shapeB, int axis);
};

class Conv2DKernel : public RefObject
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

    Conv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int inChannels, int outChannels);

    SlangResult loadParams(TorchParamReader& reader, bool loadAndFuseBNorm);
    SlangResult loadParams(int kernelSize, int outputChannelCount, float* weightsData, float* biasesData);

    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int stride, int padding);
};

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

    TransposedConv2DKernel(InferencingContext* context, int tileSize, int kernelSize, int stride, int inChannels, int outChannels);

    SlangResult loadParams(TorchParamReader& reader);
    SlangResult loadParams(int kernelSize, int outputChannelCount, float* weightsData, float* biasesData);

    ComPtr<rhi::IBuffer> queueExecute(InferencingTask& task, rhi::IBuffer* inputImage, int inputWidth, int inputHeight, int padding);
};
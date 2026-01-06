#pragma once

#include "elementwise.h"
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
    ProgramNode inputProgram;
    ProgramNode outputProgram;
    SinkExpr sinkExpr;

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
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        String name = "transConv2d");

    TransposedConv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr outputExpr = kernelOutput(),
        String name = "transConv2d");

    SlangResult loadParams(TorchParamReader& reader);
    SlangResult loadParams(
        int kernelSize,
        int outputChannelCount,
        float* weightsData,
        float* biasesData);

    TensorView allocateResultBuffer(
        ElementType elementType,
        int inputWidth,
        int inputHeight,
        int padding,
        int batchSize);

    void queueExecute(InferencingTask& task, EvalContext& ctx, TensorView output, int padding);

    void queueExecute(
        InferencingTask& task,
        TensorView outputImage,
        TensorView inputImage,
        int padding);
};
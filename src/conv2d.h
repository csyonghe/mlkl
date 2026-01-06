#pragma once

#include "elementwise.h"
#include "kernel-base.h"


// 2D Convolution Kernel
// - Weights layout: [InChannels, KernelSize, KernelSize, OutChannels]
// - Input/Output layout: [BatchSize, Height, Width, Channels]
class Conv2DKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> tilePipeline;
    ComPtr<rhi::IComputePipeline> flatPipeline;
    ComPtr<rhi::IComputePipeline> flatWaveReducePipeline;

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

    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;

    // [outChannels, kernelSize, kernelSize, inChannels]
    ComPtr<rhi::IBuffer> weightsTransposedBuffer;

    String name;

    Conv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        String name = "conv2d");

    Conv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr outputExpr = kernelOutput(),
        String name = "conv2d");

    SlangResult loadParams(TorchParamReader& reader, bool loadAndFuseBNorm);

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

    void queueExecute(InferencingTask& task, EvalContext& evalCtx, TensorView output, int padding);

    void queueExecute(InferencingTask& task, TensorView output, TensorView inputImage, int padding);
};
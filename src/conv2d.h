#pragma once

#include "elementwise.h"
#include "kernel-base.h"

class SafeTensorsReader;

// 2D Convolution Kernel
// - Weights layout: [InChannels, KernelSize, KernelSize, OutChannels]
// - Input/Output layout: [BatchSize, Height, Width, Channels]
//
// CONSTRUCTORS:
// 1. Simple (Float32, buffer input, custom output):
//    Conv2DKernel(ctx, tileSize, kernelSize, stride, inCh, outCh, outputExpr)
//
// 2. Full (custom input/output expressions):
//    Conv2DKernel(ctx, elemType, tileSize, kernelSize, stride, inCh, outCh,
//                 inputExpr, outputExpr, sinkExpr)
//
// QUEUEEXECUTE:
//   queueExecute(task, output, input, padding)
//   - padding: Use 0 for 1x1 conv, 1 for 3x3 conv (to maintain spatial dims)
//
// FUSION OPPORTUNITIES:
// - Fuse SiLU/ReLU into inputExpr:  Conv2DKernel(..., silu(buffer()), ...)
// - Fuse clamp into outputExpr:     Conv2DKernel(..., clamp(kernelOutput(), -1, 1), ...)
// - Fuse upsample into inputExpr:   Conv2DKernel(..., upsample2x(buffer()), ...)
//
// COMMON MISTAKES:
// - Forgetting padding argument in queueExecute
// - Using wrong constructor (ElementType in wrong position)
class Conv2DKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> tilePipeline;
    ComPtr<rhi::IComputePipeline> flatPipeline;
    ComPtr<rhi::IComputePipeline> flatWaveReducePipeline;

    InferencingContext* context;
    ElementType elementType;
    ProgramNode inputProgram;
    ProgramNode outputProgram;
    SinkExpr sinkExpr;

    void validateTensorElementType(const TensorView& tv, const char* name) const;

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
        ElementType elementType,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        String name = "conv2d");

    // Convenience constructor defaulting to Float32.
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
        String name = "conv2d")
        : Conv2DKernel(
              context,
              ElementType::Float32,
              tileSize,
              kernelSize,
              stride,
              inChannels,
              outChannels,
              inputExpr,
              outputExpr,
              sinkExpr,
              name)
    {
    }

    Conv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr outputExpr = kernelOutput(),
        String name = "conv2d");

    ElementType getElementType() const { return elementType; }

    SlangResult loadParams(TorchParamReader& reader, bool loadAndFuseBNorm);

    // Load from SafeTensors - weights are transposed from [OutCh, InCh, K, K] to [InCh, K, K, OutCh]
    SlangResult loadParams(
        SafeTensorsReader& reader,
        UnownedStringSlice weightName,
        UnownedStringSlice biasName);

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
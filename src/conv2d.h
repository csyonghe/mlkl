#pragma once

#include "elementwise.h"
#include "kernel-base.h"

class SafeTensorsReader;

// 2D Convolution Kernel
// - Weights layout: [InChannels, KernelSize, KernelSize, OutChannels]
// - Input/Output layout: [BatchSize, Height, Width, Channels]
//
// CONSTRUCTORS:
// 1. Simple (Float32, buffer input, custom OUTPUT transformation):
//    Conv2DKernel(ctx, tileSize, kernelSize, stride, inCh, outCh, outputExpr)
//    - outputExpr: transformation applied to conv output (e.g., kernelOutput(), clamp(...))
//    - DO NOT pass buffer() or silu(buffer()) here! Those are INPUT expressions.
//
// 2. Full (custom INPUT/output expressions):
//    Conv2DKernel(ctx, elemType, tileSize, kernelSize, stride, inCh, outCh,
//                 inputExpr, outputExpr, sinkExpr)
//    - inputExpr: transformation applied to input (e.g., silu(buffer()), upsample2x(buffer()))
//    - outputExpr: transformation applied to output (e.g., kernelOutput(), clamp(kernelOutput(), ...))
//
// QUEUEEXECUTE:
//   queueExecute(task, output, input, padding)
//   - padding: Use 0 for 1x1 conv, 1 for 3x3 conv (to maintain spatial dims)
//
// FUSION OPPORTUNITIES:
// - Fuse SiLU/ReLU into inputExpr (FULL constructor):
//     Conv2DKernel(ctx, Float32, 16, 3, 1, inCh, outCh, silu(buffer()), kernelOutput(), bufferSink())
// - Fuse clamp into outputExpr:
//     Conv2DKernel(ctx, Float32, 16, 3, 1, inCh, outCh, buffer(), clamp(kernelOutput(),-1,1), bufferSink())
// - Fuse upsample into inputExpr (FULL constructor):
//     Conv2DKernel(ctx, Float32, 16, 3, 1, inCh, outCh, upsample2x(buffer()), kernelOutput(), bufferSink())
//
// COMMON MISTAKES:
// - Forgetting padding argument in queueExecute
// - Using wrong constructor (passing silu(buffer()) to simple constructor - use FULL constructor!)
// - Confusing inputExpr vs outputExpr parameter position
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

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs,
        int padding)
    {
        EvalContext ctx = makeEvalContext(inputs);
        return queueExecute(task, ctx, output, padding);
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs,
        int padding);

    void queueExecute(InferencingTask& task, TensorView output, TensorView inputImage, int padding);
};
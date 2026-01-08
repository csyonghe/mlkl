#pragma once
#include "elementwise.h"
#include "kernel-base.h"
#include "reduce.h"

class SafeTensorsReader;

// GroupNorm Kernel
// - Input/Output layout: [BatchSize, Height, Width, Channels] (NHWC)
// - Gamma/Beta layout: [Channels]
//
// For each (batch, group), normalizes across (Height, Width, ChannelsPerGroup)
// then applies per-channel scale (gamma) and bias (beta).
//
// CONSTRUCTOR PARAMETER ORDER:
//   GroupNormKernel(ctx, numChannels, numGroups)
//   NOTE: numChannels FIRST, numGroups SECOND (easy to confuse!)
//
// COMMON MISTAKES:
// - Swapping numChannels and numGroups - causes "channels don't match" error
class GroupNormKernel : public RefObject
{
private:
    RefPtr<ReduceKernel> reduceKernel;                // Pass 1: compute sum/sumSq
    ComPtr<rhi::IComputePipeline> normalizePipeline;  // Pass 2: normalize elements
    InferencingContext* context;
    ElementType elementType;

    ProgramNode inputProgram;
    SinkExpr sinkExpr;

    void validateTensorElementType(const TensorView& tv, const char* name) const;

public:
    int numChannels;
    int numGroups;
    float epsilon;

    ComPtr<rhi::IBuffer> gammaBuffer;
    ComPtr<rhi::IBuffer> betaBuffer;

    GroupNormKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr inputExpr,
        SinkExpr sinkExpr,
        int numChannels,
        int numGroups,
        float epsilon = 1e-5f);

    // Convenience constructor defaulting to Float32 with buffer input/output
    GroupNormKernel(
        InferencingContext* ctx,
        int numChannels,
        int numGroups,
        float epsilon = 1e-5f)
        : GroupNormKernel(
              ctx,
              ElementType::Float32,
              buffer(),
              bufferSink(),
              numChannels,
              numGroups,
              epsilon)
    {
    }

    ElementType getElementType() const { return elementType; }

    // Allocate output buffer matching input shape
    TensorView allocateResultBuffer(
        ElementType elementType,
        int batchSize,
        int height,
        int width);

    // Load gamma (scale) and beta (bias) parameters from a reader
    SlangResult loadParams(TorchParamReader& reader);

    // Load from SafeTensors
    SlangResult loadParams(
        SafeTensorsReader& reader,
        UnownedStringSlice gammaName,
        UnownedStringSlice betaName);

    // Execute with full EvalContext
    void queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx);

    // Execute with multiple input tensors (for fused expressions)
    void queueExecute(InferencingTask& task, TensorView output, const std::initializer_list<InputInfo>& inputs);

    // Execute with single input tensor
    void queueExecute(InferencingTask& task, TensorView output, TensorView input);
};

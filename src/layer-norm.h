#pragma once
#include "elementwise.h"
#include "kernel-base.h"
#include "reduce.h"

class SafeTensorsReader;

// LayerNorm Kernel
// - Input/Output layout: [NumRows, NumFeatures] (2D tensor)
// - Gamma/Beta layout: [NumFeatures]
//
// Normalizes across the last dimension (features) for each row,
// then applies per-feature scale (gamma) and bias (beta).
class LayerNormKernel : public RefObject
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
    int numFeatures;
    float epsilon;

    ComPtr<rhi::IBuffer> gammaBuffer;
    ComPtr<rhi::IBuffer> betaBuffer;

    LayerNormKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr inputExpr,
        SinkExpr sinkExpr,
        int numFeatures,
        float epsilon = 1e-5f);

    // Convenience constructor defaulting to Float32 with buffer input/output
    LayerNormKernel(
        InferencingContext* ctx,
        int numFeatures,
        float epsilon = 1e-5f)
        : LayerNormKernel(
              ctx,
              ElementType::Float32,
              buffer(),
              bufferSink(),
              numFeatures,
              epsilon)
    {
    }

    ElementType getElementType() const { return elementType; }

    // Allocate output buffer matching input shape
    TensorView allocateResultBuffer(ElementType elementType, int numRows);

    // Load gamma (scale) and beta (bias) parameters from a reader
    SlangResult loadParams(TorchParamReader& reader);

    // Load from SafeTensors
    SlangResult loadParams(
        SafeTensorsReader& reader,
        UnownedStringSlice gammaName,
        UnownedStringSlice betaName);

    // Execute with full EvalContext
    void queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx);

    // Execute with single input tensor
    void queueExecute(InferencingTask& task, TensorView output, TensorView input);
};


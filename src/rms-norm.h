#pragma once
#include "elementwise.h"
#include "kernel-base.h"
#include "reduce.h"

// RMSNorm Kernel (Root Mean Square Layer Normalization)
// - Input/Output layout: [NumRows, NumFeatures] (2D tensor)
// - Gamma layout: [NumFeatures]
//
// Normalizes by dividing by the RMS of the input (no mean subtraction),
// then applies per-feature scale (gamma). No bias term.
// Used in LLaMA, Gemma, and other modern LLMs.
class RMSNormKernel : public RefObject
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

    RMSNormKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr inputExpr,
        SinkExpr sinkExpr,
        int numFeatures,
        float epsilon = 1e-5f);

    // Convenience constructor defaulting to Float32 with buffer input/output
    RMSNormKernel(
        InferencingContext* ctx,
        int numFeatures,
        float epsilon = 1e-5f)
        : RMSNormKernel(
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

    // Load gamma (scale) parameter from a reader
    SlangResult loadParams(TorchParamReader& reader);

    // Execute with full EvalContext
    void queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx);

    // Execute with single input tensor
    void queueExecute(InferencingTask& task, TensorView output, TensorView input);
};


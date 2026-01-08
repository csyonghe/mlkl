#pragma once

#include "elementwise.h"
#include "kernel-base.h"


// Flash Attention Kernel
// Implements Flash Attention 2 algorithm that computes:
//  Attention(Q, K, V) = mul(softmax(mul(Q,  transpose(K)) / sqrt(d)), V)
//
// Inputs:
// - Q: Query tensor of shape [B, H, Sq, D_head]
// - K: Key tensor of shape [B, H, Skv, D_head]
// - V: Value tensor of shape [B, H, Skv, D_head]
// - outFunc: Output TRANSFORMATION expression (use kernelOutput() for identity)
//            NOTE: This is NOT an input buffer - it transforms the output value.
// - br: Block size in rows (Sq dimension)
// - bc: Block size in columns (Skv dimension)
// - d: Head dimension (D_head)
// Output layout: Planar [B, H, Sq, D_head]
//
// IMPORTANT LIMITATIONS:
// - headDim (d) must be <= 256 due to GPU shared memory limits
// - For larger head dimensions (e.g., single-head attention with 512 dims),
//   use standard attention instead: BatchGemmKernel + SoftmaxKernel
//
// COMMON MISTAKES:
// - Using buffer() for outFunc - use kernelOutput() for no transformation
// - Passing headDim > 256 - will throw exception, use BatchGemm+Softmax instead

class FlashAttentionKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;
    ElementType elementType;

    // Expressions for generalized inputs/outputs
    Expr qExpr;
    Expr kExpr;
    Expr vExpr;
    SinkExpr sinkExpr;
    Expr outFuncExpr;

    // Programs compiled from expressions
    ProgramNode qProgram;
    ProgramNode kProgram;
    ProgramNode vProgram;
    ProgramNode outFuncProgram;

    // Tile Configuration
    int blockSizeR;
    int blockSizeC;
    int headDim;

public:
    FlashAttentionKernel(
        InferencingContext* ctx,
        ElementType elementType,
        Expr q,
        Expr k,
        Expr v,
        Expr outFunc,
        int br = 32,
        int bc = 32,
        int d = 64,
        SinkExpr sinkExpr = bufferSink());

    // Convenience constructor defaulting to Float32.
    FlashAttentionKernel(
        InferencingContext* ctx,
        Expr q,
        Expr k,
        Expr v,
        Expr outFunc,
        int br = 32,
        int bc = 32,
        int d = 64,
        SinkExpr sinkExpr = bufferSink())
        : FlashAttentionKernel(ctx, ElementType::Float32, q, k, v, outFunc, br, bc, d, sinkExpr)
    {
    }

    ElementType getElementType() const { return elementType; }

    TensorView allocateResultBuffer(
        ElementType elementType,
        uint32_t seqLenQ,
        uint32_t numHeads,
        uint32_t batchSize);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        EvalContext& ctx,
        uint32_t seqLenQ,
        uint32_t seqLenKV,
        uint32_t numHeads,
        uint32_t batchSize,
        float scale,
        bool isCausal);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs,
        uint32_t seqLenQ,
        uint32_t seqLenKV,
        uint32_t numHeads,
        uint32_t batchSize,
        float scale,
        bool isCausal)
    {
        EvalContext ctx = makeEvalContext(inputs);
        return queueExecute(task, output, ctx, seqLenQ, seqLenKV, numHeads, batchSize, scale, isCausal);
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs,
        uint32_t seqLenQ,
        uint32_t seqLenKV,
        uint32_t numHeads,
        uint32_t batchSize,
        float scale,
        bool isCausal);

private:
    void validateTensorElementType(const TensorView& tv, const char* name) const;
};

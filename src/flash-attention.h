#pragma once

#include "elementwise.h"
#include "kernel-base.h"


// Flash Attention Kernel
// Implements Flash Attention 2 algorithm that computes:
//  Attention(Q, K, V) = mul(softmax(mul(Q,  transpose(K)) / sqrt(d)), V)
// Inputs:
// - Q: Query tensor of shape [B, H, Sq, D_head]
// - K: Key tensor of shape [B, H, Skv, D_head]
// - V: Value tensor of shape [B, H, Skv, D_head]
// - outFunc: Output function to write results
// - br: Block size in rows (Sq dimension)
// - bc: Block size in columns (Skv dimension)
// - d: Head dimension (D_head)
// Output layout: Planar [B, H, Sq, D_head]

class FlashAttentionKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    InferencingContext* context;

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
        Expr q,
        Expr k,
        Expr v,
        Expr outFunc,
        int br = 32,
        int bc = 32,
        int d = 64,
        SinkExpr sinkExpr = bufferSink());

    TensorView allocateResultBuffer(
        ElementType elementType,
        uint32_t seqLenQ,
        uint32_t numHeads,
        uint32_t batchSize);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs,
        uint32_t seqLenQ,
        uint32_t seqLenKV,
        uint32_t numHeads,
        uint32_t batchSize,
        float scale,
        bool isCausal);
};

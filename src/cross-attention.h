#pragma once
#include "broadcast-add.h"
#include "flash-attention.h"
#include "kernel-base.h"
#include "linear.h"

// Apply linear projection from input latent vector `x` and context embeddings to Q, K, V,
// then apply Flash Attention and output projection, and return `x` + attention output.
//
class CrossAttentionKernel : public RefObject
{
private:
    InferencingContext* context;

    // Linear Projections
    RefPtr<LinearKernel> projQ;
    RefPtr<LinearKernel> projK;
    RefPtr<LinearKernel> projV;
    RefPtr<LinearKernel> projOut;

    int headDim;

    // Flash Attention Core
    RefPtr<FlashAttentionKernel> flashAttn;

    // Expression Handles for Fused Permutation
    Expr exprProjQ_In;
    Expr exprProjK_In;
    Expr exprProjV_In;
    Expr exprProjOut_In;

    Expr exprQ_In;
    Expr exprK_In;
    Expr exprV_In;

public:
    CrossAttentionKernel(InferencingContext* ctx, int channelDim, int contextDim, int headDim);

    SlangResult loadParams(TorchParamReader& reader);

    BufferView allocateResultBuffer(int batchSize, int seqQ, int dim);

    void queueExecute(
        InferencingTask& task,
        BufferView output,
        BufferView inputLatent,
        BufferView contextEmb,
        int batchSize,
        int seqQ,
        int seqKV,
        int numHeads);
};
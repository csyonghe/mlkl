#pragma once
#include "batch-gemm.h"
#include "broadcast-add.h"
#include "kernel-base.h"
#include "linear.h"
#include "softmax.h"

class CrossAttentionKernel : public RefObject
{
private:
    InferencingContext* context;

    // Linear Projections
    RefPtr<LinearKernel> projQ;
    RefPtr<LinearKernel> projK;
    RefPtr<LinearKernel> projV;
    RefPtr<LinearKernel> projOut;

    // Generic Gemm Kernels
    RefPtr<BatchGemmKernel> gemmScores; // Q @ K^T
    RefPtr<BatchGemmKernel> gemmValues; // Probs @ V

    // Expression Handles (to bind inputs at runtime)
    Expr exprQ_In;     // Input A for gemmScores
    Expr exprK_In;     // Input B for gemmScores (Inner buffer of transpose)
    Expr exprProbs_In; // Input A for gemmValues
    Expr exprV_In;     // Input B for gemmValues

    RefPtr<SoftmaxKernel> softmax;
    RefPtr<BroadcastAddKernel> broadcastAdd;

public:
    CrossAttentionKernel(InferencingContext* ctx, int channelDim, int contextDim);

    SlangResult loadParams(TorchParamReader& reader);

    BufferView allocResultBuffer(int batchSize, int seqQ, int dim);

    void queueExecute(
        InferencingTask& task,
        BufferView output,
        BufferView inputLatent,
        BufferView contextEmb,
        int batchSize,
        int seqQ,
        int seqKV,
        int dim,
        int numHeads);
};
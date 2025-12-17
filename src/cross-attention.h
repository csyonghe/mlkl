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

    // Sub-Kernels for Projections
    // These hold the weights (wQ, wK, wV, wOut) internally
    RefPtr<LinearKernel> projQ;
    RefPtr<LinearKernel> projK;
    RefPtr<LinearKernel> projV;
    RefPtr<LinearKernel> projOut;

    // Sub-Kernels for Attention Mechanism
    RefPtr<BatchGemmKernel> batchGemm;
    RefPtr<SoftmaxKernel> softmax;
    RefPtr<BroadcastAddKernel> broadcastAdd;

public:
    CrossAttentionKernel(InferencingContext* ctx);

    // Load weights for all 4 internal linear layers
    SlangResult loadParams(TorchParamReader& reader);

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
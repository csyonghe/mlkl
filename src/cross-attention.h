#pragma once
#if 0
#include "elementwise.h"
#include "kernel-base.h"

// Assumed Dependencies
#include "batch-gemm.h"
#include "linear.h"
#include "softmax.h"

class CrossAttentionKernel : public RefObject
{
private:
    InferencingContext* context;
    
    // -- Weights (Model Parameters) --
    // In a real engine, these would be managed by a Model/Layer class.
    // Here we own them for simplicity.
    rhi::IBuffer* wQ = nullptr;
    rhi::IBuffer* wK = nullptr;
    rhi::IBuffer* wV = nullptr;
    rhi::IBuffer* wOut = nullptr;

    // -- Sub-Kernels --
    RefPtr<LinearKernel> linear;       // Reused for Q, K, V, Out projections
    RefPtr<BatchGemmKernel> batchGemm; // Reused for Q@K and Attn@V
    RefPtr<SoftmaxKernel> softmax;
    
    // -- Intermediate Buffers --
    // We cache these to avoid allocation overhead every frame.
    ComPtr<rhi::IBuffer> bufQ;
    ComPtr<rhi::IBuffer> bufK;
    ComPtr<rhi::IBuffer> bufV;
    ComPtr<rhi::IBuffer> bufScores; // Q @ K^T
    ComPtr<rhi::IBuffer> bufProbs;  // Softmax(Scores)
    ComPtr<rhi::IBuffer> bufAttn;   // Probs @ V

    // Helper to resize buffers lazily
    void ensureBuffer(ComPtr<rhi::IBuffer>& buf, size_t size, const char* name);

public:
    CrossAttentionKernel(InferencingContext* ctx);

    // Set weights for the layer (called during model load)
    void setWeights(rhi::IBuffer* q, rhi::IBuffer* k, rhi::IBuffer* v, rhi::IBuffer* o);

    // Execute the full attention mechanism
    // Input:  [Batch, SeqQ, Dim]
    // Context: [Batch, SeqKV, Dim]
    // Output: [Batch, SeqQ, Dim]
    ComPtr<rhi::IBuffer> queueExecute(
        InferencingTask& task,
        rhi::IBuffer* inputLatent,
        rhi::IBuffer* contextEmb,
        int batchSize,
        int seqQ,
        int seqKV,
        int dim,       // Total Model Dimension (e.g. 768)
        int numHeads   // Number of Heads (e.g. 12)
    );
};
#endif
#if 0
#include "cross-attention.h"

#include <cmath>

CrossAttentionKernel::CrossAttentionKernel(InferencingContext* ctx) 
    : context(ctx)
{
    // Initialize sub-kernels
    linear = new LinearKernel(ctx);
    batchGemm = new BatchGemmKernel(ctx);
    softmax = new SoftmaxKernel(ctx);
}

void CrossAttentionKernel::setWeights(rhi::IBuffer* q, rhi::IBuffer* k, rhi::IBuffer* v, rhi::IBuffer* o)
{
    wQ = q; wK = k; wV = v; wOut = o;
}

void CrossAttentionKernel::ensureBuffer(ComPtr<rhi::IBuffer>& buf, size_t size, const char* name)
{
    if (!buf || buf->getDesc().size < size)
    {
        buf = context->allocScratchBuffer(size, rhi::ResourceState::UnorderedAccess, name);
    }
}

ComPtr<rhi::IBuffer> CrossAttentionKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* inputLatent, // Query Source
    rhi::IBuffer* contextEmb,  // Key/Value Source
    int batchSize,
    int seqQ,
    int seqKV,
    int dim,
    int numHeads)
{
    int headDim = dim / numHeads;
    float scale = 1.0f / sqrtf((float)headDim);

    // 1. Allocate / Resize Intermediate Buffers
    // ---------------------------------------------------------
    // Q, K, V, Attn are all size [Batch, Seq, Dim] (float = 4 bytes)
    size_t sizeQ = (size_t)batchSize * seqQ * dim * sizeof(float);
    size_t sizeKV = (size_t)batchSize * seqKV * dim * sizeof(float);
    
    // Scores/Probs are size [Batch, Heads, SeqQ, SeqKV]
    size_t sizeScores = (size_t)batchSize * numHeads * seqQ * seqKV * sizeof(float);

    ensureBuffer(bufQ, sizeQ, "Attn_Q");
    ensureBuffer(bufK, sizeKV, "Attn_K");
    ensureBuffer(bufV, sizeKV, "Attn_V");
    ensureBuffer(bufScores, sizeScores, "Attn_Scores");
    ensureBuffer(bufProbs, sizeScores, "Attn_Probs");
    ensureBuffer(bufAttn, sizeQ, "Attn_Output");

    auto finalOutput = task.allocateBuffer("CrossAttn_Final", sizeQ);

    // 2. Linear Projections (Input -> Q, Context -> K, V)
    // ---------------------------------------------------------
    // Viewed as: [Batch * Seq, Dim] * [Dim, Dim] -> [Batch * Seq, Dim]
    
    // Q = Input * W_q
    linear->queueExecute(task, inputLatent, wQ, nullptr, bufQ, batchSize * seqQ, dim, dim);
    
    // K = Context * W_k
    linear->queueExecute(task, contextEmb, wK, nullptr, bufK, batchSize * seqKV, dim, dim);
    
    // V = Context * W_v
    linear->queueExecute(task, contextEmb, wV, nullptr, bufV, batchSize * seqKV, dim, dim);

    // 3. Scaled Dot Product Attention: Scores = alpha * (Q @ K^T)
    // ---------------------------------------------------------
    // BatchGemm handles the multi-head view implicitly by strides.
    // Q: [Batch, Heads, SeqQ, HeadDim]
    // K: [Batch, Heads, SeqKV, HeadDim] (Transposed to [HeadDim, SeqKV] logically)
    // Out: [Batch, Heads, SeqQ, SeqKV]
    
    batchGemm->queueExecute(
        task,
        bufQ,       // Matrix A
        bufK,       // Matrix B
        bufScores,  // Matrix C
        
        batchSize * numHeads, // Batch Count for GEMM
        seqQ,                 // M
        seqKV,                // N
        headDim,              // K
        
        scale,      // alpha (Fused 1/sqrt(d) scaling here!)
        0.0f,       // beta
        
        false,      // Transpose A? No
        true        // Transpose B? Yes (K^T)
    );

    // 4. Softmax
    // ---------------------------------------------------------
    // Normalize along the last dimension (SeqKV)
    // Input: [Batch, Heads, SeqQ, SeqKV]
    softmax->queueExecute(task, bufScores, bufProbs, seqKV); 

    // 5. Weighted Sum: Output = Probs @ V
    // ---------------------------------------------------------
    // Probs: [Batch, Heads, SeqQ, SeqKV]
    // V:     [Batch, Heads, SeqKV, HeadDim]
    // Out:   [Batch, Heads, SeqQ, HeadDim]
    
    batchGemm->queueExecute(
        task,
        bufProbs,   // Matrix A
        bufV,       // Matrix B
        bufAttn,    // Matrix C
        
        batchSize * numHeads, // Batch Count
        seqQ,                 // M
        headDim,              // N
        seqKV,                // K
        
        1.0f, 0.0f, // alpha, beta
        false,      // Transpose A? No
        false       // Transpose B? No
    );

    // 6. Output Projection
    // ---------------------------------------------------------
    // View as: [Batch * SeqQ, Dim] * [Dim, Dim]
    // Final = bufAttn * W_out
    
    linear->queueExecute(
        task, 
        bufAttn, 
        wOut, 
        nullptr, 
        finalOutput, 
        batchSize * seqQ, 
        dim, 
        dim
    );

    return ComPtr<rhi::IBuffer>(finalOutput);
}
#endif
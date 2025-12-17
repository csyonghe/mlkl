#include "cross-attention.h"

#include <cmath>

CrossAttentionKernel::CrossAttentionKernel(InferencingContext* ctx, int channelDim, int contextDim)
    : context(ctx)
{
    // 1. Initialize Linear Projections
    projQ = new LinearKernel(ctx, ActivationFunction::None, 256, channelDim, channelDim);
    projK = new LinearKernel(ctx, ActivationFunction::None, 256, contextDim, channelDim);
    projV = new LinearKernel(ctx, ActivationFunction::None, 256, contextDim, channelDim);
    projOut = new LinearKernel(ctx, ActivationFunction::None, 256, channelDim, channelDim);

    // 2. Initialize Generic BatchGemms

    // --- Gemm 1: Scores = Q @ K^T ---
    // A: Q [Batch, SeqQ, Dim]
    exprQ_In = buffer();

    // B: K [Batch, SeqKV, Dim] -> Transposed to [Batch, Dim, SeqKV] logically
    // We use the 'transpose' builder which creates a TransposeNode around a buffer.
    Expr exprK_Buf = buffer();
    exprK_In = exprK_Buf;                               // Store handle to the buffer itself
    Expr exprK_Transposed = transpose(exprK_Buf, 1, 2); // Swap Seq and Dim

    // C: Zero (No bias)
    Expr exprBias1 = constant(0.0f);

    // Output: Standard
    Expr exprOut1 = kernelOutput();

    gemmScores = new BatchGemmKernel(ctx, exprQ_In, exprK_Transposed, exprBias1, exprOut1);

    // --- Gemm 2: Values = Probs @ V ---
    // A: Probs [Batch, SeqQ, SeqKV]
    exprProbs_In = buffer();

    // B: V [Batch, SeqKV, Dim]
    exprV_In = buffer();

    // C: Zero
    Expr exprBias2 = constant(0.0f);
    Expr exprOut2 = kernelOutput();

    gemmValues = new BatchGemmKernel(ctx, exprProbs_In, exprV_In, exprBias2, exprOut2);

    // 3. Other Kernels
    softmax = new SoftmaxKernel(ctx);
    broadcastAdd = new BroadcastAddKernel(ctx);
}

SlangResult CrossAttentionKernel::loadParams(TorchParamReader& reader)
{
    // Load weights.
    // Q, K, V: No Bias
    // Out: Has Bias
    SLANG_RETURN_ON_FAIL(projQ->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projK->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projV->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projOut->loadParams(reader, true));
    return SLANG_OK;
}

BufferView CrossAttentionKernel::allocResultBuffer(int batchSize, int seqQ, int dim)
{
    size_t size = (size_t)batchSize * seqQ * dim * sizeof(float);
    return context->allocScratchBuffer(size, "CrossAttn_Final");
}

void CrossAttentionKernel::queueExecute(
    InferencingTask& task,
    BufferView finalOutput,
    BufferView inputLatent,
    BufferView contextEmb,
    int batchSize,
    int seqQ,
    int seqKV,
    int dim,
    int numHeads)
{
    int headDim = dim / numHeads;
    float scale = 1.0f / sqrtf((float)headDim);

    // 1. Projections
    // Q = inputLatent @ W_q
    BufferView bufQ = projQ->allocateResultBuffer(batchSize * seqQ);
    projQ->queueExecute(task, bufQ, inputLatent, batchSize * seqQ);

    // K = contextEmb @ W_k
    BufferView bufK = projK->allocateResultBuffer(batchSize * seqKV);
    projK->queueExecute(task, bufK, contextEmb, batchSize * seqKV);

    // V = contextEmb @ W_v
    BufferView bufV = projV->allocateResultBuffer(batchSize * seqKV);
    projV->queueExecute(task, bufV, contextEmb, batchSize * seqKV);

    // 2. Attention Scores: Q @ K^T
    // Q: [Batch*Heads, SeqQ, HeadDim]
    // K: [Batch*Heads, SeqKV, HeadDim] (Physical)
    //    We treat K as B in gemm.
    //    Logic: A[M, K] x B[K, N].
    //    Here M=SeqQ, N=SeqKV, K=HeadDim.
    //    Q is [SeqQ, HeadDim]. OK.
    //    K is [SeqKV, HeadDim].
    //    We want Q x K^T.
    //    In the kernel def, we wrapped K in `transpose(..., 1, 2)`.
    //    So we pass K physical shape [Batch, SeqKV, HeadDim] to the input info.

    // Alloc Output: [Batch*Heads, SeqQ, SeqKV]
    size_t sizeScores = (size_t)batchSize * numHeads * seqQ * seqKV * sizeof(float);
    BufferView bufScores = context->allocScratchBuffer(sizeScores, "Attn_Scores");

    Dictionary<Expr, InputInfo> scoresInputs;
    // Input A: Q [Batch*Heads, SeqQ, HeadDim]
    scoresInputs.add(exprQ_In, InputInfo(Shape{batchSize * numHeads, seqQ, headDim}, bufQ));
    // Input B: K [Batch*Heads, SeqKV, HeadDim]
    scoresInputs.add(exprK_In, InputInfo(Shape{batchSize * numHeads, seqKV, headDim}, bufK));

    // Execute Gemm 1
    gemmScores->queueExecute(
        task,
        bufScores,
        seqQ,
        seqKV,
        headDim,              // M, N, K
        batchSize * numHeads, // Batch
        scale,
        0.0f, // Alpha, Beta
        scoresInputs);

    // 3. Softmax
    // Normalize over dim 1 (SeqKV)
    // Flatten batch for kernel: [Batch*Heads*SeqQ, SeqKV]
    BufferView bufProbs = context->allocScratchBuffer(sizeScores, "Attn_Probs");
    softmax->queueExecute(task, bufProbs, bufScores, batchSize * numHeads * seqQ, seqKV);

    // 4. Weighted Sum: Probs @ V
    // Probs: [Batch*Heads, SeqQ, SeqKV]
    // V:     [Batch*Heads, SeqKV, HeadDim]
    // Out:   [Batch*Heads, SeqQ, HeadDim]
    // M=SeqQ, N=HeadDim, K=SeqKV

    Dictionary<Expr, InputInfo> valuesInputs;
    valuesInputs.add(exprProbs_In, InputInfo(Shape{batchSize * numHeads, seqQ, seqKV}, bufProbs));
    valuesInputs.add(exprV_In, InputInfo(Shape{batchSize * numHeads, seqKV, headDim}, bufV));

    size_t sizeAttnOut = (size_t)batchSize * numHeads * seqQ * headDim * sizeof(float);
    BufferView bufAttnOut = context->allocScratchBuffer(sizeAttnOut, "Attn_Out");

    // Execute Gemm 2
    gemmValues->queueExecute(
        task,
        bufAttnOut,
        seqQ,
        headDim,
        seqKV, // M, N, K
        batchSize * numHeads,
        1.0f,
        0.0f,
        valuesInputs);

    // 5. Output Projection
    BufferView bufProjected = projOut->allocateResultBuffer(batchSize * seqQ);
    projOut->queueExecute(task, bufProjected, bufAttnOut, batchSize * seqQ);

    // 6. Residual
    Shape shape = {(int)batchSize, (int)seqQ, (int)dim};
    broadcastAdd->queueExecute(task, finalOutput, inputLatent, shape, bufProjected, shape);
}
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

    // A: Q
    // Physical: [Batch, SeqQ, Heads, HeadDim] (0, 1, 2, 3)
    // Logical:  [Batch, Heads, SeqQ, HeadDim] (0, 2, 1, 3)
    Expr exprQ_Buf = buffer();
    exprQ_In = exprQ_Buf;
    Expr exprQ_Permuted = permute(exprQ_Buf, {0, 2, 1, 3});

    // B: K^T
    // Physical: [Batch, SeqKV, Heads, HeadDim] (0, 1, 2, 3)
    // Logical K:   [Batch, Heads, SeqKV, HeadDim] (0, 2, 1, 3)
    // Logical K^T: [Batch, Heads, HeadDim, SeqKV] (0, 2, 3, 1)
    Expr exprK_Buf = buffer();
    exprK_In = exprK_Buf;
    Expr exprK_Permuted = permute(exprK_Buf, {0, 2, 3, 1});

    // C: Zero (No bias)
    Expr exprBias1 = constant(0.0f);
    Expr exprOut1 = kernelOutput();

    gemmScores = new BatchGemmKernel(ctx, exprQ_Permuted, exprK_Permuted, exprBias1, exprOut1);

    // --- Gemm 2: Values = Probs @ V ---

    // A: Probs
    // Physical/Logical: [Batch, Heads, SeqQ, SeqKV] (Produced by Gemm1+Softmax, already Planar)
    exprProbs_In = buffer();

    // B: V
    // Physical: [Batch, SeqKV, Heads, HeadDim] (0, 1, 2, 3)
    // Logical:  [Batch, Heads, SeqKV, HeadDim] (0, 2, 1, 3)
    Expr exprV_Buf = buffer();
    exprV_In = exprV_Buf;
    Expr exprV_Permuted = permute(exprV_Buf, {0, 2, 1, 3});

    // C: Zero
    Expr exprBias2 = constant(0.0f);
    Expr exprOut2 = kernelOutput();

    gemmValues = new BatchGemmKernel(ctx, exprProbs_In, exprV_Permuted, exprBias2, exprOut2);

    // 3. Other Kernels
    softmax = new SoftmaxKernel(ctx);
    broadcastAdd = new BroadcastAddKernel(ctx);
    permuteKernel = new PermuteKernel(ctx, {0, 2, 1, 3});
}

SlangResult CrossAttentionKernel::loadParams(TorchParamReader& reader)
{
    logInfo("Loading cross attention weights...\n");
    SLANG_RETURN_ON_FAIL(projQ->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projK->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projV->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projOut->loadParams(reader, true));
    logInfo("Finished cross attention weights.\n");
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

    // ... (Projections, GemmScores, Softmax, GemmValues same as before) ...

    // 1. Projections
    BufferView bufQ = projQ->allocateResultBuffer(batchSize * seqQ);
    projQ->queueExecute(task, bufQ, inputLatent, batchSize * seqQ);

    BufferView bufK = projK->allocateResultBuffer(batchSize * seqKV);
    projK->queueExecute(task, bufK, contextEmb, batchSize * seqKV);

    BufferView bufV = projV->allocateResultBuffer(batchSize * seqKV);
    projV->queueExecute(task, bufV, contextEmb, batchSize * seqKV);

    // 2. Scores
    size_t sizeScores = (size_t)batchSize * numHeads * seqQ * seqKV * sizeof(float);
    BufferView bufScores = context->allocScratchBuffer(sizeScores, "Attn_Scores");

    Dictionary<Expr, InputInfo> scoresInputs;
    scoresInputs.add(exprQ_In, InputInfo(Shape{batchSize, seqQ, numHeads, headDim}, bufQ));
    scoresInputs.add(exprK_In, InputInfo(Shape{batchSize, seqKV, numHeads, headDim}, bufK));

    gemmScores->queueExecute(
        task,
        bufScores,
        seqQ,
        seqKV,
        headDim,
        batchSize * numHeads,
        scale,
        0.0f,
        scoresInputs);

    // 3. Softmax
    BufferView bufProbs = context->allocScratchBuffer(sizeScores, "Attn_Probs");
    softmax->queueExecute(task, bufProbs, bufScores, batchSize * numHeads * seqQ, seqKV);

    // 4. Values
    size_t sizeAttnOut = (size_t)batchSize * numHeads * seqQ * headDim * sizeof(float);
    BufferView bufAttnOut = context->allocScratchBuffer(sizeAttnOut, "Attn_Out");

    Dictionary<Expr, InputInfo> valuesInputs;
    valuesInputs.add(exprProbs_In, InputInfo(Shape{batchSize, numHeads, seqQ, seqKV}, bufProbs));
    valuesInputs.add(exprV_In, InputInfo(Shape{batchSize, seqKV, numHeads, headDim}, bufV));

    gemmValues->queueExecute(
        task,
        bufAttnOut,
        seqQ,
        headDim,
        seqKV,
        batchSize * numHeads,
        1.0f,
        0.0f,
        valuesInputs);

    // 5. Output Permutation & Projection
    // bufAttnOut is Planar: [Batch, Heads, SeqQ, HeadDim]
    // We permute to Interleaved: [Batch, SeqQ, Heads, HeadDim] for Linear layer

    BufferView bufAttnInterleaved = context->allocScratchBuffer(sizeAttnOut, "Attn_Interleaved");

    permuteKernel->queueExecute(
        task,
        bufAttnInterleaved,
        bufAttnOut,
        Shape{batchSize, numHeads, seqQ, headDim} // Input Shape
    );

    BufferView bufProjected = projOut->allocateResultBuffer(batchSize * seqQ);
    projOut->queueExecute(task, bufProjected, bufAttnInterleaved, batchSize * seqQ);

    // 6. Residual
    Shape shape = {(int)batchSize, (int)seqQ, (int)dim};
    broadcastAdd->queueExecute(task, finalOutput, inputLatent, shape, bufProjected, shape);
}
#include "cross-attention.h"

#include <cmath>

CrossAttentionKernel::CrossAttentionKernel(InferencingContext* ctx)
    : context(ctx)
{
    // Initialize kernels
    // Note: Dimensions (input/output size) here are placeholders if your
    // LinearKernel::loadParams overwrites them based on the file content.
    // If not, you should pass the correct dims here or resize in loadParams.
    // Assuming ActivationFunction::None for all projections.
    projQ = new LinearKernel(ctx, ActivationFunction::None, 256, 0, 0);
    projK = new LinearKernel(ctx, ActivationFunction::None, 256, 0, 0);
    projV = new LinearKernel(ctx, ActivationFunction::None, 256, 0, 0);
    projOut = new LinearKernel(ctx, ActivationFunction::None, 256, 0, 0);

    batchGemm = new BatchGemmKernel(ctx);
    softmax = new SoftmaxKernel(ctx);
    broadcastAdd = new BroadcastAddKernel(ctx);
}

SlangResult CrossAttentionKernel::loadParams(TorchParamReader& reader)
{
    // The order must match your Python dump script:
    // 1. to_q
    SLANG_RETURN_ON_FAIL(projQ->loadParams(reader, false));
    // 2. to_k
    SLANG_RETURN_ON_FAIL(projK->loadParams(reader, false));
    // 3. to_v
    SLANG_RETURN_ON_FAIL(projV->loadParams(reader, false));
    // 4. to_out
    SLANG_RETURN_ON_FAIL(projOut->loadParams(reader, false));

    return SLANG_OK;
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
    // The LinearKernel computes Output = Input @ W^T + Bias

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
    // K: [Batch*Heads, SeqKV, HeadDim] -> Transposed to [HeadDim, SeqKV]
    BufferView bufScores = batchGemm->allocResultBuffer(batchSize * numHeads, seqQ, seqKV);

    batchGemm->queueExecute(
        task,
        bufScores,
        bufQ,
        bufK,
        batchSize * numHeads,
        seqQ,
        seqKV,
        headDim,
        scale,
        0.0f,
        false,
        true // Transpose B (K)
    );

    // 3. Softmax
    BufferView bufProbs = softmax->allocResultBuffer(batchSize * numHeads * seqQ, seqKV);
    softmax->queueExecute(task, bufProbs, bufScores, batchSize * numHeads * seqQ, seqKV);

    // 4. Weighted Sum: Probs @ V
    BufferView bufAttnOut = batchGemm->allocResultBuffer(batchSize * numHeads, seqQ, headDim);

    batchGemm->queueExecute(
        task,
        bufAttnOut,
        bufProbs,
        bufV,
        batchSize * numHeads,
        seqQ,
        headDim,
        seqKV,
        1.0f,
        0.0f,
        false,
        false);

    // 5. Output Projection
    // We project the result of attention back to original dim
    BufferView bufProjected = projOut->allocateResultBuffer(batchSize * seqQ);
    projOut->queueExecute(task, bufProjected, bufAttnOut, batchSize * seqQ);

    // 6. Residual Connection: Output = inputLatent + bufProjected
    Shape shape = {(int)batchSize, (int)seqQ, (int)dim};
    broadcastAdd->queueExecute(task, finalOutput, inputLatent, shape, bufProjected, shape);
}
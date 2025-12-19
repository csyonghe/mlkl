#include "cross-attention.h"

#include <cmath>

CrossAttentionKernel::CrossAttentionKernel(
    InferencingContext* ctx,
    int channelDim,
    int contextDim,
    int headDim)
    : context(ctx), headDim(headDim)
{
    // 1. Initialize Linear Projections
    projQ = new LinearKernel(ctx, channelDim, channelDim);
    projK = new LinearKernel(ctx, contextDim, channelDim);
    projV = new LinearKernel(ctx, contextDim, channelDim);
    // Fuse residual add into output projection.
    projOut = new LinearKernel(
        ctx,
        buffer(),
        kernelOutput() + buffer(),
        bufferSink(),
        channelDim,
        channelDim);

    // 2. Setup Flash Attention with Fused Permutation
    // We expect input to be [B, S, H, D] (Interleaved)
    // Flash expects [B, H, S, D] (Planar)
    // Permutation {0, 2, 1, 3} maps Interleaved -> Planar

    exprQ_In = buffer();
    Expr exprQ_Planar = permute(exprQ_In, {0, 2, 1, 3});

    exprK_In = buffer();
    Expr exprK_Planar = permute(exprK_In, {0, 2, 1, 3});

    exprV_In = buffer();
    Expr exprV_Planar = permute(exprV_In, {0, 2, 1, 3});

    Expr eOutCore = kernelOutput();

    // Output of Flash is Planar [B, H, S, D]
    // We need Interleaved [B, S, H, D] for the final projOut Linear layer.
    // We can use sinkExpr to fuse a Planar -> Interleaved permutation on the write-back!
    SinkExpr eSink = permute(bufferSink(), {0, 2, 1, 3});

    // Tile sizes 32x32 are standard for modern GPUs
    // headDim is typically 32, 64, or 128.
    flashAttn = new FlashAttentionKernel(
        ctx,
        exprQ_Planar,
        exprK_Planar,
        exprV_Planar,
        eOutCore,
        32,
        32,
        headDim,
        eSink);
}

SlangResult CrossAttentionKernel::loadParams(TorchParamReader& reader)
{
    SLANG_RETURN_ON_FAIL(projQ->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projK->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projV->loadParams(reader, false));
    SLANG_RETURN_ON_FAIL(projOut->loadParams(reader, true));
    return SLANG_OK;
}

BufferView CrossAttentionKernel::allocateResultBuffer(int batchSize, int seqQ, int dim)
{
    return context->allocScratchBuffer(
        (size_t)batchSize * seqQ * dim * sizeof(float),
        "CrossAttn_Final");
}

void CrossAttentionKernel::queueExecute(
    InferencingTask& task,
    BufferView finalOutput,
    BufferView inputLatent,
    BufferView contextEmb,
    int batchSize,
    int seqQ,
    int seqKV,
    int numHeads)
{
    task.context->pushAllocScope();
    SLANG_DEFER(task.context->popAllocScope());

    int dim = numHeads * headDim;
    float scale = 1.0f / sqrtf((float)headDim);

    // 1. Projections (Remain Interleaved: [B, S, H, D])
    BufferView bufQ = projQ->allocateResultBuffer(batchSize * seqQ);
    projQ->queueExecute(task, bufQ, inputLatent, batchSize * seqQ);

    BufferView bufK = projK->allocateResultBuffer(batchSize * seqKV);
    projK->queueExecute(task, bufK, contextEmb, batchSize * seqKV);

    BufferView bufV = projV->allocateResultBuffer(batchSize * seqKV);
    projV->queueExecute(task, bufV, contextEmb, batchSize * seqKV);

    // 2. Flash Attention Core
    size_t sizeAttn = (size_t)batchSize * numHeads * seqQ * headDim * sizeof(float);
    BufferView bufAttnInterleaved = context->allocScratchBuffer(sizeAttn, "Attn_Interleaved");

    Dictionary<Expr, InputInfo> attnInputs;
    attnInputs.add(exprQ_In, InputInfo(Shape{batchSize, seqQ, numHeads, headDim}, bufQ));
    attnInputs.add(exprK_In, InputInfo(Shape{batchSize, seqKV, numHeads, headDim}, bufK));
    attnInputs.add(exprV_In, InputInfo(Shape{batchSize, seqKV, numHeads, headDim}, bufV));

    flashAttn->queueExecute(
        task,
        bufAttnInterleaved,
        attnInputs,
        seqQ,
        seqKV,
        numHeads,
        batchSize,
        scale,
        false // Not causal for cross-attention
    );

    // 3. Final Projection and Residual Connection
    projOut->queueExecute(
        task,
        finalOutput,
        batchSize * seqQ,
        {{bufAttnInterleaved, batchSize, seqQ}, {inputLatent, batchSize, seqQ}});
}
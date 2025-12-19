#include "flash-attention.h"

namespace Slang
{

FlashAttentionKernel::FlashAttentionKernel(
    InferencingContext* ctx,
    Expr q,
    Expr k,
    Expr v,
    Expr outFunc,
    int br,
    int bc,
    int d,
    SinkExpr sinkExpr)
    : context(ctx)
    , qExpr(q)
    , kExpr(k)
    , vExpr(v)
    , outFuncExpr(outFunc)
    , blockSizeR(br)
    , blockSizeC(bc)
    , headDim(d)
    , sinkExpr(sinkExpr)
{
    // 1. Compile expressions into linear programs
    int globalRegCounter = 0;
    qProgram = compileExprToProgram(qExpr, &globalRegCounter);
    kProgram = compileExprToProgram(kExpr, &globalRegCounter);
    vProgram = compileExprToProgram(vExpr, &globalRegCounter);
    outFuncProgram = compileExprToProgram(outFuncExpr, &globalRegCounter);

    // 2. Prepare specialization arguments
    // Order: blockSizeR, blockSizeC, headDimension, TQ, TK, TV, FOut
    List<String> typeArgs;
    typeArgs.add(String(blockSizeR));
    typeArgs.add(String(blockSizeC));
    typeArgs.add(String(headDim));
    typeArgs.add(qProgram.getSlangTypeName());
    typeArgs.add(kProgram.getSlangTypeName());
    typeArgs.add(vProgram.getSlangTypeName());
    typeArgs.add(sinkExpr.node->getSlangTypeName());
    typeArgs.add(outFuncProgram.getSlangTypeName());

    // 3. Create pipeline
    pipeline = ctx->createComputePipeline("flashAttention2", typeArgs.getArrayView());
}

BufferView FlashAttentionKernel::allocateResultBuffer(
    uint32_t seqLenQ,
    uint32_t numHeads,
    uint32_t batchSize)
{
    // Output tensor shape is [Batch, Heads, SeqLenQ, HeadDim]
    // The Slang kernel writes to indices calculated as:
    // (batch_idx * batch_stride) + (head_idx * head_stride) + (row_idx * headDim) + d
    size_t elementCount = (size_t)batchSize * numHeads * seqLenQ * headDim;
    return context->allocScratchBuffer(elementCount * sizeof(float), "flash_attention_output");
}

void FlashAttentionKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    const Dictionary<Expr, InputInfo>& inputs,
    uint32_t seqLenQ,
    uint32_t seqLenKV,
    uint32_t numHeads,
    uint32_t batchSize,
    float scale,
    bool isCausal)
{
    EvalContext evalCtx;
    for (auto it : inputs)
        evalCtx.inputs.add(it.first.node, it.second);

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack according to FlashAttentionParams struct layout (ScalarDataLayout)
    // 1. Expressions (TQ, TK, TV, FOut)
    qProgram.pack(writer, evalCtx);
    kProgram.pack(writer, evalCtx);
    vProgram.pack(writer, evalCtx);
    outFuncProgram.pack(writer, evalCtx);

    SinkExprEvalContext sinkEvalCtx;
    sinkEvalCtx.outputBuffer = output;
    sinkEvalCtx.logicalShape = Shape{(int)batchSize, (int)numHeads, (int)seqLenQ, (int)headDim};

    sinkExpr.node->pack(writer, sinkEvalCtx);

    // 2. Fixed members
    writer.write(seqLenQ);
    writer.write(seqLenKV);
    writer.write(numHeads);
    writer.write(scale);

    // bool is declared as uint(4 bytes) in Slang cbuffer
    uint32_t causalVal = isCausal ? 1 : 0;
    writer.write(causalVal);

    writer.finish();

    // Dispatch:
    // X = Row blocks
    // Y = Heads
    // Z = Batch
    uint32_t gridX = (seqLenQ + blockSizeR - 1) / blockSizeR;
    uint32_t gridY = numHeads;
    uint32_t gridZ = batchSize;

    task.dispatchKernel(
        pipeline,
        gridX,
        gridY,
        gridZ,
        paramData.getBuffer(),
        (uint32_t)paramData.getCount());
}

} // namespace Slang
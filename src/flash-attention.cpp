#include "flash-attention.h"

using namespace Slang;

FlashAttentionKernel::FlashAttentionKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr q,
    Expr k,
    Expr v,
    Expr outFunc,
    int br,
    int bc,
    int d,
    SinkExpr sinkExpr)
    : context(ctx)
    , elementType(elementType)
    , qExpr(q)
    , kExpr(k)
    , vExpr(v)
    , outFuncExpr(outFunc)
    , blockSizeR(br)
    , blockSizeC(bc)
    , headDim(d)
    , sinkExpr(sinkExpr)
{
    // Validate headDim to prevent shared memory overflow
    // FlashAttention uses shared memory proportional to blockSize * headDim
    constexpr int kMaxHeadDim = 256;
    if (d > kMaxHeadDim)
    {
        throw InvalidOperationException(
            String("FlashAttentionKernel: headDim (") + String(d) + 
            ") exceeds maximum supported value (" + String(kMaxHeadDim) + 
            "). Use standard attention (BatchGemm + Softmax) for larger head dimensions.");
    }
    
    // 1. Compile expressions into linear programs
    int globalRegCounter = 0;
    qProgram = compileExprToProgram(qExpr, &globalRegCounter);
    kProgram = compileExprToProgram(kExpr, &globalRegCounter);
    vProgram = compileExprToProgram(vExpr, &globalRegCounter);
    outFuncProgram = compileExprToProgram(outFuncExpr, &globalRegCounter);

    // 2. Prepare specialization arguments
    // Order: blockSizeR, blockSizeC, headDimension, T, TQ, TK, TV, TSink, FOut
    String elemTypeName = getSlangElementTypeName(elementType);
    List<String> typeArgs;
    typeArgs.add(String(blockSizeR));
    typeArgs.add(String(blockSizeC));
    typeArgs.add(String(headDim));
    typeArgs.add(elemTypeName);
    typeArgs.add(qProgram.getSlangTypeName(elementType));
    typeArgs.add(kProgram.getSlangTypeName(elementType));
    typeArgs.add(vProgram.getSlangTypeName(elementType));
    typeArgs.add(sinkExpr.node->getSlangTypeName(elementType));
    typeArgs.add(outFuncProgram.getSlangTypeName(elementType));

    // 3. Create pipeline
    pipeline = ctx->createComputePipeline("flashAttention2", typeArgs.getArrayView());
}

void FlashAttentionKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("FlashAttentionKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

TensorView FlashAttentionKernel::allocateResultBuffer(
    ElementType elementType,
    uint32_t seqLenQ,
    uint32_t numHeads,
    uint32_t batchSize)
{
    // Output tensor shape is [Batch, Heads, SeqLenQ, HeadDim]
    // The Slang kernel writes to indices calculated as:
    // (batch_idx * batch_stride) + (head_idx * head_stride) + (row_idx * headDim) + d
    size_t elementCount = (size_t)batchSize * numHeads * seqLenQ * headDim;
    return context->allocScratchTensor(
        elementType,
        Shape(batchSize, numHeads, seqLenQ, headDim),
        "flash_attention_output");
}

void FlashAttentionKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const std::initializer_list<InputInfo>& inputs,
    uint32_t seqLenQ,
    uint32_t seqLenKV,
    uint32_t numHeads,
    uint32_t batchSize,
    float scale,
    bool isCausal)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("FlashAttentionKernel: insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : qProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : kProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : vProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : outFuncProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    queueExecute(task, output, ctx, seqLenQ, seqLenKV, numHeads, batchSize, scale, isCausal);
}

void FlashAttentionKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    EvalContext& evalCtx,
    uint32_t seqLenQ,
    uint32_t seqLenKV,
    uint32_t numHeads,
    uint32_t batchSize,
    float scale,
    bool isCausal)
{
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto it : evalCtx.inputs)
    {
        validateTensorElementType(it.second.tensorView, "input");
    }

    // Validate that Q, K, V expressions resolve to expected shapes [B, H, S, D]
    // after any permutations are applied. This catches shape format mismatches.
    Shape expectedQ{(int)batchSize, (int)numHeads, (int)seqLenQ, headDim};
    Shape expectedKV{(int)batchSize, (int)numHeads, (int)seqLenKV, headDim};

    Shape actualQ = qExpr.node->resolveShape(evalCtx);
    Shape actualK = kExpr.node->resolveShape(evalCtx);
    Shape actualV = vExpr.node->resolveShape(evalCtx);

    if (actualQ != expectedQ)
    {
        StringBuilder sb;
        sb << "FlashAttention: Q shape mismatch. Expected [B=" << batchSize << ", H=" << numHeads
           << ", S=" << seqLenQ << ", D=" << headDim << "] but got [" << actualQ[0] << ", "
           << actualQ[1] << ", " << actualQ[2] << ", " << actualQ[3]
           << "]. Check that input is in the correct format.";
        throw InvalidOperationException(sb.toString());
    }
    if (actualK != expectedKV)
    {
        StringBuilder sb;
        sb << "FlashAttention: K shape mismatch. Expected [B=" << batchSize << ", H=" << numHeads
           << ", S=" << seqLenKV << ", D=" << headDim << "] but got [" << actualK[0] << ", "
           << actualK[1] << ", " << actualK[2] << ", " << actualK[3]
           << "]. Check that input is in the correct format.";
        throw InvalidOperationException(sb.toString());
    }
    if (actualV != expectedKV)
    {
        StringBuilder sb;
        sb << "FlashAttention: V shape mismatch. Expected [B=" << batchSize << ", H=" << numHeads
           << ", S=" << seqLenKV << ", D=" << headDim << "] but got [" << actualV[0] << ", "
           << actualV[1] << ", " << actualV[2] << ", " << actualV[3]
           << "]. Check that input is in the correct format.";
        throw InvalidOperationException(sb.toString());
    }

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
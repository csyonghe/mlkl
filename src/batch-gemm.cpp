#include "batch-gemm.h"

BatchGemmKernel::BatchGemmKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr A,
    Expr B,
    Expr C,
    SinkExpr sinkExpr,
    Expr Out)
    : context(ctx), elementType(elementType), sinkExpr(sinkExpr)
{
    int globalRegCounter = 0;

    // Compile all four expressions
    programA = compileExprToProgram(A, &globalRegCounter);
    programB = compileExprToProgram(B, &globalRegCounter);
    programC = compileExprToProgram(C, &globalRegCounter);
    programOut = compileExprToProgram(Out, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);
    String typeArgs[] = {
        elemTypeName,
        programA.getSlangTypeName(elementType),
        programB.getSlangTypeName(elementType),
        programC.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType),
        programOut.getSlangTypeName(elementType)};
    pipeline = ctx->createComputePipeline("batchGemm", makeArrayView(typeArgs));
}

void BatchGemmKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("BatchGemmKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

// Allocate output buffer C [BatchSize, M, N]
TensorView BatchGemmKernel::allocateResultBuffer(
    ElementType elementType,
    int batchSize,
    int m,
    int n)
{
    return context->allocScratchTensor(elementType, {batchSize, m, n}, "BatchGemm_Result");
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    float alpha,
    float beta,
    EvalContext& ctx)
{
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto bufferNode : programA.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input A");
    }
    for (auto bufferNode : programB.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input B");
    }
    for (auto bufferNode : programC.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input C");
    }

    auto shapeA = this->programA.resolveShape(ctx);
    auto shapeB = this->programB.resolveShape(ctx);
    auto shapeC = this->programC.resolveShape(ctx);
    if (shapeA.getRank() != 3)
        throw std::runtime_error("Input A to BatchGemm must be 3 (B, M, K).");
    if (shapeB.getRank() != 3)
        throw std::runtime_error("Input B to BatchGemm must be 3 (B, K, N).");
    int M = shapeA[1];
    int K = shapeA[2];
    int N = shapeB[2];
    if (shapeB[1] != K)
        throw std::runtime_error("Input B to BatchGemm inner dimension does not match A.");
    if (shapeB[0] != shapeA[0])
        throw std::runtime_error("Input B to BatchGemm batch dimension does not match A.");
    if (!shapeC.isCompatibleWith(Shape(shapeA[0], M, N)))
    {
        throw std::runtime_error("Input C to BatchGemm shape does not match A and B.");
    }
    int batchSize = shapeA[0];

    SinkExprEvalContext sinkExprCtx;
    sinkExprCtx.outputBuffer = output;
    sinkExprCtx.logicalShape = Shape(batchSize, M, N);
    sinkExprCtx.inputs = ctx.inputs; // Pass inputs for uniformConstant resolution in sink

    // Create context for output expression with kernel output shape
    // This is needed for broadcast(x, kernelOutput()) to resolve correctly
    EvalContext outCtx = ctx;
    outCtx.kernelOutputShape = Shape(batchSize, M, N);

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack in order: A, B, C, FOut, M, N, K, alpha, beta, output
    programA.pack(writer, ctx);
    programB.pack(writer, ctx);
    programC.pack(writer, ctx);
    sinkExpr.node->pack(writer, sinkExprCtx);
    programOut.pack(writer, outCtx);
    writer.write<uint32_t>((uint32_t)M);
    writer.write<uint32_t>((uint32_t)N);
    writer.write<uint32_t>((uint32_t)K);
    writer.write(alpha);
    writer.write(beta);

    writer.finish();

    uint32_t groupX = (N + 15) / 16;
    uint32_t groupY = (M + 15) / 16;

    task.dispatchKernel(
        pipeline,
        groupX,
        groupY,
        batchSize,
        paramData.getBuffer(),
        (uint32_t)paramData.getCount());
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    float alpha,
    float beta,
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
        ctx.inputs.add(it.first.node, it.second);

    queueExecute(task, output, alpha, beta, ctx);
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    float alpha,
    float beta,
    ArrayView<InputInfo> inputs)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : programA.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : programB.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : programC.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    queueExecute(task, output, alpha, beta, ctx);
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    float alpha,
    float beta,
    const std::initializer_list<InputInfo>& inputs)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : programA.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : programB.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : programC.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    queueExecute(task, output, alpha, beta, ctx);
}
void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    float alpha,
    float beta,
    TensorView A,
    TensorView B,
    TensorView C)
{
    EvalContext ctx;
    if (programA.bufferNodes.getCount() > 1)
        throw std::runtime_error("The batch gemm expects more than one buffer for `A`.");
    if (programA.bufferNodes.getCount() < 1)
        throw std::runtime_error("The batch gemm does not need a buffer for `A`.");
    ctx.inputs.add(programA.bufferNodes[0], InputInfo(A));

    if (programB.bufferNodes.getCount() > 1)
        throw std::runtime_error("The batch gemm expects more than one buffer for `B`.");
    if (programB.bufferNodes.getCount() < 1)
        throw std::runtime_error("The batch gemm does not need a buffer for `B`.");
    ctx.inputs.add(programB.bufferNodes[0], InputInfo(B));

    if (programC.bufferNodes.getCount() > 1)
        throw std::runtime_error("The batch gemm expects more than one buffer for `C`.");
    if (programC.bufferNodes.getCount() < 1)
        throw std::runtime_error("The batch gemm does not need a buffer for `C`.");
    ctx.inputs.add(programC.bufferNodes[0], InputInfo(C));

    queueExecute(task, output, alpha, beta, ctx);
}
#include "batch-gemm.h"

BatchGemmKernel::BatchGemmKernel(
    InferencingContext* ctx,
    Expr A,
    Expr B,
    Expr C,
    SinkExpr sinkExpr,
    Expr Out)
    : context(ctx), sinkExpr(sinkExpr)
{
    int globalRegCounter = 0;

    // Compile all four expressions
    programA = compileExprToProgram(A, &globalRegCounter);
    programB = compileExprToProgram(B, &globalRegCounter);
    programC = compileExprToProgram(C, &globalRegCounter);
    programOut = compileExprToProgram(Out, &globalRegCounter);

    String typeArgs[] = {
        programA.getSlangTypeName(),
        programB.getSlangTypeName(),
        programC.getSlangTypeName(),
        sinkExpr.node->getSlangTypeName(),
        programOut.getSlangTypeName()};
    pipeline = ctx->createComputePipeline("batchGemm", makeArrayView(typeArgs));
}

// Allocate output buffer C [BatchSize, M, N]
BufferView BatchGemmKernel::allocResultBuffer(int batchSize, int m, int n)
{
    size_t size = (size_t)batchSize * m * n * sizeof(float);
    return context->allocScratchBuffer(size, "BatchGemm_Result");
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    int M,
    int N,
    int K,
    int batchSize,
    float alpha,
    float beta,
    EvalContext& ctx)
{
    SinkExprEvalContext sinkExprCtx;
    sinkExprCtx.outputBuffer = output;
    sinkExprCtx.logicalShape = Shape(batchSize, M, N);

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack in order: A, B, C, FOut, M, N, K, alpha, beta, output
    programA.pack(writer, ctx);
    programB.pack(writer, ctx);
    programC.pack(writer, ctx);
    sinkExpr.node->pack(writer, sinkExprCtx);
    programOut.pack(writer, ctx);
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
    BufferView output,
    int M,
    int N,
    int K,
    int batchSize,
    float alpha,
    float beta,
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
        ctx.inputs.add(it.first.node, it.second);

    queueExecute(task, output, M, N, K, batchSize, alpha, beta, ctx);
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    int M,
    int N,
    int K,
    int batchSize,
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
    queueExecute(task, output, M, N, K, batchSize, alpha, beta, ctx);
}

void BatchGemmKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    int M,
    int N,
    int K,
    int batchSize,
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
    queueExecute(task, output, M, N, K, batchSize, alpha, beta, ctx);
}
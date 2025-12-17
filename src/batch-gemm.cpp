#include "batch-gemm.h"

BatchGemmKernel::BatchGemmKernel(InferencingContext* ctx, Expr A, Expr B, Expr C, Expr Out)
    : context(ctx)
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
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
        ctx.inputs.add(it.first.node, it.second);

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack in order: A, B, C, FOut, M, N, K, alpha, beta, output
    programA.pack(writer, ctx);
    programB.pack(writer, ctx);
    programC.pack(writer, ctx);
    programOut.pack(writer, ctx);

    writer.write<uint32_t>((uint32_t)M);
    writer.write<uint32_t>((uint32_t)N);
    writer.write<uint32_t>((uint32_t)K);
    writer.write(alpha);
    writer.write(beta);
    writer.write(output.getDeviceAddress());

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
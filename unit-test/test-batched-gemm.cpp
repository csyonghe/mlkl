#include "batch-gemm.h"
#include "elementwise.h"
#include "test-kernels.h"

SlangResult testBatchGemm(InferencingContext* ctx)
{
    printf("Running testBatchGemm (Standard)...\n");
    int batch = 2, M = 16, N = 16, K = 64;
    List<float> A, B, Expected;
    initRandom(A, batch * M * K);
    initRandom(B, batch * N * K);
    Expected.setCount(batch * M * N);

    // Standard case: A @ B^T
    cpuBatchGemm(A.getBuffer(), B.getBuffer(), Expected.getBuffer(), batch, M, N, K, false, true);

    auto bufA = ctx->createPersistentBuffer(A, "GemmA");
    auto bufB = ctx->createPersistentBuffer(B, "GemmB");

    // Using Generic BatchGemmKernel
    // A: Buffer
    // B: Transpose(Buffer, 0, 1) - Wait, input to kernel is just expression.
    // If we use the old logic (kernel takes raw buffers and flags), we use that.
    // Assuming we migrated to the NEW Generic Kernel:

    // Construct Exprs
    auto exprA = buffer();
    // cpuBatchGemm(transB=true) means we are multiplying by B transposed.
    // Input B is [N, K] (Physical). We treat it as [K, N] logic?
    // Wait, cpuBatchGemm transB=true means:
    // Logical B = Physical B^T.
    // Physical B is [N, K]. Logical B is [K, N].
    // So in Expr terms: LogicalB = transpose(PhysicalB, 0, 1).
    // Let's stick to the setup:
    // Physical B: [batch, N, K]
    // Logical B needed for Gemm (KxN): Transpose(PhysicalB)
    auto inputB = buffer();
    auto exprB = transpose(inputB, 0, 1);
    auto exprC = constant(0.0f); // No bias for standard test
    auto exprOut = kernelOutput();

    BatchGemmKernel kernel(ctx, exprA, exprB, exprC, exprOut);
    auto bufOut = ctx->allocScratchBuffer(batch * M * N * sizeof(float), "GemmOut");

    Dictionary<Expr, InputInfo> inputs;
    inputs.add(exprA, InputInfo(Shape{batch, M, K}, BufferView(bufA)));
    inputs.add(inputB, InputInfo(Shape{batch, N, K}, BufferView(bufB)));
    auto task = ctx->createTask();
    kernel.queueExecute(task, bufOut, M, N, K, batch, 1.0f, 0.0f, inputs);
    task.execute();

    if (!checkOutput(ctx, bufOut, Expected))
    {
        printf("testBatchGemm FAILED\n");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

SlangResult testFusedBatchGemm(InferencingContext* ctx)
{
    printf("Running testFusedBatchGemm (Transpose A + Scale B + Bias + ReLU)...\n");
    int B = 2, M = 16, N = 16, K = 32;

    List<float> dataA, dataB, dataC, expected;
    initRandom(dataA, B * K * M);
    initRandom(dataB, B * K * N);
    initRandom(dataC, B * M * N);
    expected.setCount(B * M * N);

    cpuFusedGemm(
        dataA.getBuffer(),
        dataB.getBuffer(),
        dataC.getBuffer(),
        expected.getBuffer(),
        B,
        M,
        N,
        K);

    auto bufA = ctx->createPersistentBuffer(dataA, "bufA");
    auto bufB = ctx->createPersistentBuffer(dataB, "bufB");
    auto bufC = ctx->createPersistentBuffer(dataC, "bufC");

    // Expressions
    // A: [B, K, M] -> Transpose(1, 2) -> [B, M, K]
    auto bufAExpr = buffer();
    auto exprA = transpose(bufAExpr, 1, 2);
    // B: [B, K, N] -> Scale
    auto leafB = buffer();
    auto exprB = leafB * 2.0f;
    // C: Bias
    auto exprC = buffer();
    // Out: ReLU
    auto exprOut = relu(kernelOutput());

    BatchGemmKernel kernel(ctx, exprA, exprB, exprC, exprOut);

    auto task = ctx->createTask();
    auto bufOut = kernel.allocResultBuffer(B, M, N);

    Dictionary<Expr, InputInfo> inputs;
    inputs.add(bufAExpr, InputInfo(Shape{B, K, M}, BufferView(bufA)));
    inputs.add(leafB, InputInfo(Shape{B, K, N}, BufferView(bufB)));
    inputs.add(exprC, InputInfo(Shape{B, M, N}, BufferView(bufC)));

    kernel.queueExecute(task, bufOut, M, N, K, B, 1.0f, 1.0f, inputs);
    task.execute();

    if (!checkOutput(ctx, bufOut, expected))
    {
        printf("testFusedBatchGemm FAILED\n");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}
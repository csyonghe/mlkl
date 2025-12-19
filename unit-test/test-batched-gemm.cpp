#include "batch-gemm.h"
#include "elementwise.h"
#include "test-kernels.h"

void cpuBatchGemm(
    const float* A,
    const float* B,
    float* C,
    int batch,
    int M,
    int N,
    int K,
    bool transA,
    bool transB)
{
    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int strideA = M * K;
    int strideB = K * N;
    int strideC = M * N;

    for (int b = 0; b < batch; b++)
    {
        const float* matA = A + b * strideA;
        const float* matB = B + b * strideB;
        float* matC = C + b * strideC;
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    float valA = transA ? matA[k * lda + r] : matA[r * lda + k];
                    float valB = transB ? matB[c * ldb + k] : matB[k * ldb + c];
                    sum += valA * valB;
                }
                matC[r * N + c] = sum;
            }
        }
    }
}

SlangResult testBatchGemm(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    int batch = 2, M = 16, N = 16, K = 64;
    List<float> A, B, Expected;
    initRandom(A, batch * M * K);
    initRandom(B, batch * N * K);
    Expected.setCount(batch * M * N);

    // Standard case: A @ B^T
    cpuBatchGemm(A.getBuffer(), B.getBuffer(), Expected.getBuffer(), batch, M, N, K, false, true);

    auto bufA = ctx->createPersistentBuffer(A, "GemmA");
    auto bufB = ctx->createPersistentBuffer(B, "GemmB");

    // Construct Exprs
    auto exprA = buffer();
    auto inputB = buffer();
    auto exprB = transpose(inputB, 1, 2);
    auto exprC = constant(0.0f); // No bias for standard test
    auto exprOut = kernelOutput();

    BatchGemmKernel kernel(ctx, exprA, exprB, exprC, bufferSink(), exprOut);
    auto bufOut = ctx->allocScratchBuffer(batch * M * N * sizeof(float), "GemmOut");

    auto task = ctx->createTask();
    kernel.queueExecute(
        task,
        bufOut,
        M,
        N,
        K,
        batch,
        1.0f,
        0.0f,
        {InputInfo(Shape{batch, M, K}, BufferView(bufA)),
         InputInfo(Shape{batch, N, K}, BufferView(bufB))});
    task.execute();

    if (!checkOutput(ctx, bufOut, Expected))
    {
        printf("testBatchGemm FAILED\n");
        return SLANG_FAIL;
    }
    MLKL_TEST_OK();
}

static void cpuFusedGemm(
    const float* A,
    const float* B,
    const float* C,
    float* Output,
    int batch,
    int M,
    int N,
    int K)
{
    // A: [Batch, K, M] (Physical)
    // B: [Batch, K, N] (Physical)
    // C: [Batch, M, N]
    // Op: ReLU( (Trans(A) @ (B*2)) + C )
    for (int b = 0; b < batch; b++)
    {
        const float* pA = A + b * (K * M);
        const float* pB = B + b * (K * N);
        const float* pC = C + b * (M * N);
        float* pOut = Output + b * (M * N);

        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    // Transpose A: A[r, k] -> Physical pA[k * M + r]
                    float valA = pA[k * M + r];
                    // Scale B: B[k, c] -> Physical pB[k * N + c] * 2.0
                    float valB = pB[k * N + c] * 2.0f;
                    sum += valA * valB;
                }
                float res = sum + pC[r * N + c];
                pOut[c * M + r] = std::max(0.0f, res); // Transposed output.
            }
        }
    }
}


SlangResult testFusedBatchGemm(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

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
    // Sink: Permute
    auto sinkExpr = permute(bufferSink(), {0, 2, 1});
    BatchGemmKernel kernel(ctx, exprA, exprB, exprC, sinkExpr, exprOut);

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
    MLKL_TEST_OK();
}
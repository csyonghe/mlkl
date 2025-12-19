#include "batch-gemm.h"
#include "cross-attention.h"
#include "flash-attention.h"
#include "softmax.h"
#include "test-kernels.h"
#include "torch-reader.h"

#include <algorithm>
#include <cmath>
#include <random>

using namespace Slang;

void cpuCrossAttention(
    List<float>& finalOut,
    const List<float>& inputLatent,
    const List<float>& context,
    const List<float>& wQ,
    const List<float>& wK,
    const List<float>& wV,
    const List<float>& wOut,
    const List<float>& bOut,
    int B,
    int SeqQ,
    int SeqKV,
    int Dim,
    int ContextDim,
    int Heads)
{
    int headDim = Dim / Heads;
    float scale = 1.0f / sqrtf((float)headDim);

    // 1. Projections
    List<float> Q, K, V;
    Q.setCount(B * SeqQ * Dim);
    K.setCount(B * SeqKV * Dim);
    V.setCount(B * SeqKV * Dim);

    cpuLinear(inputLatent.getBuffer(), wQ.getBuffer(), nullptr, Q.getBuffer(), B, SeqQ, Dim, Dim);
    cpuLinear(
        context.getBuffer(),
        wK.getBuffer(),
        nullptr,
        K.getBuffer(),
        B,
        SeqKV,
        ContextDim,
        Dim);
    cpuLinear(
        context.getBuffer(),
        wV.getBuffer(),
        nullptr,
        V.getBuffer(),
        B,
        SeqKV,
        ContextDim,
        Dim);

    // 2. Attention
    List<float> attnOut;
    attnOut.setCount(B * SeqQ * Dim);
    List<float> scores;
    scores.setCount(SeqKV);

    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < Heads; h++)
        {
            for (int i = 0; i < SeqQ; i++)
            {
                // Score Calculation
                float maxScore = -1e38f;
                for (int j = 0; j < SeqKV; j++)
                {
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        int idxQ = (b * SeqQ + i) * Dim + (h * headDim + d);
                        int idxK = (b * SeqKV + j) * Dim + (h * headDim + d);
                        dot += Q[idxQ] * K[idxK];
                    }
                    scores[j] = dot * scale;
                    maxScore = std::max(maxScore, scores[j]);
                }
                // Softmax
                float sumProb = 0.0f;
                for (int j = 0; j < SeqKV; j++)
                {
                    scores[j] = std::exp(scores[j] - maxScore);
                    sumProb += scores[j];
                }
                for (int j = 0; j < SeqKV; j++)
                    scores[j] /= sumProb;

                // Weighted Sum
                for (int d = 0; d < headDim; d++)
                {
                    float val = 0.0f;
                    for (int j = 0; j < SeqKV; j++)
                    {
                        int idxV = (b * SeqKV + j) * Dim + (h * headDim + d);
                        val += scores[j] * V[idxV];
                    }
                    int idxOut = (b * SeqQ + i) * Dim + (h * headDim + d);
                    attnOut[idxOut] = val;
                }
            }
        }
    }

    // 3. Output Projection & Residual
    List<float> projResult;
    projResult.setCount(B * SeqQ * Dim);
    cpuLinear(
        attnOut.getBuffer(),
        wOut.getBuffer(),
        bOut.getBuffer(),
        projResult.getBuffer(),
        B,
        SeqQ,
        Dim,
        Dim);

    finalOut.setCount(B * SeqQ * Dim);
    for (int i = 0; i < finalOut.getCount(); i++)
        finalOut[i] = inputLatent[i] + projResult[i];
}

// ============================================================================
// TEST RUNNERS
// ============================================================================

SlangResult testSoftmax(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    int rows = 16, cols = 128;
    List<float> inputData, expectedOutput;
    initRandom(inputData, rows * cols);
    expectedOutput.setCount(inputData.getCount());

    cpuSoftmax(inputData.getBuffer(), expectedOutput.getBuffer(), cols, rows);

    auto inputBuf = ctx->createPersistentBuffer(inputData, "SoftmaxInput");
    SoftmaxKernel kernel(ctx);
    auto task = ctx->createTask();
    auto outputBuf = kernel.allocateResultBuffer(rows, cols);
    kernel.queueExecute(task, outputBuf, BufferView(inputBuf), rows, cols);
    task.execute();

    if (!checkOutput(ctx, outputBuf, expectedOutput))
    {
        printf("testSoftmax FAILED\n");
        return SLANG_FAIL;
    }
    MLKL_TEST_OK();
}

SlangResult testCrossAttentionFull(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    int B = 2, Dim = 64, ContextDim = 32, Heads = 1, SeqQ = 16, SeqKV = 8;

    // Inputs
    List<float> x, c;
    initRandom(x, B * SeqQ * Dim);
    initRandom(c, B * SeqKV * ContextDim);

    // Weights
    List<float> wQ, wK, wV, wOut, bOut;

    // Q, K, V: Generate weights, but NO biases
    initRandom(wQ, Dim * Dim);
    initRandom(wK, Dim * ContextDim);
    initRandom(wV, Dim * ContextDim);

    // Out: Generate weights AND biases
    initRandom(wOut, Dim * Dim);
    initRandom(bOut, Dim);

    // Write weights to memory stream.
    RefPtr<OwnedMemoryStream> memStream = new OwnedMemoryStream(FileAccess::ReadWrite);
    {
        // Pass 'false' for Q, K, V
        writeLinearWeights(memStream, wQ, List<float>());
        writeLinearWeights(memStream, wK, List<float>());
        writeLinearWeights(memStream, wV, List<float>());

        // Pass 'true' for Out
        writeLinearWeights(memStream, wOut, bOut);
    }

    // CPU Ref
    List<float> expected;
    cpuCrossAttention(
        expected,
        x,
        c,
        wQ,
        wK,
        wV,
        wOut,
        bOut,
        B,
        SeqQ,
        SeqKV,
        Dim,
        ContextDim,
        Heads);

    // GPU Run
    CrossAttentionKernel kernel(ctx, Dim, ContextDim, Dim / Heads);
    {
        // Load weights from memory stream.
        memStream->seek(SeekOrigin::Start, 0);
        TorchParamReader reader(memStream);
        if (SLANG_FAILED(kernel.loadParams(reader)))
            return SLANG_FAIL;
    }

    auto bufX = ctx->createPersistentBuffer(x, "CA_X");
    auto bufC = ctx->createPersistentBuffer(c, "CA_C");
    auto task = ctx->createTask();
    ctx->pushAllocScope();
    auto bufOut = kernel.allocateResultBuffer(B, SeqQ, Dim);
    kernel.queueExecute(task, bufOut, BufferView(bufX), BufferView(bufC), B, SeqQ, SeqKV, Heads);
    task.execute();
    ctx->popAllocScope();

    if (!checkOutput(ctx, bufOut, expected))
    {
        printf("testCrossAttentionFull FAILED\n");
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

void runCpuAttentionCore(
    List<float>& out,
    const List<float>& Q,
    const List<float>& K,
    const List<float>& V,
    int B,
    int H,
    int Sq,
    int Skv,
    int D)
{
    int headDim = D / H;
    float scale = 1.0f / sqrtf((float)headDim);
    out.setCount(B * H * Sq * headDim);

    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < H; h++)
        {
            // Calculate base offsets for this head in the planar layout
            int qHeadOffset = (b * H + h) * (Sq * headDim);
            int kvHeadOffset = (b * H + h) * (Skv * headDim);

            for (int i = 0; i < Sq; i++)
            {
                std::vector<float> scores(Skv);
                float maxScore = -1e38f;

                // 1. Compute Dot Product Scores
                for (int j = 0; j < Skv; j++)
                {
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        float qVal = Q[qHeadOffset + i * headDim + d];
                        float kVal = K[kvHeadOffset + j * headDim + d];
                        dot += qVal * kVal;
                    }
                    scores[j] = dot * scale;
                    maxScore = std::max(maxScore, scores[j]);
                }

                // 2. Softmax
                float sumProb = 0.0f;
                for (int j = 0; j < Skv; j++)
                {
                    scores[j] = std::exp(scores[j] - maxScore);
                    sumProb += scores[j];
                }
                for (int j = 0; j < Skv; j++)
                {
                    scores[j] /= sumProb;
                }

                // 3. Weighted Sum (V)
                for (int d = 0; d < headDim; d++)
                {
                    float val = 0.0f;
                    for (int j = 0; j < Skv; j++)
                    {
                        val += scores[j] * V[kvHeadOffset + j * headDim + d];
                    }
                    out[qHeadOffset + i * headDim + d] = val;
                }
            }
        }
    }
}

SlangResult testFlashAttention(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();
    // 1. Configuration (Cross-attention scenario)
    int B = 1;
    int H = 4;
    int Sq = 128; // Latent tokens
    int Skv = 64; // Context tokens
    int Dim = 128;
    int headDim = Dim / H;
    float scale = 1.0f / sqrtf((float)headDim);

    // 2. Generate Data
    auto generate = [&](int size)
    {
        List<float> list;
        list.setCount(size);
        for (int i = 0; i < size; i++)
            list[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        return list;
    };

    List<float> hostQ = generate(B * H * Sq * headDim);
    List<float> hostK = generate(B * H * Skv * headDim);
    List<float> hostV = generate(B * H * Skv * headDim);

    // 3. Run CPU Reference
    List<float> cpuOut;
    runCpuAttentionCore(cpuOut, hostQ, hostK, hostV, B, H, Sq, Skv, Dim);

    // 4. Run GPU Kernel
    // Expressions: Simple buffer pass-throughs
    Expr eQ = buffer();
    Expr eK = buffer();
    Expr eV = buffer();
    Expr eOutFunc = kernelOutput();

    // Use tile sizes that divide your sequences (e.g., 32x32)
    RefPtr<FlashAttentionKernel> kernel =
        new FlashAttentionKernel(ctx, eQ, eK, eV, eOutFunc, 32, 32, headDim);

    auto bufQ = ctx->createPersistentBuffer(hostQ, "Q");
    auto bufK = ctx->createPersistentBuffer(hostK, "K");
    auto bufV = ctx->createPersistentBuffer(hostV, "V");
    auto bufOut = kernel->allocateResultBuffer(Sq, H, B);

    Dictionary<Expr, InputInfo> inputs;
    inputs.add(eQ, InputInfo(Shape{B, H, Sq, headDim}, bufQ));
    inputs.add(eK, InputInfo(Shape{B, H, Skv, headDim}, bufK));
    inputs.add(eV, InputInfo(Shape{B, H, Skv, headDim}, bufV));

    auto task = ctx->createTask();
    kernel->queueExecute(task, bufOut, inputs, Sq, Skv, H, B, scale, false);
    task.execute();

    // 5. Comparison
    auto gpuResult = ctx->readBuffer<float>(bufOut);
    float maxDiff = 0.0f;
    for (int i = 0; i < cpuOut.getCount(); i++)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - cpuOut[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("Verification Results:\n");
        printf("  - Shape Q: [%d, %d, %d, %d]\n", B, H, Sq, headDim);
        printf("  - Shape K/V: [%d, %d, %d, %d]\n", B, H, Skv, headDim);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }
    MLKL_TEST_OK();
}

SlangResult testFlashAttentionFusedPermutation(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // 1. Configuration
    int B = 1;
    int H = 4;
    int Sq = 64;
    int Skv = 64;
    int headDim = 32;
    int Dim = H * headDim;
    float scale = 1.0f / sqrtf((float)headDim);

    // 2. Generate Data in Planar Layout for CPU [B, H, S, D]
    auto generate = [&](int size)
    {
        List<float> list;
        list.setCount(size);
        for (int i = 0; i < size; i++)
            list[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        return list;
    };

    List<float> hostQ_Planar = generate(B * H * Sq * headDim);
    List<float> hostK_Planar = generate(B * H * Skv * headDim);
    List<float> hostV_Planar = generate(B * H * Skv * headDim);

    // 3. Convert Planar to Interleaved for GPU Input [B, S, H, D]
    auto toInterleaved = [&](const List<float>& planar, int b, int h_cnt, int s_cnt, int d_cnt)
    {
        List<float> interleaved;
        interleaved.setCount(planar.getCount());
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h_cnt; hi++)
                for (int si = 0; si < s_cnt; si++)
                    for (int di = 0; di < d_cnt; di++)
                    {
                        int pIdx = (((bi * h_cnt + hi) * s_cnt + si) * d_cnt) + di;
                        int iIdx = (((bi * s_cnt + si) * h_cnt + hi) * d_cnt) + di;
                        interleaved[iIdx] = planar[pIdx];
                    }
        return interleaved;
    };

    List<float> hostQ_Interleaved = toInterleaved(hostQ_Planar, B, H, Sq, headDim);
    List<float> hostK_Interleaved = toInterleaved(hostK_Planar, B, H, Skv, headDim);
    List<float> hostV_Interleaved = toInterleaved(hostV_Planar, B, H, Skv, headDim);

    // 4. Run CPU Reference (Uses Planar inputs)
    List<float> cpuOut_Planar;
    runCpuAttentionCore(
        cpuOut_Planar,
        hostQ_Planar,
        hostK_Planar,
        hostV_Planar,
        B,
        H,
        Sq,
        Skv,
        Dim);

    // Convert CPU result to Interleaved for direct comparison with GPU fused output
    List<float> expectedOut_Interleaved = toInterleaved(cpuOut_Planar, B, H, Sq, headDim);

    // 5. Setup GPU Kernel with Permutation Expressions
    // Input: Interleaved [B, S, H, D] (indices: 0, 1, 2, 3)
    // We want kernel to see Planar [B, H, S, D] -> Permute to {0, 2, 1, 3}
    Expr eQ_Raw = buffer();
    Expr eK_Raw = buffer();
    Expr eV_Raw = buffer();

    Expr eQ_Fused = permute(eQ_Raw, {0, 2, 1, 3});
    Expr eK_Fused = permute(eK_Raw, {0, 2, 1, 3});
    Expr eV_Fused = permute(eV_Raw, {0, 2, 1, 3});

    // Output Fusion: Kernel produces Planar [B, H, S, D]
    // We want it to write Interleaved [B, S, H, D] -> Permute output index via {0, 2, 1, 3}
    Expr eOutCore = kernelOutput();
    SinkExpr eSink = permute(bufferSink(), {0, 2, 1, 3});

    RefPtr<FlashAttentionKernel> kernel = new FlashAttentionKernel(
        ctx,
        eQ_Fused,
        eK_Fused,
        eV_Fused,
        eOutCore,
        32,
        32,
        headDim,
        eSink);

    // 6. Execute GPU
    auto bufQ = ctx->createPersistentBuffer(hostQ_Interleaved, "Q_Interleaved");
    auto bufK = ctx->createPersistentBuffer(hostK_Interleaved, "K_Interleaved");
    auto bufV = ctx->createPersistentBuffer(hostV_Interleaved, "V_Interleaved");
    auto bufOut = kernel->allocateResultBuffer(Sq, H, B);

    Dictionary<Expr, InputInfo> inputs;
    // We must pass the "Physical" shape of the buffers on the GPU
    inputs.add(eQ_Raw, InputInfo(Shape{B, Sq, H, headDim}, bufQ));
    inputs.add(eK_Raw, InputInfo(Shape{B, Skv, H, headDim}, bufK));
    inputs.add(eV_Raw, InputInfo(Shape{B, Skv, H, headDim}, bufV));

    auto task = ctx->createTask();
    kernel->queueExecute(task, bufOut, inputs, Sq, Skv, H, B, scale, false);
    task.execute();

    // 7. Verify
    auto gpuResult = ctx->readBuffer<float>(bufOut);
    float maxDiff = 0.0f;
    for (int i = 0; i < expectedOut_Interleaved.getCount(); i++)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - expectedOut_Interleaved[i]));
        if (maxDiff > 1e-3f)
        {
            printf("Fused Permutation Result - Max Difference: %e\n", maxDiff);
            return SLANG_FAIL;
        }
    }

    MLKL_TEST_OK();
}

SlangResult testFlashAttentionInputPermutationOnly(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // 1. Configuration
    int B = 1;
    int H = 4;
    int Sq = 64;
    int Skv = 64;
    int headDim = 32;
    int Dim = H * headDim;
    float scale = 1.0f / sqrtf((float)headDim);

    // 2. Generate Reference Data in Planar Layout [B, H, S, D]
    auto generate = [&](int size)
    {
        List<float> list;
        list.setCount(size);
        for (int i = 0; i < size; i++)
            list[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        return list;
    };

    List<float> hostQ_Planar = generate(B * H * Sq * headDim);
    List<float> hostK_Planar = generate(B * H * Skv * headDim);
    List<float> hostV_Planar = generate(B * H * Skv * headDim);

    // 3. Convert Planar to Interleaved for GPU Input [B, S, H, D]
    auto toInterleaved = [&](const List<float>& planar, int b, int h_cnt, int s_cnt, int d_cnt)
    {
        List<float> interleaved;
        interleaved.setCount(planar.getCount());
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h_cnt; hi++)
                for (int si = 0; si < s_cnt; si++)
                    for (int di = 0; di < d_cnt; di++)
                    {
                        int pIdx = (((bi * h_cnt + hi) * s_cnt + si) * d_cnt) + di;
                        int iIdx = (((bi * s_cnt + si) * h_cnt + hi) * d_cnt) + di;
                        interleaved[iIdx] = planar[pIdx];
                    }
        return interleaved;
    };

    List<float> hostQ_Interleaved = toInterleaved(hostQ_Planar, B, H, Sq, headDim);
    List<float> hostK_Interleaved = toInterleaved(hostK_Planar, B, H, Skv, headDim);
    List<float> hostV_Interleaved = toInterleaved(hostV_Planar, B, H, Skv, headDim);

    // 4. Run CPU Reference (Standard Planar)
    List<float> cpuOut_Planar;
    runCpuAttentionCore(
        cpuOut_Planar,
        hostQ_Planar,
        hostK_Planar,
        hostV_Planar,
        B,
        H,
        Sq,
        Skv,
        Dim);

    // 5. Setup GPU Kernel
    // Inputs: Interleaved [B, S, H, D] -> Permute to Planar {0, 2, 1, 3}
    Expr eQ_Raw = buffer();
    Expr eK_Raw = buffer();
    Expr eV_Raw = buffer();

    Expr eQ_Fused = permute(eQ_Raw, {0, 2, 1, 3});
    Expr eK_Fused = permute(eK_Raw, {0, 2, 1, 3});
    Expr eV_Fused = permute(eV_Raw, {0, 2, 1, 3});

    // Output: No permutation. Flash kernel naturally writes Planar [B, H, S, D]
    Expr eOutPlanar = kernelOutput();

    RefPtr<FlashAttentionKernel> kernel =
        new FlashAttentionKernel(ctx, eQ_Fused, eK_Fused, eV_Fused, eOutPlanar, 32, 32, headDim);

    // 6. Execute GPU
    auto bufQ = ctx->createPersistentBuffer(hostQ_Interleaved, "Q_Interleaved");
    auto bufK = ctx->createPersistentBuffer(hostK_Interleaved, "K_Interleaved");
    auto bufV = ctx->createPersistentBuffer(hostV_Interleaved, "V_Interleaved");
    auto bufOut = kernel->allocateResultBuffer(Sq, H, B); // Standard Planar alloc

    Dictionary<Expr, InputInfo> inputs;
    // Map the raw expressions to Interleaved shapes
    inputs.add(eQ_Raw, InputInfo(Shape{B, Sq, H, headDim}, bufQ));
    inputs.add(eK_Raw, InputInfo(Shape{B, Skv, H, headDim}, bufK));
    inputs.add(eV_Raw, InputInfo(Shape{B, Skv, H, headDim}, bufV));

    auto task = ctx->createTask();
    kernel->queueExecute(task, bufOut, inputs, Sq, Skv, H, B, scale, false);
    task.execute();

    // 7. Verify Result against cpuOut_Planar
    auto gpuResult = ctx->readBuffer<float>(bufOut);
    float maxDiff = 0.0f;
    for (int i = 0; i < cpuOut_Planar.getCount(); i++)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - cpuOut_Planar[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] Input permutation logic failed.\n");
        printf("Input-Only Permute Result - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}
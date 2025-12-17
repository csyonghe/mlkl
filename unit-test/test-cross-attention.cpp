#include "batch-gemm.h"
#include "cross-attention.h"
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
    printf("Running testSoftmax...\n");
    int rows = 16, cols = 128;
    List<float> inputData, expectedOutput;
    initRandom(inputData, rows * cols);
    expectedOutput.setCount(inputData.getCount());

    cpuSoftmax(inputData.getBuffer(), expectedOutput.getBuffer(), cols, rows);

    auto inputBuf = ctx->createPersistentBuffer(inputData, "SoftmaxInput");
    SoftmaxKernel kernel(ctx);
    auto task = ctx->createTask();
    auto outputBuf = kernel.allocResultBuffer(rows, cols);
    kernel.queueExecute(task, outputBuf, BufferView(inputBuf), rows, cols);
    task.execute();

    if (!checkOutput(ctx, outputBuf, expectedOutput))
    {
        printf("testSoftmax FAILED\n");
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

SlangResult testCrossAttentionFull(InferencingContext* ctx)
{
    printf("Running testCrossAttentionFull...\n");
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
    CrossAttentionKernel kernel(ctx, Dim, ContextDim);
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
    auto bufOut = kernel.allocResultBuffer(B, SeqQ, Dim);
    kernel
        .queueExecute(task, bufOut, BufferView(bufX), BufferView(bufC), B, SeqQ, SeqKV, Dim, Heads);
    task.execute();
    ctx->popAllocScope();

    if (!checkOutput(ctx, bufOut, expected))
    {
        printf("testCrossAttentionFull FAILED\n");
        return SLANG_FAIL;
    }

    // Cleanup temp file if desired, or leave for debugging
    return SLANG_OK;
}

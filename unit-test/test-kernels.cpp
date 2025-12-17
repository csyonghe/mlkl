#include "test-kernels.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace Slang;

// --- Helpers ---

void initRandom(List<float>& data, int count)
{
    static std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    data.setCount(count);
    for (int i = 0; i < count; i++)
        data[i] = dist(gen);
}

bool checkOutput(InferencingContext* ctx, BufferView outputBuffer, const List<float>& expected)
{
    if (outputBuffer.size < expected.getCount() * sizeof(float))
        return false;

    auto outputData = ctx->readBuffer<float>(outputBuffer);
    float maxDiff = 0.0f;

    for (Index i = 0; i < outputData.getCount(); i++)
    {
        float val = outputData[i];
        float ref = expected[i];
        float diff = std::abs(val - ref);

        // Check for significant relative error or absolute error for small numbers
        if (diff > 1e-3f)
        {
            if (std::abs(ref) > 1e-2f && (diff / std::abs(ref)) > 1e-2f)
            {
                printf("Mismatch at %d: GPU=%f, CPU=%f (Diff: %f)\n", (int)i, val, ref, diff);
                return false;
            }
        }
    }
    return true;
}

void writeLinearWeights(Stream* fs, const List<float>& weights, const List<float>& biases)
{
    fs->write(weights.getBuffer(), weights.getCount() * sizeof(float));
    fs->write(biases.getBuffer(), biases.getCount() * sizeof(float));
}

// --- CPU Refs ---

void cpuSoftmax(const float* input, float* output, int stride, int count)
{
    for (int i = 0; i < count; i++)
    {
        const float* rowIn = input + i * stride;
        float* rowOut = output + i * stride;
        float maxVal = -1e38f;
        for (int j = 0; j < stride; j++)
            maxVal = std::max(maxVal, rowIn[j]);
        float sum = 0.0f;
        for (int j = 0; j < stride; j++)
        {
            float v = std::exp(rowIn[j] - maxVal);
            rowOut[j] = v;
            sum += v;
        }
        for (int j = 0; j < stride; j++)
            rowOut[j] /= sum;
    }
}

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

void cpuFusedGemm(
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
                pOut[r * N + c] = std::max(0.0f, res);
            }
        }
    }
}

void cpuLinear(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch,
    int seqLen,
    int inDim,
    int outDim)
{
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < seqLen; i++)
        {
            const float* inVec = input + (b * seqLen + i) * inDim;
            float* outVec = output + (b * seqLen + i) * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = (biases) ? biases[o] : 0.0f;
                for (int k = 0; k < inDim; k++)
                {
                    // [In, Out] Layout: weights[k * outDim + o]
                    sum += inVec[k] * weights[k * outDim + o];
                }
                outVec[o] = sum;
            }
        }
    }
}
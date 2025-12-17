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

SlangResult testCheck(bool condition, const char* testName, const char* message)
{
    if (!condition)
    {
        printf("%s: check failed: %s\n", testName, message);
        return SLANG_FAIL;
    }
    return SLANG_OK;
}

#include "test-kernels.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace Slang;

// --- Half Precision Conversion Helpers ---

// IEEE 754 half-precision format conversion
static uint16_t floatToHalfBits(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;
    
    if (exponent <= 0)
    {
        // Underflow to zero
        return (uint16_t)sign;
    }
    else if (exponent >= 31)
    {
        // Overflow to infinity
        return (uint16_t)(sign | 0x7C00);
    }
    
    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

static float halfBitsToFloat(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exponent == 0)
    {
        // Zero or denormal
        if (mantissa == 0)
        {
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, sizeof(float));
            return f;
        }
        // Denormal - convert to normalized float
        exponent = 1;
        while ((mantissa & 0x400) == 0)
        {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        exponent = exponent - 15 + 127;
    }
    else if (exponent == 31)
    {
        // Inf or NaN
        uint32_t bits = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &bits, sizeof(float));
        return f;
    }
    else
    {
        exponent = exponent - 15 + 127;
    }
    
    uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

void floatToHalf(const List<float>& src, List<uint16_t>& dst)
{
    dst.setCount(src.getCount());
    for (Index i = 0; i < src.getCount(); i++)
    {
        dst[i] = floatToHalfBits(src[i]);
    }
}

void halfToFloat(const List<uint16_t>& src, List<float>& dst)
{
    dst.setCount(src.getCount());
    for (Index i = 0; i < src.getCount(); i++)
    {
        dst[i] = halfBitsToFloat(src[i]);
    }
}

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

bool checkOutput(InferencingContext* ctx, TensorView outputBuffer, const List<float>& expected)
{
    return checkOutput(ctx, outputBuffer.getBufferView(), expected);
}

bool checkOutputHalf(InferencingContext* ctx, TensorView outputBuffer, const List<float>& expected)
{
    if (outputBuffer.bufferView.size < expected.getCount() * sizeof(uint16_t))
        return false;

    auto outputDataHalf = ctx->readBuffer<uint16_t>(outputBuffer.getBufferView());
    
    // Convert GPU output from half to float for comparison
    List<float> outputData;
    halfToFloat(outputDataHalf, outputData);

    // Half precision has about 3 decimal digits of precision
    // Use larger tolerance: 1e-2 relative or 1e-2 absolute
    for (Index i = 0; i < outputData.getCount(); i++)
    {
        float val = outputData[i];
        float ref = expected[i];
        float diff = std::abs(val - ref);

        // Use relative tolerance for larger values, absolute for small
        float tolerance = std::max(1e-2f, std::abs(ref) * 1e-2f);
        if (diff > tolerance)
        {
            printf("Mismatch at %d: GPU=%f, CPU=%f (Diff: %f, Tol: %f)\n", 
                   (int)i, val, ref, diff, tolerance);
            return false;
        }
    }
    return true;
}

void floatToInt(const List<float>& src, List<int32_t>& dst)
{
    dst.setCount(src.getCount());
    for (Index i = 0; i < src.getCount(); i++)
    {
        dst[i] = (int32_t)src[i];
    }
}

bool checkOutputInt(InferencingContext* ctx, TensorView outputBuffer, const List<float>& expected)
{
    if (outputBuffer.bufferView.size < expected.getCount() * sizeof(int32_t))
        return false;

    auto outputData = ctx->readBuffer<int32_t>(outputBuffer.getBufferView());

    for (Index i = 0; i < outputData.getCount(); i++)
    {
        int32_t val = outputData[i];
        int32_t ref = (int32_t)expected[i];
        int32_t diff = std::abs(val - ref);

        // For integer arithmetic, we expect exact results or very close
        if (diff > 1) // Allow 1 due to potential rounding differences
        {
            printf("Mismatch at %d: GPU=%d, CPU=%d (Diff: %d)\n", 
                   (int)i, val, ref, diff);
            return false;
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
                    // [Out In] Layout: weights[o*inDim + k]
                    sum += inVec[k] * weights[o * inDim + k];
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

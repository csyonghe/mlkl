#include "group-norm.h"
#include "test-kernels.h"

#include <cmath>

// =============================================================================
// Helper: CPU reference for GroupNorm
// =============================================================================
static void cpuGroupNorm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batchSize,
    int height,
    int width,
    int channels,
    int numGroups,
    float epsilon)
{
    int channelsPerGroup = channels / numGroups;
    int spatialSize = height * width;
    int groupSize = spatialSize * channelsPerGroup;

    for (int b = 0; b < batchSize; ++b)
    {
        for (int g = 0; g < numGroups; ++g)
        {
            int groupChannelStart = g * channelsPerGroup;

            // Compute mean and variance for this (batch, group)
            double sum = 0.0;
            double sumSq = 0.0;
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    for (int lc = 0; lc < channelsPerGroup; ++lc)
                    {
                        int c = groupChannelStart + lc;
                        // NHWC indexing
                        int idx = ((b * height + h) * width + w) * channels + c;
                        double val = input[idx];
                        sum += val;
                        sumSq += val * val;
                    }
                }
            }

            double mean = sum / groupSize;
            double variance = sumSq / groupSize - mean * mean;
            double invStd = 1.0 / std::sqrt(variance + epsilon);

            // Normalize each element in this group
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    for (int lc = 0; lc < channelsPerGroup; ++lc)
                    {
                        int c = groupChannelStart + lc;
                        int idx = ((b * height + h) * width + w) * channels + c;
                        double val = input[idx];
                        double normalized = (val - mean) * invStd;
                        output[idx] = (float)(gamma[c] * normalized + beta[c]);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Test: Basic GroupNorm with Float32
// =============================================================================
SlangResult testGroupNorm(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int batchSize = 2;
    const int height = 8;
    const int width = 8;
    const int channels = 32;
    const int numGroups = 8;
    const float epsilon = 1e-5f;

    // Create the GroupNorm kernel
    GroupNormKernel kernel(context, channels, numGroups, epsilon);

    // Prepare gamma (scale) and beta (bias) parameters
    List<float> hGamma, hBeta;
    hGamma.setCount(channels);
    hBeta.setCount(channels);
    for (int i = 0; i < channels; ++i)
    {
        hGamma[i] = 0.5f + (float)i * 0.02f;  // Scale: 0.5 to ~1.14
        hBeta[i] = (float)(i % 5) * 0.1f;     // Bias: 0, 0.1, 0.2, 0.3, 0.4 repeating
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input data in NHWC format
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        // Use varied values with some structure
        hInput[i] = std::sin((float)i * 0.01f) * 2.0f;
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, channels),
        hInput);

    // Allocate output buffer
    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, batchSize, height, width);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference
    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuGroupNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        batchSize,
        height,
        width,
        channels,
        numGroups,
        epsilon);

    // Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());
    float maxDiff = 0.0f;
    int maxDiffIdx = -1;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        float diff = std::abs(gpuResult[i] - hExpected[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
            maxDiffIdx = (int)i;
        }
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e at index %d\n", maxDiff, maxDiffIdx);
        printf("  - GPU: %f, CPU: %f\n", gpuResult[maxDiffIdx], hExpected[maxDiffIdx]);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: GroupNorm with single group (equivalent to InstanceNorm across spatial)
// =============================================================================
SlangResult testGroupNormSingleGroup(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters - single group means each (batch, channel) pair has its own stats
    const int batchSize = 2;
    const int height = 4;
    const int width = 4;
    const int channels = 8;
    const int numGroups = 1;  // All channels in one group
    const float epsilon = 1e-5f;

    // Create kernel
    GroupNormKernel kernel(context, channels, numGroups, epsilon);

    // Prepare parameters
    List<float> hGamma, hBeta;
    hGamma.setCount(channels);
    hBeta.setCount(channels);
    for (int i = 0; i < channels; ++i)
    {
        hGamma[i] = 1.0f;  // Identity scale
        hBeta[i] = 0.0f;   // Zero bias
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)(i % 7) - 3.0f;  // Values in [-3, 3]

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, channels),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, batchSize, height, width);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference
    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuGroupNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        batchSize,
        height,
        width,
        channels,
        numGroups,
        epsilon);

    // Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());
    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: GroupNorm with channels == groups (each channel is its own group)
// This is similar to LayerNorm per spatial location
// =============================================================================
SlangResult testGroupNormPerChannel(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters - numGroups == channels means each channel normalized separately
    const int batchSize = 1;
    const int height = 4;
    const int width = 4;
    const int channels = 8;
    const int numGroups = 8;  // One group per channel
    const float epsilon = 1e-5f;

    // Create kernel
    GroupNormKernel kernel(context, channels, numGroups, epsilon);

    // Prepare parameters
    List<float> hGamma, hBeta;
    hGamma.setCount(channels);
    hBeta.setCount(channels);
    for (int i = 0; i < channels; ++i)
    {
        hGamma[i] = 2.0f;   // Scale by 2
        hBeta[i] = 0.5f;    // Add 0.5
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input with different distributions per channel
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (int b = 0; b < batchSize; ++b)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                for (int c = 0; c < channels; ++c)
                {
                    int idx = ((b * height + h) * width + w) * channels + c;
                    // Each channel has a different mean
                    hInput[idx] = (float)(c * 10 + h + w);
                }
            }
        }
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, channels),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, batchSize, height, width);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference
    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuGroupNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        batchSize,
        height,
        width,
        channels,
        numGroups,
        epsilon);

    // Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());
    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: GroupNorm with larger spatial dimensions
// =============================================================================
SlangResult testGroupNormLarge(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters - larger spatial size to stress test
    const int batchSize = 2;
    const int height = 28;
    const int width = 28;
    const int channels = 64;
    const int numGroups = 8;
    const float epsilon = 1e-5f;

    // Create kernel
    GroupNormKernel kernel(context, channels, numGroups, epsilon);

    // Prepare parameters
    List<float> hGamma, hBeta;
    hGamma.setCount(channels);
    hBeta.setCount(channels);
    for (int i = 0; i < channels; ++i)
    {
        hGamma[i] = 1.0f + (float)i * 0.01f;
        hBeta[i] = 0.0f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::cos((float)i * 0.001f) * 3.0f;
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, channels),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, batchSize, height, width);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference
    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuGroupNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        batchSize,
        height,
        width,
        channels,
        numGroups,
        epsilon);

    // Verify with slightly larger tolerance for more elements
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());
    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));
    }

    if (maxDiff > 1e-2f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: GroupNorm with Half precision
// =============================================================================
SlangResult testGroupNormHalf(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters - smaller for half precision
    const int batchSize = 1;
    const int height = 4;
    const int width = 4;
    const int channels = 16;
    const int numGroups = 4;
    const float epsilon = 1e-5f;

    // Create kernel with half precision
    GroupNormKernel kernel(
        context,
        ElementType::Float16,
        buffer(),
        bufferSink(),
        channels,
        numGroups,
        epsilon);

    // Prepare parameters (in float, converted to half internally)
    List<float> hGamma, hBeta;
    hGamma.setCount(channels);
    hBeta.setCount(channels);
    for (int i = 0; i < channels; ++i)
    {
        hGamma[i] = 1.0f;
        hBeta[i] = 0.0f;
    }
    auto gammaHalf = convertFloatData(hGamma, ElementType::Float16);
    auto betaHalf = convertFloatData(hBeta, ElementType::Float16);
    kernel.gammaBuffer = context->createPersistentBuffer(gammaHalf.getBuffer(), gammaHalf.getCount());
    kernel.betaBuffer = context->createPersistentBuffer(betaHalf.getBuffer(), betaHalf.getCount());

    // Prepare input (values close to zero to avoid half precision overflow)
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = ((float)(i % 10) - 4.5f) * 0.1f;  // Small values
    }

    List<uint16_t> hInputHalf;
    floatToHalf(hInput, hInputHalf);

    auto inputBuffer = context->createTensor(
        ElementType::Float16,
        Shape(batchSize, height, width, channels),
        hInputHalf.getCount() * sizeof(uint16_t),
        hInputHalf.getBuffer());

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float16, batchSize, height, width);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference (in float)
    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuGroupNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        batchSize,
        height,
        width,
        channels,
        numGroups,
        epsilon);

    // Verify with larger tolerance for half precision
    auto gpuResultHalf = context->readBuffer<uint16_t>(outputBuffer.getBufferView());
    List<float> gpuResult;
    halfToFloat(gpuResultHalf, gpuResult);

    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));
    }

    // Half precision has lower accuracy
    if (maxDiff > 0.05f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: GroupNorm output approximately zero-mean and unit variance
// =============================================================================
SlangResult testGroupNormStats(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int batchSize = 1;
    const int height = 8;
    const int width = 8;
    const int channels = 8;
    const int numGroups = 2;  // 4 channels per group
    const float epsilon = 1e-5f;

    // Create kernel with identity transform (gamma=1, beta=0)
    GroupNormKernel kernel(context, channels, numGroups, epsilon);

    List<float> hGamma, hBeta;
    hGamma.setCount(channels);
    hBeta.setCount(channels);
    for (int i = 0; i < channels; ++i)
    {
        hGamma[i] = 1.0f;
        hBeta[i] = 0.0f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input with random-ish values
    List<float> hInput;
    hInput.setCount(batchSize * height * width * channels);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::sin((float)i * 0.123f) * 5.0f + std::cos((float)i * 0.456f) * 3.0f;
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, channels),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, batchSize, height, width);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // Read results
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());

    // Verify that each group has approximately zero mean and unit variance
    int channelsPerGroup = channels / numGroups;
    int spatialSize = height * width;
    int groupSize = spatialSize * channelsPerGroup;

    for (int b = 0; b < batchSize; ++b)
    {
        for (int g = 0; g < numGroups; ++g)
        {
            double sum = 0.0;
            double sumSq = 0.0;

            int groupChannelStart = g * channelsPerGroup;
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    for (int lc = 0; lc < channelsPerGroup; ++lc)
                    {
                        int c = groupChannelStart + lc;
                        int idx = ((b * height + h) * width + w) * channels + c;
                        double val = gpuResult[idx];
                        sum += val;
                        sumSq += val * val;
                    }
                }
            }

            double mean = sum / groupSize;
            double variance = sumSq / groupSize - mean * mean;

            // Mean should be close to 0
            if (std::abs(mean) > 1e-3)
            {
                printf(
                    "[FAILED] %s: Group (%d, %d) has non-zero mean: %f\n",
                    __func__,
                    b,
                    g,
                    mean);
                return SLANG_FAIL;
            }

            // Variance should be close to 1
            if (std::abs(variance - 1.0) > 1e-2)
            {
                printf(
                    "[FAILED] %s: Group (%d, %d) has non-unit variance: %f\n",
                    __func__,
                    b,
                    g,
                    variance);
                return SLANG_FAIL;
            }
        }
    }

    MLKL_TEST_OK();
}



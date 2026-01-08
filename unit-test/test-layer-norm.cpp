#include "layer-norm.h"
#include "rms-norm.h"
#include "test-kernels.h"

#include <cmath>

// =============================================================================
// Helper: CPU reference for LayerNorm
// =============================================================================
static void cpuLayerNorm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int numRows,
    int numFeatures,
    float epsilon)
{
    for (int row = 0; row < numRows; ++row)
    {
        const float* rowIn = input + row * numFeatures;
        float* rowOut = output + row * numFeatures;

        // Compute mean and variance for this row
        double sum = 0.0;
        double sumSq = 0.0;
        for (int col = 0; col < numFeatures; ++col)
        {
            double val = rowIn[col];
            sum += val;
            sumSq += val * val;
        }

        double mean = sum / numFeatures;
        double variance = sumSq / numFeatures - mean * mean;
        double invStd = 1.0 / std::sqrt(variance + epsilon);

        // Normalize each element
        for (int col = 0; col < numFeatures; ++col)
        {
            double val = rowIn[col];
            double normalized = (val - mean) * invStd;
            rowOut[col] = (float)(gamma[col] * normalized + beta[col]);
        }
    }
}

// =============================================================================
// Helper: CPU reference for RMSNorm
// =============================================================================
static void cpuRMSNorm(
    const float* input,
    const float* gamma,
    float* output,
    int numRows,
    int numFeatures,
    float epsilon)
{
    for (int row = 0; row < numRows; ++row)
    {
        const float* rowIn = input + row * numFeatures;
        float* rowOut = output + row * numFeatures;

        // Compute mean of squares for this row
        double sumSq = 0.0;
        for (int col = 0; col < numFeatures; ++col)
        {
            double val = rowIn[col];
            sumSq += val * val;
        }

        double meanSq = sumSq / numFeatures;
        double invRms = 1.0 / std::sqrt(meanSq + epsilon);

        // Normalize each element: gamma * x / rms
        for (int col = 0; col < numFeatures; ++col)
        {
            double val = rowIn[col];
            rowOut[col] = (float)(gamma[col] * val * invRms);
        }
    }
}

// =============================================================================
// Test: Basic LayerNorm with Float32
// =============================================================================
SlangResult testLayerNorm(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int numRows = 16;
    const int numFeatures = 64;
    const float epsilon = 1e-5f;

    // Create the LayerNorm kernel
    LayerNormKernel kernel(context, numFeatures, epsilon);

    // Prepare gamma (scale) and beta (bias) parameters
    List<float> hGamma, hBeta;
    hGamma.setCount(numFeatures);
    hBeta.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 0.8f + (float)i * 0.005f;  // Scale: 0.8 to ~1.12
        hBeta[i] = (float)(i % 3) * 0.1f;      // Bias: 0, 0.1, 0.2 repeating
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input data [numRows, numFeatures]
    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::sin((float)i * 0.02f) * 2.0f + std::cos((float)i * 0.03f);
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numFeatures),
        hInput);

    // Allocate output buffer
    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, numRows);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference
    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuLayerNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        numRows,
        numFeatures,
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
// Test: LayerNorm with identity transform (verify zero mean, unit variance)
// =============================================================================
SlangResult testLayerNormStats(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int numRows = 8;
    const int numFeatures = 32;
    const float epsilon = 1e-5f;

    // Create kernel with identity transform (gamma=1, beta=0)
    LayerNormKernel kernel(context, numFeatures, epsilon);

    List<float> hGamma, hBeta;
    hGamma.setCount(numFeatures);
    hBeta.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 1.0f;
        hBeta[i] = 0.0f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    // Prepare input with varied values
    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::sin((float)i * 0.1f) * 5.0f + (float)(i % 7);
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numFeatures),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, numRows);

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // Read results
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());

    // Verify each row has approximately zero mean and unit variance
    for (int row = 0; row < numRows; ++row)
    {
        double sum = 0.0;
        double sumSq = 0.0;
        for (int col = 0; col < numFeatures; ++col)
        {
            double val = gpuResult[row * numFeatures + col];
            sum += val;
            sumSq += val * val;
        }

        double mean = sum / numFeatures;
        double variance = sumSq / numFeatures - mean * mean;

        if (std::abs(mean) > 1e-3)
        {
            printf("[FAILED] %s: Row %d has non-zero mean: %f\n", __func__, row, mean);
            return SLANG_FAIL;
        }

        if (std::abs(variance - 1.0) > 1e-2)
        {
            printf("[FAILED] %s: Row %d has non-unit variance: %f\n", __func__, row, variance);
            return SLANG_FAIL;
        }
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: LayerNorm with large feature dimension
// =============================================================================
SlangResult testLayerNormLarge(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters - typical transformer hidden size
    const int numRows = 32;
    const int numFeatures = 512;
    const float epsilon = 1e-5f;

    LayerNormKernel kernel(context, numFeatures, epsilon);

    List<float> hGamma, hBeta;
    hGamma.setCount(numFeatures);
    hBeta.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 1.0f + (float)i * 0.001f;
        hBeta[i] = 0.0f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);
    kernel.betaBuffer = context->createPersistentBuffer(hBeta);

    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::cos((float)i * 0.001f) * 2.0f;
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numFeatures),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, numRows);

    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuLayerNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        numRows,
        numFeatures,
        epsilon);

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
// Test: LayerNorm with Half precision
// =============================================================================
SlangResult testLayerNormHalf(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    const int numRows = 8;
    const int numFeatures = 32;
    const float epsilon = 1e-5f;

    LayerNormKernel kernel(
        context,
        ElementType::Float16,
        buffer(),
        bufferSink(),
        numFeatures,
        epsilon);

    List<float> hGamma, hBeta;
    hGamma.setCount(numFeatures);
    hBeta.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 1.0f;
        hBeta[i] = 0.0f;
    }
    auto gammaHalf = convertFloatData(hGamma, ElementType::Float16);
    auto betaHalf = convertFloatData(hBeta, ElementType::Float16);
    kernel.gammaBuffer = context->createPersistentBuffer(gammaHalf.getBuffer(), gammaHalf.getCount());
    kernel.betaBuffer = context->createPersistentBuffer(betaHalf.getBuffer(), betaHalf.getCount());

    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = ((float)(i % 10) - 4.5f) * 0.2f;
    }

    List<uint16_t> hInputHalf;
    floatToHalf(hInput, hInputHalf);

    auto inputBuffer = context->createTensor(
        ElementType::Float16,
        Shape(numRows, numFeatures),
        hInputHalf.getCount() * sizeof(uint16_t),
        hInputHalf.getBuffer());

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float16, numRows);

    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuLayerNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hBeta.getBuffer(),
        hExpected.getBuffer(),
        numRows,
        numFeatures,
        epsilon);

    auto gpuResultHalf = context->readBuffer<uint16_t>(outputBuffer.getBufferView());
    List<float> gpuResult;
    halfToFloat(gpuResultHalf, gpuResult);

    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));
    }

    if (maxDiff > 0.05f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: Basic RMSNorm with Float32
// =============================================================================
SlangResult testRMSNorm(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    const int numRows = 16;
    const int numFeatures = 64;
    const float epsilon = 1e-5f;

    RMSNormKernel kernel(context, numFeatures, epsilon);

    List<float> hGamma;
    hGamma.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 0.9f + (float)i * 0.003f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);

    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::sin((float)i * 0.02f) * 2.0f + 0.5f;
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numFeatures),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, numRows);

    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuRMSNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hExpected.getBuffer(),
        numRows,
        numFeatures,
        epsilon);

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
// Test: RMSNorm with identity gamma (verify RMS scaling)
// =============================================================================
SlangResult testRMSNormIdentity(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    const int numRows = 4;
    const int numFeatures = 16;
    const float epsilon = 1e-5f;

    RMSNormKernel kernel(context, numFeatures, epsilon);

    // Identity gamma
    List<float> hGamma;
    hGamma.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 1.0f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);

    // Create input with known values
    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numFeatures; ++col)
        {
            // Each row has values that will give RMS = 1
            // For RMS=1: sqrt(mean(x^2)) = 1, so mean(x^2) = 1
            hInput[row * numFeatures + col] = (col % 2 == 0) ? 1.0f : -1.0f;
        }
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numFeatures),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, numRows);

    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // With gamma=1 and RMS=1, output should equal input
    auto gpuResult = context->readBuffer<float>(outputBuffer.getBufferView());
    float maxDiff = 0.0f;
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hInput[i]));
    }

    if (maxDiff > 1e-4f)
    {
        printf("[FAILED] %s: Output should match input when RMS=1 and gamma=1.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Test: RMSNorm with large feature dimension
// =============================================================================
SlangResult testRMSNormLarge(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    const int numRows = 32;
    const int numFeatures = 512;
    const float epsilon = 1e-6f;  // LLaMA uses smaller epsilon

    RMSNormKernel kernel(context, numFeatures, epsilon);

    List<float> hGamma;
    hGamma.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 1.0f;
    }
    kernel.gammaBuffer = context->createPersistentBuffer(hGamma);

    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = std::sin((float)i * 0.001f) * 0.5f;
    }

    auto inputBuffer = context->createTensor(
        ElementType::Float32,
        Shape(numRows, numFeatures),
        hInput);

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float32, numRows);

    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuRMSNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hExpected.getBuffer(),
        numRows,
        numFeatures,
        epsilon);

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
// Test: RMSNorm with Half precision
// =============================================================================
SlangResult testRMSNormHalf(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    const int numRows = 8;
    const int numFeatures = 32;
    const float epsilon = 1e-5f;

    RMSNormKernel kernel(
        context,
        ElementType::Float16,
        buffer(),
        bufferSink(),
        numFeatures,
        epsilon);

    List<float> hGamma;
    hGamma.setCount(numFeatures);
    for (int i = 0; i < numFeatures; ++i)
    {
        hGamma[i] = 1.0f;
    }
    auto gammaHalf = convertFloatData(hGamma, ElementType::Float16);
    kernel.gammaBuffer = context->createPersistentBuffer(gammaHalf.getBuffer(), gammaHalf.getCount());

    List<float> hInput;
    hInput.setCount(numRows * numFeatures);
    for (Index i = 0; i < hInput.getCount(); ++i)
    {
        hInput[i] = ((float)(i % 8) - 3.5f) * 0.2f;
    }

    List<uint16_t> hInputHalf;
    floatToHalf(hInput, hInputHalf);

    auto inputBuffer = context->createTensor(
        ElementType::Float16,
        Shape(numRows, numFeatures),
        hInputHalf.getCount() * sizeof(uint16_t),
        hInputHalf.getBuffer());

    auto outputBuffer = kernel.allocateResultBuffer(ElementType::Float16, numRows);

    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    List<float> hExpected;
    hExpected.setCount(hInput.getCount());
    cpuRMSNorm(
        hInput.getBuffer(),
        hGamma.getBuffer(),
        hExpected.getBuffer(),
        numRows,
        numFeatures,
        epsilon);

    auto gpuResultHalf = context->readBuffer<uint16_t>(outputBuffer.getBufferView());
    List<float> gpuResult;
    halfToFloat(gpuResultHalf, gpuResult);

    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));
    }

    if (maxDiff > 0.05f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}



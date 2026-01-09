#include "kernels.h"
#include "test-kernels.h"

SlangResult testConv2D(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
    auto inputBuffer =
        ctx->createTensor(ElementType::Float32, Shape(1, 5, 5, 1), sizeof(inputData), inputData);
    float convWeights[9] = {0.1, 0.5, 0.2, 0.5, 1.0, 0.5, 0.2, 0.5, 0.4};
    float convBiases[] = {1000.0f};
    Conv2DKernel convKernel = Conv2DKernel(ctx, 4, 3, 1, 1, 1);
    auto task = ctx->createTask();
    convKernel.loadParams(3, 1, convWeights, convBiases);
    auto outputBuffer = convKernel.allocateResultBuffer(ElementType::Float32, 5, 5, 1, 1);
    convKernel.queueExecute(task, outputBuffer, inputBuffer->getView(), 1);

    auto readWeight = [&](int x, int y) { return convWeights[y * 3 + x]; };

    renderDocBeginFrame();
    task.execute();
    auto outputData = ctx->readBuffer<float>(outputBuffer);
    renderDocEndFrame();
    float v0 = outputData[0];
    float expectedV0 = readInput(0, 0) * readWeight(1, 1) + readInput(1, 0) * readWeight(2, 1) +
                       readInput(0, 1) * readWeight(2, 1) + readInput(1, 1) * readWeight(2, 2) +
                       convBiases[0];
    TEST_CHECK("simpleConvolution", fabs(v0 - expectedV0) < 1e-3f);
    MLKL_TEST_OK();
}

SlangResult testConv2DHalf(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };

    // Convert input to half
    List<float> inputList;
    inputList.addRange(inputData, 25);
    List<uint16_t> inputHalf;
    floatToHalf(inputList, inputHalf);

    auto inputBuffer = ctx->createTensor(
        ElementType::Float16,
        Shape(1, 5, 5, 1),
        inputHalf.getCount() * sizeof(uint16_t),
        inputHalf.getBuffer());

    float convWeights[9] = {0.1f, 0.5f, 0.2f, 0.5f, 1.0f, 0.5f, 0.2f, 0.5f, 0.4f};
    float convBiases[] = {1000.0f};

    // Create kernel with Float16 element type
    Conv2DKernel convKernel = Conv2DKernel(
        ctx,
        ElementType::Float16,
        4,   // tileSize
        3,   // kernelSize
        1,   // stride
        1,   // inChannels
        1,   // outChannels
        buffer(),
        kernelOutput(),
        bufferSink(),
        "conv2d_half");

    auto task = ctx->createTask();
    convKernel.loadParams(3, 1, convWeights, convBiases);
    auto outputBuffer = convKernel.allocateResultBuffer(ElementType::Float16, 5, 5, 1, 1);
    convKernel.queueExecute(task, outputBuffer, inputBuffer->getView(), 1);

    auto readWeight = [&](int x, int y) { return convWeights[y * 3 + x]; };

    renderDocBeginFrame();
    task.execute();
    renderDocEndFrame();

    // Read output as half and convert to float
    auto outputDataHalf = ctx->readBuffer<uint16_t>(outputBuffer.getBufferView());
    List<float> outputData;
    halfToFloat(outputDataHalf, outputData);

    float v0 = outputData[0];
    float expectedV0 = readInput(0, 0) * readWeight(1, 1) + readInput(1, 0) * readWeight(2, 1) +
                       readInput(0, 1) * readWeight(2, 1) + readInput(1, 1) * readWeight(2, 2) +
                       convBiases[0];

    // Use larger tolerance for half precision
    TEST_CHECK("simpleConvolutionHalf", fabs(v0 - expectedV0) < 1.0f);
    MLKL_TEST_OK();
}

SlangResult testConv2DInt(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Use integer-friendly values (no fractional parts)
    float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    auto readInput = [&](int x, int y) { return (int)inputData[y * 5 + x]; };

    // Convert input to int32
    List<float> inputList;
    inputList.addRange(inputData, 25);
    List<int32_t> inputInt;
    floatToInt(inputList, inputInt);

    auto inputBuffer = ctx->createTensor(
        ElementType::Int32,
        Shape(1, 5, 5, 1),
        inputInt.getCount() * sizeof(int32_t),
        inputInt.getBuffer());

    // Integer weights (no fractional parts)
    float convWeights[9] = {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f};
    float convBiases[] = {100.0f};

    // Create kernel with Int32 element type
    Conv2DKernel convKernel = Conv2DKernel(
        ctx,
        ElementType::Int32,
        4,   // tileSize
        3,   // kernelSize
        1,   // stride
        1,   // inChannels
        1,   // outChannels
        buffer(),
        kernelOutput(),
        bufferSink(),
        "conv2d_int");

    auto task = ctx->createTask();
    convKernel.loadParams(3, 1, convWeights, convBiases);
    auto outputBuffer = convKernel.allocateResultBuffer(ElementType::Int32, 5, 5, 1, 1);
    convKernel.queueExecute(task, outputBuffer, inputBuffer->getView(), 1);

    auto readWeight = [&](int x, int y) { return (int)convWeights[y * 3 + x]; };

    renderDocBeginFrame();
    task.execute();
    renderDocEndFrame();

    // Read output as int32
    auto outputDataInt = ctx->readBuffer<int32_t>(outputBuffer.getBufferView());

    int v0 = outputDataInt[0];
    // For position (0,0), only center, right, bottom, and bottom-right weights contribute
    int expectedV0 = readInput(0, 0) * readWeight(1, 1) + readInput(1, 0) * readWeight(2, 1) +
                     readInput(0, 1) * readWeight(1, 2) + readInput(1, 1) * readWeight(2, 2) +
                     (int)convBiases[0];

    TEST_CHECK("simpleConvolutionInt", v0 == expectedV0);
    MLKL_TEST_OK();
}

// ============================================================================
// Test GEMM convolution with fused output expression (broadcast add)
// This tests that kernelOutputShape is correctly provided for broadcast resolution.
// The output expression adds a per-channel scale that broadcasts over spatial dims.
// If the shape is wrong, broadcast would fail or produce incorrect results.
// ============================================================================

SlangResult testConv2DGemmWithOutputExpr(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    const int batchSize = 1;
    const int inputH = 8;
    const int inputW = 8;
    const int inChannels = 4;
    const int outChannels = 8;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;
    const int outputH = 8;
    const int outputW = 8;

    // Generate random input
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inChannels);
    for (int i = 0; i < inputData.getCount(); i++)
        inputData[i] = (float)(i % 17) * 0.1f - 0.8f;

    auto inputBuffer = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, inputH, inputW, inChannels),
        inputData);

    // Generate weights and biases
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    for (int i = 0; i < weights.getCount(); i++)
        weights[i] = (float)((i * 7) % 13) * 0.05f - 0.3f;

    List<float> biases;
    biases.setCount(outChannels);
    for (int i = 0; i < outChannels; i++)
        biases[i] = (float)i * 0.1f;

    // Per-channel additive scale for output expression
    // Shape [1, 1, 1, OC] to broadcast over NHWC output [N, H, W, OC]
    List<float> outputScaleData;
    outputScaleData.setCount(outChannels);
    for (int oc = 0; oc < outChannels; oc++)
        outputScaleData[oc] = (float)(oc + 1) * 0.5f;  // 0.5, 1.0, 1.5, ...

    auto outputScaleBuffer = ctx->createTensor(
        ElementType::Float32,
        Shape(1, 1, 1, outChannels),  // Shape [1,1,1,OC] broadcasts over NHWC output
        outputScaleData);

    // Create Conv2D kernel WITHOUT output expression (for reference)
    Conv2DKernel convKernelRef(ctx, 4, kernelSize, stride, inChannels, outChannels);
    convKernelRef.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Create Conv2D kernel WITH fused output expression:
    // kernelOutput() + broadcast(buffer(), kernelOutput())
    // This adds a per-channel value that broadcasts across spatial dimensions
    Expr inputExpr = buffer();
    Expr scaleExpr = buffer();
    Expr outputExpr = kernelOutput() + broadcast(scaleExpr, kernelOutput());
    Conv2DKernel convKernelFused(ctx, ElementType::Float32, 4, kernelSize, stride, inChannels, outChannels,
                                  inputExpr, outputExpr, bufferSink());
    convKernelFused.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Allocate outputs
    auto outputRef = ctx->allocScratchTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        "output_ref");
    auto outputFused = ctx->allocScratchTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        "output_fused");

    // Run reference (flat, no output expr)
    {
        auto task = ctx->createTask();
        convKernelRef.queueExecute(task, outputRef, inputBuffer->getView(), padding, ConvolutionAlgorithm::Flat);
        task.execute();
    }

    // Apply output scale on CPU to reference output (add per-channel value)
    auto refData = ctx->readBuffer<float>(outputRef);
    List<float> expectedData;
    expectedData.setCount(refData.getCount());
    for (int n = 0; n < batchSize; n++)
    {
        for (int oh = 0; oh < outputH; oh++)
        {
            for (int ow = 0; ow < outputW; ow++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    int idx = ((n * outputH + oh) * outputW + ow) * outChannels + oc;
                    expectedData[idx] = refData[idx] + outputScaleData[oc];
                }
            }
        }
    }

    // Run fused (Gemm with broadcast add)
    {
        auto task = ctx->createTask();
        Dictionary<Expr, InputInfo> inputs;
        inputs.add(inputExpr, inputBuffer->getView());
        inputs.add(scaleExpr, outputScaleBuffer->getView());
        convKernelFused.queueExecute(task, outputFused, inputs, padding, ConvolutionAlgorithm::Gemm);
        task.execute();
    }

    auto fusedData = ctx->readBuffer<float>(outputFused);

    // Compare
    float maxDiff = 0.0f;
    for (Index i = 0; i < expectedData.getCount(); i++)
    {
        float diff = fabs(fusedData[i] - expectedData[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    TEST_CHECK("im2col_broadcast_output_matches_ref", maxDiff < 1e-3f);

    MLKL_TEST_OK();
}

// Test GEMM convolution with batched input, half precision, and fused input expressions
SlangResult testConv2DGemmBatchedHalfFused(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Input: [2, 16, 16, 8] - batch=2, 16x16 spatial, 8 channels
    const int batchSize = 2;
    const int inputH = 16;
    const int inputW = 16;
    const int inChannels = 8;
    const int outChannels = 16;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;

    const int outputH = (inputH + 2 * padding - kernelSize) / stride + 1;
    const int outputW = (inputW + 2 * padding - kernelSize) / stride + 1;

    // Generate input data (float, will convert to half)
    List<float> inputFloat;
    inputFloat.setCount(batchSize * inputH * inputW * inChannels);
    for (int i = 0; i < inputFloat.getCount(); i++)
    {
        inputFloat[i] = (float)(i % 31) * 0.05f - 0.75f;
    }

    // Convert to half precision
    List<uint16_t> inputHalf;
    floatToHalf(inputFloat, inputHalf);

    auto inputBuffer = ctx->createTensor(
        ElementType::Float16,
        Shape(batchSize, inputH, inputW, inChannels),
        inputHalf.getCount() * sizeof(uint16_t),
        inputHalf.getBuffer());

    // Generate weights
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    for (int i = 0; i < weights.getCount(); i++)
    {
        weights[i] = (float)((i * 11) % 19) * 0.03f - 0.25f;
    }

    List<float> biases;
    biases.setCount(outChannels);
    for (int i = 0; i < outChannels; i++)
    {
        biases[i] = (float)i * 0.05f - 0.4f;
    }

    // Create Conv2D kernel with fused input expression: input + 1.0
    // This tests that Gemm works with fused input transformations
    Expr inputExpr = buffer() + constant(1.0f);

    Conv2DKernel convKernel(
        ctx,
        ElementType::Float16,
        4,            // tileSize
        kernelSize,
        stride,
        inChannels,
        outChannels,
        inputExpr,    // Fused input: buffer() + 1.0
        kernelOutput(),
        bufferSink(),
        "conv2d_im2col_batched_half_fused");

    convKernel.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Allocate output buffers
    auto outputFlat = ctx->allocScratchTensor(
        ElementType::Float16,
        Shape(batchSize, outputH, outputW, outChannels),
        "flat_output");

    auto outputIm2Col = ctx->allocScratchTensor(
        ElementType::Float16,
        Shape(batchSize, outputH, outputW, outChannels),
        "im2col_output");

    // Run with Flat algorithm (reference)
    {
        auto task = ctx->createTask();
        convKernel.queueExecute(task, outputFlat, inputBuffer->getView(), padding, ConvolutionAlgorithm::Flat);
        task.execute();
    }

    // Run with Gemm algorithm
    {
        auto task = ctx->createTask();
        convKernel.queueExecute(task, outputIm2Col, inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
        task.execute();
    }

    // Compare results (read as half, convert to float)
    auto flatHalf = ctx->readBuffer<uint16_t>(outputFlat.getBufferView());
    auto im2colHalf = ctx->readBuffer<uint16_t>(outputIm2Col.getBufferView());

    List<float> flatData, im2colData;
    halfToFloat(flatHalf, flatData);
    halfToFloat(im2colHalf, im2colData);

    TEST_CHECK("im2col_batched_half_output_size", flatData.getCount() == im2colData.getCount());

    float maxDiff = 0.0f;
    for (Index i = 0; i < flatData.getCount(); i++)
    {
        float diff = fabs(flatData[i] - im2colData[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    // Half precision needs larger tolerance
    TEST_CHECK("im2col_batched_half_matches_flat", maxDiff < 0.1f);

    MLKL_TEST_OK();
}

// Test GEMM convolution with different input sizes to verify it handles various dimensions
SlangResult testConv2DGemmMultipleSizes(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    const int inChannels = 16;
    const int outChannels = 32;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;

    // Create the kernel once
    Conv2DKernel convKernel(ctx, 4, kernelSize, stride, inChannels, outChannels);

    // Generate weights
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    for (int i = 0; i < weights.getCount(); i++)
    {
        weights[i] = (float)((i * 7) % 13) * 0.02f - 0.1f;
    }

    List<float> biases;
    biases.setCount(outChannels);
    for (int i = 0; i < outChannels; i++)
    {
        biases[i] = (float)i * 0.01f;
    }

    convKernel.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Test with multiple input sizes using the SAME kernel
    int testSizes[] = {16, 32, 24};

    for (int size : testSizes)
    {
        int inputH = size;
        int inputW = size;
        int outputH = (inputH + 2 * padding - kernelSize) / stride + 1;
        int outputW = (inputW + 2 * padding - kernelSize) / stride + 1;

        // Generate input
        List<float> inputData;
        inputData.setCount(1 * inputH * inputW * inChannels);
        for (int i = 0; i < inputData.getCount(); i++)
        {
            inputData[i] = (float)(i % 23) * 0.05f - 0.5f;
        }

        auto inputBuffer = ctx->createTensor(
            ElementType::Float32,
            Shape(1, inputH, inputW, inChannels),
            inputData);

        // Allocate outputs
        auto outputFlat = ctx->allocScratchTensor(
            ElementType::Float32,
            Shape(1, outputH, outputW, outChannels),
            "flat_output");

        auto outputIm2Col = ctx->allocScratchTensor(
            ElementType::Float32,
            Shape(1, outputH, outputW, outChannels),
            "im2col_output");

        // Run both algorithms
        {
            auto task = ctx->createTask();
            convKernel.queueExecute(task, outputFlat, inputBuffer->getView(), padding, ConvolutionAlgorithm::Flat);
            task.execute();
        }
        {
            auto task = ctx->createTask();
            convKernel.queueExecute(task, outputIm2Col, inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
            task.execute();
        }

        // Compare
        auto flatData = ctx->readBuffer<float>(outputFlat);
        auto im2colData = ctx->readBuffer<float>(outputIm2Col);

        float maxDiff = 0.0f;
        for (Index i = 0; i < flatData.getCount(); i++)
        {
            float diff = fabs(flatData[i] - im2colData[i]);
            if (diff > maxDiff)
                maxDiff = diff;
        }

        char testName[64];
        snprintf(testName, sizeof(testName), "im2col_size_%d", size);
        TEST_CHECK(testName, maxDiff < 1e-3f);
    }

    MLKL_TEST_OK();
}

// ============================================================================
// GEMM-style Tiled Convolution Test
// ============================================================================
// Tests the new gemmConvolution kernel that caches both weights and input
// in shared memory for better data reuse.

SlangResult testConv2DGemm(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Input: [1, 16, 16, 32] - larger to exercise the tiled algorithm
    const int batchSize = 1;
    const int inputH = 16;
    const int inputW = 16;
    const int inChannels = 32;
    const int outChannels = 64;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;

    const int outputH = 16;
    const int outputW = 16;

    // Generate input
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inChannels);
    for (Index i = 0; i < inputData.getCount(); i++)
        inputData[i] = (float)(i % 17) * 0.1f - 0.8f;

    auto inputBuffer = ctx->createTensor(
        ElementType::Float32, Shape(batchSize, inputH, inputW, inChannels), inputData);

    // Generate weights and biases
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    for (Index i = 0; i < weights.getCount(); i++)
        weights[i] = (float)((i * 7) % 13) * 0.05f - 0.3f;

    List<float> biases;
    biases.setCount(outChannels);
    for (int i = 0; i < outChannels; i++)
        biases[i] = (float)i * 0.1f;

    // Create Conv2D kernel
    Conv2DKernel convKernel(ctx, 16, kernelSize, stride, inChannels, outChannels);
    convKernel.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Allocate output buffers
    auto outputFlat = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "flat_output");
    auto outputGemm = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "gemm_output");

    // Run with Flat algorithm (reference)
    {
        auto task = ctx->createTask();
        convKernel.queueExecute(task, outputFlat, inputBuffer->getView(), padding, ConvolutionAlgorithm::Flat);
        task.execute();
    }

    // Run with Gemm algorithm
    {
        auto task = ctx->createTask();
        convKernel.queueExecute(task, outputGemm, inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
        task.execute();
    }

    // Compare results
    auto flatData = ctx->readBuffer<float>(outputFlat);
    auto gemmData = ctx->readBuffer<float>(outputGemm);

    TEST_CHECK("gemm_output_size", flatData.getCount() == gemmData.getCount());

    float maxDiff = 0.0f;
    for (Index i = 0; i < flatData.getCount(); i++)
    {
        float diff = fabs(flatData[i] - gemmData[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    TEST_CHECK("gemm_matches_flat", maxDiff < 1e-3f);

    MLKL_TEST_OK();
}

// Test GEMM convolution with batched input
SlangResult testConv2DGemmBatched(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Batched input: [2, 8, 8, 16]
    const int batchSize = 2;
    const int inputH = 8;
    const int inputW = 8;
    const int inChannels = 16;
    const int outChannels = 32;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;

    const int outputH = 8;
    const int outputW = 8;

    // Generate input
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inChannels);
    for (Index i = 0; i < inputData.getCount(); i++)
        inputData[i] = (float)(i % 23) * 0.1f - 1.1f;

    auto inputBuffer = ctx->createTensor(
        ElementType::Float32, Shape(batchSize, inputH, inputW, inChannels), inputData);

    // Generate weights and biases
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    for (Index i = 0; i < weights.getCount(); i++)
        weights[i] = (float)((i * 11) % 19) * 0.04f - 0.38f;

    List<float> biases;
    biases.setCount(outChannels);
    for (int i = 0; i < outChannels; i++)
        biases[i] = (float)i * 0.05f;

    // Create Conv2D kernel
    Conv2DKernel convKernel(ctx, 8, kernelSize, stride, inChannels, outChannels);
    convKernel.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Allocate output buffers
    auto outputFlat = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "flat_batched");
    auto outputGemm = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "gemm_batched");

    // Run both algorithms
    {
        auto task = ctx->createTask();
        convKernel.queueExecute(task, outputFlat, inputBuffer->getView(), padding, ConvolutionAlgorithm::Flat);
        task.execute();
    }
    {
        auto task = ctx->createTask();
        convKernel.queueExecute(task, outputGemm, inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
        task.execute();
    }

    // Compare
    auto flatData = ctx->readBuffer<float>(outputFlat);
    auto gemmData = ctx->readBuffer<float>(outputGemm);

    float maxDiff = 0.0f;
    for (Index i = 0; i < flatData.getCount(); i++)
    {
        float diff = fabs(flatData[i] - gemmData[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    TEST_CHECK("gemm_batched_matches_flat", maxDiff < 1e-3f);

    MLKL_TEST_OK();
}

// ============================================================================
// Test Im2Col Expression Alone (using ElementwiseKernel to materialize)
// ============================================================================

SlangResult testIm2ColExpressionOnly(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Small input: [1, 4, 4, 2] (batch=1, 4x4, 2 channels)
    const int batchSize = 1;
    const int inputH = 4;
    const int inputW = 4;
    const int inputC = 2;
    const int kernelH = 3;
    const int kernelW = 3;
    const int strideH = 1;
    const int strideW = 1;
    const int padH = 1;
    const int padW = 1;
    const int outputH = (inputH + 2 * padH - kernelH) / strideH + 1;  // 4
    const int outputW = (inputW + 2 * padW - kernelW) / strideW + 1;  // 4

    // Generate input: value = n*1000 + h*100 + w*10 + c for easy debugging
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inputC);
    for (int n = 0; n < batchSize; n++)
        for (int h = 0; h < inputH; h++)
            for (int w = 0; w < inputW; w++)
                for (int c = 0; c < inputC; c++)
                    inputData[((n * inputH + h) * inputW + w) * inputC + c] =
                        (float)(n * 1000 + h * 100 + w * 10 + c);

    auto inputBuffer = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, inputH, inputW, inputC),
        inputData);

    // im2col output shape: [1, IC*KH*KW, N*OH*OW]
    const int K = inputC * kernelH * kernelW;  // 2 * 3 * 3 = 18
    const int N = batchSize * outputH * outputW;  // 1 * 4 * 4 = 16
    Shape im2colShape(1, K, N);

    auto outputBuffer = ctx->allocScratchTensor(ElementType::Float32, im2colShape, "im2col_output");

    // Create im2col expression and materialize it
    Expr inputExpr = buffer();
    Expr im2colExpr = im2col(inputExpr, kernelH, kernelW, strideH, strideW, padH, padW,
                              inputH, inputW, inputC, batchSize);

    ElementwiseKernel im2colKernel(ctx, ElementType::Float32, im2colExpr);

    {
        auto task = ctx->createTask();
        im2colKernel.queueExecute(task, outputBuffer, {inputBuffer->getView()});
        task.execute();
    }

    auto gpuData = ctx->readBuffer<float>(outputBuffer);

    // CPU reference: compute im2col manually
    List<float> cpuRef;
    cpuRef.setCount(K * N);

    auto getInput = [&](int n, int h, int w, int c) -> float {
        if (h < 0 || h >= inputH || w < 0 || w >= inputW)
            return 0.0f;  // Zero padding
        return inputData[((n * inputH + h) * inputW + w) * inputC + c];
    };

    for (int row = 0; row < K; row++)
    {
        for (int col = 0; col < N; col++)
        {
            // Decode col -> (n, oh, ow)
            int spatialSize = outputH * outputW;
            int n = col / spatialSize;
            int spatial = col % spatialSize;
            int oh = spatial / outputW;
            int ow = spatial % outputW;

            // Decode row -> (kh, kw, ic) using IC-minor ordering
            // row = kh * KW * IC + kw * IC + ic
            int ic = row % inputC;
            int spatialK = row / inputC;
            int kh = spatialK / kernelW;
            int kw = spatialK % kernelW;

            // Compute input position
            int ih = oh * strideH + kh - padH;
            int iw = ow * strideW + kw - padW;

            cpuRef[row * N + col] = getInput(n, ih, iw, ic);
        }
    }

    // Compare
    float maxDiff = 0.0f;
    for (int i = 0; i < K * N; i++)
    {
        float diff = fabs(gpuData[i] - cpuRef[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    TEST_CHECK("im2col_expression_correct", maxDiff < 1e-5f);

    MLKL_TEST_OK();
}

// ============================================================================
// Test Conv2DOutputSink (coordinate transformation from GEMM to NHWC)
// ============================================================================

SlangResult testConv2DOutputSink(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // Test: GEMM output [1, OC, N*OH*OW] -> NHWC output [N, OH, OW, OC]
    // We use BatchGemmKernel with identity-like setup:
    // A = [1, M, 1] (column vector of values we want to output)
    // B = [1, 1, N] (row of ones)
    // Result = A * B = [1, M, N] with each row containing the same value

    const int batchSize = 2;
    const int outputH = 3;
    const int outputW = 4;
    const int outChannels = 5;

    const int M = outChannels;
    const int spatialSize = outputH * outputW;  // 12
    const int N = batchSize * spatialSize;      // 24
    const int K = 1;

    // A: [1, M, K] = [1, 5, 1] - each row (output channel) has value oc
    List<float> aData;
    aData.setCount(M * K);
    for (int oc = 0; oc < M; oc++)
        aData[oc] = (float)(oc * 100);  // Value encodes output channel

    // B: [1, K, N] = [1, 1, 24] - each column has value col
    List<float> bData;
    bData.setCount(K * N);
    for (int col = 0; col < N; col++)
        bData[col] = (float)(col);  // Value encodes spatial position

    // C: zeros
    List<float> cData;
    cData.setCount(M * N);
    for (int i = 0; i < M * N; i++)
        cData[i] = 0.0f;

    auto aBuffer = ctx->createTensor(ElementType::Float32, Shape(1, M, K), aData);
    auto bBuffer = ctx->createTensor(ElementType::Float32, Shape(1, K, N), bData);
    auto cBuffer = ctx->createTensor(ElementType::Float32, Shape(1, M, N), cData);

    // Final NHWC shape: [N, OH, OW, OC]
    Shape nhwcShape(batchSize, outputH, outputW, outChannels);
    auto outputBuffer = ctx->allocScratchTensor(ElementType::Float32, nhwcShape, "nhwc_output");

    // Create BatchGemmKernel with Conv2DOutputSink
    Expr aExpr = buffer();
    Expr bExpr = buffer();
    Expr cExpr = buffer();
    Expr ohExpr = uniformConstant();
    Expr owExpr = uniformConstant();
    SinkExpr sink = conv2DOutputSink(bufferSink(), ohExpr, owExpr);

    BatchGemmKernel gemmKernel(ctx, ElementType::Float32, aExpr, bExpr, cExpr, sink, kernelOutput());

    {
        auto task = ctx->createTask();
        Dictionary<Expr, InputInfo> inputs;
        inputs.add(aExpr, aBuffer->getView());
        inputs.add(bExpr, bBuffer->getView());
        inputs.add(cExpr, cBuffer->getView());
        inputs.add(ohExpr, InputInfo((float)outputH));
        inputs.add(owExpr, InputInfo((float)outputW));
        gemmKernel.queueExecute(task, outputBuffer, 1.0f, 0.0f, inputs);
        task.execute();
    }

    auto gpuData = ctx->readBuffer<float>(outputBuffer);

    // GEMM result at [0, oc, col] = A[0, oc, 0] * B[0, 0, col] = (oc*100) * col
    // After Conv2DOutputSink: at NHWC [n, oh, ow, oc] we get value from [0, oc, col]
    // where col = n * OH*OW + oh * OW + ow
    float maxDiff = 0.0f;

    for (int n = 0; n < batchSize; n++)
    {
        for (int oh = 0; oh < outputH; oh++)
        {
            for (int ow = 0; ow < outputW; ow++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    int nhwcIdx = ((n * outputH + oh) * outputW + ow) * outChannels + oc;
                    int col = n * spatialSize + oh * outputW + ow;
                    float expected = (float)(oc * 100) * (float)col;
                    float got = gpuData[nhwcIdx];
                    float diff = fabs(got - expected);

                    if (diff > maxDiff)
                        maxDiff = diff;
                }
            }
        }
    }

    TEST_CHECK("conv2d_output_sink_correct", maxDiff < 1e-3f);

    MLKL_TEST_OK();
}

// ============================================================================
// Test Conv2D with non-trivial sinkExpr (permute from NHWC to NCHW)
// ============================================================================

SlangResult testConv2DWithPermuteSink(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    const int batchSize = 1;
    const int inputH = 8;
    const int inputW = 8;
    const int inChannels = 4;
    const int outChannels = 8;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;

    const int outputH = inputH;  // Same padding
    const int outputW = inputW;

    // Create input [N, H, W, C] in NHWC format
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inChannels);
    for (Index i = 0; i < inputData.getCount(); i++)
        inputData[i] = (float)(i % 17) * 0.1f - 0.8f;

    auto inputBuffer = ctx->createTensor(
        ElementType::Float32, Shape(batchSize, inputH, inputW, inChannels), inputData);

    // Weights and bias
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    List<float> biases;
    biases.setCount(outChannels);
    for (Index i = 0; i < weights.getCount(); i++)
        weights[i] = (float)(i % 13) * 0.05f - 0.3f;
    for (int i = 0; i < outChannels; i++)
        biases[i] = (float)i * 0.1f;

    // Output in NCHW format [N, C, H, W] due to permute sink
    auto outputNCHW = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outChannels, outputH, outputW), "output_nchw");

    // Reference output in NHWC format (using default bufferSink)
    auto outputNHWC = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "output_nhwc");

    // Create kernel with permute sink: NHWC [N,H,W,C] -> NCHW [N,C,H,W]
    // Sink permute: newCoords[i] = logicalCoords[pMap[i]]
    // Logical coords are [n,h,w,c], physical coords should be [n,c,h,w]
    // newCoords[0] = n = logical[0] → pMap[0] = 0
    // newCoords[1] = c = logical[3] → pMap[1] = 3
    // newCoords[2] = h = logical[1] → pMap[2] = 1
    // newCoords[3] = w = logical[2] → pMap[3] = 2
    SinkExpr permutedSink = permute(bufferSink(), {0, 3, 1, 2});
    
    Conv2DKernel kernelWithPermute(
        ctx, ElementType::Float32, 16, kernelSize, stride, inChannels, outChannels,
        buffer(), kernelOutput(), permutedSink);
    kernelWithPermute.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Reference kernel with default sink
    Conv2DKernel kernelNHWC(ctx, 16, kernelSize, stride, inChannels, outChannels);
    kernelNHWC.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    {
        auto task = ctx->createTask();
        kernelWithPermute.queueExecute(task, outputNCHW, inputBuffer->getView(), padding);
        kernelNHWC.queueExecute(task, outputNHWC, inputBuffer->getView(), padding);
        task.execute();
    }

    auto nchwData = ctx->readBuffer<float>(outputNCHW);
    auto nhwcData = ctx->readBuffer<float>(outputNHWC);

    // Verify permutation: NCHW[n,c,h,w] should equal NHWC[n,h,w,c]
    float maxDiff = 0.0f;
    for (int n = 0; n < batchSize; n++)
    {
        for (int c = 0; c < outChannels; c++)
        {
            for (int h = 0; h < outputH; h++)
            {
                for (int w = 0; w < outputW; w++)
                {
                    int nchwIdx = ((n * outChannels + c) * outputH + h) * outputW + w;
                    int nhwcIdx = ((n * outputH + h) * outputW + w) * outChannels + c;
                    float diff = fabs(nchwData[nchwIdx] - nhwcData[nhwcIdx]);
                    if (diff > maxDiff)
                        maxDiff = diff;
                }
            }
        }
    }

    TEST_CHECK("conv2d_permute_sink_correct", maxDiff < 1e-5f);

    MLKL_TEST_OK();
}

// ============================================================================
// Test Conv2D with fused residual addition in outputExpr
// This tests the scenario: outputExpr = kernelOutput() + buffer()
// The residual buffer is added to the conv output in a fused manner.
// This should use flat/tiled kernels (not Im2Col) because outputExpr has buffer nodes.
// ============================================================================

SlangResult testConv2DWithFusedResidual(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    const int batchSize = 1;
    const int inputH = 8;
    const int inputW = 8;
    const int inChannels = 4;
    const int outChannels = 8;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;

    const int outputH = inputH;  // Same padding
    const int outputW = inputW;

    // Create input [N, H, W, C]
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inChannels);
    for (Index i = 0; i < inputData.getCount(); i++)
        inputData[i] = (float)(i % 17) * 0.1f - 0.8f;

    auto inputBuffer = ctx->createTensor(
        ElementType::Float32, Shape(batchSize, inputH, inputW, inChannels), inputData);

    // Create residual buffer [N, H, W, outChannels] - same shape as output
    List<float> residualData;
    residualData.setCount(batchSize * outputH * outputW * outChannels);
    for (Index i = 0; i < residualData.getCount(); i++)
        residualData[i] = (float)(i % 11) * 0.05f;  // Different pattern from input

    auto residualBuffer = ctx->createTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), residualData);

    // Weights and bias
    List<float> weights;
    weights.setCount(outChannels * kernelSize * kernelSize * inChannels);
    List<float> biases;
    biases.setCount(outChannels);
    for (Index i = 0; i < weights.getCount(); i++)
        weights[i] = (float)(i % 13) * 0.05f - 0.3f;
    for (int i = 0; i < outChannels; i++)
        biases[i] = (float)i * 0.1f;

    // Output buffers
    auto outputFused = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "output_fused");
    auto outputSeparate = ctx->allocScratchTensor(
        ElementType::Float32, Shape(batchSize, outputH, outputW, outChannels), "output_separate");

    // Kernel with fused residual: output = conv(input) + residual
    // outputExpr = kernelOutput() + buffer(), where buffer() is the residual
    Conv2DKernel kernelFused(
        ctx, ElementType::Float32, 16, kernelSize, stride, inChannels, outChannels,
        buffer(), kernelOutput() + buffer(), bufferSink());
    kernelFused.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Reference kernel without fusion
    Conv2DKernel kernelPlain(ctx, 16, kernelSize, stride, inChannels, outChannels);
    kernelPlain.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // Run fused kernel: takes {input, residual} because outputExpr has a buffer()
    {
        auto task = ctx->createTask();
        kernelFused.queueExecute(task, outputFused, {inputBuffer->getView(), residualBuffer->getView()}, padding);
        task.execute();
    }

    // Run plain kernel and manually add residual
    {
        auto task = ctx->createTask();
        kernelPlain.queueExecute(task, outputSeparate, inputBuffer->getView(), padding);
        task.execute();
    }

    // Read results
    auto fusedData = ctx->readBuffer<float>(outputFused);
    auto separateData = ctx->readBuffer<float>(outputSeparate);

    // Verify: fusedData[i] should equal separateData[i] + residualData[i]
    float maxDiff = 0.0f;
    for (Index i = 0; i < fusedData.getCount(); i++)
    {
        float expected = separateData[i] + residualData[i];
        float diff = fabs(fusedData[i] - expected);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    TEST_CHECK("conv2d_fused_residual_correct", maxDiff < 1e-5f);

    MLKL_TEST_OK();
}
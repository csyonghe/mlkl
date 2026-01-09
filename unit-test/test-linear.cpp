#include "linear.h"
#include "test-kernels.h"

SlangResult testLinear(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int batchSize = 4;
    const int inDim = 16;
    const int outDim = 8;

    // Tiling config (using small tiles to force multiple iterations)
    const int tM = 2;
    const int tN = 4;
    const int tK = 4;

    // Define Expression Tree
    // Input: Standard buffer pull
    Expr input = buffer();
    // Output Value: Identity (no ReLU for this test)
    Expr valueTransform = kernelOutput();

    // Sink: Permute the [Batch, Out] result to [Out, Batch]
    SinkExpr outputSink = permute(bufferSink(), {1, 0});

    // Initialize Kernel
    LinearKernel kernel(context, input, valueTransform, outputSink, inDim, outDim, tM, tN, tK);

    // Prepare Mock Weights and Bias
    // Weights: [Out, In] -> [8, 16]
    List<float> hWeights;
    hWeights.setCount(outDim * inDim);
    for (Index i = 0; i < hWeights.getCount(); ++i)
        hWeights[i] = (float)i * 0.01f;
    kernel.weightsBuffer = context->createPersistentBuffer(hWeights);

    // Bias: [Out] -> [8]
    List<float> hBias;
    hBias.setCount(outDim);
    for (Index i = 0; i < outDim; ++i)
        hBias[i] = (float)i * 0.5f;
    kernel.biasesBuffer = context->createPersistentBuffer(hBias);

    // Prepare Input Data [Batch, In]
    List<float> hInput;
    hInput.setCount(batchSize * inDim);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)i * 0.1f;
    auto inputBuffer = context->createTensor(ElementType::Float32, Shape(batchSize, inDim), hInput);

    // Allocate Output Buffer
    // Note: Because of PermuteSink {1, 0}, the physical shape is [Out, Batch]
    auto outputBuffer = context->allocScratchTensor(ElementType::Float32, Shape(outDim, batchSize));

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference Calculation
    List<float> hPhysicalOutput;
    hPhysicalOutput.setCount(batchSize * outDim); // What we expect in memory

    for (int b = 0; b < batchSize; ++b)
    {
        for (int o = 0; o < outDim; ++o)
        {
            float sum = 0.0f;
            for (int i = 0; i < inDim; ++i)
            {
                sum += hInput[b * inDim + i] * hWeights[o * inDim + i];
            }
            sum += hBias[o];

            // Map logical (b, o) to physical (o, b) because of PermuteSink {1, 0}
            int physicalIdx = o * batchSize + b;
            hPhysicalOutput[physicalIdx] = sum;
        }
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < hPhysicalOutput.getCount(); i++)
    {
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hPhysicalOutput[i]));
    }

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }
    MLKL_TEST_OK();
}

SlangResult testLinearHalf(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int batchSize = 4;
    const int inDim = 16;
    const int outDim = 8;

    // Tiling config
    const int tM = 2;
    const int tN = 4;
    const int tK = 4;

    // Define Expression Tree
    Expr input = buffer();
    Expr valueTransform = kernelOutput();
    SinkExpr outputSink = bufferSink();

    // Initialize Kernel with Float16
    LinearKernel kernel(
        context,
        ElementType::Float16,
        input,
        valueTransform,
        outputSink,
        inDim,
        outDim,
        tM,
        tN,
        tK);

    // Prepare Mock Weights and Bias - convert to half precision
    List<float> hWeights;
    hWeights.setCount(outDim * inDim);
    for (Index i = 0; i < hWeights.getCount(); ++i)
        hWeights[i] = (float)i * 0.01f;
    // Convert weights to half precision
    auto weightsHalf = convertFloatData(hWeights, ElementType::Float16);
    kernel.weightsBuffer =
        context->createPersistentBuffer(weightsHalf.getBuffer(), weightsHalf.getCount());

    List<float> hBias;
    hBias.setCount(outDim);
    for (Index i = 0; i < outDim; ++i)
        hBias[i] = (float)i * 0.5f;
    // Convert bias to half precision
    auto biasHalf = convertFloatData(hBias, ElementType::Float16);
    kernel.biasesBuffer =
        context->createPersistentBuffer(biasHalf.getBuffer(), biasHalf.getCount());

    // Prepare Input Data [Batch, In] in float, then convert to half
    List<float> hInput;
    hInput.setCount(batchSize * inDim);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)i * 0.1f;

    List<uint16_t> hInputHalf;
    floatToHalf(hInput, hInputHalf);

    auto inputBuffer = context->createTensor(
        ElementType::Float16,
        Shape(batchSize, inDim),
        hInputHalf.getCount() * sizeof(uint16_t),
        hInputHalf.getBuffer());

    // Allocate Output Buffer
    auto outputBuffer = context->allocScratchTensor(ElementType::Float16, Shape(batchSize, outDim));

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference Calculation
    List<float> hExpected;
    hExpected.setCount(batchSize * outDim);

    for (int b = 0; b < batchSize; ++b)
    {
        for (int o = 0; o < outDim; ++o)
        {
            float sum = 0.0f;
            for (int i = 0; i < inDim; ++i)
            {
                sum += hInput[b * inDim + i] * hWeights[o * inDim + i];
            }
            sum += hBias[o];
            hExpected[b * outDim + o] = sum;
        }
    }

    // Verify
    if (!checkOutputHalf(context, outputBuffer, hExpected))
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        return SLANG_FAIL;
    }
    MLKL_TEST_OK();
}

SlangResult testLinearInt(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int batchSize = 4;
    const int inDim = 16;
    const int outDim = 8;

    // Tiling config
    const int tM = 2;
    const int tN = 4;
    const int tK = 4;

    // Define Expression Tree
    Expr input = buffer();
    Expr valueTransform = kernelOutput();
    SinkExpr outputSink = bufferSink();

    // Initialize Kernel with Int32
    LinearKernel kernel(
        context,
        ElementType::Int32,
        input,
        valueTransform,
        outputSink,
        inDim,
        outDim,
        tM,
        tN,
        tK);

    // Prepare Mock Weights and Bias - using small integer values
    List<float> hWeights;
    hWeights.setCount(outDim * inDim);
    for (Index i = 0; i < hWeights.getCount(); ++i)
        hWeights[i] = (float)(i % 5); // Small integers 0-4
    kernel.weightsBuffer =
        context->createPersistentBuffer(convertFloatData(hWeights, ElementType::Int32));

    List<float> hBias;
    hBias.setCount(outDim);
    for (Index i = 0; i < outDim; ++i)
        hBias[i] = (float)(i * 10); // 0, 10, 20, ...
    kernel.biasesBuffer =
        context->createPersistentBuffer(convertFloatData(hBias, ElementType::Int32));

    // Prepare Input Data [Batch, In] - using small integer values
    List<float> hInput;
    hInput.setCount(batchSize * inDim);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)(i % 3); // Small integers 0-2

    List<int32_t> hInputInt;
    floatToInt(hInput, hInputInt);

    auto inputBuffer = context->createTensor(
        ElementType::Int32,
        Shape(batchSize, inDim),
        hInputInt.getCount() * sizeof(int32_t),
        hInputInt.getBuffer());

    // Allocate Output Buffer
    auto outputBuffer = context->allocScratchTensor(ElementType::Int32, Shape(batchSize, outDim));

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // CPU Reference Calculation (integer arithmetic)
    List<float> hExpected;
    hExpected.setCount(batchSize * outDim);

    for (int b = 0; b < batchSize; ++b)
    {
        for (int o = 0; o < outDim; ++o)
        {
            int32_t sum = 0;
            for (int i = 0; i < inDim; ++i)
            {
                sum += (int32_t)hInput[b * inDim + i] * (int32_t)hWeights[o * inDim + i];
            }
            sum += (int32_t)hBias[o];
            hExpected[b * outDim + o] = (float)sum;
        }
    }

    // Verify
    if (!checkOutputInt(context, outputBuffer, hExpected))
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        return SLANG_FAIL;
    }
    MLKL_TEST_OK();
}

SlangResult testLinearPartitioned(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Parameters
    const int batchSize = 2;     // Rows (M)
    const int inDim = 16;        // Input features (K)
    const int outDim = 8;        // Logical output width (N)
    const int partitionSize = 4; // Size of D (e.g., K-dim and V-dim)
    const int numPartitions = 2; // K and V

    // 1. Define the Sink Tree
    // Logical result is [Batch, OutDim] -> [2, 8]
    // Step A: Partitioning Dim 1 (OutDim) into 2 parts of size 4.
    //        This "lifts" the rank: [Batch, OutDim] -> [Partition, Batch, LocalOut]
    //        New Indices: [0: PartitionIndex, 1: BatchIndex, 2: LocalOutIndex]
    // Step B: Permute to transpose the local matrix within each partition.
    //        We want [Partition, LocalOut, BatchIndex], so we swap 1 and 2.

    SinkExpr leaf = bufferSink();
    SinkExpr transposed = permute(leaf, {0, 2, 1}); // Swaps M and S within the partition
    SinkExpr outputSink = partition(transposed, 1, numPartitions);

    // Initialize Kernel
    // We use small tiles to ensure tiling logic works with the sink.
    LinearKernel kernel(context, buffer(), kernelOutput(), outputSink, inDim, outDim, 2, 4, 4);

    // 2. Prepare Mock Weights and Bias
    List<float> hWeights;
    hWeights.setCount(outDim * inDim);
    for (Index i = 0; i < hWeights.getCount(); ++i)
        hWeights[i] = (float)i * 0.01f;
    kernel.weightsBuffer = context->createPersistentBuffer(hWeights);

    List<float> hBias;
    hBias.setCount(outDim);
    for (Index i = 0; i < outDim; ++i)
        hBias[i] = (float)i * 0.5f;
    kernel.biasesBuffer = context->createPersistentBuffer(hBias);

    // 3. Prepare Input Data [Batch, In]
    List<float> hInput;
    hInput.setCount(batchSize * inDim);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = (float)i * 0.1f;
    auto inputBuffer = context->createTensor(ElementType::Float32, Shape(batchSize, inDim), hInput);

    // 4. Resolve Physical Shape and Allocate
    // This calls our recursive resolvePhysicalShape:
    // [2, 8] -> Partition -> [2, 2, 4] -> Permute -> [2, 4, 2]
    Shape logicalShape = {batchSize, outDim};
    Shape physicalShape = outputSink.node->resolvePhysicalShape(logicalShape);

    auto outputBuffer = context->allocScratchTensor(ElementType::Float32, physicalShape);

    // 5. Execute
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView());
    task.execute();

    // 6. CPU Reference Calculation
    List<float> hPhysicalOutput;
    hPhysicalOutput.setCount((Index)physicalShape.getElementCount());

    for (int b = 0; b < batchSize; ++b)
    {
        for (int o = 0; o < outDim; ++o)
        {
            float sum = 0.0f;
            for (int i = 0; i < inDim; ++i)
                sum += hInput[b * inDim + i] * hWeights[o * inDim + i];
            sum += hBias[o];

            // Logical to Physical Mapping Logic:
            int partitionIdx = o / partitionSize;
            int localO = o % partitionSize;

            // Physical Layout: [Partition, localO, b]
            // Size of one partition = partitionSize * batchSize
            int physicalIdx =
                (partitionIdx * (partitionSize * batchSize)) + (localO * batchSize) + b;

            hPhysicalOutput[physicalIdx] = sum;
        }
    }

    // 7. Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < hPhysicalOutput.getCount(); i++)
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hPhysicalOutput[i]));

    if (maxDiff > 1e-3f)
    {
        printf("[FAILED] %s: GPU and CPU results diverge.\n", __func__);
        printf("  - Max Difference: %e\n", maxDiff);
        return SLANG_FAIL;
    }

    MLKL_TEST_OK();
}

// =============================================================================
// Tests for Gemv vs Tiled algorithm selection
// =============================================================================

// Helper to test linear with a specific algorithm
SlangResult testLinearWithAlgorithm(
    InferencingContext* context,
    int batchSize,
    LinearAlgorithm algorithm,
    const char* testName)
{
    const int inDim = 256; // Large enough to exercise the kernel
    const int outDim = 128;

    // Small tiles to force multiple iterations
    const int tM = 8;
    const int tN = 32;
    const int tK = 16;

    Expr input = buffer();
    Expr valueTransform = kernelOutput();
    SinkExpr outputSink = bufferSink();

    LinearKernel kernel(context, input, valueTransform, outputSink, inDim, outDim, tM, tN, tK);

    // Prepare weights
    List<float> hWeights;
    hWeights.setCount(outDim * inDim);
    for (Index i = 0; i < hWeights.getCount(); ++i)
        hWeights[i] = ((i * 17 + 13) % 200 - 100) * 0.001f; // Pseudo-random small values
    kernel.weightsBuffer = context->createPersistentBuffer(hWeights);

    // Prepare bias
    List<float> hBias;
    hBias.setCount(outDim);
    for (Index i = 0; i < outDim; ++i)
        hBias[i] = (float)i * 0.01f;
    kernel.biasesBuffer = context->createPersistentBuffer(hBias);

    // Prepare input
    List<float> hInput;
    hInput.setCount(batchSize * inDim);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = ((i * 23 + 7) % 100 - 50) * 0.01f;
    auto inputBuffer = context->createTensor(ElementType::Float32, Shape(batchSize, inDim), hInput);

    // Allocate output
    auto outputBuffer = context->allocScratchTensor(ElementType::Float32, Shape(batchSize, outDim));

    // Execute with specified algorithm
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView(), algorithm);
    task.execute();

    // CPU reference
    List<float> hExpected;
    hExpected.setCount(batchSize * outDim);

    for (int b = 0; b < batchSize; ++b)
    {
        for (int o = 0; o < outDim; ++o)
        {
            float sum = 0.0f;
            for (int i = 0; i < inDim; ++i)
                sum += hInput[b * inDim + i] * hWeights[o * inDim + i];
            sum += hBias[o];
            hExpected[b * outDim + o] = sum;
        }
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer);
    float maxDiff = 0.0f;
    int mismatchCount = 0;
    for (Index i = 0; i < hExpected.getCount(); i++)
    {
        float diff = std::abs(gpuResult[i] - hExpected[i]);
        maxDiff = std::max(maxDiff, diff);
        if (diff > 1e-3f)
            mismatchCount++;
    }

    TEST_CHECK(testName, maxDiff < 1e-3f);
    return SLANG_OK;
}

SlangResult testLinearGemvBatch1(InferencingContext* context)
{
    MLKL_TEST_BEGIN();
    SLANG_RETURN_ON_FAIL(testLinearWithAlgorithm(context, 1, LinearAlgorithm::Gemv, "gemv_batch1"));
    MLKL_TEST_OK();
}

SlangResult testLinearGemvBatch4(InferencingContext* context)
{
    MLKL_TEST_BEGIN();
    SLANG_RETURN_ON_FAIL(testLinearWithAlgorithm(context, 4, LinearAlgorithm::Gemv, "gemv_batch4"));
    MLKL_TEST_OK();
}

SlangResult testLinearGemvBatch8(InferencingContext* context)
{
    MLKL_TEST_BEGIN();
    SLANG_RETURN_ON_FAIL(testLinearWithAlgorithm(context, 8, LinearAlgorithm::Gemv, "gemv_batch8"));
    MLKL_TEST_OK();
}

SlangResult testLinearTiledBatch1(InferencingContext* context)
{
    MLKL_TEST_BEGIN();
    SLANG_RETURN_ON_FAIL(
        testLinearWithAlgorithm(context, 1, LinearAlgorithm::Tiled, "tiled_batch1"));
    MLKL_TEST_OK();
}

SlangResult testLinearTiledBatch4(InferencingContext* context)
{
    MLKL_TEST_BEGIN();
    SLANG_RETURN_ON_FAIL(
        testLinearWithAlgorithm(context, 4, LinearAlgorithm::Tiled, "tiled_batch4"));
    MLKL_TEST_OK();
}

SlangResult testLinearTiledBatch16(InferencingContext* context)
{
    MLKL_TEST_BEGIN();
    SLANG_RETURN_ON_FAIL(
        testLinearWithAlgorithm(context, 16, LinearAlgorithm::Tiled, "tiled_batch16"));
    MLKL_TEST_OK();
}

SlangResult testLinearAutoSelection(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Test Auto algorithm - should select Gemv for batch<=8, Tiled for batch>8
    SLANG_RETURN_ON_FAIL(testLinearWithAlgorithm(context, 1, LinearAlgorithm::Auto, "auto_batch1"));
    SLANG_RETURN_ON_FAIL(testLinearWithAlgorithm(context, 8, LinearAlgorithm::Auto, "auto_batch8"));
    SLANG_RETURN_ON_FAIL(
        testLinearWithAlgorithm(context, 16, LinearAlgorithm::Auto, "auto_batch16"));

    MLKL_TEST_OK();
}

SlangResult testLinearGemvLargeK(InferencingContext* context)
{
    MLKL_TEST_BEGIN();

    // Test with large K dimension (SD linear layers have K up to 5120)
    const int batchSize = 2;
    const int inDim = 2048;
    const int outDim = 768;

    Expr input = buffer();
    LinearKernel kernel(context, input, kernelOutput(), bufferSink(), inDim, outDim);

    // Prepare weights
    List<float> hWeights;
    hWeights.setCount(outDim * inDim);
    for (Index i = 0; i < hWeights.getCount(); ++i)
        hWeights[i] = ((i * 17 + 13) % 200 - 100) * 0.0001f;
    kernel.weightsBuffer = context->createPersistentBuffer(hWeights);

    // Prepare bias
    List<float> hBias;
    hBias.setCount(outDim);
    for (Index i = 0; i < outDim; ++i)
        hBias[i] = (float)i * 0.001f;
    kernel.biasesBuffer = context->createPersistentBuffer(hBias);

    // Prepare input
    List<float> hInput;
    hInput.setCount(batchSize * inDim);
    for (Index i = 0; i < hInput.getCount(); ++i)
        hInput[i] = ((i * 23 + 7) % 100 - 50) * 0.01f;
    auto inputBuffer = context->createTensor(ElementType::Float32, Shape(batchSize, inDim), hInput);

    // Allocate output
    auto outputBuffer = context->allocScratchTensor(ElementType::Float32, Shape(batchSize, outDim));

    // Execute with Gemv
    auto task = context->createTask();
    kernel.queueExecute(task, outputBuffer, inputBuffer->getView(), LinearAlgorithm::Gemv);
    task.execute();

    // CPU reference
    List<float> hExpected;
    hExpected.setCount(batchSize * outDim);

    for (int b = 0; b < batchSize; ++b)
    {
        for (int o = 0; o < outDim; ++o)
        {
            float sum = 0.0f;
            for (int i = 0; i < inDim; ++i)
                sum += hInput[b * inDim + i] * hWeights[o * inDim + i];
            sum += hBias[o];
            hExpected[b * outDim + o] = sum;
        }
    }

    // Verify
    auto gpuResult = context->readBuffer<float>(outputBuffer);
    float maxDiff = 0.0f;
    for (Index i = 0; i < hExpected.getCount(); i++)
        maxDiff = std::max(maxDiff, std::abs(gpuResult[i] - hExpected[i]));

    TEST_CHECK(__func__, maxDiff < 1e-2f); // Larger tolerance for large K
    MLKL_TEST_OK();
}
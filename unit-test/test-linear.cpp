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
    auto inputBuffer = context->createPersistentBuffer(hInput);

    // Allocate Output Buffer
    // Note: Because of PermuteSink {1, 0}, the physical shape is [Out, Batch]
    auto outputBuffer = context->allocScratchBuffer(batchSize * outDim * sizeof(float));

    // Execute
    auto task = context->createTask();
    kernel.queueExecute(task, BufferView(outputBuffer), inputBuffer, batchSize);
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
    auto inputBuffer = context->createPersistentBuffer(hInput);

    // 4. Resolve Physical Shape and Allocate
    // This calls our recursive resolvePhysicalShape:
    // [2, 8] -> Partition -> [2, 2, 4] -> Permute -> [2, 4, 2]
    Shape logicalShape = {batchSize, outDim};
    Shape physicalShape = outputSink.node->resolvePhysicalShape(logicalShape);

    auto outputBuffer =
        context->allocScratchBuffer(physicalShape.getElementCount() * sizeof(float));

    // 5. Execute
    auto task = context->createTask();
    kernel.queueExecute(task, BufferView(outputBuffer), inputBuffer, batchSize);
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
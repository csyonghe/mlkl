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
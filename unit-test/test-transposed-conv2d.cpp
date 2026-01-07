#include "kernels.h"
#include "test-kernels.h"

SlangResult testTransposedConv2D(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();
    float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    float convWeights[9] = {0.1, 0.5, 0.2, 0.5, 1.0, 0.5, 0.2, 0.5, 0.4};
    float convBiases[] = {1000.0f};
    auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
    auto inputBuffer = ctx->createTensor(
        ElementType::Float32,
        Shape(1, 5, 5, 1),
        5 * 5 * sizeof(float),
        inputData);
    TransposedConv2DKernel transposedConvKernel =
        TransposedConv2DKernel(ctx, ElementType::Float32, 4, 3, 1, 1, 1);
    auto task = ctx->createTask();
    transposedConvKernel.loadParams(3, 1, convWeights, convBiases);
    auto outputBuffer = transposedConvKernel.allocateResultBuffer(ElementType::Float32, 5, 5, 1, 1);
    transposedConvKernel.queueExecute(task, outputBuffer, inputBuffer->getView(), 1);
    auto readWeight = [&](int x, int y) { return convWeights[y * 3 + x]; };
    renderDocBeginFrame();
    task.execute();
    auto outputData = ctx->readBuffer<float>(outputBuffer);
    renderDocEndFrame();
    float v0 = outputData[0];
    float expectedV0 = readInput(1, 0) * readWeight(1, 1) + readInput(0, 1) * readWeight(1, 0) +
                       readInput(1, 1) * readWeight(0, 0) + convBiases[0];
    TEST_CHECK("simpleTransposedConvolution", fabs(v0 - expectedV0) < 1e-3f);
    MLKL_TEST_OK();
}

SlangResult testTransposedConv2DHalf(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();
    float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    float convWeights[9] = {0.1f, 0.5f, 0.2f, 0.5f, 1.0f, 0.5f, 0.2f, 0.5f, 0.4f};
    float convBiases[] = {1000.0f};
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

    TransposedConv2DKernel transposedConvKernel =
        TransposedConv2DKernel(ctx, ElementType::Float16, 4, 3, 1, 1, 1);
    auto task = ctx->createTask();
    transposedConvKernel.loadParams(3, 1, convWeights, convBiases);
    auto outputBuffer = transposedConvKernel.allocateResultBuffer(ElementType::Float16, 5, 5, 1, 1);
    transposedConvKernel.queueExecute(task, outputBuffer, inputBuffer->getView(), 1);
    auto readWeight = [&](int x, int y) { return convWeights[y * 3 + x]; };
    renderDocBeginFrame();
    task.execute();
    renderDocEndFrame();

    // Read output as half and convert to float
    auto outputDataHalf = ctx->readBuffer<uint16_t>(outputBuffer.getBufferView());
    List<float> outputData;
    halfToFloat(outputDataHalf, outputData);

    float v0 = outputData[0];
    float expectedV0 = readInput(1, 0) * readWeight(1, 1) + readInput(0, 1) * readWeight(1, 0) +
                       readInput(1, 1) * readWeight(0, 0) + convBiases[0];

    // Use larger tolerance for half precision
    TEST_CHECK("simpleTransposedConvolutionHalf", fabs(v0 - expectedV0) < 1.0f);
    MLKL_TEST_OK();
}
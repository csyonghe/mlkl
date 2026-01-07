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
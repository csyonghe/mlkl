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
// Unit test for kernels

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "kernels.h"
#include "torch-reader.h"

#include <chrono>
#include <random>

static const ExampleResources resourceBase("unit-test");

struct UnitTestProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    RefPtr<InferencingContext> gInferencingCtx;

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);
        rhi::DeviceDesc deviceDesc;
        deviceDesc.slang.targetProfile = "spirv_1_6";
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;
        // rhi::getRHI()->enableDebugLayers();
        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
            return SLANG_FAIL;

        gInferencingCtx = new InferencingContext(gDevice);

        SLANG_RETURN_ON_FAIL(testSimpleConvolution());
        SLANG_RETURN_ON_FAIL(testSimpleTransposedConvolution());

        printf("all tests passed!\n");
        return SLANG_OK;
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

#define TEST_CHECK(testName, condition) \
    SLANG_RETURN_ON_FAIL(testCheck((condition), (testName), #condition))

    SlangResult testSimpleConvolution()
    {
        float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = gInferencingCtx->createBuffer(inputData, 5 * 5 * sizeof(float));
        float convWeights[9] = {0.1, 0.5, 0.2, 0.5, 1.0, 0.5, 0.2, 0.5, 0.4};
        float convBiases[] = {1000.0f};
        Conv2DKernel convKernel = Conv2DKernel(gInferencingCtx.Ptr(), 4, 3, 1, 1, 1);
        auto task = gInferencingCtx->createTask();
        convKernel.loadParams(3, 1, convWeights, convBiases);
        auto outputBuffer = convKernel.queueExecute(task, inputBuffer, 5, 5, 1);

        auto readWeight = [&](int x, int y) { return convWeights[y * 3 + x]; };

        renderDocBeginFrame();
        task.execute();
        float outputData[25];
        gDevice->getQueue(rhi::QueueType::Graphics)->waitOnHost();
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();
        float v0 = outputData[0];
        float expectedV0 = readInput(0, 0) * readWeight(1, 1) + readInput(1, 0) * readWeight(2, 1) +
                           readInput(0, 1) * readWeight(2, 1) + readInput(1, 1) * readWeight(2, 2) +
                           convBiases[0];
        TEST_CHECK("simpleConvolution", fabs(v0 - expectedV0) < 1e-3f);
        return SLANG_OK;
    }

    SlangResult testSimpleTransposedConvolution()
    {
        float inputData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
        float convWeights[9] = {0.1, 0.5, 0.2, 0.5, 1.0, 0.5, 0.2, 0.5, 0.4};
        float convBiases[] = {1000.0f};
        auto readInput = [&](int x, int y) { return inputData[y * 5 + x]; };
        auto inputBuffer = gInferencingCtx->createBuffer(inputData, 5 * 5 * sizeof(float));
        TransposedConv2DKernel transposedConvKernel =
            TransposedConv2DKernel(gInferencingCtx.Ptr(), 4, 3, 1, 1, 1);
        auto task = gInferencingCtx->createTask();
        transposedConvKernel.loadParams(3, 1, convWeights, convBiases);
        auto outputBuffer = transposedConvKernel.queueExecute(task, inputBuffer, 5, 5, 1);
        auto readWeight = [&](int x, int y) { return convWeights[y * 3 + x]; };
        renderDocBeginFrame();
        task.execute();
        float outputData[25];
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();
        float v0 = outputData[0];
        float expectedV0 = readInput(1, 0) * readWeight(1, 1) + readInput(0, 1) * readWeight(1, 0) +
                           readInput(1, 1) * readWeight(0, 0) + convBiases[0];
        TEST_CHECK("simpleTransposedConvolution", fabs(v0 - expectedV0) < 1e-3f);
        return SLANG_OK;
    }
};

int main(int argc, char** argv)
{
    UnitTestProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}

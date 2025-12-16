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

        SLANG_RETURN_ON_FAIL(testBroadcastAdd());
        SLANG_RETURN_ON_FAIL(testMaterialize());
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

    SlangResult testBroadcastAdd()
    {
        // Scenario: Add a Bias vector to an Image
        // Batch Size: 1
        // Input A: [2, 3] (Height 2, Width 3) -> 6 elements
        // Input B: [3]    (Width 3)           -> 3 elements
        //
        // Logic: B should be broadcasted to every row of A.

        int batchSize = 1;
        int height = 2;
        int width = 3;

        // 1. Setup Data
        float dataA[] = {0, 1, 2, 10, 11, 12}; // 2x3

        float dataB[] = {100, 200, 300}; // 1x3

        auto bufA = gInferencingCtx->createBuffer(dataA, sizeof(dataA));
        auto bufB = gInferencingCtx->createBuffer(dataB, sizeof(dataB));

        // 2. Prepare Kernel
        BroadcastAddKernel kernel(gInferencingCtx);
        auto task = gInferencingCtx->createTask();

        // Shapes excluding batch dimension
        int shapeA[] = {height, width};
        int shapeB[] = {width};

        // 3. Execute
        // Internally this constructs shapes [1, 2, 3] and [1, 3]
        // And broadcasts B to [1, 2, 3]
        auto outputBuffer = kernel.queueExecute(
            task,
            bufA,
            makeArrayView(shapeA, 2),
            bufB,
            makeArrayView(shapeB, 1),
            batchSize);

        // 4. Readback
        renderDocBeginFrame();
        task.execute();

        float output[6];
        gDevice->getQueue(rhi::QueueType::Graphics)->waitOnHost();
        gDevice->readBuffer(outputBuffer, 0, sizeof(output), output);
        renderDocEndFrame();

        // 5. Verify
        // Row 0: [0+100, 1+200, 2+300] -> [100, 201, 302]
        // Row 1: [10+100, 11+200, 12+300] -> [110, 211, 312]

        float expected[] = {100, 201, 302, 110, 211, 312};

        for (int i = 0; i < 6; i++)
        {
            if (fabs(output[i] - expected[i]) > 1e-3f)
            {
                printf(
                    "BroadcastAdd Mismatch at %d: Got %f, Expected %f\n",
                    i,
                    output[i],
                    expected[i]);
                return SLANG_FAIL;
            }
        }

        return SLANG_OK;
    }

    SlangResult testMaterialize()
    {
        // 1. Setup Input Data
        // Create two 4x4 arrays
        const int count = 16;
        float dataA[count];
        float dataB[count];
        for (int i = 0; i < count; i++)
        {
            dataA[i] = (float)i; // 0, 1, 2...
            dataB[i] = 100.0f;   // 100, 100...
        }

        auto bufA = gInferencingCtx->createBuffer(dataA, count * sizeof(float));
        auto bufB = gInferencingCtx->createBuffer(dataB, count * sizeof(float));

        // 2. Build Expression Tree: (A + B) * 0.5
        auto a = buffer();
        auto b = buffer();
        auto p = a + b;
        // Expression: res = (a + b) * 0.5
        // Expected result[i] = (i + 100) * 0.5
        auto expr = (p * 0.3f + p * 0.7f) * 0.5f;

        // 3. Compile Pipeline
        RefPtr<ElementwiseKernel> kernel = new ElementwiseKernel(gInferencingCtx, expr);

        // 4. Prepare Execution
        auto task = gInferencingCtx->createTask();

        Dictionary<Expr, InputInfo> inputs;
        inputs.add(a, {Shape(4, 4), bufA});
        inputs.add(b, {Shape(4, 4), bufB});

        // 5. Eval
        // Effectively dispatches `materialize<Program<9,
        //  Eval<0, BufferView>,
        //  Eval<1, BufferView>,
        //  Eval<2, Add<Reg<0>,Reg<1>>>,
        //  Eval<3, ConstantView>,
        //  Eval<4, Mul<Reg<2>,Reg<3>>>,
        //  Eval<5, ConstantView>,
        //  Eval<6, Mul<Reg<2>,Reg<5>>>,
        //  Eval<7, Add<Reg<4>,Reg<6>>>,
        //  Eval<8, ConstantView>,
        //  Eval<9, Mul<Reg<7>,Reg<8>>>
        //  >>`.
        auto outputBuffer = kernel->eval(task, inputs);

        // 6. Execute and Readback
        renderDocBeginFrame();
        task.execute();

        float outputData[count];
        gDevice->readBuffer(outputBuffer, 0, sizeof(outputData), outputData);
        renderDocEndFrame();

        // 7. Verify Results
        for (int i = 0; i < count; i++)
        {
            float expected = (dataA[i] + dataB[i]) * 0.5f;
            TEST_CHECK("materialize", fabs(outputData[i] - expected) < 1e-3f);
        }

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

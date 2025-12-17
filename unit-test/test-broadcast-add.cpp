#include "kernels.h"
#include "test-kernels.h"

SlangResult testBroadcastAdd(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

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

    auto bufA = ctx->createPersistentBuffer(dataA, sizeof(dataA));
    auto bufB = ctx->createPersistentBuffer(dataB, sizeof(dataB));

    // 2. Prepare Kernel
    BroadcastAddKernel kernel(ctx);
    auto task = ctx->createTask();

    // Shapes excluding batch dimension
    Shape shapeA = {height, width};
    Shape shapeB = {width};

    // 3. Execute
    // Internally this constructs shapes [1, 2, 3] and [1, 3]
    // And broadcasts B to [1, 2, 3]
    auto output = kernel.allocResultBuffer(shapeA, shapeB, batchSize);
    kernel.queueExecute(task, output, bufA, shapeA, bufB, shapeB, batchSize);

    // 4. Readback
    renderDocBeginFrame();
    task.execute();

    auto outputData = ctx->readBuffer<float>(output);
    renderDocEndFrame();

    // 5. Verify
    // Row 0: [0+100, 1+200, 2+300] -> [100, 201, 302]
    // Row 1: [10+100, 11+200, 12+300] -> [110, 211, 312]

    float expected[] = {100, 201, 302, 110, 211, 312};

    for (int i = 0; i < 6; i++)
    {
        if (fabs(outputData[i] - expected[i]) > 1e-3f)
        {
            printf(
                "BroadcastAdd Mismatch at %d: Got %f, Expected %f\n",
                i,
                outputData[i],
                expected[i]);
            return SLANG_FAIL;
        }
    }

    MLKL_TEST_OK();
}

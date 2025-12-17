#include "kernels.h"
#include "test-kernels.h"

SlangResult testClassifierFreeGuidance(InferencingContext* ctx)
{
    MLKL_TEST_BEGIN();

    // 1. Setup Data
    int width = 2;
    int height = 2;
    int channels = 1;
    int count = width * height * channels; // 4 elements per image

    // We pack both batches into a single buffer
    // [Batch 0 (Uncond), Batch 1 (Cond)]
    float inputData[] = {
        1.0f,
        2.0f,
        3.0f,
        4.0f, // Uncond
        10.0f,
        20.0f,
        30.0f,
        40.0f // Cond
    };

    float guidanceScale = 2.0f;

    auto inputBuf = ctx->createPersistentBuffer(inputData, sizeof(inputData));

    // 2. Prepare Kernel
    ClassifierFreeGuidanceKernel kernel(ctx);
    auto task = ctx->createTask();

    // 3. Execute
    // The kernel should:
    //  - Treat the first 4 floats as 'Uncond'
    //  - Treat the next 4 floats as 'Cond'
    //  - Apply the formula
    auto outputBuffer = kernel.allocResultBuffer(width, height, channels);
    kernel.queueExecute(
        task,
        outputBuffer,
        BufferView(inputBuf),
        width,
        height,
        channels,
        guidanceScale);

    // 4. Readback
    renderDocBeginFrame();
    task.execute();

    auto output = ctx->readBuffer<float>(outputBuffer);
    renderDocEndFrame();

    // 5. Verify
    // Formula: uncond + (cond - uncond) * scale
    float expected[] = {
        19.0f, // 1 + (10-1)*2
        38.0f, // 2 + (20-2)*2
        57.0f, // 3 + (30-3)*2
        76.0f  // 4 + (40-4)*2
    };

    for (int i = 0; i < 4; i++)
    {
        if (fabs(output[i] - expected[i]) > 1e-3f)
        {
            printf("CFG Mismatch at %d: Got %f, Expected %f\n", i, output[i], expected[i]);
            return SLANG_FAIL;
        }
    }

    MLKL_TEST_OK();
}

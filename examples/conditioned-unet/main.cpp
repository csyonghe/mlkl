#include "../shared/ddim-sampler.h"
#include "conditioned-unet.h"
#include "elementwise.h"
#include "example-base.h"
#include "torch-reader.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

using namespace Slang;

static const ExampleResources resourceBase("conditioned-unet");


// ============================================================================
// IMAGE WRITER
// ============================================================================

static void fillRandom(List<float>& list, int count)
{
    static std::mt19937 gen(7391);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    list.setCount(count);
    for (int i = 0; i < count; ++i)
        list[i] = dist(gen);
}

// ============================================================================
// MAIN TEST
// ============================================================================
struct SimpleUNetProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;

    RefPtr<InferencingContext> gInferencingCtx;

    SimpleUNetProgram() { gInferencingCtx = new InferencingContext(); }

    SlangResult testUNetModel(int targetDigit)
    {
        printf("\n=== Running Conditioned UNet Generation ===\n");

        auto ctx = gInferencingCtx.Ptr();

        // 1. Config
        int batchSize = 1;
        int imgSize = 32; // Standard simple-unet size
        int channels = 1; // MNIST is grayscale
        int train_steps = 500;
        int inference_steps = 100;

        // 2. Initialize Model
        // (1 in, 1 out, 32 tDim, 128 cDim, 64 baseCh, 10 classes)
        RefPtr<ConditionedUNet> model =
            new ConditionedUNet(ctx, channels, channels, 32, 128, 64, 10);

        {
            auto weightsPath = resourceBase.resolveResource("model_weights_conditioned.bin");
            if (!File::exists(weightsPath.getBuffer()))
            {
                printf(
                    "Model weights file not found: %s. Please run train.py to generate it first!\n",
                    weightsPath.getBuffer());
                return SLANG_FAIL;
            }
            TorchParamReader reader(weightsPath);
            if (SLANG_FAILED(model->loadParams(reader)))
            {
                printf("Error loading params\n");
                return SLANG_FAIL;
            }
            printf("Model parameters loaded.\n");
        }

        // 3. Prepare Sampler
        RefPtr<DDIMSampler> sampler = new DDIMSampler(ctx, train_steps, inference_steps);

        // 4. Prepare Buffers
        int pixelCount = batchSize * channels * imgSize * imgSize;

        // Initial Noise x_T
        List<float> noiseData;
        fillRandom(noiseData, pixelCount);
        auto bufX = ctx->createPersistentBuffer(noiseData, "x_t");

        // Buffer for x_{t-1} (Ping-Pong)
        auto bufNextX = ctx->createPersistentBuffer(noiseData, "x_prev");

        // Buffer for Predicted Noise (Eps)
        auto bufEps = ctx->allocScratchBuffer(pixelCount * sizeof(float), "eps");

        // Random Noise buffer for the update step (z)
        // We refill this every step on CPU for simplicity, or generate on GPU if we had RNG kernel
        List<float> stepNoiseData;
        stepNoiseData.setCount(pixelCount);
        auto bufZ = ctx->createPersistentBuffer(stepNoiseData, "z"); // Re-used buffer

        // Context Label (Target = 7)
        List<float> labelData;
        labelData.add((float)targetDigit);
        auto bufLabel = ctx->createPersistentBuffer(labelData, "label");


        // 5. Diffusion Loop
        // Loop backwards from T-1 to 0
        printf("Sampling...\n");
        // Loop over INFERENCE steps indices (199 -> 0)
        for (int i = inference_steps - 1; i >= 0; i--)
        {
            // Get the actual training timestep (e.g., 995)
            int t = sampler->timesteps[i];

            auto task = ctx->createTask();

            // 1. Predict Noise
            model->queueExecute(
                task,
                bufEps,   // Output
                bufX,     // Input
                bufLabel, // Condition
                imgSize,
                imgSize,
                t,
                batchSize);

            // 2. DDIM Update (No Random Noise z needed!)
            Shape shape = {batchSize, channels, imgSize, imgSize};
            sampler->step(task, bufNextX, bufX, bufEps, i, shape);

            task.execute();

            // Swap
            std::swap(bufX, bufNextX);
        }

        // 6. Save Result
        // The final result is in bufX (because we swapped after the last write to bufNextX)
        auto finalPixels = ctx->readBuffer<float>(bufX);
        writeImagePNG("result_digit.png", imgSize, imgSize, 1, finalPixels);

        printf("Done.\n");
        return SLANG_OK;
    }
};


int main(int argc, char* argv[])
{
    SimpleUNetProgram program;
    if (SLANG_FAILED(program.testUNetModel(1)))
    {
        return -1;
    }

    return 0;
}
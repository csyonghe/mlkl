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
// HELPERS
// ============================================================================

static void initImage(List<float>& list, int width, int height, int channels)
{
    static std::mt19937 gen(7391);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    list.setCount(width * height * channels);
    for (int i = 0; i < list.getCount(); ++i)
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
        // Linear schedule with default beta range
        DiffusionSchedule schedule(train_steps, 0.0001f, 0.02f, /*scaled_linear=*/false);
        DDIMSampler sampler(ctx, std::move(schedule), inference_steps);

        // 4. Prepare Tensors (NHWC layout)
        List<float> noiseData;
        initImage(noiseData, imgSize, imgSize, channels);

        // Initial Noise x_T
        auto imageAStorage = ctx->createTensor(
            ElementType::Float32,
            Shape(batchSize, imgSize, imgSize, channels),
            noiseData.getCount() * sizeof(float),
            noiseData.getBuffer(),
            "imageA");
        auto imageA = imageAStorage->getView();

        // Buffer for x_{t-1} (Ping-Pong)
        auto imageB = ctx->allocScratchTensor(
            ElementType::Float32,
            Shape(batchSize, imgSize, imgSize, channels),
            "imageB");

        // Buffer for Predicted Noise (Eps)
        auto predictedNoise = ctx->allocScratchTensor(
            ElementType::Float32,
            Shape(batchSize, imgSize, imgSize, channels),
            "predictedNoise");

        // Context Label (Target digit as float for gather kernel)
        List<float> labelData;
        labelData.add((float)targetDigit);
        auto labelStorage = ctx->createTensor(
            ElementType::Float32,
            Shape(batchSize),
            labelData.getCount() * sizeof(float),
            labelData.getBuffer(),
            "classLabel");
        auto classLabel = labelStorage->getView();

        auto outputImage = imageA;

        // 5. Diffusion Loop
        printf("Sampling digit %d...\n", targetDigit);

        ctx->pushAllocScope();
        SLANG_DEFER(ctx->popAllocScope());

        for (int step = 0; step < inference_steps; step++)
        {
            // Get the actual training timestep
            int t = sampler.timesteps[step];

            auto task = ctx->createTask();

            // 1. Predict Noise
            model->queueExecute(task, predictedNoise, imageA, classLabel, t);

            // 2. DDIM Update
            sampler.step(task, imageB, imageA, predictedNoise, step);

            task.execute();

            outputImage = imageB;
            Swap(imageA, imageB);
        }

        // 6. Save Result
        auto finalPixels = ctx->readBuffer<float>(outputImage);
        writeImagePNG("result_digit.png", imgSize, imgSize, channels, finalPixels);

        printf("Done. Saved to result_digit.png\n");
        return SLANG_OK;
    }
};


int main(int argc, char* argv[])
{
    SimpleUNetProgram program;
    if (SLANG_FAILED(program.testUNetModel(4)))
    {
        return -1;
    }

    return 0;
}
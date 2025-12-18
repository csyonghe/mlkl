#include "conditioned-unet.h"
#include "elementwise.h"
#include "example-base.h"
#include "torch-reader.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace Slang;

static const ExampleResources resourceBase("conditioned-unet");

// ============================================================================
// DIFFUSION SCHEDULE
// ============================================================================

struct DiffusionSchedule
{
    std::vector<float> alphas_cumprod;
    int train_steps;

    DiffusionSchedule(int steps = 500, float beta_min = 0.0001f, float beta_max = 0.02f)
        : train_steps(steps)
    {
        float current_alpha_cumprod = 1.0f;
        for (int i = 0; i < steps; i++)
        {
            float beta = beta_min + (beta_max - beta_min) * ((float)i / (steps - 1));
            float alpha = 1.0f - beta;
            current_alpha_cumprod *= alpha;
            alphas_cumprod.push_back(current_alpha_cumprod);
        }
    }
};

// ============================================================================
// ROBUST DDIM SAMPLER (Clipped & Trailing Schedule)
// ============================================================================

class DDIMSampler : public RefObject
{
    InferencingContext* ctx;
    RefPtr<ElementwiseKernel> updateKernel;

    Expr x_t; // Input Image
    Expr eps; // Predicted Noise
    Expr c1;  // Coeff for x_t
    Expr c2;  // Coeff for eps

public:
    DiffusionSchedule schedule;
    std::vector<int> timesteps;
    int inference_steps;

    DDIMSampler(InferencingContext* context, int train_steps, int infer_steps)
        : ctx(context), schedule(train_steps), inference_steps(infer_steps)
    {
        // 1. Improved Schedule: "Trailing" mapping
        // This ensures we always start at the max noise (t=499) and end at t=0.
        // Formula: t = i * (train_max / infer_max)
        timesteps.clear();
        if (infer_steps > 1)
        {
            for (int i = 0; i < infer_steps; i++)
            {
                int t = (int)((float)i / (float)(infer_steps - 1) * (train_steps - 1));
                timesteps.push_back(t);
            }
        }
        else
        {
            timesteps.push_back(train_steps - 1);
        }

        // 2. Build Kernel with x0 Clipping
        // x_{t-1} = c1 * x_t + c2 * eps
        x_t = buffer();
        eps = buffer();
        c1 = uniformConstant();
        c2 = uniformConstant();

        Expr next_x = x_t * c1 + eps * c2;
        updateKernel = new ElementwiseKernel(ctx, next_x);
    }

    // Takes the Loop Index 'i' (from inference_steps-1 down to 0)
    int step(
        InferencingTask& task,
        BufferView out_x_prev,
        BufferView in_x_t,
        BufferView in_eps,
        int index,
        const Shape& shape)
    {
        // 1. Map Index -> Training Timestep
        // Current Step t
        int t = timesteps[index];

        // Previous Step t_prev
        // For the very last step (index 0), we go to -1 (which effectively means alpha=1.0)
        int t_prev = (index > 0) ? timesteps[index - 1] : -1;

        // 2. Get Alphas
        float alpha_bar_t = schedule.alphas_cumprod[t];
        float alpha_bar_prev = (t_prev >= 0) ? schedule.alphas_cumprod[t_prev] : 1.0f;

        float sqrt_alpha_bar_t = std::sqrt(alpha_bar_t);

        // 3. Compute DDIM Coefficients (eta=0)
        // pred_x0 = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
        // dir_xt  = sqrt(1 - alpha_prev) * eps
        // x_prev  = sqrt(alpha_prev) * pred_x0 + dir_xt

        // Re-arranging to: x_prev = C1 * x_t + C2 * eps

        float sqrt_alpha_bar_prev = std::sqrt(alpha_bar_prev);
        float sqrt_one_minus_alpha_bar_t = std::sqrt(1.0f - alpha_bar_t);
        float sqrt_one_minus_alpha_bar_prev = std::sqrt(1.0f - alpha_bar_prev);

        // Coeff 1: Scale x_t
        float val_c1 = sqrt_alpha_bar_prev / sqrt_alpha_bar_t;

        // Coeff 2: Scale eps
        // Term 1 from pred_x0 part: - (sqrt(alpha_prev) * sqrt(1-alpha_t)) / sqrt(alpha_t)
        // Term 2 from dir_xt part:  + sqrt(1-alpha_prev)
        float val_c2 = sqrt_one_minus_alpha_bar_prev -
                       (sqrt_alpha_bar_prev * sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t);

        // 4. Execute
        Dictionary<Expr, InputInfo> inputs;
        inputs.add(x_t, InputInfo(shape, in_x_t));
        inputs.add(eps, InputInfo(shape, in_eps));
        inputs.add(c1, InputInfo(val_c1));
        inputs.add(c2, InputInfo(val_c2));

        updateKernel->eval(task, out_x_prev, inputs);

        return t; // Return actual training time for debug/logging
    }
};

// ============================================================================
// IMAGE WRITER
// ============================================================================

static void fillRandom(List<float>& list, int count)
{
    static std::mt19937 gen(1234);
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

    SlangResult testUNetModel()
    {
        printf("\n=== Running Conditioned UNet Generation ===\n");

        auto ctx = gInferencingCtx.Ptr();

        // 1. Config
        int batchSize = 1;
        int imgSize = 32; // Standard simple-unet size
        int channels = 1; // MNIST is grayscale
        int train_steps = 500;
        int inference_steps = 100;
        int targetDigit = 4;

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
        writeImagePNG("result_digit_7.png", imgSize, imgSize, 1, finalPixels);

        printf("Done.\n");
        return SLANG_OK;
    }
};


int main(int argc, char* argv[])
{
    SimpleUNetProgram program;
    if (SLANG_FAILED(program.testUNetModel()))
    {
        return -1;
    }

    return 0;
}
#pragma once

#include "inference-context.h"
#include "kernels.h"

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
    Slang::List<int> timesteps;
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
                timesteps.add(t);
            }
        }
        else
        {
            timesteps.add(train_steps - 1);
        }

        // 2. Build Kernel with x0
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
        TensorView out_x_prev,
        TensorView in_x_t,
        TensorView in_eps,
        int index)
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
        inputs.add(x_t, in_x_t);
        inputs.add(eps, in_eps);
        inputs.add(c1, InputInfo(val_c1));
        inputs.add(c2, InputInfo(val_c2));

        updateKernel->queueExecute(task, out_x_prev, inputs);

        return t; // Return actual training time for debug/logging
    }
};
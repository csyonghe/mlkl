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

    // beta_schedule: "linear" or "scaled_linear"
    // SD 1.5 uses: steps=1000, beta_start=0.00085, beta_end=0.012, scaled_linear
    DiffusionSchedule(
        int steps,
        float beta_start,
        float beta_end,
        bool scaled_linear = false)
        : train_steps(steps)
    {
        float current_alpha_cumprod = 1.0f;
        
        for (int i = 0; i < steps; i++)
        {
            float t = (float)i / (steps - 1);
            float beta;
            
            if (scaled_linear)
            {
                // scaled_linear: betas = linspace(sqrt(start), sqrt(end))^2
                float sqrt_beta = std::sqrt(beta_start) + t * (std::sqrt(beta_end) - std::sqrt(beta_start));
                beta = sqrt_beta * sqrt_beta;
            }
            else
            {
                // linear: betas = linspace(start, end)
                beta = beta_start + t * (beta_end - beta_start);
            }
            
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
    
    // Standard DDIM step kernel: x_prev = c1 * x_t + c2 * eps
    RefPtr<ElementwiseKernel> updateKernel;
    Expr x_t;   // Input latent
    Expr eps;   // Predicted noise
    Expr c1;    // Coeff for x_t
    Expr c2;    // Coeff for eps
    
    // CFG-fused step kernel: x_prev = c1 * x_t + c2 * (eps_uncond + cfg * (eps_cond - eps_uncond))
    RefPtr<ElementwiseKernel> cfgStepKernel;
    Expr cfg_x_t;        // Input latent
    Expr cfg_eps_uncond; // Unconditional noise prediction
    Expr cfg_eps_cond;   // Conditional noise prediction
    Expr cfg_c1;         // Coeff for x_t
    Expr cfg_c2;         // Coeff for combined eps
    Expr cfg_scale;      // CFG guidance scale

public:
    DiffusionSchedule schedule;
    Slang::List<int> timesteps;
    int inference_steps;

    DDIMSampler(InferencingContext* context, DiffusionSchedule sched, int infer_steps)
        : ctx(context), schedule(std::move(sched)), inference_steps(infer_steps)
    {
        int train_steps = schedule.train_steps;
        
        // Match diffusers DDIM timestep schedule exactly:
        // step_ratio = train_steps // infer_steps
        // timesteps = [0, step_ratio, 2*step_ratio, ..., (infer_steps-1)*step_ratio] reversed
        // For 1000 train, 20 infer: [950, 900, 850, ..., 50, 0]
        timesteps.clear();
        int step_ratio = train_steps / infer_steps;
        for (int i = infer_steps - 1; i >= 0; i--)
        {
            timesteps.add(i * step_ratio);
        }

        // Standard DDIM step kernel: x_{t-1} = c1 * x_t + c2 * eps
        x_t = buffer();
        eps = buffer();
        c1 = uniformConstant();
        c2 = uniformConstant();
        Expr next_x = x_t * c1 + eps * c2;
        updateKernel = new ElementwiseKernel(ctx, next_x);
        
        // CFG-fused step kernel:
        // x_prev = c1 * x_t + c2 * (eps_uncond + cfg_scale * (eps_cond - eps_uncond))
        cfg_x_t = buffer();
        cfg_eps_uncond = buffer();
        cfg_eps_cond = buffer();
        cfg_c1 = uniformConstant();
        cfg_c2 = uniformConstant();
        cfg_scale = uniformConstant();
        Expr cfg_combined = cfg_eps_uncond + cfg_scale * (cfg_eps_cond - cfg_eps_uncond);
        Expr cfg_next_x = cfg_x_t * cfg_c1 + cfg_combined * cfg_c2;
        cfgStepKernel = new ElementwiseKernel(ctx, cfg_next_x);
    }

    // Takes the Loop Index 'i' (from 0 to inference_steps-1)
    // timesteps are stored in descending order: [950, 900, ..., 50, 0]
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

        // Previous Step t_prev (next index in array since timesteps are descending)
        // For the very last step, we go to -1 (which effectively means alpha=1.0)
        int t_prev = (index < inference_steps - 1) ? timesteps[index + 1] : -1;

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
    
    // CFG-fused step: takes batched noise prediction [2, H, W, C] with [uncond, cond]
    // and applies CFG combination + DDIM step in a single kernel launch.
    // noise_pred_batched: [2, H, W, C] where [0] = uncond, [1] = cond
    // guidance_scale: CFG scale (typically 7.5)
    int stepWithCFG(
        InferencingTask& task,
        TensorView out_x_prev,
        TensorView in_x_t,
        TensorView noise_pred_batched,
        float guidance_scale,
        int index)
    {
        // 1. Map Index -> Training Timestep
        int t = timesteps[index];
        int t_prev = (index < inference_steps - 1) ? timesteps[index + 1] : -1;

        // 2. Get Alphas
        float alpha_bar_t = schedule.alphas_cumprod[t];
        float alpha_bar_prev = (t_prev >= 0) ? schedule.alphas_cumprod[t_prev] : 1.0f;

        float sqrt_alpha_bar_t = std::sqrt(alpha_bar_t);
        float sqrt_alpha_bar_prev = std::sqrt(alpha_bar_prev);
        float sqrt_one_minus_alpha_bar_t = std::sqrt(1.0f - alpha_bar_t);
        float sqrt_one_minus_alpha_bar_prev = std::sqrt(1.0f - alpha_bar_prev);

        // 3. Compute DDIM Coefficients
        float val_c1 = sqrt_alpha_bar_prev / sqrt_alpha_bar_t;
        float val_c2 = sqrt_one_minus_alpha_bar_prev -
                       (sqrt_alpha_bar_prev * sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t);

        // 4. Slice batched noise prediction into uncond [0] and cond [1]
        Shape singleShape = out_x_prev.shape;
        int elementCount = singleShape.getElementCount();
        
        TensorView eps_uncond = noise_pred_batched;
        eps_uncond.shape = singleShape;
        
        TensorView eps_cond = noise_pred_batched;
        eps_cond.bufferView.offset += elementCount * sizeof(float);
        eps_cond.shape = singleShape;

        // 5. Execute fused CFG + DDIM step
        Dictionary<Expr, InputInfo> inputs;
        inputs.add(cfg_x_t, in_x_t);
        inputs.add(cfg_eps_uncond, eps_uncond);
        inputs.add(cfg_eps_cond, eps_cond);
        inputs.add(cfg_c1, InputInfo(val_c1));
        inputs.add(cfg_c2, InputInfo(val_c2));
        inputs.add(cfg_scale, InputInfo(guidance_scale));

        cfgStepKernel->queueExecute(task, out_x_prev, inputs);

        return t;
    }
};
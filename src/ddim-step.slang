implementing mlkl;

// DDIM Sampler Shader
// Supports jumping arbitrary step sizes.
struct DDIMStepParams
{
    float* currentImage;   // x_t
    float* predictedNoise; // epsilon_theta
    float* outputImage;    // x_{t-1} (Target)

    // DDIM Parameters
    float alphaBar_t;      // Cumulative Alpha at current step
    float alphaBar_prev;   // Cumulative Alpha at NEXT step (target)
    
    uint totalElements;
};

[numthreads(256, 1, 1)]
void ddimStep(
    ConstantBuffer<DDIMStepParams> params,
    uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;
    if (idx >= params.totalElements) return;

    // 1. Load Inputs
    float x_t = params.currentImage[idx];
    float eps = params.predictedNoise[idx];

    // 2. Calculate Derived Constants
    // alphaBar_t corresponds to the current noise level
    // alphaBar_prev corresponds to the target noise level
    
    // Calculate "Predicted x_0" (The estimated clean image)
    // Formula: (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
    float sqrt_one_minus_alpha = sqrt(1.0 - params.alphaBar_t);
    float pred_x0 = (x_t - sqrt_one_minus_alpha * eps) / sqrt(params.alphaBar_t);

    // 3. Calculate "Direction to x_t" (The noise update)
    // For deterministic DDIM (eta = 0), sigma is 0.
    // Formula: sqrt(1 - alpha_bar_prev) * eps
    float dir_xt = sqrt(1.0 - params.alphaBar_prev) * eps;

    // 4. Combine to get x_{t-1}
    // Formula: sqrt(alpha_bar_prev) * pred_x0 + dir_xt
    float x_prev = sqrt(params.alphaBar_prev) * pred_x0 + dir_xt;

    params.outputImage[idx] = x_prev;
}
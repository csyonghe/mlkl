module diffusion;

__include embedding;
__include activations;
__include linear;
__include convolution;
__include concat;
__include broadcast_add;

// Simple hash function to generate deterministic noise per pixel
// Input: Thread ID + Global Seed
// Output: Random float in range [0, 1] (or Gaussian roughly)
uint hash(uint s)
{
    s ^= 2747636419u;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    return s;
}

// Box-Muller Transform: Uniform(0,1) -> Gaussian(0,1)
float getGaussianNoise(uint tid, uint seed)
{
    // Generate two uniform random numbers
    uint h1 = hash(tid + seed);
    uint h2 = hash(h1);
    
    // Convert to float [0, 1)
    // 2.3283064365386963e-10 is 1.0 / 2^32
    float u1 = float(h1) * 2.3283064365386963e-10; 
    float u2 = float(h2) * 2.3283064365386963e-10;

    // Avoid log(0)
    u1 = max(u1, 1e-6);

    // Box-Muller formula
    // We only need one sample, so we just use the Cosine part
    float radius = sqrt(-2.0 * log(u1));
    float theta = 6.28318530718 * u2; // 2 * PI
    
    return radius * cos(theta);
}

struct DiffusionStepParams
{
    // Raw Pointers (Buffer Device Address)
    float* currentImage;   // x_t (Input)
    float* predictedNoise; // epsilon_theta (Input)
    float* outputImage;    // x_{t-1} (Output)
    
    // Scalar Coefficients
    float coeff1; 
    float coeff2; 
    float coeff3; 
    
    uint totalElements;
    uint seed;             // Change this every frame/step on CPU!
};

[numthreads(256, 1, 1)]
void diffusionReverseStep(
    ConstantBuffer<DiffusionStepParams> params,
    uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;

    // Bounds Check
    if (idx >= params.totalElements) return;

    // 1. Generate Noise On-The-Fly (Save VRAM Bandwidth)
    // Only generate if coeff3 > 0 (i.e., not the final timestep)
    float z = 0.0;
    if (params.coeff3 > 0.0) 
    {
        z = getGaussianNoise(idx, params.seed);
    }

    // 2. Load Data from Pointers
    float x_t = params.currentImage[idx];
    float eps = params.predictedNoise[idx];

    // 3. The Formula
    // x_prev = c1 * (x_t - c2 * eps) + c3 * z
    
    float denoised = x_t - (params.coeff2 * eps);
    float scaled = params.coeff1 * denoised;
    float result = scaled + (params.coeff3 * z);

    // 4. Write Output
    params.outputImage[idx] = result;
}

/*
interface ITensor<T, int D>
{
    int getSize(int dim);
    T getValue<each I>(I indices) where I == int;
}
struct ResultTensor<T, int D> : ITensor<T, D>
{
    static T result;
    int size[D];
    int getSize(int dim) { return size[dim]; }

    static void setResult(T val) { result = val; }
    T getValue<each I>(I indices) where I == int { return result; }
}

int assignIf(bool cond, inout int target, int val)
{
    if (cond) target = val;
    return target;
}

int getIndex<each I>(int dim, I indices) where I == int
{
    int i = 0;
    int result = 0;
    expand assignIf(i++ == dim, result, each indices);
    return result;
}

struct ConcatTensor<T, int D, Tensor1, Tensor2, int concatDim> 
    : ITensor<T, D>
    where Tensor1 : ITensor<T, D>
    where Tensor2 : ITensor<T, D>
{
    Tensor1 tensor1;
    Tensor2 tensor2;

    int getSize(int dim) 
    { 
        if (dim == concatDim)
            return tensor1.getSize(dim) + tensor2.getSize(dim);
        else
            return tensor1.getSize(dim); 
    }

    T getValue<each I>(I indices) where I == int
    {
        int indexInConcatDim = getIndex(concatDim, indices);
        if (indexInConcatDim < tensor1.getSize(concatDim))
        {
            return tensor1.getValue(expand each indices);
        }
        else
        {
            let adjustedIndex = indexInConcatDim - tensor1.getSize(concatDim);
            int idx = 0;
            return tensor2.getValue(expand (select<int>(idx++ == concatDim, adjustedIndex, each indices)));
        }
    }
}
*/
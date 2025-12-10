implementing diffusion;

struct LinearLayerParams
{
    // Raw Pointers (mapped to Buffer Device Addresses in CUDA/Vulkan)
    float* weights;      // Layout: [inDim, outDim] (Transposed!)
    float* biases; 
    float* inputVector; 
    float* outputVector; 
    uint inDim;   // Input Features
    uint outDim;  // Output Features
};

[numthreads(tileSize, 1, 1)]
void linearLayer<int tileSize, TActivation : IActivation>(
    ConstantBuffer<LinearLayerParams> params, // Params passed via Constant Memory
    uint3 tid : SV_DispatchThreadID)
{
    uint outChannel = tid.x;

    // Bounds Check
    if (outChannel >= params.outDim) return;

    // 1. Initialize with Bias
    // Access pointer directly from the struct
    float sum = params.biases[outChannel];

    // 2. Perform Dot Product
    // We assume weights are [In, Out] for coalescing.
    for (uint i = 0; i < params.inDim; i++)
    {
        uint wIndex = i * params.outDim + outChannel;
        
        // Pointer arithmetic access
        float inVal = params.inputVector[i];
        float w = params.weights[wIndex];

        sum += inVal * w;
    }

    // 3. Apply Generic Activation
    params.outputVector[outChannel] = TActivation.apply(sum);
}
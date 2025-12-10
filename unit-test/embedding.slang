implementing diffusion;

struct TimeEmbeddingParams
{
    float* output; // Layout: [OutputDim]
    float* weights; // Layout: [EmbeddingDim * OutputDim] (Row-Major: In x Out)
    float* biases; // [OutputDim]
    uint timeStep;
    uint embeddingDim; // The size of the sin/cos vector (InChannels)
    float maxPeriod;   // Default 10000.0
};

// Helper to generate frequency coefficient (shared by sin and cos)
float getFrequency(int freqIdx, int halfDim, float maxPeriod)
{
    float divTerm = float(max(1, halfDim - 1));
    float exponent = -log(maxPeriod) * float(freqIdx) / divTerm;
    return exp(exponent);
}

// Dispatch: (OutputDim, 1, 1)
// Each thread computes exactly one final float value.
[numthreads(outputDim, 1, 1)]
void computeTimeEmbedding<int outputDim>(
    ConstantBuffer<TimeEmbeddingParams, CDataLayout> params,
    uint id : SV_DispatchThreadID)
{
    uint outChannel = id;

    // 1. Load Bias (Initialize Accumulator)
    float sum = params.biases[outChannel];

    // 2. Get Timestep
    float t = float(params.timeStep);

    uint halfDim = params.embeddingDim / 2;

    // 3. The Dot Product Loop
    // We iterate 0..halfDim to handle both Sin and Cos parts efficiently
    for (int i = 0; i < halfDim; i++)
    {
        // A. Generate Input Procedurally
        // We calculate freq once for both Sin and Cos indices
        float freq = getFrequency(i, halfDim, params.maxPeriod);
        float angle = t * freq;
        
        float valSin = sin(angle);
        float valCos = cos(angle);

        // B. Load Weights
        // The weight matrix is [EmbeddingDim, OutputDim]
        // We need weights corresponding to the Sin part and Cos part.
        
        // Index for Sin input (Logical index 'i')
        uint weightIdxSin = i * outputDim + outChannel;
        
        // Index for Cos input (Logical index 'i + halfDim')
        uint weightIdxCos = (i + halfDim) * outputDim + outChannel;

        // C. Accumulate
        sum += valSin * params.weights[weightIdxSin];
        sum += valCos * params.weights[weightIdxCos];
    }

    // 4. Activation (ReLU)
    float result = max(0.0, sum);

    // 5. Write Output
    params.output[outChannel] = result;
}
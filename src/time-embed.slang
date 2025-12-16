implementing mlkl;

struct TimeEmbeddingParams
{
    float* output; // Layout: [Batch OutputDim]
    float* weights; // Layout: [EmbeddingDim * OutputDim] (Row-Major: In x Out)
    float* biases; // [OutputDim]
    uint timeStep;
    uint embeddingDim; // The size of the sin/cos vector (InChannels)
    float maxPeriod;   // Default 10000.0
    int batchSize;
};

// Helper to generate frequency coefficient (shared by sin and cos)
float getFrequency(int freqIdx, int halfDim, float maxPeriod)
{
    float divTerm = float(max(1, halfDim - 1));
    float exponent = -log(maxPeriod) * float(freqIdx) / divTerm;
    return exp(exponent);
}

// Dispatch: (OutputDim, BatchSize, 1)
// X = Feature Dimension
// Y = Batch Dimension
[numthreads(outputDim, 1, 1)] 
void computeTimeEmbedding<int outputDim>(
    ConstantBuffer<TimeEmbeddingParams, CDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint outChannel = id.x;
    uint batchIdx = id.y;

    if (outChannel >= params.embeddingDim || batchIdx >= params.batchSize) return;

    // 1. Initialize
    float sum = params.biases[outChannel];
    float t = float(params.timeStep);
    uint halfDim = params.embeddingDim / 2;
    uint outputDim = params.embeddingDim; // Assuming OutDim == EmbDim for this kernel structure

    // 2. Dot Product
    for (int i = 0; i < halfDim; i++)
    {
        float freq = getFrequency(i, halfDim, params.maxPeriod);
        float angle = t * freq;
        float valSin = sin(angle);
        float valCos = cos(angle);

        // Weights are shared across the batch
        uint weightIdxSin = i * outputDim + outChannel;
        uint weightIdxCos = (i + halfDim) * outputDim + outChannel;

        sum += valSin * params.weights[weightIdxSin];
        sum += valCos * params.weights[weightIdxCos];
    }

    // 3. Activation (ReLU)
    float result = max(0.0, sum);

    // 4. Write Output (Broadcast to specific batch row)
    // Layout: [Batch, OutputDim]
    uint outputIdx = batchIdx * outputDim + outChannel;
    params.output[outputIdx] = result;
}
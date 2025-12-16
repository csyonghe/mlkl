implementing mlkl;

struct LinearLayerParams
{
    // Raw Pointers (mapped to Buffer Device Addresses)
    float* weights;      // [inDim, outDim] (Transposed layout for coalescing)
    float* biases;       // [outDim]
    float* inputVector;  // [batchSize, inDim]
    float* outputVector; // [batchSize, outDim]
    uint inDim;
    uint outDim;
    uint batchSize;
};

// Dispatch: X = OutputDims, Y = BatchSize
[numthreads(tileSize, 1, 1)]
void linearLayer<int tileSize, TActivation : IActivation>(
    ConstantBuffer<LinearLayerParams> params,
    uint3 tid : SV_DispatchThreadID)
{
    uint outChannel = tid.x;
    uint batchIdx = tid.y;

    // Bounds Check
    if (outChannel >= params.outDim || batchIdx >= params.batchSize) return;

    // Calculate Batch Offsets
    // Pointers in Slang support arithmetic for offsets
    // inputVector is flattened [Batch, InDim]
    // outputVector is flattened [Batch, OutDim]
    
    // We point to the start of the vector for this specific batch item
    float* batchInput = params.inputVector + (batchIdx * params.inDim);
    float* batchOutput = params.outputVector + (batchIdx * params.outDim);

    // 1. Initialize with Bias
    float sum = params.biases[outChannel];

    // 2. Perform Dot Product
    // Iterate over input dimension
    for (uint i = 0; i < params.inDim; i++)
    {
        uint wIndex = i * params.outDim + outChannel;
        float inVal = batchInput[i];
        float w = params.weights[wIndex];
        sum += inVal * w;
    }

    // 3. Apply Generic Activation
    batchOutput[outChannel] = TActivation.apply(sum);
}
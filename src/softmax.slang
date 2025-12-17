implementing mlkl;

struct SoftmaxParams
{
    float* input;
    float* output;
    uint stride;   // The size of the dimension we are normalizing (N)
    uint rowCount; // Total number of rows to process
};

// Dispatch: x=RowIndex (from 0 to Batch*Heads*SeqQ - 1)
[numthreads(256, 1, 1)]
void softmax(
    ConstantBuffer<SoftmaxParams> params,
    uint3 id : SV_DispatchThreadID)
{
    uint rowIdx = id.x;
    
    // Bounds Check: Prevent extra threads from accessing memory
    if (rowIdx >= params.rowCount) return;

    // Base pointer for this row
    float* rowInput = params.input + rowIdx * params.stride;
    float* rowOutput = params.output + rowIdx * params.stride;

    // 1. Find Max
    float maxVal = -1e38f; 
    for (uint i = 0; i < params.stride; i++)
    {
        maxVal = max(maxVal, rowInput[i]);
    }

    // 2. Compute Exp and Sum
    float sum = 0.0f;
    for (uint i = 0; i < params.stride; i++)
    {
        float val = exp(rowInput[i] - maxVal);
        rowOutput[i] = val; // Store temporarily
        sum += val;
    }

    // 3. Normalize
    float invSum = 1.0f / sum;
    for (uint i = 0; i < params.stride; i++)
    {
        rowOutput[i] *= invSum;
    }
}
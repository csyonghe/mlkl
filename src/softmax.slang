implementing mlkl;

struct SoftmaxParams<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>
{
    TInput input;
    TSink output;
    uint stride;   // The size of the dimension we are normalizing (N)
    uint rowCount; // Total number of rows to process
};

// Dispatch: x=RowIndex (from 0 to Batch*Heads*SeqQ - 1)
[numthreads(256, 1, 1)]
void softmax<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<SoftmaxParams<T, TInput, TSink>, CDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint rowIdx = id.x;
    
    // Bounds Check: Prevent extra threads from accessing memory
    if (rowIdx >= params.rowCount) return;

    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);

    // 1. Find Max (using float for numerical stability)
    float maxVal = -1e38f; 
    for (uint i = 0; i < params.stride; i++)
    {
        Coord coord = Coord(rowIdx, i);
        float val = params.input.eval(coord, emptyInput).toFloat();
        maxVal = max(maxVal, val);
    }

    // 2. Compute Exp and Sum (in float for stability)
    float sum = 0.0f;
    for (uint i = 0; i < params.stride; i++)
    {
        Coord coord = Coord(rowIdx, i);
        float val = params.input.eval(coord, emptyInput).toFloat();
        float expVal = exp(val - maxVal);
        sum += expVal;
    }

    // 3. Normalize and output
    float invSum = 1.0f / sum;
    for (uint i = 0; i < params.stride; i++)
    {
        Coord coord = Coord(rowIdx, i);
        float val = params.input.eval(coord, emptyInput).toFloat();
        float expVal = exp(val - maxVal);
        T.UnpackedType result = T.UnpackedType.fromFloat(expVal * invSum);
        params.output.store(coord, result);
    }
}

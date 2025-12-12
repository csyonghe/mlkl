implementing mlkl;

struct ConcatParams
{
    float* inputA;
    float* inputB;
    float* output;
    
    // We assume shapeA and shapeB are identical except for 'axis'
    int shapeA[8]; 
    int shapeB[8];
    
    int rank;
    int axis;

    uint innerStride; // Precomputed for efficiency (computed on CPU as follows)
    // innerStride = product of dims after 'axis'

    uint outerStride; // Precomputed for efficiency (computed on CPU as follows)
    // outerStride = product of dims before 'axis'
};

[numthreads(256, 1, 1)]
void concat(
    ConstantBuffer<ConcatParams> params, 
    uint3 id : SV_DispatchThreadID)
{
    uint tid = id.x;

    // -----------------------------------------------------------
    // 1. CALCULATE STRIDES (Can be moved to CPU for optimization)
    // -----------------------------------------------------------
    
    // Inner Stride: Product of all dims AFTER the axis.
    // e.g., if shape is [Batch, Channel, Height, Width] and axis is Channel (1),
    // innerStride = Height * Width.
    uint innerStride = params.innerStride;

    // Get dimensions along the concatenation axis
    uint axisDimA = params.shapeA[params.axis];
    uint axisDimB = params.shapeB[params.axis];
    uint axisDimOut = axisDimA + axisDimB;

    // -----------------------------------------------------------
    // 2. INVERSE MAPPING
    // -----------------------------------------------------------
    
    // We interpret the Output Tensor as flattened into 3 sections:
    // [Outer, Axis, Inner]
    //
    // tid = outerIdx * (axisDimOut * innerStride) 
    //     + axisIdx  * innerStride 
    //     + innerIdx

    uint innerIdx = tid % innerStride;
    uint temp = tid / innerStride;
    
    uint axisIdx = temp % axisDimOut;
    uint outerIdx = temp / axisDimOut;

    // -----------------------------------------------------------
    // 3. BOUNDS CHECK (Optional, depending on dispatch size)
    // -----------------------------------------------------------
    
    // Calculate Outer Dimensions product to verify bounds
    // (Or rely on tight dispatch). 
    // For safety, let's verify we aren't processing garbage if threads > elements.
    // We can do this by checking if outerIdx is within the bounds of shapeA's outer dims.
    // However, a simpler check is if we exceed the total count if passed in.
    // Assuming tight dispatch for now or valid outerIdx.
    if (outerIdx >= params.outerStride) return;

    // -----------------------------------------------------------
    // 4. COPY LOGIC
    // -----------------------------------------------------------

    if (axisIdx < axisDimA)
    {
        // READ FROM A
        // The indices (outer, inner) are same. Only axis differs.
        
        // Linear Index A = 
        // outerIdx * (axisDimA * innerStride) + axisIdx * innerStride + innerIdx
        
        uint strideA = axisDimA * innerStride;
        uint srcIdx = outerIdx * strideA + axisIdx * innerStride + innerIdx;
        
        params.output[tid] = params.inputA[srcIdx];
    }
    else
    {
        // READ FROM B
        // We need to shift the axis index back to 0-based for B
        uint axisIdxB = axisIdx - axisDimA;
        
        // Linear Index B
        uint strideB = axisDimB * innerStride;
        uint srcIdx = outerIdx * strideB + axisIdxB * innerStride + innerIdx;
        
        params.output[tid] = params.inputB[srcIdx];
    }
}
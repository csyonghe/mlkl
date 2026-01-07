implementing mlkl;

// =============================================================================
// GroupNorm Normalization Pass
// Uses precomputed stats from the reduction kernel (reduction.slang)
// =============================================================================

struct GroupNormNormalizeParams<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>
{
    TInput input;
    TSink output;
    
    uint batchSize;
    uint height;
    uint width;
    uint channels;
    uint numGroups;
    T.UnpackedType epsilon;
    
    T* stats;   // Precomputed [sum, sumSq] per (batch, group) from reduction kernel
    T* gamma;   // Scale parameters [channels]
    T* beta;    // Bias parameters [channels]
};

// Dispatch: ceil(totalElements / 256) thread groups
// Fully parallel - each thread handles one element
[numthreads(256, 1, 1)]
void groupNormNormalize<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<GroupNormNormalizeParams<T, TInput, TSink>, CDataLayout> params,
    uint3 dispatchId : SV_DispatchThreadID)
{
    uint elementIdx = dispatchId.x;
    
    uint totalElements = params.batchSize * params.height * params.width * params.channels;
    if (elementIdx >= totalElements) return;
    
    // Decode element index to (batch, h, w, channel) in NHWC layout
    uint c = elementIdx % params.channels;
    uint temp = elementIdx / params.channels;
    uint w = temp % params.width;
    temp = temp / params.width;
    uint h = temp % params.height;
    uint b = temp / params.height;
    
    // Determine which group this channel belongs to
    uint channelsPerGroup = params.channels / params.numGroups;
    uint groupIdx = c / channelsPerGroup;
    uint batchGroupIdx = b * params.numGroups + groupIdx;
    
    // Read precomputed stats
    uint statsIdx = batchGroupIdx * 2;
    T.UnpackedType sum = params.stats[statsIdx].unpack();
    T.UnpackedType sumSq = params.stats[statsIdx + 1].unpack();
    
    // Compute mean and variance
    uint spatialSize = params.height * params.width;
    uint groupSize = spatialSize * channelsPerGroup;
    T.UnpackedType invGroupSize = T.UnpackedType(1) / T.UnpackedType(groupSize);
    T.UnpackedType mean = sum * invGroupSize;
    T.UnpackedType variance = sumSq * invGroupSize - mean * mean;
    T.UnpackedType invStd = T.UnpackedType(rsqrt((variance + params.epsilon).toFloat()));
    
    // Read input value
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    Coord coord = Coord(b, h, w, c);
    T.UnpackedType val = params.input.eval(coord, emptyInput);
    
    // Normalize and apply affine transform: gamma * (x - mean) / std + beta
    T.UnpackedType gamma = params.gamma[c].unpack();
    T.UnpackedType beta = params.beta[c].unpack();
    T.UnpackedType normalized = (val - mean) * invStd;
    T.UnpackedType result = gamma * normalized + beta;
    
    // Store output
    params.output.store(coord, result);
}

// =============================================================================
// LayerNorm Normalization Pass  
// Normalizes across the last dimension (channels/features)
// =============================================================================

struct LayerNormNormalizeParams<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>
{
    TInput input;
    TSink output;
    
    uint numRows;       // Product of all dims except the last (B * H * W for NHWC, or B * S for sequences)
    uint numFeatures;   // Size of the last dimension to normalize over
    T.UnpackedType epsilon;
    
    T* stats;   // Precomputed [sum, sumSq] per row from reduction kernel
    T* gamma;   // Scale parameters [numFeatures]
    T* beta;    // Bias parameters [numFeatures]
};

// Dispatch: ceil(totalElements / 256) thread groups
[numthreads(256, 1, 1)]
void layerNormNormalize<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<LayerNormNormalizeParams<T, TInput, TSink>, CDataLayout> params,
    uint3 dispatchId : SV_DispatchThreadID)
{
    uint elementIdx = dispatchId.x;
    
    uint totalElements = params.numRows * params.numFeatures;
    if (elementIdx >= totalElements) return;
    
    uint row = elementIdx / params.numFeatures;
    uint col = elementIdx % params.numFeatures;
    
    // Read precomputed stats for this row
    uint statsIdx = row * 2;
    T.UnpackedType sum = params.stats[statsIdx].unpack();
    T.UnpackedType sumSq = params.stats[statsIdx + 1].unpack();
    
    // Compute mean and variance
    T.UnpackedType invN = T.UnpackedType(1) / T.UnpackedType(params.numFeatures);
    T.UnpackedType mean = sum * invN;
    T.UnpackedType variance = sumSq * invN - mean * mean;
    T.UnpackedType invStd = T.UnpackedType(rsqrt((variance + params.epsilon).toFloat()));
    
    // Read input value
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    Coord coord = Coord(row, col);
    T.UnpackedType val = params.input.eval(coord, emptyInput);
    
    // Normalize and apply affine transform
    T.UnpackedType gamma = params.gamma[col].unpack();
    T.UnpackedType beta = params.beta[col].unpack();
    T.UnpackedType normalized = (val - mean) * invStd;
    T.UnpackedType result = gamma * normalized + beta;
    
    params.output.store(coord, result);
}

// =============================================================================
// RMSNorm Normalization Pass
// Root Mean Square normalization (no mean subtraction, only variance scaling)
// =============================================================================

struct RMSNormNormalizeParams<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>
{
    TInput input;
    TSink output;
    
    uint numRows;
    uint numFeatures;
    T.UnpackedType epsilon;
    
    T* stats;   // Precomputed [sum, sumSq] per row - we only use sumSq
    T* gamma;   // Scale parameters [numFeatures]
};

// Dispatch: ceil(totalElements / 256) thread groups
[numthreads(256, 1, 1)]
void rmsNormNormalize<T : ITensorElement, TInput : IExpr<T>, TSink : ISink<T>>(
    ConstantBuffer<RMSNormNormalizeParams<T, TInput, TSink>, CDataLayout> params,
    uint3 dispatchId : SV_DispatchThreadID)
{
    uint elementIdx = dispatchId.x;
    
    uint totalElements = params.numRows * params.numFeatures;
    if (elementIdx >= totalElements) return;
    
    uint row = elementIdx / params.numFeatures;
    uint col = elementIdx % params.numFeatures;
    
    // Read precomputed sumSq (we don't need sum for RMSNorm)
    uint statsIdx = row * 2 + 1;  // +1 to get sumSq
    T.UnpackedType sumSq = params.stats[statsIdx].unpack();
    
    // Compute RMS: sqrt(mean(x^2) + eps)
    T.UnpackedType meanSq = sumSq / T.UnpackedType(params.numFeatures);
    T.UnpackedType invRms = T.UnpackedType(rsqrt((meanSq + params.epsilon).toFloat()));
    
    // Read input value
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    Coord coord = Coord(row, col);
    T.UnpackedType val = params.input.eval(coord, emptyInput);
    
    // Normalize: gamma * x / rms
    T.UnpackedType gamma = params.gamma[col].unpack();
    T.UnpackedType result = gamma * val * invRms;
    
    params.output.store(coord, result);
}

implementing mlkl;

struct BroadcastedAddParams
{
    float* inputLhs;       // The Big Tensor (e.g. Image [B, C, H, W])
    float* inputRhs;       // The Small Tensor (e.g. Time [B, C] or [C])
    float* output;
    
    uint lhsRank;          // e.g. 4
    uint rhsRank;          // e.g. 1 or 2
    
    // We store the shape of the LHS (Big Tensor)
    // because total threads = product of lhsShape.
    uint lhsShape[8]; 
    
    // We store the shape of RHS to compute the broadcast index.
    // If a dimension is missing (rhsRank < lhsRank), we assume alignment on the right?
    // Actually, standard Numpy/PyTorch broadcasting aligns on the *right*.
    // e.g. LHS [4, 320, 64, 64] + RHS [320, 1, 1]
    uint rhsShape[8];
};

// Helper to map a linear index back to a specific dimension index
// and then map that to the broadcasted index for RHS.
uint mapToRhsIndex(uint linearId, BroadcastedAddParams params)
{
    uint rhsLinear = 0;
    uint rhsStride = 1;
    uint lhsStride = 1;
    
    // We traverse dimensions from right-to-left (fastest moving axis first)
    // to decompose the linearId.
    // NOTE: This assumes Row-Major memory layout (Rightmost dim is stride 1).
    
    // Calculate alignment offset.
    // PyTorch broadcasting aligns on the trailing dimensions.
    // If LHS is Rank 4 and RHS is Rank 1, RHS corresponds to Dim 3?
    // Actually, in Diffusion U-Nets, Time Embed is usually [Batch, Channels].
    // Image is [Batch, Channels, Height, Width].
    // So we want to add [B, C] to [B, C, H, W].
    // This requires RHS to be reshaped to [B, C, 1, 1] conceptually.
    
    // To keep this generic, we assume 'rhsShape' has already been padded 
    // with 1s to match 'lhsRank' in the host setup code.
    // e.g. if adding vector [C] to [B, C, H, W], host should pass rhsShape as [1, C, 1, 1].
    
    uint tempId = linearId;

    [unroll]
    for (int i = int(params.lhsRank) - 1; i >= 0; i--)
    {
        // 1. Get position in current dimension
        uint dimSizeLhs = params.lhsShape[i];
        uint coord = tempId % dimSizeLhs;
        tempId /= dimSizeLhs;

        // 2. Map to RHS
        // If RHS dimension is 1, index is always 0.
        // If RHS dimension matches LHS, index is coord.
        uint dimSizeRhs = params.rhsShape[i];
        
        // This 'coord % dimSizeRhs' handles broadcasting:
        // If dimSizeRhs is 1, coord % 1 == 0.
        // If dimSizeRhs == dimSizeLhs, coord % size == coord.
        uint rhsCoord = coord % dimSizeRhs;

        // 3. Accumulate RHS linear index
        rhsLinear += rhsCoord * rhsStride;
        rhsStride *= dimSizeRhs;
    }
    
    return rhsLinear;
}

[numthreads(256, 1, 1)]
void broadcastAdd(
    ConstantBuffer<BroadcastedAddParams> params,
    uint3 id : SV_DispatchThreadID)
{
    uint tid = id.x;

    // 1. Calculate Total Elements
    uint totalElements = 1;
    for(uint i = 0; i < params.lhsRank; i++) 
        totalElements *= params.lhsShape[i];

    if (tid >= totalElements) return;

    // 2. Compute RHS Index (Broadcasting Logic)
    uint rhsIdx = mapToRhsIndex(tid, params);

    // 3. Perform Add
    float lhsVal = params.inputLhs[tid];
    float rhsVal = params.inputRhs[rhsIdx];
    
    params.output[tid] = lhsVal + rhsVal;
}

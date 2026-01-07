implementing mlkl;

// =============================================================================
// Reduction Layout Interface
// Abstracts coordinate computation for different reduction patterns.
// Each layout is specialized at compile time - zero runtime overhead.
// =============================================================================

interface IReductionLayout
{
    // Get the coordinate for a given (groupIdx, elementIdx) pair
    Coord getCoord(uint groupIdx, uint elementIdx);
    
    // Get the number of elements to reduce per group
    uint getElementsPerGroup();
}

// =============================================================================
// Layout: Reduce last dimension of a 2D tensor
// Input shape: [numRows, numCols]
// Groups: numRows, Elements per group: numCols
// =============================================================================
struct LastDimLayout : IReductionLayout
{
    uint numRows;
    uint numCols;
    
    Coord getCoord(uint groupIdx, uint elementIdx)
    {
        return Coord(groupIdx, elementIdx);
    }
    
    uint getElementsPerGroup()
    {
        return numCols;
    }
}

// =============================================================================
// Layout: GroupNorm with NHWC tensor
// Reduces over (H, W, channelsPerGroup) for each (batch, group)
// =============================================================================
struct GroupNormLayout : IReductionLayout
{
    uint batchSize;
    uint height;
    uint width;
    uint numGroups;
    uint channelsPerGroup;
    
    Coord getCoord(uint groupIdx, uint elementIdx)
    {
        uint batchIdx = groupIdx / numGroups;
        uint groupNum = groupIdx % numGroups;
        uint groupChannelStart = groupNum * channelsPerGroup;
        
        uint spatialIdx = elementIdx / channelsPerGroup;
        uint localChannel = elementIdx % channelsPerGroup;
        uint globalChannel = groupChannelStart + localChannel;
        
        uint h = spatialIdx / width;
        uint w = spatialIdx % width;
        
        return Coord(batchIdx, h, w, globalChannel);
    }
    
    uint getElementsPerGroup()
    {
        return height * width * channelsPerGroup;
    }
}

// =============================================================================
// Layout: General axis reduction for up to 8D tensor
// Reduces over a single specified axis
// =============================================================================
struct AxisLayout : IReductionLayout
{
    uint shape[8];
    uint rank;  // Number of dimensions (1-8)
    uint axis;
    uint elementsPerGroup;
    
    Coord getCoord(uint groupIdx, uint elementIdx)
    {
        uint coord[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        
        // Decode groupIdx to non-reduced dimensions
        uint remaining = groupIdx;
        for (int d = int(rank) - 1; d >= 0; d--)
        {
            if (uint(d) == axis)
                coord[d] = elementIdx;
            else
            {
                coord[d] = remaining % shape[d];
                remaining = remaining / shape[d];
            }
        }
        
        return Coord(0, coord);
    }
    
    uint getElementsPerGroup()
    {
        return elementsPerGroup;
    }
}

// =============================================================================
// Generic Reduction Kernel
// Works with any layout that implements IReductionLayout
// =============================================================================
struct ReductionParams<T : ITensorElement, TInput : IExpr<T>, TLayout : IReductionLayout>
{
    TInput input;
    TLayout layout;
    uint numGroups;
    T* stats;  // Output: [numGroups * 2] = [sum0, sumSq0, sum1, sumSq1, ...]
}

// Dispatch: numGroups thread groups
[numthreads(256, 1, 1)]
void reduce<T : ITensorElement, TInput : IExpr<T>, TLayout : IReductionLayout>(
    ConstantBuffer<ReductionParams<T, TInput, TLayout>, CDataLayout> params,
    uint3 groupId : SV_GroupID,
    uint3 threadId : SV_GroupThreadID)
{
    static groupshared T.UnpackedType sharedSum[8];
    static groupshared T.UnpackedType sharedSumSq[8];
    
    uint groupIdx = groupId.x;
    if (groupIdx >= params.numGroups) return;
    
    uint tid = threadId.x;
    uint numThreads = 256;
    uint elementsPerGroup = params.layout.getElementsPerGroup();
    
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    
    // Each thread accumulates over its strided portion
    T.UnpackedType localSum = T.UnpackedType(0);
    T.UnpackedType localSumSq = T.UnpackedType(0);
    
    for (uint i = tid; i < elementsPerGroup; i += numThreads)
    {
        Coord coord = params.layout.getCoord(groupIdx, i);
        T.UnpackedType val = params.input.eval(coord, emptyInput);
        
        localSum = localSum + val;
        localSumSq = localSumSq + val * val;
    }
    
    // Wave-level reduction
    T.UnpackedType waveSum = localSum.waveActiveSum();
    T.UnpackedType waveSumSq = localSumSq.waveActiveSum();
    
    // Cross-wave reduction
    uint waveIndex = tid / WaveGetLaneCount();
    uint numWaves = (numThreads + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    
    if (WaveIsFirstLane())
    {
        sharedSum[waveIndex] = waveSum;
        sharedSumSq[waveIndex] = waveSumSq;
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (tid == 0)
    {
        T.UnpackedType totalSum = T.UnpackedType(0);
        T.UnpackedType totalSumSq = T.UnpackedType(0);
        for (uint w = 0; w < numWaves; w++)
        {
            totalSum = totalSum + sharedSum[w];
            totalSumSq = totalSumSq + sharedSumSq[w];
        }
        
        uint statsIdx = groupIdx * 2;
        params.stats[statsIdx] = T.pack(totalSum);
        params.stats[statsIdx + 1] = T.pack(totalSumSq);
    }
}

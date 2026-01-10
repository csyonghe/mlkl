implementing mlkl;

struct Input<T : ITensorElement>
{
    T.UnpackedType value;
}

struct Coord
{
    uint baseOffset;
    uint coords[8];
    __subscript(int index) -> uint
    {
        get { return coords[index]; }
        set { coords[index] = newValue; }
    }
    __init()
    {
        baseOffset = 0; coords = makeArrayFromElement<uint,8>(0);
    }
    __init(uint x)
    {
        baseOffset = 0; coords = {x, 0, 0, 0, 0, 0, 0, 0};
    }
    __init(uint x, uint y)
    {
        baseOffset = 0; coords = {x, y, 0, 0, 0, 0, 0, 0};
    }
    __init(uint x, uint y, uint z)
    {
        baseOffset = 0; coords = {x, y, z, 0, 0, 0, 0, 0};
    }
    __init(uint x, uint y, uint z, uint w)
    {
        baseOffset = 0; coords = {x, y, z, w, 0, 0, 0, 0};
    }
    // Int overloads for convenience
    __init(int x)
    {
        baseOffset = 0; coords = {uint(x), 0, 0, 0, 0, 0, 0, 0};
    }
    __init(int x, int y)
    {
        baseOffset = 0; coords = {uint(x), uint(y), 0, 0, 0, 0, 0, 0};
    }
    __init(int x, int y, int z)
    {
        baseOffset = 0; coords = {uint(x), uint(y), uint(z), 0, 0, 0, 0, 0};
    }
    __init(int x, int y, int z, int w)
    {
        baseOffset = 0; coords = {uint(x), uint(y), uint(z), uint(w), 0, 0, 0, 0};
    }
    __init(uint offset, uint coords[8])
    {
        baseOffset = offset;
        this.coords = coords;
    }
    
    // Create a Coord from a linear index (for backward compatibility)
    static Coord fromLinearIndex(uint linearIndex)
    {
        Coord c;
        c.baseOffset = linearIndex;
        c.coords = makeArrayFromElement<uint,8>(0);
        return c;
    }
}

interface IExpr<T : ITensorElement>
{
    T.UnpackedType eval(Coord coord, Input<T> input);
}

#if 0
extension<T : ITensorElement, E : IExpr<T>> E
{
    __subscript(int linearIndex) -> T.UnpackedType
    {
        get
        {
            Input<T> input;
            input.value = T.UnpackedType(0);
            return eval(Coord.fromLinearIndex(linearIndex), input);
        }
    }
}
#endif

interface IStmt<T : ITensorElement>
{
    void execute(Coord coord, Input<T> input);
}

interface ISink<T : ITensorElement>
{
    void store(Coord coord, T.UnpackedType value);
};

// Acts as both the Storage and the Expression
struct Reg<T : ITensorElement, let ID : int> : IExpr<T>
{
    // Thread-local static storage unique to this specialization <T, ID>
    static T.UnpackedType value;

    // IExpr implementation: Read the value
    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        return value;
    }
}

// Assignment: %ID = Expr
struct Eval<T : ITensorElement, let ID : int, E : IExpr<T>> : IStmt<T>
{
    E expr;

    void execute(Coord coord, Input<T> input)
    {
        // Write to the static storage for this ID
        Reg<T, ID>.value = expr.eval(coord, input);
    }
}

struct Program<T : ITensorElement, int resultReg, each S : IStmt<T>> : IExpr<T>
{
    Tuple<expand each S> stmts;

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Execute each statement in order
        expand (each stmts).execute(coord, input);
        // Return the value of the result register
        return Reg<T, resultReg>.value;
    }
}

struct BufferView<T : ITensorElement> : IExpr<T>
{
    T* data;
    int rank;
    uint strides[8];
    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // baseOffset is used for direct linear indexing (subscript operator)
        // coords are used for N-D coordinate access
        uint linearIdx = coord.baseOffset;
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            if (i >= rank) break;
            linearIdx += coord[i] * strides[i];
        }
        return data[linearIdx].unpack();
    }
}

struct ConstantView<T : ITensorElement> : IExpr<T>
{
    float value;  // Always store as float for simplicity
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return T.UnpackedType.fromFloat(value); 
    }
}

struct KernelOutput<T : ITensorElement> : IExpr<T>
{
    T.UnpackedType eval(Coord coord, Input<T> input) { return input.value; }
}

struct Broadcast<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint rank;
    uint innerStrides[8];  // 0 means broadcast, >0 means take from coord

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Transform output coord to inner coord by zeroing broadcast dimensions
        Coord innerCoord;
        innerCoord.baseOffset = coord.baseOffset;
        innerCoord.coords = makeArrayFromElement<uint, 8>(0);
        
        [ForceUnroll]
        for (int i = 0; i < 8; i++)
        {
            // If stride is 0, the dimension is broadcast (set coord to 0)
            // Otherwise, pass through the coordinate
            innerCoord[i] = (innerStrides[i] == 0) ? 0 : coord[i];
        }
        return inner.eval(innerCoord, input);
    }
}

struct Permute<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint rank;
    // permMap[i] = j means output coord[i] maps to inner coord[j]
    uint permMap[8];

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Transform output coord to inner coord via permutation
        // Initialize all coords to 0 first
        Coord innerCoord;
        innerCoord.baseOffset = coord.baseOffset;
        innerCoord.coords = makeArrayFromElement<uint, 8>(0);
        
        // Apply permutation: output dimension i maps to inner dimension permMap[i]
        // Manually unrolled to avoid double-indirection issues
        if (0 < rank) innerCoord[permMap[0]] = coord[0];
        if (1 < rank) innerCoord[permMap[1]] = coord[1];
        if (2 < rank) innerCoord[permMap[2]] = coord[2];
        if (3 < rank) innerCoord[permMap[3]] = coord[3];
        if (4 < rank) innerCoord[permMap[4]] = coord[4];
        if (5 < rank) innerCoord[permMap[5]] = coord[5];
        if (6 < rank) innerCoord[permMap[6]] = coord[6];
        if (7 < rank) innerCoord[permMap[7]] = coord[7];
        
        return inner.eval(innerCoord, input);
    }
}

struct Transpose<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint dim0; // Dimension to swap
    uint dim1; // Dimension to swap

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Swap coordinates at dim0 and dim1
        Coord innerCoord = coord;
        innerCoord[dim0] = coord[dim1];
        innerCoord[dim1] = coord[dim0];
        return inner.eval(innerCoord, input);
    }
}

struct Concat<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L left;
    R right;
    
    uint axis;
    uint splitPoint; // Size of Left dimension along axis

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        uint axisCoord = coord[axis];
        
        if (axisCoord < splitPoint)
        {
            // Left tensor - pass coord through unchanged
            return left.eval(coord, input);
        }
        else
        {
            // Right tensor - shift coordinate along axis
            Coord rightCoord = coord;
            rightCoord[axis] = axisCoord - splitPoint;
            return right.eval(rightCoord, input);
        }
    }
}

// ============================================================================
// Slice along an axis
// ============================================================================

// Slice extracts a contiguous range [start, start+size) along a specific axis
// The output shape has the sliced dimension reduced to `size`
struct Slice<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint axis;      // Which axis to slice
    uint start;     // Start index along the axis

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Offset the coordinate at the slice axis by start
        Coord innerCoord = coord;
        innerCoord[axis] = coord[axis] + start;
        return inner.eval(innerCoord, input);
    }
}

// ============================================================================
// Duplicate (tile) a dimension
// ============================================================================

// Duplicate (tile) values along a dimension
// Output dimension = input dimension * times (implicit from output shape)
// E.g., input [1, H, W, C] with dim=0 → output [2, H, W, C]
// Maps output coord to input coord via modulo on the duplicated dimension
struct Duplicate<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint dim;           // Which dimension to duplicate
    uint originalSize;  // Original size of the dimension (e.g., 1)
    uint stride;        // Stride for the dimension (elements per increment)

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Map output coord back to input coord via modulo
        Coord innerCoord = coord;
        innerCoord[dim] = coord[dim] % originalSize;
        return inner.eval(innerCoord, input);
    }
}

// ============================================================================
// Upsampling (nearest-neighbor)
// ============================================================================

// Upsample spatial dimensions using nearest-neighbor interpolation
// Assumes NHWC layout by default (dims 1 and 2 are H and W)
// factor: upsampling factor (e.g., 2 for 2x upsampling)
struct Upsample<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint factor;        // Upsampling factor
    uint heightDim;     // Which dimension is height (default: 1)
    uint widthDim;      // Which dimension is width (default: 2)

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // Nearest-neighbor: divide output coord by factor (floor division)
        Coord innerCoord = coord;
        innerCoord[heightDim] = coord[heightDim] / factor;
        innerCoord[widthDim] = coord[widthDim] / factor;
        
        return inner.eval(innerCoord, input);
    }
}

// ============================================================================
// Im2Col - Transform input tensor for convolution via GEMM
// ============================================================================
//
// Transforms [N, H, W, IC] input into [IC*KH*KW, N*OH*OW] matrix.
// K-dimension uses IC-minor ordering for optimal NHWC memory coalescing:
//   row = kh * KW * IC + kw * IC + ic
//
// This ensures consecutive K values access consecutive memory addresses
// when reading from NHWC layout tensors.
//
struct Im2Col<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint kernelH, kernelW;
    uint strideH, strideW;
    uint padH, padW;
    uint inputH, inputW, inputC;
    uint outputH, outputW;
    uint batchSize;

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        // coord = [batch, row, col] in im2col matrix (batch dimension is always 0)
        // row: K dimension (0..IC*KH*KW-1)
        // col: spatial dimension (0..N*OH*OW-1)
        uint row = coord[1];
        uint col = coord[2];

        // Decode output spatial position from col
        uint spatialSize = outputH * outputW;
        uint n = col / spatialSize;
        uint spatial = col % spatialSize;
        uint oh = spatial / outputW;
        uint ow = spatial % outputW;

        // Decode kernel position from row using IC-minor ordering:
        // row = kh * KW * IC + kw * IC + ic
        // This gives optimal memory coalescing for NHWC input
        uint ic = row % inputC;
        uint spatialK = row / inputC;
        uint kh = spatialK / kernelW;
        uint kw = spatialK % kernelW;

        // Compute input position
        int ih = int(oh * strideH + kh) - int(padH);
        int iw = int(ow * strideW + kw) - int(padW);

        // Zero padding for out-of-bounds
        if (ih < 0 || ih >= int(inputH) || iw < 0 || iw >= int(inputW))
            return T.UnpackedType(0);

        // Evaluate inner at [n, ih, iw, ic] (NHWC layout)
        Coord innerCoord = Coord(n, uint(ih), uint(iw), ic);
        return inner.eval(innerCoord, input);
    }
}

// ============================================================================

struct Gather<T : ITensorElement, Table : IExpr<T>, Indices : IExpr<T>> : IExpr<T>
{
    Table table;     // The lookup table (e.g., Weights [NumClasses, Dim])
    Indices indices; // The indices to gather (e.g., Labels [Batch])
    
    // For simplicity, this implementation assumes gathering along Axis 0 
    // and that 'Table' is 2D and 'Indices' is 1D.
    // Output Shape: [Batch, Dim]
    // coord[0] = batch index, coord[1] = embedding dim

    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        uint b = coord[0]; // Batch index
        uint d = coord[1]; // Embedding dimension

        // 1. Evaluate Index at position 'b'
        Coord idxCoord = Coord(b);
        T.UnpackedType idxVal = indices.eval(idxCoord, input);
        int row = idxVal.toInt(); // Convert to int index

        // 2. Evaluate Table at position (row, d)
        Coord tableCoord = Coord(row, d);
        
        return table.eval(tableCoord, input);
    }
}

// Binary Ops
struct Add<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L l; R r;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return l.eval(coord, input) + r.eval(coord, input); 
    }
}

struct Sub<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L l; R r;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return l.eval(coord, input) - r.eval(coord, input); 
    }
}

struct Mul<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L l; R r;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return l.eval(coord, input) * r.eval(coord, input); 
    }
}

struct Div<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L l; R r;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return l.eval(coord, input) / r.eval(coord, input); 
    }
}

struct Min<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L left; R right;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return T.UnpackedType.min(left.eval(coord, input), right.eval(coord, input)); 
    }
}

struct Max<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L left; R right;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return T.UnpackedType.max(left.eval(coord, input), right.eval(coord, input)); 
    }
}

struct Pow<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L base; R exp;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return base.eval(coord, input).pow(exp.eval(coord, input)); 
    }
}

// --- Unary Operations ---

struct Neg<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return -inner.eval(coord, input); 
    }
}

struct Exp<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).exp(); 
    }
}

struct Log<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).log(); 
    }
}

struct Sin<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).sin(); 
    }
}

struct Cos<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).cos(); 
    }
}

struct Abs<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).abs(); 
    }
}

struct Sqrt<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).sqrt(); 
    }
}

struct Floor<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).floor(); 
    }
}

struct Ceil<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).ceil(); 
    }
}

struct Rsqrt<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).rsqrt(); 
    }
}

struct Tanh<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return inner.eval(coord, input).tanh(); 
    }
}

// --- Activations (composed from primitives) ---

struct Relu<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        return T.UnpackedType.max(T.UnpackedType(0), inner.eval(coord, input)); 
    }
}

struct Sigmoid<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        let v = inner.eval(coord, input);
        return T.UnpackedType(1) / (T.UnpackedType(1) + (-v).exp()); 
    }
}

struct Silu<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        let x = inner.eval(coord, input);
        // x * sigmoid(x)
        return x * (T.UnpackedType(1) / (T.UnpackedType(1) + (-x).exp()));
    }
}

struct Gelu<T : ITensorElement, I : IExpr<T>> : IExpr<T>
{
    I inner;
    T.UnpackedType eval(Coord coord, Input<T> input) 
    { 
        let x = inner.eval(coord, input);
        // Tanh approximation
        let c1 = T.UnpackedType.fromFloat(0.7978845608f);
        let c2 = T.UnpackedType.fromFloat(0.044715f);
        let half = T.UnpackedType.fromFloat(0.5f);
        return half * x * (T.UnpackedType(1) + (c1 * (x + c2 * x * x * x)).tanh());
    }
}


// BufferSink: The terminal node that actually writes to memory.
struct BufferSink<T : ITensorElement> : ISink<T>
{
    T* data;
    int rank;
    int strides[8];

    void store(Coord coord, T.UnpackedType value)
    {
        uint offset = coord.baseOffset;
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            if (i >= rank)
                break;
            offset += coord[i] * strides[i];
        }
        data[offset] = T.pack(value);
    }
};

// PermuteSink: permutes output dimension ordering.
struct PermuteSink<T : ITensorElement, Child : ISink<T>> : ISink<T>
{
    Child child;
    uint pMap[8]; // Permutation map

    void store(Coord coords, T.UnpackedType value)
    {
        Coord newCoords = {coords.baseOffset, makeArrayFromElement<uint, 8>(0)};
        [unroll]
        for (uint i = 0; i < 8; i++)
        {
            if (pMap[i] == 0xFFFFFFFF)
                break;
            newCoords[i] = coords[pMap[i]];
        }
        child.store(newCoords, value);
    }
};

/**
 * PartitionSink: Physically re-orders data by splitting a single logical dimension 
 * into multiple contiguous physical "piles" or "partitions".

 * Example:
 * Consider a tensor of shape: [H=2, W=6]. Partition at dimIndex = 1, and partitionSize = 3.
 * (Results in N=2 partitions).
 * Logically, this partitions the tensor as follows:
 * Row 0: [0, 1, 2] | [3, 4, 5]  (Part 0 | Part 1)
 * Row 1: [6, 7, 8] | [9, 10, 11] (Part 0 | Part 1)
 * We will store each partition contiguously in physical memory:
    * Physical Layout:
    * Part 0: [0, 1, 2, 6, 7, 8]
    * Part 1: [3, 4, 5, 9, 10, 11]
 */
struct PartitionSink<T : ITensorElement, Child : ISink<T>> : ISink<T>
{
    Child child;
    
    uint dimIndex;             // Which coordinate index to split (e.g., 1 for Column)
    uint partitionSize;        // The size of a single partition block (e.g., HeadDim)

    void store(Coord logicalCoord, T.UnpackedType value)
    {
        uint pIdx = logicalCoord[dimIndex] / partitionSize;
        uint localCoord = logicalCoord[dimIndex] % partitionSize;

        // Create the (N+1)-D Physical Coordinate
        Coord physicalCoord;
        physicalCoord.baseOffset = logicalCoord.baseOffset;
        
        // Dim 0 is the Partition ID
        physicalCoord[0] = pIdx;
        
        // Shift all original dimensions down by 1
        [ForceUnroll]
        for (int i = 0; i < 7; i++)
        {
            uint val = logicalCoord[i];
            if (i == (int)dimIndex) val = localCoord;
            
            physicalCoord[i + 1] = val;
        }

        child.store(physicalCoord, value);
    }
};

// Conv2DOutputSink: Transforms GEMM output [1, OC, N*OH*OW] → [N, OH, OW, OC]
// Used for im2col-based convolution to write directly to NHWC output format.
struct Conv2DOutputSink<T : ITensorElement, Child : ISink<T>> : ISink<T>
{
    Child child;
    uint outputH;
    uint outputW;

    void store(Coord logicalCoord, T.UnpackedType value)
    {
        // Input logical coord: [1, oc, col] where col = n * OH * OW + oh * OW + ow
        // Note: logicalCoord[0] is always 0 (batch dim of GEMM is 1)
        uint oc = logicalCoord[1];
        uint col = logicalCoord[2];

        // Decompose col → (n, oh, ow)
        uint spatialSize = outputH * outputW;
        uint n = col / spatialSize;
        uint spatial = col % spatialSize;
        uint oh = spatial / outputW;
        uint ow = spatial % outputW;

        // Output physical coord: [n, oh, ow, oc]
        Coord physicalCoord;
        physicalCoord.baseOffset = logicalCoord.baseOffset;
        physicalCoord.coords = makeArrayFromElement<uint, 8>(0);
        physicalCoord[0] = n;
        physicalCoord[1] = oh;
        physicalCoord[2] = ow;
        physicalCoord[3] = oc;

        child.store(physicalCoord, value);
    }
};

struct MaterializeParams<T : ITensorElement, P : IExpr<T>>
{
    T* outputBuffer;
    uint count;
    uint rank;
    uint shape[8];
    P program;
}

[numthreads(256, 1, 1)]
void materialize<T : ITensorElement, P : IExpr<T>>(
    ConstantBuffer<MaterializeParams<T, P>, ScalarDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint tid = id.x;
    if (tid >= params.count) return;
    // Decompose linear index to N-D coordinate (row-major order)
    Coord coord;
    coord.baseOffset = 0;
    uint temp = tid;
    
    [ForceUnroll]
    for (int i = 7; i >= 0; i--)
    {
        if (i < int(params.rank))
        {
            uint dimSize = params.shape[i];
            coord[i] = temp % dimSize;
            temp /= dimSize;
        }
        else
        {
            coord[i] = 0;
        }
    }

    Input<T> input = {};

    // Execute Program with N-D coordinate
    let result = params.program.eval(coord, input);

    // Write Output
    params.outputBuffer[tid] = T.pack(result);
}

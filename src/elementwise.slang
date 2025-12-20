implementing mlkl;

struct Input
{
    float value;
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
    __init(int x)
    {
        baseOffset = 0; coords = {x};
    }
    __init(int x, int y)
    {
        baseOffset = 0; coords = {x, y};
    }
    __init(int x, int y, int z)
    {
        baseOffset = 0; coords = {x, y, z};
    }
    __init(int x, int y, int z, int w)
    {
        baseOffset = 0; coords = {x, y, z, w};
    }
    __init(uint offset, uint coords[8])
    {
        baseOffset = offset;
        this.coords = coords;
    }
}

interface IExpr
{
    float eval(uint tid, Input input);
}

extension<T:IExpr> T
{
    __subscript(int linearIndex)->float
    {
        get
        {
            return eval(linearIndex, Input(0.0f));
        }
    }
}

interface IStmt
{
    void execute(uint tid, Input input);
}

interface ISink
{
    void store(Coord coord, float value);
};

// Acts as both the Storage and the Expression
struct Reg<let ID : int> : IExpr
{
    // Thread-local static storage unique to this specialization <ID>
    static float value;

    // IExpr implementation: Read the value
    float eval(uint tid, Input input)
    {
        return value;
    }
}

// Assignment: %ID = Expr
struct Eval<let ID : int, E : IExpr> : IStmt
{
    E expr;

    void execute(uint tid, Input input)
    {
        // Write to the static storage for this ID
        Reg<ID>.value = expr.eval(tid, input);
    }
}

struct Program<int resultReg, each S : IStmt> : IExpr
{
    Tuple<expand each S> stmts;

    float eval(uint tid, Input input)
    {
        // Execute each statement in order
        expand (each stmts).execute(tid, input);
        // Return the value of the result register
        return Reg<resultReg>.value;
    }
}

struct BufferView : IExpr
{
    float* data;
    float eval(uint tid, Input input) { return data[tid]; }
}

struct ConstantView : IExpr
{
    float value;
    float eval(uint tid, Input input) { return value; }
}

struct KernelOutput : IExpr
{
    float eval(uint tid, Input input) { return input.value; }
}

struct Broadcast<Inner : IExpr> : IExpr
{
    Inner inner;
    uint rank;
    uint outputShape[8];
    uint innerStrides[8];

    float eval(uint tid, Input input)
    {
        uint innerLinear = 0;
        uint temp = tid;
        
        [unroll]
        for (int i = 7; i >= 0; i--) 
        {
            if (i < int(rank))
            {
                uint dimSize = outputShape[i];
                uint coord = temp % dimSize;
                temp /= dimSize;
                innerLinear += coord * innerStrides[i];
            }
        }
        return inner.eval(innerLinear, input);
    }
}
struct Permute<Inner : IExpr> : IExpr
{
    Inner inner;
    uint rank;
    // Dimensions packed in REVERSE order (index 0 is the innermost/last dimension)
    uint outputDims[8];
    // Strides in the physical buffer mapped to the reversed logical dimensions
    uint mappedStrides[8]; 

    float eval(uint tid, Input input)
    {
        // For scalars (rank 0), the loop is skipped and it returns inner.eval(0)
        uint innerIdx = 0;
        uint temp = tid;
        
        // We iterate through the packed metadata. 
        // Because we packed in reverse in C++, index 0 is the fastest changing dim.
        [unroll]
        for (uint i = 0; i < 8; i++)
        {
            if (i < rank)
            {
                uint d = outputDims[i];
                
                // Extract the coordinate for this logical dimension
                // If d is 1 (e.g. for a scalar or padding), c is always 0.
                uint c = temp % d;
                temp /= d;
                
                // Add the contribution to the physical buffer offset
                innerIdx += c * mappedStrides[i];
            }
        }
        
        return inner.eval(innerIdx, input);
    }
}

struct Transpose<Inner : IExpr> : IExpr
{
    Inner inner;
    uint dim0; // Dimension to swap
    uint dim1; // Dimension to swap
    
    uint shape[8];   // Output shape
    uint strides[8]; // Inner tensor strides

    // To evaluate Transpose(A) at index 'i':
    // 1. Convert 'i' to coords (in output shape)
    // 2. Swap coord[dim0] and coord[dim1]
    // 3. Re-linearize using inner strides
    float eval(uint tid, Input input)
    {
        // 1. Decompose
        uint coord[8];
        uint temp = tid;
        
        // Unroll assuming max rank 8
        [unroll]
        for (int i = 7; i >= 0; i--)
        {
             // Note: We use the output shape for decomposition
             uint d = shape[i]; 
             // Avoid div/mod by 0 if shape is uninitialized/padding
             if (d > 1) 
             {
                 coord[i] = temp % d;
                 temp /= d;
             }
             else
             {
                 coord[i] = 0;
             }
        }

        // 2. Swap coordinates
        uint t = coord[dim0];
        coord[dim0] = coord[dim1];
        coord[dim1] = t;

        // 3. Re-linearize
        uint innerIdx = 0;
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            innerIdx += coord[i] * strides[i];
        }
        
        return inner.eval(innerIdx, input);
    }
}

struct Concat<L : IExpr, R : IExpr> : IExpr
{
    L left;
    R right;
    
    uint axis;
    uint splitPoint; // Size of Left dimension along axis
    uint rank;
    
    // Metadata for coordinate transformation
    uint outputDims[8];    // To decompose tid -> coords
    uint leftStrides[8];   // To compose coords -> tid_L
    uint rightStrides[8];  // To compose coords -> tid_R

    float eval(uint tid, Input input)
    {
        // 1. Decompose Linear Output ID -> N-D Coordinates
        uint coord[8];
        uint temp = tid;
        
        // Standard row-major decomposition (last dim is contiguous)
        [unroll]
        for (int i = 7; i >= 0; i--)
        {
            if (i < int(rank))
            {
                uint dimSize = outputDims[i];
                coord[i] = temp % dimSize;
                temp /= dimSize;
            }
            else 
            {
                coord[i] = 0;
            }
        }

        // 2. Branch based on Axis
        uint axisCoord = coord[axis];
        
        if (axisCoord < splitPoint)
        {
            // Case: Left Tensor
            // Re-linearize using Left Strides
            uint idx = 0;
            [unroll]
            for (int i = 0; i < 8; ++i) {
                if (i < int(rank)) idx += coord[i] * leftStrides[i];
            }
            return left.eval(idx, input);
        }
        else
        {
            // Case: Right Tensor
            // Shift coordinate relative to Right's start
            coord[axis] = axisCoord - splitPoint;
            
            // Re-linearize using Right Strides
            uint idx = 0;
            [unroll]
            for (int i = 0; i < 8; ++i) {
                 if (i < int(rank)) idx += coord[i] * rightStrides[i];
            }
            return right.eval(idx, input);
        }
    }
}
struct Gather<Table : IExpr, Indices : IExpr> : IExpr
{
    Table table;     // The lookup table (e.g., Weights [NumClasses, Dim])
    Indices indices; // The indices to gather (e.g., Labels [Batch])
    
    uint embeddingDim; // Size of the inner dimension (Dim)
    // For simplicity, this implementation assumes gathering along Axis 0 
    // and that 'Table' is 2D and 'Indices' is 1D.

    float eval(uint tid, Input input)
    {
        // Output Shape: [Batch, Dim]
        // Map linear TID -> (batch, dim)
        uint d = tid % embeddingDim;
        uint b = tid / embeddingDim;

        // 1. Evaluate Index at position 'b'
        // We pass 'b' as the TID to the indices expression
        float idxVal = indices.eval(b, input);
        int row = int(idxVal); // Cast float to int index

        // 2. Evaluate Table at position (row, d)
        // Table is [NumClasses, Dim], Row-Major.
        // Linear Index = row * embeddingDim + d
        uint tableTid = row * embeddingDim + d;
        
        return table.eval(tableTid, input);
    }
}

// Binary Ops
struct Add<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid, Input input) { return l.eval(tid, input) + r.eval(tid, input); }
}
struct Sub<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid, Input input) { return l.eval(tid, input) - r.eval(tid, input); }
}
struct Mul<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid, Input input) { return l.eval(tid, input) * r.eval(tid, input); }
}
struct Div<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid, Input input) { return l.eval(tid, input) / r.eval(tid, input); }
}

struct Min<L : IExpr, R : IExpr> : IExpr
{
    L left; R right;
    float eval(uint tid, Input input) { return min(left.eval(tid, input), right.eval(tid, input)); }
}

struct Max<L : IExpr, R : IExpr> : IExpr
{
    L left; R right;
    float eval(uint tid, Input input) { return max(left.eval(tid, input), right.eval(tid, input)); }
}

// --- Unary Operations ---

struct Neg<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return -inner.eval(tid, input); }
}

struct Exp<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return exp(inner.eval(tid, input)); }
}

struct Log<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return log(inner.eval(tid, input)); }
}

struct Sin<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return sin(inner.eval(tid, input)); }
}

struct Cos<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return cos(inner.eval(tid, input)); }
}

struct Abs<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return abs(inner.eval(tid, input)); }
}

struct Sqrt<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return sqrt(inner.eval(tid, input)); }
}

struct Pow<L : IExpr, R : IExpr> : IExpr
{
    L base; R exp;
    // HLSL pow(x, y)
    float eval(uint tid, Input input) { return pow(base.eval(tid, input), exp.eval(tid, input)); }
}
struct Floor<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return floor(inner.eval(tid, input)); }
}

struct Ceil<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return ceil(inner.eval(tid, input)); }
}

struct Rsqrt<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return rsqrt(inner.eval(tid, input)); }
}

// --- Activations ---

struct Relu<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return max(0.0, inner.eval(tid, input)); }
}

struct Sigmoid<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) 
    { 
        float v = inner.eval(tid, input);
        return 1.0 / (1.0 + exp(-v)); 
    }
}

struct Tanh<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) { return tanh(inner.eval(tid, input)); }
}

struct Silu<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) 
    { 
        float x = inner.eval(tid, input);
        // x * sigmoid(x)
        return x * (1.0 / (1.0 + exp(-x)));
    }
}

struct Gelu<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid, Input input) 
    { 
        float x = inner.eval(tid, input);
        // Tanh approximation
        return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    }
}


// BufferSink: The terminal node that actually writes to memory.
struct BufferSink : ISink
{
    float* data;
    int rank;
    int strides[8];

    void store(Coord coord, float value)
    {
        uint offset = coord.baseOffset;
        [unroll]
        for (int i = 0; i < 8; i++)
        {
            if (i >= rank)
                break;
            offset += coord[i] * strides[i];
        }
        data[offset] = value;
    }
};

// PermuteSink: permutes output dimension ordering.
struct PermuteSink<Child : ISink> : ISink
{
    Child child;
    uint pMap[8]; // Permutation map

    void store(Coord coords, float value)
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
struct PartitionSink<Child : ISink> : ISink
{
    Child child;
    
    uint dimIndex;             // Which coordinate index to split (e.g., 1 for Column)
    uint partitionSize;        // The size of a single partition block (e.g., HeadDim)

    void store(Coord logicalCoord, float value)
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

struct MaterializeParams<P : IExpr>
{
    float* outputBuffer;
    uint count;
    P program;
}

[numthreads(256, 1, 1)]
void materialize<P:IExpr>(
    ConstantBuffer<MaterializeParams<P>, ScalarDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint tid = id.x;
    if (tid >= params.count) return;

    Input input = {};

    // 1. Execute Program
    let result = params.program.eval(tid, input);

    // 2. Write Output
    params.outputBuffer[tid] = result;
}


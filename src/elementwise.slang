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

interface IExpr
{
    float eval(Coord coord, Input input);
}

extension<T:IExpr> T
{
    __subscript(int linearIndex)->float
    {
        get
        {
            return eval(Coord.fromLinearIndex(linearIndex), Input(0.0f));
        }
    }
}

interface IStmt
{
    void execute(Coord coord, Input input);
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
    float eval(Coord coord, Input input)
    {
        return value;
    }
}

// Assignment: %ID = Expr
struct Eval<let ID : int, E : IExpr> : IStmt
{
    E expr;

    void execute(Coord coord, Input input)
    {
        // Write to the static storage for this ID
        Reg<ID>.value = expr.eval(coord, input);
    }
}

struct Program<int resultReg, each S : IStmt> : IExpr
{
    Tuple<expand each S> stmts;

    float eval(Coord coord, Input input)
    {
        // Execute each statement in order
        expand (each stmts).execute(coord, input);
        // Return the value of the result register
        return Reg<resultReg>.value;
    }
}

struct BufferView: IExpr
{
    float* data;
    int rank;
    uint strides[8];
    float eval(Coord coord, Input input)
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
        return data[linearIdx];
    }
}

struct ConstantView : IExpr
{
    float value;
    float eval(Coord coord, Input input) { return value; }
}

struct KernelOutput : IExpr
{
    float eval(Coord coord, Input input) { return input.value; }
}

struct Broadcast<Inner : IExpr> : IExpr
{
    Inner inner;
    uint rank;
    uint innerStrides[8];  // 0 means broadcast, >0 means take from coord

    float eval(Coord coord, Input input)
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
struct Permute<Inner : IExpr> : IExpr
{
    Inner inner;
    uint rank;
    // permMap[i] = j means output coord[i] maps to inner coord[j]
    uint permMap[8];

    float eval(Coord coord, Input input)
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

struct Transpose<Inner : IExpr> : IExpr
{
    Inner inner;
    uint dim0; // Dimension to swap
    uint dim1; // Dimension to swap

    float eval(Coord coord, Input input)
    {
        // Swap coordinates at dim0 and dim1
        Coord innerCoord = coord;
        innerCoord[dim0] = coord[dim1];
        innerCoord[dim1] = coord[dim0];
        return inner.eval(innerCoord, input);
    }
}

struct Concat<L : IExpr, R : IExpr> : IExpr
{
    L left;
    R right;
    
    uint axis;
    uint splitPoint; // Size of Left dimension along axis

    float eval(Coord coord, Input input)
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
struct Gather<Table : IExpr, Indices : IExpr> : IExpr
{
    Table table;     // The lookup table (e.g., Weights [NumClasses, Dim])
    Indices indices; // The indices to gather (e.g., Labels [Batch])
    
    // For simplicity, this implementation assumes gathering along Axis 0 
    // and that 'Table' is 2D and 'Indices' is 1D.
    // Output Shape: [Batch, Dim]
    // coord[0] = batch index, coord[1] = embedding dim

    float eval(Coord coord, Input input)
    {
        uint b = coord[0]; // Batch index
        uint d = coord[1]; // Embedding dimension

        // 1. Evaluate Index at position 'b'
        Coord idxCoord = Coord(b);
        float idxVal = indices.eval(idxCoord, input);
        int row = int(idxVal); // Cast float to int index

        // 2. Evaluate Table at position (row, d)
        Coord tableCoord = Coord(row, d);
        
        return table.eval(tableCoord, input);
    }
}

// Binary Ops
struct Add<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(Coord coord, Input input) { return l.eval(coord, input) + r.eval(coord, input); }
}
struct Sub<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(Coord coord, Input input) { return l.eval(coord, input) - r.eval(coord, input); }
}
struct Mul<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(Coord coord, Input input) { return l.eval(coord, input) * r.eval(coord, input); }
}
struct Div<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(Coord coord, Input input) { return l.eval(coord, input) / r.eval(coord, input); }
}

struct Min<L : IExpr, R : IExpr> : IExpr
{
    L left; R right;
    float eval(Coord coord, Input input) { return min(left.eval(coord, input), right.eval(coord, input)); }
}

struct Max<L : IExpr, R : IExpr> : IExpr
{
    L left; R right;
    float eval(Coord coord, Input input) { return max(left.eval(coord, input), right.eval(coord, input)); }
}

// --- Unary Operations ---

struct Neg<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return -inner.eval(coord, input); }
}

struct Exp<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return exp(inner.eval(coord, input)); }
}

struct Log<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return log(inner.eval(coord, input)); }
}

struct Sin<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return sin(inner.eval(coord, input)); }
}

struct Cos<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return cos(inner.eval(coord, input)); }
}

struct Abs<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return abs(inner.eval(coord, input)); }
}

struct Sqrt<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return sqrt(inner.eval(coord, input)); }
}

struct Pow<L : IExpr, R : IExpr> : IExpr
{
    L base; R exp;
    // HLSL pow(x, y)
    float eval(Coord coord, Input input) { return pow(base.eval(coord, input), exp.eval(coord, input)); }
}
struct Floor<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return floor(inner.eval(coord, input)); }
}

struct Ceil<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return ceil(inner.eval(coord, input)); }
}

struct Rsqrt<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return rsqrt(inner.eval(coord, input)); }
}

// --- Activations ---

struct Relu<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return max(0.0, inner.eval(coord, input)); }
}

struct Sigmoid<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) 
    { 
        float v = inner.eval(coord, input);
        return 1.0 / (1.0 + exp(-v)); 
    }
}

struct Tanh<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) { return tanh(inner.eval(coord, input)); }
}

struct Silu<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) 
    { 
        float x = inner.eval(coord, input);
        // x * sigmoid(x)
        return x * (1.0 / (1.0 + exp(-x)));
    }
}

struct Gelu<I : IExpr> : IExpr
{
    I inner;
    float eval(Coord coord, Input input) 
    { 
        float x = inner.eval(coord, input);
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
    uint rank;
    uint shape[8];
    P program;
}

[numthreads(256, 1, 1)]
void materialize<P:IExpr>(
    ConstantBuffer<MaterializeParams<P>, ScalarDataLayout> params,
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

    Input input = {};

    // Execute Program with N-D coordinate
    let result = params.program.eval(coord, input);

    // Write Output
    params.outputBuffer[tid] = result;
}


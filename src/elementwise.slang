implementing mlkl;

struct Input
{
    float value;
}

interface IExpr
{
    float eval(uint tid, Input input);
}

interface IStmt
{
    void execute(uint tid, Input input);
}

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
    uint outputDims[8];
    // Stride of the dimension in the Inner tensor that corresponds to the i-th output dimension
    uint mappedStrides[8]; 

    float eval(uint tid, Input input)
    {
        uint innerIdx = 0;
        uint temp = tid;
        
        [unroll]
        for (int i = 7; i >= 0; i--)
        {
            if (i < int(rank))
            {
                uint d = outputDims[i];
                uint c = temp % d;
                temp /= d;
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
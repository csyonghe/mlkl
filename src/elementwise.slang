implementing mlkl;

interface IExpr
{
    float eval(uint tid);
}

interface IStmt
{
    void execute(uint tid);
}

// Acts as both the Storage and the Expression
struct Reg<let ID : int> : IExpr
{
    // Thread-local static storage unique to this specialization <ID>
    static float value;

    // IExpr implementation: Read the value
    float eval(uint tid)
    {
        return value;
    }
}

struct BufferView : IExpr
{
    float* data;
    float eval(uint tid) { return data[tid]; }
}

struct ConstantView : IExpr
{
    float value;
    float eval(uint tid) { return value; }
}

struct Broadcast<Inner : IExpr> : IExpr
{
    Inner inner;
    uint rank;
    uint outputShape[8];
    uint innerStrides[8];

    float eval(uint tid)
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
        return inner.eval(innerLinear);
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

    float eval(uint tid)
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
            return left.eval(idx);
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
            return right.eval(idx);
        }
    }
}

// Binary Ops
struct Add<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid) { return l.eval(tid) + r.eval(tid); }
}
struct Sub<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid) { return l.eval(tid) - r.eval(tid); }
}
struct Mul<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid) { return l.eval(tid) * r.eval(tid); }
}
struct Div<L : IExpr, R : IExpr> : IExpr {
    L l; R r;
    float eval(uint tid) { return l.eval(tid) / r.eval(tid); }
}

struct Min<L : IExpr, R : IExpr> : IExpr
{
    L left; R right;
    float eval(uint tid) { return min(left.eval(tid), right.eval(tid)); }
}

struct Max<L : IExpr, R : IExpr> : IExpr
{
    L left; R right;
    float eval(uint tid) { return max(left.eval(tid), right.eval(tid)); }
}

// --- Unary Operations ---

struct Neg<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return -inner.eval(tid); }
}

struct Exp<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return exp(inner.eval(tid)); }
}

struct Log<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return log(inner.eval(tid)); }
}

struct Sin<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return sin(inner.eval(tid)); }
}

struct Cos<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return cos(inner.eval(tid)); }
}

struct Abs<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return abs(inner.eval(tid)); }
}

struct Sqrt<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return sqrt(inner.eval(tid)); }
}

struct Pow<L : IExpr, R : IExpr> : IExpr
{
    L base; R exp;
    // HLSL pow(x, y)
    float eval(uint tid) { return pow(base.eval(tid), exp.eval(tid)); }
}
struct Floor<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return floor(inner.eval(tid)); }
}

struct Ceil<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return ceil(inner.eval(tid)); }
}

struct Rsqrt<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return rsqrt(inner.eval(tid)); }
}

// --- Activations ---

struct Relu<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return max(0.0, inner.eval(tid)); }
}

struct Sigmoid<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) 
    { 
        float v = inner.eval(tid);
        return 1.0 / (1.0 + exp(-v)); 
    }
}

struct Tanh<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) { return tanh(inner.eval(tid)); }
}

struct Silu<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) 
    { 
        float x = inner.eval(tid);
        // x * sigmoid(x)
        return x * (1.0 / (1.0 + exp(-x)));
    }
}

struct Gelu<I : IExpr> : IExpr
{
    I inner;
    float eval(uint tid) 
    { 
        float x = inner.eval(tid);
        // Tanh approximation
        return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    }
}

// Assignment: %ID = Expr
struct Eval<let ID : int, E : IExpr> : IStmt
{
    E expr;

    void execute(uint tid)
    {
        // Write to the static storage for this ID
        Reg<ID>.value = expr.eval(tid);
    }
}

struct Program<int resultReg, each S : IStmt> : IExpr
{
    Tuple<expand each S> stmts;

    float eval(uint tid)
    {
        // Execute each statement in order
        expand (each stmts).execute(tid);
        // Return the value of the result register
        return Reg<resultReg>.value;
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

    // 1. Execute Program
    let result = params.program.eval(tid);

    // 2. Write Output
    params.outputBuffer[tid] = result;
}
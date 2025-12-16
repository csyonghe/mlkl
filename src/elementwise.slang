implementing mlkl;

interface IExpr
{
    float eval(uint tid);
}

interface IStmt
{
    void execute(uint tid);
}

interface IProgram
{
    float execute(uint tid);
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
            if (tid == 0)
            {
                printf("rank = %d\n", rank);
                printf("outputShape[%d] = %d\n", i, outputShape[i]);
                printf("innerStrides[%d] = %d\n", i, innerStrides[i]);
            }
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

struct Program<int resultReg, each S : IStmt> : IProgram
{
    Tuple<expand each S> stmts;

    float execute(uint tid)
    {
        // Execute each statement in order
        expand (each stmts).execute(tid);
        // Return the value of the result register
        return Reg<resultReg>.value;
    }
}

struct MaterializeParams<P : IProgram>
{
    float* outputBuffer;
    uint count;
    P program;
}

[numthreads(256, 1, 1)]
void materialize<P:IProgram>(
    ConstantBuffer<MaterializeParams<P>, ScalarDataLayout> params,
    uint3 id : SV_DispatchThreadID)
{
    uint tid = id.x;
    if (tid >= params.count) return;

    // 1. Execute Program
    let result = params.program.execute(tid);

    // 2. Write Output
    params.outputBuffer[tid] = result;
}
#pragma once

#include "kernel-base.h"

#include <initializer_list>

enum class BinaryOp
{
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow
};
String getSlangBinaryOpName(BinaryOp op);

enum class UnaryOp
{
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Sqrt,
    Relu,
    Sigmoid,
    Tanh,
    Silu,
    Gelu,
    Floor,
    Ceil,
    Rsqrt
};
String getSlangUnaryOpName(UnaryOp op);

struct InputInfo
{
    // For Buffer inputs
    TensorView tensorView;

    // For Uniform Constant inputs
    float scalarValue = 0.0f;

    InputInfo() = default;
    InputInfo(TensorView tensorView)
        : tensorView(tensorView)
    {
    }
    InputInfo(float c)
        : scalarValue(c)
    {
    }
};

class ExprNode;
class ProgramNode;

struct EvalContext
{
    Dictionary<ExprNode*, InputInfo> inputs;
    Dictionary<ExprNode*, Shape>* additionalShapeMap = nullptr;
    Shape getShapeForNode(ExprNode* node) const
    {
        if (auto info = inputs.tryGetValue(node))
        {
            return info->tensorView.shape;
        }
        if (additionalShapeMap)
        {
            if (auto shape = additionalShapeMap->tryGetValue(node))
            {
                return *shape;
            }
        }
        return Shape();
    }
    EvalContext() = default;
};

struct SinkExprEvalContext
{
    Shape logicalShape; // The shape provided by the kernel (e.g., [B, H, S, D])
    TensorView outputBuffer;
};

struct ParameterWriter
{
    List<uint8_t>& buffer;
    Index maxAlignment = 1;
    template<typename T>
    void write(const T& val)
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&val);
        buffer.addRange(ptr, sizeof(T));
    }

    void writeBytes(const void* src, size_t size)
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(src);
        buffer.addRange(ptr, size);
    }

    void finish() { align(maxAlignment); }

    void align(size_t alignment)
    {
        maxAlignment = Slang::Math::Max(maxAlignment, Index(alignment));
        Index newSize = (buffer.getCount() + alignment - 1) / alignment * alignment;
        for (Index i = buffer.getCount(); i < newSize; i++)
        {
            buffer.add(0);
        }
    }
};

class ExprNode : public RefObject
{
public:
    virtual ~ExprNode() = default;

    virtual String getSlangTypeName() const = 0;
    virtual Shape resolveShape(const EvalContext& ctx) const = 0;

    // Packs the DATA required for this node's DEFINITION in the shader.
    // - BufferNode: Packs float*
    // - ConstantNode: Packs float
    // - BroadcastNode: Packs Rank/Shape/Strides (Inner is Reg<ID>, so size 0)
    // - BinaryNode: Packs NOTHING (Operands are Reg<ID>, so size 0)
    virtual void pack(ParameterWriter& writer, const EvalContext& ctx) const = 0;

    virtual size_t getAlignment() const { return sizeof(int32_t); }
};


struct Expr
{
    RefPtr<ExprNode> node;

    Expr() = default;
    Expr(ExprNode* n)
        : node(n)
    {
    }
    Expr(const RefPtr<ExprNode>& n)
        : node(n)
    {
    }

    ExprNode* operator->() { return node; }
    const ExprNode* operator->() const { return node; }
    operator bool() const { return node != nullptr; }
    HashCode getHashCode() const { return node ? node.getHashCode() : 0; }
    bool operator==(const Expr& other) const { return node == other.node; }
};


// SinkExpr represent transformations on the output shape.
// Maps to `ISinkExpr` in Slang.
class SinkExprNode : public RefObject
{
public:
    virtual String getSlangTypeName() const = 0;

    // Recursive Top-Down shape resolution:
    // Takes the logical shape entering THIS node and returns the
    // physical shape of the final terminal buffer.
    virtual Shape resolvePhysicalShape(const Shape& logicalShape) const = 0;

    virtual void pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const = 0;
    virtual size_t getParameterAlignment() const { return sizeof(int32_t); }
};

struct SinkExpr
{
    RefPtr<SinkExprNode> node;
    SinkExpr(SinkExprNode* n)
        : node(n)
    {
    }
};

class BufferNode;

class ProgramNode : public ExprNode
{
public:
    List<RefPtr<ExprNode>> linearNodes;
    List<BufferNode*> bufferNodes;
    int resultRegID = -1;
    Dictionary<ExprNode*, int> nodeToRegID;
    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
    virtual size_t getAlignment() const override;
};

class LeafNode : public ExprNode
{
};

class BufferNode : public LeafNode
{
public:
    uint64_t sequenceNumber; // To identify the buffer at runtime

    String getSlangTypeName() const override { return "BufferView"; }
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
    virtual size_t getAlignment() const { return sizeof(void*); }
};

class ConstantNode : public LeafNode
{
    float value;

public:
    ConstantNode(float v)
        : value(v)
    {
    }
    Shape resolveShape(const EvalContext&) const override { return Shape(); }
    String getSlangTypeName() const override { return "ConstantView"; }
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

// Represents a runtime constant provided via InputInfo
class UniformConstantNode : public LeafNode
{
public:
    UniformConstantNode() = default;

    Shape resolveShape(const EvalContext&) const override { return Shape(); }

    // Reuses the Slang-side "ConstantView" struct
    String getSlangTypeName() const override { return "ConstantView"; }

    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class BroadcastNode : public ExprNode
{
public:
    Expr inner;
    Expr targetShapeOf;

    // Cached linear order of the inner DAG
    RefPtr<ProgramNode> innerProgram;

    BroadcastNode(Expr inner, Expr targetShape);

    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class PermuteNode : public ExprNode
{
public:
    Expr inner;
    List<int> dims; // Permutation

    RefPtr<ProgramNode> innerProgram;

    PermuteNode(Expr inner, ArrayView<int> dims);
    PermuteNode(Expr inner, const std::initializer_list<int>& dims);

    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
    void validateDims();
};

class TransposeNode : public ExprNode
{
public:
    Expr inner;
    int dim0;
    int dim1;

    RefPtr<ProgramNode> innerProgram;

    TransposeNode(Expr inner, int d0, int d1);

    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class GatherNode : public ExprNode
{
public:
    Expr table;
    Expr indices;

    // We cache the program for table/indices dependencies
    RefPtr<ProgramNode> tableProgram;
    RefPtr<ProgramNode> indicesProgram;

    GatherNode(Expr table, Expr indices);

    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class ConcatNode : public ExprNode
{
public:
    Expr left;
    Expr right;
    Expr axis;

    // Compiled inner programs
    RefPtr<ProgramNode> leftProgram;
    RefPtr<ProgramNode> rightProgram;

    ConcatNode(Expr left, Expr right, Expr axis);

    int getAxis(const EvalContext& ctx) const;
    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class BinaryNode : public ExprNode
{
public:
    Expr left;
    Expr right;
    BinaryOp op;
    Shape shape;

    BinaryNode(Expr l, Expr r, BinaryOp op);

    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;

    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class UnaryNode : public ExprNode
{
public:
    Expr inner;
    UnaryOp op;

    UnaryNode(Expr inner, UnaryOp op)
        : inner(inner), op(op)
    {
    }

    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;

    // Normal pack does nothing (Register operands)
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override {}
};

class KernelOutputNode : public LeafNode
{
public:
    Shape resolveShape(const EvalContext&) const override { return Shape(); }
    String getSlangTypeName() const override { return "KernelOutput"; }
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override {}
};

class BufferSinkNode : public SinkExprNode
{
public:
    String getSlangTypeName() const override { return "BufferSink"; }
    Shape resolvePhysicalShape(const Shape& logicalOutputShape) const override
    {
        // The leaf represents the final memory. Its physical shape IS the logical
        // shape that reached it after all transformations.
        return logicalOutputShape;
    }
    virtual void pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const override;
    size_t getParameterAlignment() const override { return sizeof(void*); }
};

class PermuteSinkNode : public SinkExprNode
{
public:
    SinkExpr child;
    List<int> dims; // Permutation mapping
public:
    PermuteSinkNode(SinkExpr child, const std::initializer_list<int>& dims);
    String getSlangTypeName() const override;
    Shape resolvePhysicalShape(const Shape& logicalOutputShape) const override;
    virtual void pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const override;
    size_t getParameterAlignment() const override;
};

class PartitionSinkNode : public SinkExprNode
{
public:
    SinkExpr child;
    uint32_t dimIndex;       // Which coordinate index to split (e.g., 1 for Column)
    uint32_t partitionCount; // Number of partitions to divide dimension `dimIndex` into.
public:
    PartitionSinkNode(SinkExpr child, uint32_t dimIndex, uint32_t partitionCount);
    Shape getChildLogicalShape(const Shape& logicalShape) const;
    String getSlangTypeName() const override;
    Shape resolvePhysicalShape(const Shape& logicalOutputShape) const override;
    virtual void pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const override;
    size_t getParameterAlignment() const override;
};

// Represent an input buffer to the kernel.
Expr buffer();
// Represent a static constant value.
Expr constant(float v);
// Represent a uniform constant value provided at runtime.
Expr uniformConstant();
// Represent the output of the current kernel before applying any output transformations.
Expr kernelOutput();

Expr broadcast(Expr inner, Expr shapeOf);
Expr concat(Expr left, Expr right, Expr axis);
Expr permute(Expr inner, ArrayView<int> dims);
Expr permute(Expr inner, const std::initializer_list<int>& dims);
Expr gather(Expr table, Expr indices);
Expr transpose(Expr inner, int dim0, int dim1);

// Represent the output buffer that the kernel writes to.
SinkExpr bufferSink();

// Represent a permutation transformation (reorders dimensions) on the output buffer.
SinkExpr permute(SinkExpr child, const std::initializer_list<int>& dims);

// Represent a partitioning transformation on the output buffer. The logical output
// tensor of the kernel is split into `partitionCount` partitions along dimension `dimIndex`,
// and each partition becomes a new outer-most dimension in the physical output buffer.
// For example, if a kernel produces the following 2D tensor:
//   row0: [1 2 3 4]
//   row1: [5 6 7 8]
// Then applying partition(bufferSink(), 1, 2) would produce two partitions along dimension 1
// (columns):
//   partition 0:
//     row0: [1 2]
//     row1: [5 6]
//   partition 1:
//     row0: [3 4]
//     row1: [7 8]
// The partitions would be laid out in the physical buffer as:
//   partition 0, row0: [1 2]
//   partition 0, row1: [5 6]
//   partition 1, row0: [3 4]
//   partition 1, row1: [7 8]
// In short, the final physical shape would be
// [partitionCount, originalDim0, originalDim1 / partitionCount].
SinkExpr partition(SinkExpr child, uint32_t dimIndex, uint32_t partitionCount);

Expr min(Expr l, Expr r);
Expr max(Expr l, Expr r);

Expr neg(Expr i); // -Expr
Expr exp(Expr i);
Expr log(Expr i);
Expr sin(Expr i);
Expr cos(Expr i);
Expr abs(Expr i);
Expr sqrt(Expr i);
Expr pow(Expr base, Expr exponent);
Expr floor(Expr i);
Expr ceil(Expr i);
Expr rsqrt(Expr i);

// Activations

// Represent the ReLU activation function, f(x) = max(0, x)
Expr relu(Expr i);
// Represent the Sigmoid activation function, f(x) = 1 / (1 + exp(-x))
Expr sigmoid(Expr i);
// Represent the Tanh activation function, f(x) = tanh(x)
Expr tanh(Expr i);
// Represent the SiLU (Swish) activation function, f(x) = x * sigmoid(x)
Expr silu(Expr i);
// Represent the GELU activation function, f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
Expr gelu(Expr i);

// Operator Overload for Negation
inline Expr operator-(Expr i)
{
    return neg(i);
}

Expr operator+(Expr l, Expr r);
Expr operator-(Expr l, Expr r);
Expr operator*(Expr l, Expr r);
Expr operator/(Expr l, Expr r);

inline Expr operator*(Expr l, float r)
{
    return l * constant(r);
}
inline Expr operator+(Expr l, float r)
{
    return l + constant(r);
}
inline Expr operator-(Expr l, float r)
{
    return l - constant(r);
}
inline Expr operator/(Expr l, float r)
{
    return l / constant(r);
}
inline Expr operator*(float l, Expr r)
{
    return constant(l) * r;
}
inline Expr operator+(float l, Expr r)
{
    return constant(l) + r;
}

// --- Composite Helpers (No new shader code needed!) ---

// clamp(x, min, max) -> min(max(x, min_val), max_val)
inline Expr clamp(Expr x, Expr minVal, Expr maxVal)
{
    return min(max(x, minVal), maxVal);
}

inline Expr clamp(Expr x, float minVal, float maxVal)
{
    return clamp(x, constant(minVal), constant(maxVal));
}

// lerp(a, b, t) -> a + (b - a) * t
inline Expr lerp(Expr a, Expr b, Expr t)
{
    return a + (b - a) * t;
}

// leakyRelu(x, alpha) -> max(x, x * alpha)
inline Expr leakyRelu(Expr x, float alpha)
{
    return max(x, x * alpha);
}

// =========================================================================
// 6. Kernel Wrapper
// =========================================================================

class ElementwiseKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    Expr root;
    InferencingContext* context;

    ProgramNode program;

public:
    ElementwiseKernel(InferencingContext* ctx, Expr rootNode);
    TensorView allocateResultBuffer(
        ElementType elementType,
        const Dictionary<Expr, InputInfo>& inputs);
    void queueExecute(InferencingTask& task, EvalContext& ctx, TensorView output);
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs);
    void queueExecute(InferencingTask& task, TensorView output, ArrayView<InputInfo> inputs);
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs);

    void queueExecute(InferencingTask& task, TensorView output) { queueExecute(task, output, {}); }

    void queueExecute(InferencingTask& task, TensorView output, const InputInfo& input1)
    {
        queueExecute(task, output, {input1});
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const InputInfo& input1,
        const InputInfo& input2)
    {
        queueExecute(task, output, {input1, input2});
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const InputInfo& input1,
        const InputInfo& input2,
        const InputInfo& input3)
    {
        queueExecute(task, output, {input1, input2, input3});
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const InputInfo& input1,
        const InputInfo& input2,
        const InputInfo& input3,
        const InputInfo& input4)
    {
        queueExecute(task, output, {input1, input2, input3, input4});
    }
};

inline EvalContext makeEvalContext(const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (const auto& kv : inputs)
    {
        ctx.inputs.add(kv.first.node.get(), kv.second);
    }
    return ctx;
}

ProgramNode compileExprToProgram(Expr root, int* globalRegCounter);
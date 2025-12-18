#pragma once

#include "kernel-base.h"

#include <initializer_list>

// =========================================================================
// 1. Shape Type
// =========================================================================

struct Shape
{
    Array<int, 8> dims;

    Shape() = default;
    Shape(int i0) { dims.add(i0); }
    Shape(int i0, int i1)
    {
        dims.add(i0);
        dims.add(i1);
    }
    Shape(int i0, int i1, int i2)
    {
        dims.add(i0);
        dims.add(i1);
        dims.add(i2);
    }
    Shape(int i0, int i1, int i2, int i3)
    {
        dims.add(i0);
        dims.add(i1);
        dims.add(i2);
        dims.add(i3);
    }
    Shape(ArrayView<int> d)
    {
        for (auto v : d)
            dims.add(v);
    }
    Shape(const std::initializer_list<int>& d)
    {
        for (auto v : d)
            dims.add(v);
    }

    bool isScalar() const { return dims.getCount() == 0; }
    int getRank() const { return (int)dims.getCount(); }
    int operator[](int i) const { return dims[i]; }
    ArrayView<int> getDims() const
    {
        return ArrayView<int>((int*)dims.getBuffer(), dims.getCount());
    }
    size_t getElementCount() const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const { return !(*this == other); }
    bool isCompatibleWith(const Shape& other) const;
};

// =========================================================================
// 2. Enums & Forward Declarations
// =========================================================================

class ExprNode;

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

// =========================================================================
// 3. Evaluation Context
// =========================================================================
struct InputInfo
{
    // For Buffer inputs
    BufferView buffer;
    Shape shape;

    // For Uniform Constant inputs
    float scalarValue = 0.0f;

    InputInfo() = default;
    InputInfo(Shape shape, BufferView buf)
        : shape(shape), buffer(buf) {};
    InputInfo(float c)
        : scalarValue(c) {};
};

struct EvalContext
{
    Dictionary<ExprNode*, InputInfo> inputs;
    Dictionary<ExprNode*, Shape>* additionalShapeMap = nullptr;
    Shape getShapeForNode(ExprNode* node) const
    {
        if (auto info = inputs.tryGetValue(node))
        {
            return info->shape;
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

// =========================================================================
// 4. Expression Nodes
// =========================================================================

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
};

class ProgramNode : public ExprNode
{
public:
    List<RefPtr<ExprNode>> linearNodes;
    int resultRegID = -1;
    Dictionary<ExprNode*, int> nodeToRegID;
    String getSlangTypeName() const override;
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};


class BufferNode : public ExprNode
{
public:
    String getSlangTypeName() const override { return "BufferView"; }
    Shape resolveShape(const EvalContext& ctx) const override;
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override;
};

class ConstantNode : public ExprNode
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
class UniformConstantNode : public ExprNode
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

class KernelOutputNode : public ExprNode
{
public:
    Shape resolveShape(const EvalContext&) const override { return Shape(); }
    String getSlangTypeName() const override { return "KernelOutput"; }
    void pack(ParameterWriter& writer, const EvalContext& ctx) const override {}
};

// =========================================================================
// 5. Builder API
// =========================================================================

Expr buffer();
Expr constant(float v);
Expr broadcast(Expr inner, Expr shapeOf);
Expr concat(Expr left, Expr right, Expr axis);
Expr permute(Expr inner, ArrayView<int> dims);
Expr permute(Expr inner, const std::initializer_list<int>& dims);
Expr gather(Expr table, Expr indices);
Expr transpose(Expr inner, int dim0, int dim1);
Expr uniformConstant();
Expr kernelOutput();

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
Expr relu(Expr i);
Expr sigmoid(Expr i);
Expr tanh(Expr i);
Expr silu(Expr i);
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
    BufferView allocResultBuffer(const Dictionary<Expr, InputInfo>& inputs);
    void eval(InferencingTask& task, BufferView output, const Dictionary<Expr, InputInfo>& inputs);
};

ProgramNode compileExprToProgram(Expr root, int* globalRegCounter);
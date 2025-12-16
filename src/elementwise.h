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
    Div
};

String getSlangBinaryOpName(BinaryOp op);

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
    rhi::IBuffer* buffer = nullptr;
    size_t offset = 0;

    Shape shape;

    // For Uniform Constant inputs
    float scalarValue = 0.0f;

    InputInfo() = default;
    InputInfo(Shape shape, rhi::IBuffer* buf, size_t off = 0)
        : shape(shape), buffer(buf), offset(off) {};
    InputInfo(Shape shape, const ComPtr<rhi::IBuffer>& buf, size_t off = 0)
        : shape(shape), buffer(buf), offset(off) {};
    InputInfo(float c)
        : scalarValue(c) {};
};

struct EvalContext
{
    Dictionary<ExprNode*, InputInfo> inputs;
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

    BroadcastNode(Expr inner, Expr targetShape);

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

// =========================================================================
// 5. Builder API
// =========================================================================

Expr buffer();
Expr constant(float v);
Expr broadcast(Expr inner, Expr shapeOf);
Expr uniformConstant();

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

// =========================================================================
// 6. Kernel Wrapper
// =========================================================================

class ElementwiseKernel : public RefObject
{
    ComPtr<rhi::IComputePipeline> pipeline;
    Expr root;
    InferencingContext* context;

    List<ExprNode*> linearNodes;

public:
    ElementwiseKernel(InferencingContext* ctx, Expr rootNode);
    ComPtr<rhi::IBuffer> eval(InferencingTask& task, const Dictionary<Expr, InputInfo>& inputs);
};
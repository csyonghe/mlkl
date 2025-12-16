#include "elementwise.h"

#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// =========================================================================
// Global Pipeline Cache
// =========================================================================

struct ContextPipelineCache
{
    Dictionary<String, ComPtr<rhi::IComputePipeline>> pipelines;
};

static std::mutex gCacheMutex;
static Dictionary<InferencingContext*, ContextPipelineCache> gPipelineCache;

// =========================================================================
// Helpers
// =========================================================================

String getSlangBinaryOpName(BinaryOp op)
{
    switch (op)
    {
    case BinaryOp::Add:
        return "Add";
    case BinaryOp::Sub:
        return "Sub";
    case BinaryOp::Mul:
        return "Mul";
    case BinaryOp::Div:
        return "Div";
    default:
        return "Unknown";
    }
}

// [Shape Implementation Unchanged] ...
size_t Shape::getElementCount() const
{
    if (isScalar())
        return 1;
    size_t count = 1;
    for (Index i = 0; i < dims.getCount(); i++)
        count *= dims[i];
    return count;
}
bool Shape::operator==(const Shape& other) const
{
    if (dims.getCount() != other.dims.getCount())
        return false;
    for (Index i = 0; i < dims.getCount(); i++)
        if (dims[i] != other.dims[i])
            return false;
    return true;
}
bool Shape::isCompatibleWith(const Shape& other) const
{
    if (this->isScalar() || other.isScalar())
        return true;
    return *this == other;
}

// =========================================================================
// Node Implementations
// =========================================================================

// --- BufferNode ---

Shape BufferNode::resolveShape(const EvalContext& ctx) const
{
    InputInfo info;
    if (ctx.inputs.tryGetValue((ExprNode*)this, info))
        return info.shape;
    return Shape();
}

void BufferNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    InputInfo info;
    writer.align(8);
    if (ctx.inputs.tryGetValue((ExprNode*)this, info))
    {
        if (info.buffer)
        {
            writer.write(info.buffer->getDeviceAddress() + info.offset);
            return;
        }
    }
    writer.write<uint64_t>(0);
}

// --- ConstantNode ---
void ConstantNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    writer.write(value);
}

// --- UniformConstantNode ---
void UniformConstantNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    InputInfo info;
    if (ctx.inputs.tryGetValue((ExprNode*)this, info))
    {
        // Scalar value is packed directly (size 4 bytes)
        writer.write(info.scalarValue);
        return;
    }
    writer.write(0.0f);
}


// --- BroadcastNode ---

BroadcastNode::BroadcastNode(Expr inner, Expr targetShape)
    : inner(inner), targetShapeOf(targetShape)
{
}

String BroadcastNode::getSlangTypeName() const
{
    return "Broadcast<" + inner.node->getSlangTypeName() + ">";
}

Shape BroadcastNode::resolveShape(const EvalContext& ctx) const
{
    auto outputShape = targetShapeOf->resolveShape(ctx);
    auto inputShape = inner->resolveShape(ctx);

    // 1. Allow scalars (Rank 0 broadcasts to anything)
    if (inputShape.isScalar())
        return outputShape;

    // 2. Allow Inner Rank <= Target Rank
    if (inputShape.getRank() > outputShape.getRank())
    {
        throw std::runtime_error("BroadcastNode: Inner rank cannot exceed target rank");
    }

    // 3. Right-Aligned Compatibility Check
    int rankDiff = outputShape.getRank() - inputShape.getRank();

    for (int i = 0; i < inputShape.getRank(); i++)
    {
        // Map Inner[i] to Output[i + rankDiff]
        int inDim = inputShape[i];
        int outDim = outputShape[i + rankDiff];

        // Standard Broadcasting Rule: Dims must match or one must be 1
        if (inDim != outDim && inDim != 1)
        {
            // Optional: You might allow outDim=1 if inDim!=1?
            // Usually broadcasting allows 1 -> N, but not N -> M.
            throw std::runtime_error("BroadcastNode: Dimension mismatch");
        }
    }

    return outputShape;
}

void BroadcastNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    // Use pack inlined to deeply inline the inner nodes.
    inner.node->packInlined(writer, ctx);

    // We compute metadata at runtime based on shapes
    Shape innerShape = inner.node->resolveShape(ctx);
    const Shape& targetShape = targetShapeOf->resolveShape(ctx);

    int rankDiff = targetShape.getRank() - innerShape.getRank();

    // Compute Strides
    List<int> strides;
    for (Index i = 0; i < targetShape.getRank(); i++)
        strides.add(0);

    // Inner contiguous strides
    List<int> trueInnerStrides;
    for (Index i = 0; i < innerShape.getRank(); i++)
        trueInnerStrides.add(0);
    int s = 1;
    for (int i = innerShape.getRank() - 1; i >= 0; i--)
    {
        trueInnerStrides[i] = s;
        s *= innerShape[i];
    }

    for (int i = targetShape.getRank() - 1; i >= 0; i--)
    {
        int innerIdx = i - rankDiff;
        if (innerIdx >= 0)
        {
            int iDim = innerShape[innerIdx];
            int tDim = targetShape[i];
            if (iDim == tDim)
                strides[i] = trueInnerStrides[innerIdx];
            else
                strides[i] = 0; // Broadcast (dim=1 or mismatch assumed broadcast)
        }
    }

    // Pack Metadata
    uint32_t rank = (uint32_t)targetShape.getRank();
    writer.write(rank);

    uint32_t outShapeArr[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (Index i = 0; i < targetShape.getRank(); ++i)
        outShapeArr[i] = (uint32_t)targetShape[i];
    writer.writeBytes(outShapeArr, sizeof(outShapeArr));

    uint32_t strideArr[8] = {0};
    for (Index i = 0; i < strides.getCount(); ++i)
        strideArr[i] = (uint32_t)strides[i];
    writer.writeBytes(strideArr, sizeof(strideArr));
}

// --- BinaryNode ---
BinaryNode::BinaryNode(Expr l, Expr r, BinaryOp op)
    : left(l), right(r), op(op)
{
}

String BinaryNode::getSlangTypeName() const
{
    return getSlangBinaryOpName(op) + "<" + left.node->getSlangTypeName() + "," +
           right.node->getSlangTypeName() + ">";
}

Shape BinaryNode::resolveShape(const EvalContext& ctx) const
{
    Shape l = left.node->resolveShape(ctx);
    Shape r = right.node->resolveShape(ctx);
    if (!l.isCompatibleWith(r))
    {
        throw std::runtime_error("BinaryNode: Operand shapes are not compatible");
    }
    if (!l.isScalar())
        return l;
    return r;
}

void BinaryNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    // Operands are Reg<ID>, size 0. Nothing to pack!
}

void BinaryNode::packInlined(ParameterWriter& writer, const EvalContext& ctx) const
{
    left.node->packInlined(writer, ctx);
    right.node->packInlined(writer, ctx);
}

// [Factories]
Expr buffer()
{
    return Expr(new BufferNode());
}
Expr constant(float v)
{
    return Expr(new ConstantNode(v));
}
Expr uniformConstant()
{
    return Expr(new UniformConstantNode());
}
Expr broadcast(Expr inner, Expr shapeOf)
{
    return Expr(new BroadcastNode(inner, shapeOf));
}

Expr operator+(Expr l, Expr r)
{
    return Expr(new BinaryNode(l, r, BinaryOp::Add));
}
Expr operator-(Expr l, Expr r)
{
    return Expr(new BinaryNode(l, r, BinaryOp::Sub));
}
Expr operator*(Expr l, Expr r)
{
    return Expr(new BinaryNode(l, r, BinaryOp::Mul));
}
Expr operator/(Expr l, Expr r)
{
    return Expr(new BinaryNode(l, r, BinaryOp::Div));
}


// =========================================================================
// SSA Type Generator
// =========================================================================

struct SSAGenContext
{
    std::unordered_map<ExprNode*, int> regIDs;
    List<ExprNode*> topoOrder;
    std::set<ExprNode*> visited;
    int regCounter = 0;
};

void topoVisit(ExprNode* node, SSAGenContext& ctx)
{
    if (ctx.visited.count(node))
        return;

    if (auto b = dynamic_cast<BinaryNode*>(node))
    {
        topoVisit(b->left.node, ctx);
        topoVisit(b->right.node, ctx);
    }
    else if (auto br = dynamic_cast<BroadcastNode*>(node))
    {
        // Stop recursion here to inline inner nodes.
    }

    ctx.visited.insert(node);
    ctx.topoOrder.add(node);
    ctx.regIDs[node] = ctx.regCounter++;
}

String genExprType(ExprNode* node, SSAGenContext& ctx);

String genDefType(ExprNode* node, SSAGenContext& ctx)
{
    if (auto b = dynamic_cast<BinaryNode*>(node))
    {
        return getSlangBinaryOpName(b->op) + "<" + genExprType(b->left.node, ctx) + "," +
               genExprType(b->right.node, ctx) + ">";
    }
    else if (auto br = dynamic_cast<BroadcastNode*>(node))
    {
        return "Broadcast<" + genExprType(br->inner.node, ctx) + ">";
    }
    else if (dynamic_cast<BufferNode*>(node))
        return "BufferView";
    else if (dynamic_cast<ConstantNode*>(node))
        return "ConstantView";
    else if (dynamic_cast<UniformConstantNode*>(node))
        return "ConstantView";
    return "Error";
}

String genExprType(ExprNode* node, SSAGenContext& ctx)
{
    auto it = ctx.regIDs.find(node);
    if (it != ctx.regIDs.end())
    {
        // Visited node -> Use Register
        std::stringstream ss;
        ss << "Reg<" << it->second << ">";
        return ss.str().c_str();
    }
    // CHANGE: Not visited (child of Broadcast) -> Generate full definition (Inline)
    return genDefType(node, ctx);
}

// =========================================================================
// Pipeline Implementation
// =========================================================================

ElementwiseKernel::ElementwiseKernel(InferencingContext* ctx, Expr rootNode)
    : context(ctx), root(rootNode)
{
    // 1. Traverse and Generate Signature
    SSAGenContext ssaCtx;
    topoVisit(root.node, ssaCtx);

    int resultReg = ssaCtx.regIDs[root.node];

    StringBuilder typeBuilder;
    typeBuilder << "Program<" << resultReg;

    // Append Statements
    for (ExprNode* node : ssaCtx.topoOrder)
    {
        int id = ssaCtx.regIDs[node];
        String defType = genDefType(node, ssaCtx);
        typeBuilder << ", Eval<" << id << ", " << defType << ">";

        linearNodes.add(node);
    }

    typeBuilder << ">";

    String finalProgramType = typeBuilder.produceString();

    String typeArgs[] = {finalProgramType};
    pipeline = ctx->createComputePipeline("materialize", makeArrayView(typeArgs));
}

ComPtr<rhi::IBuffer> ElementwiseKernel::eval(
    InferencingTask& task,
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
        ctx.inputs.add(it.first.node, it.second);

    // 1. VALIDATION: Resolve shapes for the whole tree to ensure compatibility
    Shape resultShape = root.node->resolveShape(ctx);
    size_t count = resultShape.getElementCount();
    auto outBuf = task.allocateBuffer("eval_out", count * sizeof(float));

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    writer.write(outBuf->getDeviceAddress());
    writer.write((uint32_t)count);

    for (ExprNode* node : linearNodes)
    {
        // pack() for BroadcastNode also validates dimensions implicitly
        node->pack(writer, ctx);
    }
    writer.finish();

    uint32_t groups = (uint32_t)((count + 255) / 256);
    task.dispatchKernel(
        pipeline,
        groups,
        1,
        1,
        paramData.getBuffer(),
        (uint32_t)paramData.getCount());

    return ComPtr<rhi::IBuffer>(outBuf);
}
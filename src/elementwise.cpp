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
    case BinaryOp::Min:
        return "Min";
    case BinaryOp::Max:
        return "Max";
    case BinaryOp::Pow:
        return "Pow";
    default:
        return "Unknown";
    }
}

String getSlangUnaryOpName(UnaryOp op)
{
    switch (op)
    {
    case UnaryOp::Neg:
        return "Neg";
    case UnaryOp::Exp:
        return "Exp";
    case UnaryOp::Log:
        return "Log";
    case UnaryOp::Sin:
        return "Sin";
    case UnaryOp::Cos:
        return "Cos";
    case UnaryOp::Abs:
        return "Abs";
    case UnaryOp::Sqrt:
        return "Sqrt";
    case UnaryOp::Relu:
        return "Relu";
    case UnaryOp::Sigmoid:
        return "Sigmoid";
    case UnaryOp::Tanh:
        return "Tanh";
    case UnaryOp::Silu:
        return "Silu";
    case UnaryOp::Gelu:
        return "Gelu";
    case UnaryOp::Floor:
        return "Floor";
    case UnaryOp::Ceil:
        return "Ceil";
    case UnaryOp::Rsqrt:
        return "Rsqrt";
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

List<int> computeDenseStrides(const Shape& shape)
{
    List<int> strides;
    for (int i = 0; i < shape.getRank(); i++)
        strides.add(0);

    int s = 1;
    for (int i = shape.getRank() - 1; i >= 0; i--)
    {
        strides[i] = s;
        s *= shape[i];
    }
    return strides;
}

String genExprType(ExprNode* node, const Dictionary<ExprNode*, int>& mapNodeToId);
String genDefType(ExprNode* node, const Dictionary<ExprNode*, int>& mapNodeToId);
ProgramNode compileExprToProgram(Expr root, int* globalRegCounter);

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
            writer.write(info.buffer.getDeviceAddress() + info.offset);
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
    return "Broadcast<" + innerProgram->getSlangTypeName() + ">";
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
    // Pack parameters for the inner program first.
    innerProgram->pack(writer, ctx);

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

// --- PermuteNode ---

PermuteNode::PermuteNode(Expr inner, ArrayView<int> dims)
    : inner(inner), dims(dims)
{
}

String PermuteNode::getSlangTypeName() const
{
    return "Permute<" + innerProgram->getSlangTypeName() + ">";
}

Shape PermuteNode::resolveShape(const EvalContext& ctx) const
{
    Shape innerShape = inner.node->resolveShape(ctx);
    if (innerShape.getRank() != dims.getCount())
        throw std::runtime_error("Permute: Rank mismatch with permutation indices");

    List<int> newDims;
    for (int i : dims)
    {
        if (i < 0 || i >= innerShape.getRank())
            throw std::runtime_error("Permute: Invalid dimension index");
        newDims.add(innerShape[i]);
    }
    return Shape(newDims.getArrayView());
}

void PermuteNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    innerProgram->pack(writer, ctx);

    Shape innerShape = inner.node->resolveShape(ctx);
    Shape outShape = resolveShape(ctx);

    List<int> innerStrides = computeDenseStrides(innerShape);
    List<int> mappedStrides;

    // For output dimension i, it corresponds to inner dimension dims[i].
    // So if we move along output[i], we move along inner[dims[i]].
    // Stride is innerStrides[dims[i]].
    for (int i = 0; i < dims.getCount(); ++i)
    {
        mappedStrides.add(innerStrides[dims[i]]);
    }

    uint32_t rank = (uint32_t)outShape.getRank();
    writer.write(rank);

    uint32_t outDimsArr[8] = {1};
    for (int i = 0; i < outShape.getRank(); ++i)
        outDimsArr[i] = (uint32_t)outShape[i];
    writer.writeBytes(outDimsArr, sizeof(outDimsArr));

    uint32_t mapStrideArr[8] = {0};
    for (int i = 0; i < mappedStrides.getCount(); ++i)
        mapStrideArr[i] = (uint32_t)mappedStrides[i];
    writer.writeBytes(mapStrideArr, sizeof(mapStrideArr));
}

// --- TransposeNode ---

TransposeNode::TransposeNode(Expr inner, int d0, int d1)
    : inner(inner), dim0(d0), dim1(d1)
{
}

String TransposeNode::getSlangTypeName() const
{
    return "Transpose<" + innerProgram->getSlangTypeName() + ">";
}

Shape TransposeNode::resolveShape(const EvalContext& ctx) const
{
    Shape s = inner.node->resolveShape(ctx);
    int rank = s.getRank();

    // Handle negative indices
    int d0 = dim0 < 0 ? rank + dim0 : dim0;
    int d1 = dim1 < 0 ? rank + dim1 : dim1;

    if (d0 < 0 || d0 >= rank || d1 < 0 || d1 >= rank)
        throw std::runtime_error("Transpose: Invalid dimension index");

    // Swap dimensions
    List<int> newDims;
    for (int i = 0; i < rank; i++)
        newDims.add(s[i]);
    std::swap(newDims[d0], newDims[d1]);

    return Shape(newDims.getArrayView());
}

void TransposeNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    innerProgram->pack(writer, ctx);

    Shape innerShape = inner.node->resolveShape(ctx);
    Shape outShape = resolveShape(ctx);
    List<int> innerStrides = computeDenseStrides(innerShape);
    int rank = innerShape.getRank();

    // Handle negative indices again for packing
    int d0 = dim0 < 0 ? rank + dim0 : dim0;
    int d1 = dim1 < 0 ? rank + dim1 : dim1;

    writer.write((uint32_t)d0);
    writer.write((uint32_t)d1);

    uint32_t shapeArr[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < rank; ++i)
        shapeArr[i] = (uint32_t)outShape[i];
    writer.writeBytes(shapeArr, sizeof(shapeArr));

    uint32_t strideArr[8] = {0};
    for (int i = 0; i < rank; ++i)
        strideArr[i] = (uint32_t)innerStrides[i];
    writer.writeBytes(strideArr, sizeof(strideArr));
}

// --- ConcatNode ---

ConcatNode::ConcatNode(Expr left, Expr right, Expr axis)
    : left(left), right(right), axis(axis)
{
}

int ConcatNode::getAxis(const EvalContext& ctx) const
{
    auto axisInput = ctx.inputs.tryGetValue(axis.node);
    if (!axisInput)
        throw std::runtime_error("axis of concat not provided");
    if (axisInput->buffer)
        throw std::runtime_error("axis of concat must be static const");
    return (int)axisInput->scalarValue;
}

String ConcatNode::getSlangTypeName() const
{
    return StringBuilder() << "Concat<" << leftProgram->getSlangTypeName() << ","
                           << rightProgram->getSlangTypeName() << ">";
}

Shape ConcatNode::resolveShape(const EvalContext& ctx) const
{
    Shape l = left.node->resolveShape(ctx);
    Shape r = right.node->resolveShape(ctx);

    if (l.getRank() != r.getRank())
        throw std::runtime_error("Concat: Rank mismatch");

    // Handle negative axis
    int trueAxis = getAxis(ctx);
    if (trueAxis < 0)
        trueAxis += l.getRank();

    if (trueAxis < 0 || trueAxis >= l.getRank())
        throw std::runtime_error("Concat: Invalid axis");

    List<int> dims;
    for (int i = 0; i < l.getRank(); i++)
    {
        if (i == trueAxis)
        {
            dims.add(l[i] + r[i]);
        }
        else
        {
            if (l[i] != r[i])
                throw std::runtime_error("Concat: Dimension mismatch off-axis");
            dims.add(l[i]);
        }
    }
    return Shape(dims.getArrayView());
}

void ConcatNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    // 1. Pack Inner Programs
    leftProgram->pack(writer, ctx);
    rightProgram->pack(writer, ctx);

    // 2. Resolve Shapes
    Shape lShape = left.node->resolveShape(ctx);
    Shape rShape = right.node->resolveShape(ctx);
    Shape outShape = resolveShape(ctx); // Output shape

    int trueAxis = getAxis(ctx);

    if (trueAxis < 0)
        trueAxis += lShape.getRank();

    // 3. Compute Metadata
    List<int> lStrides = computeDenseStrides(lShape);
    List<int> rStrides = computeDenseStrides(rShape);

    // 4. Pack Scalars (Axis, Split, Rank)
    // Axis
    writer.write((uint32_t)trueAxis);
    // Split Point (Size of Left at Axis)
    writer.write((uint32_t)lShape[trueAxis]);
    // Rank
    writer.write((uint32_t)outShape.getRank());

    // 5. Pack Arrays (OutputDims, LeftStrides, RightStrides)

    // Output Dims
    uint32_t outDimsArr[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < outShape.getRank(); ++i)
        outDimsArr[i] = (uint32_t)outShape[i];
    writer.writeBytes(outDimsArr, sizeof(outDimsArr));

    // Left Strides
    uint32_t lStridesArr[8] = {0};
    for (int i = 0; i < lStrides.getCount(); ++i)
        lStridesArr[i] = (uint32_t)lStrides[i];
    writer.writeBytes(lStridesArr, sizeof(lStridesArr));

    // Right Strides
    uint32_t rStridesArr[8] = {0};
    for (int i = 0; i < rStrides.getCount(); ++i)
        rStridesArr[i] = (uint32_t)rStrides[i];
    writer.writeBytes(rStridesArr, sizeof(rStridesArr));
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

// --- UnaryNode ---

String UnaryNode::getSlangTypeName() const
{
    return getSlangUnaryOpName(op) + "<" + inner.node->getSlangTypeName() + ">";
}

Shape UnaryNode::resolveShape(const EvalContext& ctx) const
{
    // Shape passes through Unary ops unchanged
    return inner.node->resolveShape(ctx);
}

// --- ProgramNode ---

String ProgramNode::getSlangTypeName() const
{
    StringBuilder typeBuilder;
    typeBuilder << "Program<" << resultRegID;

    // Append Statements
    for (ExprNode* node : linearNodes)
    {
        auto id = nodeToRegID.tryGetValue(node);
        if (!id)
            throw std::runtime_error("ProgramNode: Node missing in nodeToRegID map");
        String defType = genDefType(node, nodeToRegID);
        typeBuilder << ", Eval<" << *id << ", " << defType << ">";
    }

    typeBuilder << ">";
    return typeBuilder.toString();
}

Shape ProgramNode::resolveShape(const EvalContext& ctx) const
{
    if (linearNodes.getCount() == 0)
        return Shape();

    Dictionary<ExprNode*, Shape> shapes;
    EvalContext localCtx = ctx;
    localCtx.additionalShapeMap = &shapes;
    for (ExprNode* node : linearNodes)
    {
        shapes[node] = node->resolveShape(localCtx);
    }
    return shapes[linearNodes.getLast()];
}

void ProgramNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    for (ExprNode* node : linearNodes)
    {
        node->pack(writer, ctx);
    }
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
Expr kernelOutput()
{
    return Expr(new KernelOutputNode());
}
Expr broadcast(Expr inner, Expr shapeOf)
{
    return Expr(new BroadcastNode(inner, shapeOf));
}
Expr permute(Expr inner, ArrayView<int> dims)
{
    return Expr(new PermuteNode(inner, dims));
}
Expr transpose(Expr inner, int dim0, int dim1)
{
    return Expr(new TransposeNode(inner, dim0, dim1));
}
Expr concat(Expr left, Expr right, Expr axis)
{
    return Expr(new ConcatNode(left, right, axis));
}
Expr min(Expr l, Expr r)
{
    return Expr(new BinaryNode(l, r, BinaryOp::Min));
}
Expr max(Expr l, Expr r)
{
    return Expr(new BinaryNode(l, r, BinaryOp::Max));
}

Expr neg(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Neg));
}
Expr exp(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Exp));
}
Expr log(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Log));
}
Expr sin(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Sin));
}
Expr cos(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Cos));
}
Expr abs(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Abs));
}
Expr sqrt(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Sqrt));
}

Expr relu(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Relu));
}
Expr sigmoid(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Sigmoid));
}
Expr tanh(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Tanh));
}
Expr silu(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Silu));
}
Expr gelu(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Gelu));
}
Expr pow(Expr base, Expr exponent)
{
    return Expr(new BinaryNode(base, exponent, BinaryOp::Pow));
}
Expr floor(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Floor));
}
Expr ceil(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Ceil));
}
Expr rsqrt(Expr i)
{
    return Expr(new UnaryNode(i, UnaryOp::Rsqrt));
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
    Dictionary<ExprNode*, int> regIDs;
    List<ExprNode*> topoOrder;
    std::set<ExprNode*> visited;
    int* globalRegCounter = nullptr;
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
    else if (auto u = dynamic_cast<UnaryNode*>(node))
    {
        topoVisit(u->inner.node, ctx);
    }
    else if (auto br = dynamic_cast<BroadcastNode*>(node))
    {
        // Local SSA generation for the inner graph
        RefPtr<ProgramNode> innerProgram = new ProgramNode();
        *innerProgram = compileExprToProgram(br->inner, ctx.globalRegCounter);
        br->innerProgram = innerProgram;
    }
    else if (auto p = dynamic_cast<PermuteNode*>(node))
    {
        RefPtr<ProgramNode> innerProgram = new ProgramNode();
        *innerProgram = compileExprToProgram(p->inner, ctx.globalRegCounter);
        p->innerProgram = innerProgram;
    }
    else if (auto t = dynamic_cast<TransposeNode*>(node))
    {
        RefPtr<ProgramNode> innerProgram = new ProgramNode();
        *innerProgram = compileExprToProgram(t->inner, ctx.globalRegCounter);
        t->innerProgram = innerProgram;
    }
    else if (auto c = dynamic_cast<ConcatNode*>(node))
    {
        // Concat creates TWO inner programs for conditional execution
        c->leftProgram = new ProgramNode();
        *c->leftProgram = compileExprToProgram(c->left, ctx.globalRegCounter);

        c->rightProgram = new ProgramNode();
        *c->rightProgram = compileExprToProgram(c->right, ctx.globalRegCounter);
    }
    ctx.visited.insert(node);
    ctx.topoOrder.add(node);
    ctx.regIDs[node] = (*ctx.globalRegCounter)++;
}

ProgramNode compileExprToProgram(Expr root, int* globalRegCounter)
{
    SSAGenContext ssaCtx;
    ssaCtx.globalRegCounter = globalRegCounter;
    topoVisit(root.node, ssaCtx);

    int resultReg = ssaCtx.regIDs[root.node];

    ProgramNode program;
    program.linearNodes = _Move(ssaCtx.topoOrder);
    program.resultRegID = ssaCtx.regIDs[root.node];
    program.nodeToRegID = _Move(ssaCtx.regIDs);
    return _Move(program);
}

String genDefType(ExprNode* node, const Dictionary<ExprNode*, int>& mapNodeToId)
{
    if (auto b = dynamic_cast<BinaryNode*>(node))
    {
        return getSlangBinaryOpName(b->op) + "<" + genExprType(b->left.node, mapNodeToId) + "," +
               genExprType(b->right.node, mapNodeToId) + ">";
    }
    else if (auto u = dynamic_cast<UnaryNode*>(node))
    {
        return getSlangUnaryOpName(u->op) + "<" + genExprType(u->inner.node, mapNodeToId) + ">";
    }
    return node->getSlangTypeName();
}

String genExprType(ExprNode* node, const Dictionary<ExprNode*, int>& mapNodeToId)
{
    if (auto it = mapNodeToId.tryGetValue(node))
    {
        // Visited node -> Use Register
        StringBuilder sb;
        sb << "Reg<" << *it << ">";
        return sb.produceString();
    }
    // CHANGE: Not visited (child of Broadcast) -> Generate full definition (Inline)
    return genDefType(node, mapNodeToId);
}

// =========================================================================
// Pipeline Implementation
// =========================================================================

ElementwiseKernel::ElementwiseKernel(InferencingContext* ctx, Expr rootNode)
    : context(ctx), root(rootNode)
{
    int globalRegCounter = 0;
    program = compileExprToProgram(root, &globalRegCounter);

    String finalProgramType = program.getSlangTypeName();

    String typeArgs[] = {finalProgramType};
    pipeline = ctx->createComputePipeline("materialize", makeArrayView(typeArgs));
}

BufferView ElementwiseKernel::allocResultBuffer(const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
    {
        ctx.inputs.add(it.first.node, it.second);
    }
    Shape resultShape = root.node->resolveShape(ctx);
    size_t count = resultShape.getElementCount();
    return context->allocScratchBuffer(count * sizeof(float), "elementwise");
}

void ElementwiseKernel::eval(
    InferencingTask& task,
    BufferView output,
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
        ctx.inputs.add(it.first.node, it.second);

    Shape resultShape = root.node->resolveShape(ctx);
    size_t count = resultShape.getElementCount();

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    writer.write(output.getDeviceAddress());
    writer.write((uint32_t)count);

    program.pack(writer, ctx);
    writer.finish();

    uint32_t groups = (uint32_t)((count + 255) / 256);
    task.dispatchKernel(
        pipeline,
        groups,
        1,
        1,
        paramData.getBuffer(),
        (uint32_t)paramData.getCount());
}
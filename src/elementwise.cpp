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

String genExprType(
    ExprNode* node,
    const Dictionary<ExprNode*, int>& mapNodeToId,
    ElementType elemType);
String genDefType(
    ExprNode* node,
    const Dictionary<ExprNode*, int>& mapNodeToId,
    ElementType elemType);
ProgramNode compileExprToProgram(Expr root, int* globalRegCounter);

// =========================================================================
// Node Implementations
// =========================================================================

// --- BufferNode ---

Shape BufferNode::resolveShape(const EvalContext& ctx) const
{
    InputInfo info;
    if (ctx.inputs.tryGetValue((ExprNode*)this, info))
        return info.tensorView.shape;
    return Shape();
}

void BufferNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    InputInfo info = {};
    writer.align(8);
    bool hasAddr = false;
    if (ctx.inputs.tryGetValue((ExprNode*)this, info))
    {
        if (info.tensorView)
        {
            writer.write(info.tensorView.getDeviceAddress());
            hasAddr = true;
        }
    }
    if (!hasAddr)
    {
        // If a buffer isn't provided, write nullptr for buffer address.
        writer.write<uint64_t>(0);
    }
    writer.write<int>(info.tensorView.shape.getRank());
    auto strides = computeDenseStrides(info.tensorView.shape);
    writer.writeBytes(strides.getBuffer(), strides.getCapacity() * sizeof(uint32_t));
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

String BroadcastNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return "Broadcast<" + elemTypeName + ", " + innerProgram->getSlangTypeName(elemType) + ">";
}

Shape BroadcastNode::resolveShape(const EvalContext& ctx) const
{
    auto outputShape = targetShapeOf->resolveShape(ctx);
    auto inputShape = inner->resolveShape(ctx);

    // 1. Allow scalars (Rank 0 broadcasts to anything)
    if (inputShape.isScalar())
        return outputShape;

    // 2. Allow Inner Rank <= Target Rank
    if (inputShape.getRank() != outputShape.getRank())
    {
        throw std::runtime_error("BroadcastNode: rank mismatch!");
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

    // Compute innerStrides: 0 means broadcast (zero out coord), 1 means pass through
    // When a dimension is broadcast (inner dim = 1, target dim > 1), we set stride to 0
    // When dimensions match, we set stride to 1 (pass through the coord)
    uint32_t innerStrides[8] = {0};

    for (int i = 0; i < targetShape.getRank(); i++)
    {
        int innerIdx = i - rankDiff;
        if (innerIdx >= 0 && innerIdx < innerShape.getRank())
        {
            int iDim = innerShape[innerIdx];
            int tDim = targetShape[i];
            // If dimensions match, pass through; if inner is 1 (broadcast), zero out
            innerStrides[i] = (iDim == tDim) ? 1 : 0;
        }
        else
        {
            // Dimension doesn't exist in inner (leading dims from rank difference)
            innerStrides[i] = 0;
        }
    }

    // Pack Metadata
    uint32_t rank = (uint32_t)targetShape.getRank();
    writer.write(rank);

    writer.writeBytes(innerStrides, sizeof(innerStrides));
}

// --- PermuteNode ---

PermuteNode::PermuteNode(Expr inner, ArrayView<int> dims)
    : inner(inner), dims(dims)
{
    validateDims();
}

PermuteNode::PermuteNode(Expr inner, const std::initializer_list<int>& dims)
    : inner(inner)
{
    for (auto d : dims)
        this->dims.add(d);
    validateDims();
}


String PermuteNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return "Permute<" + elemTypeName + ", " + innerProgram->getSlangTypeName(elemType) + ">";
}

Shape PermuteNode::resolveShape(const EvalContext& ctx) const
{
    Shape innerShape = inner.node->resolveShape(ctx);
    if (innerShape.isScalar())
        return innerShape;
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

    uint32_t rank = (uint32_t)outShape.getRank();
    writer.write(rank);

    // Build permutation map: permMap[i] = j means output coord[i] maps to inner coord[j]
    // This is exactly the dims array we already have.
    uint32_t permMap[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < (int)dims.getCount(); ++i)
    {
        permMap[i] = (uint32_t)dims[i];
    }

    writer.writeBytes(permMap, sizeof(permMap));
}

void PermuteNode::validateDims()
{
    Index maxRank = dims.getCount();
    if (maxRank > 8)
        throw std::runtime_error("Permute: Max rank supported is 8");

    Array<bool, 8> seen;
    seen.setCount(maxRank);
    for (Index i = 0; i < seen.getCount(); i++)
        seen[i] = false;
    for (auto d : dims)
    {
        if (d < 0 || d >= maxRank)
            throw std::runtime_error("Permute: Dimension index out of range");
        if (seen[d])
            throw std::runtime_error("Permute: Duplicate dimension index");
        seen[d] = true;
    }
}

// --- GatherNode ---

GatherNode::GatherNode(Expr table, Expr indices)
    : table(table), indices(indices)
{
}

String GatherNode::getSlangTypeName(ElementType elemType) const
{
    // Gather<T, TableProgram, IndicesProgram>
    String elemTypeName = getSlangElementTypeName(elemType);
    return "Gather<" + elemTypeName + ", " + tableProgram->getSlangTypeName(elemType) + ", " +
           indicesProgram->getSlangTypeName(elemType) + ">";
}

Shape GatherNode::resolveShape(const EvalContext& ctx) const
{
    Shape tableShape = table.node->resolveShape(ctx);
    Shape idxShape = indices.node->resolveShape(ctx);

    // Basic Embedding Validation:
    // Table should be [NumClasses, Dim]
    // Indices should be [Batch]
    // Output is [Batch, Dim]

    if (tableShape.getRank() != 2)
        throw std::runtime_error("Gather: Table must be 2D");
    if (idxShape.getRank() != 1)
        throw std::runtime_error("Gather: Indices must be 1D");

    return Shape{idxShape[0], tableShape[1]};
}

void GatherNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    // Pack dependencies - no additional fields needed since coords are passed directly
    tableProgram->pack(writer, ctx);
    indicesProgram->pack(writer, ctx);
}

// --- TransposeNode ---

TransposeNode::TransposeNode(Expr inner, int d0, int d1)
    : inner(inner), dim0(d0), dim1(d1)
{
}

String TransposeNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return "Transpose<" + elemTypeName + ", " + innerProgram->getSlangTypeName(elemType) + ">";
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
    // First, pack the innerProgram in the main linear list.
    innerProgram->pack(writer, ctx);

    Shape innerShape = inner.node->resolveShape(ctx);
    int rank = innerShape.getRank();

    int d0 = dim0 < 0 ? rank + dim0 : dim0;
    int d1 = dim1 < 0 ? rank + dim1 : dim1;

    // Only need to pack the two dimension indices to swap
    writer.write((uint32_t)d0);
    writer.write((uint32_t)d1);
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
    if (axisInput->tensorView)
        throw std::runtime_error("axis of concat must be static const");
    return (int)axisInput->scalarValue;
}

String ConcatNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return StringBuilder() << "Concat<" << elemTypeName << ", "
                           << leftProgram->getSlangTypeName(elemType) << ", "
                           << rightProgram->getSlangTypeName(elemType) << ">";
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

    int trueAxis = getAxis(ctx);

    if (trueAxis < 0)
        trueAxis += lShape.getRank();

    // 3. Pack Scalars (Axis, Split Point)
    // Axis
    writer.write((uint32_t)trueAxis);
    // Split Point (Size of Left at Axis)
    writer.write((uint32_t)lShape[trueAxis]);
}

// =========================================================================
// BUFFER SINK (The Terminal Node)
// =========================================================================

void BufferSinkNode::pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const
{
    // Verify that the logical shape matches the output buffer shape.
    if (evalCtx.logicalShape != evalCtx.outputBuffer.shape)
    {
        throw std::runtime_error("BufferSink: Logical shape does not match output buffer shape.");
    }

    // Ensure the output pointer is aligned for the GPU
    writer.align(8);
    writer.write(evalCtx.outputBuffer.getDeviceAddress());

    uint32_t rank = (uint32_t)evalCtx.logicalShape.getRank();
    writer.write(rank);

    // Calculate and write physical strides based on the buffer's physical shape.
    auto strides = computeDenseStrides(evalCtx.outputBuffer.shape);

    uint32_t strideArr[8] = {0};
    for (int i = 0; i < (int)rank; ++i)
    {
        strideArr[i] = (uint32_t)strides[i];
    }
    writer.writeBytes(strideArr, sizeof(strideArr));
}

// =========================================================================
// PERMUTE SINK (The Address Transformer)
// =========================================================================


PermuteSinkNode::PermuteSinkNode(SinkExpr child, const std::initializer_list<int>& dims)
    : child(child)
{
    for (auto d : dims)
        this->dims.add(d);
}

String PermuteSinkNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return "PermuteSink<" + elemTypeName + ", " + child.node->getSlangTypeName(elemType) + ">";
}

Shape PermuteSinkNode::resolvePhysicalShape(const Shape& logicalShape) const
{
    // 1. "Peel" or transform the current logical shape into the child's space.
    // E.g., incoming [B, H, S, D] with dims {0, 2, 1, 3} -> transformed [B, S, H, D]
    ShortList<int, 8> transformedDims;
    for (int i = 0; i < dims.getCount(); ++i)
    {
        transformedDims.add(logicalShape[dims[i]]);
    }
    Shape transformedShape(transformedDims.getArrayView().arrayView);

    // 2. Delegate to child to find the ultimate physical layout
    return child.node->resolvePhysicalShape(transformedShape);
}

// Alignment is the max of child and local uints (4)
size_t PermuteSinkNode::getParameterAlignment() const
{
    return std::max((size_t)4, child.node->getParameterAlignment());
}

void PermuteSinkNode::pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const
{
    writer.align(getParameterAlignment());

    // 1. Transform the logical shape for the child
    // e.g., if parent is [2, 3] and we swap {1, 0}, child sees [3, 2]
    List<int> childDims;
    for (int d : dims)
    {
        // dims is our permutation map
        childDims.add(evalCtx.logicalShape[d]);
    }
    Shape childLogicalShape(childDims.getArrayView());

    // 2. Update Context and pass down
    SinkExprEvalContext childCtx = evalCtx;
    childCtx.logicalShape = childLogicalShape;
    child.node->pack(writer, childCtx);

    // 3. Write the Permutation Map (pMap) for the shader
    // We use 0xFFFFFFFF as a sentinel for "End of Map"
    uint32_t pMapArr[8];
    memset(pMapArr, 0xFF, sizeof(pMapArr));

    for (int i = 0; i < (int)dims.getCount(); ++i)
    {
        pMapArr[i] = (uint32_t)dims[i];
    }
    writer.writeBytes(pMapArr, sizeof(pMapArr));
}

// =========================================================================
// PARTITION SINK
// =========================================================================

PartitionSinkNode::PartitionSinkNode(SinkExpr child, uint32_t dimIndex, uint32_t partitionCount)
    : child(child), dimIndex(dimIndex), partitionCount(partitionCount)
{
}

String PartitionSinkNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return "PartitionSink<" + elemTypeName + ", " + child.node->getSlangTypeName(elemType) + ">";
}

Shape PartitionSinkNode::getChildLogicalShape(const Shape& logicalShape) const
{
    // Transform the incoming logical shape into the "Partitioned" layout
    // Logical [M, W] -> Physical [N, M, S]
    int rank = logicalShape.getRank();
    ShortList<int, 8> partitionedDims;

    // Insert the "Partition ID" as the new outermost dimension (Dim 0)
    int partitionSize = logicalShape[dimIndex] / (int)partitionCount;
    if (logicalShape[dimIndex] % partitionCount != 0)
    {
        throw std::runtime_error("PartitionSink: Dimension size not divisible by partition count");
    }
    partitionedDims.add(partitionCount);

    for (int i = 0; i < rank; ++i)
    {
        if (i == (int)dimIndex)
        {
            // The split dimension is reduced to the size of a single partition
            partitionedDims.add(partitionSize);
        }
        else
        {
            partitionedDims.add(logicalShape[i]);
        }
    }

    Shape partitionedShape(partitionedDims.getArrayView().arrayView);
    return partitionedShape;
}

Shape PartitionSinkNode::resolvePhysicalShape(const Shape& logicalShape) const
{

    Shape partitionedShape = getChildLogicalShape(logicalShape);

    // Ask the child to resolve its physical shape based on this new layout
    // If the child is a Permute, it will swap these new 3D dims.
    // If the child is a BufferSink, it will accept this 3D shape as final.
    return child.node->resolvePhysicalShape(partitionedShape);
}

// Alignment is the max of child and local uints (4)
size_t PartitionSinkNode::getParameterAlignment() const
{
    return std::max((size_t)4, child.node->getParameterAlignment());
}

void PartitionSinkNode::pack(ParameterWriter& writer, const SinkExprEvalContext& evalCtx) const
{
    Shape childLogicalShape = getChildLogicalShape(evalCtx.logicalShape);

    // Update Context and pass down
    SinkExprEvalContext childCtx = evalCtx;
    childCtx.logicalShape = childLogicalShape;
    child.node->pack(writer, childCtx);

    // Write metadata for the shader
    writer.write((uint32_t)dimIndex);
    int partitionSize = evalCtx.logicalShape[dimIndex] / (int)partitionCount;
    writer.write((uint32_t)partitionSize);
}

// --- BinaryNode ---
BinaryNode::BinaryNode(Expr l, Expr r, BinaryOp op)
    : left(l), right(r), op(op)
{
}

String BinaryNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return getSlangBinaryOpName(op) + "<" + elemTypeName + ", " +
           left.node->getSlangTypeName(elemType) + ", " + right.node->getSlangTypeName(elemType) +
           ">";
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

String UnaryNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    return getSlangUnaryOpName(op) + "<" + elemTypeName + ", " +
           inner.node->getSlangTypeName(elemType) + ">";
}

Shape UnaryNode::resolveShape(const EvalContext& ctx) const
{
    // Shape passes through Unary ops unchanged
    return inner.node->resolveShape(ctx);
}

// --- ProgramNode ---

String ProgramNode::getSlangTypeName(ElementType elemType) const
{
    String elemTypeName = getSlangElementTypeName(elemType);
    StringBuilder typeBuilder;
    typeBuilder << "Program<" << elemTypeName << ", " << resultRegID;

    // Append Statements
    for (ExprNode* node : linearNodes)
    {
        auto id = nodeToRegID.tryGetValue(node);
        if (!id)
            throw std::runtime_error("ProgramNode: Node missing in nodeToRegID map");
        String defType = genDefType(node, nodeToRegID, elemType);
        typeBuilder << ", Eval<" << elemTypeName << ", " << *id << ", " << defType << ">";
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

size_t ProgramNode::getAlignment() const
{
    size_t alignment = 1;
    for (ExprNode* node : linearNodes)
    {
        alignment = Math::Max(alignment, node->getAlignment());
    }
    return alignment;
}

void ProgramNode::pack(ParameterWriter& writer, const EvalContext& ctx) const
{
    // First, go through all nodes and find max alignment, that
    // is the alignment of the entire Program struct, and we need
    // to align our packing location with it.
    auto alignment = getAlignment();
    writer.align(alignment);
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
Expr permute(Expr inner, const std::initializer_list<int>& dims)
{
    return Expr(new PermuteNode(inner, dims));
}
Expr gather(Expr table, Expr indices)
{
    return Expr(new GatherNode(table, indices));
}
Expr transpose(Expr inner, int dim0, int dim1)
{
    return Expr(new TransposeNode(inner, dim0, dim1));
}
Expr concat(Expr left, Expr right, Expr axis)
{
    return Expr(new ConcatNode(left, right, axis));
}

// Represent the output buffer that the kernel will write into.
SinkExpr bufferSink()
{
    return SinkExpr(new BufferSinkNode());
}

SinkExpr permute(SinkExpr child, const std::initializer_list<int>& dims)
{
    return SinkExpr(new PermuteSinkNode(child, dims));
}

SinkExpr partition(SinkExpr child, uint32_t dimIndex, uint32_t numParititons)
{
    return SinkExpr(new PartitionSinkNode(child, dimIndex, numParititons));
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

template<typename Func>
void visitAllExpr(HashSet<ExprNode*>& visited, ExprNode* node, Func f)
{
    if (visited.contains(node))
        return;
    visited.add(node);
    f(node);
    if (auto b = dynamic_cast<BinaryNode*>(node))
    {
        visitAllExpr(visited, b->left.node, f);
        visitAllExpr(visited, b->right.node, f);
    }
    else if (auto u = dynamic_cast<UnaryNode*>(node))
    {
        visitAllExpr(visited, u->inner.node, f);
    }
    else if (auto br = dynamic_cast<BroadcastNode*>(node))
    {
        visitAllExpr(visited, br->inner.node, f);
    }
    else if (auto p = dynamic_cast<PermuteNode*>(node))
    {
        visitAllExpr(visited, p->inner.node, f);
    }
    else if (auto t = dynamic_cast<TransposeNode*>(node))
    {
        visitAllExpr(visited, t->inner.node, f);
    }
    else if (auto c = dynamic_cast<ConcatNode*>(node))
    {
        visitAllExpr(visited, c->left.node, f);
        visitAllExpr(visited, c->right.node, f);
    }
    else if (auto g = dynamic_cast<GatherNode*>(node))
    {
        visitAllExpr(visited, g->table.node, f);
        visitAllExpr(visited, g->indices.node, f);
    }
    else if (auto leaf = dynamic_cast<LeafNode*>(node))
    {
    }
    else
    {
        throw std::runtime_error("Unknown ExprNode type in visitAllExpr");
    }
}

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
    else if (auto g = dynamic_cast<GatherNode*>(node))
    {
        // Compile sub-programs for both inputs
        g->tableProgram = new ProgramNode();
        *g->tableProgram = compileExprToProgram(g->table, ctx.globalRegCounter);

        g->indicesProgram = new ProgramNode();
        *g->indicesProgram = compileExprToProgram(g->indices, ctx.globalRegCounter);
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
    HashSet<ExprNode*> visited;
    List<BufferNode*> bufferNodes;
    visitAllExpr(
        visited,
        root.node,
        [&](ExprNode* node)
        {
            if (auto bufferNode = as<BufferNode>(node))
            {
                bufferNodes.add(bufferNode);
            }
        });
    bufferNodes.sort([](BufferNode* b1, BufferNode* b2)
                     { return b1->sequenceNumber < b2->sequenceNumber; });

    int resultReg = ssaCtx.regIDs[root.node];

    ProgramNode program;
    for (auto node : ssaCtx.topoOrder)
    {
        program.linearNodes.add(node);
    }
    program.resultRegID = ssaCtx.regIDs[root.node];
    program.nodeToRegID = _Move(ssaCtx.regIDs);
    program.bufferNodes = _Move(bufferNodes);
    return _Move(program);
}

String genDefType(
    ExprNode* node,
    const Dictionary<ExprNode*, int>& mapNodeToId,
    ElementType elemType)
{
    String elemTypeName = getSlangElementTypeName(elemType);
    if (auto b = dynamic_cast<BinaryNode*>(node))
    {
        return getSlangBinaryOpName(b->op) + "<" + elemTypeName + ", " +
               genExprType(b->left.node, mapNodeToId, elemType) + ", " +
               genExprType(b->right.node, mapNodeToId, elemType) + ">";
    }
    else if (auto u = dynamic_cast<UnaryNode*>(node))
    {
        return getSlangUnaryOpName(u->op) + "<" + elemTypeName + ", " +
               genExprType(u->inner.node, mapNodeToId, elemType) + ">";
    }
    return node->getSlangTypeName(elemType);
}

String genExprType(
    ExprNode* node,
    const Dictionary<ExprNode*, int>& mapNodeToId,
    ElementType elemType)
{
    String elemTypeName = getSlangElementTypeName(elemType);
    if (auto it = mapNodeToId.tryGetValue(node))
    {
        // Visited node -> Use Register
        StringBuilder sb;
        sb << "Reg<" << elemTypeName << ", " << *it << ">";
        return sb.produceString();
    }
    // CHANGE: Not visited (child of Broadcast) -> Generate full definition (Inline)
    return genDefType(node, mapNodeToId, elemType);
}

// =========================================================================
// Pipeline Implementation
// =========================================================================

ElementwiseKernel::ElementwiseKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr rootNode)
    : context(ctx), elementType(elementType), root(rootNode)
{
    int globalRegCounter = 0;
    program = compileExprToProgram(root, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);
    String finalProgramType = program.getSlangTypeName(elementType);

    String typeArgs[] = {elemTypeName, finalProgramType};
    pipeline = ctx->createComputePipeline("materialize", makeArrayView(typeArgs));
}

void ElementwiseKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("ElementwiseKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

TensorView ElementwiseKernel::allocateResultBuffer(
    ElementType elementType,
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
    {
        ctx.inputs.add(it.first.node, it.second);
    }
    Shape resultShape = root.node->resolveShape(ctx);
    size_t count = resultShape.getElementCount();
    return context->allocScratchTensor(elementType, resultShape, "elementwise");
}

void ElementwiseKernel::queueExecute(InferencingTask& task, EvalContext& ctx, TensorView output)
{
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto bufferNode : program.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input");
    }

    Shape resultShape = root.node->resolveShape(ctx);
    size_t count = resultShape.getElementCount();

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    if (output.shape != resultShape)
    {
        throw std::runtime_error("ElementwiseKernel: Output shape mismatch");
    }

    writer.write(output.getDeviceAddress());
    writer.write((uint32_t)count);

    // Write rank and shape for coordinate decomposition in the materialize kernel
    uint32_t rank = (uint32_t)resultShape.getRank();
    writer.write(rank);

    uint32_t shapeArr[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < (int)rank; ++i)
        shapeArr[i] = (uint32_t)resultShape[i];
    writer.writeBytes(shapeArr, sizeof(shapeArr));

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

void ElementwiseKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const Dictionary<Expr, InputInfo>& inputs)
{
    EvalContext ctx;
    for (auto it : inputs)
        ctx.inputs.add(it.first.node, it.second);
    queueExecute(task, ctx, output);
}

void ElementwiseKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    ArrayView<InputInfo> inputs)
{
    EvalContext ctx;
    if (inputs.getCount() != inputs.getCount())
    {
        throw std::runtime_error("ElementwiseKernel::queueExecute: Input count mismatch");
    }
    for (Index i = 0; i < inputs.getCount(); i++)
    {
        ctx.inputs.add(program.bufferNodes[i], inputs[i]);
    }
    queueExecute(task, ctx, output);
}

void ElementwiseKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const std::initializer_list<InputInfo>& inputs)
{
    EvalContext ctx;
    if (inputs.size() != program.bufferNodes.getCount())
    {
        throw std::runtime_error("ElementwiseKernel::queueExecute: Input count mismatch");
    }
    Index i = 0;
    for (auto& inputInfo : inputs)
    {
        ctx.inputs.add(program.bufferNodes[i], inputInfo);
        i++;
    }
    queueExecute(task, ctx, output);
}

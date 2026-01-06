#include "concat.h"

#include "elementwise.h"

// Helper to recursively build the Concat(Left, Right) tree
// Returns: The root Expr of the sub-tree
// Updates: 'bufferMap' maps the generated Buffer Exprs back to their runtime index
//          so we can bind InputInfo later.
Expr buildConcatTree(int startIndex, int count, Expr axis, Dictionary<int, Expr>& bufferMap)
{
    if (count == 1)
    {
        // Leaf: Create a Buffer placeholder
        auto b = buffer();
        bufferMap.add(startIndex, b);
        return b;
    }

    // Split inputs: Left gets 1, Right gets (N-1)
    // This creates a chain: Concat(A, Concat(B, Concat(C, D)))
    // This is efficient because the shader evaluation logic checks bounds
    // and terminates early (if in A, it reads A and stops).

    Expr left = buildConcatTree(startIndex, 1, axis, bufferMap);
    Expr right = buildConcatTree(startIndex + 1, count - 1, axis, bufferMap);

    return concat(left, right, axis);
}

ConcatKernel::ConcatKernel(InferencingContext* ctx, int operandCount)
    : context(ctx), operandCount(operandCount)
{
    axisExpr = uniformConstant();
    // 1. Build the generic expression tree for N inputs
    // This creates a balanced binary tree of Concat nodes.
    // The leaves are Buffer() expressions that will be bound at runtime.
    Expr rootExpr = buildConcatTree(0, operandCount, axisExpr, mapOperandToExprNode);
    // 2. Compile the kernel (this caches the pipeline internally)
    elementwiseKernel = new ElementwiseKernel(context, rootExpr);
}

TensorView ConcatKernel::allocateResultBuffer(
    ElementType elementType,
    ArrayView<Shape> inputShapes,
    int axis)
{
    if (inputShapes.getCount() != operandCount)
        throw std::runtime_error(
            "ConcatKernel: Mismatched kernel operand count configuration and actual operand count");
    if (inputShapes.getCount() == 0)
        throw std::runtime_error("ConcatKernel: No inputs provided");

    // 3. Bind Inputs
    Dictionary<Expr, InputInfo> bindings;

    bindings.add(axisExpr, axis);
    for (int i = 0; i < inputShapes.getCount(); i++)
    {
        // Retrieve the Expr node created for this index
        Expr e = mapOperandToExprNode[i];

        // Bind runtime data: Shape + Buffer Pointer
        bindings.add(e, TensorView(BufferView(), elementType, inputShapes[i]));
    }
    return elementwiseKernel->allocateResultBuffer(elementType, bindings);
}

void ConcatKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    ArrayView<TensorView> inputs,
    int axis)
{
    if (inputs.getCount() != operandCount)
        throw std::runtime_error(
            "ConcatKernel: Mismatched kernel operand count configuration and actual operand count");
    if (inputs.getCount() == 0)
        throw std::runtime_error("ConcatKernel: No inputs provided");

    // 3. Bind Inputs
    Dictionary<Expr, InputInfo> bindings;

    bindings.add(axisExpr, axis);
    for (int i = 0; i < inputs.getCount(); i++)
    {
        // Retrieve the Expr node created for this index
        Expr e = mapOperandToExprNode[i];

        // Bind runtime data: Shape + Buffer Pointer
        bindings.add(e, inputs[i]);
    }

    // 4. Execute
    // The kernel will resolve the total output shape automatically
    // by propagating shapes through the ConcatNodes.
    elementwiseKernel->queueExecute(task, output, bindings);
}
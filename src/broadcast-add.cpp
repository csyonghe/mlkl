#include "broadcast-add.h"

BroadcastAddKernel::BroadcastAddKernel(InferencingContext* context)
    : context(context)
{
    // 1. Build the generic expression tree
    // We want to calculate: A + Broadcast(B, to: A)

    inputAExpr = buffer();
    inputBExpr = buffer();

    // Explicitly say B should be broadcast to match A's shape
    auto b_broad = broadcast(inputBExpr, inputAExpr);
    auto resultExpr = inputAExpr + b_broad;

    // 2. Compile the kernel (this caches the pipeline internally)
    kernel = new ElementwiseKernel(context, resultExpr);
}

BufferView BroadcastAddKernel::allocResultBuffer(
    const Shape& shapeA,
    const Shape& shapeB,
    int batchSize)
{
    return context->allocScratchBuffer(
        shapeA.getElementCount() * batchSize * sizeof(float),
        "broadcastAdd");
}


void BroadcastAddKernel::queueExecute(
    InferencingTask& task,
    BufferView result,
    BufferView inputA,
    const Shape& shapeA,
    BufferView inputB,
    const Shape& shapeB,
    int batchSize)
{
    // 1. Construct Runtime Shapes
    // Prepend batch dimension to shapes
    List<int> dimsA;
    dimsA.add(batchSize);
    for (int s : shapeA.getDims())
        dimsA.add(s);

    List<int> dimsB;
    dimsB.add(batchSize);
    for (int s : shapeB.getDims())
        dimsB.add(s);

    Shape fullShapeA(dimsA.getArrayView());
    Shape fullShapeB(dimsB.getArrayView());

    // 2. Bind Inputs
    Dictionary<Expr, InputInfo> inputs;

    // Bind A: InputInfo(Shape, Buffer, Offset)
    inputs.add(inputAExpr, InputInfo(fullShapeA, inputA, 0));

    // Bind B
    inputs.add(inputBExpr, InputInfo(fullShapeB, inputB, 0));

    // 3. Execute
    // The kernel will resolve the output shape (which is A's shape) and dispatch.
    kernel->eval(task, result, inputs);
}
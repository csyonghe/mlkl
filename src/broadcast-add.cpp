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

ComPtr<rhi::IBuffer> BroadcastAddKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* inputA,
    ArrayView<int> shapeA,
    rhi::IBuffer* inputB,
    ArrayView<int> shapeB,
    int batchSize)
{
    // 1. Construct Runtime Shapes
    // Prepend batch dimension to shapes
    List<int> dimsA;
    dimsA.add(batchSize);
    for (int s : shapeA)
        dimsA.add(s);

    List<int> dimsB;
    dimsB.add(batchSize);
    for (int s : shapeB)
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
    return kernel->eval(task, inputs);
}
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

TensorView BroadcastAddKernel::allocateResultBuffer(
    ElementType elementType,
    const Shape& shapeA,
    const Shape& shapeB)
{
    return context->allocScratchTensor(elementType, shapeA, "broadcastAdd");
}


void BroadcastAddKernel::queueExecute(
    InferencingTask& task,
    TensorView result,
    TensorView inputA,
    TensorView inputB)
{
    Dictionary<Expr, InputInfo> inputs;

    // Bind A: InputInfo(Shape, Buffer, Offset)
    inputs.add(inputAExpr, InputInfo(inputA));

    // Bind B
    inputs.add(inputBExpr, InputInfo(inputB));

    // The kernel will resolve the output shape (which is A's shape) and dispatch.
    kernel->queueExecute(task, result, inputs);
}
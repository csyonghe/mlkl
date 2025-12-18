#include "cfg.h"

ClassifierFreeGuidanceKernel::ClassifierFreeGuidanceKernel(InferencingContext* context)
    : context(context)
{
    // 1. Create Graph Nodes
    uncond = buffer();
    cond = buffer();
    scale = uniformConstant(); // Runtime constant!

    // 2. Define Formula: Uncond + Scale * (Cond - Uncond)
    // Scale is scalar, others are vectors.
    // Automatic broadcast logic in BinaryNode will handle Scalar * Vector.

    auto diff = cond - uncond;
    auto result = uncond + diff * scale;

    // 3. Compile Kernel
    kernel = new ElementwiseKernel(context, result);
}

BufferView ClassifierFreeGuidanceKernel::allocResultBuffer(int width, int height, int channels)
{
    // For CFG, shapeA and shapeB should be the same.
    size_t elementCount = Shape(width, height, channels).getElementCount();
    return context->allocScratchBuffer(elementCount * sizeof(float), "cfg");
}

void ClassifierFreeGuidanceKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    BufferView batchedInput,
    int width,
    int height,
    int channels,
    float guidanceScale)
{
    auto shape = Shape(width, height, channels);

    Dictionary<Expr, InputInfo> inputs;
    // Batch 0: Uncond (Offset 0)
    inputs[uncond] = InputInfo(shape, batchedInput);

    // Batch 1: Cond (Offset count * sizeof(float))
    inputs[cond] = InputInfo(shape, batchedInput.tail(shape.getElementCount() * sizeof(float)));

    // Scale (Scalar Value)
    inputs[scale] = guidanceScale;

    kernel->eval(task, output, inputs);
}
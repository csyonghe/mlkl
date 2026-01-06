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

TensorView ClassifierFreeGuidanceKernel::allocateResultBuffer(
    ElementType elementType,
    int width,
    int height,
    int channels)
{
    // For CFG, shapeA and shapeB should be the same.
    return context->allocScratchTensor(elementType, Shape(width, height, channels), "cfg");
}

void ClassifierFreeGuidanceKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView batchedInput,
    float guidanceScale)
{
    if (batchedInput.shape.getRank() != output.shape.getRank() + 1)
    {
        throw std::runtime_error(
            "Batched input rank must be output rank + 1 (for batch dimension).");
    }

    if (batchedInput.shape.dims[0] != 2)
    {
        throw std::runtime_error("Batched input first dimension (batch) must be 2 (uncond, cond).");
    }

    Dictionary<Expr, InputInfo> inputs;
    auto uncondSlice = batchedInput.slice(0, 1).reshape(batchedInput.shape.tail(1));
    auto condSlice = batchedInput.slice(1, 1).reshape(uncondSlice.shape);

    // Batch 0: Uncond (Offset 0)
    inputs[uncond] = uncondSlice;

    // Batch 1: Cond (Offset count * sizeof(float))
    inputs[cond] = condSlice;

    // Scale (Scalar Value)
    inputs[scale] = guidanceScale;

    kernel->queueExecute(task, output, inputs);
}
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

ComPtr<rhi::IBuffer> ClassifierFreeGuidanceKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* batchedInput,
    int width,
    int height,
    int channels,
    float guidanceScale)
{
    int count = (int)width * height * channels;

    Dictionary<Expr, InputInfo> inputs;

    // Batch 0: Uncond (Offset 0)
    InputInfo uncondInfo;
    uncondInfo.buffer = batchedInput;
    uncondInfo.offset = 0;
    inputs.add(uncond, uncondInfo);

    // Batch 1: Cond (Offset count * sizeof(float))
    InputInfo condInfo;
    condInfo.buffer = batchedInput;
    condInfo.offset = count * sizeof(float);
    inputs.add(cond, condInfo);

    // Scale (Scalar Value)
    InputInfo scaleInfo;
    scaleInfo.scalarValue = guidanceScale;
    inputs.add(scale, scaleInfo);

    return kernel->eval(task, inputs);
}
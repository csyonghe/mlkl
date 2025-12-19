#include "softmax.h"

SoftmaxKernel::SoftmaxKernel(InferencingContext* ctx)
    : context(ctx)
{
    pipeline = context->createComputePipeline("softmax", {});
}

BufferView SoftmaxKernel::allocateResultBuffer(int rows, int cols)
{
    size_t size = (size_t)rows * cols * sizeof(float);
    return context->allocScratchBuffer(size, "Softmax_Result");
}

void SoftmaxKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    BufferView input,
    int rows,
    int cols)
{
    struct
    {
        rhi::DeviceAddress input, output;
        uint32_t stride;
        uint32_t rowCount; // Added
    } params;

    params.input = input.getDeviceAddress();
    params.output = output.getDeviceAddress();
    params.stride = cols;
    params.rowCount = rows; // Pass total rows

    // Calculate dispatch size (rounding up)
    uint32_t groupX = (rows + 255) / 256;
    task.dispatchKernel(pipeline, groupX, 1, 1, params);
}
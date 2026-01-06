#include "softmax.h"

SoftmaxKernel::SoftmaxKernel(InferencingContext* ctx)
    : context(ctx)
{
    pipeline = context->createComputePipeline("softmax", {});
}

TensorView SoftmaxKernel::allocateResultBuffer(ElementType elementType, int rows, int cols)
{
    return context->allocScratchTensor(elementType, Shape(rows, cols), "Softmax_Result");
}

void SoftmaxKernel::queueExecute(InferencingTask& task, TensorView output, TensorView input)
{
    struct
    {
        rhi::DeviceAddress input, output;
        uint32_t stride;
        uint32_t rowCount; // Added
    } params;

    if (input.shape.getRank() != 2)
    {
        throw std::runtime_error("SoftmaxKernel: Input must be a rank 2 tensor.");
    }
    if (input.shape != output.shape)
    {
        throw std::runtime_error("SoftmaxKernel: Input and output shapes must match.");
    }
    int rows = (int)input.shape.dims[0];
    int cols = (int)input.shape.dims[1];

    params.input = input.getDeviceAddress();
    params.output = output.getDeviceAddress();
    params.stride = cols;
    params.rowCount = rows; // Pass total rows

    // Calculate dispatch size (rounding up)
    uint32_t groupX = (rows + 255) / 256;
    task.dispatchKernel(pipeline, groupX, 1, 1, params);
}
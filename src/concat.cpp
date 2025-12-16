#include "concat.h"

ConcatKernel::ConcatKernel(InferencingContext* context)
    : context(context)
{
    pipeline = context->createComputePipeline("concat", ArrayView<String>());
}

ComPtr<rhi::IBuffer> ConcatKernel::queueExecute(
    InferencingTask& task,
    rhi::IBuffer* inputA,
    ArrayView<int> shapeA,
    rhi::IBuffer* inputB,
    ArrayView<int> shapeB,
    int axis,
    int batchSize)
{
    struct ConcatParams
    {
        rhi::DeviceAddress inputA;
        rhi::DeviceAddress inputB;
        rhi::DeviceAddress output;
        int shapeA[8];
        int shapeB[8];
        int rank;
        int axis;
        int innerStride;
        int outerStride;
    } params;

    // 1. Prepare Shapes with Batch Dimension (Dim 0)
    // We shift the original shapes to start at index 1, and place batchSize at index 0.

    params.shapeA[0] = batchSize;
    params.shapeB[0] = batchSize;

    for (int i = 0; i < 7; i++)
    {
        params.shapeA[i + 1] = (i < shapeA.getCount()) ? shapeA[i] : 1;
        params.shapeB[i + 1] = (i < shapeB.getCount()) ? shapeB[i] : 1;
    }

    // 2. Adjust Rank and Axis
    // Rank increases by 1 due to batch dimension
    params.rank = (int)shapeA.getCount() + 1;

    // Axis shifts by 1 (e.g. Channel 0 becomes Channel 1)
    params.axis = axis + 1;

    // 3. Calculate Output Size
    size_t elementCountA = batchSize;
    size_t elementCountB = batchSize;

    for (int i = 0; i < shapeA.getCount(); i++)
    {
        elementCountA *= shapeA[i];
        elementCountB *= shapeB[i];
    }

    size_t outputSize = (elementCountA + elementCountB);
    auto outputBuffer = task.allocateBuffer("concat_out", outputSize * sizeof(float));

    params.inputA = inputA->getDeviceAddress();
    params.inputB = inputB->getDeviceAddress();
    params.output = outputBuffer->getDeviceAddress();

    // 4. Calculate Strides using the Augmented Shapes

    // Inner Stride: Product of dims AFTER axis
    params.innerStride = 1;
    for (int i = params.rank - 1; i > params.axis; i--)
    {
        params.innerStride *= params.shapeA[i];
    }

    // Outer Stride: Product of dims BEFORE axis
    // This now includes the Batch Dimension (since j=0 is batch)
    params.outerStride = 1;
    for (int j = 0; j < params.axis; j++)
    {
        params.outerStride *= params.shapeA[j];
    }

    uint32_t threadGroupCount = (uint32_t)((outputSize + 255) / 256);
    task.dispatchKernel(pipeline, threadGroupCount, 1, 1, params);

    return ComPtr<rhi::IBuffer>(outputBuffer);
}
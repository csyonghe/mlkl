#include "kernels.h"

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
    int axis)
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
    for (int i = 0; i < 8; i++)
    {
        params.shapeA[i] = (i < shapeA.getCount()) ? shapeA[i] : 1;
        params.shapeB[i] = (i < shapeB.getCount()) ? shapeB[i] : 1;
    }
    params.rank = (int)shapeA.getCount();
    size_t elementCountA = 1, elementCountB = 1;
    for (int i = 0; i < shapeA.getCount(); i++)
    {
        elementCountA *= shapeA[i];
        elementCountB *= shapeB[i];
    }
    size_t outputSize = (elementCountA + elementCountB);
    auto outputBuffer = context->createBuffer(nullptr, outputSize * sizeof(float));
    params.inputA = inputA->getDeviceAddress();
    params.inputB = inputB->getDeviceAddress();
    params.output = outputBuffer->getDeviceAddress();
    params.rank = (int)shapeA.getCount();
    params.axis = axis;
    params.innerStride = 1;
    for (int i = params.rank - 1; i > params.axis; i--)
    {
        params.innerStride *= params.shapeA[i];
    }
    params.outerStride = 1;
    for (int j = 0; j < params.axis; j++)
    {
        params.outerStride *= params.shapeA[j];
    }
    uint32_t threadGroupCount = (uint32_t)((outputSize + 255) / 256);
    task.dispatchKernel(
        pipeline,
        threadGroupCount,
        1,
        1,
        params);
    return outputBuffer;
}
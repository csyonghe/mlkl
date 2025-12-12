
#include "broadcast-add.h"

BroadcastAddKernel::BroadcastAddKernel(InferencingContext* context)
    : context(context)
{
    pipeline = context->createComputePipeline("broadcastAdd", ArrayView<String>());
}

ComPtr<rhi::IBuffer> BroadcastAddKernel::queueExecute(InferencingTask& task, rhi::IBuffer* inputA, ArrayView<int> shapeA, rhi::IBuffer* inputB, ArrayView<int> shapeB)
{
    struct BroadcastAddParamsData
    {
        rhi::DeviceAddress inputLhs;
        rhi::DeviceAddress inputRhs;
        rhi::DeviceAddress output;
        int lhsRank;
        int rhsRank;
        int lhsShape[8];
        int rhsShape[8];
    } paramsData = {};
    int outputSize = 1;
    for (Index i = 0; i < shapeA.getCount(); i++)
    {
        outputSize *= shapeA[i];
    }
    auto outputBuffer = task.allocateBuffer("broadcast_add_output", outputSize * sizeof(float));
    paramsData.inputLhs = inputA->getDeviceAddress();
    paramsData.inputRhs = inputB->getDeviceAddress();
    paramsData.output = outputBuffer->getDeviceAddress();
    paramsData.lhsRank = (int)shapeA.getCount();
    paramsData.rhsRank = (int)shapeB.getCount();
    for (int i = 0; i < 8; i++)
    {
        paramsData.lhsShape[i] = (i < shapeA.getCount()) ? shapeA[i] : 1;
        paramsData.rhsShape[i] = (i < shapeB.getCount()) ? shapeB[i] : 1;
    }
    uint32_t threadGroupCountX = (uint32_t)((outputSize + 256 - 1) / 256);
    task.dispatchKernel(
        pipeline,
        threadGroupCountX,
        1,
        1,
        paramsData);
    return ComPtr<rhi::IBuffer>(outputBuffer);
}
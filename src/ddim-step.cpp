#include "ddim-step.h"

DDIMStepKernel::DDIMStepKernel(RefPtr<InferencingContext> inferencingCtx)
    : inferencingCtx(inferencingCtx)
{
    slang::TypeLayoutReflection* paramTypeLayout = nullptr;
    pipeline = inferencingCtx->createComputePipeline("ddimStep", {});
}

void DDIMStepKernel::forward(
    InferencingTask& task,
    rhi::IBuffer* currentImage,
    rhi::IBuffer* predictedNoise,
    rhi::IBuffer* outputImage,
    float alphaBar_t,
    float alphaBar_prev,
    int width,
    int height,
    int channels,
    int batchSize)
{
    struct DDIMParams
    {
        rhi::DeviceAddress currentImage;
        rhi::DeviceAddress predictedNoise;
        rhi::DeviceAddress outputImage;
        float alphaBar_t;
        float alphaBar_prev;
        uint32_t totalElements;
    } params;

    params.currentImage = currentImage->getDeviceAddress();
    params.predictedNoise = predictedNoise->getDeviceAddress();
    params.outputImage = outputImage->getDeviceAddress();
    params.alphaBar_t = alphaBar_t;
    params.alphaBar_prev = alphaBar_prev;

    // Calculate total elements properly
    params.totalElements = width * height * channels * batchSize;

    task.dispatchKernel(pipeline, (params.totalElements + 255) / 256, 1, 1, params);
}

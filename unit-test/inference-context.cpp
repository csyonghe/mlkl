#include "inference-context.h"

#include "slang-rhi/shader-cursor.h"

InferencingContext::InferencingContext(rhi::IDevice* inDevice)
{
    this->device = inDevice;
    this->slangSession = device->getSlangSession();

    ComPtr<ISlangBlob> diagnosticBlob;
    this->slangModule = slangSession->loadModule("diffusion", diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
}

ComPtr<rhi::IComputePipeline> InferencingContext::createComputePipeline(const char* entryPointName, Slang::ConstArrayView<String> specArgs)
{
    ComPtr<slang::IEntryPoint> entryPoint;
    slangModule->findEntryPointByName(entryPointName, entryPoint.writeRef());
    if (!entryPoint)
    {
        reportError("Failed to find entry point '%s'\n", entryPointName);
        return nullptr;
    }
    ComPtr<slang::IComponentType> specializedComponentType;
    List<slang::SpecializationArg> args;
    for (const auto& arg : specArgs)
    {
        args.add(slang::SpecializationArg::fromExpr(arg.getBuffer()));
    }
    ComPtr<ISlangBlob> diagnosticBlob;
    entryPoint->specialize(
        args.getBuffer(),
        (SlangInt)args.getCount(),
        specializedComponentType.writeRef(),
        diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
    if (!specializedComponentType)
    {
        reportError("Failed to specialize entry point '%s'\n", entryPointName);
        return nullptr;
    }
    ComPtr<slang::IComponentType> linkedComponentType;
    specializedComponentType->link(linkedComponentType.writeRef(), diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
    if (!linkedComponentType)
    {
        reportError("Failed to link specialized entry point '%s'\n", entryPointName);
        return nullptr;
    }
    ComPtr<rhi::IShaderProgram> shaderProgram;
    shaderProgram = device->createShaderProgram(linkedComponentType, diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
    if (!shaderProgram)
    {
        reportError("Failed to create shader program for entry point '%s'\n", entryPointName);
        return nullptr;
    }
    rhi::ComputePipelineDesc desc = {};
    desc.program = shaderProgram;
    desc.label = entryPointName;
    return device->createComputePipeline(desc);
}

void InferencingContext::diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
{
    if (diagnosticsBlob != nullptr)
    {
        reportError("%s", (const char*)diagnosticsBlob->getBufferPointer());
    }
}

InferencingTask InferencingContext::createTask()
{
    InferencingTask task = InferencingTask(
        device->getQueue(rhi::QueueType::Graphics)->createCommandEncoder(),
        this);
    return task;
}

ComPtr<rhi::IBuffer> InferencingContext::createBuffer(const void* data, size_t size)
{
    rhi::BufferDesc bufferDesc = {};
    bufferDesc.size = size;
    bufferDesc.defaultState = rhi::ResourceState::UnorderedAccess;
    bufferDesc.usage = rhi::BufferUsage::CopySource | rhi::BufferUsage::CopyDestination |
        rhi::BufferUsage::UnorderedAccess;
    bufferDesc.memoryType = rhi::MemoryType::DeviceLocal;
    auto buffer = device->createBuffer(bufferDesc, data);
    if (!buffer)
    {
        reportError("Failed to create buffer of size %zu\n", size);
        return nullptr;
    }
    return buffer;
}

ComPtr<rhi::IShaderObject> InferencingTask::createKernelParamObject(slang::TypeLayoutReflection* typeLayout)
{
    ComPtr<rhi::IShaderObject> obj;
    context->getDevice()->createShaderObjectFromTypeLayout(typeLayout, obj.writeRef());
    if (!obj)
    {
        InferencingContext::reportError("Failed to create kernel parameter object\n");
        return nullptr;
    }
    kernelParamObjects.add(obj);
    return obj;
}

void InferencingTask::dispatchKernel(
    rhi::IComputePipeline* pipeline,
    uint32_t threadGroupCountX,
    uint32_t threadGroupCountY,
    uint32_t threadGroupCountZ,
    const void* paramData,
    size_t paramDataSize)
{
    auto computeEncoder = encoder->beginComputePass();
    auto rootShaderObject = computeEncoder->bindPipeline(pipeline);
    rhi::ShaderCursor cursor(rootShaderObject->getEntryPoint(0));
    auto pParams = cursor["params"];
    auto typeLayout = pParams.getTypeLayout();
    if (!typeLayout)
    {
        InferencingContext::reportError("Failed to get kernel parameter type layout, kernel must have only one uniform parameter named 'params'\n");
        return;
    }
    auto paramsTypeLayout = typeLayout->getElementTypeLayout();
    auto obj = createKernelParamObject(paramsTypeLayout);
    if (paramsTypeLayout->getStride() != paramDataSize)
    {
        InferencingContext::reportError("Kernel parameter size mismatch: expected %zu, got %zu\n", paramsTypeLayout->getSize(), paramDataSize);
        return;
    }
    obj->setData(rhi::ShaderOffset(), paramData, paramDataSize);
    pParams.setObject(obj);
    computeEncoder->dispatchCompute(threadGroupCountX, threadGroupCountY, threadGroupCountZ);
    computeEncoder->end();
}

rhi::IBuffer* InferencingTask::allocateBuffer(size_t size, void* initData)
{
    ComPtr<rhi::IBuffer> buffer = context->createBuffer(initData, size);
    buffers.add(buffer);
    return buffer;
}

void InferencingTask::execute()
{
    auto commandBuffer = encoder->finish();
    auto result = context->getDevice()->getQueue(rhi::QueueType::Graphics)->submit(commandBuffer);
    if (SLANG_FAILED(result))
    {
        InferencingContext::reportError("Failed to submit command buffer\n");
    }
}

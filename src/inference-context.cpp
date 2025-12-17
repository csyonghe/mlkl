#include "inference-context.h"

#include "example-base.h"
#include "inference-context.h"
#include "slang-rhi/shader-cursor.h"

InferencingContext::InferencingContext(rhi::IDevice* inDevice, size_t defaultPageSize)
{
    this->device = inDevice;
    this->slangSession = device->getSlangSession();
    this->allocator = new StackAllocator(this, defaultPageSize);

    ComPtr<ISlangBlob> diagnosticBlob;
    this->slangModule = slangSession->loadModule("mlkl", diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);
}

ComPtr<rhi::IComputePipeline> InferencingContext::createComputePipeline(
    const char* entryPointName,
    Slang::ConstArrayView<String> specArgs)
{
    DigestBuilder<MD5> digestBuilder;
    digestBuilder.append(UnownedStringSlice(entryPointName));
    for (auto arg : specArgs)
    {
        digestBuilder.append(arg);
    }
    auto digest = digestBuilder.finalize();
    ComPtr<rhi::IComputePipeline> pipeline;
    if (pipelineCache.tryGetValue(digest, pipeline))
    {
        return pipeline;
    }
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
    pipeline = device->createComputePipeline(desc);
    pipelineCache.add(digest, pipeline);
    return pipeline;
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
#if !INTERMEDIATE_MODE
        device->getQueue(rhi::QueueType::Graphics)->createCommandEncoder(),
#else
        nullptr,
#endif
        this);
    return task;
}

BufferView InferencingContext::allocScratchBuffer(size_t size, const char* label)
{
    auto buffer = allocator->allocate(size);
    if (!buffer)
    {
        reportError("Failed to create buffer of size %zu\n", size);
        return {};
    }
    return buffer;
}

ComPtr<rhi::IBuffer> InferencingContext::createPersistentBuffer(
    const void* data,
    size_t size,
    const char* label)
{
    rhi::BufferDesc bufferDesc = {};
    bufferDesc.size = size;
    bufferDesc.label = label;
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

ComPtr<rhi::IShaderObject> InferencingTask::createKernelParamObject(
    slang::TypeLayoutReflection* typeLayout)
{
    ComPtr<rhi::IShaderObject> obj;
    context->getDevice()->createShaderObjectFromTypeLayout(typeLayout, obj.writeRef());
    if (!obj)
    {
        reportError("Failed to create kernel parameter object\n");
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
#if INTERMEDIATE_MODE
    auto queue = context->getDevice()->getQueue(rhi::QueueType::Graphics);
    encoder = queue->createCommandEncoder();
#endif
    auto computeEncoder = encoder->beginComputePass();
    auto rootShaderObject = computeEncoder->bindPipeline(pipeline);
    rhi::ShaderCursor cursor(rootShaderObject->getEntryPoint(0));
    auto pParams = cursor["params"];
    auto typeLayout = pParams.getTypeLayout();
    if (!typeLayout)
    {
        reportError("Failed to get kernel parameter type layout, kernel must have only one uniform "
                    "parameter named 'params'\n");
        return;
    }
    auto paramsTypeLayout = typeLayout->getElementTypeLayout();
    auto obj = createKernelParamObject(paramsTypeLayout);
    if (paramsTypeLayout->getStride() != paramDataSize)
    {
        reportError(
            "Kernel parameter size mismatch: expected %zu, got %zu\n",
            paramsTypeLayout->getStride(),
            paramDataSize);
        return;
    }
    obj->setData(rhi::ShaderOffset(), paramData, paramDataSize);
    pParams.setObject(obj);
    computeEncoder->dispatchCompute(threadGroupCountX, threadGroupCountY, threadGroupCountZ);
    computeEncoder->end();
    encoder->globalBarrier();
#if INTERMEDIATE_MODE
    if (SLANG_FAILED(queue->submit(encoder->finish())))
    {
        InferencingContext::reportError("Failed to submit compute command buffer\n");
    }
    if (SLANG_FAILED(queue->waitOnHost()))
    {
        InferencingContext::reportError("Failed to wait on compute queue\n");
    }
    encoder = nullptr;
#endif
}

void InferencingTask::execute()
{
#if !INTERMEDIATE_MODE
    auto commandBuffer = encoder->finish();
    auto queue = context->getDevice()->getQueue(rhi::QueueType::Graphics);
    auto result = queue->submit(commandBuffer);
    if (SLANG_FAILED(result))
    {
        reportError("Failed to submit command buffer\n");
    }
    queue->waitOnHost();
#endif
}

#include "inference-context.h"

#include "example-base.h"
#include "inference-context.h"
#include "slang-rhi/shader-cursor.h"

InferencingContext::InferencingContext(size_t defaultPageSize)
{
    // Create persistent shader/pipeline cache
    shaderCache = new FileShaderCache();

    rhi::DeviceDesc deviceDesc;
    deviceDesc.slang.targetProfile = "spirv_1_6";
    List<slang::CompilerOptionEntry> compilerOptionsEntries;
    const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
    for (auto cap : capabilities)
    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::Capability;
        slang::CompilerOptionValue value;
        value.kind = slang::CompilerOptionValueKind::String;
        value.stringValue0 = cap;
        entry.value = value;
        compilerOptionsEntries.add(entry);
    }
    deviceDesc.slang.compilerOptionEntries = compilerOptionsEntries.getBuffer();
    deviceDesc.slang.compilerOptionEntryCount = (uint32_t)compilerOptionsEntries.getCount();
    deviceDesc.deviceType = rhi::DeviceType::Vulkan;

    // Enable persistent caching for both shaders and pipelines
    deviceDesc.persistentShaderCache = shaderCache;
    deviceDesc.persistentPipelineCache = shaderCache;

    // rhi::getRHI()->enableDebugLayers();
    device = rhi::getRHI()->createDevice(deviceDesc);
    if (!device)
        throw std::runtime_error("cannot create vulkan device.");
    initWithDevice(defaultPageSize);
}

InferencingContext::InferencingContext(rhi::IDevice* inDevice, size_t defaultPageSize)
{
    this->device = inDevice;
    initWithDevice(defaultPageSize);
}

void InferencingContext::initWithDevice(size_t defaultPageSize)
{
    this->slangSession = device->getSlangSession();
    this->allocator = new StackAllocator(this, defaultPageSize);

    ComPtr<ISlangBlob> diagnosticBlob;
    this->slangModule = slangSession->loadModule("mlkl", diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);

    // Get timestamp frequency for performance measurements
    timestampFrequency = device->getInfo().timestampFrequency;
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
#if !IMMEDIATE_MODE
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

RefPtr<Tensor> InferencingContext::createTensor(
    ElementType elementType,
    const Shape& shape,
    size_t initialDataSize,
    const void* initialData,
    const char* label)
{
    RefPtr<Tensor> tensor = new Tensor();
    tensor->elementType = elementType;
    tensor->shape = shape;
    size_t bufferSize = tensor->getBufferSize();
    if (initialData != nullptr && bufferSize > initialDataSize)
    {
        reportError(
            "Initial data size %zu is smaller than tensor buffer size %zu\n",
            initialDataSize,
            bufferSize);
        return nullptr;
    }
    tensor->buffer = createPersistentBuffer(initialData, bufferSize, label);
    return tensor;
}

TensorView InferencingContext::allocScratchTensor(
    ElementType elementType,
    const Shape& shape,
    const char* label)
{
    TensorView tensorView = {};
    tensorView.elementType = elementType;
    tensorView.shape = shape;
    size_t bufferSize = tensorView.getBufferSize();
    tensorView.bufferView = allocScratchBuffer(bufferSize, label);
    return tensorView;
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
#if IMMEDIATE_MODE
    auto queue = context->getDevice()->getQueue(rhi::QueueType::Graphics);
    encoder = queue->createCommandEncoder();
#endif

    // Get kernel name for profiling
    const char* kernelName = pipeline->getDesc().label;
    if (!kernelName)
        kernelName = "unknown";

    // Write start timestamp if profiling is enabled
    context->recordKernelTimestamp(encoder, kernelName, true);

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

    // Write end timestamp if profiling is enabled
    context->recordKernelTimestamp(encoder, kernelName, false);

#if IMMEDIATE_MODE
    if (SLANG_FAILED(queue->submit(encoder->finish())))
    {
        reportError("Failed to submit compute command buffer\n");
    }
    if (SLANG_FAILED(queue->waitOnHost()))
    {
        reportError("Failed to wait on compute queue\n");
    }
    encoder = nullptr;
#endif
}

void InferencingTask::execute()
{
#if !IMMEDIATE_MODE
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

// ============================================================================
// Performance Measurement Implementation
// ============================================================================

void InferencingContext::setCollectPerfMeasurements(bool enable)
{
    collectPerfMeasurements = enable;

    if (enable && !queryPool)
    {
        // Create query pool on first enable
        rhi::QueryPoolDesc desc = {};
        desc.type = rhi::QueryType::Timestamp;
        desc.count = kMaxTimestampQueries;
        desc.label = "PerfTimestamps";
        if (SLANG_FAILED(device->createQueryPool(desc, queryPool.writeRef())))
        {
            reportError("Failed to create timestamp query pool\n");
            collectPerfMeasurements = false;
        }
    }
}

void InferencingContext::resetPerfMeasurements()
{
    timestampRecords.clear();
    nextQueryIndex = 0;
    if (queryPool)
    {
        queryPool->reset();
    }
}

uint32_t InferencingContext::allocateTimestampQuery()
{
    if (nextQueryIndex >= kMaxTimestampQueries)
    {
        reportError("Timestamp query pool exhausted (max %u queries)\n", kMaxTimestampQueries);
        return UINT32_MAX;
    }
    return nextQueryIndex++;
}

void InferencingContext::recordKernelTimestamp(
    rhi::ICommandEncoder* encoder,
    const char* kernelName,
    bool isStart)
{
    if (!collectPerfMeasurements || !queryPool)
        return;

    uint32_t queryIndex = allocateTimestampQuery();
    if (queryIndex == UINT32_MAX)
        return;

    encoder->writeTimestamp(queryPool, queryIndex);

    if (isStart)
    {
        // Start of kernel - create new record
        TimestampRecord record;
        record.startIndex = queryIndex;
        record.endIndex = UINT32_MAX; // Will be filled on end
        record.kernelName = kernelName;
        timestampRecords.add(record);
    }
    else
    {
        // End of kernel - update last record
        if (timestampRecords.getCount() > 0)
        {
            timestampRecords.getLast().endIndex = queryIndex;
        }
    }
}

List<KernelPerfEntry> InferencingContext::getPerfMeasurements()
{
    Dictionary<String, KernelPerfEntry> aggregated;

    if (!queryPool || timestampRecords.getCount() == 0 || timestampFrequency == 0)
    {
        return List<KernelPerfEntry>();
    }

    // Read all timestamps
    List<uint64_t> timestamps;
    timestamps.setCount(nextQueryIndex);
    if (SLANG_FAILED(queryPool->getResult(0, nextQueryIndex, timestamps.getBuffer())))
    {
        reportError("Failed to read timestamp query results\n");
        return List<KernelPerfEntry>();
    }

    // Process each record
    for (const auto& record : timestampRecords)
    {
        if (record.endIndex == UINT32_MAX || record.endIndex >= nextQueryIndex)
            continue;

        uint64_t startTime = timestamps[record.startIndex];
        uint64_t endTime = timestamps[record.endIndex];
        double durationMs = (double)(endTime - startTime) / timestampFrequency * 1000.0;

        KernelPerfEntry* entry = aggregated.tryGetValue(record.kernelName);
        if (entry)
        {
            entry->totalTimeMs += durationMs;
            entry->callCount++;
        }
        else
        {
            KernelPerfEntry newEntry;
            newEntry.name = record.kernelName;
            newEntry.totalTimeMs = durationMs;
            newEntry.callCount = 1;
            aggregated[record.kernelName] = newEntry;
        }
    }

    // Convert to list and sort by total time (descending)
    List<KernelPerfEntry> result;
    for (const auto& kv : aggregated)
    {
        result.add(kv.second);
    }

    // Sort by total time descending
    result.sort([](const KernelPerfEntry& a, const KernelPerfEntry& b)
                { return a.totalTimeMs > b.totalTimeMs; });

    return result;
}

void InferencingContext::printPerfMeasurements()
{
    auto entries = getPerfMeasurements();

    if (entries.getCount() == 0)
    {
        printf("No performance measurements collected.\n");
        return;
    }

    // Calculate totals
    double totalTimeMs = 0.0;
    uint32_t totalCalls = 0;
    for (const auto& e : entries)
    {
        totalTimeMs += e.totalTimeMs;
        totalCalls += e.callCount;
    }

    printf("\n=== GPU Kernel Performance ===\n");
    printf("%-50s %10s %8s %10s %6s\n", "Kernel", "Total(ms)", "Calls", "Avg(ms)", "%");
    printf("--------------------------------------------------------------------------------\n");

    for (const auto& e : entries)
    {
        double pct = totalTimeMs > 0 ? (e.totalTimeMs / totalTimeMs * 100.0) : 0.0;
        printf(
            "%-50s %10.2f %8u %10.3f %5.1f%%\n",
            e.name.getBuffer(),
            e.totalTimeMs,
            e.callCount,
            e.avgTimeMs(),
            pct);
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("%-50s %10.2f %8u\n", "TOTAL", totalTimeMs, totalCalls);
    printf("\n");
}

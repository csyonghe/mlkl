#pragma once

#include "core/slang-basic.h"
#include "core/slang-crypto.h"
#include "external/slang-rhi/include/slang-rhi.h"
#include "half.h"
#include "shader-cache.h"
#include "slang-com-ptr.h"
#include "slang.h"
#include "stack-allocator.h"
#include "tensor.h"

using namespace Slang;

// Define this to 1 to enable intermediate mode that runs kernels synchronously
// at each queueExecute for easier debugging.
#define IMMEDIATE_MODE 0

class InferencingContext;

class InferencingTask
{
public:
    ComPtr<rhi::ICommandEncoder> encoder;
    InferencingContext* context;
    List<ComPtr<rhi::IShaderObject>> kernelParamObjects;
    List<ComPtr<rhi::IBuffer>> buffers;
    ComPtr<rhi::IShaderObject> createKernelParamObject(slang::TypeLayoutReflection* typeLayout);

public:
    InferencingTask(ComPtr<rhi::ICommandEncoder>&& encoder, InferencingContext* context)
        : encoder(_Move(encoder)), context(context)
    {
    }

    void dispatchKernel(
        rhi::IComputePipeline* pipeline,
        uint32_t threadGroupCountX,
        uint32_t threadGroupCountY,
        uint32_t threadGroupCountZ,
        const void* paramData,
        size_t paramDataSize);

    template<typename TParams>
    void dispatchKernel(
        rhi::IComputePipeline* pipeline,
        uint32_t threadGroupCountX,
        uint32_t threadGroupCountY,
        uint32_t threadGroupCountZ,
        const TParams& paramData)
    {
        dispatchKernel(
            pipeline,
            threadGroupCountX,
            threadGroupCountY,
            threadGroupCountZ,
            &paramData,
            sizeof(TParams));
    }

    void dispatchKernel(
        rhi::IComputePipeline* pipeline,
        uint32_t threadGroupCountX,
        uint32_t threadGroupCountY,
        uint32_t threadGroupCountZ,
        const List<uint8_t>& paramData)
    {
        dispatchKernel(
            pipeline,
            threadGroupCountX,
            threadGroupCountY,
            threadGroupCountZ,
            paramData.getBuffer(),
            (size_t)paramData.getCount());
    }
    void execute();
};

class InferencingContext : public RefObject
{
private:
    ComPtr<rhi::IDevice> device;
    ComPtr<slang::ISession> slangSession;
    ComPtr<slang::IModule> slangModule;
    Dictionary<MD5::Digest, ComPtr<rhi::IComputePipeline>> pipelineCache;
    RefPtr<StackAllocator> allocator;
    RefPtr<FileShaderCache> shaderCache;
    void initWithDevice(size_t defaultPageSize);

public:
    InferencingContext(size_t defaultPageSize = 1024 * 1024 * 1024);
    InferencingContext(rhi::IDevice* device, size_t defaultPageSize = 1024 * 1024 * 1024);
    ComPtr<rhi::IComputePipeline> createComputePipeline(
        const char* entryPointName,
        Slang::ConstArrayView<String> specArgs);
    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob);

    inline rhi::IDevice* getDevice() const { return device; }

    StackAllocator* getAllocator() const { return allocator; }

    InferencingTask createTask();

    void pushAllocScope() { allocator->push(); }
    void popAllocScope() { allocator->pop(); }

    // RAII helper for scoped scratch allocations.
    // All scratch tensors allocated within the scope are freed when the scope exits.
    //
    // IMPORTANT: Only use ScratchScope AFTER task.execute() completes!
    // With async GPU execution, kernels are queued but not executed until task.execute().
    // If ScratchScope exits before task.execute(), buffers may be freed while GPU is still using
    // them.
    //
    // CORRECT usage:
    //   ctx->pushAllocScope();
    //   model.queueExecute(task, output, input);
    //   task.execute();  // GPU actually runs here
    //   ctx->popAllocScope();  // Safe to free now
    //
    // WRONG usage (inside queueExecute):
    //   void Model::queueExecute(...) {
    //       ScratchScope scope(ctx);  // BAD: exits before GPU runs!
    //       ...
    //   }
    class ScratchScope
    {
        InferencingContext* ctx;

    public:
        ScratchScope(InferencingContext* ctx)
            : ctx(ctx)
        {
            ctx->pushAllocScope();
        }
        ~ScratchScope() { ctx->popAllocScope(); }
        ScratchScope(const ScratchScope&) = delete;
        ScratchScope& operator=(const ScratchScope&) = delete;
    };
    BufferView allocScratchBuffer(size_t size, const char* label = nullptr);

    ComPtr<rhi::IBuffer> createPersistentBuffer(
        const void* data,
        size_t size,
        const char* label = nullptr);

    // ========================================================================
    // TENSOR ALLOCATION METHODS
    // ========================================================================

    // createTensor: Creates a PERSISTENT tensor with optional initial data.
    // - Returns RefPtr<Tensor> - caller owns the reference
    // - Tensor persists as long as the RefPtr is held
    // - Use for: model weights, input data, output buffers that outlive a single inference call
    //
    // IMPORTANT: Do NOT create tensors with createTensor inside queueExecute() methods!
    // The RefPtr will be destroyed when the function returns, but async GPU commands
    // may still reference the tensor. This causes crashes with INTERMEDIATE_MODE=0.
    // Instead, store such tensors as class members or use allocScratchTensor.
    RefPtr<Tensor> createTensor(
        ElementType elementType,
        const Shape& shape,
        size_t initialDataSize,
        const void* initialData = nullptr,
        const char* label = nullptr);

    // allocScratchTensor: Allocates a TEMPORARY tensor from the scratch pool.
    // - Returns TensorView - memory managed by InferencingContext
    // - Memory may be reused after popAllocScope() is called
    // - Use for: intermediate results within queueExecute() methods
    //
    // This is the preferred method for allocating intermediate buffers during
    // inference. The scratch allocator efficiently reuses memory and handles
    // lifetime management automatically.
    TensorView allocScratchTensor(
        ElementType elementType,
        const Shape& shape,
        const char* label = nullptr);

    template<typename T>
    List<T> readBuffer(BufferView buffer)
    {
        List<T> result;
        size_t count = buffer.size / sizeof(T);
        result.setCount(count);
        getDevice()->readBuffer(buffer.buffer, buffer.offset, buffer.size, result.getBuffer());
        return result;
    }

    template<typename T>
    List<T> readBuffer(TensorView tensorView)
    {
        return readBuffer<T>(tensorView.bufferView);
    }

    template<typename T>
    ComPtr<rhi::IBuffer> createPersistentBuffer(const List<T>& data, const char* label = nullptr)
    {
        return createPersistentBuffer(data.getBuffer(), data.getCount() * sizeof(T), label);
    }

    // Convenience overload of createTensor with typed data.
    // See createTensor() above for usage guidelines and warnings.
    template<typename T>
    RefPtr<Tensor> createTensor(
        ElementType elementType,
        const Shape& shape,
        const List<T>& data,
        const char* label = nullptr)
    {
        if (data.getCount() * sizeof(T) !=
            shape.getElementCount() * getElementTypeSize(elementType))
        {
            throw std::runtime_error("InferencingContext::createTensor: Data size does not match "
                                     "shape and element type");
        }
        return createTensor(
            elementType,
            shape,
            data.getCount() * sizeof(T),
            data.getBuffer(),
            label);
    }
};

template<typename... TArgs>
inline void logInfo(const char* format, TArgs... args)
{
    // Verbose logging disabled by default
    // Uncomment to enable: printf(format, std::forward<TArgs>(args)...);
    (void)format; // Suppress unused parameter warnings
    ((void)args, ...);
}

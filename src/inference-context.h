#pragma once

#include "core/slang-basic.h"
#include "core/slang-crypto.h"
#include "external/slang-rhi/include/slang-rhi.h"
#include "slang-com-ptr.h"
#include "slang.h"

using namespace Slang;

// Define this to 1 to enable intermediate mode that runs kernels synchronously
// on the CPU for easier debugging.
#define INTERMEDIATE_MODE 0

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

    rhi::IBuffer* allocateBuffer(const char* name, size_t size, void* initData = nullptr);

    void execute();
};

class InferencingContext : public RefObject
{
private:
    ComPtr<rhi::IDevice> device;
    ComPtr<slang::ISession> slangSession;
    ComPtr<slang::IModule> slangModule;
    Dictionary<MD5::Digest, ComPtr<rhi::IComputePipeline>> pipelineCache;

public:
    InferencingContext(rhi::IDevice* device);
    ComPtr<rhi::IComputePipeline> createComputePipeline(
        const char* entryPointName,
        Slang::ConstArrayView<String> specArgs);
    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob);

    inline rhi::IDevice* getDevice() const { return device; }

    InferencingTask createTask();

    ComPtr<rhi::IBuffer> createBuffer(const void* data, size_t size, const char* label = nullptr);

    template<typename T>
    ComPtr<rhi::IBuffer> createBuffer(const List<T>& data)
    {
        return createBuffer(data.getBuffer(), data.getCount() * sizeof(T));
    }
};

template<typename... TArgs>
inline void logInfo(const char* format, TArgs... args)
{
    printf(format, std::forward<TArgs>(args)...);
}

inline unsigned short floatToHalf(float val)
{
    uint32_t x = 0;
    memcpy(&x, &val, sizeof(float));

    unsigned short bits = (x >> 16) & 0x8000;
    unsigned short m = (x >> 12) & 0x07ff;
    unsigned int e = (x >> 23) & 0xff;
    if (e < 103)
        return bits;
    if (e > 142)
    {
        bits |= 0x7c00u;
        bits |= e == 255 && (x & 0x007fffffu);
        return bits;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

#pragma once

#include "core/slang-basic.h"
#include "core/slang-io.h"
#include "external/slang-rhi/include/slang-rhi.h"

#include <atomic>

using namespace Slang;

// ============================================================================
// FileShaderCache - Persistent shader/pipeline cache using filesystem
// ============================================================================
//
// Usage:
//   FileShaderCache cache;  // Creates .shadercache folder if needed
//
//   DeviceDesc desc;
//   desc.persistentShaderCache = &cache;
//   desc.persistentPipelineCache = &cache;
//   rhiCreateDevice(&desc, device.writeRef());
//
// The cache stores entries as files in .shadercache/ folder.
// File names are hex-encoded SHA1 hashes of the keys.
//

class FileShaderCache : public rhi::IPersistentCache, public Slang::ComObject
{
public:
    static const char* kCacheDir; // ".shadercache"

private:
    // In-memory cache: key (hex string) -> data
    Dictionary<String, List<uint8_t>> cache;

    // Set of known files (populated on construction)
    HashSet<String> knownFiles;

    // Reference counting
    std::atomic<uint32_t> refCount{1};

public:
    FileShaderCache();

    // IPersistentCache interface
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    writeCache(ISlangBlob* key, ISlangBlob* data) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryCache(ISlangBlob* key, ISlangBlob** outData) override;

    void* getInterface(const SlangUUID& uuid)
    {
        if (uuid == rhi::IPersistentCache::getTypeGuid() || uuid == ISlangUnknown::getTypeGuid())
            return static_cast<rhi::IPersistentCache*>(this);
        return nullptr;
    }

    // ISlangUnknown interface
    SLANG_COM_OBJECT_IUNKNOWN_ALL

private:
    // Convert SHA1 bytes to hex string for filenames
    static String bytesToHex(const uint8_t* data, size_t size);

    // Get full path for a cache entry
    String getCachePath(const String& hexKey) const;

    // Load a file from disk into cache
    bool loadFromDisk(const String& hexKey);
};

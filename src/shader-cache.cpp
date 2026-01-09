#include "shader-cache.h"

#include "core/slang-blob.h"

const char* FileShaderCache::kCacheDir = ".shadercache";

// ============================================================================
// Helper: Convert bytes to hex string
// ============================================================================

String FileShaderCache::bytesToHex(const uint8_t* data, size_t size)
{
    static const char hexChars[] = "0123456789abcdef";
    StringBuilder sb;
    sb.ensureCapacity(size * 2);
    for (size_t i = 0; i < size; i++)
    {
        sb.appendChar(hexChars[(data[i] >> 4) & 0xF]);
        sb.appendChar(hexChars[data[i] & 0xF]);
    }
    return sb.toString();
}

String FileShaderCache::getCachePath(const String& hexKey) const
{
    return Path::combine(kCacheDir, hexKey);
}

// ============================================================================
// Constructor - Initialize cache directory and populate known files
// ============================================================================

// Visitor to collect filenames
class CacheFileVisitor : public Path::Visitor
{
public:
    HashSet<String>& files;
    CacheFileVisitor(HashSet<String>& f)
        : files(f)
    {
    }

    virtual void accept(Path::Type type, const UnownedStringSlice& filename) override
    {
        if (type == Path::Type::File)
        {
            files.add(String(filename));
        }
    }
};

FileShaderCache::FileShaderCache()
{
    // Create cache directory if it doesn't exist
    if (!File::exists(kCacheDir))
    {
        Path::createDirectory(kCacheDir);
    }

    // Enumerate existing cache files
    CacheFileVisitor visitor(knownFiles);
    Path::find(kCacheDir, nullptr, &visitor);
}

// ============================================================================
// Load from disk (lazy loading)
// ============================================================================

bool FileShaderCache::loadFromDisk(const String& hexKey)
{
    String path = getCachePath(hexKey);

    List<uint8_t> data;
    if (SLANG_FAILED(File::readAllBytes(path, data)))
    {
        return false;
    }

    cache.add(hexKey, std::move(data));
    return true;
}

// ============================================================================
// IPersistentCache: queryCache
// ============================================================================

SlangResult FileShaderCache::queryCache(ISlangBlob* key, ISlangBlob** outData)
{
    if (!key || !outData)
        return SLANG_E_INVALID_ARG;

    // Convert key to hex string
    String hexKey =
        bytesToHex(static_cast<const uint8_t*>(key->getBufferPointer()), key->getBufferSize());

    // Check in-memory cache first
    if (cache.containsKey(hexKey))
    {
        const List<uint8_t>& data = cache[hexKey];
        *outData = RawBlob::create(data.getBuffer(), data.getCount()).detach();
        return SLANG_OK;
    }

    // Check if file exists on disk
    if (knownFiles.contains(hexKey))
    {
        if (loadFromDisk(hexKey))
        {
            const List<uint8_t>& data = cache[hexKey];
            *outData = RawBlob::create(data.getBuffer(), data.getCount()).detach();
            return SLANG_OK;
        }
    }

    // Cache miss
    *outData = nullptr;
    return SLANG_E_NOT_FOUND;
}

// ============================================================================
// IPersistentCache: writeCache
// ============================================================================

SlangResult FileShaderCache::writeCache(ISlangBlob* key, ISlangBlob* data)
{
    if (!key || !data)
        return SLANG_E_INVALID_ARG;

    // Convert key to hex string
    String hexKey =
        bytesToHex(static_cast<const uint8_t*>(key->getBufferPointer()), key->getBufferSize());

    // Store in memory
    List<uint8_t> dataList;
    dataList.setCount(data->getBufferSize());
    memcpy(dataList.getBuffer(), data->getBufferPointer(), data->getBufferSize());
    cache[hexKey] = std::move(dataList);

    // Write to disk
    String path = getCachePath(hexKey);
    SLANG_RETURN_ON_FAIL(
        File::writeAllBytes(path, data->getBufferPointer(), data->getBufferSize()));

    // Add to known files
    knownFiles.add(hexKey);

    return SLANG_OK;
}

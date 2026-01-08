#include "test-utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// Search prefixes to try when looking for test files
static const char* kSearchPrefixes[] = {
    "",
    "../",
    "../../",
    "../../../",
};

String getTestFilePath(String subpath)
{
    for (const char* prefix : kSearchPrefixes)
    {
        String fullPath = String(prefix) + subpath;
        if (File::exists(fullPath))
        {
            return fullPath;
        }
    }

    // Not found - return empty string
    return String();
}

SlangResult loadBinaryTensor(const String& path, List<float>& outData)
{
    List<uint8_t> bytes;
    SLANG_RETURN_ON_FAIL(File::readAllBytes(path, bytes));

    size_t numFloats = bytes.getCount() / sizeof(float);
    outData.setCount(numFloats);
    memcpy(outData.getBuffer(), bytes.getBuffer(), bytes.getCount());

    return SLANG_OK;
}

bool checkApproxEqual(
    const List<float>& actual,
    const List<float>& expected,
    float rtol,
    float atol)
{
    if (actual.getCount() != expected.getCount())
    {
        printf(
            "Size mismatch: got %lld, expected %lld\n",
            (long long)actual.getCount(),
            (long long)expected.getCount());
        return false;
    }

    int errorCount = 0;
    for (Index i = 0; i < actual.getCount(); i++)
    {
        float diff = std::abs(actual[i] - expected[i]);
        float threshold = atol + rtol * std::abs(expected[i]);
        if (diff > threshold)
        {
            if (errorCount < 10)
            {
                printf(
                    "Mismatch at %lld: got %f, expected %f (diff=%f, threshold=%f)\n",
                    (long long)i,
                    actual[i],
                    expected[i],
                    diff,
                    threshold);
            }
            errorCount++;
        }
    }

    if (errorCount > 0)
    {
        printf("Total errors: %d / %lld\n", errorCount, (long long)actual.getCount());
        return false;
    }
    return true;
}


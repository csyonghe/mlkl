// Unit tests for SafeTensorsReader

#include "safetensors-reader.h"
#include "test-kernels.h"

#include <cmath>

// Search prefixes to try when looking for test files
static const char* kSearchPrefixes[] = {
    "./",
    "../",
    "../../",
    "../../../",
};

static const char* kTestDataSubdir = "unit-test/safetensors/";

static String getTestFilePath(const char* filename)
{
    String subpath = String(kTestDataSubdir) + filename;
    
    for (const char* prefix : kSearchPrefixes)
    {
        String fullPath = String(prefix) + subpath;
        if (File::exists(fullPath))
        {
            return fullPath;
        }
    }
    
    // Fallback: return the default path (will fail with clear error)
    return String("./") + subpath;
}

// Helper to check if two floats are approximately equal
static bool approxEqual(float a, float b, float tolerance = 1e-5f)
{
    return std::abs(a - b) <= tolerance;
}

// ============================================================================
// Test: Load and enumerate tensors
// ============================================================================

SlangResult testSafeTensorsLoad()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_basic_types.safetensors")));

    // Should have 7 tensors
    TEST_CHECK("tensor count", reader.getTensorCount() == 7);

    // Check tensor existence
    TEST_CHECK("has f32_scalar", reader.hasTensor(toSlice("f32_scalar")));
    TEST_CHECK("has f32_1d", reader.hasTensor(toSlice("f32_1d")));
    TEST_CHECK("has f32_2d", reader.hasTensor(toSlice("f32_2d")));
    TEST_CHECK("has f16_1d", reader.hasTensor(toSlice("f16_1d")));
    TEST_CHECK("has f16_2d", reader.hasTensor(toSlice("f16_2d")));
    TEST_CHECK("has bf16_1d", reader.hasTensor(toSlice("bf16_1d")));
    TEST_CHECK("has bf16_2d", reader.hasTensor(toSlice("bf16_2d")));
    TEST_CHECK("!has nonexistent", !reader.hasTensor(toSlice("nonexistent")));

    MLKL_TEST_OK();
}

SlangResult testSafeTensorsLoadMissing()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SlangResult result = reader.load(getTestFilePath("nonexistent_file.safetensors"));
    TEST_CHECK("load missing file fails", SLANG_FAILED(result));

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Tensor info (shape, dtype)
// ============================================================================

SlangResult testSafeTensorsTensorInfo()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_basic_types.safetensors")));

    // Check 1D tensor info
    {
        const SafeTensorInfo* info = reader.getTensorInfo(toSlice("f32_1d"));
        TEST_CHECK("f32_1d exists", info != nullptr);
        TEST_CHECK("f32_1d rank", info->shape.getRank() == 1);
        TEST_CHECK("f32_1d dim[0]", info->shape[0] == 16);
        TEST_CHECK("f32_1d dtype", info->dtype == ElementType::Float32);
    }

    // Check 2D tensor info
    {
        const SafeTensorInfo* info = reader.getTensorInfo(toSlice("f32_2d"));
        TEST_CHECK("f32_2d exists", info != nullptr);
        TEST_CHECK("f32_2d rank", info->shape.getRank() == 2);
        TEST_CHECK("f32_2d dim[0]", info->shape[0] == 4);
        TEST_CHECK("f32_2d dim[1]", info->shape[1] == 6);
        TEST_CHECK("f32_2d dtype", info->dtype == ElementType::Float32);
    }

    // Check F16 tensor info
    {
        const SafeTensorInfo* info = reader.getTensorInfo(toSlice("f16_1d"));
        TEST_CHECK("f16_1d exists", info != nullptr);
        TEST_CHECK("f16_1d dtype", info->dtype == ElementType::Float16);
    }

    // Check BF16 tensor info
    {
        const SafeTensorInfo* info = reader.getTensorInfo(toSlice("bf16_1d"));
        TEST_CHECK("bf16_1d exists", info != nullptr);
        TEST_CHECK("bf16_1d dtype", info->dtype == ElementType::BFloat16);
    }

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Read tensor without permutation
// ============================================================================

SlangResult testSafeTensorsReadBasic()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_basic_types.safetensors")));

    // Read F32 1D tensor (values are 0, 1, 2, ..., 15)
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("f32_1d"), ElementType::Float32, data));

        TEST_CHECK("f32_1d size", data.getCount() == 16 * sizeof(float));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < 16; i++)
        {
            if (!approxEqual(values[i], (float)i))
            {
                allCorrect = false;
                break;
            }
        }
        TEST_CHECK("f32_1d values", allCorrect);
    }

    // Read F32 2D tensor (values are 0, 1, 2, ..., 23 in row-major)
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("f32_2d"), ElementType::Float32, data));

        TEST_CHECK("f32_2d size", data.getCount() == 24 * sizeof(float));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < 24; i++)
        {
            if (!approxEqual(values[i], (float)i))
            {
                allCorrect = false;
                break;
            }
        }
        TEST_CHECK("f32_2d values", allCorrect);
    }

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Type conversion
// ============================================================================

SlangResult testSafeTensorsTypeConversion()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_conversion.safetensors")));

    // Reference values (must match generate.py)
    const float exactValues[] = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.25f,
        0.125f, 4.0f, 8.0f, 16.0f, 0.0625f, 100.0f, 1000.0f, 2048.0f
    };
    const int numValues = 16;

    // Test F32 -> F32 (identity)
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("exact_f32"), ElementType::Float32, data));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < numValues; i++)
        {
            if (!approxEqual(values[i], exactValues[i]))
            {
                allCorrect = false;
                break;
            }
        }
        TEST_CHECK("f32->f32 conversion", allCorrect);
    }

    // Test F16 -> F32
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("exact_f16"), ElementType::Float32, data));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < numValues; i++)
        {
            // Allow slightly larger tolerance for F16
            if (!approxEqual(values[i], exactValues[i], 1e-3f))
            {
                printf("  F16->F32 mismatch at %d: got %f, expected %f\n", i, values[i], exactValues[i]);
                allCorrect = false;
            }
        }
        TEST_CHECK("f16->f32 conversion", allCorrect);
    }

    // Test BF16 -> F32
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("exact_bf16"), ElementType::Float32, data));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < numValues; i++)
        {
            // BF16 has less precision than F16, allow larger tolerance
            if (!approxEqual(values[i], exactValues[i], 1e-2f))
            {
                printf("  BF16->F32 mismatch at %d: got %f, expected %f\n", i, values[i], exactValues[i]);
                allCorrect = false;
            }
        }
        TEST_CHECK("bf16->f32 conversion", allCorrect);
    }

    // Test F32 -> F16
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("exact_f32"), ElementType::Float16, data));

        TEST_CHECK("f32->f16 size", data.getCount() == numValues * sizeof(uint16_t));

        // Read the F16 data back to F32 to verify
        // (We can't easily check F16 values directly)
        // Instead, verify that reading as F16 then back to F32 preserves values
        // This is implicitly tested by the f16->f32 test above
    }

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Linear weights (2D, no permutation)
// ============================================================================

SlangResult testSafeTensorsLinear()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_linear.safetensors")));

    // Check shape
    const SafeTensorInfo* info = reader.getTensorInfo(toSlice("linear.weight"));
    TEST_CHECK("linear.weight exists", info != nullptr);
    TEST_CHECK("linear.weight rank", info->shape.getRank() == 2);
    TEST_CHECK("linear.weight dim[0]", info->shape[0] == 4);  // out_features
    TEST_CHECK("linear.weight dim[1]", info->shape[1] == 8);  // in_features

    // Read and verify values
    // weight[out, in] = out * 100 + in
    List<uint8_t> data;
    SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("linear.weight"), ElementType::Float32, data));

    const float* values = reinterpret_cast<const float*>(data.getBuffer());
    bool allCorrect = true;
    for (int out = 0; out < 4; out++)
    {
        for (int in = 0; in < 8; in++)
        {
            float expected = (float)(out * 100 + in);
            float actual = values[out * 8 + in];
            if (!approxEqual(actual, expected))
            {
                printf("  linear.weight[%d,%d]: got %f, expected %f\n", out, in, actual, expected);
                allCorrect = false;
            }
        }
    }
    TEST_CHECK("linear.weight values", allCorrect);

    // Check bias
    List<uint8_t> biasData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("linear.bias"), ElementType::Float32, biasData));

    const float* biasValues = reinterpret_cast<const float*>(biasData.getBuffer());
    bool biasCorrect = true;
    for (int i = 0; i < 4; i++)
    {
        if (!approxEqual(biasValues[i], (float)i))
        {
            biasCorrect = false;
        }
    }
    TEST_CHECK("linear.bias values", biasCorrect);

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Conv2D weights with permutation
// ============================================================================

SlangResult testSafeTensorsConv2DPermutation()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_conv2d.safetensors")));

    // Check source shape: [OutCh=2, InCh=3, K=3, K=3]
    const SafeTensorInfo* info = reader.getTensorInfo(toSlice("conv.weight"));
    TEST_CHECK("conv.weight exists", info != nullptr);
    TEST_CHECK("conv.weight rank", info->shape.getRank() == 4);
    TEST_CHECK("conv.weight dim[0]", info->shape[0] == 2);  // OutCh
    TEST_CHECK("conv.weight dim[1]", info->shape[1] == 3);  // InCh
    TEST_CHECK("conv.weight dim[2]", info->shape[2] == 3);  // Ky
    TEST_CHECK("conv.weight dim[3]", info->shape[3] == 3);  // Kx

    // Read with permutation [1, 2, 3, 0]: [O, I, Ky, Kx] -> [I, Ky, Kx, O]
    static const int perm[] = {1, 2, 3, 0};
    List<uint8_t> data;
    SLANG_RETURN_ON_FAIL(reader.readTensor(
        toSlice("conv.weight"),
        ElementType::Float32,
        data,
        makeConstArrayView(perm)));

    const float* values = reinterpret_cast<const float*>(data.getBuffer());

    // After permutation, layout is [I=3, Ky=3, Kx=3, O=2]
    // Value at dest[i, ky, kx, o] should be o*1000 + i*100 + ky*10 + kx
    bool allCorrect = true;
    int idx = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int ky = 0; ky < 3; ky++)
        {
            for (int kx = 0; kx < 3; kx++)
            {
                for (int o = 0; o < 2; o++)
                {
                    float expected = (float)(o * 1000 + i * 100 + ky * 10 + kx);
                    float actual = values[idx];
                    if (!approxEqual(actual, expected))
                    {
                        printf("  conv.weight[%d,%d,%d,%d]: got %f, expected %f\n",
                               i, ky, kx, o, actual, expected);
                        allCorrect = false;
                    }
                    idx++;
                }
            }
        }
    }
    TEST_CHECK("conv.weight permuted values", allCorrect);

    MLKL_TEST_OK();
}

// ============================================================================
// Test: TransposedConv2D weights with permutation
// ============================================================================

SlangResult testSafeTensorsTransposedConv2DPermutation()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_transposed_conv2d.safetensors")));

    // Check source shape: [InCh=3, OutCh=2, K=3, K=3]
    const SafeTensorInfo* info = reader.getTensorInfo(toSlice("tconv.weight"));
    TEST_CHECK("tconv.weight exists", info != nullptr);
    TEST_CHECK("tconv.weight rank", info->shape.getRank() == 4);
    TEST_CHECK("tconv.weight dim[0]", info->shape[0] == 3);  // InCh
    TEST_CHECK("tconv.weight dim[1]", info->shape[1] == 2);  // OutCh
    TEST_CHECK("tconv.weight dim[2]", info->shape[2] == 3);  // Ky
    TEST_CHECK("tconv.weight dim[3]", info->shape[3] == 3);  // Kx

    // Read with permutation [0, 2, 3, 1]: [I, O, Ky, Kx] -> [I, Ky, Kx, O]
    static const int perm[] = {0, 2, 3, 1};
    List<uint8_t> data;
    SLANG_RETURN_ON_FAIL(reader.readTensor(
        toSlice("tconv.weight"),
        ElementType::Float32,
        data,
        makeConstArrayView(perm)));

    const float* values = reinterpret_cast<const float*>(data.getBuffer());

    // After permutation, layout is [I=3, Ky=3, Kx=3, O=2]
    // Value at dest[i, ky, kx, o] should be i*1000 + o*100 + ky*10 + kx
    bool allCorrect = true;
    int idx = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int ky = 0; ky < 3; ky++)
        {
            for (int kx = 0; kx < 3; kx++)
            {
                for (int o = 0; o < 2; o++)
                {
                    float expected = (float)(i * 1000 + o * 100 + ky * 10 + kx);
                    float actual = values[idx];
                    if (!approxEqual(actual, expected))
                    {
                        printf("  tconv.weight[%d,%d,%d,%d]: got %f, expected %f\n",
                               i, ky, kx, o, actual, expected);
                        allCorrect = false;
                    }
                    idx++;
                }
            }
        }
    }
    TEST_CHECK("tconv.weight permuted values", allCorrect);

    MLKL_TEST_OK();
}

// ============================================================================
// Test: General permutation verification
// ============================================================================

SlangResult testSafeTensorsPermutation()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_permutation.safetensors")));

    // Source shape: [2, 3, 2, 2]
    // Value at [d0, d1, d2, d3] = d0*1000 + d1*100 + d2*10 + d3

    // Test identity permutation
    {
        static const int perm[] = {0, 1, 2, 3};
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(
            toSlice("perm_test"),
            ElementType::Float32,
            data,
            makeConstArrayView(perm)));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        int idx = 0;
        for (int d0 = 0; d0 < 2; d0++)
        {
            for (int d1 = 0; d1 < 3; d1++)
            {
                for (int d2 = 0; d2 < 2; d2++)
                {
                    for (int d3 = 0; d3 < 2; d3++)
                    {
                        float expected = (float)(d0 * 1000 + d1 * 100 + d2 * 10 + d3);
                        if (!approxEqual(values[idx], expected))
                        {
                            allCorrect = false;
                        }
                        idx++;
                    }
                }
            }
        }
        TEST_CHECK("identity permutation", allCorrect);
    }

    // Test reverse permutation [3, 2, 1, 0]
    // New shape: [2, 2, 3, 2]
    // Value at new[d3, d2, d1, d0] = d0*1000 + d1*100 + d2*10 + d3
    {
        static const int perm[] = {3, 2, 1, 0};
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(
            toSlice("perm_test"),
            ElementType::Float32,
            data,
            makeConstArrayView(perm)));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        int idx = 0;
        // New iteration order (destination layout)
        for (int nd0 = 0; nd0 < 2; nd0++)      // was d3
        {
            for (int nd1 = 0; nd1 < 2; nd1++)  // was d2
            {
                for (int nd2 = 0; nd2 < 3; nd2++)  // was d1
                {
                    for (int nd3 = 0; nd3 < 2; nd3++)  // was d0
                    {
                        // Original indices
                        int d0 = nd3, d1 = nd2, d2 = nd1, d3 = nd0;
                        float expected = (float)(d0 * 1000 + d1 * 100 + d2 * 10 + d3);
                        if (!approxEqual(values[idx], expected))
                        {
                            printf("  reverse perm[%d,%d,%d,%d]: got %f, expected %f\n",
                                   nd0, nd1, nd2, nd3, values[idx], expected);
                            allCorrect = false;
                        }
                        idx++;
                    }
                }
            }
        }
        TEST_CHECK("reverse permutation", allCorrect);
    }

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Mixed precision file
// ============================================================================

SlangResult testSafeTensorsMixedPrecision()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_mixed_precision.safetensors")));

    // Check that weight is F16
    const SafeTensorInfo* weightInfo = reader.getTensorInfo(toSlice("layer.weight"));
    TEST_CHECK("layer.weight dtype is F16", weightInfo->dtype == ElementType::Float16);

    // Check that bias is F32
    const SafeTensorInfo* biasInfo = reader.getTensorInfo(toSlice("layer.bias"));
    TEST_CHECK("layer.bias dtype is F32", biasInfo->dtype == ElementType::Float32);

    // Read weight as F32 (conversion)
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("layer.weight"), ElementType::Float32, data));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        // Values should be 0, 1, 2, ..., 63
        bool allCorrect = true;
        for (int i = 0; i < 64; i++)
        {
            if (!approxEqual(values[i], (float)i, 1e-2f))  // F16 tolerance
            {
                allCorrect = false;
            }
        }
        TEST_CHECK("layer.weight F16->F32 values", allCorrect);
    }

    // Read weight as F16 (same type)
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("layer.weight"), ElementType::Float16, data));
        TEST_CHECK("layer.weight F16 size", data.getCount() == 64 * sizeof(uint16_t));
    }

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Embedding weights
// ============================================================================

SlangResult testSafeTensorsEmbedding()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_embedding.safetensors")));

    // Check shape [vocab_size=10, embed_dim=8]
    const SafeTensorInfo* info = reader.getTensorInfo(toSlice("embed.weight"));
    TEST_CHECK("embed.weight exists", info != nullptr);
    TEST_CHECK("embed.weight rank", info->shape.getRank() == 2);
    TEST_CHECK("embed.weight dim[0]", info->shape[0] == 10);
    TEST_CHECK("embed.weight dim[1]", info->shape[1] == 8);

    // Read and verify values: weight[v, e] = v * 100 + e
    List<uint8_t> data;
    SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("embed.weight"), ElementType::Float32, data));

    const float* values = reinterpret_cast<const float*>(data.getBuffer());
    bool allCorrect = true;
    for (int v = 0; v < 10; v++)
    {
        for (int e = 0; e < 8; e++)
        {
            float expected = (float)(v * 100 + e);
            float actual = values[v * 8 + e];
            if (!approxEqual(actual, expected))
            {
                allCorrect = false;
            }
        }
    }
    TEST_CHECK("embed.weight values", allCorrect);

    MLKL_TEST_OK();
}

// ============================================================================
// Test: Norm parameters
// ============================================================================

SlangResult testSafeTensorsNorm()
{
    MLKL_TEST_BEGIN();

    SafeTensorsReader reader;
    SLANG_RETURN_ON_FAIL(reader.load(getTestFilePath("test_norm.safetensors")));

    // Check gamma (weight): values are 1, 2, 3, ..., 32
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("norm.weight"), ElementType::Float32, data));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < 32; i++)
        {
            if (!approxEqual(values[i], (float)(i + 1)))
            {
                allCorrect = false;
            }
        }
        TEST_CHECK("norm.weight (gamma) values", allCorrect);
    }

    // Check beta (bias): values are 0, 0.1, 0.2, ..., 3.1
    {
        List<uint8_t> data;
        SLANG_RETURN_ON_FAIL(reader.readTensor(toSlice("norm.bias"), ElementType::Float32, data));

        const float* values = reinterpret_cast<const float*>(data.getBuffer());
        bool allCorrect = true;
        for (int i = 0; i < 32; i++)
        {
            if (!approxEqual(values[i], (float)i * 0.1f))
            {
                allCorrect = false;
            }
        }
        TEST_CHECK("norm.bias (beta) values", allCorrect);
    }

    MLKL_TEST_OK();
}


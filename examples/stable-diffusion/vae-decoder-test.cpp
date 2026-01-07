#include "vae-decoder-test.h"

#include "safetensors-reader.h"
#include "vae-decoder.h"

#include <cmath>

// Search prefixes to try when looking for test files
static const char* kSearchPrefixes[] = {
    "./",
    "../",
    "../../",
    "../../../",
};

static String getTestFilePath(String subpath)
{
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


// Helper to check if two float arrays are approximately equal
static bool checkApproxEqual(
    const List<float>& actual,
    const List<float>& expected,
    float rtol = 1e-3f,
    float atol = 1e-5f)
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

// Load binary tensor from file
static SlangResult loadBinaryTensor(const String& path, List<float>& outData)
{
    List<uint8_t> bytes;
    SLANG_RETURN_ON_FAIL(File::readAllBytes(path, bytes));

    size_t numFloats = bytes.getCount() / sizeof(float);
    outData.setCount(numFloats);
    memcpy(outData.getBuffer(), bytes.getBuffer(), bytes.getCount());

    return SLANG_OK;
}

SlangResult testVAEResNetBlock(InferencingContext* ctx)
{
    printf("Running testVAEResNetBlock...\n");

    // Test construction with various channel configurations
    {
        VAEResNetBlock block(ctx, 64, 64, 32); // Same in/out channels
        printf("  Created block 64->64, 32 groups\n");
    }
    {
        VAEResNetBlock block(ctx, 256, 128, 32); // Different in/out (needs skip conv)
        printf("  Created block 256->128, 32 groups\n");
    }
    {
        VAEResNetBlock block(ctx, 512, 512, 32); // SD VAE typical size
        printf("  Created block 512->512, 32 groups\n");
    }

    // Verify allocateResultBuffer returns correct shape
    VAEResNetBlock block(ctx, 64, 128, 32);
    auto output = block.allocateResultBuffer(ElementType::Float32, 16, 16, 2);
    if (output.shape[0] != 2 || output.shape[1] != 16 || output.shape[2] != 16 ||
        output.shape[3] != 128)
    {
        printf("testVAEResNetBlock: FAILED (wrong output shape)\n");
        return SLANG_FAIL;
    }

    printf("testVAEResNetBlock: PASSED\n");
    return SLANG_OK;
}

SlangResult testVAEAttentionBlock(InferencingContext* ctx)
{
    printf("Running testVAEAttentionBlock...\n");

    // Test construction with various configurations
    {
        VAEAttentionBlock block(ctx, 64, 1); // 64 channels, 1 head (head_dim=64)
        printf("  Created attention block 64 channels, 1 head\n");
    }
    {
        VAEAttentionBlock block(ctx, 512, 1); // SD VAE typical size
        printf("  Created attention block 512 channels, 1 head\n");
    }
    {
        VAEAttentionBlock block(ctx, 256, 8); // 256 channels, 8 heads (head_dim=32)
        printf("  Created attention block 256 channels, 8 heads\n");
    }

    // Verify allocateResultBuffer returns correct shape
    VAEAttentionBlock block(ctx, 512, 1);
    auto output = block.allocateResultBuffer(ElementType::Float32, 8, 8, 1);
    if (output.shape[0] != 1 || output.shape[1] != 8 || output.shape[2] != 8 ||
        output.shape[3] != 512)
    {
        printf("testVAEAttentionBlock: FAILED (wrong output shape)\n");
        return SLANG_FAIL;
    }

    printf("testVAEAttentionBlock: PASSED\n");
    return SLANG_OK;
}

SlangResult testVAEUpBlock(InferencingContext* ctx)
{
    printf("Running testVAEUpBlock...\n");

    // Test construction with upsample
    {
        VAEUpBlock block(ctx, 512, 512, true, 3); // SD VAE up_block_0
        printf("  Created up block 512->512 with upsample, 3 resnets\n");
    }
    {
        VAEUpBlock block(ctx, 512, 256, true, 3); // SD VAE up_block_2
        printf("  Created up block 512->256 with upsample, 3 resnets\n");
    }
    {
        VAEUpBlock block(ctx, 256, 128, false, 3); // SD VAE up_block_3 (no upsample)
        printf("  Created up block 256->128 without upsample, 3 resnets\n");
    }

    // Verify allocateResultBuffer returns correct shape with upsample
    {
        VAEUpBlock block(ctx, 512, 512, true, 3);
        auto output = block.allocateResultBuffer(ElementType::Float32, 64, 64, 1);
        // With 2x upsample: 64 -> 128
        if (output.shape[0] != 1 || output.shape[1] != 128 || output.shape[2] != 128 ||
            output.shape[3] != 512)
        {
            printf("testVAEUpBlock: FAILED (wrong output shape with upsample)\n");
            return SLANG_FAIL;
        }
    }

    // Verify allocateResultBuffer without upsample
    {
        VAEUpBlock block(ctx, 256, 128, false, 3);
        auto output = block.allocateResultBuffer(ElementType::Float32, 512, 512, 1);
        // No upsample: 512 stays 512
        if (output.shape[0] != 1 || output.shape[1] != 512 || output.shape[2] != 512 ||
            output.shape[3] != 128)
        {
            printf("testVAEUpBlock: FAILED (wrong output shape without upsample)\n");
            return SLANG_FAIL;
        }
    }

    printf("testVAEUpBlock: PASSED\n");
    return SLANG_OK;
}

SlangResult testVAEDecoderSmall(InferencingContext* ctx)
{
    printf("Running testVAEDecoderSmall...\n");

    // Test that the full decoder can be instantiated
    VAEDecoder decoder(ctx);
    printf("  Created full VAE decoder\n");

    // Verify allocateResultBuffer returns correct shape
    // For latent [1, 64, 64, 4], output should be [1, 512, 512, 3]
    auto output = decoder.allocateResultBuffer(ElementType::Float32, 64, 64, 1);
    if (output.shape[0] != 1 || output.shape[1] != 512 || output.shape[2] != 512 ||
        output.shape[3] != 3)
    {
        printf("testVAEDecoderSmall: FAILED (wrong output shape)\n");
        printf(
            "  Expected [1, 512, 512, 3], got [%d, %d, %d, %d]\n",
            output.shape[0],
            output.shape[1],
            output.shape[2],
            output.shape[3]);
        return SLANG_FAIL;
    }

    // Small latent test
    auto smallOutput = decoder.allocateResultBuffer(ElementType::Float32, 8, 8, 1);
    if (smallOutput.shape[1] != 64 || smallOutput.shape[2] != 64)
    {
        printf("testVAEDecoderSmall: FAILED (wrong small output shape)\n");
        return SLANG_FAIL;
    }

    printf("testVAEDecoderSmall: PASSED\n");
    return SLANG_OK;
}

SlangResult testVAEDecoderSD15(InferencingContext* ctx)
{
    printf("Running testVAEDecoderSD15...\n");

    // Try to load SD 1.5 VAE weights
    SafeTensorsReader reader;
    SlangResult loadResult = reader.load(
        getTestFilePath("examples/stable-diffusion/model/diffusion_pytorch_model.safetensors"));
    List<String> names;
    reader.getTensorNames(names);
    for (auto& name : names)
    {
        if (name.indexOf("attention") != -1 || name.indexOf("attn") != -1)
            printf("  %s\n", name.getBuffer());
    }

    if (SLANG_FAILED(loadResult))
    {
        printf("testVAEDecoderSD15: SKIPPED (weights not found)\n");
        printf("  To run this test, download SD 1.5 VAE weights:\n");
        printf("  huggingface-cli download stabilityai/sd-vae-ft-mse\n");
        return SLANG_OK; // Skip, don't fail
    }

    VAEDecoder decoder(ctx);
    SLANG_RETURN_ON_FAIL(decoder.loadParams(reader));

    // Load test input (if available)
    List<float> inputData;
    SlangResult inputResult =
        loadBinaryTensor(getTestFilePath("test_data/vae_latent_input.bin"), inputData);

    if (SLANG_FAILED(inputResult))
    {
        // Generate random input
        printf("  Using random input (no test_data/vae_latent_input.bin found)\n");
        inputData.setCount(1 * 64 * 64 * 4);
        for (Index i = 0; i < inputData.getCount(); i++)
        {
            inputData[i] = ((float)(i % 1000) / 1000.0f - 0.5f) * 2.0f;
        }
    }

    auto inputTensor =
        ctx->createTensor(ElementType::Float32, Shape(1, 64, 64, 4), inputData, "LatentInput");

    auto outputTensor = decoder.allocateResultBuffer(ElementType::Float32, 64, 64, 1);

    auto task = ctx->createTask();
    decoder.queueExecute(task, outputTensor, inputTensor->getView());
    task.execute();

    auto output = ctx->readBuffer<float>(outputTensor);

    // Verify output shape
    if (output.getCount() != 1 * 512 * 512 * 3)
    {
        printf(
            "testVAEDecoderSD15: FAILED (expected %d elements, got %lld)\n",
            1 * 512 * 512 * 3,
            (long long)output.getCount());
        return SLANG_FAIL;
    }

    // Note: clamp to [-1, 1] is fused into the decoder's final conv layer

    // Check against reference if available
    List<float> expectedOutput;
    if (SLANG_SUCCEEDED(
            loadBinaryTensor(getTestFilePath("test_data/vae_decoder_output.bin"), expectedOutput)))
    {
        if (!checkApproxEqual(output, expectedOutput, 1e-2f, 1e-4f))
        {
            printf("testVAEDecoderSD15: FAILED (output mismatch)\n");
            return SLANG_FAIL;
        }
    }
    else
    {
        printf("  No reference output found, skipping numerical validation\n");
    }

    printf("testVAEDecoderSD15: PASSED\n");
    return SLANG_OK;
}

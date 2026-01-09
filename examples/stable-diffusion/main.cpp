// Stable Diffusion Example
// With --test: runs all component tests
// Without --test: generates an image from a hardcoded prompt

#include "clip-encoder-test.h"
#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "example-base/test-utils.h"
#include "inference-context.h"
#include "sd-image-generator.h"
#include "tokenizer-test.h"
#include "unet-test.h"
#include "vae-decoder-test.h"

#include <cstdio>
#include <cstring>

using namespace Slang;

static const ExampleResources resourceBase("stable-diffusion");

// ============================================================================
// Test Mode
// ============================================================================
static SlangResult runTests(InferencingContext* ctx)
{
    printf("=== Running Stable Diffusion Component Tests ===\n\n");

    // Test tokenizer
    printf("--- Tokenizer Tests ---\n");
    SLANG_RETURN_ON_FAIL(testCLIPTokenizer());

    // Test VAE decoder
    printf("\n--- VAE Decoder Tests ---\n");
    SLANG_RETURN_ON_FAIL(testVAEDecoderSD15(ctx));

    // Test UNet
    printf("\n--- UNet Tests ---\n");
    SLANG_RETURN_ON_FAIL(testSDUNet(ctx));

    // Test CLIP encoder
    printf("\n--- CLIP Encoder Tests ---\n");
    SLANG_RETURN_ON_FAIL(testCLIPEncoderSD15(ctx));

    printf("\n=== All tests passed! ===\n");
    return SLANG_OK;
}

// ============================================================================
// Image Generation Mode
// ============================================================================
static SlangResult generateImage(InferencingContext* ctx)
{
    // Configuration
    const char* prompt = "a beautiful sunset over mountains, digital art, highly detailed";
    const unsigned int seed = 1377;
    const int inferenceSteps = 30;
    const float guidanceScale = 7.5f; // CFG: 1.0 = no guidance, 7.5 = typical, higher = stronger
    const char* outputPath = "result.png";

    printf("=== Stable Diffusion Image Generation ===\n\n");
    printf("Prompt: \"%s\"\n", prompt);
    printf("Seed: %u\n", seed);
    printf("Steps: %d\n", inferenceSteps);
    printf("Guidance scale: %.1f\n\n", guidanceScale);

    // ========================================================================
    // Initialize generator
    // ========================================================================
    printf("Initializing SD generator (compiling shaders)...\n");
    RefPtr<SDImageGenerator> generator = new SDImageGenerator(ctx);

    // ========================================================================
    // Load models
    // ========================================================================
    printf("Loading models...\n");
    String modelsDir = getTestFilePath("models");
    SLANG_RETURN_ON_FAIL(generator->loadModels(modelsDir));

    // ========================================================================
    // Generate image
    // ========================================================================
    printf("Generating image...\n");
    TensorView imageView = generator->generateImage(prompt, seed, inferenceSteps, guidanceScale);
    if (!imageView)
    {
        printf("Failed to generate image\n");
        return SLANG_FAIL;
    }

    // Print performance stats
    const auto& perf = generator->getLastPerfStats();
    printf("\n=== Performance ===\n");
    printf("  CLIP encoding:        %7.1f ms\n", perf.clipTimeMs);
    printf(
        "  Diffusion (%2d steps): %7.1f ms (%.1f ms/step)\n",
        perf.inferenceSteps,
        perf.diffusionTimeMs,
        perf.msPerStep());
    printf("  VAE decoding:         %7.1f ms\n", perf.vaeTimeMs);
    printf("  Total:                %7.1f ms\n", perf.totalTimeMs);

    // ========================================================================
    // Save image
    // ========================================================================
    printf("\nSaving image to %s...\n", outputPath);
    auto imageData = ctx->readBuffer<float>(imageView);
    writeImagePNG(
        outputPath,
        generator->getImageWidth(),
        generator->getImageHeight(),
        3,
        imageData);

    printf("Done! Image saved to %s\n", outputPath);
    return SLANG_OK;
}

// ============================================================================
// Main Entry Point
// ============================================================================
int main(int argc, char** argv)
{
    // Check for --test flag
    bool testMode = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--test") == 0)
        {
            testMode = true;
            break;
        }
    }

    // Create inferencing context
    RefPtr<InferencingContext> ctx = new InferencingContext();

    SlangResult result;
    if (testMode)
    {
        result = runTests(ctx);
    }
    else
    {
        result = generateImage(ctx);
    }

    return SLANG_SUCCEEDED(result) ? 0 : 1;
}

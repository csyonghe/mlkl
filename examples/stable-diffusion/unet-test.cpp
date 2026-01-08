#include "unet-test.h"

#include "example-base/test-utils.h"
#include "unet.h"

// ============================================================================
// Helper to print tensor names from SafeTensors
// ============================================================================
static void printUNetWeightNames(SafeTensorsReader& reader)
{
    printf("  Available UNet weight names:\n");
    List<String> names;
    reader.getTensorNames(names);
    for (auto& name : names)
    {
        // Filter to show just a few representative ones
        if (name.indexOf("down_blocks.0.resnets.0") != -1 ||
            name.indexOf("mid_block.resnets.0") != -1 ||
            name.indexOf("up_blocks.0.resnets.0") != -1 || name.indexOf("time_embedding") != -1 ||
            name.indexOf("conv_in") != -1 || name.indexOf("conv_out") != -1)
        {
            printf("    %s\n", name.getBuffer());
        }
    }
}

// ============================================================================
// Test SDResNetBlock
// ============================================================================
SlangResult testSDResNetBlock(InferencingContext* ctx)
{
    printf("testSDResNetBlock: Testing construction and shapes...\n");

    // Test with channel change
    SDResNetBlock block(ctx, 320, 640, 1280);

    // Verify construction
    if (!block.norm1 || !block.conv1 || !block.timeProj || !block.norm2 || !block.conv2)
    {
        printf("testSDResNetBlock: FAILED (null kernels)\n");
        return SLANG_FAIL;
    }

    if (!block.hasResidualConv || !block.residualConv)
    {
        printf("testSDResNetBlock: FAILED (should have residual conv)\n");
        return SLANG_FAIL;
    }

    // Test without channel change
    SDResNetBlock block2(ctx, 640, 640, 1280);
    if (block2.hasResidualConv)
    {
        printf("testSDResNetBlock: FAILED (should not have residual conv)\n");
        return SLANG_FAIL;
    }

    printf("testSDResNetBlock: PASSED\n");
    return SLANG_OK;
}

// ============================================================================
// Test SDSelfAttention
// ============================================================================
SlangResult testSDSelfAttention(InferencingContext* ctx)
{
    printf("testSDSelfAttention: Testing construction...\n");

    SDSelfAttention attn(ctx, 640, 8); // 640 dim, 8 heads

    if (!attn.toQ || !attn.toK || !attn.toV || !attn.toOut)
    {
        printf("testSDSelfAttention: FAILED (null linear kernels)\n");
        return SLANG_FAIL;
    }

    if (!attn.flashAttn)
    {
        printf("testSDSelfAttention: FAILED (null flash attention kernel)\n");
        return SLANG_FAIL;
    }

    if (attn.headDim != 80) // 640 / 8 = 80
    {
        printf("testSDSelfAttention: FAILED (wrong headDim: %d, expected 80)\n", attn.headDim);
        return SLANG_FAIL;
    }

    printf("testSDSelfAttention: PASSED\n");
    return SLANG_OK;
}

// ============================================================================
// Test SDCrossAttention
// ============================================================================
SlangResult testSDCrossAttention(InferencingContext* ctx)
{
    printf("testSDCrossAttention: Testing construction...\n");

    SDCrossAttention attn(ctx, 640, 768, 8); // query=640, context=768 (CLIP), 8 heads

    if (!attn.toQ || !attn.toK || !attn.toV || !attn.toOut)
    {
        printf("testSDCrossAttention: FAILED (null linear kernels)\n");
        return SLANG_FAIL;
    }

    if (attn.headDim != 80)
    {
        printf("testSDCrossAttention: FAILED (wrong headDim: %d, expected 80)\n", attn.headDim);
        return SLANG_FAIL;
    }

    printf("testSDCrossAttention: PASSED\n");
    return SLANG_OK;
}

// ============================================================================
// Test SDSpatialTransformer
// ============================================================================
SlangResult testSDSpatialTransformer(InferencingContext* ctx)
{
    printf("testSDSpatialTransformer: Testing construction...\n");

    SDSpatialTransformer transformer(ctx, 640, 8, 768, 1); // 640 channels, 8 heads, 768 context

    if (!transformer.norm || !transformer.projIn || !transformer.projOut)
    {
        printf("testSDSpatialTransformer: FAILED (null projection kernels)\n");
        return SLANG_FAIL;
    }

    if (transformer.blocks.getCount() != 1)
    {
        printf(
            "testSDSpatialTransformer: FAILED (wrong block count: %lld)\n",
            (long long)transformer.blocks.getCount());
        return SLANG_FAIL;
    }

    printf("testSDSpatialTransformer: PASSED\n");
    return SLANG_OK;
}

// ============================================================================
// Test Full SDUNet with real weights
// ============================================================================
SlangResult testSDUNet(InferencingContext* ctx)
{
    printf("testSDUNet: Loading SD 1.5 UNet...\n");

    // Find the UNet weights file
    String weightsPath = getTestFilePath("models/unet.safetensors");
    if (weightsPath.getLength() == 0)
    {
        printf("  UNet weights not found. Run: python download_models.py\n");
        printf("testSDUNet: SKIPPED (no weights)\n");
        return SLANG_OK;
    }

    printf("  Found weights: %s\n", weightsPath.getBuffer());

    // Load weights
    SafeTensorsReader reader;
    if (SLANG_FAILED(reader.load(weightsPath)))
    {
        printf("testSDUNet: FAILED (couldn't load weights)\n");
        return SLANG_FAIL;
    }

    // Create UNet
    printf("  Creating UNet model...\n");
    SDUNet unet(ctx);

    // Load parameters
    printf("  Loading parameters...\n");
    if (SLANG_FAILED(unet.loadParams(reader)))
    {
        printf("testSDUNet: FAILED (couldn't load parameters)\n");
        return SLANG_FAIL;
    }

    // Load test inputs
    String latentPath = getTestFilePath("test_data/unet_input_latent.bin");
    String contextPath = getTestFilePath("test_data/unet_input_context.bin");
    String outputPath = getTestFilePath("test_data/unet_output.bin");

    List<float> latentData, contextData, expectedOutput;

    if (SLANG_FAILED(loadBinaryTensor(latentPath, latentData)) ||
        SLANG_FAILED(loadBinaryTensor(contextPath, contextData)) ||
        SLANG_FAILED(loadBinaryTensor(outputPath, expectedOutput)))
    {
        printf("  Test data not found. Run unet-test-generate.py to create test data.\n");
        printf("testSDUNet: SKIPPED (no test data)\n");
        return SLANG_OK;
    }

    // Create input tensors
    int batchSize = 1;
    int height = 64;
    int width = 64;
    int latentChannels = 4;
    int seqLen = 77;
    int contextDim = 768;
    int timestep = 500;

    auto latentTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, height, width, latentChannels),
        latentData,
        "LatentInput");

    auto contextTensor = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, seqLen, contextDim),
        contextData,
        "ContextInput");

    auto outputTensor = unet.allocateResultBuffer(ElementType::Float32, height, width, batchSize);

    // Run inference with debug output
    printf("  Running UNet inference (timestep=%d)...\n", timestep);
    {
        auto task = ctx->createTask();
        unet.queueExecute(
            task,
            outputTensor,
            latentTensor->getView(),
            timestep,
            contextTensor->getView());
        task.execute();
    }
    printf("  Inference complete.\n");

    // Read output and compare
    auto output = ctx->readBuffer<float>(outputTensor);

    // UNet is a very deep network (~100+ layers), so allow slightly higher tolerance
    // for floating-point accumulation differences
    if (!checkApproxEqual(output, expectedOutput, 1e-2f, 2e-4f))
    {
        printf("testSDUNet: FAILED (output mismatch)\n");
        return SLANG_FAIL;
    }

    printf("testSDUNet: PASSED\n");
    return SLANG_OK;
}

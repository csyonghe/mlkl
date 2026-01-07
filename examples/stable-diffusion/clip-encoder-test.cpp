#include "clip-encoder-test.h"

#include "clip-encoder.h"
#include "safetensors-reader.h"

#include <cmath>

// Search prefixes to try when looking for test files
static const char* kSearchPrefixes[] = {
    "./",
    "../",
    "../../",
    "../../../",
};

static String getTestFilePath(const char* subpath)
{
    for (auto prefix : kSearchPrefixes)
    {
        String path = String(prefix) + subpath;
        if (File::exists(path))
            return path;
    }
    return String("./") + subpath;
}

// Load binary tensor from file
static SlangResult loadBinaryTensor(const String& path, List<float>& outData)
{
    if (!File::exists(path))
        return SLANG_E_NOT_FOUND;
    
    List<uint8_t> bytes;
    SLANG_RETURN_ON_FAIL(File::readAllBytes(path, bytes));
    
    size_t numFloats = bytes.getCount() / sizeof(float);
    outData.setCount(numFloats);
    memcpy(outData.getBuffer(), bytes.getBuffer(), bytes.getCount());
    
    return SLANG_OK;
}

// Check approximate equality
static bool checkApproxEqual(
    const List<float>& actual,
    const List<float>& expected,
    float rtol = 1e-3f,
    float atol = 1e-5f)
{
    if (actual.getCount() != expected.getCount())
    {
        printf("Size mismatch: got %lld, expected %lld\n",
            (long long)actual.getCount(), (long long)expected.getCount());
        return false;
    }
    
    int errorCount = 0;
    for (Index i = 0; i < actual.getCount() && errorCount < 10; i++)
    {
        float diff = std::abs(actual[i] - expected[i]);
        float threshold = atol + rtol * std::abs(expected[i]);
        if (diff > threshold)
        {
            printf("Mismatch at %lld: got %.6f, expected %.6f (diff=%.6f, threshold=%.6f)\n",
                (long long)i, actual[i], expected[i], diff, threshold);
            errorCount++;
        }
    }
    
    if (errorCount > 0)
    {
        // Count total errors
        int totalErrors = 0;
        for (Index i = 0; i < actual.getCount(); i++)
        {
            float diff = std::abs(actual[i] - expected[i]);
            float threshold = atol + rtol * std::abs(expected[i]);
            if (diff > threshold)
                totalErrors++;
        }
        printf("Total errors: %d / %lld\n", totalErrors, (long long)actual.getCount());
        return false;
    }
    
    return true;
}

// Debug helper: print all tensor names from a SafeTensors file
static void printAllTensorNames(SafeTensorsReader& reader, const char* label)
{
    List<String> names;
    reader.getTensorNames(names);
    
    printf("\n=== Tensor names in %s (%lld tensors) ===\n", label, (long long)names.getCount());
    for (const auto& name : names)
    {
        const SafeTensorInfo* info = reader.getTensorInfo(name.getUnownedSlice());
        if (info)
        {
            printf("  %s: shape=[", name.getBuffer());
            for (int i = 0; i < info->shape.getRank(); i++)
            {
                if (i > 0) printf(",");
                printf("%d", info->shape[i]);
            }
            printf("]\n");
        }
    }
    printf("=== End tensor names ===\n\n");
}

// Debug helper: check if a weight exists and print warning if not
static bool checkWeightExists(SafeTensorsReader& reader, const String& name, const char* context)
{
    if (!reader.hasTensor(name.getUnownedSlice()))
    {
        printf("WARNING: Weight not found: '%s' (context: %s)\n", name.getBuffer(), context);
        return false;
    }
    return true;
}

// Verify all expected CLIP weights exist in the file
static void verifyClipWeights(SafeTensorsReader& reader, const String& prefix)
{
    printf("\nVerifying CLIP weights with prefix '%s'...\n", prefix.getBuffer());
    
    List<String> expectedWeights;
    
    // Embeddings
    expectedWeights.add(prefix + "embeddings.token_embedding.weight");
    expectedWeights.add(prefix + "embeddings.position_embedding.weight");
    
    // 12 transformer layers
    for (int i = 0; i < 12; i++)
    {
        String layerPrefix = prefix + "encoder.layers." + String(i) + ".";
        
        // Layer norms
        expectedWeights.add(layerPrefix + "layer_norm1.weight");
        expectedWeights.add(layerPrefix + "layer_norm1.bias");
        expectedWeights.add(layerPrefix + "layer_norm2.weight");
        expectedWeights.add(layerPrefix + "layer_norm2.bias");
        
        // Self attention
        expectedWeights.add(layerPrefix + "self_attn.q_proj.weight");
        expectedWeights.add(layerPrefix + "self_attn.q_proj.bias");
        expectedWeights.add(layerPrefix + "self_attn.k_proj.weight");
        expectedWeights.add(layerPrefix + "self_attn.k_proj.bias");
        expectedWeights.add(layerPrefix + "self_attn.v_proj.weight");
        expectedWeights.add(layerPrefix + "self_attn.v_proj.bias");
        expectedWeights.add(layerPrefix + "self_attn.out_proj.weight");
        expectedWeights.add(layerPrefix + "self_attn.out_proj.bias");
        
        // MLP
        expectedWeights.add(layerPrefix + "mlp.fc1.weight");
        expectedWeights.add(layerPrefix + "mlp.fc1.bias");
        expectedWeights.add(layerPrefix + "mlp.fc2.weight");
        expectedWeights.add(layerPrefix + "mlp.fc2.bias");
    }
    
    // Final layer norm
    expectedWeights.add(prefix + "final_layer_norm.weight");
    expectedWeights.add(prefix + "final_layer_norm.bias");
    
    int missingCount = 0;
    for (const auto& name : expectedWeights)
    {
        if (!reader.hasTensor(name.getUnownedSlice()))
        {
            printf("  MISSING: %s\n", name.getBuffer());
            missingCount++;
        }
    }
    
    if (missingCount == 0)
    {
        printf("  All %lld expected weights found.\n", (long long)expectedWeights.getCount());
    }
    else
    {
        printf("  %d/%lld weights missing!\n", missingCount, (long long)expectedWeights.getCount());
    }
}

SlangResult testCLIPEncoderSD15(InferencingContext* ctx)
{
    printf("testCLIPEncoderSD15: Loading CLIP text encoder...\n");
    
    // Load CLIP model weights
    String modelPath = getTestFilePath("test_data/clip_text_model.safetensors");
    
    if (!File::exists(modelPath))
    {
        printf("  CLIP model not found at %s\n", modelPath.getBuffer());
        printf("  Run clip-encoder-test-generate.py to download and prepare the model.\n");
        printf("  testCLIPEncoderSD15: SKIPPED\n");
        return SLANG_OK;
    }
    
    SafeTensorsReader reader;
    if (SLANG_FAILED(reader.load(modelPath)))
    {
        printf("  Failed to load CLIP model from %s\n", modelPath.getBuffer());
        return SLANG_FAIL;
    }
    
    // Debug: print all tensor names
    printAllTensorNames(reader, "clip_text_model.safetensors");
    
    // Determine the weight prefix by checking first tensor name
    List<String> tensorNames;
    reader.getTensorNames(tensorNames);
    String clipPrefix = "";
    if (tensorNames.getCount() > 0 && tensorNames[0].startsWith("text_model."))
    {
        clipPrefix = "text_model.";
        printf("  Using weight prefix: 'text_model.'\n");
    }
    else
    {
        printf("  Using weight prefix: '' (empty)\n");
    }
    
    // Verify expected weights exist
    verifyClipWeights(reader, clipPrefix);
    
    // Create encoder
    printf("  Creating CLIP encoder...\n");
    CLIPTextEncoder encoder(ctx);
    
    // Load parameters
    printf("  Loading parameters...\n");
    if (SLANG_FAILED(encoder.loadParams(reader, clipPrefix)))
    {
        printf("  Failed to load CLIP parameters\n");
        printf("  testCLIPEncoderSD15: FAILED\n");
        return SLANG_FAIL;
    }
    
    // Load test input
    String inputPath = getTestFilePath("test_data/clip_input_tokens.bin");
    List<float> inputTokens;
    if (SLANG_FAILED(loadBinaryTensor(inputPath, inputTokens)))
    {
        printf("  No test input found at %s\n", inputPath.getBuffer());
        printf("  Run clip-encoder-test-generate.py to generate test data.\n");
        printf("  testCLIPEncoderSD15: SKIPPED\n");
        return SLANG_OK;
    }
    
    int batchSize = 1;
    int seqLen = 77;
    
    if (inputTokens.getCount() != (Index)(batchSize * seqLen))
    {
        printf("  Input size mismatch: got %lld, expected %d\n",
            (long long)inputTokens.getCount(), batchSize * seqLen);
        return SLANG_FAIL;
    }
    
    // Create input tensor
    auto inputTensor = ctx->createTensor(
        ElementType::Float32, Shape(batchSize, seqLen), inputTokens, "InputTokens");
    
    // Allocate output
    auto outputTensor = encoder.allocateResultBuffer(ElementType::Float32, seqLen, batchSize);
    
    // Execute
    printf("  Running encoder...\n");
    auto task = ctx->createTask();
    encoder.queueExecute(task, outputTensor, inputTensor->getView(), seqLen, batchSize);
    task.execute();
    
    // Read output
    auto output = ctx->readBuffer<float>(outputTensor);
    
    // Verify shape
    int expectedSize = batchSize * seqLen * encoder.hiddenSize;
    if (output.getCount() != expectedSize)
    {
        printf("  Output size mismatch: got %lld, expected %d\n",
            (long long)output.getCount(), expectedSize);
        return SLANG_FAIL;
    }
    
    // Check against reference if available
    String refPath = getTestFilePath("test_data/clip_output.bin");
    List<float> expectedOutput;
    if (SLANG_SUCCEEDED(loadBinaryTensor(refPath, expectedOutput)))
    {
        if (!checkApproxEqual(output, expectedOutput, 1e-2f, 1e-4f))
        {
            printf("testCLIPEncoderSD15: FAILED (output mismatch)\n");
            return SLANG_FAIL;
        }
    }
    else
    {
        printf("  No reference output found, skipping numerical validation\n");
        
        // Print first few output values for inspection
        printf("  First 8 output values: ");
        for (int i = 0; i < 8 && i < output.getCount(); i++)
        {
            printf("%.4f ", output[i]);
        }
        printf("\n");
    }
    
    printf("testCLIPEncoderSD15: PASSED\n");
    return SLANG_OK;
}


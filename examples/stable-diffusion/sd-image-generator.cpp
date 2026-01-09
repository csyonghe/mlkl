#include "sd-image-generator.h"

#include "concat.h"
#include "safetensors-reader.h"

#include <chrono>
#include <cmath>
#include <cstdlib>

// ============================================================================
// Constructor
// ============================================================================

SDImageGenerator::SDImageGenerator(InferencingContext* context)
    : ctx(context)
{
    // Initialize tokenizer
    tokenizer = new CLIPTokenizer();

    // Initialize model components (this compiles shaders)
    clipEncoder = new CLIPTextEncoder(ctx);
    unet = new SDUNet(ctx);
    vaeDecoder = new VAEDecoder(ctx);

    // Initialize kernels
    initializeKernels();

    // Initialize DDIM sampler
    // SD 1.5 uses: 1000 train steps, beta_start=0.00085, beta_end=0.012, scaled_linear schedule
    DiffusionSchedule schedule(kTrainSteps, 0.00085f, 0.012f, /*scaled_linear=*/true);
    sampler = new DDIMSampler(ctx, std::move(schedule), kDefaultInferenceSteps);

    // Allocate persistent buffers
    allocateBuffers();
}

// ============================================================================
// Kernel and Buffer Initialization
// ============================================================================

void SDImageGenerator::initializeKernels()
{
    // VAE input scaling kernel: output = input / 0.18215
    const float vaeScalingFactor = 0.18215f;
    vaeScaleKernel = new ElementwiseKernel(ctx, buffer() * constant(1.0f / vaeScalingFactor));

    // Duplicate kernel: [1, H, W, C] → [2, H, W, C] (duplicates batch dim for CFG)
    latentDuplicateKernel = new ElementwiseKernel(ctx, duplicate(buffer(), 0, 2));
    
    // Note: CFG combination is now fused into DDIMSampler::stepWithCFG()
}

void SDImageGenerator::allocateBuffers()
{
    const int latentSize = 1 * kLatentHeight * kLatentWidth * kLatentChannels;

    // Note: tokenIds and latent are created in generateImage() with actual data

    // Latent buffer for DDIM sampling (second buffer for double buffering)
    latentNext = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kLatentHeight, kLatentWidth, kLatentChannels},
        latentSize * sizeof(float),
        nullptr,
        "LatentNext");

    // Noise prediction buffer for batched CFG [2, H, W, C] = [uncond, cond]
    noisePredBatched = ctx->createTensor(
        ElementType::Float32,
        Shape{2, kLatentHeight, kLatentWidth, kLatentChannels},
        2 * latentSize * sizeof(float),
        nullptr,
        "NoisePredBatched");

    // Batched latent for CFG (holds [uncond_latent, cond_latent])
    latentBatched = ctx->createTensor(
        ElementType::Float32,
        Shape{2, kLatentHeight, kLatentWidth, kLatentChannels},
        2 * latentSize * sizeof(float),
        nullptr,
        "LatentBatched");

    // Scaled latent for VAE input
    scaledLatent = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kLatentHeight, kLatentWidth, kLatentChannels},
        latentSize * sizeof(float),
        nullptr,
        "ScaledLatent");

    // Output image buffer
    outputImage = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kImageHeight, kImageWidth, 3},
        1 * kImageHeight * kImageWidth * 3 * sizeof(float),
        nullptr,
        "OutputImage");
}

// ============================================================================
// Model Loading
// ============================================================================

SlangResult SDImageGenerator::loadModels(const String& modelsDir)
{
    // Ensure trailing separator
    String dir = modelsDir;
    if (!dir.endsWith("/") && !dir.endsWith("\\"))
        dir = dir + "/";

    // Load tokenizer vocabulary and merges
    String vocabPath = dir + "vocab.json";
    String mergesPath = dir + "merges.txt";
    SLANG_RETURN_ON_FAIL(tokenizer->load(vocabPath, mergesPath));

    // Load CLIP text encoder
    {
        String clipPath = dir + "clip.safetensors";
        SafeTensorsReader reader;
        SLANG_RETURN_ON_FAIL(reader.load(clipPath));
        
        // Auto-detect weight prefix
        List<String> tensorNames;
        reader.getTensorNames(tensorNames);
        String clipPrefix = "";
        if (tensorNames.getCount() > 0 && tensorNames[0].startsWith("text_model."))
        {
            clipPrefix = "text_model.";
        }
        SLANG_RETURN_ON_FAIL(clipEncoder->loadParams(reader, clipPrefix));
    }

    // Load UNet
    {
        String unetPath = dir + "unet.safetensors";
        SafeTensorsReader reader;
        SLANG_RETURN_ON_FAIL(reader.load(unetPath));
        SLANG_RETURN_ON_FAIL(unet->loadParams(reader));
    }

    // Load VAE decoder
    {
        String vaePath = dir + "vae.safetensors";
        SafeTensorsReader reader;
        SLANG_RETURN_ON_FAIL(reader.load(vaePath));
        SLANG_RETURN_ON_FAIL(vaeDecoder->loadParams(reader));
    }

    modelsLoaded = true;
    return SLANG_OK;
}

// ============================================================================
// Random Latent Generation
// ============================================================================

void SDImageGenerator::fillRandomLatent(float* data, int count, uint32_t seed)
{
    // Simple Box-Muller transform for Gaussian noise
    srand(seed);
    for (int i = 0; i < count; i += 2)
    {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 2);
        float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 2);
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * 3.14159265358979f * u2;
        data[i] = r * std::cos(theta);
        if (i + 1 < count)
            data[i + 1] = r * std::sin(theta);
    }
}

// ============================================================================
// Image Generation
// ============================================================================

TensorView SDImageGenerator::generateImage(
    const String& prompt,
    uint32_t seed,
    int inferenceSteps,
    float guidanceScale)
{
    if (!modelsLoaded)
    {
        return TensorView();
    }

    // Recreate sampler if inference steps changed
    if (inferenceSteps != sampler->inference_steps)
    {
        DiffusionSchedule schedule(kTrainSteps, 0.00085f, 0.012f, /*scaled_linear=*/true);
        sampler = new DDIMSampler(ctx, std::move(schedule), inferenceSteps);
    }

    bool useCFG = guidanceScale > 1.0f;

    // Performance timing
    using Clock = std::chrono::high_resolution_clock;
    auto totalStart = Clock::now();
    double clipTimeMs = 0.0, diffusionTimeMs = 0.0, vaeTimeMs = 0.0;

    // ========================================================================
    // Step 1: Tokenize prompt (and empty prompt for CFG)
    // ========================================================================
    List<int> tokens = tokenizer->encode(prompt, kMaxTokenLength);

    // Create conditional token IDs tensor (CLIP expects Float32)
    List<float> tokenData;
    tokenData.setCount(kMaxTokenLength);
    for (int i = 0; i < kMaxTokenLength; i++)
    {
        tokenData[i] = (float)((i < (int)tokens.getCount()) ? tokens[i] : CLIPTokenizer::kEndOfText);
    }
    tokenIds = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kMaxTokenLength},
        tokenData,
        "TokenIds");

    // For CFG, also create unconditional (empty prompt) tokens
    RefPtr<Tensor> uncondTokenIds;
    if (useCFG)
    {
        List<int> uncondTokens = tokenizer->encode("", kMaxTokenLength);
        List<float> uncondTokenData;
        uncondTokenData.setCount(kMaxTokenLength);
        for (int i = 0; i < kMaxTokenLength; i++)
        {
            uncondTokenData[i] =
                (float)((i < (int)uncondTokens.getCount()) ? uncondTokens[i] : CLIPTokenizer::kEndOfText);
        }
        uncondTokenIds = ctx->createTensor(
            ElementType::Float32,
            Shape{1, kMaxTokenLength},
            uncondTokenData,
            "UncondTokenIds");
    }

    // ========================================================================
    // Step 2: CLIP text encoding
    // ========================================================================
    auto clipStart = Clock::now();
    
    // For CFG, we create batched embeddings [2, 77, 768] = [uncond, cond]
    // For non-CFG, just [1, 77, 768]
    TensorView textEmbeddings;
    RefPtr<Tensor> textEmbeddingsTensor;  // Keep tensor alive for the function duration

    if (useCFG)
    {
        // Encode unconditional (empty prompt)
        TensorView uncondEmbed = clipEncoder->allocateResultBuffer(ElementType::Float32, kMaxTokenLength, 1);
        {
            auto task = ctx->createTask();
            clipEncoder->queueExecute(task, uncondEmbed, uncondTokenIds->getView(), kMaxTokenLength, 1);
            task.execute();
        }

        // Encode conditional (user prompt)
        TensorView condEmbed = clipEncoder->allocateResultBuffer(ElementType::Float32, kMaxTokenLength, 1);
        {
            auto task = ctx->createTask();
            clipEncoder->queueExecute(task, condEmbed, tokenIds->getView(), kMaxTokenLength, 1);
            task.execute();
        }

        // Concatenate along batch dimension: [uncond, cond] → [2, 77, 768]
        textEmbeddingsTensor = ctx->createTensor(
            ElementType::Float32,
            Shape{2, kMaxTokenLength, 768},
            2 * kMaxTokenLength * 768 * sizeof(float),
            nullptr,
            "BatchedTextEmbeddings");
        {
            auto task = ctx->createTask();
            ConcatKernel concatKernel(ctx, 2); // 2 operands
            TensorView inputs[] = {uncondEmbed, condEmbed};
            concatKernel.queueExecute(task, textEmbeddingsTensor->getView(), makeArrayView(inputs), 0); // axis 0
            task.execute();
        }
        textEmbeddings = textEmbeddingsTensor->getView();
    }
    else
    {
        textEmbeddings = clipEncoder->allocateResultBuffer(ElementType::Float32, kMaxTokenLength, 1);
        auto task = ctx->createTask();
        clipEncoder->queueExecute(task, textEmbeddings, tokenIds->getView(), kMaxTokenLength, 1);
        task.execute();
    }
    
    clipTimeMs = std::chrono::duration<double, std::milli>(Clock::now() - clipStart).count();

    // ========================================================================
    // Step 3: Initialize random latent
    // ========================================================================
    const int latentSize = 1 * kLatentHeight * kLatentWidth * kLatentChannels;
    List<float> latentData;
    latentData.setCount(latentSize);
    fillRandomLatent(latentData.getBuffer(), latentSize, seed);
    latent = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kLatentHeight, kLatentWidth, kLatentChannels},
        latentData,
        "Latent");

    // ========================================================================
    // Step 4: DDIM sampling loop with CFG
    // ========================================================================
    auto diffusionStart = Clock::now();
    
    // Use RefPtr for swapping
    RefPtr<Tensor> currentLatent = latent;
    RefPtr<Tensor> nextLatent = latentNext;

    for (int i = 0; i < inferenceSteps; i++)
    {
        int t = sampler->timesteps[i];

        if (useCFG)
        {
            // Duplicate latent to batch dimension: [1, H, W, C] → [2, H, W, C]
            {
                auto task = ctx->createTask();
                latentDuplicateKernel->queueExecute(task, latentBatched->getView(), currentLatent->getView());
                task.execute();
            }

            // Run UNet once with batched inputs: [2, H, W, C] latent, [2, 77, 768] context
            {
                auto task = ctx->createTask();
                unet->queueExecute(
                    task, noisePredBatched->getView(), latentBatched->getView(), t, textEmbeddings);
                task.execute();
            }

            // Fused CFG + DDIM step: applies CFG combination and DDIM update in one kernel
            {
                auto task = ctx->createTask();
                sampler->stepWithCFG(
                    task, nextLatent->getView(), currentLatent->getView(),
                    noisePredBatched->getView(), guidanceScale, i);
                task.execute();
            }
        }
        else
        {
            // No CFG - single UNet pass
            TensorView noisePred = noisePredBatched->getView();
            noisePred.shape = Shape{1, kLatentHeight, kLatentWidth, kLatentChannels};
            {
                auto task = ctx->createTask();
                unet->queueExecute(task, noisePred, currentLatent->getView(), t, textEmbeddings);
                task.execute();
            }

            // DDIM update step
            {
                auto task = ctx->createTask();
                sampler->step(task, nextLatent->getView(), currentLatent->getView(), noisePred, i);
                task.execute();
            }
        }

        // Swap latent buffers
        std::swap(currentLatent, nextLatent);
    }
    
    diffusionTimeMs = std::chrono::duration<double, std::milli>(Clock::now() - diffusionStart).count();

    // ========================================================================
    // Step 5: Scale latent for VAE
    // ========================================================================
    auto vaeStart = Clock::now();
    
    {
        auto task = ctx->createTask();
        vaeScaleKernel->queueExecute(task, scaledLatent->getView(), currentLatent->getView());
        task.execute();
    }

    // ========================================================================
    // Step 6: VAE decode
    // ========================================================================
    {
        auto task = ctx->createTask();
        vaeDecoder->queueExecute(task, outputImage->getView(), scaledLatent->getView());
        task.execute();
    }
    
    vaeTimeMs = std::chrono::duration<double, std::milli>(Clock::now() - vaeStart).count();
    
    // Store performance stats
    lastPerfStats.clipTimeMs = clipTimeMs;
    lastPerfStats.diffusionTimeMs = diffusionTimeMs;
    lastPerfStats.vaeTimeMs = vaeTimeMs;
    lastPerfStats.totalTimeMs = std::chrono::duration<double, std::milli>(Clock::now() - totalStart).count();
    lastPerfStats.inferenceSteps = inferenceSteps;

    return outputImage->getView();
}

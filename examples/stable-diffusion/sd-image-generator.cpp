#include "sd-image-generator.h"

#include "safetensors-reader.h"

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

    // CFG combination kernel: output = uncond + scale * (cond - uncond)
    // Store expressions so we can map them to inputs later
    cfgUncondExpr = buffer();
    cfgCondExpr = buffer();
    cfgScaleExpr = uniformConstant();
    cfgKernel = new ElementwiseKernel(ctx, cfgUncondExpr + cfgScaleExpr * (cfgCondExpr - cfgUncondExpr));
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

    // Noise prediction buffers (for CFG we need conditional, unconditional, and combined)
    noisePredCond = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kLatentHeight, kLatentWidth, kLatentChannels},
        latentSize * sizeof(float),
        nullptr,
        "NoisePredCond");

    noisePredUncond = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kLatentHeight, kLatentWidth, kLatentChannels},
        latentSize * sizeof(float),
        nullptr,
        "NoisePredUncond");

    noisePredCombined = ctx->createTensor(
        ElementType::Float32,
        Shape{1, kLatentHeight, kLatentWidth, kLatentChannels},
        latentSize * sizeof(float),
        nullptr,
        "NoisePredCombined");

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
        SLANG_RETURN_ON_FAIL(clipEncoder->loadParams(reader));
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
    TensorView condEmbeddings = clipEncoder->allocateResultBuffer(ElementType::Float32, kMaxTokenLength, 1);
    {
        auto task = ctx->createTask();
        clipEncoder->queueExecute(task, condEmbeddings, tokenIds->getView(), kMaxTokenLength, 1);
        task.execute();
    }

    TensorView uncondEmbeddings;
    if (useCFG)
    {
        uncondEmbeddings = clipEncoder->allocateResultBuffer(ElementType::Float32, kMaxTokenLength, 1);
        auto task = ctx->createTask();
        clipEncoder->queueExecute(task, uncondEmbeddings, uncondTokenIds->getView(), kMaxTokenLength, 1);
        task.execute();
    }

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
    // Use RefPtr for swapping
    RefPtr<Tensor> currentLatent = latent;
    RefPtr<Tensor> nextLatent = latentNext;

    for (int i = 0; i < inferenceSteps; i++)
    {
        int t = sampler->timesteps[i];

        TensorView noisePredForDDIM;

        if (useCFG)
        {
            // Run UNet twice: unconditional and conditional
            {
                auto task = ctx->createTask();
                unet->queueExecute(
                    task, noisePredUncond->getView(), currentLatent->getView(), t, uncondEmbeddings);
                task.execute();
            }
            {
                auto task = ctx->createTask();
                unet->queueExecute(
                    task, noisePredCond->getView(), currentLatent->getView(), t, condEmbeddings);
                task.execute();
            }

            // Combine: output = uncond + scale * (cond - uncond)
            {
                auto task = ctx->createTask();
                Dictionary<Expr, InputInfo> cfgInputs;
                cfgInputs.add(cfgUncondExpr, noisePredUncond->getView());
                cfgInputs.add(cfgCondExpr, noisePredCond->getView());
                cfgInputs.add(cfgScaleExpr, InputInfo(guidanceScale));
                cfgKernel->queueExecute(task, noisePredCombined->getView(), cfgInputs);
                task.execute();
            }

            noisePredForDDIM = noisePredCombined->getView();
        }
        else
        {
            // No CFG - single UNet pass
            auto task = ctx->createTask();
            unet->queueExecute(task, noisePredCond->getView(), currentLatent->getView(), t, condEmbeddings);
            task.execute();
            noisePredForDDIM = noisePredCond->getView();
        }

        // DDIM update step
        {
            auto task = ctx->createTask();
            sampler->step(task, nextLatent->getView(), currentLatent->getView(), noisePredForDDIM, i);
            task.execute();
        }

        // Swap latent buffers
        std::swap(currentLatent, nextLatent);
    }

    // ========================================================================
    // Step 5: Scale latent for VAE
    // ========================================================================
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

    return outputImage->getView();
}

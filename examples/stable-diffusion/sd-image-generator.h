#pragma once

#include "clip-encoder.h"
#include "core/slang-basic.h"
#include "examples/shared/ddim-sampler.h"
#include "inference-context.h"
#include "tokenizer.h"
#include "unet.h"
#include "vae-decoder.h"

using namespace Slang;

// ============================================================================
// SDImageGenerator - Complete Stable Diffusion 1.5 Image Generator
// ============================================================================
// 
// Usage:
//   RefPtr<InferencingContext> ctx = new InferencingContext();
//   RefPtr<SDImageGenerator> generator = new SDImageGenerator(ctx);
//   generator->loadModels("models/");  // Load all model weights
//   
//   // Generate with CFG (guidance_scale=7.5 for best quality)
//   TensorView imageView = generator->generateImage("a beautiful sunset", 42, 20, 7.5f);
//   auto imageData = ctx->readBuffer<float>(imageView);
//   writeImagePNG("output.png", 512, 512, 3, imageData);
//
class SDImageGenerator : public RefObject
{
public:
    // Configuration
    static const int kImageWidth = 512;
    static const int kImageHeight = 512;
    static const int kLatentWidth = 64;   // 512 / 8
    static const int kLatentHeight = 64;  // 512 / 8
    static const int kLatentChannels = 4;
    static const int kMaxTokenLength = 77;
    static const int kTrainSteps = 1000;
    static const int kDefaultInferenceSteps = 20;

private:
    InferencingContext* ctx;

    // Components
    RefPtr<CLIPTokenizer> tokenizer;
    RefPtr<CLIPTextEncoder> clipEncoder;
    RefPtr<SDUNet> unet;
    RefPtr<VAEDecoder> vaeDecoder;

    // Scaling kernel for VAE input
    RefPtr<ElementwiseKernel> vaeScaleKernel;

    // CFG combination kernel: output = uncond + scale * (cond - uncond)
    // Takes sliced views of batched noise prediction
    RefPtr<ElementwiseKernel> cfgKernel;
    Expr cfgUncondExpr;
    Expr cfgCondExpr;
    Expr cfgScaleExpr;

    // Latent duplicate kernel: duplicates [1,H,W,C] → [2,H,W,C] for batched CFG
    RefPtr<ElementwiseKernel> latentDuplicateKernel;

    // DDIM sampler
    RefPtr<DDIMSampler> sampler;

    // Pre-allocated persistent buffers
    RefPtr<Tensor> tokenIds;
    RefPtr<Tensor> latent;
    RefPtr<Tensor> latentNext;
    RefPtr<Tensor> noisePredBatched;   // [2, H, W, C] for batched CFG
    RefPtr<Tensor> noisePredCombined;  // [1, H, W, C] Combined noise after CFG
    RefPtr<Tensor> latentBatched;      // [2, H, W, C] duplicated latent for batched UNet
    RefPtr<Tensor> scaledLatent;
    RefPtr<Tensor> outputImage;

    bool modelsLoaded = false;

public:
    SDImageGenerator(InferencingContext* context);

    // Load all model weights from a directory
    // Expected files: vocab.json, merges.txt, clip.safetensors, unet.safetensors, vae.safetensors
    SlangResult loadModels(const String& modelsDir);

    // Generate an image from a text prompt
    // Returns TensorView with RGB float data in range [-1, 1] with shape [1, height, width, 3]
    // guidanceScale: 1.0 = no CFG, 7.5 = typical for good results
    TensorView generateImage(
        const String& prompt,
        uint32_t seed,
        int inferenceSteps = kDefaultInferenceSteps,
        float guidanceScale = 7.5f);

    // Get image dimensions
    int getImageWidth() const { return kImageWidth; }
    int getImageHeight() const { return kImageHeight; }

private:
    void initializeKernels();
    void allocateBuffers();
    void fillRandomLatent(float* data, int count, uint32_t seed);
};

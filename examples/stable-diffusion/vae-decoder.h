#pragma once

#include "inference-context.h"
#include "kernels.h"
#include "safetensors-reader.h"

using namespace Slang;

// ============================================================================
// SD 1.5 VAE Decoder Architecture
// ============================================================================
// Input:  [B, 4, H/8, W/8] latent (e.g., [1, 4, 64, 64] for 512x512 output)
// Output: [B, 3, H, W] RGB image (e.g., [1, 3, 512, 512])
//
// Structure:
// 1. post_quant_conv: Conv2D 4 -> 4
// 2. decoder.conv_in: Conv2D 4 -> 512
// 3. decoder.mid_block:
//    - resnets[0]: ResNetBlock 512 -> 512
//    - attentions[0]: Self-Attention 512
//    - resnets[1]: ResNetBlock 512 -> 512
// 4. decoder.up_blocks[0..3]: Each contains 3 ResNetBlocks + optional upsample
//    - up_blocks[0]: 512 -> 512, upsample (64 -> 128)
//    - up_blocks[1]: 512 -> 512, upsample (128 -> 256)
//    - up_blocks[2]: 512 -> 256, upsample (256 -> 512)
//    - up_blocks[3]: 256 -> 128, no upsample
// 5. decoder.conv_norm_out: GroupNorm(32, 128)
// 6. decoder.conv_out: Conv2D 128 -> 3
// ============================================================================

// ResNet block used in VAE decoder
class VAEResNetBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    // Layers (SiLU is fused into conv input expressions)
    RefPtr<GroupNormKernel> norm1;
    RefPtr<Conv2DKernel> conv1;  // Has SiLU fused in input
    RefPtr<GroupNormKernel> norm2;
    RefPtr<Conv2DKernel> conv2;  // Has SiLU fused in input
    
    // Skip connection (if in_channels != out_channels)
    RefPtr<Conv2DKernel> skipConv;
    
    // Residual addition
    RefPtr<ElementwiseKernel> residualAdd;
    
    int inChannels;
    int outChannels;
    
public:
    VAEResNetBlock(
        RefPtr<InferencingContext> ctx,
        int inChannels,
        int outChannels,
        int numGroups = 32);
    
    SlangResult loadParams(
        SafeTensorsReader& reader,
        const String& prefix);
    
    TensorView allocateResultBuffer(
        ElementType elementType,
        int height,
        int width,
        int batchSize);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input);
};

// Self-attention block for VAE mid-block
// Uses separate Q, K, V linear projections + standard attention (BatchGemm + Softmax)
// Note: FlashAttention can't handle headDim=512 (SD VAE uses single-head with 512 channels)
class VAEAttentionBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<GroupNormKernel> groupNorm;
    
    // Q, K, V projections (all from same input for self-attention)
    RefPtr<LinearKernel> projQ;
    RefPtr<LinearKernel> projK;
    RefPtr<LinearKernel> projV;
    RefPtr<LinearKernel> projOut;
    
    // Standard attention: Q @ K^T -> softmax -> @ V
    // Q @ K^T with fused transpose on K
    RefPtr<BatchGemmKernel> qkGemm;
    Expr qExpr, kExpr;  // Input expressions for Q and K (K is transposed internally)
    
    RefPtr<SoftmaxKernel> softmax;
    
    // attn_weights @ V
    RefPtr<BatchGemmKernel> attnVGemm;
    Expr attnExpr, vExpr;  // Input expressions for attention weights and V
    
    // Residual connection
    RefPtr<ElementwiseKernel> residualAdd;
    
    int channels;
    
public:
    VAEAttentionBlock(
        RefPtr<InferencingContext> ctx,
        int channels,
        int numHeads = 1);
    
    SlangResult loadParams(
        SafeTensorsReader& reader,
        const String& prefix);
    
    TensorView allocateResultBuffer(
        ElementType elementType,
        int height,
        int width,
        int batchSize);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input);
};

// Up-sampling block (3 ResNet blocks + optional upsample)
class VAEUpBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    List<RefPtr<VAEResNetBlock>> resnets;
    
    // Upsampling (upsample2x fused into conv input expression)
    RefPtr<Conv2DKernel> upsampleConv;  // 3x3 conv with fused 2x upsample
    
    bool hasUpsample;
    int outChannels;
    
public:
    VAEUpBlock(
        RefPtr<InferencingContext> ctx,
        int inChannels,
        int outChannels,
        bool hasUpsample,
        int numResBlocks = 3);
    
    SlangResult loadParams(
        SafeTensorsReader& reader,
        const String& prefix);
    
    TensorView allocateResultBuffer(
        ElementType elementType,
        int height,
        int width,
        int batchSize);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input);
};

// Complete VAE Decoder
class VAEDecoder : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    // Post-quantization conv (before decoder)
    RefPtr<Conv2DKernel> postQuantConv;
    
    // Decoder input conv
    RefPtr<Conv2DKernel> convIn;
    
    // Mid block
    RefPtr<VAEResNetBlock> midResnet1;
    RefPtr<VAEAttentionBlock> midAttn;
    RefPtr<VAEResNetBlock> midResnet2;
    
    // Up blocks (processed in reverse order: 0 first, then 1, 2, 3)
    List<RefPtr<VAEUpBlock>> upBlocks;
    
    // Output layers (SiLU fused into convOut input, clamp fused into output)
    RefPtr<GroupNormKernel> normOut;
    RefPtr<Conv2DKernel> convOut;
    
    // Configuration
    int latentChannels = 4;
    int outputChannels = 3;
    
public:
    VAEDecoder(RefPtr<InferencingContext> ctx);
    
    // Load weights from SafeTensors file
    SlangResult loadParams(SafeTensorsReader& reader);
    
    // Decode latent to image
    // Input: [B, 4, H, W] latent in NCHW (will be converted to NHWC internally)
    // Output: [B, H*8, W*8, 3] RGB image in NHWC
    TensorView allocateResultBuffer(
        ElementType elementType,
        int latentHeight,
        int latentWidth,
        int batchSize);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView latentInput);
};


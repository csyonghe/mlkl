#pragma once

#include "inference-context.h"
#include "kernels.h"
#include "safetensors-reader.h"

// ============================================================================
// Stable Diffusion 1.5 UNet
// ============================================================================
// Architecture:
// - Input: [B, H, W, 4] latent + timestep + [B, 77, 768] text conditioning
// - Output: [B, H, W, 4] noise prediction
//
// Structure:
// 1. Time embedding: sinusoidal → Linear → SiLU → Linear (320 → 1280)
// 2. Input conv: 4 → 320
// 3. Down blocks (4 stages): ResNet + Transformer (with cross-attention)
// 4. Mid block: ResNet + Transformer + ResNet
// 5. Up blocks (4 stages): ResNet + Transformer (with skip connections)
// 6. Output: GroupNorm → SiLU → Conv (320 → 4)
//
// Channel progression: 320 → 640 → 1280 → 1280
// Attention at: 32x32 and 16x16 (not at 64x64)
// ============================================================================

// Forward declarations
class SDResNetBlock;
class SDTransformerBlock;
class SDDownBlock;
class SDMidBlock;
class SDUpBlock;

// ============================================================================
// SD ResNet Block
// ============================================================================
// Structure: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + residual
// Time embedding is added after first conv via projection
//
// For up blocks, can optionally fuse a concat operation into norm1's inputExpr.
// When fuseConcatInput=true, queueExecute takes two inputs (current, skip) that
// are concatenated along the channel dimension before normalization.
class SDResNetBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<GroupNormKernel> norm1;
    RefPtr<Conv2DKernel> conv1;
    RefPtr<LinearKernel> timeProj;  // Projects time embedding to channel dim
    RefPtr<GroupNormKernel> norm2;
    RefPtr<Conv2DKernel> conv2;
    RefPtr<Conv2DKernel> residualConv;  // Only if inChannels != outChannels
    
    int inChannels;
    int outChannels;
    int timeEmbedDim;
    bool hasResidualConv;
    bool hasFusedConcat;  // True if norm1 has fused concat inputExpr
    
    // Expressions for fused concat (stored to map inputs at runtime)
    Expr norm1Buf0, norm1Buf1, norm1ConcatAxis;
    Expr resBuf0, resBuf1, residualConvConcatAxis;
    
public:
    // Standard constructor (no concat fusion)
    SDResNetBlock(
        RefPtr<InferencingContext> ctx,
        int inChannels,
        int outChannels,
        int timeEmbedDim = 1280);
    
    // Constructor with optional concat fusion for up blocks
    // When fuseConcatInput=true, inChannels should be currentChannels + skipChannels
    SDResNetBlock(
        RefPtr<InferencingContext> ctx,
        int inChannels,
        int outChannels,
        int timeEmbedDim,
        bool fuseConcatInput);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    // Standard execution (single input)
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        TensorView timeEmbed);  // [B, timeEmbedDim]
    
    // Execution with fused concat (two inputs concatenated along channel dim)
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView current,
        TensorView skip,
        TensorView timeEmbed);
};

// ============================================================================
// SD Cross-Attention Block
// ============================================================================
// Used in transformer blocks for text conditioning
// Q from image features, K/V from text embeddings
class SDCrossAttention : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<LinearKernel> toQ;
    RefPtr<LinearKernel> toK;
    RefPtr<LinearKernel> toV;
    RefPtr<LinearKernel> toOut;
    
    RefPtr<FlashAttentionKernel> flashAttn;
    
    Expr qExpr, kExpr, vExpr;
    
    int queryDim;
    int contextDim;  // Text embedding dim (768 for CLIP)
    int numHeads;
    int headDim;
    
public:
    SDCrossAttention(
        RefPtr<InferencingContext> ctx,
        int queryDim,
        int contextDim,
        int numHeads);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView query,      // [B, seqLen, queryDim]
        TensorView context,    // [B, contextSeqLen, contextDim] - text embeddings
        TensorView residual);  // [B, seqLen, queryDim] - adds residual to output
};

// ============================================================================
// SD Self-Attention Block
// ============================================================================
// Same as cross-attention but Q, K, V all from same input
class SDSelfAttention : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<LinearKernel> toQ;
    RefPtr<LinearKernel> toK;
    RefPtr<LinearKernel> toV;
    RefPtr<LinearKernel> toOut;
    
    RefPtr<FlashAttentionKernel> flashAttn;
    
    Expr qExpr, kExpr, vExpr;
    
    int dim;
    int numHeads;
    int headDim;
    
public:
    SDSelfAttention(
        RefPtr<InferencingContext> ctx,
        int dim,
        int numHeads);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        TensorView residual);  // [B, seqLen, dim] - adds residual to output
};

// ============================================================================
// SD Feed-Forward Block
// ============================================================================
// Structure: Linear → GEGLU → Linear
class SDFeedForward : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<LinearKernel> proj1;  // dim → innerDim * 2 (for GEGLU gate)
    RefPtr<LinearKernel> proj2;  // innerDim → dim
    
    int dim;
    int innerDim;
    
public:
    SDFeedForward(
        RefPtr<InferencingContext> ctx,
        int dim,
        int mult = 4);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        TensorView residual);  // Adds residual to output
};

// ============================================================================
// SD Basic Transformer Block
// ============================================================================
// Structure: LayerNorm → SelfAttn → LayerNorm → CrossAttn → LayerNorm → FF
class SDBasicTransformerBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<LayerNormKernel> norm1;
    RefPtr<SDSelfAttention> selfAttn;
    RefPtr<LayerNormKernel> norm2;
    RefPtr<SDCrossAttention> crossAttn;
    RefPtr<LayerNormKernel> norm3;
    RefPtr<SDFeedForward> ff;
    
    int dim;
    int contextDim;
    int numHeads;
    
public:
    SDBasicTransformerBlock(
        RefPtr<InferencingContext> ctx,
        int dim,
        int contextDim,
        int numHeads);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        TensorView context);  // Text embeddings
};

// ============================================================================
// SD Spatial Transformer (wraps BasicTransformerBlock with projection)
// ============================================================================
// Structure: GroupNorm → proj_in → TransformerBlocks → proj_out + residual
class SDSpatialTransformer : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<GroupNormKernel> norm;
    RefPtr<Conv2DKernel> projIn;   // 1x1 conv
    List<RefPtr<SDBasicTransformerBlock>> blocks;
    RefPtr<Conv2DKernel> projOut;  // 1x1 conv
    
    int inChannels;
    int numHeads;
    int contextDim;
    int numLayers;  // Usually 1 for SD 1.5
    
public:
    SDSpatialTransformer(
        RefPtr<InferencingContext> ctx,
        int inChannels,
        int numHeads,
        int contextDim,
        int numLayers = 1);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,     // [B, H, W, C]
        TensorView context);  // [B, 77, 768]
};

// ============================================================================
// SD Down Block
// ============================================================================
// Contains ResNet blocks + optional Transformer blocks + optional downsample
class SDDownBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    List<RefPtr<SDResNetBlock>> resnets;
    List<RefPtr<SDSpatialTransformer>> transformers;  // Empty if no attention
    RefPtr<Conv2DKernel> downsample;  // nullptr if no downsampling
    
    int inChannels;
    int outChannels;
    int timeEmbedDim;
    bool hasAttention;
    bool hasDownsample;
    int numLayers;
    
public:
    SDDownBlock(
        RefPtr<InferencingContext> ctx,
        int inChannels,
        int outChannels,
        int timeEmbedDim,
        bool hasAttention,
        bool hasDownsample,
        int numHeads = 8,
        int contextDim = 768,
        int numLayers = 2);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    // Returns list of hidden states for skip connections
    void queueExecute(
        InferencingTask& task,
        List<TensorView>& hiddenStates,  // Output: skip connection states
        TensorView input,
        TensorView timeEmbed,
        TensorView context);
};

// ============================================================================
// SD Mid Block
// ============================================================================
// Structure: ResNet → Transformer → ResNet
class SDMidBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<SDResNetBlock> resnet1;
    RefPtr<SDSpatialTransformer> transformer;
    RefPtr<SDResNetBlock> resnet2;
    
    int channels;
    int timeEmbedDim;
    
public:
    SDMidBlock(
        RefPtr<InferencingContext> ctx,
        int channels,
        int timeEmbedDim,
        int numHeads = 8,
        int contextDim = 768);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        TensorView timeEmbed,
        TensorView context);
};

// ============================================================================
// SD Up Block
// ============================================================================
// Contains ResNet blocks + optional Transformer blocks + optional upsample
// ResNet blocks receive concatenated skip connections
class SDUpBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    List<RefPtr<SDResNetBlock>> resnets;
    List<RefPtr<SDSpatialTransformer>> transformers;
    RefPtr<Conv2DKernel> upsample;  // nullptr if no upsampling
    
    int outChannels;
    int prevChannels;  // Channels from previous up block
    int timeEmbedDim;
    bool hasAttention;
    bool hasUpsample;
    List<int> skipChannelsList;  // Skip channels for each resnet
    
public:
    // skipChannels: list of skip channel counts for each resnet
    // e.g., [1280, 1280, 640] means resnet[0] gets 1280ch skip, resnet[2] gets 640ch skip
    SDUpBlock(
        RefPtr<InferencingContext> ctx,
        int outChannels,
        int prevChannels,
        List<int> skipChannels,
        int timeEmbedDim,
        bool hasAttention,
        bool hasUpsample,
        int numHeads = 8,
        int contextDim = 768);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        List<TensorView>& skipConnections,  // Consumed from back to front
        TensorView timeEmbed,
        TensorView context);
};

// ============================================================================
// SD UNet (Full Model)
// ============================================================================
class SDUNet : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    // Time embedding
    RefPtr<LinearKernel> timeProj1;  // 320 → 1280
    RefPtr<LinearKernel> timeProj2;  // 1280 → 1280
    
    // Input
    RefPtr<Conv2DKernel> convIn;  // 4 → 320
    
    // Down blocks
    List<RefPtr<SDDownBlock>> downBlocks;
    
    // Mid block
    RefPtr<SDMidBlock> midBlock;
    
    // Up blocks
    List<RefPtr<SDUpBlock>> upBlocks;
    
    // Output
    RefPtr<GroupNormKernel> normOut;
    RefPtr<Conv2DKernel> convOut;  // 320 → 4
    
    // Configuration
    int inChannels = 4;
    int outChannels = 4;
    int timeEmbedDim = 1280;
    int contextDim = 768;
    List<int> channelMult;  // [320, 640, 1280, 1280]
    
    // Cached timestep embedding (persistent to avoid async issues)
    RefPtr<Tensor> sinEmbedTensor;
    int cachedTimestep = -1;
    
public:
    SDUNet(RefPtr<InferencingContext> ctx);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix = "");
    
    TensorView allocateResultBuffer(
        ElementType elementType,
        int height,
        int width,
        int batchSize);
    
    // Main inference
    // latent: [B, H, W, 4]
    // timestep: scalar (will be converted to time embedding internally)
    // context: [B, 77, 768] text embeddings from CLIP
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView latent,
        int timestep,
        TensorView context);
};

// Sinusoidal time embedding generation
void getSinusoidalEmbedding(float* output, int timestep, int dim);


#pragma once

#include "inference-context.h"
#include "kernels.h"
#include "safetensors-reader.h"

using namespace Slang;

// ============================================================================
// CLIP Text Encoder (SD 1.5 uses CLIP ViT-L/14)
// ============================================================================
// Input:  [B, 77] token IDs (int32 or float with integer values)
// Output: [B, 77, 768] text embeddings for UNet cross-attention
//
// Architecture:
// 1. Token embedding (49408 vocab -> 768 dim)
// 2. Positional embedding (77 positions -> 768 dim)  
// 3. 12x Transformer blocks (LayerNorm -> Attn -> LayerNorm -> MLP)
// 4. Final LayerNorm
//
// Configuration for SD 1.5:
//   - vocab_size: 49408
//   - hidden_size: 768
//   - num_heads: 12
//   - head_dim: 64
//   - num_layers: 12
//   - max_position_embeddings: 77
//   - intermediate_size: 3072 (4x hidden)

// MLP block: Linear -> QuickGELU -> Linear
class CLIPMLP : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<LinearKernel> fc1;  // hidden_size -> intermediate_size (with fused QuickGELU)
    RefPtr<LinearKernel> fc2;  // intermediate_size -> hidden_size
    
    int hiddenSize;
    int intermediateSize;
    
public:
    CLIPMLP(RefPtr<InferencingContext> ctx, int hiddenSize, int intermediateSize);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input);
};

// Self-attention block with causal masking
class CLIPSelfAttention : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    // Q, K, V projections
    RefPtr<LinearKernel> qProj;
    RefPtr<LinearKernel> kProj;
    RefPtr<LinearKernel> vProj;
    RefPtr<LinearKernel> outProj;
    
    // Flash attention (headDim=64 is within limits)
    RefPtr<FlashAttentionKernel> flashAttn;
    Expr qExpr, kExpr, vExpr;
    
    int hiddenSize;
    int numHeads;
    int headDim;
    
public:
    CLIPSelfAttention(
        RefPtr<InferencingContext> ctx,
        int hiddenSize,
        int numHeads);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        int seqLen,
        int batchSize);
};

// Transformer block: LayerNorm -> Attn -> Residual -> LayerNorm -> MLP -> Residual
class CLIPTransformerBlock : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    RefPtr<LayerNormKernel> layerNorm1;
    RefPtr<CLIPSelfAttention> selfAttn;
    RefPtr<LayerNormKernel> layerNorm2;
    RefPtr<CLIPMLP> mlp;
    
    // Residual additions (fused)
    RefPtr<ElementwiseKernel> residualAdd1;
    RefPtr<ElementwiseKernel> residualAdd2;
    
    int hiddenSize;
    
public:
    CLIPTransformerBlock(
        RefPtr<InferencingContext> ctx,
        int hiddenSize,
        int numHeads,
        int intermediateSize);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix);
    
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView input,
        int seqLen,
        int batchSize);
};

// Full CLIP Text Encoder
class CLIPTextEncoder : public RefObject
{
public:
    RefPtr<InferencingContext> ctx;
    
    // Fused embeddings: gather(tokenTable, tokenIds) + gather(posTable, posIds)
    // This replaces 3 separate kernels with 1 fused kernel
    RefPtr<ElementwiseKernel> embeddingKernel;
    Expr tokenTableExpr;   // Token embedding table [vocab, hidden]
    Expr tokenIdsExpr;     // Token ID indices
    Expr posTableExpr;     // Position embedding table [maxSeq, hidden]
    Expr posIdsExpr;       // Position ID indices
    
    // Embedding table buffers (loaded from SafeTensors)
    ComPtr<rhi::IBuffer> tokenTableBuffer;
    ComPtr<rhi::IBuffer> posTableBuffer;
    
    // Transformer layers
    List<RefPtr<CLIPTransformerBlock>> layers;
    
    // Final layer norm
    RefPtr<LayerNormKernel> finalLayerNorm;
    
    // Configuration
    int vocabSize;
    int hiddenSize;
    int numHeads;
    int numLayers;
    int maxSeqLen;
    int intermediateSize;
    
public:
    CLIPTextEncoder(
        RefPtr<InferencingContext> ctx,
        int vocabSize = 49408,
        int hiddenSize = 768,
        int numHeads = 12,
        int numLayers = 12,
        int maxSeqLen = 77,
        int intermediateSize = 3072);
    
    SlangResult loadParams(SafeTensorsReader& reader, const String& prefix = "text_model.");
    
    TensorView allocateResultBuffer(ElementType elementType, int seqLen, int batchSize);
    
    // Encode token IDs to embeddings
    // tokenIds: [B, seqLen] tensor with integer token IDs
    // output: [B, seqLen, hiddenSize] text embeddings
    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView tokenIds,
        int seqLen,
        int batchSize);
};


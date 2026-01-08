# Stable Diffusion 1.5 Implementation

This directory contains a high-performance implementation of Stable Diffusion 1.5 using the MLKL inference engine.

## Components

- **CLIP Text Encoder** (`clip-encoder.h/cpp`) - ViT-L/14 text encoder
- **VAE Decoder** (`vae-decoder.h/cpp`) - Latent to image decoder
- **UNet** (`unet.h/cpp`) - Noise prediction network

## Kernel Fusion Optimizations

The implementation uses aggressive kernel fusion to minimize memory bandwidth and kernel launch overhead. Below is a comprehensive list of all fusions applied.

---

### SDResNetBlock

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| concat (up blocks only) | Concat fused into norm1 inputExpr | `norm1 = GroupNormKernel(..., concat(buf0, buf1, 3), ...)` |
| norm1 → SiLU → conv1 | SiLU fused into conv1 inputExpr | `conv1 = Conv2DKernel(..., silu(buffer()), ...)` |
| SiLU → timeProj | SiLU fused into timeProj inputExpr | `timeProj = LinearKernel(..., silu(buffer()), ...)` |
| conv1 + time_emb → norm2 | Time add fused into norm2 inputExpr | `norm2 = GroupNormKernel(..., buf0 + broadcast(buf1, buf0), ...)` |
| norm2 → SiLU → conv2 | SiLU fused into conv2 inputExpr | `conv2 = Conv2DKernel(..., silu(buffer()), ...)` |
| conv2 + residual | Residual fused into conv2 outputExpr | `conv2 = Conv2DKernel(..., kernelOutput() + buffer(), ...)` |
| residualConv (with concat) | Concat fused into residualConv inputExpr | `residualConv = Conv2DKernel(..., concat(buf0, buf1, 3), ...)` |

**All operations fused - no standalone kernels!**

---

### SDSelfAttention

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| Q/K/V reshape + transpose | Fused into FlashAttention via permute expressions | `permute(qExpr, {0,2,1,3})` in FlashAttentionKernel |
| Output transpose | Fused into FlashAttention sink expression | `permute(bufferSink(), {0,2,1,3})` |
| toOut + residual | Residual fused into toOut outputExpr | `toOut = LinearKernel(..., kernelOutput() + buffer(), ...)` |

---

### SDCrossAttention

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| Q/K/V reshape + transpose | Fused into FlashAttention via permute expressions | `permute(qExpr, {0,2,1,3})` in FlashAttentionKernel |
| Output transpose | Fused into FlashAttention sink expression | `permute(bufferSink(), {0,2,1,3})` |
| toOut + residual | Residual fused into toOut outputExpr | `toOut = LinearKernel(..., kernelOutput() + buffer(), ...)` |

---

### SDFeedForward (GEGLU)

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| hidden_states * gelu(gate) | GEGLU fused into proj2 inputExpr | `proj2 = LinearKernel(..., buffer() * gelu(buffer()), ...)` |
| proj2 + residual | Residual fused into proj2 outputExpr | `proj2 = LinearKernel(..., kernelOutput() + buffer(), ...)` |

**Remaining standalone kernel:** Permute for GEGLU split (required for contiguous slice operation)

---

### SDSpatialTransformer

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| projOut + residual | Residual fused into projOut outputExpr | `projOut = Conv2DKernel(..., kernelOutput() + buffer(), ...)` |

---

### SDUpBlock

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| Nearest-neighbor 2x upsample + conv | Upsample fused into conv inputExpr | `upsample = Conv2DKernel(..., upsample2x(buffer()), ...)` |
| Final copy elimination | Last resnet/transformer writes directly to output | Conditional destination selection in loop |

---

### CLIP Encoder

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| fc1 + QuickGELU | QuickGELU fused into fc1 outputExpr | `fc1 = LinearKernel(..., quickGelu(kernelOutput()), ...)` |

---

### VAE Decoder

| Operation | Fusion | Implementation |
|-----------|--------|----------------|
| GroupNorm → SiLU → Conv | SiLU fused into conv inputExpr | `conv = Conv2DKernel(..., silu(buffer()), ...)` |
| conv + residual | Residual fused into conv outputExpr | `conv = Conv2DKernel(..., kernelOutput() + buffer(), ...)` |
| Final clamp | Clamp fused into convOut outputExpr | `convOut = Conv2DKernel(..., clamp(kernelOutput(), -1, 1), ...)` |

---

## Fusion Summary

| Category | Before | After | Reduction |
|----------|--------|-------|-----------|
| UNet ElementwiseKernels | 9 | 1 | 89% |
| UNet standalone add/copy ops | 6 | 0 | 100% |

### Remaining Standalone Kernels (UNet)

1. **GEGLU permute** - Required for contiguous memory layout before slice

Note: ConcatKernel has been eliminated by fusing `concat()` expression into the ResNet's norm1 and residualConv inputExprs.

---

## Expression System

Fusions are enabled by the IExpr system which allows composing input/output transformations:

```cpp
// Input expression: transform input before kernel computation
Conv2DKernel(ctx, ..., silu(buffer()), kernelOutput(), bufferSink());

// Output expression: transform output after kernel computation  
LinearKernel(ctx, ..., buffer(), kernelOutput() + buffer(), bufferSink());

// Combined: both input and output transformations
LinearKernel(ctx, ..., 
    buffer() * gelu(buffer()),      // GEGLU input
    kernelOutput() + buffer(),       // residual output
    bufferSink());
```

### Available Expression Nodes

| Expression | Description |
|------------|-------------|
| `buffer()` | Input tensor reference |
| `kernelOutput()` | Kernel computation result |
| `constant(f)` | Scalar constant |
| `silu(x)` | SiLU activation |
| `gelu(x)` | GELU activation |
| `quickGelu(x)` | QuickGELU (CLIP) |
| `clamp(x, lo, hi)` | Value clamping |
| `broadcast(x, shape)` | Broadcasting |
| `permute(x, dims)` | Dimension permutation |
| `transpose(x, d0, d1)` | Dimension swap |
| `upsample2x(x)` | 2x nearest-neighbor upsample |

### Sink Expressions

| Expression | Description |
|------------|-------------|
| `bufferSink()` | Write to output buffer |
| `permute(bufferSink(), dims)` | Permute then write |

---

## Usage

```cpp
// Initialize context
auto ctx = InferencingContext::create(device);

// Load models
auto clip = new CLIPTextEncoder(ctx, ...);
clip->loadParams(reader, "text_model.");

auto unet = new SDUNet(ctx, ...);
unet->loadParams(reader, "");

auto vae = new VAEDecoder(ctx, ...);
vae->loadParams(reader, "decoder.");

// Run inference
InferencingTask task(ctx);
clip->queueExecute(task, textEmbed, tokenIds);
unet->queueExecute(task, noise, latent, context, timestep);
vae->queueExecute(task, image, latent);
task.execute();
```

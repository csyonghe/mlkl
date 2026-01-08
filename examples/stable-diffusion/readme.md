# Stable Diffusion 1.5 on MLKL

This document provides a comprehensive overview of our Stable Diffusion 1.5 implementation, focusing on the architecture, kernel scheduling, fusion opportunities, and how MLKL's **IExpr system** enables aggressive optimization.

---

## Table of Contents

1. [Stable Diffusion Architecture](#stable-diffusion-architecture)
2. [The Kernel Scheduling Problem](#the-kernel-scheduling-problem)
3. [Understanding Kernel Fusion](#understanding-kernel-fusion)
4. [The IExpr System](#the-iexpr-system)
5. [Fusion Catalog](#fusion-catalog)
6. [Results: Detailed Kernel Count Analysis](#results-detailed-kernel-count-analysis)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

---

## Stable Diffusion Architecture

Stable Diffusion 1.5 is a latent diffusion model consisting of three main components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STABLE DIFFUSION 1.5                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐                                                      │
│   │    CLIP      │  "a photo of a cat"                                  │
│   │ Text Encoder │  ───────────────────►  [77, 768] text embeddings     │
│   └──────────────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐      ┌─────────────┐                                 │
│   │    UNet      │◄─────│   DDIM      │  (50 denoising steps)           │
│   │  Denoiser    │─────►│  Scheduler  │                                 │
│   └──────────────┘      └─────────────┘                                 │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │     VAE      │  [4, 64, 64] latent  ─►  [3, 512, 512] RGB image     │
│   │   Decoder    │                                                      │
│   └──────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### CLIP Text Encoder (ViT-L/14)

Transforms text prompts into conditioning embeddings that guide image generation.

```
Input: Token IDs [B, 77]
  │
  ├─► Token Embedding ─────────┐
  │                            ├─► Add ─► LayerNorm ─► TransformerBlock ×12
  └─► Position Embedding ──────┘
                                        │
                                        ▼
                               Output: [B, 77, 768]
```

**TransformerBlock:**
```
Input ─► LayerNorm ─► SelfAttention ─► + ─► LayerNorm ─► MLP ─► + ─► Output
   │                                   ▲                        ▲
   └───────────────────────────────────┴────────────────────────┘
                              (residual connections)
```

### UNet Noise Predictor

The heart of diffusion - predicts noise to be removed at each timestep.

```
                              ┌─────────────────┐
                              │  Time Embedding │
                              │  sinusoidal +   │
                              │  2× Linear+SiLU │
                              └────────┬────────┘
                                       │
Input [4,64,64] ─► ConvIn ─────────────┼──────────────────────► ConvOut ─► Output
                     │                 │                             ▲
              ┌──────┴──────┐          │                    ┌────────┴────────┐
              │  DownBlock  │          │                    │     UpBlock     │
              │  320 ch     ├──────────┼───────────────────►│     320 ch      │
              ├─────────────┤          │                    ├─────────────────┤
              │  DownBlock  │          │                    │     UpBlock     │
              │  640 ch     ├──────────┼───────────────────►│     640 ch      │
              ├─────────────┤          │                    ├─────────────────┤
              │  DownBlock  │          │                    │     UpBlock     │
              │  1280 ch    ├──────────┼───────────────────►│     1280 ch     │
              ├─────────────┤          │                    ├─────────────────┤
              │  DownBlock  │          │                    │     UpBlock     │
              │  1280 ch    ├──────────┼───────────────────►│     1280 ch     │
              └──────┬──────┘          │                    └────────▲────────┘
                     │                 │                             │
                     └────────► MidBlock ────────────────────────────┘
                                  │
                            ResNet + Attention + ResNet
```

**ResNetBlock (the workhorse):**
```
Input ────────────────────────────────────────────────────►(+)─► Output
  │                                                          ▲
  └─► GroupNorm ─► SiLU ─► Conv3×3 ─► (+TimeEmb) ─► GroupNorm ─► SiLU ─► Conv3×3
                                         ▲
                               TimeEmb ──┘ (broadcast add)
```

**SpatialTransformer (cross-attention with text):**
```
Input ─► GroupNorm ─► ProjIn ─► BasicTransformerBlock ─► ProjOut ─► (+) ─► Output
   │                                                                  ▲
   └──────────────────────────────────────────────────────────────────┘

BasicTransformerBlock:
  ─► LayerNorm ─► SelfAttn ─► (+) ─► LayerNorm ─► CrossAttn ─► (+) ─► LayerNorm ─► FFN ─► (+)
```

**FeedForward with GEGLU:**
```
Input ─► Linear(dim → 2×inner) ─► GEGLU ─► Linear(inner → dim) ─► Output

GEGLU: split(x) → [hidden, gate] → hidden * gelu(gate)
```

### VAE Decoder

Upscales 4-channel latents to 3-channel RGB images.

```
Latent [4,64,64]
    │
    ├─► Conv3×3 ─► MidBlock (ResNet + Attention + ResNet)
    │
    ├─► UpBlock 512ch (3× ResNet + Upsample)
    ├─► UpBlock 512ch (3× ResNet + Upsample)  
    ├─► UpBlock 256ch (3× ResNet + Upsample)
    └─► UpBlock 128ch (3× ResNet)
            │
            └─► GroupNorm ─► SiLU ─► Conv3×3 ─► Output [3,512,512]
```

---

## The Kernel Scheduling Problem

Each box in the diagrams above represents a GPU kernel launch. Without fusion, a single UNet forward pass requires:

| Component | Kernel Launches |
|-----------|-----------------|
| Time embedding | 4 |
| ConvIn/ConvOut | 2 |
| DownBlocks (4×) | ~160 |
| MidBlock | ~40 |
| UpBlocks (4×) | ~200 |
| **Total per step** | **~400** |
| **50 DDIM steps** | **~20,000** |

Each kernel launch has overhead:
- **Launch latency**: ~5-10μs per kernel
- **Memory bandwidth**: Reading/writing intermediate tensors
- **Synchronization**: GPU idle time between kernels

For memory-bound operations (element-wise ops, small convolutions), the kernel launch overhead can exceed the actual compute time!

### The Naive Kernel Schedule

Consider a ResNetBlock without fusion:

```
1. GroupNorm(input)           → temp1
2. SiLU(temp1)                → temp2      ← ElementwiseKernel  
3. Conv3×3(temp2)             → temp3
4. TimeProj(timeEmb)          → temp4
5. SiLU(temp4)                → temp5      ← ElementwiseKernel
6. BroadcastAdd(temp3, temp5) → temp6      ← ElementwiseKernel
7. GroupNorm(temp6)           → temp7
8. SiLU(temp7)                → temp8      ← ElementwiseKernel
9. Conv3×3(temp8)             → temp9
10. Add(temp9, residual)      → output     ← ElementwiseKernel
```

**5 standalone ElementwiseKernels** just for element-wise operations!

---

## Understanding Kernel Fusion

**Kernel fusion** combines multiple operations into a single kernel launch, eliminating:
1. Intermediate memory allocations
2. Memory round-trips (write → read)
3. Kernel launch overhead

### Types of Fusion

**1. Input Fusion (Pre-processing)**
```
Before: temp = SiLU(input);  output = Conv(temp);
After:  output = Conv(SiLU(input));  // SiLU computed on-the-fly during Conv
```

**2. Output Fusion (Post-processing)**
```
Before: temp = Conv(input);  output = temp + residual;
After:  output = Conv(input) + residual;  // Addition done as Conv writes
```

**3. Multi-input Fusion**
```
Before: temp = Concat(a, b);  output = GroupNorm(temp);
After:  output = GroupNorm(Concat(a, b));  // Concat computed on-the-fly
```

### Why Fusion is Hard

Traditional approaches require writing custom fused kernels for each combination:
- `Conv + SiLU`
- `Conv + ReLU`
- `Conv + SiLU + Add`
- `GroupNorm + SiLU + Conv + Add + ...`

This leads to **combinatorial explosion** - you can't anticipate every useful combination.

---

## The IExpr System

MLKL solves fusion through **IExpr** (Input/Output Expressions) - a composable type system for describing data transformations. The key insight is that we don't generate Slang code - instead, we generate **Slang type signatures** that specialize generic kernel entrypoints.

### The Slang IExpr Interface

All expression types conform to a simple interface:

```slang
interface IExpr<T : ITensorElement>
{
    // Given a coordinate and optional input value, compute the output value
    T.UnpackedType eval(Coord coord, Input<T> input);
}
```

Each expression type is a struct that implements this interface:

```slang
// Leaf: reads from a GPU buffer at the given coordinate
struct BufferView<T : ITensorElement> : IExpr<T>
{
    T* data;
    int rank;
    uint strides[8];
    
    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        uint linearIdx = 0;
        for (int i = 0; i < rank; i++)
            linearIdx += coord[i] * strides[i];
        return data[linearIdx].unpack();
    }
}

// Unary: applies sigmoid to inner expression
struct Sigmoid<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    
    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        let v = inner.eval(coord, input);
        return 1.0 / (1.0 + (-v).exp());
    }
}

// Binary: adds two expressions element-wise
struct Add<T : ITensorElement, L : IExpr<T>, R : IExpr<T>> : IExpr<T>
{
    L left;
    R right;
    
    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        return left.eval(coord, input) + right.eval(coord, input);
    }
}

// Slice: offsets coordinate along an axis
struct Slice<T : ITensorElement, Inner : IExpr<T>> : IExpr<T>
{
    Inner inner;
    uint axis;
    uint start;
    
    T.UnpackedType eval(Coord coord, Input<T> input)
    {
        Coord innerCoord = coord;
        innerCoord[axis] = coord[axis] + start;
        return inner.eval(innerCoord, input);
    }
}
```

### How C++ Expression Trees Become Slang Types

When you write C++ code like:

```cpp
auto buf = buffer();
auto hidden = sliceLastDim(buf, 0, 1280);
auto gate = sliceLastDim(buf, 1280, 1280);
auto gegluExpr = hidden * gelu(gate);
```

Each C++ `Expr` node has a `getSlangTypeName()` method that generates a **type signature**:

```cpp
// BufferNode::getSlangTypeName() returns:
"BufferView<float>"

// SliceNode::getSlangTypeName() returns (recursively):
"Slice<float, BufferView<float>>"

// The full GEGLU expression becomes:
"Mul<float, Slice<float, BufferView<float>>, GELU<float, Slice<float, BufferView<float>>>>"
```

### Kernel Entrypoints Use Type Specialization

Kernel entrypoints are **generic** over the expression type. For example, LinearKernel:

```slang
// Generic entrypoint - TInput can be ANY type conforming to IExpr<T>
void linearTiled<
    let TILE_M : int, let TILE_N : int, let TILE_K : int,
    T : ITensorElement,
    TInput : IExpr<T>,      // ← Generic input expression type
    TSink : ISink<T>,       // ← Generic output sink type  
    TOutput : IExpr<T>      // ← Generic output expression type
>(
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    ConstantBuffer<LinearParams<T, TInput, TSink, TOutput>> params)
{
    // ... tiled matrix multiply ...
    
    // Read input through the expression tree
    s_A[row][k] = params.input.eval(Coord(globalRow, globalK), emptyInput);
    
    // ... compute ...
    
    // Write output through the sink
    params.outputSink.store(outputCoord, params.outFunc.eval(outputCoord, input));
}
```

When MLKL creates a kernel, it **specializes** this generic entrypoint with the concrete type signature:

```cpp
// C++ kernel creation
String specArgs[] = {
    "16", "16", "16",                                    // Tile sizes
    "float",                                              // Element type
    inputProgram.getSlangTypeName(elementType),          // e.g., "Slice<float, BufferView<float>>"
    sinkExpr.node->getSlangTypeName(elementType),        // e.g., "BufferSink<float>"
    outputProgram.getSlangTypeName(elementType)          // e.g., "Add<float, KernelOutput<float>, BufferView<float>>"
};
pipeline = context->createComputePipeline("linearTiled", specArgs);
```

The Slang compiler then generates a **fully specialized** GPU kernel where:
- All `eval()` calls are inlined
- The expression tree traversal becomes straight-line code
- No virtual dispatch, no indirection - just optimized math

### Core Concept

Instead of hardcoding fused kernels, we describe transformations as **expression trees**:

```cpp
// Expression tree for: SiLU(input)
Expr inputExpr = silu(buffer());

// Expression tree for: output + residual  
Expr outputExpr = kernelOutput() + buffer();

// Create kernel with fused input/output transformations
auto conv = new Conv2DKernel(ctx, ..., inputExpr, outputExpr, bufferSink());
```

At kernel creation time, MLKL:
1. Walks the expression tree
2. Generates a Slang type signature (e.g., `"SiLU<float, BufferView<float>>"`)
3. Specializes the generic entrypoint with this type
4. Compiles the specialized kernel to GPU code

At runtime:
1. Maps `buffer()` nodes to actual tensor data pointers
2. Packs expression parameters into a constant buffer
3. Dispatches the specialized kernel

### Expression Building Blocks

**Leaf Nodes:**
```cpp
buffer()          // Reference to an input tensor
kernelOutput()    // The kernel's computed result
constant(1.5f)    // Compile-time constant
uniformConstant() // Runtime-provided constant
```

**Unary Operations:**
```cpp
silu(x)           // SiLU activation: x * sigmoid(x)
gelu(x)           // GELU activation
sigmoid(x)        // Sigmoid
tanh(x)           // Tanh
exp(x), log(x)    // Exponential, logarithm
neg(x), abs(x)    // Negation, absolute value
sqrt(x), rsqrt(x) // Square root, reciprocal sqrt
```

**Binary Operations:**
```cpp
x + y             // Addition (with broadcast)
x - y             // Subtraction
x * y             // Multiplication
x / y             // Division
```

**Shape-changing Operations:**
```cpp
broadcast(x, shapeOf)         // Broadcast to target shape
concat(a, b, axis)            // Concatenate along axis
slice(x, axis, start, size)   // Extract slice
sliceLastDim(x, start, size)  // Slice last dimension
permute(x, {dims})            // Reorder dimensions
transpose(x, dim0, dim1)      // Swap two dimensions
upsample2x(x)                 // 2× nearest-neighbor upsample
```

### How Expression Trees Work

Consider the GEGLU operation:
```
hidden_states, gate = proj1_output.chunk(2, dim=-1)
output = hidden_states * gelu(gate)
```

Without IExpr, this requires:
1. Allocate temp buffer
2. Permute kernel to make chunks contiguous
3. Slice to get hidden_states
4. Slice to get gate
5. GELU kernel on gate
6. Multiply kernel

With IExpr:
```cpp
auto buf = buffer();  // proj1 output: [N, 2*D]
auto hidden = sliceLastDim(buf, 0, innerDim);      // [N, D] - first half
auto gate = sliceLastDim(buf, innerDim, innerDim); // [N, D] - second half
auto gegluExpr = hidden * gelu(gate);

// Fuse into the next Linear layer's input
proj2 = new LinearKernel(ctx, ..., gegluExpr, ...);
```

The expression tree:
```
         (*)
        /   \
   slice     gelu
     |         |
   buffer    slice
               |
             buffer  ← Same buffer node (shared reference)
```

When `proj2` executes, for each output element it:
1. Computes which input elements are needed
2. Slices into the correct positions
3. Applies GELU to the gate half
4. Multiplies hidden × gelu(gate)
5. Feeds result to the Linear computation

**All in a single kernel launch, no intermediate buffers!**

### Multi-Buffer Expressions

When an expression contains multiple `buffer()` calls, each maps to a different input:

```cpp
// Time embedding addition with broadcast
auto conv1Out = buffer();    // [B, H, W, C]
auto timeProj = buffer();    // [B, 1, 1, C] 
auto timeAddExpr = conv1Out + broadcast(timeProj, conv1Out);

norm2 = new GroupNormKernel(ctx, ..., timeAddExpr, ...);

// At runtime, provide both inputs:
norm2->queueExecute(task, output, {afterConv1, timeProjReshaped});
```

Buffer nodes are matched to inputs **in order of creation**:
- First `buffer()` → first input in list
- Second `buffer()` → second input in list

### Sink Expressions (Output Transformations)

While input expressions transform data *before* the kernel computes, **sink expressions** transform data *as it's written*:

```cpp
// Permute output layout as it's written
auto outSink = permute(bufferSink(), {0, 2, 1, 3});  // BSHD → BHSD

attention = new FlashAttentionKernel(ctx, qExpr, kExpr, vExpr, outSink);
```

This eliminates the need for a separate permute kernel after attention.

---

## Fusion Catalog

### ResNetBlock: 5 Fusions → 0 Standalone Kernels

**Before (naive implementation):**
```
1. GroupNorm(input)
2. SiLU(temp1)                    ← Standalone ElementwiseKernel
3. Conv3×3(temp2)
4. SiLU(timeEmb)                  ← Standalone ElementwiseKernel  
5. TimeProj(temp4)
6. BroadcastAdd(conv1Out, time)   ← Standalone ElementwiseKernel
7. GroupNorm(temp6)
8. SiLU(temp7)                    ← Standalone ElementwiseKernel
9. Conv3×3(temp8)
10. Add(conv2Out, residual)       ← Standalone ElementwiseKernel
```

**After (with IExpr fusion):**
```cpp
// Fusion 1: SiLU into Conv1 input
conv1 = new Conv2DKernel(ctx, ..., silu(buffer()), kernelOutput(), bufferSink());

// Fusion 2: SiLU into TimeProj input
timeProj = new LinearKernel(ctx, ..., silu(buffer()), kernelOutput(), bufferSink());

// Fusion 3: BroadcastAdd into Norm2 input
auto buf0 = buffer();  // conv1 output
auto buf1 = buffer();  // time projection
norm2 = new GroupNormKernel(ctx, ..., buf0 + broadcast(buf1, buf0), bufferSink());

// Fusion 4: SiLU into Conv2 input
// Fusion 5: Residual add into Conv2 output  
conv2 = new Conv2DKernel(ctx, ..., silu(buffer()), kernelOutput() + buffer(), bufferSink());
```

**Kernel schedule after fusion:**
```
1. GroupNorm(input)                      // norm1
2. Conv3×3(SiLU(input))                  // conv1 with fused SiLU
3. Linear(SiLU(timeEmb))                 // timeProj with fused SiLU
4. GroupNorm(conv1Out + broadcast(time)) // norm2 with fused add
5. Conv3×3(SiLU(input)) + residual       // conv2 with fused SiLU + residual add
```

**5 kernels eliminated!**

### ResNetBlock in UpBlocks: +2 More Fusions

UpBlocks concatenate skip connections, adding another fusion opportunity:

```cpp
// Fusion 6: Concat into Norm1 input
auto current = buffer();
auto skip = buffer();
auto axis = uniformConstant();  // Axis provided at runtime
norm1 = new GroupNormKernel(ctx, ..., concat(current, skip, axis), bufferSink());

// Fusion 7: Concat into ResidualConv input (if channel mismatch)
residualConv = new Conv2DKernel(ctx, ..., concat(current, skip, axis), kernelOutput(), bufferSink());
```

**ConcatKernel eliminated!**

### FeedForward GEGLU: 1 Fusion → 0 Standalone Kernels

**Before:**
```
1. Linear(input) → [N, 2*D]
2. Permute to make chunks contiguous   ← Standalone ElementwiseKernel
3. Slice hidden_states
4. Slice gate  
5. GELU(gate)                          ← Standalone ElementwiseKernel (or fused)
6. Multiply(hidden, gelu_gate)         ← Standalone ElementwiseKernel
7. Linear(result)
8. Add(residual)                       ← Standalone ElementwiseKernel
```

**After (with sliceLastDim):**
```cpp
auto proj1Out = buffer();  // [N, 2*innerDim]
auto hidden = sliceLastDim(proj1Out, 0, innerDim);
auto gate = sliceLastDim(proj1Out, innerDim, innerDim);
auto gegluExpr = hidden * gelu(gate);

// All fused into proj2
proj2 = new LinearKernel(ctx, ..., gegluExpr, kernelOutput() + buffer(), bufferSink());
```

**Kernel schedule after fusion:**
```
1. Linear(input) → proj1Out
2. Linear(GEGLU(proj1Out)) + residual  // proj2 with fused GEGLU + residual
```

**Permute, GELU, Multiply, and Add all eliminated!**

### Attention: Fused Permutations

FlashAttention naturally expects `[B, H, S, D]` but linear projections output `[B, S, H*D]`:

```cpp
// Q projection with reshape + permute fused
auto qExpr = permute(reshape(buffer(), {B, S, H, D}), {0, 2, 1, 3});
auto kExpr = permute(reshape(buffer(), {B, S, H, D}), {0, 2, 1, 3});
auto vExpr = permute(reshape(buffer(), {B, S, H, D}), {0, 2, 1, 3});

// Output permute fused into sink
auto outSink = permute(bufferSink(), {0, 2, 1, 3});  // BHSD → BSHD

attention = new FlashAttentionKernel(ctx, qExpr, kExpr, vExpr, outSink);
```

### Upsampling: Fused into Convolution

```cpp
// 2× nearest-neighbor upsample fused into 3×3 conv
upsampleConv = new Conv2DKernel(ctx, ..., upsample2x(buffer()), kernelOutput(), bufferSink());
```

The `upsample2x()` expression divides coordinates by 2 when reading, effectively computing the upsample on-the-fly.

---

## Results: Detailed Kernel Count Analysis

### UNet Architecture Breakdown

The SD 1.5 UNet consists of:

| Block Type | Count | ResNets | Attentions | Total Layers |
|------------|-------|---------|------------|--------------|
| DownBlock 0 | 1 | 2 | 1 | 3 |
| DownBlock 1 | 1 | 2 | 1 | 3 |
| DownBlock 2 | 1 | 2 | 1 | 3 |
| DownBlock 3 | 1 | 2 | 0 | 2 |
| MidBlock | 1 | 2 | 1 | 3 |
| UpBlock 0 | 1 | 3 | 0 | 3 |
| UpBlock 1 | 1 | 3 | 1 | 4 |
| UpBlock 2 | 1 | 3 | 1 | 4 |
| UpBlock 3 | 1 | 3 | 1 | 4 |
| **Total** | 9 | **22** | **7** | **29** |

### Per-Component Kernel Counts

#### ResNetBlock (Naive vs Fused)

| Operation | Naive | Fused | Notes |
|-----------|-------|-------|-------|
| GroupNorm1 | 1 | 1 | |
| SiLU1 | 1 | 0 | Fused into Conv1 inputExpr |
| Conv1 | 1 | 1 | |
| SiLU(timeEmb) | 1 | 0 | Fused into TimeProj inputExpr |
| TimeProj | 1 | 1 | |
| BroadcastAdd | 1 | 0 | Fused into GroupNorm2 inputExpr |
| GroupNorm2 | 1 | 1 | |
| SiLU2 | 1 | 0 | Fused into Conv2 inputExpr |
| Conv2 | 1 | 1 | |
| ResidualAdd | 1 | 0 | Fused into Conv2 outputExpr |
| **Total** | **10** | **5** | **50% reduction** |

#### ResNetBlock in UpBlock (with skip concat)

| Operation | Naive | Fused | Notes |
|-----------|-------|-------|-------|
| Concat(current, skip) | 1 | 0 | Fused into GroupNorm1 inputExpr |
| GroupNorm1 | 1 | 1 | |
| SiLU1 | 1 | 0 | Fused into Conv1 |
| Conv1 | 1 | 1 | |
| SiLU(timeEmb) | 1 | 0 | Fused into TimeProj |
| TimeProj | 1 | 1 | |
| BroadcastAdd | 1 | 0 | Fused into GroupNorm2 |
| GroupNorm2 | 1 | 1 | |
| SiLU2 | 1 | 0 | Fused into Conv2 |
| Conv2 | 1 | 1 | |
| ResidualConv (concat input) | 1 | 1 | Concat fused into inputExpr |
| ResidualAdd | 1 | 0 | Fused into Conv2 outputExpr |
| **Total** | **12** | **6** | **50% reduction** |

#### SpatialTransformer (BasicTransformerBlock)

| Operation | Naive | Fused | Notes |
|-----------|-------|-------|-------|
| GroupNorm | 1 | 1 | |
| ProjIn (Conv1×1) | 1 | 1 | |
| LayerNorm1 | 1 | 1 | |
| Q Linear | 1 | 1 | |
| K Linear | 1 | 1 | |
| V Linear | 1 | 1 | |
| Q Permute | 1 | 0 | Fused into FlashAttention qExpr |
| K Permute | 1 | 0 | Fused into FlashAttention kExpr |
| V Permute | 1 | 0 | Fused into FlashAttention vExpr |
| FlashAttention | 1 | 1 | |
| Out Permute | 1 | 0 | Fused into FlashAttention sinkExpr |
| ToOut Linear | 1 | 1 | |
| SelfAttn Add | 1 | 0 | Fused into ToOut outputExpr |
| LayerNorm2 | 1 | 1 | |
| (Cross-attention: same pattern) | +10 | +4 | 6 fused |
| LayerNorm3 | 1 | 1 | |
| FFN Proj1 | 1 | 1 | |
| FFN Permute (GEGLU) | 1 | 0 | Eliminated by sliceLastDim |
| FFN GELU | 1 | 0 | Fused into Proj2 inputExpr |
| FFN Multiply | 1 | 0 | Fused into Proj2 inputExpr |
| FFN Proj2 | 1 | 1 | |
| FFN Add | 1 | 0 | Fused into Proj2 outputExpr |
| ProjOut (Conv1×1) | 1 | 1 | |
| Transformer Add | 1 | 0 | Fused into ProjOut outputExpr |
| **Total** | **~32** | **~16** | **50% reduction** |

### Full UNet Kernel Count

| Component | Count | Naive Kernels | Fused Kernels |
|-----------|-------|---------------|---------------|
| Time Embedding | 1 | 4 | 2 |
| ConvIn | 1 | 1 | 1 |
| DownBlock ResNets | 8 | 80 | 40 |
| DownBlock Attentions | 3 | 96 | 48 |
| MidBlock ResNets | 2 | 20 | 10 |
| MidBlock Attention | 1 | 32 | 16 |
| UpBlock ResNets | 12 | 144 | 72 |
| UpBlock Attentions | 3 | 96 | 48 |
| UpBlock Upsamples | 3 | 6 | 3 |
| ConvOut | 1 | 1 | 1 |
| **Total per step** | - | **480** | **241** |
| **50 DDIM steps** | - | **24,000** | **12,050** |

### Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Kernels per UNet step | 480 | 241 | **50% fewer launches** |
| Kernels per 50-step generation | 24,000 | 12,050 | **~12,000 launches saved** |
| Standalone ElementwiseKernels | ~100 | 0 | **100% eliminated** |
| Standalone ConcatKernels | ~12 | 0 | **100% eliminated** |
| Standalone PermuteKernels | ~40 | 0 | **100% eliminated** |

### Memory Bandwidth Savings

Each eliminated kernel saves one write + one read of the intermediate tensor:

| Tensor Size (at 64×64 resolution) | Bytes Saved Per Fusion |
|-----------------------------------|------------------------|
| [1, 64, 64, 320] | 5.24 MB |
| [1, 32, 32, 640] | 2.62 MB |
| [1, 16, 16, 1280] | 1.31 MB |
| [1, 8, 8, 1280] | 0.33 MB |

**Conservative estimate**: 200+ MB of memory bandwidth saved per UNet forward pass.

At 50 DDIM steps: **10+ GB** of memory bandwidth saved per image generation!

---

## The Power of Composability

The IExpr system's true power is **composability**. New operations can be added without modifying existing kernels:

```cpp
// Today: SiLU activation
conv = new Conv2DKernel(ctx, ..., silu(buffer()), ...);

// Tomorrow: New activation? Just add the expression node
conv = new Conv2DKernel(ctx, ..., mish(buffer()), ...);

// Complex transformations compose naturally
auto expr = silu(buffer() + broadcast(buffer(), buffer()));
```

This stands in contrast to traditional approaches where each fusion requires a custom kernel implementation.

---

## Available Expression Nodes

### Input/Buffer Expressions

| Expression | Description |
|------------|-------------|
| `buffer()` | Reference to an input tensor (each call creates a new input slot) |
| `kernelOutput()` | The kernel's computed result (for output expressions) |
| `constant(f)` | Compile-time constant |
| `uniformConstant()` | Runtime-provided scalar (via `EvalContext`) |

### Unary Operations

| Expression | Description |
|------------|-------------|
| `silu(x)` | SiLU: x × σ(x) |
| `gelu(x)` | GELU: x × Φ(x) |
| `quickGelu(x)` | QuickGELU: x × σ(1.702x) |
| `sigmoid(x)` | Sigmoid: 1/(1+e^(-x)) |
| `tanh(x)` | Hyperbolic tangent |
| `relu(x)` | ReLU: max(0, x) |
| `exp(x)`, `log(x)` | Exponential, natural log |
| `sqrt(x)`, `rsqrt(x)` | Square root, reciprocal sqrt |
| `neg(x)`, `abs(x)` | Negation, absolute value |
| `clamp(x, lo, hi)` | Clamp to range |

### Binary Operations

| Expression | Description |
|------------|-------------|
| `x + y` | Addition (with automatic broadcast) |
| `x - y` | Subtraction |
| `x * y` | Multiplication |
| `x / y` | Division |

### Shape Transformations

| Expression | Description |
|------------|-------------|
| `broadcast(x, shapeOf)` | Broadcast x to match shapeOf's dimensions |
| `concat(a, b, axis)` | Concatenate along axis |
| `slice(x, axis, start, size)` | Extract slice [start, start+size) along axis |
| `sliceLastDim(x, start, size)` | Slice along last dimension (convenience) |
| `permute(x, {d0, d1, ...})` | Reorder dimensions |
| `transpose(x, dim0, dim1)` | Swap two dimensions |
| `upsample2x(x)` | 2× nearest-neighbor upsample (spatial dims) |
| `gather(table, indices)` | Embedding lookup |

### Sink Expressions (Output Transformations)

| Expression | Description |
|------------|-------------|
| `bufferSink()` | Write directly to output buffer |
| `permute(bufferSink(), {dims})` | Permute as writing |

---

## Usage

```cpp
#include "clip-encoder.h"
#include "unet.h"
#include "vae-decoder.h"
#include "ddim-sampler.h"

// Initialize
auto ctx = InferencingContext::create(device);

// Load models
auto clip = new CLIPTextEncoder(ctx, 49408, 768, 12, 12);
clip->loadParams(clipReader, "text_model.");

auto unet = new SDUNet(ctx, 4, 320, 768);
unet->loadParams(unetReader, "");

auto vae = new VAEDecoder(ctx, 4, 128);
vae->loadParams(vaeReader, "decoder.");

// Text encoding
InferencingTask clipTask(ctx);
clip->queueExecute(clipTask, textEmbeddings, tokenIds, seqLen, batchSize);
clipTask.execute();

// Diffusion loop (50 steps)
DDIMSampler sampler(1000, 50);
auto latent = /* random noise */;

for (int i = 0; i < 50; i++) {
    int timestep = sampler.getTimestep(i);
    
    InferencingTask unetTask(ctx);
    unet->queueExecute(unetTask, noisePred, latent, textEmbeddings, timestep);
    unetTask.execute();
    
    sampler.step(latent, noisePred, i);
}

// VAE decode
InferencingTask vaeTask(ctx);
vae->queueExecute(vaeTask, image, latent);
vaeTask.execute();
```

---

## Files

| File | Description |
|------|-------------|
| `clip-encoder.h/cpp` | CLIP ViT-L/14 text encoder |
| `clip-encoder-test.cpp` | CLIP unit tests |
| `unet.h/cpp` | UNet noise predictor with all fusions |
| `unet-test.cpp` | UNet unit tests |
| `vae-decoder.h/cpp` | VAE image decoder |
| `vae-decoder-test.cpp` | VAE unit tests |

---

## Conclusion

The IExpr system transforms kernel fusion from a tedious, combinatorial problem into a composable, declarative one. By expressing transformations as trees of operations, we:

1. **Eliminate boilerplate** - No custom fused kernel implementations
2. **Enable composition** - Complex fusions from simple building blocks
3. **Preserve flexibility** - Add new operations without modifying kernels
4. **Maximize performance** - Minimize memory bandwidth and kernel launches

For Stable Diffusion 1.5, this approach eliminated **100% of standalone element-wise, concat, and permute kernels**, reducing kernel launches by ~50% and saving hundreds of megabytes of memory bandwidth per UNet forward pass.

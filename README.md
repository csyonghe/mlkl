# (Unofficial) Slang Machine Learning Kernel Library

This repository hosts a collection of machine learning kernels implemented in Slang,
a shading language designed for high-performance computing on GPUs.
These kernels can be used for various machine learning inferencing tasks.

**Disclaimer**: This work is exploratory, and is not a part of the Slang project. Use at your own risk.

## File Structure

- `/src/mlkl.slang` is the main Slang module file containing the implementations of various machine learning kernels.
- `/src/kernels.h` is a C++ header file that provides declarations for the Slang kernels, allowing them to be called from C++ code. The C++ host logic
   for launching and managing these kernels is built on top of `slang-rhi`, a graphics hardware abstraction layer that tightly integrates the Slang
   shading language for GPU programming.
- Each kernel implementation is separated into three files, `<kernel-name>.slang`, `<kernel-name>.h` and `<kernel-name>.cpp`, all located in the `/src/`
  directory. The `.slang` file contains the Slang code for the kernel, the `.h` file contains the C++ declarations, and the `.cpp` file contains the C++ host logic for launching the kernel.

## Supported Kernels

### Linear Algebra

| Kernel | Description |
|--------|-------------|
| **LinearKernel** | Fully-connected layer: `Out = In @ W^T + Bias`. Supports tiled matrix multiplication with configurable tile sizes. Input: `[Batch, InDim]`, Output: `[Batch, OutDim]`. |
| **BatchGemmKernel** | Batched matrix multiplication: `alpha * A[i] @ B[i] + beta * C[i]`. Supports transpose flags for A and B. Input: `[Batch, M, K]` and `[Batch, K, N]`, Output: `[Batch, M, N]`. |

### Convolution

| Kernel | Description |
|--------|-------------|
| **Conv2DKernel** | 2D convolution with configurable kernel size, stride, and padding. Supports BatchNorm fusion. Layout: NHWC `[Batch, Height, Width, Channels]`. |
| **TransposedConv2DKernel** | Transposed 2D convolution (deconvolution) for upsampling. Layout: NHWC `[Batch, Height, Width, Channels]`. |

### Normalization

| Kernel | Description |
|--------|-------------|
| **GroupNormKernel** | Group Normalization for NHWC tensors. Normalizes across `(H, W, C/G)` for each `(batch, group)`, then applies learnable scale (gamma) and bias (beta). |
| **LayerNormKernel** | Layer Normalization. Normalizes across the last dimension (features) for each row, with learnable gamma and beta. Input: `[NumRows, NumFeatures]`. |
| **RMSNormKernel** | Root Mean Square Normalization (used in LLaMA, Gemma, Mistral). Normalizes by RMS without mean subtraction. Only gamma parameter (no beta). Input: `[NumRows, NumFeatures]`. |
| **SoftmaxKernel** | Row-wise softmax. Input: `[Rows, Cols]`, Output: `[Rows, Cols]` where each row sums to 1.0. |

### Attention

| Kernel | Description |
|--------|-------------|
| **FlashAttentionKernel** | Flash Attention 2 implementation: `softmax(Q @ K^T / sqrt(d)) @ V`. Memory-efficient with O(1) memory overhead. Supports fused input/output permutations. Input: `[B, H, S, D]` (planar layout). |
| **CrossAttentionKernel** | Complete cross-attention module with Q/K/V projections, Flash Attention core, and output projection with residual connection. |

### Reduction

| Kernel | Description |
|--------|-------------|
| **ReduceKernel** | Generic parallel reduction computing `(sum, sumSq)` per group. Supports multiple layouts: `LastDimLayout` (2D row reduction), `GroupNormLayout` (NHWC group reduction), `AxisLayout` (N-D single axis reduction up to 8 dimensions). Uses wave intrinsics for efficiency. |

### Tensor Operations

| Kernel | Description |
|--------|-------------|
| **ElementwiseKernel** | Fused elementwise operations via expression tree composition. Supports activations (ReLU, Sigmoid, Tanh, SiLU, GELU), arithmetic (+, -, *, /), and composites (clamp, lerp, leakyRelu). |
| **ConcatKernel** | Concatenate multiple tensors along a specified axis. |
| **PermuteKernel** | Reorder tensor dimensions (transpose). |
| **BroadcastAddKernel** | Element-wise addition with NumPy-style broadcasting. |
| **GatherKernel** | Embedding lookup / gather operation. Maps integer indices to embedding vectors. |

### Embeddings

| Kernel | Description |
|--------|-------------|
| **TimeEmbedingKernel** | Sinusoidal positional embedding for diffusion models. Generates time step embeddings with configurable dimension. |

### Diffusion Model Utilities

| Kernel | Description |
|--------|-------------|
| **ClassifierFreeGuidanceKernel** | Classifier-free guidance blending for diffusion models. Combines conditional and unconditional predictions. |

### Data Types

All kernels support multiple precision modes:
- **Float32** - Single precision (default)
- **Float16** - Half precision for memory efficiency
- **Int32** - Integer operations where applicable

## Kernel Fusion

### Philosophy

Kernel fusion in this library is **explicitly controlled by the user**. Rather than relying on compiler magic or runtime JIT compilation, users specify exactly what operations should be fused into the prologue and epilogue of each "big kernel" (convolution, linear, attention, etc.).

This is achieved through two complementary mechanisms:
- **`Expr` (IExpr)** - Defines input transformations (prologue) that run before the main kernel computation
- **`SinkExpr` (ISink)** - Defines output transformations (epilogue) that run after the main kernel computation

**Key benefit**: All fusion is specified in **100% C++** - you never need to write or modify GPU shader code.

### How It Works

When you create a kernel, you can optionally provide `Expr` and `SinkExpr` arguments to define fused operations:

```cpp
// Without fusion: just a convolution
Conv2DKernel conv(ctx, tileSize, kernelSize, stride, inCh, outCh, kernelOutput());

// With epilogue fusion: convolution + ReLU in a single kernel launch
Conv2DKernel convRelu(ctx, tileSize, kernelSize, stride, inCh, outCh, relu(kernelOutput()));

// With prologue fusion: input permutation + linear projection
LinearKernel linear(ctx, permute(buffer(), {1, 0}), kernelOutput(), bufferSink(), inDim, outDim);

// With both: permuted input -> linear -> ReLU -> permuted output
LinearKernel fused(ctx, 
    permute(buffer(), {1, 0}),           // Prologue: transpose input
    relu(kernelOutput()),                 // Epilogue: apply ReLU
    permute(bufferSink(), {1, 0}),       // Output: transpose result
    inDim, outDim);
```

The fused operations execute within the same GPU kernel launch - no intermediate buffers, no extra memory traffic.

### Expression Types

#### Input Expressions (`Expr`)

These define how data flows into the kernel:

| Expression | Description |
|------------|-------------|
| `buffer()` | Read from an input buffer |
| `constant(v)` | Static compile-time constant value |
| `uniformConstant()` | Runtime constant provided via parameters |
| `kernelOutput()` | The raw output of the kernel (for epilogue fusion) |

#### Tensor Operations

| Expression | Description |
|------------|-------------|
| `permute(expr, dims)` | Reorder dimensions (e.g., NHWC ↔ NCHW) |
| `transpose(expr, d0, d1)` | Swap two dimensions |
| `broadcast(expr, shapeOf)` | Broadcast to match another tensor's shape |
| `gather(table, indices)` | Index into a lookup table |
| `concat(left, right, axis)` | Concatenate along an axis |

#### Math Operations

| Expression | Description |
|------------|-------------|
| `expr + expr`, `expr - expr` | Addition, subtraction |
| `expr * expr`, `expr / expr` | Multiplication, division |
| `-expr` | Negation |
| `min(a, b)`, `max(a, b)` | Element-wise min/max |
| `exp(x)`, `log(x)` | Exponential, natural log |
| `sin(x)`, `cos(x)` | Trigonometric functions |
| `sqrt(x)`, `rsqrt(x)` | Square root, reciprocal square root |
| `pow(base, exp)` | Power function |
| `abs(x)` | Absolute value |
| `floor(x)`, `ceil(x)` | Rounding functions |

#### Activation Functions

| Expression | Description |
|------------|-------------|
| `relu(x)` | ReLU: `max(0, x)` |
| `sigmoid(x)` | Sigmoid: `1 / (1 + exp(-x))` |
| `tanh(x)` | Hyperbolic tangent |
| `silu(x)` | SiLU/Swish: `x * sigmoid(x)` |
| `gelu(x)` | GELU: `0.5 * x * (1 + erf(x / sqrt(2)))` |
| `leakyRelu(x, alpha)` | Leaky ReLU: `max(x, x * alpha)` |
| `clamp(x, min, max)` | Clamp to range |
| `lerp(a, b, t)` | Linear interpolation |

#### Output Sink Expressions (`SinkExpr`)

These define how data is written to the output buffer:

| Expression | Description |
|------------|-------------|
| `bufferSink()` | Write directly to output buffer |
| `permute(sink, dims)` | Permute dimensions on write |
| `partition(sink, dim, n)` | Split output into n partitions along dimension |

### ElementwiseKernel

For purely elementwise operations without a "big kernel" core, use `ElementwiseKernel`:

```cpp
// Create a fused elementwise operation
Expr x = buffer();
Expr result = silu(x * 0.5f + x);  // Custom activation
ElementwiseKernel kernel(ctx, result);

// Execute
kernel.queueExecute(task, output, input);
```

Multiple inputs are supported:

```cpp
Expr a = buffer();  // First input
Expr b = buffer();  // Second input  
Expr result = a * b + sqrt(abs(a - b));
ElementwiseKernel kernel(ctx, result);

// Execute with multiple inputs
EvalContext evalCtx;
evalCtx.inputs.add(a.node, InputInfo{tensorA});
evalCtx.inputs.add(b.node, InputInfo{tensorB});
kernel.queueExecute(task, evalCtx, output);
```

### Real-World Example: Flash Attention with Fused Permutations

Flash Attention expects planar layout `[B, H, S, D]` but your data might be in interleaved format `[B, S, H, D]`. Instead of launching separate permutation kernels, fuse the permutations:

```cpp
// Input expressions: permute interleaved -> planar
Expr exprQ = permute(buffer(), {0, 2, 1, 3});  // [B,S,H,D] -> [B,H,S,D]
Expr exprK = permute(buffer(), {0, 2, 1, 3});
Expr exprV = permute(buffer(), {0, 2, 1, 3});

// Output sink: permute planar -> interleaved on write
SinkExpr sink = permute(bufferSink(), {0, 2, 1, 3});  // [B,H,S,D] -> [B,S,H,D]

FlashAttentionKernel attn(ctx, exprQ, exprK, exprV, kernelOutput(), 
                          tileR, tileC, headDim, sink);
```

This executes the permutations "for free" within the attention kernel - no extra memory bandwidth.

## Building the Library

To build the Slang Machine Learning Kernel Library, follow these steps:
1. Clone the repository.
1. Run `setup.bat` or `setup.sh` to initiate the environment setup. This only needs to be done once.
1. Run `build.bat` or `build.sh` to compile the library.

## Testing Your Build

After building the library, you can run the provided test cases to verify that everything is working correctly. Use the command:

```
<build_diretory>/unit-test.exe
```

You should see `All tests passed!` if everything is functioning as expected.

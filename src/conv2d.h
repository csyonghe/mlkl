#pragma once

#include "batch-gemm.h"
#include "elementwise.h"
#include "kernel-base.h"

class SafeTensorsReader;

// Convolution algorithm selection
enum class ConvolutionAlgorithm
{
    Auto,           // Automatically select based on input size (uses Gemm)
    Flat,           // Flat kernel (good for small spatial sizes)
    Tiled,          // Tiled kernel (good for large spatial sizes)
    Gemm,           // GEMM-style tiled conv (caches both weights and input in shared mem)
    GemmWaveShuffle,// GEMM with wave shuffle (weights via warp shuffle, less shared mem)
    Winograd,       // Winograd F(4x4, 3x3) for 3x3 stride=1 convolutions
};

// GEMM convolution tile configuration for tuning/experimentation
struct GemmTileConfig
{
    int tileOH = 16;      // Output spatial tile height
    int tileOW = 16;      // Output spatial tile width
    int tileOC = 16;      // Output channels per block
    int tileIC = 8;       // Input channels per K-iteration
    int threadOH = 1;     // Spatial rows per thread (register blocking)
    int threadOW = 1;     // Spatial cols per thread (register blocking)

    // Default configuration (current best for general case)
    static GemmTileConfig defaultConfig() { return GemmTileConfig{}; }

    // Configuration name for logging
    String getName() const
    {
        StringBuilder sb;
        sb << "OH" << tileOH << "_OW" << tileOW << "_OC" << tileOC << "_IC" << tileIC;
        return sb.produceString();
    }

    // Compute thread block size
    int getBlockSize() const { return (tileOW / threadOW) * (tileOH / threadOH); }
};

// 2D Convolution Kernel
// - Weights layout: [InChannels, KernelSize, KernelSize, OutChannels]
// - Input/Output layout: [BatchSize, Height, Width, Channels]
//
// CONSTRUCTORS:
// 1. Simple (Float32, buffer input, custom OUTPUT transformation):
//    Conv2DKernel(ctx, tileSize, kernelSize, stride, inCh, outCh, outputExpr)
//    - outputExpr: transformation applied to conv output (e.g., kernelOutput(), clamp(...))
//    - DO NOT pass buffer() or silu(buffer()) here! Those are INPUT expressions.
//
// 2. Full (custom INPUT/output expressions):
//    Conv2DKernel(ctx, elemType, tileSize, kernelSize, stride, inCh, outCh,
//                 inputExpr, outputExpr, sinkExpr)
//    - inputExpr: transformation applied to input (e.g., silu(buffer()), upsample2x(buffer()))
//    - outputExpr: transformation applied to output (e.g., kernelOutput(), clamp(kernelOutput(), ...))
//
// QUEUEEXECUTE:
//   queueExecute(task, output, input, padding)
//   - padding: Use 0 for 1x1 conv, 1 for 3x3 conv (to maintain spatial dims)
//
// FUSION OPPORTUNITIES:
// - Fuse SiLU/ReLU into inputExpr (FULL constructor):
//     Conv2DKernel(ctx, Float32, 16, 3, 1, inCh, outCh, silu(buffer()), kernelOutput(), bufferSink())
// - Fuse clamp into outputExpr:
//     Conv2DKernel(ctx, Float32, 16, 3, 1, inCh, outCh, buffer(), clamp(kernelOutput(),-1,1), bufferSink())
// - Fuse upsample into inputExpr (FULL constructor):
//     Conv2DKernel(ctx, Float32, 16, 3, 1, inCh, outCh, upsample2x(buffer()), kernelOutput(), bufferSink())
//
// COMMON MISTAKES:
// - Forgetting padding argument in queueExecute
// - Using wrong constructor (passing silu(buffer()) to simple constructor - use FULL constructor!)
// - Confusing inputExpr vs outputExpr parameter position
class Conv2DKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> tilePipeline;
    ComPtr<rhi::IComputePipeline> flatPipeline;
    ComPtr<rhi::IComputePipeline> flatWaveReducePipeline;
    ComPtr<rhi::IComputePipeline> gemmPipeline;
    ComPtr<rhi::IComputePipeline> gemmSmallSpatialPipeline; // For small outputs (<=8x8)
    ComPtr<rhi::IComputePipeline> gemmWaveShufflePipeline;  // Wave shuffle version (no weight shared mem)
    ComPtr<rhi::IComputePipeline> winogradPipeline;         // Winograd F(4x4, 3x3)

    InferencingContext* context;
    ElementType elementType;
    ProgramNode inputProgram;
    ProgramNode outputProgram;
    SinkExpr sinkExpr;
    GemmTileConfig gemmConfig;              // Tile configuration used for gemmPipeline
    GemmTileConfig gemmSmallSpatialConfig;  // Config for small spatial outputs

    // Winograd-transformed weights buffer [OC, IC, 6, 6] for F(4x4, 3x3)
    ComPtr<rhi::IBuffer> winogradWeightsBuffer;

    void validateTensorElementType(const TensorView& tv, const char* name) const;
    ComPtr<rhi::IComputePipeline> createGemmPipelineWithConfig(const GemmTileConfig& config);
    ComPtr<rhi::IComputePipeline> createGemmWaveShufflePipeline(const GemmTileConfig& config);
    void createWinogradPipeline();
    void transformWeightsToWinograd();

public:
    int tileSize;
    int kernelSize;
    int inChannels;
    int outChannels;
    int stride;

    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;

    // [outChannels, kernelSize, kernelSize, inChannels]
    ComPtr<rhi::IBuffer> weightsTransposedBuffer;

    String name;

    Conv2DKernel(
        InferencingContext* context,
        ElementType elementType,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        String name = "conv2d");

    // Convenience constructor defaulting to Float32.
    Conv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        String name = "conv2d")
        : Conv2DKernel(
              context,
              ElementType::Float32,
              tileSize,
              kernelSize,
              stride,
              inChannels,
              outChannels,
              inputExpr,
              outputExpr,
              sinkExpr,
              name)
    {
    }

    Conv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        Expr outputExpr = kernelOutput(),
        String name = "conv2d");

    // Constructor with custom GEMM tile configuration (for benchmarking/tuning)
    Conv2DKernel(
        InferencingContext* context,
        int tileSize,
        int kernelSize,
        int stride,
        int inChannels,
        int outChannels,
        const GemmTileConfig& gemmTileConfig,
        String name = "conv2d");

    ElementType getElementType() const { return elementType; }
    const GemmTileConfig& getGemmConfig() const { return gemmConfig; }

    SlangResult loadParams(TorchParamReader& reader, bool loadAndFuseBNorm);

    // Load from SafeTensors - weights are transposed from [OutCh, InCh, K, K] to [InCh, K, K, OutCh]
    SlangResult loadParams(
        SafeTensorsReader& reader,
        UnownedStringSlice weightName,
        UnownedStringSlice biasName);

    SlangResult loadParams(
        int kernelSize,
        int outputChannelCount,
        float* weightsData,
        float* biasesData);

    TensorView allocateResultBuffer(
        ElementType elementType,
        int inputWidth,
        int inputHeight,
        int padding,
        int batchSize);

    void queueExecute(
        InferencingTask& task,
        EvalContext& evalCtx,
        TensorView output,
        int padding,
        ConvolutionAlgorithm algorithm = ConvolutionAlgorithm::Auto);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs,
        int padding,
        ConvolutionAlgorithm algorithm = ConvolutionAlgorithm::Auto)
    {
        EvalContext ctx = makeEvalContext(inputs);
        return queueExecute(task, ctx, output, padding, algorithm);
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs,
        int padding,
        ConvolutionAlgorithm algorithm = ConvolutionAlgorithm::Auto);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView inputImage,
        int padding,
        ConvolutionAlgorithm algorithm = ConvolutionAlgorithm::Auto);

private:
    // GEMM-style tiled convolution (called when algorithm == Gemm)
    void executeGemmConv(
        InferencingTask& task,
        EvalContext& ctx,
        TensorView output,
        int padding);

    // GEMM with wave shuffle (called when algorithm == GemmWaveShuffle)
    void executeGemmWaveShuffleConv(
        InferencingTask& task,
        EvalContext& ctx,
        TensorView output,
        int padding);

    // Winograd F(4x4, 3x3) convolution (called when algorithm == Winograd)
    void executeWinogradConv(
        InferencingTask& task,
        EvalContext& ctx,
        TensorView output,
        int padding);
};
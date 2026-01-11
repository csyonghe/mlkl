// Convolution Benchmark for Stable Diffusion-relevant configurations
//
// Tests gemmConvolution kernel performance across:
// - Spatial sizes: 64x64, 32x32, 16x16, 8x8 (SD 1.5 latent sizes)
// - Channel configurations: 320, 640, 1280 (SD 1.5 UNet channels)
// - Stride 1 (regular convolutions) and stride 2 (downsampling)
//
// Build: Add to CMakeLists.txt as a separate executable
// Run: bench-conv2d.exe

#include "core/slang-basic.h"
#include "example-base/example-base.h"
#include "inference-context.h"
#include "kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

using namespace Slang;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct ConvConfig
{
    const char* name;
    int inputH;
    int inputW;
    int inChannels;
    int outChannels;
    int kernelSize;
    int stride;
    int padding;
};

// SD 1.5 UNet convolution configurations
// These represent the most performance-critical convolutions in the model
static const ConvConfig kBenchConfigs[] = {
    // ========================================
    // STRIDE 1: Regular convolutions
    // ========================================

    // 64x64 spatial (initial latent size after VAE encode)
    {"64x64_4->320_s1", 64, 64, 4, 320, 3, 1, 1},     // Initial conv_in
    {"64x64_320->320_s1", 64, 64, 320, 320, 3, 1, 1}, // Down block 0 resblocks

    // 32x32 spatial (after first downsample)
    {"32x32_320->320_s1", 32, 32, 320, 320, 3, 1, 1}, // Down block 0 output
    {"32x32_320->640_s1", 32, 32, 320, 640, 3, 1, 1}, // Down block 1 first conv
    {"32x32_640->640_s1", 32, 32, 640, 640, 3, 1, 1}, // Down block 1 resblocks

    // 16x16 spatial (after second downsample)
    {"16x16_640->640_s1", 16, 16, 640, 640, 3, 1, 1},     // Down block 1 output
    {"16x16_640->1280_s1", 16, 16, 640, 1280, 3, 1, 1},   // Down block 2 first conv
    {"16x16_1280->1280_s1", 16, 16, 1280, 1280, 3, 1, 1}, // Down block 2 resblocks

    // 8x8 spatial (bottleneck)
    {"8x8_1280->1280_s1", 8, 8, 1280, 1280, 3, 1, 1}, // Bottleneck resblocks

    // ========================================
    // STRIDE 2: Downsampling convolutions
    // ========================================

    // Downsample 64x64 -> 32x32
    {"64x64->32x32_320_s2", 64, 64, 320, 320, 3, 2, 1},

    // Downsample 32x32 -> 16x16
    {"32x32->16x16_640_s2", 32, 32, 640, 640, 3, 2, 1},

    // Downsample 16x16 -> 8x8
    {"16x16->8x8_1280_s2", 16, 16, 1280, 1280, 3, 2, 1},

    // ========================================
    // UPSAMPLING PATH (stride 1, after concat)
    // ========================================

    // Up block convolutions (input channels doubled due to skip connection)
    {"8x8_2560->1280_s1", 8, 8, 2560, 1280, 3, 1, 1}, // Up block after concat
    {"16x16_2560->1280_s1", 16, 16, 2560, 1280, 3, 1, 1},
    {"16x16_1920->640_s1", 16, 16, 1920, 640, 3, 1, 1},
    {"32x32_1280->640_s1", 32, 32, 1280, 640, 3, 1, 1},
    {"32x32_960->320_s1", 32, 32, 960, 320, 3, 1, 1},
    {"64x64_640->320_s1", 64, 64, 640, 320, 3, 1, 1},
};

// ============================================================================
// Timing Helpers
// ============================================================================

struct BenchResult
{
    const char* name;
    double avgMs;
    double minMs;
    double maxMs;
    double stddevMs;
    int numCalls;
    double gflops; // Effective GFLOPS
};

// Calculate FLOPS for a conv2d operation
// FLOPS = 2 * outputH * outputW * outChannels * inChannels * kernelH * kernelW
double calcConvGFLOPS(const ConvConfig& cfg, double timeMs)
{
    int outputH = (cfg.inputH + 2 * cfg.padding - cfg.kernelSize) / cfg.stride + 1;
    int outputW = (cfg.inputW + 2 * cfg.padding - cfg.kernelSize) / cfg.stride + 1;

    // MACs = output_elements * kernel_elements * input_channels
    // FLOPS = 2 * MACs (multiply + add)
    double macs = (double)outputH * outputW * cfg.outChannels * cfg.inChannels * cfg.kernelSize *
                  cfg.kernelSize;
    double flops = 2.0 * macs;

    // GFLOPS = FLOPS / (time_in_ms * 1e6)
    return flops / (timeMs * 1e6);
}

// ============================================================================
// Benchmark Runner
// ============================================================================

// Pre-created kernel with its associated buffers for accurate benchmarking
struct PreparedKernel
{
    RefPtr<Conv2DKernel> kernel;
    RefPtr<Tensor> inputBuffer;
    TensorView outputBuffer;
    int padding;
    const ConvConfig* cfg;
};

// Kernel input expression type (to match real SD usage)
enum class ConvInputExpr
{
    Buffer,     // buffer() - simple input
    SiLU,       // silu(buffer()) - most ResNet convolutions in SD
    Upsample2x, // upsample2x(buffer()) - upsample layers
};

class ConvBenchmark
{
public:
    InferencingContext* ctx;
    int warmupRuns = 10;  // Increased warmup
    int benchRuns = 50;   // More runs for accuracy
    int batchSize = 2;    // Match SD CFG batch size

    ConvBenchmark(InferencingContext* context)
        : ctx(context)
    {
    }

    // Pre-create kernel and buffers for a config
    PreparedKernel prepareKernel(const ConvConfig& cfg, ConvInputExpr inputExprType = ConvInputExpr::Buffer)
    {
        int outputH = (cfg.inputH + 2 * cfg.padding - cfg.kernelSize) / cfg.stride + 1;
        int outputW = (cfg.inputW + 2 * cfg.padding - cfg.kernelSize) / cfg.stride + 1;

        // Create random input data
        List<float> inputData;
        int inputSize = batchSize * cfg.inputH * cfg.inputW * cfg.inChannels;
        inputData.setCount(inputSize);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < inputSize; i++)
            inputData[i] = dist(rng);

        auto inputBuffer = ctx->createTensor(
            ElementType::Float32,
            Shape(batchSize, cfg.inputH, cfg.inputW, cfg.inChannels),
            inputData);

        // Create random weights and biases
        int weightsSize = cfg.outChannels * cfg.kernelSize * cfg.kernelSize * cfg.inChannels;
        List<float> weights;
        weights.setCount(weightsSize);
        for (int i = 0; i < weightsSize; i++)
            weights[i] = dist(rng) * 0.1f;

        List<float> biases;
        biases.setCount(cfg.outChannels);
        for (int i = 0; i < cfg.outChannels; i++)
            biases[i] = dist(rng) * 0.01f;

        // Create kernel with appropriate input expression
        Conv2DKernel* kernel = nullptr;
        switch (inputExprType)
        {
        case ConvInputExpr::SiLU:
            // Most common in SD ResNet blocks: SiLU(input) → Conv
            kernel = new Conv2DKernel(ctx, ElementType::Float32, 16, cfg.kernelSize, cfg.stride,
                                      cfg.inChannels, cfg.outChannels,
                                      silu(buffer()), kernelOutput(), bufferSink());
            break;
        case ConvInputExpr::Upsample2x:
            // Upsample layers in SD
            kernel = new Conv2DKernel(ctx, ElementType::Float32, 16, cfg.kernelSize, cfg.stride,
                                      cfg.inChannels, cfg.outChannels,
                                      upsample2x(buffer()), kernelOutput(), bufferSink());
            break;
        case ConvInputExpr::Buffer:
        default:
            // Simple buffer input
            kernel = new Conv2DKernel(ctx, 16, cfg.kernelSize, cfg.stride, cfg.inChannels, cfg.outChannels);
            break;
        }
        kernel->loadParams(cfg.kernelSize, cfg.outChannels, weights.getBuffer(), biases.getBuffer());

        // Allocate output
        auto outputBuffer = ctx->allocScratchTensor(
            ElementType::Float32,
            Shape(batchSize, outputH, outputW, cfg.outChannels),
            "bench_output");

        PreparedKernel result;
        result.kernel = kernel;
        result.inputBuffer = inputBuffer;
        result.outputBuffer = outputBuffer;
        result.padding = cfg.padding;
        result.cfg = &cfg;
        return result;
    }

    // Warmup a prepared kernel
    void warmupKernel(PreparedKernel& pk, ConvolutionAlgorithm algorithm)
    {
        for (int i = 0; i < warmupRuns; i++)
        {
            auto task = ctx->createTask();
            pk.kernel->queueExecute(task, pk.outputBuffer, pk.inputBuffer->getView(), pk.padding, algorithm);
            task.execute();
        }
    }

    // Run benchmark on pre-warmed kernel
    BenchResult runBenchmark(PreparedKernel& pk, ConvolutionAlgorithm algorithm)
    {
        // Reset timing
        ctx->resetPerfMeasurements();

        // Benchmark runs
        for (int i = 0; i < benchRuns; i++)
        {
            auto task = ctx->createTask();
            pk.kernel->queueExecute(task, pk.outputBuffer, pk.inputBuffer->getView(), pk.padding, algorithm);
            task.execute();
        }

        // Get timing results
        auto perfEntries = ctx->getPerfMeasurements();

        // Find the gemmConvolution entry (or whichever kernel was used)
        double totalMs = 0;
        int callCount = 0;
        for (const auto& entry : perfEntries)
        {
            // Match any convolution kernel
            if (strstr(entry.name.getBuffer(), "Convolution") ||
                strstr(entry.name.getBuffer(), "convolution"))
            {
                totalMs = entry.totalTimeMs;
                callCount = entry.callCount;
                break;
            }
        }

        double avgMs = callCount > 0 ? totalMs / callCount : 0;

        BenchResult result;
        result.name = pk.cfg->name;
        result.avgMs = avgMs;
        result.minMs = avgMs;
        result.maxMs = avgMs;
        result.stddevMs = 0;
        result.numCalls = callCount;
        result.gflops = calcConvGFLOPS(*pk.cfg, avgMs);

        return result;
    }

    // Legacy interface for algorithm comparison section
    BenchResult runBenchmark(const ConvConfig& cfg, ConvolutionAlgorithm algorithm)
    {
        PreparedKernel pk = prepareKernel(cfg);
        warmupKernel(pk, algorithm);
        return runBenchmark(pk, algorithm);
    }
};

// ============================================================================
// Main Program
// ============================================================================

// Forward declaration for external linkage
int runConvBenchmark(int argc, char* argv[]);

struct ConvBenchProgram : public TestBase
{
    ComPtr<rhi::IDevice> gDevice;
    RefPtr<InferencingContext> gInferencingCtx;

    void printHeader()
    {
        printf("\n");
        printf(
            "================================================================================\n");
        printf("                    CONVOLUTION BENCHMARK - SD 1.5 Configurations\n");
        printf(
            "================================================================================\n");
        printf("\n");
    }

    void printTableHeader()
    {
        printf(
            "%-28s %8s %8s %8s %10s %10s\n",
            "Configuration",
            "InCh",
            "OutCh",
            "Stride",
            "Time(ms)",
            "GFLOPS");
        printf(
            "--------------------------------------------------------------------------------\n");
    }

    void printResult(const ConvConfig& cfg, const BenchResult& result)
    {
        printf(
            "%-28s %8d %8d %8d %10.3f %10.1f\n",
            result.name,
            cfg.inChannels,
            cfg.outChannels,
            cfg.stride,
            result.avgMs,
            result.gflops);
    }

    SlangResult execute(int argc, char* argv[])
    {
        parseOption(argc, argv);

        printHeader();

        // Initialize device
        rhi::DeviceDesc deviceDesc;
        deviceDesc.slang.targetProfile = "spirv_1_6";
        List<slang::CompilerOptionEntry> compilerOptionsEntries;
        const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
        for (auto cap : capabilities)
        {
            slang::CompilerOptionEntry entry;
            entry.name = slang::CompilerOptionName::Capability;
            slang::CompilerOptionValue value;
            value.kind = slang::CompilerOptionValueKind::String;
            value.stringValue0 = cap;
            entry.value = value;
            compilerOptionsEntries.add(entry);
        }
        deviceDesc.slang.compilerOptionEntries = compilerOptionsEntries.getBuffer();
        deviceDesc.slang.compilerOptionEntryCount = (uint32_t)compilerOptionsEntries.getCount();
        deviceDesc.deviceType = rhi::DeviceType::Vulkan;

        // Match unit-test device init:
        // - Add executable directory as a shader search path so `*.slang` files copied next to the exe are found.
        // - Enable persistent cache to reduce shader compile overhead during benchmark setup.
        String exePath = Path::getParentDirectory(Path::getExecutablePath()).getBuffer();
        const char* exePathStr = exePath.getBuffer();
        deviceDesc.slang.searchPaths = &exePathStr;
        deviceDesc.slang.searchPathCount = 1;

        ComPtr<rhi::IPersistentCache> shaderCache =
            ComPtr<rhi::IPersistentCache>(new FileShaderCache());
        deviceDesc.persistentShaderCache = shaderCache.get();

        gDevice = rhi::getRHI()->createDevice(deviceDesc);
        if (!gDevice)
        {
            printf("ERROR: Failed to create GPU device\n");
            return SLANG_FAIL;
        }

        gInferencingCtx = new InferencingContext(gDevice);
        gInferencingCtx->setCollectPerfMeasurements(true);

        ConvBenchmark bench(gInferencingCtx);

        // ============================================
        // Phase 1: Pre-create and warmup ALL kernels
        // ============================================
        printf("Preparing kernels...\n");
        List<PreparedKernel> preparedKernels;
        for (const auto& cfg : kBenchConfigs)
        {
            preparedKernels.add(bench.prepareKernel(cfg));
        }

        printf("Warming up %d kernels...\n", (int)preparedKernels.getCount());
        for (auto& pk : preparedKernels)
        {
            bench.warmupKernel(pk, ConvolutionAlgorithm::Gemm);
        }
        printf("Warmup complete.\n\n");

        // ============================================
        // Stride 1 benchmarks
        // ============================================
        printf("=== STRIDE 1 CONVOLUTIONS ===\n\n");
        printTableHeader();

        double totalStrideOneTime = 0;
        int strideOneCount = 0;

        for (auto& pk : preparedKernels)
        {
            if (pk.cfg->stride != 1)
                continue;

            BenchResult result = bench.runBenchmark(pk, ConvolutionAlgorithm::Gemm);
            printResult(*pk.cfg, result);
            totalStrideOneTime += result.avgMs;
            strideOneCount++;
        }

        printf(
            "--------------------------------------------------------------------------------\n");
        printf(
            "Stride 1 Total (per inference): %.3f ms (%d layers)\n",
            totalStrideOneTime,
            strideOneCount);

        // ============================================
        // Stride 2 benchmarks
        // ============================================
        printf("\n=== STRIDE 2 CONVOLUTIONS (Downsample) ===\n\n");
        printTableHeader();

        double totalStrideTwoTime = 0;
        int strideTwoCount = 0;

        for (auto& pk : preparedKernels)
        {
            if (pk.cfg->stride != 2)
                continue;

            BenchResult result = bench.runBenchmark(pk, ConvolutionAlgorithm::Gemm);
            printResult(*pk.cfg, result);
            totalStrideTwoTime += result.avgMs;
            strideTwoCount++;
        }

        printf(
            "--------------------------------------------------------------------------------\n");
        printf(
            "Stride 2 Total (per inference): %.3f ms (%d layers)\n",
            totalStrideTwoTime,
            strideTwoCount);

        // ============================================
        // Algorithm comparison for selected configs
        // ============================================
        printf("\n=== ALGORITHM COMPARISON (Selected Configs) ===\n\n");

        // Test a few representative configs with different algorithms
        ConvConfig testConfigs[] = {
            {"64x64_320->320_s1", 64, 64, 320, 320, 3, 1, 1},
            {"32x32_640->640_s1", 32, 32, 640, 640, 3, 1, 1},
            {"16x16_1280->1280_s1", 16, 16, 1280, 1280, 3, 1, 1},
            {"8x8_1280->1280_s1", 8, 8, 1280, 1280, 3, 1, 1},
            {"64x64->32x32_320_s2", 64, 64, 320, 320, 3, 2, 1},
        };

        printf("%-28s %12s %12s %12s\n", "Configuration", "Gemm(ms)", "Flat(ms)", "Tiled(ms)");
        printf(
            "--------------------------------------------------------------------------------\n");

        for (const auto& cfg : testConfigs)
        {
            BenchResult gemmResult = bench.runBenchmark(cfg, ConvolutionAlgorithm::Gemm);
            BenchResult flatResult = bench.runBenchmark(cfg, ConvolutionAlgorithm::Flat);
            BenchResult tiledResult = bench.runBenchmark(cfg, ConvolutionAlgorithm::Tiled);

            printf(
                "%-28s %12.3f %12.3f %12.3f\n",
                cfg.name,
                gemmResult.avgMs,
                flatResult.avgMs,
                tiledResult.avgMs);
        }

        // ============================================
        // Winograd vs GEMM for 3x3 stride=1 convolutions
        // ============================================
        printf("\n=== WINOGRAD vs GEMM (3x3 stride=1 only) ===\n\n");
        printf("Winograd F(4x4,3x3) reduces multiplications by ~2.25x for 3x3 stride=1 convs.\n\n");

        // Winograd-applicable configs (3x3, stride=1)
        ConvConfig winogradConfigs[] = {
            {"8x8_1280->1280", 8, 8, 1280, 1280, 3, 1, 1},
            {"8x8_2560->1280", 8, 8, 2560, 1280, 3, 1, 1},
            {"16x16_1280->1280", 16, 16, 1280, 1280, 3, 1, 1},
            {"16x16_2560->1280", 16, 16, 2560, 1280, 3, 1, 1},
            {"32x32_640->640", 32, 32, 640, 640, 3, 1, 1},
            {"64x64_320->320", 64, 64, 320, 320, 3, 1, 1},
        };

        printf("%-24s %12s %12s %10s\n", "Configuration", "GEMM(ms)", "Winograd(ms)", "Speedup");
        printf("--------------------------------------------------------------------------------\n");

        double totalGemmTime = 0, totalWinogradTime = 0;
        for (const auto& cfg : winogradConfigs)
        {
            BenchResult gemmResult = bench.runBenchmark(cfg, ConvolutionAlgorithm::Gemm);
            BenchResult winogradResult = bench.runBenchmark(cfg, ConvolutionAlgorithm::Winograd);

            double speedup = gemmResult.avgMs / winogradResult.avgMs;
            totalGemmTime += gemmResult.avgMs;
            totalWinogradTime += winogradResult.avgMs;

            printf("%-24s %12.3f %12.3f %9.2fx\n",
                   cfg.name,
                   gemmResult.avgMs,
                   winogradResult.avgMs,
                   speedup);
        }

        printf("--------------------------------------------------------------------------------\n");
        printf("%-24s %12.3f %12.3f %9.2fx\n",
               "TOTAL",
               totalGemmTime,
               totalWinogradTime,
               totalGemmTime / totalWinogradTime);

        // ============================================
        // Summary
        // ============================================
        printf("\n=== SUMMARY ===\n\n");
        printf("Total convolution time per inference:\n");
        printf("  Stride 1: %.3f ms\n", totalStrideOneTime);
        printf("  Stride 2: %.3f ms\n", totalStrideTwoTime);
        printf("  Combined: %.3f ms\n", totalStrideOneTime + totalStrideTwoTime);
        printf("\n");
        printf("Note: SD 1.5 has ~62 conv layers per step, 20 steps = ~1240 conv calls total\n");
        printf("\n");

        // ============================================
        // REALISTIC Benchmark: SiLU input vs Buffer input comparison
        // ============================================
        printf("=== REALISTIC BENCHMARK: SiLU vs Buffer input ===\n\n");
        printf("Comparing performance with silu(buffer()) vs plain buffer() input...\n\n");

        // Test configs matching SD layer shapes
        ConvConfig realisticConfigs[] = {
            {"8x8_1280->1280", 8, 8, 1280, 1280, 3, 1, 1},
            {"16x16_1280->1280", 16, 16, 1280, 1280, 3, 1, 1},
            {"32x32_640->640", 32, 32, 640, 640, 3, 1, 1},
            {"64x64_320->320", 64, 64, 320, 320, 3, 1, 1},
        };

        printf("%-24s %12s %12s %10s\n", "Configuration", "Buffer(ms)", "SiLU(ms)", "Overhead");
        printf("--------------------------------------------------------------------------------\n");

        for (const auto& cfg : realisticConfigs)
        {
            // Test with buffer() input
            PreparedKernel bufferKernel = bench.prepareKernel(cfg, ConvInputExpr::Buffer);
            bench.warmupKernel(bufferKernel, ConvolutionAlgorithm::Gemm);
            BenchResult bufferResult = bench.runBenchmark(bufferKernel, ConvolutionAlgorithm::Gemm);

            // Test with silu(buffer()) input
            PreparedKernel siluKernel = bench.prepareKernel(cfg, ConvInputExpr::SiLU);
            bench.warmupKernel(siluKernel, ConvolutionAlgorithm::Gemm);
            BenchResult siluResult = bench.runBenchmark(siluKernel, ConvolutionAlgorithm::Gemm);

            double overhead = siluResult.avgMs > 0 ? (siluResult.avgMs - bufferResult.avgMs) / bufferResult.avgMs * 100 : 0;

            printf("%-24s %12.3f %12.3f %9.1f%%\n",
                   cfg.name,
                   bufferResult.avgMs,
                   siluResult.avgMs,
                   overhead);
        }

        // ============================================
        // Complete SD Layer Mix (weighted by call count)
        // ============================================
        printf("\n=== FULL SD LAYER MIX (batch_size=2, 30 steps) ===\n\n");
        
        // SD profile: 2976 conv calls / 30 steps = ~99 conv layers per UNet pass
        // Each layer called once per step with batch_size=2 for CFG
        // Calls = layers_of_this_type × 30 steps
        struct LayerWeight
        {
            ConvConfig cfg;
            int callsPerInference;  // = num_layers × 30 steps
        };

        LayerWeight sdLayerMix[] = {
            // High-channel bottleneck (most expensive per call)
            // Mid block: 2 resblocks × 2 convs = 4 convs at 8x8_1280
            // Down3: 2 resblocks × 2 convs = 4 convs at 8x8_1280  
            // Up0: 3 resblocks × 2 convs = 6 convs (with skip concat: 2560->1280)
            {{"8x8_1280->1280", 8, 8, 1280, 1280, 3, 1, 1}, 8 * 30},     // mid + down3
            {{"8x8_2560->1280", 8, 8, 2560, 1280, 3, 1, 1}, 6 * 30},     // up0 with skip
            
            // 16x16 spatial (down2/up1)
            {{"16x16_1280->1280", 16, 16, 1280, 1280, 3, 1, 1}, 8 * 30}, // down2
            {{"16x16_2560->1280", 16, 16, 2560, 1280, 3, 1, 1}, 6 * 30}, // up1 with skip
            
            // 32x32 spatial (down1/up2)  
            {{"32x32_640->640", 32, 32, 640, 640, 3, 1, 1}, 8 * 30},     // down1
            {{"32x32_1280->640", 32, 32, 1280, 640, 3, 1, 1}, 6 * 30},   // up2 with skip
            
            // 64x64 spatial (down0/up3)
            {{"64x64_320->320", 64, 64, 320, 320, 3, 1, 1}, 8 * 30},     // down0
            {{"64x64_640->320", 64, 64, 640, 320, 3, 1, 1}, 6 * 30},     // up3 with skip
            
            // conv_in and conv_out (1 each per step)
            {{"64x64_4->320", 64, 64, 4, 320, 3, 1, 1}, 1 * 30},         // conv_in
            {{"64x64_320->4", 64, 64, 320, 4, 3, 1, 1}, 1 * 30},         // conv_out
            
            // Stride 2 downsamplers (1 each per level, 3 total)
            {{"64x64->32x32_320_s2", 64, 64, 320, 320, 3, 2, 1}, 1 * 30},
            {{"32x32->16x16_640_s2", 32, 32, 640, 640, 3, 2, 1}, 1 * 30},
            {{"16x16->8x8_1280_s2", 16, 16, 1280, 1280, 3, 2, 1}, 1 * 30},
            
            // Upsample convs (1 each per level, 3 total) - with nearest-neighbor fused
            {{"16x16_1280->1280_up", 8, 8, 1280, 1280, 3, 1, 1}, 1 * 30},  // 8x8 input, 16x16 output
            {{"32x32_640->640_up", 16, 16, 640, 640, 3, 1, 1}, 1 * 30},    // 16x16 input, 32x32 output
            {{"64x64_320->320_up", 32, 32, 320, 320, 3, 1, 1}, 1 * 30},    // 32x32 input, 64x64 output
        };

        printf("%-24s %8s %10s %12s %12s\n", "Configuration", "Calls", "Avg(ms)", "Total(ms)", "GFLOPS");
        printf("--------------------------------------------------------------------------------\n");

        double grandTotalMs = 0;
        int totalCalls = 0;

        for (const auto& layer : sdLayerMix)
        {
            PreparedKernel pk = bench.prepareKernel(layer.cfg, ConvInputExpr::SiLU);
            bench.warmupKernel(pk, ConvolutionAlgorithm::Gemm);
            BenchResult result = bench.runBenchmark(pk, ConvolutionAlgorithm::Gemm);

            double layerTotal = result.avgMs * layer.callsPerInference;
            grandTotalMs += layerTotal;
            totalCalls += layer.callsPerInference;

            printf("%-24s %8d %10.3f %12.2f %12.1f\n",
                   layer.cfg.name,
                   layer.callsPerInference,
                   result.avgMs,
                   layerTotal,
                   result.gflops);
        }

        printf("--------------------------------------------------------------------------------\n");
        printf("%-24s %8d %10.3f %12.2f\n",
               "TOTAL",
               totalCalls,
               grandTotalMs / totalCalls,
               grandTotalMs);
        printf("\nPredicted conv time for SD: %.1f ms\n", grandTotalMs);
        printf("Actual from SD profile:     3768 ms\n");
        printf("Ratio: %.2fx\n", 3768.0 / grandTotalMs);

        // ============================================
        // TILE CONFIG EXPLORATION (for high-channel layers)
        // ============================================
        printf("\n=== TILE CONFIG EXPLORATION (batch_size=2) ===\n\n");
        printf("Testing different TILE_OC values for high-channel bottleneck layers...\n\n");

        // Configs to explore - focus on 8x8 outputs (using our small spatial optimization)
        ConvConfig exploreConfigs[] = {
            {"8x8_1280->1280", 8, 8, 1280, 1280, 3, 1, 1},
            {"8x8_2560->1280", 8, 8, 2560, 1280, 3, 1, 1},  // up0 with skip
        };

        // Tile configs to test
        struct TileVariant
        {
            const char* name;
            GemmTileConfig config;
        };

        TileVariant tileVariants[] = {
            // Current best for 8x8: 8x8 spatial + OC8 + IC8
            {"8x8_IC8_OC8", []() { GemmTileConfig c; c.tileOH = 8; c.tileOW = 8; c.tileOC = 8; return c; }()},
            // Test IC4 with 8x8 spatial
            {"8x8_IC4_OC8", []() { GemmTileConfig c; c.tileOH = 8; c.tileOW = 8; c.tileIC = 4; c.tileOC = 8; return c; }()},
            // Default for comparison
            {"16x16_IC8_OC16", GemmTileConfig::defaultConfig()},
        };

        printf("%-20s", "Configuration");
        for (const auto& tv : tileVariants)
        {
            printf(" %12s", tv.name);
        }
        printf("     Best\n");
        printf("--------------------------------------------------------------------------------");
        printf("------------------\n");

        for (const auto& cfg : exploreConfigs)
        {
            int outputH = (cfg.inputH + 2 * cfg.padding - cfg.kernelSize) / cfg.stride + 1;
            int outputW = (cfg.inputW + 2 * cfg.padding - cfg.kernelSize) / cfg.stride + 1;

            printf("%-20s", cfg.name);

            double bestTime = 1e9;
            const char* bestName = "";

            for (const auto& tv : tileVariants)
            {
                // Create random weights/biases
                int weightsSize = cfg.outChannels * cfg.kernelSize * cfg.kernelSize * cfg.inChannels;
                List<float> weights;
                weights.setCount(weightsSize);
                std::mt19937 rng(42);
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (int i = 0; i < weightsSize; i++)
                    weights[i] = dist(rng) * 0.1f;

                List<float> biases;
                biases.setCount(cfg.outChannels);
                for (int i = 0; i < cfg.outChannels; i++)
                    biases[i] = dist(rng) * 0.01f;

                // Create kernel with specific tile config
                Conv2DKernel kernel(
                    gInferencingCtx,
                    16,  // tileSize
                    cfg.kernelSize,
                    cfg.stride,
                    cfg.inChannels,
                    cfg.outChannels,
                    tv.config,
                    cfg.name);
                kernel.loadParams(cfg.kernelSize, cfg.outChannels, weights.getBuffer(), biases.getBuffer());

                // Create input/output buffers
                List<float> inputData;
                int inputSize = bench.batchSize * cfg.inputH * cfg.inputW * cfg.inChannels;
                inputData.setCount(inputSize);
                for (int i = 0; i < inputSize; i++)
                    inputData[i] = dist(rng);

                auto inputBuffer = gInferencingCtx->createTensor(
                    ElementType::Float32,
                    Shape(bench.batchSize, cfg.inputH, cfg.inputW, cfg.inChannels),
                    inputData);

                auto outputBuffer = gInferencingCtx->allocScratchTensor(
                    ElementType::Float32,
                    Shape(bench.batchSize, outputH, outputW, cfg.outChannels),
                    "explore_output");

                // Warmup
                for (int i = 0; i < 20; i++)
                {
                    auto task = gInferencingCtx->createTask();
                    kernel.queueExecute(task, outputBuffer, inputBuffer->getView(), cfg.padding, ConvolutionAlgorithm::Gemm);
                    task.execute();
                }

                // Benchmark
                gInferencingCtx->resetPerfMeasurements();
                for (int i = 0; i < 50; i++)
                {
                    auto task = gInferencingCtx->createTask();
                    kernel.queueExecute(task, outputBuffer, inputBuffer->getView(), cfg.padding, ConvolutionAlgorithm::Gemm);
                    task.execute();
                }

                auto perfEntries = gInferencingCtx->getPerfMeasurements();
                double avgMs = 0;
                for (const auto& entry : perfEntries)
                {
                    if (strstr(entry.name.getBuffer(), "Convolution") ||
                        strstr(entry.name.getBuffer(), "convolution"))
                    {
                        avgMs = entry.totalTimeMs / entry.callCount;
                        break;
                    }
                }

                printf(" %12.3f", avgMs);

                if (avgMs < bestTime)
                {
                    bestTime = avgMs;
                    bestName = tv.name;
                }
            }

            printf("     %s\n", bestName);
        }

        printf("\n");

        return SLANG_OK;
    }
};

// Entry point for running benchmark from unit-test.exe --bench-conv2d
int runConvBenchmark(int argc, char* argv[])
{
    ConvBenchProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}

// ============================================================================
// Convolution Profiling Mode for Nsight
// ============================================================================
// Runs a single convolution config in a tight loop for GPU profiling.
// Usage: unit-test.exe --profile-conv [iterations]

int runConvProfile(int argc, char* argv[])
{
    // Parse iteration count (default 1000)
    int iterations = 1000;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--profile-conv") == 0 && i + 1 < argc)
        {
            iterations = atoi(argv[i + 1]);
            if (iterations <= 0) iterations = 1000;
        }
    }

    printf("=== Convolution Profiling Mode ===\n");
    printf("Iterations: %d\n\n", iterations);

    // Initialize device
    rhi::DeviceDesc deviceDesc;
    deviceDesc.slang.targetProfile = "spirv_1_6";
    List<slang::CompilerOptionEntry> compilerOptionsEntries;
    const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
    for (auto cap : capabilities)
    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::Capability;
        slang::CompilerOptionValue value;
        value.kind = slang::CompilerOptionValueKind::String;
        value.stringValue0 = cap;
        entry.value = value;
        compilerOptionsEntries.add(entry);
    }
    deviceDesc.slang.compilerOptionEntries = compilerOptionsEntries.getBuffer();
    deviceDesc.slang.compilerOptionEntryCount = (uint32_t)compilerOptionsEntries.getCount();
    deviceDesc.deviceType = rhi::DeviceType::Vulkan;

    // Match InferencingContext's default device initialization so Slang can find `mlkl.slang`
    // and so we get persistent shader/pipeline caching (also avoids past access violations).
    RefPtr<FileShaderCache> shaderCache = new FileShaderCache();
    String exePath = Path::getParentDirectory(Path::getExecutablePath()).getBuffer();
    const char* exePathStr = exePath.getBuffer();
    deviceDesc.slang.searchPaths = &exePathStr;
    deviceDesc.slang.searchPathCount = 1;
    deviceDesc.persistentShaderCache = shaderCache;
    deviceDesc.persistentPipelineCache = shaderCache;

    auto device = rhi::getRHI()->createDevice(deviceDesc);
    if (!device)
    {
        printf("ERROR: Failed to create GPU device\n");
        return -1;
    }

    RefPtr<InferencingContext> ctx = new InferencingContext(device);
    
    // Use the slowest config from SD: 16x16 with 2560->1280 channels
    const int inputH = 16;
    const int inputW = 16;
    const int inChannels = 2560;
    const int outChannels = 1280;
    const int kernelSize = 3;
    const int stride = 1;
    const int padding = 1;
    const int batchSize = 2;  // CFG batching
    
    printf("Config: %dx%d, %d->%d channels, 3x3 stride=%d, batch=%d\n",
           inputH, inputW, inChannels, outChannels, stride, batchSize);
    printf("This is the slowest SD convolution config (~2.6 ms/call)\n\n");
    
    // Create baseline kernel (GEMM)
    Conv2DKernel kernel(ctx, 16, kernelSize, stride, inChannels, outChannels);
    
    // Create random weights and biases
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    List<float> weights;
    weights.setCount(inChannels * kernelSize * kernelSize * outChannels);
    for (Index i = 0; i < weights.getCount(); i++)
        weights[i] = dist(rng);
    
    List<float> biases;
    biases.setCount(outChannels);
    for (Index i = 0; i < biases.getCount(); i++)
        biases[i] = dist(rng) * 0.01f;
    
    kernel.loadParams(kernelSize, outChannels, weights.getBuffer(), biases.getBuffer());

    // ------------------------------------------------------------------------
    // Winograd prototype setup (standalone kernel, no Conv2DKernel/IExpr/ISink)
    // ------------------------------------------------------------------------
    // Prototype constraints (for this profiling config they are satisfied):
    // - inChannels % 8 == 0
    // - outChannels % 4 == 0
    if ((inChannels % 8) != 0 || (outChannels % 4) != 0)
    {
        printf("ERROR: winogradConvProto requires inChannels%%8==0 and outChannels%%4==0\n");
        return -1;
    }

    // Create prototype pipeline (fused, single kernel)
    List<String> emptyArgs;
    auto winogradProtoPipeline = ctx->createComputePipeline("winogradConvProtoBlock16", emptyArgs.getArrayView());
    if (!winogradProtoPipeline)
    {
        printf("ERROR: Failed to create winogradConvProto pipeline\n");
        return -1;
    }

    // Create nonfused Winograd pipelines (cuDNN-style: input transform -> GEMM -> output transform)
    auto winogradInputPipeline = ctx->createComputePipeline("winogradInputTransformBlock16", emptyArgs.getArrayView());
    auto winogradGemmPipeline = ctx->createComputePipeline("winogradDomainGemmF43", emptyArgs.getArrayView());
    auto winogradGemmPipelineV2 = ctx->createComputePipeline("winogradDomainGemmF43_v2", emptyArgs.getArrayView());
    auto winogradFused23PipelineV2 = ctx->createComputePipeline("winogradDomainGemmOutputFusedF43_v2", emptyArgs.getArrayView());
    auto winogradOutputPipeline = ctx->createComputePipeline("winogradOutputTransformBlock16", emptyArgs.getArrayView());
    if (!winogradInputPipeline || !winogradGemmPipeline || !winogradGemmPipelineV2 || !winogradFused23PipelineV2 || !winogradOutputPipeline)
    {
        printf("ERROR: Failed to create one or more nonfused Winograd pipelines\n");
        return -1;
    }

    // ------------------------------------------------------------------------
    // GEMM conv v2 prototype (NHWC float4 packed fast path)
    // ------------------------------------------------------------------------
    // gemmConvolution_v2 is a specialized entrypoint (like gemmConvolution): we must pass its specialization args.
    // Keep it aligned with the default GEMM tile config used in this profile (16x16 tile, tileOC=16, tileIC=8, 1x1 per thread).
    static const int kGemmV2TileOH = 16;
    static const int kGemmV2TileOW = 16;
    static const int kGemmV2TileOC = 16;
    static const int kGemmV2TileIC = 8;
    static const int kGemmV2ThreadOH = 1;
    static const int kGemmV2ThreadOW = 1;

    String gemmV2Args[] = {
        String(kGemmV2TileOH),
        String(kGemmV2TileOW),
        String(kGemmV2TileOC),
        String(kGemmV2TileIC),
        String(kGemmV2ThreadOH),
        String(kGemmV2ThreadOW),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
    };
    auto gemmConvV2Pipeline = ctx->createComputePipeline("gemmConvolution_v2", makeArrayView(gemmV2Args));
    if (!gemmConvV2Pipeline)
    {
        printf("ERROR: Failed to create gemmConvolution_v2 pipeline\n");
        return -1;
    }

    struct Float4 { float x, y, z, w; };

    auto transformWeightsF43Packed = [&](const float* wIn_IKKO, int inC, int outC) -> List<Float4>
    {
        // Input layout: [inC, 3, 3, outC]
        // Output packed layout (Float4 array):
        //   weightsU[pos][ocVec][icVec][o] where each entry is float4 of 4 input-channel weights
        // Flattened:
        //   idx = (((pos * ocVecCount + ocVec) * icVecCount + icVec) * 4 + o)
        const int K = 3;
        const int ocVecCount = outC / 4;
        const int icVecCount = inC / 4;

        List<Float4> out;
        out.setCount((Index)36 * ocVecCount * icVecCount * 4);

        // G matrix for Winograd F(4,3) (matches Conv2DKernel::transformWeightsToWinograd)
        const float G[6][3] = {
            { 1.0f / 4,       0,       0},
            {-1.0f / 6, -1.0f / 6, -1.0f / 6},
            {-1.0f / 6,  1.0f / 6, -1.0f / 6},
            { 1.0f / 24, 1.0f / 12,  1.0f / 6},
            { 1.0f / 24,-1.0f / 12,  1.0f / 6},
            {      0,       0,       1},
        };

        auto getW = [&](int ic, int ky, int kx, int oc) -> float
        {
            int idx = ((ic * K + ky) * K + kx) * outC + oc;
            return wIn_IKKO[idx];
        };

        // For each (oc, ic) compute U[6][6] once, then pack into the float4 layout.
        // Pack order: pos-major, then ocVec, then icVec, then o(0..3).
        for (int ocVec = 0; ocVec < ocVecCount; ++ocVec)
        {
            for (int icVec = 0; icVec < icVecCount; ++icVec)
            {
                for (int o = 0; o < 4; ++o)
                {
                    int oc = ocVec * 4 + o;

                    // Compute full U[6][6] (stored as 36 values) for the 4 input channels in this icVec.
                    float Uvals[4][36];
                    for (int i = 0; i < 4; ++i)
                    {
                        int ic = icVec * 4 + i;

                        // Load 3x3 g
                        float g[3][3];
                        for (int ky = 0; ky < 3; ++ky)
                            for (int kx = 0; kx < 3; ++kx)
                                g[ky][kx] = getW(ic, ky, kx, oc);

                        // temp = G * g  (6x3)
                        float temp[6][3];
                        for (int r = 0; r < 6; ++r)
                            for (int c = 0; c < 3; ++c)
                                temp[r][c] = G[r][0] * g[0][c] + G[r][1] * g[1][c] + G[r][2] * g[2][c];

                        // U = temp * G^T (6x6)
                        for (int py = 0; py < 6; ++py)
                        {
                            for (int px = 0; px < 6; ++px)
                            {
                                float U = temp[py][0] * G[px][0] + temp[py][1] * G[px][1] + temp[py][2] * G[px][2];
                                Uvals[i][py * 6 + px] = U;
                            }
                        }
                    }

                    // Pack into float4 for each pos: {U(ic0), U(ic1), U(ic2), U(ic3)}
                    for (int pos = 0; pos < 36; ++pos)
                    {
                        Float4 wVec = {Uvals[0][pos], Uvals[1][pos], Uvals[2][pos], Uvals[3][pos]};
                        int idx = (((pos * ocVecCount + ocVec) * icVecCount + icVec) * 4 + o);
                        out[idx] = wVec;
                    }
                }
            }
        }

        return out;
    };

    auto winogradWeightsPacked = transformWeightsF43Packed(weights.getBuffer(), inChannels, outChannels);
    auto winogradWeightsBuffer = ctx->createPersistentBuffer(
        winogradWeightsPacked.getBuffer(),
        (size_t)winogradWeightsPacked.getCount() * sizeof(Float4),
        "winograd_proto_weights");
    if (!winogradWeightsBuffer)
    {
        printf("ERROR: Failed to create winograd prototype weights buffer\n");
        return -1;
    }

    // Pack weights for gemmConvolution_v2:
    // Expected weightsP index:
    //   idx = (((((kh * K + kw) * icVecCount + icVec) * ocVecCount + ocVec) * 4) + o)
    // where each entry is float4 of 4 scalar IC weights, and o selects the output lane within ocVec.
    ComPtr<rhi::IBuffer> gemmConvV2WeightsBuffer;
    {
        if ((inChannels % 4) != 0 || (outChannels % 4) != 0)
        {
            printf("ERROR: gemmConvolution_v2 requires inChannels%%4==0 and outChannels%%4==0\n");
            return -1;
        }

        const int K = kernelSize;
        const int icVecCountLocal = inChannels / 4;
        const int ocVecCountLocal = outChannels / 4;

        List<Float4> weightsP;
        weightsP.setCount((Index)K * K * icVecCountLocal * ocVecCountLocal * 4);

        auto getW = [&](int ic, int ky, int kx, int oc) -> float
        {
            // weights input layout here is [inC, K, K, outC] (IKKO)
            int idx = ((ic * K + ky) * K + kx) * outChannels + oc;
            return weights[idx];
        };

        for (int ky = 0; ky < K; ++ky)
        {
            for (int kx = 0; kx < K; ++kx)
            {
                for (int icVec = 0; icVec < icVecCountLocal; ++icVec)
                {
                    for (int ocVec = 0; ocVec < ocVecCountLocal; ++ocVec)
                    {
                        for (int o = 0; o < 4; ++o)
                        {
                            int oc = ocVec * 4 + o;
                            float w0 = getW(icVec * 4 + 0, ky, kx, oc);
                            float w1 = getW(icVec * 4 + 1, ky, kx, oc);
                            float w2 = getW(icVec * 4 + 2, ky, kx, oc);
                            float w3 = getW(icVec * 4 + 3, ky, kx, oc);

                            int outIdx = (((((ky * K + kx) * icVecCountLocal + icVec) * ocVecCountLocal + ocVec) * 4) + o);
                            weightsP[(Index)outIdx] = {w0, w1, w2, w3};
                        }
                    }
                }
            }
        }

        gemmConvV2WeightsBuffer = ctx->createPersistentBuffer(
            weightsP.getBuffer(),
            (size_t)weightsP.getCount() * sizeof(Float4),
            "gemm_conv_v2_weightsP");
        if (!gemmConvV2WeightsBuffer)
        {
            printf("ERROR: Failed to create gemmConvolution_v2 weights buffer\n");
            return -1;
        }
    }

    // Repack weights for winogradDomainGemmF43_v2:
    // v1 layout: idx = (((pos * ocVecCount + ocVec) * icVecCount + icVec) * 4 + o)
    // v2 layout: idx = (((pos * icVecCount + icVec) * ocVecCount + ocVec) * 4 + o)
    ComPtr<rhi::IBuffer> winogradWeightsBufferV2;
    {
        List<Float4> winogradWeightsPackedV2;
        winogradWeightsPackedV2.setCount(winogradWeightsPacked.getCount());

        const int ocVecCountLocal = outChannels / 4;
        const int icVecCountLocal = inChannels / 4;

        for (int pos = 0; pos < 36; ++pos)
        {
            for (int ocVec = 0; ocVec < ocVecCountLocal; ++ocVec)
            {
                for (int icVec = 0; icVec < icVecCountLocal; ++icVec)
                {
                    for (int o = 0; o < 4; ++o)
                    {
                        int srcIdx = (((pos * ocVecCountLocal + ocVec) * icVecCountLocal + icVec) * 4 + o);
                        int dstIdx = (((pos * icVecCountLocal + icVec) * ocVecCountLocal + ocVec) * 4 + o);
                        winogradWeightsPackedV2[(Index)dstIdx] = winogradWeightsPacked[(Index)srcIdx];
                    }
                }
            }
        }

        auto winogradWeightsBufferV2Local = ctx->createPersistentBuffer(
            winogradWeightsPackedV2.getBuffer(),
            (size_t)winogradWeightsPackedV2.getCount() * sizeof(Float4),
            "winograd_proto_weights_v2");
        if (!winogradWeightsBufferV2Local)
        {
            printf("ERROR: Failed to create winograd prototype weights buffer (v2)\n");
            return -1;
        }
        // Keep it alive for the rest of this function.
        winogradWeightsBufferV2 = winogradWeightsBufferV2Local;
    }
    
    // Create input buffer
    List<float> inputData;
    inputData.setCount(batchSize * inputH * inputW * inChannels);
    for (Index i = 0; i < inputData.getCount(); i++)
        inputData[i] = dist(rng);
    
    auto inputBuffer = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, inputH, inputW, inChannels),
        inputData);
    
    // Create output buffers
    int outputH = (inputH + 2 * padding - kernelSize) / stride + 1;
    int outputW = (inputW + 2 * padding - kernelSize) / stride + 1;
    auto outputGemm = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        0,
        nullptr,
        "conv_output_gemm");
    auto outputProto = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        0,
        nullptr,
        "conv_output_winograd_proto");
    if (!outputGemm || !outputProto)
    {
        printf("ERROR: Failed to allocate output tensors\n");
        return -1;
    }

    auto outputGemmV2 = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        0,
        nullptr,
        "conv_output_gemm_v2");
    if (!outputGemmV2)
    {
        printf("ERROR: Failed to allocate output tensor for gemmConvolution_v2\n");
        return -1;
    }

    struct GemmConvV2Params
    {
        rhi::DeviceAddress input;
        rhi::DeviceAddress output;
        rhi::DeviceAddress bias;
        rhi::DeviceAddress weightsP;
        int H;
        int W;
        int padding;
        int batchSize;
    };

    GemmConvV2Params gemmV2Params = {};
    gemmV2Params.input = inputBuffer->getView().getDeviceAddress();
    gemmV2Params.output = outputGemmV2->getView().getDeviceAddress();
    gemmV2Params.bias = kernel.biasesBuffer->getDeviceAddress();
    gemmV2Params.weightsP = gemmConvV2WeightsBuffer->getDeviceAddress();
    gemmV2Params.H = outputH; // output H==inputH for pad=1, stride=1 in this profile config
    gemmV2Params.W = outputW;
    gemmV2Params.padding = padding;
    gemmV2Params.batchSize = batchSize;

    // Nonfused Winograd intermediate buffers (A/V and C/M), plus output
    int numTilesH = (outputH + 3) / 4;
    int numTilesW = (outputW + 3) / 4;
    int numTiles = batchSize * numTilesH * numTilesW;
    int icVecCount = inChannels / 4;
    int ocVecCount = outChannels / 4;

    auto winogradV = ctx->createTensor(
        ElementType::Float32,
        Shape(numTiles, 36, icVecCount * 4),
        0,
        nullptr,
        "winograd_V");
    auto winogradM = ctx->createTensor(
        ElementType::Float32,
        Shape(numTiles, 36, ocVecCount * 4),
        0,
        nullptr,
        "winograd_M");
    auto outputNonFused = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        0,
        nullptr,
        "conv_output_winograd_nonfused");
    auto outputFused23 = ctx->createTensor(
        ElementType::Float32,
        Shape(batchSize, outputH, outputW, outChannels),
        0,
        nullptr,
        "conv_output_winograd_fused23");
    if (!winogradV || !winogradM || !outputNonFused)
    {
        printf("ERROR: Failed to allocate Winograd nonfused intermediate tensors\n");
        return -1;
    }
    if (!outputFused23)
    {
        printf("ERROR: Failed to allocate Winograd fused(2+3) output tensor\n");
        return -1;
    }
    
    // Warmup (10 iterations)
    printf("Warming up (10 iterations)...\n");
    for (int i = 0; i < 10; i++)
    {
        auto task = ctx->createTask();
        kernel.queueExecute(task, outputGemm->getView(), inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
        task.execute();
    }
    
    // Benchmark GEMM
    printf("\n=== GEMM (shared memory weights) ===\n");
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++)
    {
        auto task = ctx->createTask();
        kernel.queueExecute(task, outputGemm->getView(), inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
        task.execute();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double gemmMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    printf("Average per iteration: %.3f ms\n", gemmMs / iterations);

    // Benchmark GEMM v2 (float4 packed)
    printf("\n=== GEMM v2 (float4 packed, channels%%4==0) ===\n");
    // Warmup
    for (int i = 0; i < 10; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            gemmConvV2Pipeline,
            (uint32_t)((outputW + kGemmV2TileOW - 1) / kGemmV2TileOW),
            (uint32_t)((outputH + kGemmV2TileOH - 1) / kGemmV2TileOH),
            (uint32_t)batchSize * (uint32_t)(((outChannels / 4) + (kGemmV2TileOC / 4) - 1) / (kGemmV2TileOC / 4)),
            gemmV2Params);
        task.execute();
    }
    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            gemmConvV2Pipeline,
            (uint32_t)((outputW + kGemmV2TileOW - 1) / kGemmV2TileOW),
            (uint32_t)((outputH + kGemmV2TileOH - 1) / kGemmV2TileOH),
            (uint32_t)batchSize * (uint32_t)(((outChannels / 4) + (kGemmV2TileOC / 4) - 1) / (kGemmV2TileOC / 4)),
            gemmV2Params);
        task.execute();
    }
    endTime = std::chrono::high_resolution_clock::now();
    double gemmV2Ms = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    printf("Average per iteration: %.3f ms\n", gemmV2Ms / iterations);
    
    // NOTE: gemmConvolutionWaveShuffle was a failed experiment; intentionally not benchmarked.
    double waveMs = 0.0;

    // ------------------------------------------------------------------------
    // Warmup + benchmark Winograd prototype
    // ------------------------------------------------------------------------
    struct WinogradProtoParams
    {
        rhi::DeviceAddress input;
        rhi::DeviceAddress output;
        rhi::DeviceAddress bias;
        rhi::DeviceAddress weightsU;
        int H;
        int W;
        int inChannels;
        int outChannels;
        int batchSize;
    };

    WinogradProtoParams protoParams = {};
    protoParams.input = inputBuffer->getView().getDeviceAddress();
    protoParams.output = outputProto->getView().getDeviceAddress();
    protoParams.bias = kernel.biasesBuffer->getDeviceAddress();
    protoParams.weightsU = winogradWeightsBuffer->getDeviceAddress();
    protoParams.H = inputH;
    protoParams.W = inputW;
    protoParams.inChannels = inChannels;
    protoParams.outChannels = outChannels;
    protoParams.batchSize = batchSize;

    struct WinogradNonFusedParams
    {
        rhi::DeviceAddress input;
        rhi::DeviceAddress output;
        rhi::DeviceAddress bias;
        rhi::DeviceAddress weightsU;
        rhi::DeviceAddress V;
        rhi::DeviceAddress M;
        int H;
        int W;
        int padding;
        int batchSize;
        int inChannels;
        int outChannels;
        int numTilesH;
        int numTilesW;
        int numTiles;
    };

    WinogradNonFusedParams nonfusedParams = {};
    nonfusedParams.input = inputBuffer->getView().getDeviceAddress();
    nonfusedParams.output = outputNonFused->getView().getDeviceAddress();
    nonfusedParams.bias = kernel.biasesBuffer->getDeviceAddress();
    nonfusedParams.weightsU = winogradWeightsBuffer->getDeviceAddress();
    nonfusedParams.V = winogradV->getView().getDeviceAddress();
    nonfusedParams.M = winogradM->getView().getDeviceAddress();
    nonfusedParams.H = inputH;
    nonfusedParams.W = inputW;
    nonfusedParams.padding = padding;
    nonfusedParams.batchSize = batchSize;
    nonfusedParams.inChannels = inChannels;
    nonfusedParams.outChannels = outChannels;
    nonfusedParams.numTilesH = numTilesH;
    nonfusedParams.numTilesW = numTilesW;
    nonfusedParams.numTiles = numTiles;

    // v2 params only differ by weights packing
    WinogradNonFusedParams nonfusedParamsV2 = nonfusedParams;
    {
        nonfusedParamsV2.weightsU = winogradWeightsBufferV2->getDeviceAddress();
    }

    printf("\n=== Winograd Prototype (standalone, float4 IO/weights) ===\n");
    for (int i = 0; i < 10; i++)
    {
        auto task = ctx->createTask();
        static const int kWinogradProtoOcVecPerCta = 2; // must match OCV_PER_CTA in src/winograd.slang
        task.dispatchKernel(
            winogradProtoPipeline,
            (outputW + 15) / 16,
            (outputH + 15) / 16,
            (uint32_t)batchSize * (uint32_t)(((outChannels / 4) + kWinogradProtoOcVecPerCta - 1) / kWinogradProtoOcVecPerCta),
            protoParams);
        task.execute();
    }

    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        auto task = ctx->createTask();
        static const int kWinogradProtoOcVecPerCta = 2; // must match OCV_PER_CTA in src/winograd.slang
        task.dispatchKernel(
            winogradProtoPipeline,
            (outputW + 15) / 16,
            (outputH + 15) / 16,
            (uint32_t)batchSize * (uint32_t)(((outChannels / 4) + kWinogradProtoOcVecPerCta - 1) / kWinogradProtoOcVecPerCta),
            protoParams);
        task.execute();
    }
    endTime = std::chrono::high_resolution_clock::now();
    double winoProtoMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    printf("Average per iteration: %.3f ms\n", winoProtoMs / iterations);

    // ------------------------------------------------------------------------
    // Warmup + benchmark Winograd nonfused (3-kernel pipeline)
    // ------------------------------------------------------------------------
    printf("\n=== Winograd Nonfused (input transform + GEMM + output transform) ===\n");

    // GPU timestamp breakdown for the 3-kernel pipeline
    ctx->setCollectPerfMeasurements(true);
    ctx->resetPerfMeasurements();

    // Warmup (10 iterations)
    for (int i = 0; i < 10; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            winogradInputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)icVecCount,
            nonfusedParams);
        task.dispatchKernel(
            winogradGemmPipeline,
            (uint32_t)((numTiles + 31) / 32),
            (uint32_t)((ocVecCount + 15) / 16),
            36,
            nonfusedParams);
        task.dispatchKernel(
            winogradOutputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)ocVecCount,
            nonfusedParams);
        task.execute();
    }

    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            winogradInputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)icVecCount,
            nonfusedParams);
        task.dispatchKernel(
            winogradGemmPipeline,
            (uint32_t)((numTiles + 31) / 32),
            (uint32_t)((ocVecCount + 15) / 16),
            36,
            nonfusedParams);
        task.dispatchKernel(
            winogradOutputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)ocVecCount,
            nonfusedParams);
        task.execute();
    }
    endTime = std::chrono::high_resolution_clock::now();
    double winoNonFusedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    printf("Average per iteration: %.3f ms\n", winoNonFusedMs / iterations);

    // Print GPU timestamp breakdown across the benchmarked iterations.
    ctx->printPerfMeasurements();
    ctx->resetPerfMeasurements();
    ctx->setCollectPerfMeasurements(false);

    // ------------------------------------------------------------------------
    // Warmup + benchmark Winograd nonfused (v2 GEMM kernel, v2 weight packing)
    // ------------------------------------------------------------------------
    printf("\n=== Winograd Nonfused (v2 GEMM + v2 packed weights) ===\n");

    ctx->setCollectPerfMeasurements(true);
    ctx->resetPerfMeasurements();

    for (int i = 0; i < 10; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            winogradInputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)icVecCount,
            nonfusedParamsV2);
        task.dispatchKernel(
            winogradGemmPipelineV2,
            (uint32_t)((numTiles + 31) / 32),
            (uint32_t)((ocVecCount + 15) / 16),
            36,
            nonfusedParamsV2);
        task.dispatchKernel(
            winogradOutputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)ocVecCount,
            nonfusedParamsV2);
        task.execute();
    }

    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            winogradInputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)icVecCount,
            nonfusedParamsV2);
        task.dispatchKernel(
            winogradGemmPipelineV2,
            (uint32_t)((numTiles + 31) / 32),
            (uint32_t)((ocVecCount + 15) / 16),
            36,
            nonfusedParamsV2);
        task.dispatchKernel(
            winogradOutputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)ocVecCount,
            nonfusedParamsV2);
        task.execute();
    }
    endTime = std::chrono::high_resolution_clock::now();
    double winoNonFusedV2Ms = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    printf("Average per iteration: %.3f ms\n", winoNonFusedV2Ms / iterations);

    ctx->printPerfMeasurements();
    ctx->resetPerfMeasurements();
    ctx->setCollectPerfMeasurements(false);

    // ------------------------------------------------------------------------
    // Warmup + benchmark Winograd fused (2+3) using v2 packed weights
    // Pipeline: input transform -> fused(2+3)
    // ------------------------------------------------------------------------
    printf("\n=== Winograd Fused(2+3) (v2 packed weights, no global M) ===\n");

    // Use the fused output buffer
    WinogradNonFusedParams fused23ParamsV2 = nonfusedParamsV2;
    fused23ParamsV2.output = outputFused23->getView().getDeviceAddress();

    ctx->setCollectPerfMeasurements(true);
    ctx->resetPerfMeasurements();

    // Warmup
    for (int i = 0; i < 10; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            winogradInputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)icVecCount,
            fused23ParamsV2);
        static const int kWinogradFused23OcVecPerCta = 4; // must match OCV_PER_CTA in src/winograd.slang
        task.dispatchKernel(
            winogradFused23PipelineV2,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)((ocVecCount + kWinogradFused23OcVecPerCta - 1) / kWinogradFused23OcVecPerCta),
            fused23ParamsV2);
        task.execute();
    }

    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        auto task = ctx->createTask();
        task.dispatchKernel(
            winogradInputPipeline,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)icVecCount,
            fused23ParamsV2);
        static const int kWinogradFused23OcVecPerCta = 4; // must match OCV_PER_CTA in src/winograd.slang
        task.dispatchKernel(
            winogradFused23PipelineV2,
            (uint32_t)((numTilesW + 3) / 4),
            (uint32_t)((numTilesH + 3) / 4),
            (uint32_t)batchSize * (uint32_t)((ocVecCount + kWinogradFused23OcVecPerCta - 1) / kWinogradFused23OcVecPerCta),
            fused23ParamsV2);
        task.execute();
    }
    endTime = std::chrono::high_resolution_clock::now();
    double winoFused23V2Ms = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    printf("Average per iteration: %.3f ms\n", winoFused23V2Ms / iterations);

    ctx->printPerfMeasurements();
    ctx->resetPerfMeasurements();
    ctx->setCollectPerfMeasurements(false);

    // Correctness: compare one GEMM vs one Winograd nonfused run
    {
        size_t outSizeBytes = (size_t)batchSize * outputH * outputW * outChannels * sizeof(float);
        List<float> gemmOut;
        gemmOut.setCount((Index)(outSizeBytes / sizeof(float)));

        // Run GEMM once and read it back
        {
            auto task = ctx->createTask();
            kernel.queueExecute(task, outputGemm->getView(), inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
            task.execute();
            ctx->getDevice()->readBuffer(outputGemm->buffer, 0, outSizeBytes, gemmOut.getBuffer());
        }

        auto printMaxErr = [&](const char* label, const List<float>& other)
        {
            double maxAbs = 0.0;
            double maxRel = 0.0;
            for (Index i = 0; i < gemmOut.getCount(); i++)
            {
                double a = (double)gemmOut[i];
                double b = (double)other[i];
                double absErr = fabs(a - b);
                double relErr = absErr / (fabs(a) + 1e-6);
                if (absErr > maxAbs) maxAbs = absErr;
                if (relErr > maxRel) maxRel = relErr;
            }
            printf("%s: maxAbs=%.6g  maxRel=%.6g\n", label, maxAbs, maxRel);
        };

        // Run nonfused v1 once
        {
            auto task = ctx->createTask();
            task.dispatchKernel(
                winogradInputPipeline,
                (uint32_t)((numTilesW + 3) / 4),
                (uint32_t)((numTilesH + 3) / 4),
                (uint32_t)batchSize * (uint32_t)icVecCount,
                nonfusedParams);
            task.dispatchKernel(
                winogradGemmPipeline,
                (uint32_t)((numTiles + 31) / 32),
                (uint32_t)((ocVecCount + 15) / 16),
                36,
                nonfusedParams);
            task.dispatchKernel(
                winogradOutputPipeline,
                (uint32_t)((numTilesW + 3) / 4),
                (uint32_t)((numTilesH + 3) / 4),
                (uint32_t)batchSize * (uint32_t)ocVecCount,
                nonfusedParams);
            task.execute();
        }
        {
            List<float> nfOut;
            nfOut.setCount((Index)(outSizeBytes / sizeof(float)));
            ctx->getDevice()->readBuffer(outputNonFused->buffer, 0, outSizeBytes, nfOut.getBuffer());
            printMaxErr("Correctness nonfused(v1) vs GEMM", nfOut);
        }

        // Run nonfused v2 once
        {
            auto task = ctx->createTask();
            task.dispatchKernel(
                winogradInputPipeline,
                (uint32_t)((numTilesW + 3) / 4),
                (uint32_t)((numTilesH + 3) / 4),
                (uint32_t)batchSize * (uint32_t)icVecCount,
                nonfusedParamsV2);
            task.dispatchKernel(
                winogradGemmPipelineV2,
                (uint32_t)((numTiles + 31) / 32),
                (uint32_t)((ocVecCount + 15) / 16),
                36,
                nonfusedParamsV2);
            task.dispatchKernel(
                winogradOutputPipeline,
                (uint32_t)((numTilesW + 3) / 4),
                (uint32_t)((numTilesH + 3) / 4),
                (uint32_t)batchSize * (uint32_t)ocVecCount,
                nonfusedParamsV2);
            task.execute();
        }
        {
            List<float> nfOutV2;
            nfOutV2.setCount((Index)(outSizeBytes / sizeof(float)));
            ctx->getDevice()->readBuffer(outputNonFused->buffer, 0, outSizeBytes, nfOutV2.getBuffer());
            printMaxErr("Correctness nonfused(v2) vs GEMM", nfOutV2);
        }

        // Run fused(2+3) v2 once (input transform -> fused kernel)
        {
            auto task = ctx->createTask();
            task.dispatchKernel(
                winogradInputPipeline,
                (uint32_t)((numTilesW + 3) / 4),
                (uint32_t)((numTilesH + 3) / 4),
                (uint32_t)batchSize * (uint32_t)icVecCount,
                fused23ParamsV2);
            static const int kWinogradFused23OcVecPerCta = 4; // must match OCV_PER_CTA in src/winograd.slang
            task.dispatchKernel(
                winogradFused23PipelineV2,
                (uint32_t)((numTilesW + 3) / 4),
                (uint32_t)((numTilesH + 3) / 4),
                (uint32_t)batchSize * (uint32_t)((ocVecCount + kWinogradFused23OcVecPerCta - 1) / kWinogradFused23OcVecPerCta),
                fused23ParamsV2);
            task.execute();
        }
        {
            List<float> fusedOut;
            fusedOut.setCount((Index)(outSizeBytes / sizeof(float)));
            ctx->getDevice()->readBuffer(outputFused23->buffer, 0, outSizeBytes, fusedOut.getBuffer());
            printMaxErr("Correctness fused(2+3,v2) vs GEMM", fusedOut);
        }
    }

    // Correctness: compare one GEMM vs one GEMM v2 run
    {
        size_t outSizeBytes = (size_t)batchSize * outputH * outputW * outChannels * sizeof(float);
        List<float> gemmOut;
        List<float> gemmV2Out;
        gemmOut.setCount((Index)(outSizeBytes / sizeof(float)));
        gemmV2Out.setCount((Index)(outSizeBytes / sizeof(float)));

        // GEMM baseline
        {
            auto task = ctx->createTask();
            kernel.queueExecute(task, outputGemm->getView(), inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
            task.execute();
            ctx->getDevice()->readBuffer(outputGemm->buffer, 0, outSizeBytes, gemmOut.getBuffer());
        }

        // GEMM v2
        {
            auto task = ctx->createTask();
            task.dispatchKernel(
                gemmConvV2Pipeline,
                (uint32_t)((outputW + kGemmV2TileOW - 1) / kGemmV2TileOW),
                (uint32_t)((outputH + kGemmV2TileOH - 1) / kGemmV2TileOH),
                (uint32_t)batchSize * (uint32_t)(((outChannels / 4) + (kGemmV2TileOC / 4) - 1) / (kGemmV2TileOC / 4)),
                gemmV2Params);
            task.execute();
            ctx->getDevice()->readBuffer(outputGemmV2->buffer, 0, outSizeBytes, gemmV2Out.getBuffer());
        }

        double maxAbs = 0.0;
        double maxRel = 0.0;
        for (Index i = 0; i < gemmOut.getCount(); i++)
        {
            double a = (double)gemmOut[i];
            double b = (double)gemmV2Out[i];
            double absErr = fabs(a - b);
            double relErr = absErr / (fabs(a) + 1e-6);
            if (absErr > maxAbs) maxAbs = absErr;
            if (relErr > maxRel) maxRel = relErr;
        }
        printf("Correctness gemm(v2) vs gemm: maxAbs=%.6g  maxRel=%.6g\n", maxAbs, maxRel);
    }

    // ------------------------------------------------------------------------
    // Correctness: compare one GEMM vs one Winograd prototype run
    // ------------------------------------------------------------------------
    {
        // Run GEMM once
        {
            auto task = ctx->createTask();
            kernel.queueExecute(task, outputGemm->getView(), inputBuffer->getView(), padding, ConvolutionAlgorithm::Gemm);
            task.execute();
        }
        // Run prototype once
        {
            auto task = ctx->createTask();
            static const int kWinogradProtoOcVecPerCta = 2; // must match OCV_PER_CTA in src/winograd.slang
            task.dispatchKernel(
                winogradProtoPipeline,
                (outputW + 15) / 16,
                (outputH + 15) / 16,
                (uint32_t)batchSize * (uint32_t)(((outChannels / 4) + kWinogradProtoOcVecPerCta - 1) / kWinogradProtoOcVecPerCta),
                protoParams);
            task.execute();
        }

        size_t outSizeBytes = (size_t)batchSize * outputH * outputW * outChannels * sizeof(float);
        List<float> gemmOut;
        List<float> protoOut;
        gemmOut.setCount((Index)(outSizeBytes / sizeof(float)));
        protoOut.setCount((Index)(outSizeBytes / sizeof(float)));

        ctx->getDevice()->readBuffer(outputGemm->buffer, 0, outSizeBytes, gemmOut.getBuffer());
        ctx->getDevice()->readBuffer(outputProto->buffer, 0, outSizeBytes, protoOut.getBuffer());

        double maxAbs = 0.0;
        double maxRel = 0.0;
        for (Index i = 0; i < gemmOut.getCount(); i++)
        {
            double a = (double)gemmOut[i];
            double b = (double)protoOut[i];
            double absErr = fabs(a - b);
            double relErr = absErr / (fabs(a) + 1e-6);
            if (absErr > maxAbs) maxAbs = absErr;
            if (relErr > maxRel) maxRel = relErr;
        }
        printf("Correctness vs GEMM: maxAbs=%.6g  maxRel=%.6g\n", maxAbs, maxRel);
    }
    
    // Summary
    printf("\n=== Summary ===\n");
    printf("GEMM:         %.3f ms\n", gemmMs / iterations);
    printf("GEMM v2:      %.3f ms\n", gemmV2Ms / iterations);
    if (waveMs > 0.0)
        printf("Wave Shuffle: %.3f ms\n", waveMs / iterations);
    printf("WinogradProto:%.3f ms\n", winoProtoMs / iterations);
    printf("WinoNonFused: %.3f ms\n", winoNonFusedMs / iterations);
    printf("WinoNonFusedV2: %.3f ms\n", winoNonFusedV2Ms / iterations);
    printf("WinoFused23V2: %.3f ms\n", winoFused23V2Ms / iterations);
    if (waveMs > 0.0)
        printf("Wave speedup: %.2fx\n", gemmMs / waveMs);
    printf("Wino speedup: %.2fx\n", gemmMs / winoProtoMs);
    printf("WinoNF speedup: %.2fx\n", gemmMs / winoNonFusedMs);
    printf("WinoNFv2 speedup: %.2fx\n", gemmMs / winoNonFusedV2Ms);
    printf("WinoFused23v2 speedup: %.2fx\n", gemmMs / winoFused23V2Ms);
    
    return 0;
}
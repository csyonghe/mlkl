#pragma once

#include "core/slang-io.h"
#include "inference-context.h"

using namespace Slang;

// ============================================================================
// SHARED HELPERS
// ============================================================================

// Fills a list with random float values (Normal distribution)
void initRandom(List<float>& data, int count);

// Convert float data to half precision (stored as uint16_t)
void floatToHalf(const List<float>& src, List<uint16_t>& dst);

// Convert half precision data (stored as uint16_t) to float
void halfToFloat(const List<uint16_t>& src, List<float>& dst);

// Compares GPU buffer content with expected CPU results
bool checkOutput(InferencingContext* ctx, BufferView outputBuffer, const List<float>& expected);
bool checkOutput(InferencingContext* ctx, TensorView outputBuffer, const List<float>& expected);

// Compares GPU buffer content (half precision) with expected CPU results (float)
// Uses larger tolerance appropriate for half precision
bool checkOutputHalf(InferencingContext* ctx, TensorView outputBuffer, const List<float>& expected);

// Compares GPU buffer content (int32) with expected CPU results (float, truncated to int)
bool checkOutputInt(InferencingContext* ctx, TensorView outputBuffer, const List<float>& expected);

// Convert float data to int32 (for creating test input tensors)
void floatToInt(const List<float>& src, List<int32_t>& dst);

// Writes weights/biases to file stream (compatible with TorchParamReader)
void writeLinearWeights(Stream* fs, const List<float>& weights, const List<float>& biases);

// ============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// ============================================================================

void cpuSoftmax(const float* input, float* output, int stride, int count);

void cpuBatchGemm(
    const float* A,
    const float* B,
    float* C,
    int batch,
    int M,
    int N,
    int K,
    bool transA,
    bool transB);

// Standard Linear: Out = In @ W^T + Bias (Weights expected in [In, Out] layout)
void cpuLinear(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch,
    int seqLen,
    int inDim,
    int outDim);


SlangResult testCheck(bool condition, const char* testName, const char* message);

#define TEST_CHECK(testName, condition) \
    SLANG_RETURN_ON_FAIL(testCheck((condition), (testName), #condition))

struct TestReportingRAII
{
    const char* testName;
    bool testOK = false;
    TestReportingRAII(const char* name)
        : testName(name)
    {
    }
    void markSuccess() { testOK = true; }
    ~TestReportingRAII() { printf("%s test: %s\n", (testOK ? "Passed" : "Failed"), testName); }
};

#define MLKL_TEST_BEGIN() TestReportingRAII testReportingRAII(__func__);
#define MLKL_TEST_OK()               \
    testReportingRAII.markSuccess(); \
    return SLANG_OK;

// ============================================================================
// TEST ENTRY POINTS
// ============================================================================

// Defined in test-elementwise.cpp
SlangResult testTranspose(InferencingContext* ctx);
SlangResult testMaterialize(InferencingContext* ctx);
SlangResult testReluNegSin(InferencingContext* ctx);
SlangResult testLeakyReluComposite(InferencingContext* ctx);
SlangResult testMultiConcat(InferencingContext* ctx);
SlangResult testIdentityPermute(InferencingContext* ctx);
SlangResult testNonTrivialPermute(InferencingContext* ctx);
SlangResult testUpsample(InferencingContext* ctx);

// Defined in test-gemm.cpp
SlangResult testBatchGemm(InferencingContext* ctx);
SlangResult testFusedBatchGemm(InferencingContext* ctx);
SlangResult testBatchGemmHalf(InferencingContext* ctx);

// Defined in test-cross-attention.cpp
SlangResult testFlashAttention(InferencingContext* ctx);
SlangResult testFlashAttentionHalf(InferencingContext* ctx);
SlangResult testFlashAttentionInputPermutationOnly(InferencingContext* ctx);
SlangResult testFlashAttentionFusedPermutation(InferencingContext* ctx);
SlangResult testSoftmax(InferencingContext* ctx);
SlangResult testSoftmaxHalf(InferencingContext* ctx);
SlangResult testCrossAttentionFull(InferencingContext* ctx);

// Defined in test-conv2d.cpp
SlangResult testConv2D(InferencingContext* ctx);
SlangResult testConv2DHalf(InferencingContext* ctx);
SlangResult testConv2DInt(InferencingContext* ctx);
SlangResult testConv2DGemmWithOutputExpr(InferencingContext* ctx);
SlangResult testConv2DGemmBatchedHalfFused(InferencingContext* ctx);
SlangResult testConv2DGemmMultipleSizes(InferencingContext* ctx);
SlangResult testIm2ColExpressionOnly(InferencingContext* ctx);
SlangResult testConv2DOutputSink(InferencingContext* ctx);
SlangResult testConv2DWithPermuteSink(InferencingContext* ctx);
SlangResult testConv2DWithFusedResidual(InferencingContext* ctx);
SlangResult testConv2DGemm(InferencingContext* ctx);
SlangResult testConv2DGemmBatched(InferencingContext* ctx);
SlangResult testConv2DWinograd(InferencingContext* ctx);

// Defined in test-transposed-conv2d.cpp
SlangResult testTransposedConv2D(InferencingContext* ctx);
SlangResult testTransposedConv2DHalf(InferencingContext* ctx);

// Defined in test-broadcast-add.cpp
SlangResult testBroadcastAdd(InferencingContext* ctx);

// Defined in test-classifier-free-guidance.cpp
SlangResult testClassifierFreeGuidance(InferencingContext* ctx);

// Defined in test-linear.cpp
SlangResult testLinear(InferencingContext* ctx);
SlangResult testLinearPartitioned(InferencingContext* ctx);
SlangResult testLinearHalf(InferencingContext* ctx);
SlangResult testLinearInt(InferencingContext* ctx);
SlangResult testLinearGemvBatch1(InferencingContext* ctx);
SlangResult testLinearGemvBatch4(InferencingContext* ctx);
SlangResult testLinearGemvBatch8(InferencingContext* ctx);
SlangResult testLinearTiledBatch1(InferencingContext* ctx);
SlangResult testLinearTiledBatch4(InferencingContext* ctx);
SlangResult testLinearTiledBatch16(InferencingContext* ctx);
SlangResult testLinearAutoSelection(InferencingContext* ctx);
SlangResult testLinearGemvLargeK(InferencingContext* ctx);// Defined in test-reduce.cpp
SlangResult testReduceLastDim(InferencingContext* ctx);
SlangResult testReduceGroupNorm(InferencingContext* ctx);
SlangResult testReduceAxis(InferencingContext* ctx);
SlangResult testReduceAxis4D(InferencingContext* ctx);
SlangResult testReduceLarge(InferencingContext* ctx);
SlangResult testReduceHalf(InferencingContext* ctx);// Defined in test-group-norm.cpp
SlangResult testGroupNorm(InferencingContext* ctx);
SlangResult testGroupNormSingleGroup(InferencingContext* ctx);
SlangResult testGroupNormPerChannel(InferencingContext* ctx);
SlangResult testGroupNormLarge(InferencingContext* ctx);
SlangResult testGroupNormHalf(InferencingContext* ctx);
SlangResult testGroupNormStats(InferencingContext* ctx);

// Defined in test-layer-norm.cpp
SlangResult testLayerNorm(InferencingContext* ctx);
SlangResult testLayerNormStats(InferencingContext* ctx);
SlangResult testLayerNormLarge(InferencingContext* ctx);
SlangResult testLayerNormHalf(InferencingContext* ctx);
SlangResult testRMSNorm(InferencingContext* ctx);
SlangResult testRMSNormIdentity(InferencingContext* ctx);
SlangResult testRMSNormLarge(InferencingContext* ctx);
SlangResult testRMSNormHalf(InferencingContext* ctx);

// Defined in test-safetensors.cpp (no InferencingContext needed)
SlangResult testSafeTensorsLoad();
SlangResult testSafeTensorsLoadMissing();
SlangResult testSafeTensorsTensorInfo();
SlangResult testSafeTensorsReadBasic();
SlangResult testSafeTensorsTypeConversion();
SlangResult testSafeTensorsLinear();
SlangResult testSafeTensorsConv2DPermutation();
SlangResult testSafeTensorsTransposedConv2DPermutation();
SlangResult testSafeTensorsPermutation();
SlangResult testSafeTensorsMixedPrecision();
SlangResult testSafeTensorsEmbedding();
SlangResult testSafeTensorsNorm();

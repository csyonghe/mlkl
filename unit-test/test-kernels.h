#pragma once

#include "core/slang-io.h"
#include "inference-context.h"

using namespace Slang;

// ============================================================================
// SHARED HELPERS
// ============================================================================

// Fills a list with random float values (Normal distribution)
void initRandom(List<float>& data, int count);

// Compares GPU buffer content with expected CPU results
bool checkOutput(InferencingContext* ctx, BufferView outputBuffer, const List<float>& expected);

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

// Defined in test-gemm.cpp
SlangResult testBatchGemm(InferencingContext* ctx);
SlangResult testFusedBatchGemm(InferencingContext* ctx);

// Defined in test-cross-attention.cpp
SlangResult testFlashAttention(InferencingContext* ctx);
SlangResult testFlashAttentionInputPermutationOnly(InferencingContext* ctx);
SlangResult testFlashAttentionFusedPermutation(InferencingContext* ctx);
SlangResult testSoftmax(InferencingContext* ctx);
SlangResult testCrossAttentionFull(InferencingContext* ctx);

// Defined in test-conv2d.cpp
SlangResult testConv2D(InferencingContext* ctx);

// Defined in test-transposed-conv2d.cpp
SlangResult testTransposedConv2D(InferencingContext* ctx);

// Defined in test-broadcast-add.cpp
SlangResult testBroadcastAdd(InferencingContext* ctx);

// Defined in test-classifier-free-guidance.cpp
SlangResult testClassifierFreeGuidance(InferencingContext* ctx);

// Defined in test-linear.cpp
SlangResult testLinear(InferencingContext* ctx);
SlangResult testLinearPartitioned(InferencingContext* ctx);

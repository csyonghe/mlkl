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

// Fused: ReLU( (Trans(A) @ (B*2)) + C )
void cpuFusedGemm(
    const float* A,
    const float* B,
    const float* C,
    float* Output,
    int batch,
    int M,
    int N,
    int K);

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

// ============================================================================
// TEST ENTRY POINTS
// ============================================================================

// Defined in test-elementwise.cpp
SlangResult testSoftmax(InferencingContext* ctx);

// Defined in test-gemm.cpp
SlangResult testBatchGemm(InferencingContext* ctx);
SlangResult testFusedBatchGemm(InferencingContext* ctx);

// Defined in test-cross-attention.cpp
SlangResult testCrossAttentionFull(InferencingContext* ctx);

#pragma once
#include "elementwise.h"
#include "kernel-base.h"

class SafeTensorsReader;

enum class LinearAlgorithm
{
    Auto,   // Automatically select based on batch size (Gemv for <=8, Tiled for >8)
    Tiled,  // Tiled GEMM (good for large batch sizes)
    Gemv,   // Batched GEMV (optimized for batch size 1-8)
};

// Linear Kernel: Out = In @ W^T + Bias (Weights expected in [In, Out] layout)
// W is of shape [outputVectorLength, inputVectorLength]
// In is of shape [batchSize, inputVectorLength]
// Out is of shape [batchSize, outputVectorLength]
// Underlying implementation uses tiled matrix multiplication.
//
// IMPORTANT: Input must be 2D tensor [batchSize, inputVectorLength]
// For 4D tensors [B, H, W, C], reshape to [B*H*W, C] before calling.
//
// CONSTRUCTOR: LinearKernel(ctx, inputDim, outputDim)
//
// COMMON MISTAKES:
// - Passing 4D tensor - causes "Input tensor must be rank 2" error
// - Confusing input/output dimensions
class LinearKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> tiledPipeline;
    ComPtr<rhi::IComputePipeline> gemvPipeline;
    InferencingContext* context;
    ElementType elementType;
    int tileM, tileN, tileK;
    int outputVectorLength; // Rows of weight matrix.
    int inputVectorLength;  // Columns of weight matrix.


    // For binding input buffer.
    ProgramNode inputProgram;
    ProgramNode outputProgram;
    SinkExpr sinkExpr;

    void validateTensorElementType(const TensorView& tv, const char* name) const;

public:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;

    LinearKernel(
        InferencingContext* context,
        ElementType elementType,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        int inputVectorLength,
        int outputVectorLength,
        int tileM = 64,
        int tileN = 64,
        int tileK = 8);

    // Convenience constructor defaulting to Float32.
    LinearKernel(
        InferencingContext* context,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        int inputVectorLength,
        int outputVectorLength,
        int tileM = 64,
        int tileN = 64,
        int tileK = 8)
        : LinearKernel(
              context,
              ElementType::Float32,
              inputExpr,
              outputExpr,
              sinkExpr,
              inputVectorLength,
              outputVectorLength,
              tileM,
              tileN,
              tileK)
    {
    }

    LinearKernel(
        InferencingContext* context,
        int inputVectorLength,
        int outputVectorLength,
        int tileM = 64,
        int tileN = 64,
        int tileK = 8)
        : LinearKernel(
              context,
              ElementType::Float32,
              buffer(),
              kernelOutput(),
              bufferSink(),
              inputVectorLength,
              outputVectorLength,
              tileM,
              tileN,
              tileK)
    {
    }

    ElementType getElementType() const { return elementType; }

    SlangResult loadParams(TorchParamReader& reader, bool loadBias = true);

    // Load from SafeTensors - weights expected in [Out, In] layout (same as PyTorch Linear)
    SlangResult loadParams(
        SafeTensorsReader& reader,
        UnownedStringSlice weightName,
        UnownedStringSlice biasName = UnownedStringSlice());

    TensorView allocateResultBuffer(ElementType elementType, int batchSize);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const EvalContext& ctx,
        LinearAlgorithm algorithm = LinearAlgorithm::Auto);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs,
        LinearAlgorithm algorithm = LinearAlgorithm::Auto)
    {
        return queueExecute(task, output, makeEvalContext(inputs), algorithm);
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs,
        LinearAlgorithm algorithm = LinearAlgorithm::Auto);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView inputVector,
        LinearAlgorithm algorithm = LinearAlgorithm::Auto)
    {
        return queueExecute(task, output, std::initializer_list<InputInfo>{inputVector}, algorithm);
    }
};

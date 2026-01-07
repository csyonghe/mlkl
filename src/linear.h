#pragma once
#include "elementwise.h"
#include "kernel-base.h"

// Linear Kernel: Out = In @ W^T + Bias (Weights expected in [In, Out] layout)
// W is of shape [outputVectorLength, inputVectorLength]
// In is of shape [batchSize, inputVectorLength]
// Out is of shape [batchSize, outputVectorLength]
// Underlying implementation uses tiled matrix multiplication.
class LinearKernel : public RefObject
{
private:
    ComPtr<rhi::IComputePipeline> pipeline;
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
        int tileM = 8,
        int tileN = 32,
        int tileK = 16);

    // Convenience constructor defaulting to Float32.
    LinearKernel(
        InferencingContext* context,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        int inputVectorLength,
        int outputVectorLength,
        int tileM = 8,
        int tileN = 32,
        int tileK = 16)
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
        int tileM = 2,
        int tileN = 8,
        int tileK = 16)
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

    SlangResult loadParams(TorchParamReader& reader, bool loadBiass = true);

    TensorView allocateResultBuffer(ElementType elementType, int batchSize);

    void queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const Dictionary<Expr, InputInfo>& inputs)
    {
        return queueExecute(task, output, makeEvalContext(inputs));
    }

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        const std::initializer_list<InputInfo>& inputs);

    void queueExecute(InferencingTask& task, TensorView output, TensorView inputVector)
    {
        return queueExecute(task, output, std::initializer_list<InputInfo>{inputVector});
    }
};

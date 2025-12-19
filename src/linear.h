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
    int tileM, tileN, tileK;
    int outputVectorLength; // Rows of weight matrix.
    int inputVectorLength;  // Columns of weight matrix.


    // For binding input buffer.
    ProgramNode inputProgram;
    ProgramNode outputProgram;
    SinkExpr sinkExpr;

public:
    ComPtr<rhi::IBuffer> weightsBuffer, biasesBuffer;

    LinearKernel(
        InferencingContext* context,
        Expr inputExpr,
        Expr outputExpr,
        SinkExpr sinkExpr,
        int inputVectorLength,
        int outputVectorLength,
        int tileM = 2,
        int tileN = 8,
        int tileK = 16);

    LinearKernel(
        InferencingContext* context,
        int inputVectorLength,
        int outputVectorLength,
        int tileM = 2,
        int tileN = 8,
        int tileK = 16)
        : LinearKernel(
              context,
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

    SlangResult loadParams(TorchParamReader& reader, bool loadBiass = true);

    BufferView allocateResultBuffer(int batchSize);

    void queueExecute(
        InferencingTask& task,
        BufferView output,
        int batchSize,
        const EvalContext& ctx);

    void queueExecute(
        InferencingTask& task,
        BufferView output,
        int batchSize,
        const Dictionary<Expr, InputInfo>& inputs)
    {
        return queueExecute(task, output, batchSize, makeEvalContext(inputs));
    }

    void queueExecute(
        InferencingTask& task,
        BufferView output,
        int batchSize,
        const std::initializer_list<InputInfo>& inputs)
    {
        return queueExecute(task, output, batchSize, EvalContext(&inputProgram, inputs));
    }

    void queueExecute(
        InferencingTask& task,
        BufferView output,
        BufferView inputVector,
        int batchSize = 1)
    {
        return queueExecute(
            task,
            output,
            batchSize,
            {InputInfo(Shape(batchSize, inputVectorLength), inputVector)});
    }
};

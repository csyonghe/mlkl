#pragma once
#include "elementwise.h"
#include "kernel-base.h"

class GatherKernel : public RefObject
{
private:
    InferencingContext* context;
    RefPtr<ElementwiseKernel> kernel;

    // The Expression handles
    Expr tableExpr;   // Represents the Weight Matrix
    Expr indicesExpr; // Represents the Input Labels

    // Internal storage for the weights
    ComPtr<rhi::IBuffer> weightsBuffer;
    int numClasses;
    int embeddingDim;

public:
    GatherKernel(InferencingContext* ctx, int nClasses, int dim);

    // Loads weights from the reader and uploads them to the GPU
    SlangResult loadParams(TorchParamReader& reader);

    // Helper to allocate [Batch, EmbeddingDim]
    TensorView allocateResultBuffer(ElementType elementType, int batchSize);

    // Executes the gather: Output[b, d] = Weights[Input[b], d]
    // NOTE: inputLabels must be a buffer of FLOATs (e.g. 3.0f for class 3)
    // because the generic IExpr system currently operates on floats.
    void queueExecute(InferencingTask& task, TensorView output, TensorView inputLabels);
};
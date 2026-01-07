#pragma once
#include "elementwise.h"
#include "kernel-base.h"

class SafeTensorsReader;

// Gather/Embedding lookup kernel: Output[b, d] = Weights[Input[b], d]
//
// FUSION OPPORTUNITY: Consider fusing into the consuming kernel!
// - For token + positional embeddings: gather both and add in single kernel
// - For embedding -> Linear: fuse gather into Linear input expression
//
// Only use standalone GatherKernel when:
// - The embedding result is used by multiple downstream kernels
// - No adjacent kernel supports input expression fusion
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

    // Load from SafeTensors - embeddings expected in [NumClasses, EmbeddingDim] layout
    SlangResult loadParams(SafeTensorsReader& reader, UnownedStringSlice weightName);

    // Helper to allocate [Batch, EmbeddingDim]
    TensorView allocateResultBuffer(ElementType elementType, int batchSize);

    // Executes the gather: Output[b, d] = Weights[Input[b], d]
    // inputLabels contains token/class indices (values are converted to int internally)
    void queueExecute(InferencingTask& task, TensorView output, TensorView inputLabels);
};
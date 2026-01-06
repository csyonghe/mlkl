#pragma once

#include "inference-context.h"
#include "kernels.h"
#include "torch-reader.h"

enum class UNetBlockKind
{
    Down,
    Up
};

class UNetBlock : public RefObject
{
public:
    RefPtr<InferencingContext> inferencingCtx;
    RefPtr<Conv2DKernel> conv1, conv2;
    RefPtr<Conv2DKernel> downTransform;
    RefPtr<TransposedConv2DKernel> upTransform;
    RefPtr<LinearKernel> timeEmbedTransform;
    RefPtr<BroadcastAddKernel> broadcastAdd;

public:
    int inChannels;
    int outChannels;
    UNetBlock(
        RefPtr<InferencingContext> inferencingCtx,
        UNetBlockKind kind,
        int inChannels,
        int outChannels,
        int timeEmbedDim);

    SlangResult loadParams(TorchParamReader& reader);

    void writeResult(const char* name, BufferView buffer);

    TensorView allocateResultBuffer(
        ElementType elementType,
        int inputWidth,
        int inputHeight,
        int batchSize);

    void queueExecute(
        InferencingTask& task,
        TensorView output,
        TensorView inputImage,
        TensorView timeEmbedding);
};

class UNetModel : public RefObject
{
public:
    RefPtr<InferencingContext> inferencingCtx;
    RefPtr<TimeEmbedingKernel> timeEmbedKernel;
    RefPtr<Conv2DKernel> initialConv;
    RefPtr<Conv2DKernel> finalConv;
    RefPtr<ConcatKernel> concat;

public:
    List<RefPtr<UNetBlock>> downBlocks;
    List<RefPtr<UNetBlock>> upBlocks;
    UNetModel(RefPtr<InferencingContext> inferencingCtx, int inputChannels, int outputChannels);

    SlangResult loadParams(TorchParamReader& reader);

    void queueExecute(
        InferencingTask& task,
        TensorView outputImage,
        TensorView inputImage,
        int timeStep);
};
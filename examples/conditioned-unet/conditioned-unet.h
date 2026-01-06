#pragma once

#include "cross-attention.h"
#include "inference-context.h"
#include "kernels.h"
#include "torch-reader.h"

using namespace Slang;

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

class ConditionedUNet : public RefObject
{
public:
    RefPtr<InferencingContext> context;

    // Model Components
    RefPtr<TimeEmbedingKernel> timeEmbed;
    RefPtr<Conv2DKernel> initialConv;
    RefPtr<Conv2DKernel> finalConv;
    RefPtr<GatherKernel> classEmbed;

    // Blocks
    List<RefPtr<UNetBlock>> downBlocks;
    List<RefPtr<UNetBlock>> upBlocks;
    RefPtr<CrossAttentionKernel> midAttn;

    // Utilities
    RefPtr<ConcatKernel> concat;

    // Configuration
    int inputChannels;
    int outputChannels;
    int timeEmbedDim;
    int contextDim;
    int baseChannels;
    int classCount;
    List<int> channelMultipliers;

public:
    ConditionedUNet(
        RefPtr<InferencingContext> ctx,
        int inChannels = 1,
        int outChannels = 1,
        int tDim = 32,
        int cDim = 128,
        int baseCh = 64,
        int numClasses = 10);

    // Loads weights from the binary dump.
    // NOTE: This assumes the standard diffmodel.py order.
    SlangResult loadParams(TorchParamReader& reader);

    // Runs the network.
    void queueExecute(
        InferencingTask& task,
        TensorView outputImage,
        TensorView inputImage,
        TensorView classLabels,
        int timeStep);
};
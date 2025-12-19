#pragma once

#include "core/slang-basic.h"
#include "core/slang-io.h"
#include "example-base/example-base.h"
#include "external/slang-rhi/include/slang-rhi.h"
#include "slang-com-ptr.h"
#include "slang-rhi/shader-cursor.h"
#include "slang.h"

#include <string>

using namespace Slang;

struct LinearLayerParams
{
    int inputFeatures;
    int outputFeatures;
    List<float> weights; // layout: outputFeatures x inputFeatures elements
    List<float> biases;  // layout: outputFeatures elements
};

struct BatchNorm2DLayerParams
{
    int numFeatures;
    List<float> weights;
    List<float> biases;
    List<float> runningMean;
    List<float> runningVar;
};

struct Conv2DLayerParams
{
    int inChannels;
    int outChannels;
    int kernelSize;
    List<float> weights; // layout: outChannels x inChannels x kernelSize x kernelSize elements
    List<float> biases;  // layout: outChannels elements

    void fuseBatchNorm(const BatchNorm2DLayerParams& bnParams, float epsilon = 1e-5f);
};

struct TransposedConv2DLayerParams
{
    int inChannels;
    int outChannels;
    int kernelSize;
    List<float> weights; // layout: outChannels x inChannels x kernelSize x kernelSize elements
    List<float> biases;  // layout: outChannels elements

    void fuseBatchNorm(const BatchNorm2DLayerParams& bnParams, float epsilon = 1e-5f);
};

class TorchParamReader
{
private:
    RefPtr<Stream> stream;

public:
    TorchParamReader(RefPtr<Stream> inputStream);
    TorchParamReader(String path);

    SlangResult readParams(List<float>& result, int count);

    // Read parameters for a linear layer, assuming weights layout to be [outFeatures, inFeatures].
    SlangResult readLinearLayer(
        int inFeatures,
        int outFeatures,
        bool hasBias,
        LinearLayerParams& params);

    // Read parameters for the conv2d layer, assuming weights layout to be
    // [InChannels, KernelSize, KernelSize, OutChannels].
    SlangResult readConv2DLayer(
        int inChannels,
        int outChannels,
        int kernelSize,
        Conv2DLayerParams& params);

    // Read parameters for the conv2d layer, assuming weights layout to be
    // [InChannels, KernelSize, KernelSize, OutChannels].
    SlangResult readTransposedConv2DLayer(
        int inChannels,
        int outChannels,
        int kernelSize,
        TransposedConv2DLayerParams& params);

    SlangResult readBatchNorm2DLayer(int numFeatures, BatchNorm2DLayerParams& params);
};
#pragma once

#include "core/slang-basic.h"
#include "core/slang-io.h"
#include "example-base/example-base.h"
#include "slang-rhi/shader-cursor.h"
#include "external/slang-rhi/include/slang-rhi.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <string>

using namespace Slang;

struct LinearLayerParams
{
    int inputFeatures;
    int outputFeatures;
    List<float> weights; // layout: outputFeatures x inputFeatures elements
    List<float> biases; // layout: outputFeatures elements
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
    SlangResult readParams(List<float>& result, int count);
public:
    TorchParamReader(RefPtr<Stream> inputStream);
    TorchParamReader(String path);

    // Read torch's exported parameters for a linear layer, and swap the weights layout to be [outFeatures, inFeatures].
    SlangResult readLinearLayer(int inFeatures, int outFeatures, LinearLayerParams& params);

    // Read torch's exported parameters for a conv2d layer, the layout is unchanged (remains [outChannels, inChannels, kernelSize, kernelSize]).
    SlangResult readConv2DLayer(int inChannels, int outChannels, int kernelSize, Conv2DLayerParams& params);

    // Read torch's exported parameters for a transposed conv2d layer, and swap the weights layout to be [outChannels, inChannels, kernelSize, kernelSize].
    SlangResult readTransposedConv2DLayer(int inChannels, int outChannels, int kernelSize, TransposedConv2DLayerParams& params);

    SlangResult readBatchNorm2DLayer(int numFeatures, BatchNorm2DLayerParams& params);
};
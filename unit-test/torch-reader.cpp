#include "torch-reader.h"
#include "inference-context.h"

TorchParamReader::TorchParamReader(RefPtr<Stream> inputStream)
    : stream(inputStream)
{
}

SlangResult TorchParamReader::readParams(List<float>& result, int count)
{
    result.setCount(count);
    logInfo("reading %d floats\n", count);
    SLANG_RETURN_ON_FAIL(stream->readExactly(result.getBuffer(), count * sizeof(float)));
    return SLANG_OK;
}

SlangResult TorchParamReader::readLinearLayer(int inFeatures, int outFeatures, LinearLayerParams& params)
{
    params.inputFeatures = inFeatures;
    params.outputFeatures = outFeatures;
    size_t weightCount = inFeatures * outFeatures;
    // Read weights
    SLANG_RETURN_ON_FAIL(readParams(params.weights, weightCount));
    // Read biases
    SLANG_RETURN_ON_FAIL(readParams(params.biases, outFeatures));
    return SLANG_OK;
}

SlangResult TorchParamReader::readConv2DLayer(int inChannels, int outChannels, int kernelSize, Conv2DLayerParams& params)
{
    params.inChannels = inChannels;
    params.outChannels = outChannels;
    params.kernelSize = kernelSize;
    size_t weightCount = inChannels * outChannels * kernelSize * kernelSize;
    // Read weights
    // assuming layout being [inChannels, kernelSize, kernelSize, outChannels]
    SLANG_RETURN_ON_FAIL(readParams(params.weights, weightCount));
    // Read biases
    SLANG_RETURN_ON_FAIL(readParams(params.biases, outChannels));
    return SLANG_OK;
}

SlangResult TorchParamReader::readTransposedConv2DLayer(int inChannels, int outChannels, int kernelSize, TransposedConv2DLayerParams& params)
{
    params.inChannels = inChannels;
    params.outChannels = outChannels;
    params.kernelSize = kernelSize;
    size_t weightCount = inChannels * outChannels * kernelSize * kernelSize;
    // Read weights
    // assuming layout being [inChannels, kernelSize, kernelSize, outChannels]
    SLANG_RETURN_ON_FAIL(readParams(params.weights, weightCount));
    // Read biases
    SLANG_RETURN_ON_FAIL(readParams(params.biases, outChannels));
    return SLANG_OK;
}

SlangResult TorchParamReader::readBatchNorm2DLayer(int numFeatures, BatchNorm2DLayerParams& params)
{
    params.numFeatures = numFeatures;
    // Read weights
    SLANG_RETURN_ON_FAIL(readParams(params.weights, numFeatures));
    // Read biases
    SLANG_RETURN_ON_FAIL(readParams(params.biases, numFeatures));
    // Read running mean
    SLANG_RETURN_ON_FAIL(readParams(params.runningMean, numFeatures));
    // Read running var
    SLANG_RETURN_ON_FAIL(readParams(params.runningVar, numFeatures));
    return SLANG_OK;
}

// This function works for BOTH Conv2d and TransposedConv2d
// provided that BOTH use the [In, K, K, Out] weight layout.
void fuseBatchNormUnified(
    List<float>& weights,
    List<float>& biases,
    const BatchNorm2DLayerParams& bnParams,
    int outChannels,
    float epsilon)
{
    // 1. Pre-calculate Scale Factors
    List<float> scales;
    scales.setCount(outChannels);

    for (int oc = 0; oc < outChannels; oc++)
    {
        float gamma = bnParams.weights[oc];
        float beta = bnParams.biases[oc];
        float mean = bnParams.runningMean[oc];
        float var = bnParams.runningVar[oc];

        float invStd = 1.0f / sqrtf(var + epsilon);
        scales[oc] = gamma * invStd;

        // Adjust Bias
        biases[oc] = (biases[oc] - mean) * scales[oc] + beta;
    }

    // 2. Adjust Weights (Linear Scan)
    // Works for [In, K, K, Out] because 'oc' is the fastest moving dimension.
    Index totalWeights = weights.getCount();

    for (Index i = 0; i < totalWeights; i++)
    {
        // The pattern of output channels repeats 0..N, 0..N
        int oc = i % outChannels;
        weights[i] *= scales[oc];
    }
}

void Conv2DLayerParams::fuseBatchNorm(const BatchNorm2DLayerParams& bnParams, float epsilon)
{
    fuseBatchNormUnified(weights, biases, bnParams, outChannels, epsilon);
}

void TransposedConv2DLayerParams::fuseBatchNorm(const BatchNorm2DLayerParams& bnParams, float epsilon)
{
    fuseBatchNormUnified(weights, biases, bnParams, outChannels, epsilon);
}

#include "torch-reader.h"

TorchParamReader::TorchParamReader(RefPtr<Stream> inputStream)
    : stream(inputStream)
{
}

SlangResult TorchParamReader::readLinearLayer(int inFeatures, int outFeatures, LinearLayerParams& params)
{
    params.inputFeatures = inFeatures;
    params.outputFeatures = outFeatures;
    size_t weightCount = inFeatures * outFeatures;
    params.weights.setCount(weightCount);
    params.biases.setCount(outFeatures);
    // Read weights
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.weights.getBuffer(), weightCount * sizeof(float)));
    // Swap layout from [inFeatures, outFeatures] to [outFeatures, inFeatures]
    List<float> swappedWeights;
    swappedWeights.setCount(weightCount);
    for (int o = 0; o < outFeatures; o++)
    {
        for (int i = 0; i < inFeatures; i++)
        {
            swappedWeights[o * inFeatures + i] = params.weights[i * outFeatures + o];
        }
    }
    params.weights = _Move(swappedWeights);
    // Read biases
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.biases.getBuffer(), outFeatures * sizeof(float)));
    return SLANG_OK;
}

SlangResult TorchParamReader::readConv2DLayer(int inChannels, int outChannels, int kernelSize, Conv2DLayerParams& params)
{
    params.inChannels = inChannels;
    params.outChannels = outChannels;
    params.kernelSize = kernelSize;
    size_t weightCount = inChannels * outChannels * kernelSize * kernelSize;
    params.weights.setCount(weightCount);
    params.biases.setCount(outChannels);
    // Read weights
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.weights.getBuffer(), weightCount * sizeof(float)));
    // Read biases
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.biases.getBuffer(), outChannels * sizeof(float)));
    return SLANG_OK;
}

SlangResult TorchParamReader::readTransposedConv2DLayer(int inChannels, int outChannels, int kernelSize, TransposedConv2DLayerParams& params)
{
    params.inChannels = inChannels;
    params.outChannels = outChannels;
    params.kernelSize = kernelSize;
    size_t weightCount = inChannels * outChannels * kernelSize * kernelSize;
    params.weights.setCount(weightCount);
    params.biases.setCount(outChannels);
    // Read weights
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.weights.getBuffer(), weightCount * sizeof(float)));
    // Swap layout from [inChannels, outChannels, kernelSize, kernelSize] to [outChannels, inChannels, kernelSize, kernelSize]
    List<float> swappedWeights;
    swappedWeights.setCount(weightCount);
    for (int o = 0; o < outChannels; o++)
    {
        for (int i = 0; i < inChannels; i++)
        {
            for (int ky = 0; ky < kernelSize; ky++)
            {
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    swappedWeights[o * (inChannels * kernelSize * kernelSize) +
                                   i * (kernelSize * kernelSize) +
                                   ky * kernelSize + kx] =
                        params.weights[i * (outChannels * kernelSize * kernelSize) +
                                       o * (kernelSize * kernelSize) +
                                       ky * kernelSize + kx];
                }
            }
        }
    }
    params.weights = _Move(swappedWeights);
    // Read biases
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.biases.getBuffer(), outChannels * sizeof(float)));
    return SLANG_OK;
}

SlangResult TorchParamReader::readBatchNorm2DLayer(int numFeatures, BatchNorm2DLayerParams& params)
{
    params.numFeatures = numFeatures;
    params.weights.setCount(numFeatures);
    params.biases.setCount(numFeatures);
    params.runningMean.setCount(numFeatures);
    params.runningVar.setCount(numFeatures);
    // Read weights
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.weights.getBuffer(), numFeatures * sizeof(float)));
    // Read biases
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.biases.getBuffer(), numFeatures * sizeof(float)));
    // Read running mean
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.runningMean.getBuffer(), numFeatures * sizeof(float)));
    // Read running var
    SLANG_RETURN_ON_FAIL(stream->readExactly(params.runningVar.getBuffer(), numFeatures * sizeof(float)));
    return SLANG_OK;
}

void Conv2DLayerParams::fuseBatchNorm(const BatchNorm2DLayerParams& bnParams, float epsilon)
{
    for (int oc = 0; oc < outChannels; oc++)
    {
        float gamma = bnParams.weights[oc];
        float beta = bnParams.biases[oc];
        float mean = bnParams.runningMean[oc];
        float var = bnParams.runningVar[oc];
        float invStd = 1.0f / sqrtf(var + epsilon);
        // Adjust weights
        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int ky = 0; ky < kernelSize; ky++)
            {
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    int weightIndex = oc * (inChannels * kernelSize * kernelSize) +
                        ic * (kernelSize * kernelSize) +
                        ky * kernelSize +
                        kx;
                    weights[weightIndex] *= gamma * invStd;
                }
            }
        }
        // Adjust biases
        biases[oc] = (biases[oc] - mean) * gamma * invStd + beta;
    }
}

void TransposedConv2DLayerParams::fuseBatchNorm(const BatchNorm2DLayerParams& bnParams, float epsilon)
{
    for (int oc = 0; oc < outChannels; oc++)
    {
        float gamma = bnParams.weights[oc];
        float beta = bnParams.biases[oc];
        float mean = bnParams.runningMean[oc];
        float var = bnParams.runningVar[oc];
        float invStd = 1.0f / sqrtf(var + epsilon);
        // Adjust weights
        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int ky = 0; ky < kernelSize; ky++)
            {
                for (int kx = 0; kx < kernelSize; kx++)
                {
                    int weightIndex = oc * (inChannels * kernelSize * kernelSize) +
                        ic * (kernelSize * kernelSize) +
                        ky * kernelSize +
                        kx;
                    weights[weightIndex] *= gamma * invStd;
                }
            }
        }
        // Adjust biases
        biases[oc] = (biases[oc] - mean) * gamma * invStd + beta;
    }
}

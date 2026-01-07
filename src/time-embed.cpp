#include "time-embed.h"
#include "safetensors-reader.h"

TimeEmbedingKernel::TimeEmbedingKernel(InferencingContext* context, int outputChannels)
    : context(context), outputChannels(outputChannels)
{
    pipeline = context->createComputePipeline(
        "computeTimeEmbedding",
        makeConstArrayViewSingle(String(outputChannels)));
}

SlangResult TimeEmbedingKernel::loadParams(TorchParamReader& reader)
{
    logInfo("Loading TimeEmbed Linear Layer: outputChannel %d\n", outputChannels);
    LinearLayerParams linearParams;
    reader.readLinearLayer(outputChannels, outputChannels, true, linearParams);
    biasesBuffer = context->createPersistentBuffer(linearParams.biases);
    if (!biasesBuffer)
        return SLANG_FAIL;
    weightsBuffer = context->createPersistentBuffer(linearParams.weights);
    if (!weightsBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

SlangResult TimeEmbedingKernel::loadParams(
    SafeTensorsReader& reader,
    UnownedStringSlice linear1WeightName,
    UnownedStringSlice linear1BiasName,
    UnownedStringSlice linear2WeightName,
    UnownedStringSlice linear2BiasName)
{
    logInfo("Loading TimeEmbed from SafeTensors: outputChannel %d\n", outputChannels);

    // For now, this kernel only uses one linear layer internally
    // (the sinusoidal embedding is computed on the fly)
    // Load the first linear layer weights (TimeEmbed uses Float32)
    List<uint8_t> weightsData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(linear1WeightName, ElementType::Float32, weightsData));

    List<uint8_t> biasData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(linear1BiasName, ElementType::Float32, biasData));

    weightsBuffer = context->createPersistentBuffer(weightsData.getBuffer(), weightsData.getCount());
    if (!weightsBuffer)
        return SLANG_FAIL;

    biasesBuffer = context->createPersistentBuffer(biasData.getBuffer(), biasData.getCount());
    if (!biasesBuffer)
        return SLANG_FAIL;

    return SLANG_OK;
}

TensorView TimeEmbedingKernel::allocateResultBuffer(ElementType elementType, int batchSize)
{
    return context->allocScratchTensor(
        elementType,
        Shape(batchSize, outputChannels),
        "time_embed_output");
}


struct TimeEmbeddingKernelParams
{
    rhi::DeviceAddress output;  // Layout: [OutputDim]
    rhi::DeviceAddress weights; // Layout: [EmbeddingDim * OutputDim] (Row-Major: In x Out)
    rhi::DeviceAddress biases;  // [OutputDim]
    uint32_t timeStep;
    uint32_t embeddingDim; // The size of the sin/cos vector (InChannels)
    float maxPeriod;       // Default 10000.0
    uint32_t batchSize;
};

void TimeEmbedingKernel::queueExecute(InferencingTask& task, TensorView output, uint32_t timeStep)
{
    output = output.ensureRank(2);

    TimeEmbeddingKernelParams params = {};
    params.output = output.getDeviceAddress();
    params.weights = weightsBuffer->getDeviceAddress();
    params.biases = biasesBuffer->getDeviceAddress();
    params.embeddingDim = outputChannels;
    params.maxPeriod = 10000.0f;
    params.timeStep = timeStep;
    params.batchSize = (uint32_t)output.shape.dims[0];

    // Dispatch: X=Channels, Y=Batch
    // Group size is 32 in X.
    uint32_t threadsX = (outputChannels + 31) / 32;
    task.dispatchKernel(pipeline, threadsX, params.batchSize, 1, params);
}
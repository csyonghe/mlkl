#include "group-norm.h"
#include "safetensors-reader.h"

GroupNormKernel::GroupNormKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr inputExpr,
    SinkExpr sinkExpr,
    int numChannels,
    int numGroups,
    float epsilon)
    : context(ctx)
    , elementType(elementType)
    , sinkExpr(sinkExpr)
    , numChannels(numChannels)
    , numGroups(numGroups)
    , epsilon(epsilon)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);

    // Pass 1: Reduce kernel using the generic ReduceKernel with GroupNormLayout
    reduceKernel = new ReduceKernel(ctx, elementType, inputExpr, ReductionLayoutType::GroupNorm);

    // Pass 2: Normalize kernel
    String normalizeArgs[] = {
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    normalizePipeline = context->createComputePipeline("groupNormNormalize", makeArrayView(normalizeArgs));
}

void GroupNormKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("GroupNormKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

SlangResult GroupNormKernel::loadParams(TorchParamReader& reader)
{
    logInfo("Loading GroupNorm Layer: channels=%d, groups=%d\n", numChannels, numGroups);

    // Read gamma (scale) - [numChannels]
    List<float> gamma;
    SLANG_RETURN_ON_FAIL(reader.readParams(gamma, numChannels));

    // Read beta (bias) - [numChannels]
    List<float> beta;
    SLANG_RETURN_ON_FAIL(reader.readParams(beta, numChannels));

    // Convert to kernel's element type and create buffers
    auto gammaData = convertFloatData(gamma, elementType);
    gammaBuffer = context->createPersistentBuffer(gammaData.getBuffer(), gammaData.getCount());

    auto betaData = convertFloatData(beta, elementType);
    betaBuffer = context->createPersistentBuffer(betaData.getBuffer(), betaData.getCount());

    return SLANG_OK;
}

SlangResult GroupNormKernel::loadParams(
    SafeTensorsReader& reader,
    UnownedStringSlice gammaName,
    UnownedStringSlice betaName)
{
    logInfo("Loading GroupNorm from SafeTensors: channels=%d, groups=%d\n", numChannels, numGroups);

    // Read gamma (scale/weight) directly to target element type
    List<uint8_t> gammaData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(gammaName, elementType, gammaData));
    gammaBuffer = context->createPersistentBuffer(gammaData.getBuffer(), gammaData.getCount());

    // Read beta (bias) directly to target element type
    List<uint8_t> betaData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(betaName, elementType, betaData));
    betaBuffer = context->createPersistentBuffer(betaData.getBuffer(), betaData.getCount());

    return SLANG_OK;
}

TensorView GroupNormKernel::allocateResultBuffer(
    ElementType elementType,
    int batchSize,
    int height,
    int width)
{
    return context->allocScratchTensor(
        elementType,
        Shape(batchSize, height, width, numChannels),
        "groupnorm_output");
}

void GroupNormKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const EvalContext& ctx)
{
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto bufferNode : inputProgram.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input");
    }

    // Resolve input shape - expecting NHWC layout
    auto inputShape = inputProgram.resolveShape(ctx);
    if (inputShape.getRank() != 4)
    {
        throw std::runtime_error("GroupNormKernel: Input must be a rank 4 tensor (NHWC).");
    }

    int batchSize = inputShape[0];
    int height = inputShape[1];
    int width = inputShape[2];
    int channels = inputShape[3];

    if (channels != numChannels)
    {
        throw std::runtime_error("GroupNormKernel: Input channels don't match configured channels.");
    }

    int channelsPerGroup = channels / numGroups;
    int totalGroups = batchSize * numGroups;

    // Get input tensor from context
    TensorView inputTensor;
    for (auto bufferNode : inputProgram.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
        {
            inputTensor = info->tensorView;
            break;
        }
    }

    // =========================================================================
    // Pass 1: Reduce - compute sum and sumSq per (batch, group)
    // =========================================================================
    auto statsBuffer = reduceKernel->allocateStatsBuffer(totalGroups);

    GroupNormLayoutParams layoutParams;
    layoutParams.batchSize = batchSize;
    layoutParams.height = height;
    layoutParams.width = width;
    layoutParams.numGroups = numGroups;
    layoutParams.channelsPerGroup = channelsPerGroup;

    reduceKernel->queueExecute(task, statsBuffer, inputTensor, layoutParams);

    // =========================================================================
    // Pass 2: Normalize - apply normalization to each element
    // =========================================================================
    {
        List<uint8_t> paramData;
        ParameterWriter writer{paramData};

        // Pack input expression
        inputProgram.pack(writer, ctx);

        // Pack sink expression
        SinkExprEvalContext sinkCtx;
        sinkCtx.outputBuffer = output;
        sinkCtx.logicalShape = Shape{batchSize, height, width, channels};
        sinkExpr.node->pack(writer, sinkCtx);

        // Pack scalar parameters
        writer.write<uint32_t>(batchSize);
        writer.write<uint32_t>(height);
        writer.write<uint32_t>(width);
        writer.write<uint32_t>(channels);
        writer.write<uint32_t>(numGroups);
        writer.write<float>(epsilon);

        // Pack buffer addresses (8-byte aligned)
        writer.align(8);
        writer.write(statsBuffer.getDeviceAddress());
        writer.write(gammaBuffer->getDeviceAddress());
        writer.write(betaBuffer->getDeviceAddress());
        writer.finish();

        // Dispatch: one thread per element, grouped into thread groups of 256
        uint32_t totalElements = batchSize * height * width * channels;
        uint32_t numNormalizeGroups = (totalElements + 255) / 256;
        task.dispatchKernel(normalizePipeline, numNormalizeGroups, 1, 1, paramData);
    }
}

void GroupNormKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input)
{
    EvalContext ctx;
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, InputInfo{input});
    return queueExecute(task, output, ctx);
}

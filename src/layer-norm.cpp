#include "layer-norm.h"

LayerNormKernel::LayerNormKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr inputExpr,
    SinkExpr sinkExpr,
    int numFeatures,
    float epsilon)
    : context(ctx)
    , elementType(elementType)
    , sinkExpr(sinkExpr)
    , numFeatures(numFeatures)
    , epsilon(epsilon)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);

    // Pass 1: Reduce kernel using LastDimLayout (reduce last dimension)
    reduceKernel = new ReduceKernel(ctx, elementType, inputExpr, ReductionLayoutType::LastDim);

    // Pass 2: Normalize kernel
    String normalizeArgs[] = {
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    normalizePipeline = context->createComputePipeline("layerNormNormalize", makeArrayView(normalizeArgs));
}

void LayerNormKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("LayerNormKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

SlangResult LayerNormKernel::loadParams(TorchParamReader& reader)
{
    logInfo("Loading LayerNorm Layer: features=%d\n", numFeatures);

    // Read gamma (scale) - [numFeatures]
    List<float> gamma;
    SLANG_RETURN_ON_FAIL(reader.readParams(gamma, numFeatures));

    // Read beta (bias) - [numFeatures]
    List<float> beta;
    SLANG_RETURN_ON_FAIL(reader.readParams(beta, numFeatures));

    // Convert to kernel's element type and create buffers
    auto gammaData = convertFloatData(gamma, elementType);
    gammaBuffer = context->createPersistentBuffer(gammaData.getBuffer(), gammaData.getCount());

    auto betaData = convertFloatData(beta, elementType);
    betaBuffer = context->createPersistentBuffer(betaData.getBuffer(), betaData.getCount());

    return SLANG_OK;
}

TensorView LayerNormKernel::allocateResultBuffer(ElementType elementType, int numRows)
{
    return context->allocScratchTensor(
        elementType,
        Shape(numRows, numFeatures),
        "layernorm_output");
}

void LayerNormKernel::queueExecute(
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

    // Resolve input shape - expecting 2D [NumRows, NumFeatures]
    auto inputShape = inputProgram.resolveShape(ctx);
    if (inputShape.getRank() != 2)
    {
        throw std::runtime_error("LayerNormKernel: Input must be a rank 2 tensor [NumRows, NumFeatures].");
    }

    int numRows = inputShape[0];
    int features = inputShape[1];

    if (features != numFeatures)
    {
        throw std::runtime_error("LayerNormKernel: Input features don't match configured features.");
    }

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
    // Pass 1: Reduce - compute sum and sumSq per row
    // =========================================================================
    auto statsBuffer = reduceKernel->allocateStatsBuffer(numRows);

    LastDimLayoutParams layoutParams;
    layoutParams.numRows = numRows;
    layoutParams.numCols = numFeatures;

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
        sinkCtx.logicalShape = Shape{numRows, numFeatures};
        sinkExpr.node->pack(writer, sinkCtx);

        // Pack scalar parameters
        writer.write<uint32_t>(numRows);
        writer.write<uint32_t>(numFeatures);
        writer.write<float>(epsilon);

        // Pack buffer addresses (8-byte aligned)
        writer.align(8);
        writer.write(statsBuffer.getDeviceAddress());
        writer.write(gammaBuffer->getDeviceAddress());
        writer.write(betaBuffer->getDeviceAddress());
        writer.finish();

        // Dispatch: one thread per element
        uint32_t totalElements = numRows * numFeatures;
        uint32_t numDispatchGroups = (totalElements + 255) / 256;
        task.dispatchKernel(normalizePipeline, numDispatchGroups, 1, 1, paramData);
    }
}

void LayerNormKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input)
{
    EvalContext ctx;
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, InputInfo{input});
    return queueExecute(task, output, ctx);
}


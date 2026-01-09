#include "linear.h"

#include "kernels.h"
#include "safetensors-reader.h"

using namespace Slang;

LinearKernel::LinearKernel(
    InferencingContext* context,
    ElementType elementType,
    Expr inputExpr,
    Expr outputExpr,
    SinkExpr sinkExpr,
    int inputVectorLength,
    int outputVectorLength,
    int tileM,
    int tileN,
    int tileK)
    : context(context)
    , elementType(elementType)
    , inputVectorLength(inputVectorLength)
    , outputVectorLength(outputVectorLength)
    , sinkExpr(sinkExpr)
    , tileM(tileM)
    , tileN(tileN)
    , tileK(tileK)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);
    outputProgram = compileExprToProgram(outputExpr, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);
    String specArgs[] = {
        String(tileM),
        String(tileN),
        String(tileK),
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType),
        outputProgram.getSlangTypeName(elementType)};
    tiledPipeline = context->createComputePipeline("linearTiled", makeConstArrayView(specArgs));
    
    // Create GEMV pipeline (uses same params struct but different dispatch)
    gemvPipeline = context->createComputePipeline("linearGemv", makeConstArrayView(specArgs));
}

void LinearKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("LinearKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

SlangResult LinearKernel::loadParams(TorchParamReader& reader, bool loadBias)
{
    logInfo(
        "Loading Linear Layer: inputSize=%d, outputSize=%d\n",
        inputVectorLength,
        outputVectorLength);
    LinearLayerParams params;
    SLANG_RETURN_ON_FAIL(
        reader.readLinearLayer(inputVectorLength, outputVectorLength, loadBias, params));
    
    // Convert weights/biases to kernel's element type
    auto weightsData = convertFloatData(params.weights, elementType);
    weightsBuffer = context->createPersistentBuffer(weightsData.getBuffer(), weightsData.getCount());
    
    if (loadBias)
    {
        auto biasData = convertFloatData(params.biases, elementType);
        biasesBuffer = context->createPersistentBuffer(biasData.getBuffer(), biasData.getCount());
    }
    return SLANG_OK;
}

SlangResult LinearKernel::loadParams(
    SafeTensorsReader& reader,
    UnownedStringSlice weightName,
    UnownedStringSlice biasName)
{
    logInfo(
        "Loading Linear Layer from SafeTensors: inputSize=%d, outputSize=%d\n",
        inputVectorLength,
        outputVectorLength);

    // Verify shape
    const SafeTensorInfo* weightInfo = reader.getTensorInfo(weightName);
    if (!weightInfo ||
        weightInfo->shape.getRank() != 2 ||
        weightInfo->shape[0] != outputVectorLength ||
        weightInfo->shape[1] != inputVectorLength)
    {
        return SLANG_E_INVALID_ARG;
    }

    // Read weights directly to target element type
    List<uint8_t> weightsData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(weightName, elementType, weightsData));
    weightsBuffer = context->createPersistentBuffer(weightsData.getBuffer(), weightsData.getCount());

    // Read bias if provided
    if (biasName.getLength() > 0 && reader.hasTensor(biasName))
    {
        List<uint8_t> biasData;
        SLANG_RETURN_ON_FAIL(reader.readTensor(biasName, elementType, biasData));
        biasesBuffer = context->createPersistentBuffer(biasData.getBuffer(), biasData.getCount());
    }

    return SLANG_OK;
}


TensorView LinearKernel::allocateResultBuffer(ElementType elementType, int batchSize)
{
    auto outputBuffer = context->allocScratchTensor(
        elementType,
        Shape(batchSize, outputVectorLength),
        "linear_output");
    return outputBuffer;
}

void LinearKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const EvalContext& ctx,
    LinearAlgorithm algorithm)
{
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto bufferNode : inputProgram.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input");
    }

    // Validate Input/Output Shapes.
    auto inputShape = inputProgram.resolveShape(ctx);
    int batchSize = inputShape[0];
    if (inputShape.getRank() != 2)
    {
        throw std::runtime_error("Input tensor must be rank 2 (batch_size, input_size).");
    }
    if (inputShape.getDims().getLast() != inputVectorLength)
    {
        throw std::runtime_error("Input tensor's last dimension does not match input size.");
    }
    if (inputShape.getRank() > 1)
    {
        if (inputShape.getDims()[0] != batchSize)
            throw std::runtime_error("Input tensor's batch size does not match output batch size.");
    }
    else
    {
        if (batchSize != 1)
            throw std::runtime_error("Input tensor is missing batch dimension.");
    }

    int outputSize = outputVectorLength;

    // 2. Prepare Parameter Data (same for both algorithms)
    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // A. Pack inputProgram (TIn)
    inputProgram.pack(writer, ctx);

    // B. Pack outputProgram (FOut)
    outputProgram.pack(writer, ctx);

    // C. Pack Sink (TSink)
    SinkExprEvalContext sinkCtx;
    sinkCtx.outputBuffer = output;
    sinkCtx.logicalShape = Shape{batchSize, outputSize};
    this->sinkExpr.node->pack(writer, sinkCtx);

    // D. Pack Weight and Bias Pointers (8-byte aligned)
    writer.align(8);
    writer.write(weightsBuffer->getDeviceAddress());
    writer.write(biasesBuffer ? biasesBuffer->getDeviceAddress() : (uint64_t)0);

    // E. Pack Scalars (M, K, N, has_bias)
    writer.write((uint32_t)batchSize);         // M
    writer.write((uint32_t)inputVectorLength); // K
    writer.write((uint32_t)outputSize);        // N
    writer.write((uint32_t)(biasesBuffer ? 1 : 0));
    writer.finish();

    // Determine effective algorithm
    LinearAlgorithm effectiveAlgorithm = algorithm;
    if (algorithm == LinearAlgorithm::Auto)
    {
        // Use GEMV for small batch sizes (<=8), tiled GEMM for larger
        static const int GEMV_MAX_BATCH = 8;
        effectiveAlgorithm = (batchSize <= GEMV_MAX_BATCH)
            ? LinearAlgorithm::Gemv
            : LinearAlgorithm::Tiled;
    }

    // Dispatch based on algorithm
    if (effectiveAlgorithm == LinearAlgorithm::Gemv)
    {
        // Warp-cooperative GEMV dispatch:
        // - Block has 8 warps (256 threads)
        // - Each warp handles 4 output columns
        // - Each block handles 32 output columns total
        // - Dispatch: x = ceil(N / 32), y = batchSize, z = 1
        static const int GEMV_COLS_PER_BLOCK = 32;  // Must match slang constants
        int gridX = (outputSize + GEMV_COLS_PER_BLOCK - 1) / GEMV_COLS_PER_BLOCK;
        task.dispatchKernel(gemvPipeline, (uint32_t)gridX, (uint32_t)batchSize, 1, paramData);
    }
    else
    {
        // Tiled GEMM
        int gridX = (outputSize + tileN - 1) / tileN;
        int gridY = (batchSize + tileM - 1) / tileM;
        task.dispatchKernel(tiledPipeline, (uint32_t)gridX, (uint32_t)gridY, 1, paramData);
    }
}

void LinearKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const std::initializer_list<InputInfo>& inputs,
    LinearAlgorithm algorithm)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    for (auto bufferNode : outputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    return queueExecute(task, output, ctx, algorithm);
}
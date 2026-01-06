#include "linear.h"

#include "kernels.h"

using namespace Slang;

LinearKernel::LinearKernel(
    InferencingContext* context,
    Expr inputExpr,
    Expr outputExpr,
    SinkExpr sinkExpr,
    int inputVectorLength,
    int outputVectorLength,
    int tileM,
    int tileN,
    int tileK)
    : context(context)
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

    String specArgs[] = {
        String(tileM),
        String(tileN),
        String(tileK),
        inputProgram.getSlangTypeName(),
        sinkExpr.node->getSlangTypeName(),
        outputProgram.getSlangTypeName()};
    pipeline = context->createComputePipeline("linearTiled", makeConstArrayView(specArgs));
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
    weightsBuffer = context->createPersistentBuffer(params.weights);
    if (loadBias)
        biasesBuffer = context->createPersistentBuffer(params.biases);
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

void LinearKernel::queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx)
{
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

    // 1. Calculate Grid Dimensions
    // tileN covers the output features (N), tileM covers the batch/rows (M)
    int outputSize = outputVectorLength;
    int gridX = (outputSize + tileN - 1) / tileN;
    int gridY = (batchSize + tileM - 1) / tileM;

    // 2. Prepare Parameter Data
    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // --- PACKING ACCORDING TO LinearParams<TIn, TSink, FOut> ---

    // A. Pack inputProgram (TIn)
    inputProgram.pack(writer, ctx);

    // B. Pack outputProgram (FOut)
    outputProgram.pack(writer, ctx);


    // C. Pack Sink (TSink)
    // Since SinkExpr isn't a ProgramNode member, we pack it using the node logic.
    // Note: You should ideally store SinkExpr in your header or use a sinkProgram.
    // Assuming you have access to the sink node used in the constructor:
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

    // 3. Dispatch
    task.dispatchKernel(pipeline, (uint32_t)gridX, (uint32_t)gridY, 1, paramData);
}

void LinearKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const std::initializer_list<InputInfo>& inputs)
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
    return queueExecute(task, output, ctx);
}
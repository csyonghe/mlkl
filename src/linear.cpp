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


BufferView LinearKernel::allocateResultBuffer(int batchSize)
{
    auto outputBuffer = context->allocScratchBuffer(
        batchSize * outputVectorLength * sizeof(float),
        "linear_output");
    return outputBuffer;
}

void LinearKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    int batchSize,
    const EvalContext& ctx)
{
    // 1. Calculate Grid Dimensions
    // tileN covers the output features (N), tileM covers the batch/rows (M)
    // The outputSize (N) is implicitly handled by the dispatch grid and weights
    // We'll calculate N based on the weights buffer size if not stored,
    // but usually, it's safer to have outputSize as a member.
    // For now, let's assume outputSize is (weightsBuffer->getSize() / sizeof(float)) /
    // inputVectorLength;
    int outputSize = (int)((weightsBuffer->getDesc().size / sizeof(float)) / inputVectorLength);

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
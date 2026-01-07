#include "softmax.h"

SoftmaxKernel::SoftmaxKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr inputExpr,
    SinkExpr sinkExpr)
    : context(ctx)
    , elementType(elementType)
    , sinkExpr(sinkExpr)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);
    String specArgs[] = {
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        sinkExpr.node->getSlangTypeName(elementType)};
    pipeline = context->createComputePipeline("softmax", makeArrayView(specArgs));
}

void SoftmaxKernel::validateTensorElementType(const TensorView& tv, const char* name) const
{
    if (tv && tv.elementType != elementType)
    {
        throw InvalidOperationException(
            String("SoftmaxKernel: ") + name + " element type mismatch. Expected " +
            getSlangElementTypeName(elementType) + " but got " +
            getSlangElementTypeName(tv.elementType));
    }
}

TensorView SoftmaxKernel::allocateResultBuffer(ElementType elementType, int rows, int cols)
{
    return context->allocScratchTensor(elementType, Shape(rows, cols), "Softmax_Result");
}

void SoftmaxKernel::queueExecute(InferencingTask& task, TensorView output, const EvalContext& ctx)
{
    // Validate element types
    validateTensorElementType(output, "output");
    for (auto bufferNode : inputProgram.bufferNodes)
    {
        if (auto info = ctx.inputs.tryGetValue(bufferNode))
            validateTensorElementType(info->tensorView, "input");
    }

    // Resolve input shape
    auto inputShape = inputProgram.resolveShape(ctx);
    if (inputShape.getRank() != 2)
    {
        throw std::runtime_error("SoftmaxKernel: Input must be a rank 2 tensor.");
    }

    int rows = inputShape[0];
    int cols = inputShape[1];

    // Build parameter buffer
    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack input expression
    inputProgram.pack(writer, ctx);

    // Pack sink expression
    SinkExprEvalContext sinkCtx;
    sinkCtx.outputBuffer = output;
    sinkCtx.logicalShape = Shape{rows, cols};
    sinkExpr.node->pack(writer, sinkCtx);

    // Pack scalar parameters
    writer.write<uint32_t>(cols);  // stride
    writer.write<uint32_t>(rows);  // rowCount
    writer.finish();

    // Dispatch
    uint32_t groupX = (rows + 255) / 256;
    task.dispatchKernel(pipeline, groupX, 1, 1, paramData);
}

void SoftmaxKernel::queueExecute(
    InferencingTask& task,
    TensorView output,
    const std::initializer_list<InputInfo>& inputs)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("SoftmaxKernel: insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    return queueExecute(task, output, ctx);
}

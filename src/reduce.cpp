#include "reduce.h"

static const char* getLayoutTypeName(ReductionLayoutType type)
{
    switch (type)
    {
    case ReductionLayoutType::LastDim:
        return "LastDimLayout";
    case ReductionLayoutType::GroupNorm:
        return "GroupNormLayout";
    case ReductionLayoutType::Axis:
        return "AxisLayout";
    default:
        return "LastDimLayout";
    }
}

ReduceKernel::ReduceKernel(
    InferencingContext* ctx,
    ElementType elementType,
    Expr inputExpr,
    ReductionLayoutType layoutType)
    : context(ctx), elementType(elementType), layoutType(layoutType)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);

    String elemTypeName = getSlangElementTypeName(elementType);
    String specArgs[] = {
        elemTypeName,
        inputProgram.getSlangTypeName(elementType),
        getLayoutTypeName(layoutType)};
    pipeline = context->createComputePipeline("reduce", makeArrayView(specArgs));
}

BufferView ReduceKernel::allocateStatsBuffer(int numGroups)
{
    int elementSize = getElementTypeSize(elementType);
    int statsSize = numGroups * 2 * elementSize;
    return context->allocScratchBuffer(statsSize, "reduce_stats");
}

void ReduceKernel::queueExecute(
    InferencingTask& task,
    BufferView statsOutput,
    TensorView input,
    const LastDimLayoutParams& layout)
{
    if (layoutType != ReductionLayoutType::LastDim)
    {
        throw std::runtime_error("ReduceKernel: Layout type mismatch (expected LastDim)");
    }

    EvalContext ctx;
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, InputInfo{input});

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack input expression
    inputProgram.pack(writer, ctx);

    // Pack LastDimLayout fields
    writer.write<uint32_t>(layout.numRows);
    writer.write<uint32_t>(layout.numCols);

    // Pack numGroups
    writer.write<uint32_t>(layout.numRows);

    // Pack stats buffer address (8-byte aligned)
    writer.align(8);
    writer.write(statsOutput.getDeviceAddress());
    writer.finish();

    // Dispatch: one thread group per row
    task.dispatchKernel(pipeline, layout.numRows, 1, 1, paramData);
}

void ReduceKernel::queueExecute(
    InferencingTask& task,
    BufferView statsOutput,
    const EvalContext& ctx,
    const GroupNormLayoutParams& layout)
{
    if (layoutType != ReductionLayoutType::GroupNorm)
    {
        throw std::runtime_error("ReduceKernel: Layout type mismatch (expected GroupNorm)");
    }

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack input expression
    inputProgram.pack(writer, ctx);

    // Pack GroupNormLayout fields
    writer.write<uint32_t>(layout.batchSize);
    writer.write<uint32_t>(layout.height);
    writer.write<uint32_t>(layout.width);
    writer.write<uint32_t>(layout.numGroups);
    writer.write<uint32_t>(layout.channelsPerGroup);

    // Pack numGroups
    int numGroups = layout.batchSize * layout.numGroups;
    writer.write<uint32_t>(numGroups);

    // Pack stats buffer address (8-byte aligned)
    writer.align(8);
    writer.write(statsOutput.getDeviceAddress());
    writer.finish();

    // Dispatch: one thread group per (batch, group) pair
    task.dispatchKernel(pipeline, numGroups, 1, 1, paramData);
}

void ReduceKernel::queueExecute(
    InferencingTask& task,
    BufferView statsOutput,
    const std::initializer_list<InputInfo>& inputs,
    const GroupNormLayoutParams& layout)
{
    EvalContext ctx;
    auto iter = inputs.begin();
    auto consume = [&]()
    {
        if (iter == inputs.end())
            throw std::runtime_error("ReduceKernel: insufficient input buffers.");
        return *(iter++);
    };
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, consume());
    return queueExecute(task, statsOutput, ctx, layout);
}

void ReduceKernel::queueExecute(
    InferencingTask& task,
    BufferView statsOutput,
    TensorView input,
    const GroupNormLayoutParams& layout)
{
    return queueExecute(task, statsOutput, std::initializer_list<InputInfo>{input}, layout);
}

void ReduceKernel::queueExecute(
    InferencingTask& task,
    BufferView statsOutput,
    TensorView input,
    const AxisLayoutParams& layout)
{
    if (layoutType != ReductionLayoutType::Axis)
    {
        throw std::runtime_error("ReduceKernel: Layout type mismatch (expected Axis)");
    }

    EvalContext ctx;
    for (auto bufferNode : inputProgram.bufferNodes)
        ctx.inputs.add(bufferNode, InputInfo{input});

    // Compute numGroups (product of non-reduced dimensions)
    int rank = layout.shape.getRank();
    int numGroups = 1;
    for (int d = 0; d < rank; d++)
    {
        if (d != layout.axis)
            numGroups *= layout.shape[d];
    }

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};

    // Pack input expression
    inputProgram.pack(writer, ctx);

    // Pack AxisLayout fields (8 shape values, padded with 1s)
    for (int d = 0; d < 8; d++)
        writer.write<uint32_t>(d < rank ? layout.shape[d] : 1);
    writer.write<uint32_t>(rank);
    writer.write<uint32_t>(layout.axis);
    writer.write<uint32_t>(layout.elementsPerGroup);

    // Pack numGroups
    writer.write<uint32_t>(numGroups);

    // Pack stats buffer address (8-byte aligned)
    writer.align(8);
    writer.write(statsOutput.getDeviceAddress());
    writer.finish();

    // Dispatch
    task.dispatchKernel(pipeline, numGroups, 1, 1, paramData);
}

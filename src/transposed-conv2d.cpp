#include "transposed-conv2d.h"

TransposedConv2DKernel::TransposedConv2DKernel(
    InferencingContext* context,
    int tileSize,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    Expr inputExpr,
    Expr outputExpr,
    SinkExpr sinkExpr,
    String name)
    : context(context)
    , tileSize(tileSize)
    , stride(stride)
    , kernelSize(kernelSize)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , sinkExpr(sinkExpr)
    , name(name)
{
    int globalRegCounter = 0;
    inputProgram = compileExprToProgram(inputExpr, &globalRegCounter);
    outputProgram = compileExprToProgram(outputExpr, &globalRegCounter);

    String specArgs[] = {
        String(tileSize),
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        inputProgram.getSlangTypeName(),
        outputProgram.getSlangTypeName(),
        sinkExpr.node->getSlangTypeName()};
    pipeline =
        context->createComputePipeline("tiledTransposedConvolution", makeArrayView(specArgs));

    // Create Flat Pipeline
    String flatArgs[] = {
        String(kernelSize),
        String(stride),
        String(inChannels),
        String(outChannels),
        inputProgram.getSlangTypeName(),
        outputProgram.getSlangTypeName(),
        sinkExpr.node->getSlangTypeName()};
    // Note: tileSize is NOT needed for flat kernel generic
    flatPipeline =
        context->createComputePipeline("flatTransposedConvolution", makeArrayView(flatArgs));
}

TransposedConv2DKernel::TransposedConv2DKernel(
    InferencingContext* context,
    int tileSize,
    int kernelSize,
    int stride,
    int inChannels,
    int outChannels,
    Expr outputExpr,
    String name)
    : TransposedConv2DKernel(
          context,
          tileSize,
          kernelSize,
          stride,
          inChannels,
          outChannels,
          buffer(),
          outputExpr,
          bufferSink(),
          name)
{
}

SlangResult TransposedConv2DKernel::loadParams(TorchParamReader& reader)
{
    logInfo(
        "Loading TransposedConv2D Layer: inChannels=%d, outChannels=%d, kernelSize=%d\n",
        inChannels,
        outChannels,
        kernelSize);
    TransposedConv2DLayerParams convParams;
    SLANG_RETURN_ON_FAIL(
        reader.readTransposedConv2DLayer(inChannels, outChannels, kernelSize, convParams));
    return loadParams(
        kernelSize,
        outChannels,
        convParams.weights.getBuffer(),
        convParams.biases.getBuffer());
}

SlangResult TransposedConv2DKernel::loadParams(
    int kernelSize,
    int outputChannelCount,
    float* weightsData,
    float* biasesData)
{
    weightsBuffer = context->createPersistentBuffer(
        weightsData,
        kernelSize * kernelSize * inChannels * outputChannelCount * sizeof(float));
    if (!weightsBuffer)
        return SLANG_FAIL;
    biasesBuffer = context->createPersistentBuffer(biasesData, outputChannelCount * sizeof(float));
    if (!biasesBuffer)
        return SLANG_FAIL;
    return SLANG_OK;
}

TensorView TransposedConv2DKernel::allocateResultBuffer(
    ElementType elementType,
    int inputWidth,
    int inputHeight,
    int padding,
    int batchSize)
{
    int outputWidth = (inputWidth - 1) * stride - 2 * padding + kernelSize;
    int outputHeight = (inputHeight - 1) * stride - 2 * padding + kernelSize;

    // Updated name for debug clarity
    String resultBufferName = StringBuilder() << name << "_" + outputWidth << "x" << outputHeight
                                              << "x" << outChannels << "_B" << batchSize;

    auto outputBuffer = context->allocScratchTensor(
        elementType,
        Shape(batchSize, outputHeight, outputWidth, outChannels),
        resultBufferName.getBuffer());
    return outputBuffer;
}


struct TransposedConv2DKernelParams
{
    rhi::DeviceAddress weights;
    rhi::DeviceAddress biases;
    rhi::DeviceAddress inputImage;
    rhi::DeviceAddress outputImage;
    int inputImageWidth;
    int inputImageHeight;
    int outputImageWidth;
    int outputImageHeight;
    int stride;
    int padding;
    int batchSize;
};


void TransposedConv2DKernel::queueExecute(
    InferencingTask& task,
    EvalContext& ctx,
    TensorView output,
    int padding)
{
    auto inputShape = inputProgram.resolveShape(ctx);
    if (inputShape.getRank() != 4)
        throw std::runtime_error("TransposedConv2DKernel: Input rank must be 4.");
    int batchSize = inputShape[0];
    int inputHeight = inputShape[1];
    int inputWidth = inputShape[2];
    int inputChannels = inputShape[3];
    if (inputChannels != inChannels)
    {
        throw std::runtime_error("TransposedConv2DKernel: Input channel count mismatch.");
    }
    int outputWidth = (inputWidth - 1) * stride - 2 * padding + kernelSize;
    int outputHeight = (inputHeight - 1) * stride - 2 * padding + kernelSize;

    List<uint8_t> paramData;
    ParameterWriter writer{paramData};
    inputProgram.pack(writer, ctx);
    outputProgram.pack(writer, ctx);

    SinkExprEvalContext sinkContext;
    sinkContext.outputBuffer = output;
    sinkContext.logicalShape = Shape(batchSize, outputHeight, outputWidth, outChannels);
    sinkExpr.node->pack(writer, sinkContext);

    writer.align(8);
    writer.write(weightsBuffer->getDeviceAddress());
    writer.write(biasesBuffer->getDeviceAddress());
    writer.write<int>(inputWidth);
    writer.write<int>(inputHeight);
    writer.write<int>(outputWidth);
    writer.write<int>(outputHeight);
    writer.write<int>(stride);
    writer.write<int>(padding);
    writer.write<int>(batchSize); // Set Batch Size
    writer.finish();

    if (outputWidth * outputHeight <= 1024)
    {
        // Dispatch 1D Grid (Flat Kernel)
        // Global Index covers Batch * H * W * C
        int totalElementsPerImage = outputWidth * outputHeight * outChannels;
        int totalElements = totalElementsPerImage * batchSize;

        int groupSize = 256;
        int numGroups = (totalElements + groupSize - 1) / groupSize;

        task.dispatchKernel(flatPipeline, numGroups, 1, 1, paramData);
    }
    else
    {
        // Dispatch 3D Grid (Tiled Kernel)
        // Z Dimension covers (ChannelGroups * Batch)
        static const int batchOutChannels = 32;
        int zBlocksPerImage = (outChannels + batchOutChannels - 1) / batchOutChannels;
        int totalZBlocks = zBlocksPerImage * batchSize;

        task.dispatchKernel(
            pipeline,
            (outputWidth + tileSize - 1) / tileSize,
            (outputHeight + tileSize - 1) / tileSize,
            totalZBlocks,
            paramData);
    }
}


void TransposedConv2DKernel::queueExecute(
    InferencingTask& task,
    TensorView outputImage,
    TensorView inputImage,
    int padding)
{
    EvalContext ctx;
    if (inputProgram.bufferNodes.getCount() > 1)
    {
        throw std::runtime_error("insufficient input buffers for TransposeConv2D kernel.");
    }
    if (inputProgram.bufferNodes.getCount() < 1)
    {
        throw std::runtime_error("The TransposeConv2D kernel does not take any input buffers.");
    }
    ctx.inputs.add(inputProgram.bufferNodes[0], inputImage);

    queueExecute(task, ctx, outputImage, padding);
}

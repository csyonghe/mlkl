#include "gather.h"
#include "safetensors-reader.h"

ElementType kGatherKernelWeightsElementType = ElementType::Float32;

GatherKernel::GatherKernel(InferencingContext* ctx, int nClasses, int dim)
    : context(ctx), numClasses(nClasses), embeddingDim(dim)
{
    // Define the Compute Graph
    tableExpr = buffer();   // Input 0: The Weights
    indicesExpr = buffer(); // Input 1: The Labels

    // Root = Gather(Table, Indices)
    Expr root = gather(tableExpr, indicesExpr);

    // Compile the kernel
    kernel = new ElementwiseKernel(ctx, root);
}

SlangResult GatherKernel::loadParams(TorchParamReader& reader)
{
    // 1. Read Weights into temporary CPU memory
    List<float> weightsRaw;
    weightsRaw.setCount(numClasses * embeddingDim);

    if (SLANG_FAILED(reader.readParams(weightsRaw, weightsRaw.getCount())))
    {
        return SLANG_FAIL;
    }

    // 2. Upload to GPU Persistent Buffer
    // We store the BufferView in the class so we can bind it every frame
    weightsBuffer = context->createPersistentBuffer(
        weightsRaw.getBuffer(),
        weightsRaw.getCount() * sizeof(float),
        "GatherWeights");

    return SLANG_OK;
}

SlangResult GatherKernel::loadParams(SafeTensorsReader& reader, UnownedStringSlice weightName)
{
    // Verify shape
    const SafeTensorInfo* info = reader.getTensorInfo(weightName);
    if (!info ||
        info->shape.getRank() != 2 ||
        info->shape[0] != numClasses ||
        info->shape[1] != embeddingDim)
    {
        return SLANG_E_INVALID_ARG;
    }

    // Read weights directly to target element type (F32 for GatherKernel)
    List<uint8_t> weightsData;
    SLANG_RETURN_ON_FAIL(reader.readTensor(weightName, kGatherKernelWeightsElementType, weightsData));

    // Upload to GPU
    weightsBuffer = context->createPersistentBuffer(
        weightsData.getBuffer(),
        weightsData.getCount(),
        "GatherWeights");

    return SLANG_OK;
}

TensorView GatherKernel::allocateResultBuffer(ElementType elementType, int batchSize)
{
    size_t size = (size_t)batchSize * embeddingDim * sizeof(float);
    return context->allocScratchTensor(elementType, Shape(batchSize, embeddingDim), "GatherOutput");
}

void GatherKernel::queueExecute(InferencingTask& task, TensorView output, TensorView inputLabels)
{
    // Validate shapes.
    if (inputLabels.shape.getRank() != 1)
    {
        throw std::runtime_error("Input labels must be a 1D tensor [BatchSize]");
    }
    if (output.shape.getRank() != 2)
    {
        throw std::runtime_error("Output tensor must be a 2D tensor [BatchSize, EmbeddingDim]");
    }
    if (output.shape[0] != inputLabels.shape[0])
    {
        throw std::runtime_error(
            "Output tensor first dimension (BatchSize) must match input labels size.");
    }
    if (output.shape[1] != embeddingDim)
    {
        throw std::runtime_error("Output tensor second dimension must match embedding dimension.");
    }

    Dictionary<Expr, InputInfo> inputs;

    // Bind the Weights (Table)
    // Shape: [NumClasses, EmbeddingDim]
    inputs.add(
        tableExpr,
        TensorView(
            weightsBuffer,
            kGatherKernelWeightsElementType,
            Shape{numClasses, embeddingDim}));

    // Bind the Labels (Indices)
    // Shape: [BatchSize]
    inputs.add(indicesExpr, inputLabels);

    // Execute
    kernel->queueExecute(task, output, inputs);
}
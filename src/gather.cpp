#include "gather.h"

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

BufferView GatherKernel::allocateResultBuffer(int batchSize)
{
    size_t size = (size_t)batchSize * embeddingDim * sizeof(float);
    return context->allocScratchBuffer(size, "GatherOutput");
}

void GatherKernel::queueExecute(
    InferencingTask& task,
    BufferView output,
    BufferView inputLabels,
    int batchSize)
{
    Dictionary<Expr, InputInfo> inputs;

    // Bind the Weights (Table)
    // Shape: [NumClasses, EmbeddingDim]
    inputs.add(tableExpr, InputInfo(Shape{numClasses, embeddingDim}, weightsBuffer));

    // Bind the Labels (Indices)
    // Shape: [BatchSize]
    inputs.add(indicesExpr, InputInfo(Shape{batchSize}, inputLabels));

    // Execute
    kernel->queueExecute(task, output, inputs);
}
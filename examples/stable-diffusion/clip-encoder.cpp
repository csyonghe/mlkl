#include "clip-encoder.h"

// ============================================================================
// CLIPMLP
// ============================================================================

CLIPMLP::CLIPMLP(RefPtr<InferencingContext> ctx, int hiddenSize, int intermediateSize)
    : ctx(ctx), hiddenSize(hiddenSize), intermediateSize(intermediateSize)
{
    // fc1: hidden_size -> intermediate_size with fused QuickGELU
    auto input1 = buffer();
    fc1 = new LinearKernel(
        ctx,
        ElementType::Float32,
        input1,
        quickGelu(kernelOutput()),
        bufferSink(),
        hiddenSize,
        intermediateSize);

    // fc2: intermediate_size -> hidden_size
    fc2 = new LinearKernel(ctx, intermediateSize, hiddenSize);
}

SlangResult CLIPMLP::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // CLIP uses fc1, fc2 naming
    SLANG_RETURN_ON_FAIL(fc1->loadParams(
        reader,
        (prefix + "fc1.weight").getUnownedSlice(),
        (prefix + "fc1.bias").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(fc2->loadParams(
        reader,
        (prefix + "fc2.weight").getUnownedSlice(),
        (prefix + "fc2.bias").getUnownedSlice()));

    return SLANG_OK;
}

void CLIPMLP::queueExecute(InferencingTask& task, TensorView output, TensorView input)
{
    auto fc1Out =
        ctx->allocScratchTensor(ElementType::Float32, Shape(input.shape[0], intermediateSize));
    fc1->queueExecute(task, fc1Out, input); // fc1 with fused QuickGELU
    fc2->queueExecute(task, output, fc1Out);
}

// ============================================================================
// CLIPSelfAttention
// ============================================================================

CLIPSelfAttention::CLIPSelfAttention(RefPtr<InferencingContext> ctx, int hiddenSize, int numHeads)
    : ctx(ctx), hiddenSize(hiddenSize), numHeads(numHeads)
{
    headDim = hiddenSize / numHeads; // 768/12 = 64

    // Q, K, V projections
    qProj = new LinearKernel(ctx, hiddenSize, hiddenSize);
    kProj = new LinearKernel(ctx, hiddenSize, hiddenSize);
    vProj = new LinearKernel(ctx, hiddenSize, hiddenSize);
    outProj = new LinearKernel(ctx, hiddenSize, hiddenSize);

    // Flash attention with fused permutations
    // Input from linear is [B*seqLen, hiddenSize], reshape to [B, seqLen, numHeads, headDim]
    // FlashAttention expects [B, numHeads, seqLen, headDim]
    // Use permute in expression to convert
    qExpr = buffer();
    kExpr = buffer();
    vExpr = buffer();

    auto qPlanar = permute(qExpr, {0, 2, 1, 3}); // [B, S, H, D] -> [B, H, S, D]
    auto kPlanar = permute(kExpr, {0, 2, 1, 3});
    auto vPlanar = permute(vExpr, {0, 2, 1, 3});

    // Output needs to go back to [B, S, H, D], fuse into sink
    auto outSink = permute(bufferSink(), {0, 2, 1, 3});

    flashAttn = new FlashAttentionKernel(
        ctx,
        qPlanar,
        kPlanar,
        vPlanar,
        kernelOutput(),
        32,
        32,
        headDim,
        outSink);
}

SlangResult CLIPSelfAttention::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // CLIP uses q_proj, k_proj, v_proj, out_proj naming
    SLANG_RETURN_ON_FAIL(qProj->loadParams(
        reader,
        (prefix + "q_proj.weight").getUnownedSlice(),
        (prefix + "q_proj.bias").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(kProj->loadParams(
        reader,
        (prefix + "k_proj.weight").getUnownedSlice(),
        (prefix + "k_proj.bias").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(vProj->loadParams(
        reader,
        (prefix + "v_proj.weight").getUnownedSlice(),
        (prefix + "v_proj.bias").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(outProj->loadParams(
        reader,
        (prefix + "out_proj.weight").getUnownedSlice(),
        (prefix + "out_proj.bias").getUnownedSlice()));

    return SLANG_OK;
}

void CLIPSelfAttention::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    int seqLen,
    int batchSize)
{
    int totalTokens = batchSize * seqLen;

    TensorView inputFlat = input;
    inputFlat.shape = Shape(totalTokens, hiddenSize);

    // Q, K, V projections
    auto qOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    auto kOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    auto vOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));

    qProj->queueExecute(task, qOut, inputFlat);
    kProj->queueExecute(task, kOut, inputFlat);
    vProj->queueExecute(task, vOut, inputFlat);

    // Reshape for attention: [B*S, H*D] -> [B, S, H, D]
    TensorView qReshaped = qOut;
    TensorView kReshaped = kOut;
    TensorView vReshaped = vOut;
    qReshaped.shape = Shape(batchSize, seqLen, numHeads, headDim);
    kReshaped.shape = Shape(batchSize, seqLen, numHeads, headDim);
    vReshaped.shape = Shape(batchSize, seqLen, numHeads, headDim);

    // Flash attention with causal mask
    auto attnOut =
        ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, seqLen, numHeads, headDim));

    Dictionary<Expr, InputInfo> attnInputs;
    attnInputs.add(qExpr, qReshaped);
    attnInputs.add(kExpr, kReshaped);
    attnInputs.add(vExpr, vReshaped);

    float scale = 1.0f / sqrtf((float)headDim);
    flashAttn->queueExecute(
        task,
        attnOut,
        attnInputs,
        seqLen,
        seqLen,
        numHeads,
        batchSize,
        scale,
        true); // is_causal=true

    // Output projection
    TensorView attnOutFlat = attnOut;
    attnOutFlat.shape = Shape(totalTokens, hiddenSize);
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalTokens, hiddenSize);
    outProj->queueExecute(task, outputFlat, attnOutFlat);
}

// ============================================================================
// CLIPTransformerBlock
// ============================================================================

CLIPTransformerBlock::CLIPTransformerBlock(
    RefPtr<InferencingContext> ctx,
    int hiddenSize,
    int numHeads,
    int intermediateSize)
    : ctx(ctx), hiddenSize(hiddenSize)
{
    // LayerNorm before attention
    layerNorm1 = new LayerNormKernel(ctx, hiddenSize);

    // Self-attention
    selfAttn = new CLIPSelfAttention(ctx, hiddenSize, numHeads);

    // LayerNorm before MLP
    layerNorm2 = new LayerNormKernel(ctx, hiddenSize);

    // MLP
    mlp = new CLIPMLP(ctx, hiddenSize, intermediateSize);

    // Residual additions (each needs its own buffer expressions)
    {
        auto a = buffer();
        auto b = buffer();
        residualAdd1 = new ElementwiseKernel(ctx, a + b);
    }
    {
        auto a = buffer();
        auto b = buffer();
        residualAdd2 = new ElementwiseKernel(ctx, a + b);
    }
}

SlangResult CLIPTransformerBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // layer_norm1 -> self_attn -> layer_norm2 -> mlp
    SLANG_RETURN_ON_FAIL(layerNorm1->loadParams(
        reader,
        (prefix + "layer_norm1.weight").getUnownedSlice(),
        (prefix + "layer_norm1.bias").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(selfAttn->loadParams(reader, prefix + "self_attn."));

    SLANG_RETURN_ON_FAIL(layerNorm2->loadParams(
        reader,
        (prefix + "layer_norm2.weight").getUnownedSlice(),
        (prefix + "layer_norm2.bias").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(mlp->loadParams(reader, prefix + "mlp."));

    return SLANG_OK;
}

void CLIPTransformerBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    int seqLen,
    int batchSize)
{
    int totalTokens = batchSize * seqLen;

    // LayerNorm1
    TensorView inputFlat = input;
    inputFlat.shape = Shape(totalTokens, hiddenSize);

    auto normed1 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    layerNorm1->queueExecute(task, normed1, inputFlat);

    // Self-attention
    TensorView normed1_3d = normed1;
    normed1_3d.shape = Shape(batchSize, seqLen, hiddenSize);

    auto attnOut =
        ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, seqLen, hiddenSize));
    selfAttn->queueExecute(task, attnOut, normed1_3d, seqLen, batchSize);

    // Residual 1
    TensorView attnOutFlat = attnOut;
    attnOutFlat.shape = Shape(totalTokens, hiddenSize);
    auto residual1 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    residualAdd1->queueExecute(task, residual1, inputFlat, attnOutFlat);

    // LayerNorm2
    auto normed2 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    layerNorm2->queueExecute(task, normed2, residual1);

    // MLP
    auto mlpOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    mlp->queueExecute(task, mlpOut, normed2);

    // Residual 2
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalTokens, hiddenSize);
    residualAdd2->queueExecute(task, outputFlat, residual1, mlpOut);
}

// ============================================================================
// CLIPTextEncoder
// ============================================================================

CLIPTextEncoder::CLIPTextEncoder(
    RefPtr<InferencingContext> ctx,
    int vocabSize,
    int hiddenSize,
    int numHeads,
    int numLayers,
    int maxSeqLen,
    int intermediateSize)
    : ctx(ctx)
    , vocabSize(vocabSize)
    , hiddenSize(hiddenSize)
    , numHeads(numHeads)
    , numLayers(numLayers)
    , maxSeqLen(maxSeqLen)
    , intermediateSize(intermediateSize)
{
    // Embeddings (separate kernels for now)
    tokenEmbedding = new GatherKernel(ctx, vocabSize, hiddenSize);
    positionEmbedding = new GatherKernel(ctx, maxSeqLen, hiddenSize);

    // Embedding addition
    auto a = buffer();
    auto b = buffer();
    embeddingAdd = new ElementwiseKernel(ctx, a + b);

    // Transformer layers
    for (int i = 0; i < numLayers; i++)
    {
        layers.add(new CLIPTransformerBlock(ctx, hiddenSize, numHeads, intermediateSize));
    }

    // Final layer norm
    finalLayerNorm = new LayerNormKernel(ctx, hiddenSize);
    
    // Pre-create position IDs tensor (0, 1, 2, ..., maxSeqLen-1)
    // Stored as member to persist for async GPU execution
    List<float> posIds;
    posIds.setCount(maxSeqLen);
    for (int i = 0; i < maxSeqLen; i++)
        posIds[i] = (float)i;
    positionIdsTensor = ctx->createTensor(
        ElementType::Float32, Shape(maxSeqLen), posIds, "PositionIds");
}

SlangResult CLIPTextEncoder::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // Embeddings
    SLANG_RETURN_ON_FAIL(tokenEmbedding->loadParams(
        reader,
        (prefix + "embeddings.token_embedding.weight").getUnownedSlice()));

    SLANG_RETURN_ON_FAIL(positionEmbedding->loadParams(
        reader,
        (prefix + "embeddings.position_embedding.weight").getUnownedSlice()));

    // Transformer layers
    for (Index i = 0; i < layers.getCount(); i++)
    {
        String layerPrefix = prefix + "encoder.layers." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(layers[i]->loadParams(reader, layerPrefix));
    }

    // Final layer norm
    SLANG_RETURN_ON_FAIL(finalLayerNorm->loadParams(
        reader,
        (prefix + "final_layer_norm.weight").getUnownedSlice(),
        (prefix + "final_layer_norm.bias").getUnownedSlice()));

    return SLANG_OK;
}

TensorView CLIPTextEncoder::allocateResultBuffer(ElementType elementType, int seqLen, int batchSize)
{
    return ctx->allocScratchTensor(elementType, Shape(batchSize, seqLen, hiddenSize));
}

void CLIPTextEncoder::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView tokenIds,
    int seqLen,
    int batchSize)
{
    int totalTokens = batchSize * seqLen;

    // ========================================================================
    // PRE-ALLOCATE ALL BUFFERS before queuing any commands
    // (required for async execution - allocating during queuing can cause issues)
    // ========================================================================
    
    // Embedding buffers
    auto tokEmb = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    auto posEmb = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    auto hidden = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    
    // Layer output buffers
    List<TensorView> layerOuts;
    for (Index i = 0; i < layers.getCount(); i++)
    {
        layerOuts.add(ctx->allocScratchTensor(
            ElementType::Float32, Shape(batchSize, seqLen, hiddenSize)));
    }

    // ========================================================================
    // QUEUE ALL COMMANDS
    // ========================================================================
    
    // Token embeddings
    TensorView tokenIdsFlat = tokenIds;
    tokenIdsFlat.shape = Shape(totalTokens);
    tokenEmbedding->queueExecute(task, tokEmb, tokenIdsFlat);

    // Position embeddings
    // For batchSize=1: use pre-created position IDs directly
    // For batchSize>1: would need to tile position IDs (not currently supported)
    if (batchSize != 1)
    {
        throw std::runtime_error("CLIPTextEncoder: batchSize > 1 not yet supported");
    }
    TensorView posIdsView = positionIdsTensor->getView();
    posIdsView.shape = Shape(seqLen);
    positionEmbedding->queueExecute(task, posEmb, posIdsView);

    // Add token + position embeddings
    embeddingAdd->queueExecute(task, hidden, tokEmb, posEmb);

    // Transformer layers
    TensorView current = hidden;
    current.shape = Shape(batchSize, seqLen, hiddenSize);

    for (Index i = 0; i < layers.getCount(); i++)
    {
        layers[i]->queueExecute(task, layerOuts[i], current, seqLen, batchSize);
        current = layerOuts[i];
    }

    // Final layer norm
    TensorView currentFlat = current;
    currentFlat.shape = Shape(totalTokens, hiddenSize);
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalTokens, hiddenSize);
    finalLayerNorm->queueExecute(task, outputFlat, currentFlat);
}

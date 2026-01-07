#include "clip-encoder.h"

// ============================================================================
// CLIPMLP
// ============================================================================

CLIPMLP::CLIPMLP(RefPtr<InferencingContext> ctx, int hiddenSize, int intermediateSize)
    : ctx(ctx), hiddenSize(hiddenSize), intermediateSize(intermediateSize)
{
    // fc1: hidden_size -> intermediate_size with fused QuickGELU
    auto input1 = buffer();
    fc1 = new LinearKernel(ctx, ElementType::Float32, input1, quickGelu(kernelOutput()), 
                           bufferSink(), hiddenSize, intermediateSize);
    
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

void CLIPMLP::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input)
{
    // fc1 with fused QuickGELU
    auto fc1Out = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(input.shape[0], intermediateSize));
    fc1->queueExecute(task, fc1Out, input);
    
    // fc2
    fc2->queueExecute(task, output, fc1Out);
}

// ============================================================================
// CLIPSelfAttention
// ============================================================================

CLIPSelfAttention::CLIPSelfAttention(
    RefPtr<InferencingContext> ctx,
    int hiddenSize,
    int numHeads)
    : ctx(ctx), hiddenSize(hiddenSize), numHeads(numHeads)
{
    headDim = hiddenSize / numHeads;  // 768/12 = 64
    
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
    
    auto qPlanar = permute(qExpr, {0, 2, 1, 3});  // [B, S, H, D] -> [B, H, S, D]
    auto kPlanar = permute(kExpr, {0, 2, 1, 3});
    auto vPlanar = permute(vExpr, {0, 2, 1, 3});
    
    // Output needs to go back to [B, S, H, D], fuse into sink
    auto outSink = permute(bufferSink(), {0, 2, 1, 3});
    
    flashAttn = new FlashAttentionKernel(ctx, qPlanar, kPlanar, vPlanar, 
                                          kernelOutput(), 32, 32, headDim, outSink);
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
    
    // Input is [B, seqLen, hiddenSize], reshape for linear
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
    // Output will be [B, S, H, D] due to fused permute in sink
    auto attnOut = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, seqLen, numHeads, headDim));
    
    Dictionary<Expr, InputInfo> attnInputs;
    attnInputs.add(qExpr, qReshaped);
    attnInputs.add(kExpr, kReshaped);
    attnInputs.add(vExpr, vReshaped);
    
    float scale = 1.0f / sqrtf((float)headDim);
    flashAttn->queueExecute(task, attnOut, attnInputs, seqLen, seqLen, 
                            numHeads, batchSize, scale, true);  // is_causal=true
    
    // Reshape for output projection: [B, S, H, D] -> [B*S, H*D]
    TensorView attnOutFlat = attnOut;
    attnOutFlat.shape = Shape(totalTokens, hiddenSize);
    
    // Output projection
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
    // Input is [B, seqLen, hiddenSize], reshape to [B*seqLen, hiddenSize] for LayerNorm
    TensorView inputFlat = input;
    inputFlat.shape = Shape(totalTokens, hiddenSize);
    
    auto normed1 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    layerNorm1->queueExecute(task, normed1, inputFlat);
    
    // Reshape back for attention
    TensorView normed1_3d = normed1;
    normed1_3d.shape = Shape(batchSize, seqLen, hiddenSize);
    
    // Self-attention
    auto attnOut = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, seqLen, hiddenSize));
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
    // Fused embeddings: gather(tokenTable, tokenIds) + gather(posTable, posIds)
    // This fuses 3 kernels (tokenGather, posGather, add) into 1 kernel
    tokenTableExpr = buffer();  // Token embedding table [vocab, hidden]
    tokenIdsExpr = buffer();    // Token ID indices [totalTokens]
    posTableExpr = buffer();    // Position embedding table [maxSeq, hidden]
    posIdsExpr = buffer();      // Position ID indices [totalTokens]
    
    auto tokenEmb = gather(tokenTableExpr, tokenIdsExpr);
    auto posEmb = gather(posTableExpr, posIdsExpr);
    embeddingKernel = new ElementwiseKernel(ctx, tokenEmb + posEmb);
    
    // Transformer layers
    for (int i = 0; i < numLayers; i++)
    {
        layers.add(new CLIPTransformerBlock(ctx, hiddenSize, numHeads, intermediateSize));
    }
    
    // Final layer norm
    finalLayerNorm = new LayerNormKernel(ctx, hiddenSize);
}

SlangResult CLIPTextEncoder::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // Load token embedding table [vocab_size, hidden_size]
    {
        String weightName = prefix + "embeddings.token_embedding.weight";
        const SafeTensorInfo* info = reader.getTensorInfo(weightName.getUnownedSlice());
        if (!info || info->shape.getRank() != 2 ||
            info->shape[0] != vocabSize || info->shape[1] != hiddenSize)
        {
            return SLANG_E_INVALID_ARG;
        }
        List<uint8_t> weightsData;
        SLANG_RETURN_ON_FAIL(reader.readTensor(weightName.getUnownedSlice(), ElementType::Float32, weightsData));
        tokenTableBuffer = ctx->createPersistentBuffer(
            weightsData.getBuffer(), weightsData.getCount(), "TokenEmbedding");
    }
    
    // Load position embedding table [max_seq_len, hidden_size]
    {
        String weightName = prefix + "embeddings.position_embedding.weight";
        const SafeTensorInfo* info = reader.getTensorInfo(weightName.getUnownedSlice());
        if (!info || info->shape.getRank() != 2 ||
            info->shape[0] != maxSeqLen || info->shape[1] != hiddenSize)
        {
            return SLANG_E_INVALID_ARG;
        }
        List<uint8_t> weightsData;
        SLANG_RETURN_ON_FAIL(reader.readTensor(weightName.getUnownedSlice(), ElementType::Float32, weightsData));
        posTableBuffer = ctx->createPersistentBuffer(
            weightsData.getBuffer(), weightsData.getCount(), "PositionEmbedding");
    }
    
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
    
    // Flatten token IDs: [B, seqLen] -> [totalTokens]
    TensorView tokenIdsFlat = tokenIds;
    tokenIdsFlat.shape = Shape(totalTokens);
    
    // Position IDs: [0, 1, 2, ..., seqLen-1] repeated for each batch
    List<float> posIds;
    posIds.setCount(totalTokens);
    for (int b = 0; b < batchSize; b++)
    {
        for (int s = 0; s < seqLen; s++)
        {
            posIds[b * seqLen + s] = (float)s;
        }
    }
    auto posIdsTensor = ctx->createTensor(ElementType::Float32, Shape(totalTokens), posIds, "PositionIds");
    
    // Fused embeddings: gather(tokenTable, tokenIds) + gather(posTable, posIds)
    // Single kernel replaces 3 separate kernels
    auto hidden = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, hiddenSize));
    
    Dictionary<Expr, InputInfo> embInputs;
    embInputs.add(tokenTableExpr, TensorView(tokenTableBuffer, ElementType::Float32, Shape(vocabSize, hiddenSize)));
    embInputs.add(tokenIdsExpr, tokenIdsFlat);
    embInputs.add(posTableExpr, TensorView(posTableBuffer, ElementType::Float32, Shape(maxSeqLen, hiddenSize)));
    embInputs.add(posIdsExpr, posIdsTensor->getView());
    embeddingKernel->queueExecute(task, hidden, embInputs);
    
    // Reshape to 3D for transformer blocks
    TensorView current = hidden;
    current.shape = Shape(batchSize, seqLen, hiddenSize);
    
    // Transformer layers
    for (Index i = 0; i < layers.getCount(); i++)
    {
        auto layerOut = ctx->allocScratchTensor(ElementType::Float32, 
            Shape(batchSize, seqLen, hiddenSize));
        layers[i]->queueExecute(task, layerOut, current, seqLen, batchSize);
        current = layerOut;
    }
    
    // Final layer norm
    TensorView currentFlat = current;
    currentFlat.shape = Shape(totalTokens, hiddenSize);
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalTokens, hiddenSize);
    finalLayerNorm->queueExecute(task, outputFlat, currentFlat);
}


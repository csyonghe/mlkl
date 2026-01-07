#include "vae-decoder.h"

// ============================================================================
// VAEResNetBlock
// ============================================================================

VAEResNetBlock::VAEResNetBlock(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int outChannels,
    int numGroups)
    : ctx(ctx), inChannels(inChannels), outChannels(outChannels)
{
    // GroupNorm -> SiLU -> Conv pattern (twice)
    // Fuse SiLU into Conv2D input expression for efficiency
    // Conv2DKernel(ctx, tileSize, kernelSize, stride, inChannels, outChannels, inputExpr, outputExpr, sinkExpr)
    // GroupNormKernel(ctx, numChannels, numGroups)
    norm1 = new GroupNormKernel(ctx, inChannels, numGroups);
    auto input1 = buffer();
    conv1 = new Conv2DKernel(ctx, ElementType::Float32, 8, 3, 1, inChannels, outChannels,
                              silu(input1), kernelOutput(), bufferSink());
    
    norm2 = new GroupNormKernel(ctx, outChannels, numGroups);
    auto input2 = buffer();
    conv2 = new Conv2DKernel(ctx, ElementType::Float32, 8, 3, 1, outChannels, outChannels,
                              silu(input2), kernelOutput(), bufferSink());
    
    // Skip connection if channel dimensions differ
    if (inChannels != outChannels)
    {
        skipConv = new Conv2DKernel(ctx, 8, 1, 1, inChannels, outChannels);
    }
    
    // Residual addition
    auto a = buffer();
    auto b = buffer();
    residualAdd = new ElementwiseKernel(ctx, a + b);
}

SlangResult VAEResNetBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // Load norm1, conv1, norm2, conv2
    SLANG_RETURN_ON_FAIL(norm1->loadParams(
        reader,
        (prefix + "norm1.weight").getUnownedSlice(),
        (prefix + "norm1.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(conv1->loadParams(
        reader,
        (prefix + "conv1.weight").getUnownedSlice(),
        (prefix + "conv1.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(norm2->loadParams(
        reader,
        (prefix + "norm2.weight").getUnownedSlice(),
        (prefix + "norm2.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(conv2->loadParams(
        reader,
        (prefix + "conv2.weight").getUnownedSlice(),
        (prefix + "conv2.bias").getUnownedSlice()));
    
    // Skip connection
    if (skipConv)
    {
        SLANG_RETURN_ON_FAIL(skipConv->loadParams(
            reader,
            (prefix + "conv_shortcut.weight").getUnownedSlice(),
            (prefix + "conv_shortcut.bias").getUnownedSlice()));
    }
    
    return SLANG_OK;
}

TensorView VAEResNetBlock::allocateResultBuffer(
    ElementType elementType,
    int height,
    int width,
    int batchSize)
{
    return ctx->allocScratchTensor(
        elementType,
        Shape(batchSize, height, width, outChannels));
}

void VAEResNetBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input)
{
    auto shape = input.shape;
    int B = shape[0], H = shape[1], W = shape[2];
    
    // Allocate intermediates (SiLU is fused into conv input expression)
    auto afterNorm1 = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, inChannels));
    auto afterConv1 = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, outChannels));
    auto afterNorm2 = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, outChannels));
    auto afterConv2 = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, outChannels));
    
    // Forward pass: GroupNorm -> Conv(SiLU fused) -> GroupNorm -> Conv(SiLU fused)
    norm1->queueExecute(task, afterNorm1, input);
    conv1->queueExecute(task, afterConv1, afterNorm1, 1);  // padding=1, SiLU fused in input
    
    norm2->queueExecute(task, afterNorm2, afterConv1);
    conv2->queueExecute(task, afterConv2, afterNorm2, 1);  // padding=1, SiLU fused in input
    
    // Residual connection
    if (skipConv)
    {
        auto skipOut = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, outChannels));
        skipConv->queueExecute(task, skipOut, input, 0);  // padding=0 for 1x1 conv
        residualAdd->queueExecute(task, output, afterConv2, skipOut);
    }
    else
    {
        residualAdd->queueExecute(task, output, afterConv2, input);
    }
}

// ============================================================================
// VAEAttentionBlock
// ============================================================================

VAEAttentionBlock::VAEAttentionBlock(
    RefPtr<InferencingContext> ctx,
    int channels,
    int numHeads)
    : ctx(ctx), channels(channels)
{
    // SD VAE uses single-head attention with 512 channels.
    // FlashAttention can't handle headDim=512 (exceeds shared memory limits).
    // Use standard attention instead: Q @ K^T -> softmax -> @ V
    (void)numHeads;  // Not used - VAE attention is always single-head
    
    // GroupNormKernel(ctx, numChannels, numGroups)
    groupNorm = new GroupNormKernel(ctx, channels, 32);
    
    // Q, K, V projections (channels -> channels)
    // LinearKernel(ctx, inputDim, outputDim)
    projQ = new LinearKernel(ctx, channels, channels);
    projK = new LinearKernel(ctx, channels, channels);
    projV = new LinearKernel(ctx, channels, channels);
    projOut = new LinearKernel(ctx, channels, channels);
    
    // Standard attention using BatchGemm + Softmax
    // Q @ K^T: [B, seqLen, C] @ [B, C, seqLen] -> [B, seqLen, seqLen]
    // Fuse transpose into K expression
    qExpr = buffer();
    kExpr = buffer();  // Will be transposed via expression
    auto kTransposed = transpose(kExpr, 1, 2);  // [B, seqLen, C] -> [B, C, seqLen]
    qkGemm = new BatchGemmKernel(ctx, qExpr, kTransposed, constant(0.0f), bufferSink(), kernelOutput());
    
    // Softmax over attention scores
    softmax = new SoftmaxKernel(ctx);
    
    // attn_weights @ V: [B, seqLen, seqLen] @ [B, seqLen, C] -> [B, seqLen, C]
    attnExpr = buffer();
    vExpr = buffer();
    attnVGemm = new BatchGemmKernel(ctx, attnExpr, vExpr, constant(0.0f), bufferSink(), kernelOutput());
    
    // Residual addition
    auto a = buffer();
    auto b = buffer();
    residualAdd = new ElementwiseKernel(ctx, a + b);
}

SlangResult VAEAttentionBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    SLANG_RETURN_ON_FAIL(groupNorm->loadParams(
        reader,
        (prefix + "group_norm.weight").getUnownedSlice(),
        (prefix + "group_norm.bias").getUnownedSlice()));
    
    // Load Q, K, V projections
    // SD VAE uses: query, key, value, proj_attn (not to_q, to_k, to_v, to_out)
    SLANG_RETURN_ON_FAIL(projQ->loadParams(
        reader,
        (prefix + "query.weight").getUnownedSlice(),
        (prefix + "query.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(projK->loadParams(
        reader,
        (prefix + "key.weight").getUnownedSlice(),
        (prefix + "key.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(projV->loadParams(
        reader,
        (prefix + "value.weight").getUnownedSlice(),
        (prefix + "value.bias").getUnownedSlice()));
    
    // Output projection
    SLANG_RETURN_ON_FAIL(projOut->loadParams(
        reader,
        (prefix + "proj_attn.weight").getUnownedSlice(),
        (prefix + "proj_attn.bias").getUnownedSlice()));
    
    return SLANG_OK;
}

TensorView VAEAttentionBlock::allocateResultBuffer(
    ElementType elementType,
    int height,
    int width,
    int batchSize)
{
    return ctx->allocScratchTensor(
        elementType,
        Shape(batchSize, height, width, channels));
}

void VAEAttentionBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input)
{
    auto shape = input.shape;
    int B = shape[0], H = shape[1], W = shape[2], C = shape[3];
    int seqLen = H * W;
    int totalTokens = B * seqLen;
    
    // GroupNorm
    auto normed = ctx->allocScratchTensor(ElementType::Float32, shape);
    groupNorm->queueExecute(task, normed, input);
    
    // Reshape to [B*H*W, C] for linear projections (LinearKernel expects rank 2)
    TensorView normedFlat = normed;
    normedFlat.shape = Shape(totalTokens, C);
    
    // Q, K, V projections: [B*seqLen, C] -> [B*seqLen, C]
    auto qOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, C));
    auto kOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, C));
    auto vOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, C));
    
    projQ->queueExecute(task, qOut, normedFlat);
    projK->queueExecute(task, kOut, normedFlat);
    projV->queueExecute(task, vOut, normedFlat);
    
    // Standard attention: Q @ K^T -> softmax -> @ V
    // Reshape for BatchGemm: [B*seqLen, C] -> [B, seqLen, C]
    TensorView qBatch = qOut;
    TensorView kBatch = kOut;
    TensorView vBatch = vOut;
    qBatch.shape = Shape(B, seqLen, C);
    kBatch.shape = Shape(B, seqLen, C);
    vBatch.shape = Shape(B, seqLen, C);
    
    // Q @ K^T: [B, seqLen, C] @ [B, C, seqLen] -> [B, seqLen, seqLen]
    // K transpose is fused into qkGemm via expression
    auto scores = ctx->allocScratchTensor(ElementType::Float32, Shape(B, seqLen, seqLen));
    float scale = 1.0f / sqrtf((float)C);
    qkGemm->queueExecute(task, scores, scale, 0.0f, {qBatch, kBatch});
    
    // Softmax: [B, seqLen, seqLen] -> normalize each row
    TensorView scoresFlat = scores;
    scoresFlat.shape = Shape(B * seqLen, seqLen);
    auto attnWeights = ctx->allocScratchTensor(ElementType::Float32, Shape(B * seqLen, seqLen));
    softmax->queueExecute(task, attnWeights, scoresFlat);
    
    // Reshape back to [B, seqLen, seqLen]
    TensorView attnWeightsBatch = attnWeights;
    attnWeightsBatch.shape = Shape(B, seqLen, seqLen);
    
    // attn_weights @ V: [B, seqLen, seqLen] @ [B, seqLen, C] -> [B, seqLen, C]
    auto attnOut = ctx->allocScratchTensor(ElementType::Float32, Shape(B, seqLen, C));
    attnVGemm->queueExecute(task, attnOut, 1.0f, 0.0f, {attnWeightsBatch, vBatch});
    
    // Reshape to [B*seqLen, C] for linear projection
    TensorView attnOutFlat = attnOut;
    attnOutFlat.shape = Shape(totalTokens, C);
    
    // Output projection (LinearKernel expects rank 2)
    auto projOutResult = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, C));
    projOut->queueExecute(task, projOutResult, attnOutFlat);
    
    // Reshape to original: [B*seqLen, C] -> [B, H, W, C]
    TensorView projOutReshaped = projOutResult;
    projOutReshaped.shape = Shape(B, H, W, C);
    
    // Residual
    residualAdd->queueExecute(task, output, projOutReshaped, input);
}

// ============================================================================
// VAEUpBlock
// ============================================================================

VAEUpBlock::VAEUpBlock(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int outChannels,
    bool hasUpsample,
    int numResBlocks)
    : ctx(ctx), hasUpsample(hasUpsample), outChannels(outChannels)
{
    // Create ResNet blocks
    // First block: inChannels -> outChannels
    // Remaining blocks: outChannels -> outChannels
    for (int i = 0; i < numResBlocks; i++)
    {
        int blockIn = (i == 0) ? inChannels : outChannels;
        resnets.add(new VAEResNetBlock(ctx, blockIn, outChannels));
    }
    
    // Upsampling: nearest-neighbor 2x fused into conv input
    if (hasUpsample)
    {
        auto x = buffer();
        upsampleConv = new Conv2DKernel(ctx, ElementType::Float32, 8, 3, 1, outChannels, outChannels,
                                         upsample2x(x), kernelOutput(), bufferSink());
    }
}

SlangResult VAEUpBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // Load ResNet blocks
    for (Index i = 0; i < resnets.getCount(); i++)
    {
        String resPrefix = prefix + "resnets." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(resnets[i]->loadParams(reader, resPrefix));
    }
    
    // Load upsampler
    if (hasUpsample)
    {
        SLANG_RETURN_ON_FAIL(upsampleConv->loadParams(
            reader,
            (prefix + "upsamplers.0.conv.weight").getUnownedSlice(),
            (prefix + "upsamplers.0.conv.bias").getUnownedSlice()));
    }
    
    return SLANG_OK;
}

TensorView VAEUpBlock::allocateResultBuffer(
    ElementType elementType,
    int height,
    int width,
    int batchSize)
{
    int outH = hasUpsample ? height * 2 : height;
    int outW = hasUpsample ? width * 2 : width;
    return ctx->allocScratchTensor(
        elementType,
        Shape(batchSize, outH, outW, outChannels));
}

void VAEUpBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input)
{
    auto shape = input.shape;
    int B = shape[0], H = shape[1], W = shape[2];
    
    TensorView current = input;
    
    // Process ResNet blocks
    for (Index i = 0; i < resnets.getCount(); i++)
    {
        auto resOut = resnets[i]->allocateResultBuffer(ElementType::Float32, H, W, B);
        resnets[i]->queueExecute(task, resOut, current);
        current = resOut;
    }
    
    // Upsample if needed (upsample2x is fused into conv input expression)
    if (hasUpsample)
    {
        // Conv with fused 2x nearest-neighbor upsample in input
        upsampleConv->queueExecute(task, output, current, 1);  // padding=1
    }
    else
    {
        // No upsample - copy to output
        // TODO: Could optimize by having last resnet write directly to output
        auto copyExpr = buffer();
        ElementwiseKernel copyKernel(ctx, copyExpr);
        copyKernel.queueExecute(task, output, current);
    }
}

// ============================================================================
// VAEDecoder
// ============================================================================

VAEDecoder::VAEDecoder(RefPtr<InferencingContext> ctx)
    : ctx(ctx)
{
    // Post-quant conv: 4 -> 4 (1x1 conv)
    postQuantConv = new Conv2DKernel(ctx, 8, 1, 1, 4, 4);
    
    // Decoder input: 4 -> 512 (3x3 conv)
    convIn = new Conv2DKernel(ctx, 8, 3, 1, 4, 512);
    
    // Mid block
    midResnet1 = new VAEResNetBlock(ctx, 512, 512);
    midAttn = new VAEAttentionBlock(ctx, 512, 1);
    midResnet2 = new VAEResNetBlock(ctx, 512, 512);
    
    // Up blocks (reverse order in architecture, but we process 0 first)
    // Block 0: 512 -> 512, upsample
    // Block 1: 512 -> 512, upsample
    // Block 2: 512 -> 256, upsample
    // Block 3: 256 -> 128, no upsample
    upBlocks.add(new VAEUpBlock(ctx, 512, 512, true));   // 64 -> 128
    upBlocks.add(new VAEUpBlock(ctx, 512, 512, true));   // 128 -> 256
    upBlocks.add(new VAEUpBlock(ctx, 512, 256, true));   // 256 -> 512
    upBlocks.add(new VAEUpBlock(ctx, 256, 128, false));  // 512 (no change)
    
    // Output - GroupNormKernel(ctx, numChannels, numGroups)
    normOut = new GroupNormKernel(ctx, 128, 32);
    // 3x3 conv, 128 -> 3 channels
    // Fuse SiLU into input and clamp to [-1, 1] into output
    auto x = buffer();
    convOut = new Conv2DKernel(ctx, ElementType::Float32, 8, 3, 1, 128, 3,
                                silu(x), clamp(kernelOutput(), -1.0f, 1.0f), bufferSink());
}

SlangResult VAEDecoder::loadParams(SafeTensorsReader& reader)
{
    // Post-quant conv
    SLANG_RETURN_ON_FAIL(postQuantConv->loadParams(
        reader,
        toSlice("post_quant_conv.weight"),
        toSlice("post_quant_conv.bias")));
    
    // Decoder conv_in
    SLANG_RETURN_ON_FAIL(convIn->loadParams(
        reader,
        toSlice("decoder.conv_in.weight"),
        toSlice("decoder.conv_in.bias")));
    
    // Mid block
    SLANG_RETURN_ON_FAIL(midResnet1->loadParams(reader, "decoder.mid_block.resnets.0."));
    SLANG_RETURN_ON_FAIL(midAttn->loadParams(reader, "decoder.mid_block.attentions.0."));
    SLANG_RETURN_ON_FAIL(midResnet2->loadParams(reader, "decoder.mid_block.resnets.1."));
    
    // Up blocks
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        String prefix = "decoder.up_blocks." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(upBlocks[i]->loadParams(reader, prefix));
    }
    
    // Output
    SLANG_RETURN_ON_FAIL(normOut->loadParams(
        reader,
        toSlice("decoder.conv_norm_out.weight"),
        toSlice("decoder.conv_norm_out.bias")));
    
    SLANG_RETURN_ON_FAIL(convOut->loadParams(
        reader,
        toSlice("decoder.conv_out.weight"),
        toSlice("decoder.conv_out.bias")));
    
    return SLANG_OK;
}

TensorView VAEDecoder::allocateResultBuffer(
    ElementType elementType,
    int latentHeight,
    int latentWidth,
    int batchSize)
{
    // Output is 8x the latent resolution
    return ctx->allocScratchTensor(
        elementType,
        Shape(batchSize, latentHeight * 8, latentWidth * 8, outputChannels));
}

void VAEDecoder::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView latentInput)
{
    auto shape = latentInput.shape;
    int B = shape[0], H = shape[1], W = shape[2];
    
    // Post-quant conv (1x1, no padding)
    auto postQuant = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, 4));
    postQuantConv->queueExecute(task, postQuant, latentInput, 0);
    
    // Conv in (3x3, padding=1)
    auto afterConvIn = ctx->allocScratchTensor(ElementType::Float32, Shape(B, H, W, 512));
    convIn->queueExecute(task, afterConvIn, postQuant, 1);
    
    // Mid block
    auto midOut1 = midResnet1->allocateResultBuffer(ElementType::Float32, H, W, B);
    midResnet1->queueExecute(task, midOut1, afterConvIn);
    
    auto midAttnOut = midAttn->allocateResultBuffer(ElementType::Float32, H, W, B);
    midAttn->queueExecute(task, midAttnOut, midOut1);
    
    auto midOut2 = midResnet2->allocateResultBuffer(ElementType::Float32, H, W, B);
    midResnet2->queueExecute(task, midOut2, midAttnOut);
    
    // Up blocks
    TensorView current = midOut2;
    int currentH = H, currentW = W;
    
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        auto upOut = upBlocks[i]->allocateResultBuffer(ElementType::Float32, currentH, currentW, B);
        upBlocks[i]->queueExecute(task, upOut, current);
        current = upOut;
        
        if (upBlocks[i]->hasUpsample)
        {
            currentH *= 2;
            currentW *= 2;
        }
    }
    
    // Output: GroupNorm -> Conv (SiLU fused in input, clamp fused in output)
    auto normed = ctx->allocScratchTensor(ElementType::Float32, Shape(B, currentH, currentW, 128));
    normOut->queueExecute(task, normed, current);
    
    convOut->queueExecute(task, output, normed, 1);  // 3x3, padding=1
}


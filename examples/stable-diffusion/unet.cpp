#include "unet.h"
#include <cmath>

// ============================================================================
// Utility: Sinusoidal Time Embedding
// ============================================================================

void getSinusoidalEmbedding(float* output, int timestep, int dim)
{
    // Matches diffusers get_timestep_embedding for SD 1.5:
    // - flip_sin_to_cos=True, downscale_freq_shift=0
    // exponent = -log(10000) * arange(half_dim) / half_dim
    // emb = timestep * exp(exponent)
    // output = [cos(emb), sin(emb)]
    int halfDim = dim / 2;
    float logBase = logf(10000.0f) / (float)halfDim;  // SD 1.5 uses half_dim, not half_dim-1
    
    for (int i = 0; i < halfDim; i++)
    {
        float freq = expf(-(float)i * logBase);
        float angle = (float)timestep * freq;
        output[i] = cosf(angle);            // cos first (flip_sin_to_cos=True)
        output[i + halfDim] = sinf(angle);  // sin second
    }
}

// ============================================================================
// SDResNetBlock
// ============================================================================

SDResNetBlock::SDResNetBlock(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int outChannels,
    int timeEmbedDim)
    : SDResNetBlock(ctx, inChannels, outChannels, timeEmbedDim, false)
{
}

SDResNetBlock::SDResNetBlock(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int outChannels,
    int timeEmbedDim,
    bool fuseConcatInput)
    : ctx(ctx)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , timeEmbedDim(timeEmbedDim)
    , hasResidualConv(inChannels != outChannels)
    , hasFusedConcat(fuseConcatInput)
{
    // norm1: optionally with fused concat for up blocks
    if (fuseConcatInput)
    {
        // norm1 with fused concat: norm(concat(current, skip, axis=3))
        // Store expressions as members to map inputs at runtime
        norm1Buf0 = buffer();  // current [B, H, W, C1]
        norm1Buf1 = buffer();  // skip [B, H, W, C2]
        norm1ConcatAxis = uniformConstant();  // axis provided at runtime
        auto concatExpr = concat(norm1Buf0, norm1Buf1, norm1ConcatAxis);
        norm1 = new GroupNormKernel(ctx, ElementType::Float32, concatExpr, bufferSink(), inChannels, 32);
    }
    else
    {
        norm1 = new GroupNormKernel(ctx, inChannels, 32);
    }
    
    // GroupNorm → SiLU → Conv3x3
    conv1 = new Conv2DKernel(ctx, ElementType::Float32, 16, 3, 1, inChannels, outChannels,
                              silu(buffer()), kernelOutput(), bufferSink());
    
    // Time embedding projection: SiLU → Linear (diffusers applies silu to time emb first)
    timeProj = new LinearKernel(ctx, ElementType::Float32, silu(buffer()), kernelOutput(),
                                 bufferSink(), timeEmbedDim, outChannels);
    
    // GroupNorm with fused time embedding add: norm(conv1_out + broadcast(time_proj))
    // Buffer order: buf0 = conv1 output, buf1 = time projection (broadcast)
    auto timeBuf0 = buffer();  // conv1 output [B, H, W, C]
    auto timeBuf1 = buffer();  // time projection [B, 1, 1, C] (will be broadcast)
    auto timeAddExpr = timeBuf0 + broadcast(timeBuf1, timeBuf0);
    norm2 = new GroupNormKernel(ctx, ElementType::Float32, timeAddExpr, bufferSink(), outChannels, 32);
    
    // SiLU → Conv3x3 with fused residual addition
    conv2 = new Conv2DKernel(ctx, ElementType::Float32, 16, 3, 1, outChannels, outChannels,
                              silu(buffer()), kernelOutput() + buffer(), bufferSink());
    
    // Residual projection if channel mismatch
    if (hasResidualConv)
    {
        if (fuseConcatInput)
        {
            // residualConv with fused concat: conv(concat(current, skip))
            // Store expressions as members to map inputs at runtime
            resBuf0 = buffer();  // current
            resBuf1 = buffer();  // skip
            residualConvConcatAxis = uniformConstant();  // axis provided at runtime
            auto resConcatExpr = concat(resBuf0, resBuf1, residualConvConcatAxis);
            residualConv = new Conv2DKernel(ctx, ElementType::Float32, 16, 1, 1, inChannels, outChannels,
                                            resConcatExpr, kernelOutput(), bufferSink());
        }
        else
        {
            residualConv = new Conv2DKernel(ctx, 16, 1, 1, inChannels, outChannels);
        }
    }
}

SlangResult SDResNetBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    SLANG_RETURN_ON_FAIL(norm1->loadParams(
        reader,
        (prefix + "norm1.weight").getUnownedSlice(),
        (prefix + "norm1.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(conv1->loadParams(
        reader,
        (prefix + "conv1.weight").getUnownedSlice(),
        (prefix + "conv1.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(timeProj->loadParams(
        reader,
        (prefix + "time_emb_proj.weight").getUnownedSlice(),
        (prefix + "time_emb_proj.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(norm2->loadParams(
        reader,
        (prefix + "norm2.weight").getUnownedSlice(),
        (prefix + "norm2.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(conv2->loadParams(
        reader,
        (prefix + "conv2.weight").getUnownedSlice(),
        (prefix + "conv2.bias").getUnownedSlice()));
    
    if (hasResidualConv)
    {
        SLANG_RETURN_ON_FAIL(residualConv->loadParams(
            reader,
            (prefix + "conv_shortcut.weight").getUnownedSlice(),
            (prefix + "conv_shortcut.bias").getUnownedSlice()));
    }
    
    return SLANG_OK;
}

void SDResNetBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    TensorView timeEmbed)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    
    // norm1 → silu → conv1
    auto normed1 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, inChannels));
    norm1->queueExecute(task, normed1, input);
    
    auto afterConv1 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, outChannels));
    conv1->queueExecute(task, afterConv1, normed1, 1);
    
    // Time embedding projection: [B, timeEmbedDim] → [B, outChannels]
    auto timeProj_out = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, outChannels));
    timeProj->queueExecute(task, timeProj_out, timeEmbed);
    
    // Reshape for broadcast: [B, outChannels] → [B, 1, 1, outChannels]
    TensorView timeProjReshaped = timeProj_out;
    timeProjReshaped.shape = Shape(batchSize, 1, 1, outChannels);
    
    // norm2 with fused time embedding add: norm(conv1 + broadcast(time))
    // Buffer order matches constructor: {buf0=afterConv1, buf1=timeProjReshaped}
    auto normed2 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, outChannels));
    norm2->queueExecute(task, normed2, {afterConv1, timeProjReshaped});
    
    if (hasResidualConv)
    {
        // Residual needs channel projection first
        auto residual = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, outChannels));
        residualConv->queueExecute(task, residual, input, 0);
        conv2->queueExecute(task, output, {normed2, residual}, 1);
    }
    else
    {
        // Input is the residual directly
        conv2->queueExecute(task, output, {normed2, input}, 1);
    }
}

void SDResNetBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView current,
    TensorView skip,
    TensorView timeEmbed)
{
    if (!hasFusedConcat)
    {
        throw std::runtime_error("SDResNetBlock: Two-input queueExecute requires fuseConcatInput=true");
    }
    
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = current.shape[0];
    int height = current.shape[1];
    int width = current.shape[2];
    
    // norm1 with fused concat: norm(concat(current, skip))
    // Build EvalContext to map expressions to inputs including the axis
    auto normed1 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, inChannels));
    {
        EvalContext norm1Ctx;
        norm1Ctx.inputs.add(norm1Buf0.node, InputInfo{current});
        norm1Ctx.inputs.add(norm1Buf1.node, InputInfo{skip});
        norm1Ctx.inputs.add(norm1ConcatAxis.node, InputInfo{3.0f});
        norm1->queueExecute(task, normed1, norm1Ctx);
    }
    
    auto afterConv1 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, outChannels));
    conv1->queueExecute(task, afterConv1, normed1, 1);
    
    // Time embedding projection: [B, timeEmbedDim] → [B, outChannels]
    auto timeProj_out = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, outChannels));
    timeProj->queueExecute(task, timeProj_out, timeEmbed);
    
    // Reshape for broadcast: [B, outChannels] → [B, 1, 1, outChannels]
    TensorView timeProjReshaped = timeProj_out;
    timeProjReshaped.shape = Shape(batchSize, 1, 1, outChannels);
    
    // norm2 with fused time embedding add
    auto normed2 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, outChannels));
    norm2->queueExecute(task, normed2, {afterConv1, timeProjReshaped});
    
    // For fused concat case, residualConv also has fused concat in its inputExpr
    if (hasResidualConv)
    {
        // residualConv with fused concat: build EvalContext with axis
        auto residual = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, outChannels));
        {
            EvalContext resConvCtx;
            resConvCtx.inputs.add(resBuf0.node, InputInfo{current});
            resConvCtx.inputs.add(resBuf1.node, InputInfo{skip});
            resConvCtx.inputs.add(residualConvConcatAxis.node, InputInfo{3.0f});
            residualConv->queueExecute(task, resConvCtx, residual, 0);
        }
        conv2->queueExecute(task, output, {normed2, residual}, 1);
    }
    else
    {
        // No residual conv means inChannels == outChannels
        // But with concat, inChannels = currentCh + skipCh, outChannels = some fixed value
        // This case shouldn't happen for up blocks since they always have channel changes
        throw std::runtime_error("SDResNetBlock: Fused concat without residual conv not supported");
    }
}

// ============================================================================
// SDSelfAttention
// ============================================================================

SDSelfAttention::SDSelfAttention(
    RefPtr<InferencingContext> ctx,
    int dim,
    int numHeads)
    : ctx(ctx)
    , dim(dim)
    , numHeads(numHeads)
    , headDim(dim / numHeads)
{
    toQ = new LinearKernel(ctx, dim, dim);
    toK = new LinearKernel(ctx, dim, dim);
    toV = new LinearKernel(ctx, dim, dim);
    // toOut with fused residual addition: output = linear(attn) + residual
    toOut = new LinearKernel(ctx, ElementType::Float32,
                             buffer(), kernelOutput() + buffer(), bufferSink(),
                             dim, dim);
    
    // Use FlashAttentionKernel (handles head transpose internally)
    // Input format: [B, S, H, headDim] - needs permute to [B, H, S, headDim]
    qExpr = buffer();
    kExpr = buffer();
    vExpr = buffer();
    
    auto qPlanar = permute(qExpr, {0, 2, 1, 3});  // [B,S,H,D] -> [B,H,S,D]
    auto kPlanar = permute(kExpr, {0, 2, 1, 3});
    auto vPlanar = permute(vExpr, {0, 2, 1, 3});
    // Output comes as [B, H, S, D], permute back to [B, S, H, D]
    auto outSink = permute(bufferSink(), {0, 2, 1, 3});
    
    flashAttn = new FlashAttentionKernel(
        ctx, qPlanar, kPlanar, vPlanar, kernelOutput(),
        32, 32, headDim, outSink);
}

SlangResult SDSelfAttention::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // SD 1.5 UNet attention has NO biases for Q/K/V projections
    SLANG_RETURN_ON_FAIL(toQ->loadParams(
        reader,
        (prefix + "to_q.weight").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(toK->loadParams(
        reader,
        (prefix + "to_k.weight").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(toV->loadParams(
        reader,
        (prefix + "to_v.weight").getUnownedSlice()));
    
    // to_out does have bias
    SLANG_RETURN_ON_FAIL(toOut->loadParams(
        reader,
        (prefix + "to_out.0.weight").getUnownedSlice(),
        (prefix + "to_out.0.bias").getUnownedSlice()));
    
    return SLANG_OK;
}

void SDSelfAttention::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    TensorView residual)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = input.shape[0];
    int seqLen = input.shape[1];
    int totalTokens = batchSize * seqLen;
    
    // Flatten for linear layers: [B, S, D] → [B*S, D]
    TensorView inputFlat = input;
    inputFlat.shape = Shape(totalTokens, dim);
    
    // Q, K, V projections
    auto qOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, dim));
    auto kOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, dim));
    auto vOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, dim));
    
    toQ->queueExecute(task, qOut, inputFlat);
    toK->queueExecute(task, kOut, inputFlat);
    toV->queueExecute(task, vOut, inputFlat);
    
    // Reshape for FlashAttention: [B*S, D] → [B, S, H, headDim]
    TensorView qReshaped = qOut;
    qReshaped.shape = Shape(batchSize, seqLen, numHeads, headDim);
    TensorView kReshaped = kOut;
    kReshaped.shape = Shape(batchSize, seqLen, numHeads, headDim);
    TensorView vReshaped = vOut;
    vReshaped.shape = Shape(batchSize, seqLen, numHeads, headDim);
    
    // Flash attention (handles transpose and scaling internally)
    auto attnOut = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, seqLen, numHeads, headDim));
    
    Dictionary<Expr, InputInfo> attnInputs;
    attnInputs.add(qExpr, qReshaped);
    attnInputs.add(kExpr, kReshaped);
    attnInputs.add(vExpr, vReshaped);
    
    float scale = 1.0f / sqrtf((float)headDim);
    flashAttn->queueExecute(task, attnOut, attnInputs,
        seqLen, seqLen, numHeads, batchSize, scale, false);
    
    // Reshape back: [B, S, H, headDim] → [B*S, D]
    TensorView attnOutFlat = attnOut;
    attnOutFlat.shape = Shape(totalTokens, dim);
    
    // Flatten residual for toOut
    TensorView residualFlat = residual;
    residualFlat.shape = Shape(totalTokens, dim);
    
    // Output projection with fused residual: toOut(attn) + residual
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalTokens, dim);
    toOut->queueExecute(task, outputFlat, {attnOutFlat, residualFlat});
}

// ============================================================================
// SDCrossAttention
// ============================================================================

SDCrossAttention::SDCrossAttention(
    RefPtr<InferencingContext> ctx,
    int queryDim,
    int contextDim,
    int numHeads)
    : ctx(ctx)
    , queryDim(queryDim)
    , contextDim(contextDim)
    , numHeads(numHeads)
    , headDim(queryDim / numHeads)
{
    toQ = new LinearKernel(ctx, queryDim, queryDim);
    toK = new LinearKernel(ctx, contextDim, queryDim);  // Context → query dim
    toV = new LinearKernel(ctx, contextDim, queryDim);
    // toOut with fused residual addition: output = linear(attn) + residual
    toOut = new LinearKernel(ctx, ElementType::Float32,
                             buffer(), kernelOutput() + buffer(), bufferSink(),
                             queryDim, queryDim);
    
    // Use FlashAttentionKernel (handles head transpose internally)
    // Input format: [B, S, H, headDim] - needs permute to [B, H, S, headDim]
    qExpr = buffer();
    kExpr = buffer();
    vExpr = buffer();
    
    auto qPlanar = permute(qExpr, {0, 2, 1, 3});  // [B,S,H,D] -> [B,H,S,D]
    auto kPlanar = permute(kExpr, {0, 2, 1, 3});
    auto vPlanar = permute(vExpr, {0, 2, 1, 3});
    // Output comes as [B, H, S, D], permute back to [B, S, H, D]
    auto outSink = permute(bufferSink(), {0, 2, 1, 3});
    
    flashAttn = new FlashAttentionKernel(
        ctx, qPlanar, kPlanar, vPlanar, kernelOutput(),
        32, 32, headDim, outSink);
}

SlangResult SDCrossAttention::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // SD 1.5 UNet attention has NO biases for Q/K/V projections
    SLANG_RETURN_ON_FAIL(toQ->loadParams(
        reader,
        (prefix + "to_q.weight").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(toK->loadParams(
        reader,
        (prefix + "to_k.weight").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(toV->loadParams(
        reader,
        (prefix + "to_v.weight").getUnownedSlice()));
    
    // to_out does have bias
    SLANG_RETURN_ON_FAIL(toOut->loadParams(
        reader,
        (prefix + "to_out.0.weight").getUnownedSlice(),
        (prefix + "to_out.0.bias").getUnownedSlice()));
    
    return SLANG_OK;
}

void SDCrossAttention::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView query,
    TensorView context,
    TensorView residual)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = query.shape[0];
    int querySeqLen = query.shape[1];
    int contextSeqLen = context.shape[1];
    int totalQueryTokens = batchSize * querySeqLen;
    int totalContextTokens = batchSize * contextSeqLen;
    
    // Flatten for linear layers
    TensorView queryFlat = query;
    queryFlat.shape = Shape(totalQueryTokens, queryDim);
    TensorView contextFlat = context;
    contextFlat.shape = Shape(totalContextTokens, contextDim);
    
    // Q from query, K/V from context
    auto qOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalQueryTokens, queryDim));
    auto kOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalContextTokens, queryDim));
    auto vOut = ctx->allocScratchTensor(ElementType::Float32, Shape(totalContextTokens, queryDim));
    
    toQ->queueExecute(task, qOut, queryFlat);
    toK->queueExecute(task, kOut, contextFlat);
    toV->queueExecute(task, vOut, contextFlat);
    
    // Reshape for FlashAttention: [B*S, D] → [B, S, H, headDim]
    TensorView qReshaped = qOut;
    qReshaped.shape = Shape(batchSize, querySeqLen, numHeads, headDim);
    TensorView kReshaped = kOut;
    kReshaped.shape = Shape(batchSize, contextSeqLen, numHeads, headDim);
    TensorView vReshaped = vOut;
    vReshaped.shape = Shape(batchSize, contextSeqLen, numHeads, headDim);
    
    // Flash attention (handles transpose and scaling internally)
    auto attnOut = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, querySeqLen, numHeads, headDim));
    
    Dictionary<Expr, InputInfo> attnInputs;
    attnInputs.add(qExpr, qReshaped);
    attnInputs.add(kExpr, kReshaped);
    attnInputs.add(vExpr, vReshaped);
    
    float scale = 1.0f / sqrtf((float)headDim);
    flashAttn->queueExecute(task, attnOut, attnInputs,
        querySeqLen, contextSeqLen, numHeads, batchSize, scale, false);
    
    // Reshape back and output projection: [B, S, H, headDim] → [B*S, D]
    TensorView attnOutFlat = attnOut;
    attnOutFlat.shape = Shape(totalQueryTokens, queryDim);
    
    // Flatten residual for toOut
    TensorView residualFlat = residual;
    residualFlat.shape = Shape(totalQueryTokens, queryDim);
    
    // Output projection with fused residual: toOut(attn) + residual
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalQueryTokens, queryDim);
    toOut->queueExecute(task, outputFlat, {attnOutFlat, residualFlat});
}

// ============================================================================
// SDFeedForward
// ============================================================================

SDFeedForward::SDFeedForward(
    RefPtr<InferencingContext> ctx,
    int dim,
    int mult)
    : ctx(ctx)
    , dim(dim)
    , innerDim(dim * mult)
{
    // GEGLU: proj1 outputs 2x innerDim, split and gate
    proj1 = new LinearKernel(ctx, dim, innerDim * 2);
    
    // proj2 with fused GEGLU (inputExpr) and residual addition (outputExpr)
    // inputExpr: hidden_states * gelu(gate) - takes two input buffers
    // outputExpr: kernelOutput() + buffer() - adds residual (third input buffer)
    proj2 = new LinearKernel(ctx, ElementType::Float32,
                             buffer() * gelu(buffer()), kernelOutput() + buffer(), bufferSink(),
                             innerDim, dim);
}

SlangResult SDFeedForward::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    SLANG_RETURN_ON_FAIL(proj1->loadParams(
        reader,
        (prefix + "net.0.proj.weight").getUnownedSlice(),
        (prefix + "net.0.proj.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(proj2->loadParams(
        reader,
        (prefix + "net.2.weight").getUnownedSlice(),
        (prefix + "net.2.bias").getUnownedSlice()));
    
    return SLANG_OK;
}

void SDFeedForward::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    TensorView residual)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int totalTokens = input.shape[0] * input.shape[1];
    
    TensorView inputFlat = input;
    inputFlat.shape = Shape(totalTokens, dim);
    
    // proj1 → [totalTokens, innerDim * 2]  
    auto proj1Out = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, innerDim * 2));
    proj1->queueExecute(task, proj1Out, inputFlat);
    
    // GEGLU: split along last dim, apply gelu to gate, multiply with value
    // Reshape to [totalTokens, 2, innerDim] then permute to [2, totalTokens, innerDim]
    // This makes gate (idx 0) and value (idx 1) contiguous blocks that can be sliced
    TensorView proj1Reshaped = proj1Out;
    proj1Reshaped.shape = Shape(totalTokens, 2, innerDim);
    
    // Permute [totalTokens, 2, innerDim] -> [2, totalTokens, innerDim]
    auto permuteExpr = permute(buffer(), {1, 0, 2});
    auto permutedBuf = ctx->allocScratchTensor(ElementType::Float32, Shape(2, totalTokens, innerDim));
    auto permuteKernel = new ElementwiseKernel(ctx, permuteExpr);
    permuteKernel->queueExecute(task, permutedBuf, proj1Reshaped);
    
    // Use slice() to get hidden_states [0,:,:] and gate [1,:,:] as contiguous views
    // Then squeezeFront(1) to remove the leading singleton dimension
    // Diffusers GEGLU: hidden_states, gate = proj.chunk(2, dim=-1); return hidden_states * gelu(gate)
    TensorView hiddenStatesView = permutedBuf.slice(0, 1).squeezeFront(1);  // first half
    TensorView gateView = permutedBuf.slice(1, 1).squeezeFront(1);          // second half
    
    // Flatten residual
    TensorView residualFlat = residual;
    residualFlat.shape = Shape(totalTokens, dim);
    
    // proj2 with fused GEGLU + residual: (hidden_states * gelu(gate)) + residual
    TensorView outputFlat = output;
    outputFlat.shape = Shape(totalTokens, dim);
    proj2->queueExecute(task, outputFlat, {hiddenStatesView, gateView, residualFlat});
}

// ============================================================================
// SDBasicTransformerBlock
// ============================================================================

SDBasicTransformerBlock::SDBasicTransformerBlock(
    RefPtr<InferencingContext> ctx,
    int dim,
    int contextDim,
    int numHeads)
    : ctx(ctx)
    , dim(dim)
    , contextDim(contextDim)
    , numHeads(numHeads)
{
    norm1 = new LayerNormKernel(ctx, dim);
    selfAttn = new SDSelfAttention(ctx, dim, numHeads);
    norm2 = new LayerNormKernel(ctx, dim);
    crossAttn = new SDCrossAttention(ctx, dim, contextDim, numHeads);
    norm3 = new LayerNormKernel(ctx, dim);
    ff = new SDFeedForward(ctx, dim);
}

SlangResult SDBasicTransformerBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    SLANG_RETURN_ON_FAIL(norm1->loadParams(
        reader,
        (prefix + "norm1.weight").getUnownedSlice(),
        (prefix + "norm1.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(selfAttn->loadParams(reader, prefix + "attn1."));
    
    SLANG_RETURN_ON_FAIL(norm2->loadParams(
        reader,
        (prefix + "norm2.weight").getUnownedSlice(),
        (prefix + "norm2.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(crossAttn->loadParams(reader, prefix + "attn2."));
    
    SLANG_RETURN_ON_FAIL(norm3->loadParams(
        reader,
        (prefix + "norm3.weight").getUnownedSlice(),
        (prefix + "norm3.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(ff->loadParams(reader, prefix + "ff."));
    
    return SLANG_OK;
}

void SDBasicTransformerBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    TensorView context)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = input.shape[0];
    int seqLen = input.shape[1];
    int totalTokens = batchSize * seqLen;
    
    // Self-attention with fused residual: output = selfAttn(norm(input)) + input
    TensorView inputFlat = input;
    inputFlat.shape = Shape(totalTokens, dim);
    
    auto normed1 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, dim));
    norm1->queueExecute(task, normed1, inputFlat);
    
    TensorView normed1_3d = normed1;
    normed1_3d.shape = Shape(batchSize, seqLen, dim);
    
    auto residual1 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, seqLen, dim));
    selfAttn->queueExecute(task, residual1, normed1_3d, input);  // fused: selfAttn + input
    
    // Cross-attention with fused residual: output = crossAttn(norm(residual1)) + residual1
    TensorView residual1Flat = residual1;
    residual1Flat.shape = Shape(totalTokens, dim);
    
    auto normed2 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, dim));
    norm2->queueExecute(task, normed2, residual1Flat);
    
    TensorView normed2_3d = normed2;
    normed2_3d.shape = Shape(batchSize, seqLen, dim);
    
    auto residual2 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, seqLen, dim));
    crossAttn->queueExecute(task, residual2, normed2_3d, context, residual1);  // fused: crossAttn + residual1
    
    // Feed-forward with fused residual: output = ff(norm(residual2)) + residual2
    TensorView residual2Flat = residual2;
    residual2Flat.shape = Shape(totalTokens, dim);
    
    auto normed3 = ctx->allocScratchTensor(ElementType::Float32, Shape(totalTokens, dim));
    norm3->queueExecute(task, normed3, residual2Flat);
    
    TensorView normed3_3d = normed3;
    normed3_3d.shape = Shape(batchSize, seqLen, dim);
    
    ff->queueExecute(task, output, normed3_3d, residual2);  // fused: ff + residual2
}

// ============================================================================
// SDSpatialTransformer
// ============================================================================

SDSpatialTransformer::SDSpatialTransformer(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int numHeads,
    int contextDim,
    int numLayers)
    : ctx(ctx)
    , inChannels(inChannels)
    , numHeads(numHeads)
    , contextDim(contextDim)
    , numLayers(numLayers)
{
    norm = new GroupNormKernel(ctx, inChannels, 32);
    projIn = new Conv2DKernel(ctx, 16, 1, 1, inChannels, inChannels);
    
    for (int i = 0; i < numLayers; i++)
    {
        blocks.add(new SDBasicTransformerBlock(ctx, inChannels, contextDim, numHeads));
    }
    
    // projOut with fused residual addition: conv_output + residual
    projOut = new Conv2DKernel(ctx, ElementType::Float32, 16, 1, 1, inChannels, inChannels,
                               buffer(), kernelOutput() + buffer(), bufferSink());
}

SlangResult SDSpatialTransformer::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    SLANG_RETURN_ON_FAIL(norm->loadParams(
        reader,
        (prefix + "norm.weight").getUnownedSlice(),
        (prefix + "norm.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(projIn->loadParams(
        reader,
        (prefix + "proj_in.weight").getUnownedSlice(),
        (prefix + "proj_in.bias").getUnownedSlice()));
    
    for (Index i = 0; i < blocks.getCount(); i++)
    {
        String blockPrefix = prefix + "transformer_blocks." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(blocks[i]->loadParams(reader, blockPrefix));
    }
    
    SLANG_RETURN_ON_FAIL(projOut->loadParams(
        reader,
        (prefix + "proj_out.weight").getUnownedSlice(),
        (prefix + "proj_out.bias").getUnownedSlice()));
    
    return SLANG_OK;
}

void SDSpatialTransformer::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    TensorView context)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    int seqLen = height * width;
    
    // GroupNorm
    auto normed = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, inChannels));
    norm->queueExecute(task, normed, input);
    
    // proj_in (1x1 conv)
    auto projected = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, height, width, inChannels));
    projIn->queueExecute(task, projected, normed, 0);
    
    // Reshape to sequence: [B, H, W, C] → [B, H*W, C]
    TensorView projectedSeq = projected;
    projectedSeq.shape = Shape(batchSize, seqLen, inChannels);
    
    // Transformer blocks
    TensorView current = projectedSeq;
    for (Index i = 0; i < blocks.getCount(); i++)
    {
        auto blockOut = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, seqLen, inChannels));
        blocks[i]->queueExecute(task, blockOut, current, context);
        current = blockOut;
    }
    
    // Reshape back: [B, H*W, C] → [B, H, W, C]
    TensorView currentSpatial = current;
    currentSpatial.shape = Shape(batchSize, height, width, inChannels);
    
    // proj_out with fused residual: {currentSpatial, input} → output
    projOut->queueExecute(task, output, {currentSpatial, input}, 0);
}

// ============================================================================
// SDDownBlock
// ============================================================================

SDDownBlock::SDDownBlock(
    RefPtr<InferencingContext> ctx,
    int inChannels,
    int outChannels,
    int timeEmbedDim,
    bool hasAttention,
    bool hasDownsample,
    int numHeads,
    int contextDim,
    int numLayers)
    : ctx(ctx)
    , inChannels(inChannels)
    , outChannels(outChannels)
    , timeEmbedDim(timeEmbedDim)
    , hasAttention(hasAttention)
    , hasDownsample(hasDownsample)
    , numLayers(numLayers)
{
    // First ResNet takes inChannels, rest take outChannels
    resnets.add(new SDResNetBlock(ctx, inChannels, outChannels, timeEmbedDim));
    for (int i = 1; i < numLayers; i++)
    {
        resnets.add(new SDResNetBlock(ctx, outChannels, outChannels, timeEmbedDim));
    }
    
    if (hasAttention)
    {
        for (int i = 0; i < numLayers; i++)
        {
            transformers.add(new SDSpatialTransformer(ctx, outChannels, numHeads, contextDim));
        }
    }
    
    if (hasDownsample)
    {
        downsample = new Conv2DKernel(ctx, 16, 3, 2, outChannels, outChannels);
    }
}

SlangResult SDDownBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    for (Index i = 0; i < resnets.getCount(); i++)
    {
        String resnetPrefix = prefix + "resnets." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(resnets[i]->loadParams(reader, resnetPrefix));
    }
    
    if (hasAttention)
    {
        for (Index i = 0; i < transformers.getCount(); i++)
        {
            String attnPrefix = prefix + "attentions." + String(i) + ".";
            SLANG_RETURN_ON_FAIL(transformers[i]->loadParams(reader, attnPrefix));
        }
    }
    
    if (hasDownsample)
    {
        SLANG_RETURN_ON_FAIL(downsample->loadParams(
            reader,
            (prefix + "downsamplers.0.conv.weight").getUnownedSlice(),
            (prefix + "downsamplers.0.conv.bias").getUnownedSlice()));
    }
    
    return SLANG_OK;
}

void SDDownBlock::queueExecute(
    InferencingTask& task,
    List<TensorView>& hiddenStates,
    TensorView input,
    TensorView timeEmbed,
    TensorView context)
{
    int batchSize = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    
    TensorView current = input;
    
    for (Index i = 0; i < resnets.getCount(); i++)
    {
        int currentChannels = (i == 0) ? inChannels : outChannels;
        int nextHeight = current.shape[1];
        int nextWidth = current.shape[2];
        
        auto resnetOut = ctx->allocScratchTensor(ElementType::Float32, 
            Shape(batchSize, nextHeight, nextWidth, outChannels));
        resnets[i]->queueExecute(task, resnetOut, current, timeEmbed);
        current = resnetOut;
        
        if (hasAttention && i < transformers.getCount())
        {
            auto attnOut = ctx->allocScratchTensor(ElementType::Float32, 
                Shape(batchSize, nextHeight, nextWidth, outChannels));
            transformers[i]->queueExecute(task, attnOut, current, context);
            current = attnOut;
        }
        
        // Save for skip connection
        hiddenStates.add(current);
    }
    
    if (hasDownsample)
    {
        int newHeight = current.shape[1] / 2;
        int newWidth = current.shape[2] / 2;
        auto downsampled = ctx->allocScratchTensor(ElementType::Float32, 
            Shape(batchSize, newHeight, newWidth, outChannels));
        downsample->queueExecute(task, downsampled, current, 1);
        hiddenStates.add(downsampled);
    }
}

// ============================================================================
// SDMidBlock
// ============================================================================

SDMidBlock::SDMidBlock(
    RefPtr<InferencingContext> ctx,
    int channels,
    int timeEmbedDim,
    int numHeads,
    int contextDim)
    : ctx(ctx)
    , channels(channels)
    , timeEmbedDim(timeEmbedDim)
{
    resnet1 = new SDResNetBlock(ctx, channels, channels, timeEmbedDim);
    transformer = new SDSpatialTransformer(ctx, channels, numHeads, contextDim);
    resnet2 = new SDResNetBlock(ctx, channels, channels, timeEmbedDim);
}

SlangResult SDMidBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    SLANG_RETURN_ON_FAIL(resnet1->loadParams(reader, prefix + "resnets.0."));
    SLANG_RETURN_ON_FAIL(transformer->loadParams(reader, prefix + "attentions.0."));
    SLANG_RETURN_ON_FAIL(resnet2->loadParams(reader, prefix + "resnets.1."));
    return SLANG_OK;
}

void SDMidBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    TensorView timeEmbed,
    TensorView context)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    
    auto resnet1Out = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, height, width, channels));
    resnet1->queueExecute(task, resnet1Out, input, timeEmbed);
    
    auto attnOut = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, height, width, channels));
    transformer->queueExecute(task, attnOut, resnet1Out, context);
    
    resnet2->queueExecute(task, output, attnOut, timeEmbed);
}

// ============================================================================
// SDUpBlock
// ============================================================================

SDUpBlock::SDUpBlock(
    RefPtr<InferencingContext> ctx,
    int outChannels,
    int prevChannels,
    List<int> skipChannels,
    int timeEmbedDim,
    bool hasAttention,
    bool hasUpsample,
    int numHeads,
    int contextDim)
    : ctx(ctx)
    , outChannels(outChannels)
    , prevChannels(prevChannels)
    , timeEmbedDim(timeEmbedDim)
    , hasAttention(hasAttention)
    , hasUpsample(hasUpsample)
    , skipChannelsList(skipChannels)
{
    // ResNets take concatenated input (skip connection + current)
    // Each resnet may have different skip channel count
    // Use fused concat to eliminate separate ConcatKernel
    int numLayers = (int)skipChannels.getCount();
    for (int i = 0; i < numLayers; i++)
    {
        int currentChannels = (i == 0) ? prevChannels : outChannels;
        int skipCh = skipChannels[i];
        // fuseConcatInput=true: norm1 and residualConv will have fused concat
        resnets.add(new SDResNetBlock(ctx, currentChannels + skipCh, outChannels, timeEmbedDim, true));
    }
    
    if (hasAttention)
    {
        for (int i = 0; i < numLayers; i++)
        {
            transformers.add(new SDSpatialTransformer(ctx, outChannels, numHeads, contextDim));
        }
    }
    
    if (hasUpsample)
    {
        // Fuse nearest-neighbor 2x upsample into conv input expression
        // Must use full constructor: inputExpr=upsample2x(buffer()), outputExpr=kernelOutput(), sinkExpr=bufferSink()
        upsample = new Conv2DKernel(ctx, 16, 3, 1, outChannels, outChannels,
                                     upsample2x(buffer()), kernelOutput(), bufferSink());
    }
}

SlangResult SDUpBlock::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    for (Index i = 0; i < resnets.getCount(); i++)
    {
        String resnetPrefix = prefix + "resnets." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(resnets[i]->loadParams(reader, resnetPrefix));
    }
    
    if (hasAttention)
    {
        for (Index i = 0; i < transformers.getCount(); i++)
        {
            String attnPrefix = prefix + "attentions." + String(i) + ".";
            SLANG_RETURN_ON_FAIL(transformers[i]->loadParams(reader, attnPrefix));
        }
    }
    
    if (hasUpsample)
    {
        SLANG_RETURN_ON_FAIL(upsample->loadParams(
            reader,
            (prefix + "upsamplers.0.conv.weight").getUnownedSlice(),
            (prefix + "upsamplers.0.conv.bias").getUnownedSlice()));
    }
    
    return SLANG_OK;
}

void SDUpBlock::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView input,
    List<TensorView>& skipConnections,
    TensorView timeEmbed,
    TensorView context)
{
    InferencingContext::ScratchScope scope(ctx);
    
    int batchSize = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    
    TensorView current = input;
    Index numResnets = resnets.getCount();
    
    for (Index i = 0; i < numResnets; i++)
    {
        // Pop skip connection from back
        TensorView skip = skipConnections[skipConnections.getCount() - 1];
        skipConnections.removeLast();
        
        int currentHeight = current.shape[1];
        int currentWidth = current.shape[2];
        
        bool isLastIteration = (i == numResnets - 1);
        bool hasTransformerThisIteration = hasAttention && i < transformers.getCount();
        bool resnetWritesToOutput = isLastIteration && !hasUpsample && !hasTransformerThisIteration;
        
        TensorView resnetDest = resnetWritesToOutput 
            ? output 
            : ctx->allocScratchTensor(ElementType::Float32, 
                Shape(batchSize, currentHeight, currentWidth, outChannels));
        
        // Use two-input queueExecute with fused concat (current, skip)
        resnets[i]->queueExecute(task, resnetDest, current, skip, timeEmbed);
        current = resnetDest;
        
        if (hasTransformerThisIteration)
        {
            bool transformerWritesToOutput = isLastIteration && !hasUpsample;
            TensorView attnDest = transformerWritesToOutput
                ? output
                : ctx->allocScratchTensor(ElementType::Float32, 
                    Shape(batchSize, currentHeight, currentWidth, outChannels));
            transformers[i]->queueExecute(task, attnDest, current, context);
            current = attnDest;
        }
    }
    
    if (hasUpsample)
    {
        // Conv with fused nearest-neighbor 2x upsample writes directly to output
        upsample->queueExecute(task, output, current, 1);
    }
    // else: last resnet/transformer already wrote to output
}

// ============================================================================
// SDUNet
// ============================================================================

SDUNet::SDUNet(RefPtr<InferencingContext> ctx)
    : ctx(ctx)
{
    // Channel progression: 320, 640, 1280, 1280
    channelMult.add(320);
    channelMult.add(640);
    channelMult.add(1280);
    channelMult.add(1280);
    
    // Time embedding: sinusoidal (320) → Linear → SiLU → Linear (1280)
    // Fuse SiLU into timeProj1 output expression
    timeProj1 = new LinearKernel(ctx, ElementType::Float32, buffer(), silu(kernelOutput()), 
                                  bufferSink(), 320, timeEmbedDim);
    timeProj2 = new LinearKernel(ctx, timeEmbedDim, timeEmbedDim);
    
    // Input conv (use default kernelOutput() for output expression)
    convIn = new Conv2DKernel(ctx, 16, 3, 1, inChannels, channelMult[0]);
    
    // Down blocks (SD 1.5: CrossAttnDownBlock2D x3, then DownBlock2D)
    // Block 0: 320 → 320, HAS attention, downsample
    downBlocks.add(new SDDownBlock(ctx, channelMult[0], channelMult[0], timeEmbedDim, 
                                    true, true, 8, contextDim, 2));
    // Block 1: 320 → 640, HAS attention, downsample
    downBlocks.add(new SDDownBlock(ctx, channelMult[0], channelMult[1], timeEmbedDim, 
                                    true, true, 8, contextDim, 2));
    // Block 2: 640 → 1280, HAS attention, downsample
    downBlocks.add(new SDDownBlock(ctx, channelMult[1], channelMult[2], timeEmbedDim, 
                                    true, true, 8, contextDim, 2));
    // Block 3: 1280 → 1280, NO attention, NO downsample
    downBlocks.add(new SDDownBlock(ctx, channelMult[2], channelMult[3], timeEmbedDim, 
                                    false, false, 8, contextDim, 2));
    
    // Mid block
    midBlock = new SDMidBlock(ctx, channelMult[3], timeEmbedDim, 8, contextDim);
    
    // Up blocks (SD 1.5: UpBlock2D, then CrossAttnUpBlock2D x3)
    // Skip channels are consumed in reverse order from down blocks
    // Total skips: 11 (down[0,1,2] produce 3 each, down[3] produces 2)
    // 
    // Skip list (in order stored): 
    //   [0-2]: down[0] = 320,320,320
    //   [3-5]: down[1] = 640,640,640
    //   [6-8]: down[2] = 1280,1280,1280
    //   [9-10]: down[3] = 1280,1280
    //
    // Consumption (from back): 
    //   up[0]: 3 skips (10,9,8) = 1280,1280,1280
    //   up[1]: 3 skips (7,6,5) = 1280,1280,640
    //   up[2]: 3 skips (4,3,2) = 640,640,320
    //   up[3]: 2 skips (1,0) = 320,320
    
    // Block 0: consumes skips 10,9,8 (down[3].res1, down[3].res0, down[2].downsample)
    upBlocks.add(new SDUpBlock(ctx, channelMult[3], channelMult[3], 
                                List<int>({1280, 1280, 1280}),
                                timeEmbedDim, false, true, 8, contextDim));
    // Block 1: consumes skips 7,6,5 (down[2].res1, down[2].res0, down[1].downsample)
    upBlocks.add(new SDUpBlock(ctx, channelMult[2], channelMult[3], 
                                List<int>({1280, 1280, 640}),
                                timeEmbedDim, true, true, 8, contextDim));
    // Block 2: consumes skips 4,3,2 (down[1].res1, down[1].res0, down[0].downsample)
    upBlocks.add(new SDUpBlock(ctx, channelMult[1], channelMult[2], 
                                List<int>({640, 640, 320}),
                                timeEmbedDim, true, true, 8, contextDim));
    // Block 3: consumes skips 2,1,0 (down[0].res1, down[0].res0, conv_in)
    upBlocks.add(new SDUpBlock(ctx, channelMult[0], channelMult[1], 
                                List<int>({320, 320, 320}),
                                timeEmbedDim, true, false, 8, contextDim));
    
    // Output
    normOut = new GroupNormKernel(ctx, channelMult[0], 32);
    convOut = new Conv2DKernel(ctx, ElementType::Float32, 16, 3, 1, channelMult[0], outChannels,
                                silu(buffer()), kernelOutput(), bufferSink());
}

SlangResult SDUNet::loadParams(SafeTensorsReader& reader, const String& prefix)
{
    // Time embedding
    SLANG_RETURN_ON_FAIL(timeProj1->loadParams(
        reader,
        (prefix + "time_embedding.linear_1.weight").getUnownedSlice(),
        (prefix + "time_embedding.linear_1.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(timeProj2->loadParams(
        reader,
        (prefix + "time_embedding.linear_2.weight").getUnownedSlice(),
        (prefix + "time_embedding.linear_2.bias").getUnownedSlice()));
    
    // Input conv
    SLANG_RETURN_ON_FAIL(convIn->loadParams(
        reader,
        (prefix + "conv_in.weight").getUnownedSlice(),
        (prefix + "conv_in.bias").getUnownedSlice()));
    
    // Down blocks
    for (Index i = 0; i < downBlocks.getCount(); i++)
    {
        String blockPrefix = prefix + "down_blocks." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(downBlocks[i]->loadParams(reader, blockPrefix));
    }
    
    // Mid block
    SLANG_RETURN_ON_FAIL(midBlock->loadParams(reader, prefix + "mid_block."));
    
    // Up blocks
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        String blockPrefix = prefix + "up_blocks." + String(i) + ".";
        SLANG_RETURN_ON_FAIL(upBlocks[i]->loadParams(reader, blockPrefix));
    }
    
    // Output
    SLANG_RETURN_ON_FAIL(normOut->loadParams(
        reader,
        (prefix + "conv_norm_out.weight").getUnownedSlice(),
        (prefix + "conv_norm_out.bias").getUnownedSlice()));
    
    SLANG_RETURN_ON_FAIL(convOut->loadParams(
        reader,
        (prefix + "conv_out.weight").getUnownedSlice(),
        (prefix + "conv_out.bias").getUnownedSlice()));
    
    return SLANG_OK;
}

TensorView SDUNet::allocateResultBuffer(
    ElementType elementType,
    int height,
    int width,
    int batchSize)
{
    return ctx->allocScratchTensor(elementType, Shape(batchSize, height, width, outChannels));
}

void SDUNet::queueExecute(
    InferencingTask& task,
    TensorView output,
    TensorView latent,
    int timestep,
    TensorView context)
{
    int batchSize = latent.shape[0];
    int height = latent.shape[1];
    int width = latent.shape[2];
    
    // ========================================================================
    // PRE-ALLOCATE key buffers
    // ========================================================================
    
    // Time embedding: sinusoidal → MLP
    // Recreate tensor only if timestep changed (member variable persists across async calls)
    if (cachedTimestep != timestep)
    {
        List<float> sinEmbed;
        sinEmbed.setCount(320);
        getSinusoidalEmbedding(sinEmbed.getBuffer(), timestep, 320);
        
        sinEmbedTensor = ctx->createTensor(ElementType::Float32, Shape(1, 320), sinEmbed, "SinEmbed");
        cachedTimestep = timestep;
    }
    
    // Expand for batch if needed
    TensorView sinEmbedView = sinEmbedTensor->getView();
    if (batchSize > 1)
    {
        // For simplicity, assume batch size 1 for now
        // TODO: Tile the embedding for larger batches
    }
    
    auto timeEmbed1 = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, timeEmbedDim));
    timeProj1->queueExecute(task, timeEmbed1, sinEmbedView);
    
    auto timeEmbed = ctx->allocScratchTensor(ElementType::Float32, Shape(batchSize, timeEmbedDim));
    timeProj2->queueExecute(task, timeEmbed, timeEmbed1);
    
    // Input conv
    auto hidden = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, height, width, channelMult[0]));
    convIn->queueExecute(task, hidden, latent, 1);
    
    // Down blocks - collect skip connections
    List<TensorView> skipConnections;
    skipConnections.add(hidden);  // Initial hidden state
    
    TensorView current = hidden;
    for (Index i = 0; i < downBlocks.getCount(); i++)
    {
        List<TensorView> blockSkips;
        downBlocks[i]->queueExecute(task, blockSkips, current, timeEmbed, context);
        
        // Add all skip connections
        for (Index j = 0; j < blockSkips.getCount(); j++)
        {
            skipConnections.add(blockSkips[j]);
        }
        
        // Last one is the output for next block
        current = skipConnections[skipConnections.getCount() - 1];
    }
    
    // Mid block
    int midHeight = current.shape[1];
    int midWidth = current.shape[2];
    auto midOut = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, midHeight, midWidth, channelMult[3]));
    midBlock->queueExecute(task, midOut, current, timeEmbed, context);
    current = midOut;
    
    // Up blocks - consume skip connections
    for (Index i = 0; i < upBlocks.getCount(); i++)
    {
        int outChannels = upBlocks[i]->outChannels;
        int upHeight = current.shape[1];
        int upWidth = current.shape[2];
        
        if (upBlocks[i]->hasUpsample)
        {
            upHeight *= 2;
            upWidth *= 2;
        }
        
        auto upOut = ctx->allocScratchTensor(ElementType::Float32, 
            Shape(batchSize, upHeight, upWidth, outChannels));
        upBlocks[i]->queueExecute(task, upOut, current, skipConnections, timeEmbed, context);
        current = upOut;
    }
    
    // Output: GroupNorm → SiLU → Conv
    auto normed = ctx->allocScratchTensor(ElementType::Float32, 
        Shape(batchSize, height, width, channelMult[0]));
    normOut->queueExecute(task, normed, current);
    
    convOut->queueExecute(task, output, normed, 1);
}


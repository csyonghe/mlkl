implementing mlkl;

// Generalized Parameter Structure for Cross-Attention
struct FlashAttentionParams<T : ITensorElement, TQ : IExpr<T>, TK : IExpr<T>, TV : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>> 
{
    TQ Q;
    TK K;
    TV V;
    FOut outFunc;
    TSink sink;
    uint seq_len_q;
    uint seq_len_kv;
    uint num_heads;
    float scale;  // Keep as float for scalar parameter
    uint is_causal;
}

[numthreads(blockSizeR, 1, 1)]
void flashAttention2<let blockSizeR : int, let blockSizeC : int, let headDimension : int, 
                     T : ITensorElement, TQ : IExpr<T>, TK : IExpr<T>, TV : IExpr<T>, TSink : ISink<T>, FOut : IExpr<T>>(
    ConstantBuffer<FlashAttentionParams<T, TQ, TK, TV, TSink, FOut>, CDataLayout> params,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID)
{
    uint tid = groupThreadID.x;
    uint row_idx = groupID.x * blockSizeR + tid;
    uint head_idx = groupID.y;
    uint batch_idx = groupID.z;

    // Use T.UnpackedType for shared memory and accumulation
    static groupshared T.UnpackedType s_K[blockSizeC][headDimension];
    static groupshared T.UnpackedType s_V[blockSizeC][headDimension];

    // 1. Online Softmax Accumulators
    // Keep max and sum in float for numerical stability (can be very large/small values)
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    
    // Accumulator in T.UnpackedType
    T.UnpackedType acc[headDimension];
    for (uint i = 0; i < headDimension; i++) acc[i] = T.UnpackedType(0);

    // 2. Load Q (Static for the duration of this row block)
    T.UnpackedType q_local[headDimension];
    Input<T> emptyInput = {};
    emptyInput.value = T.UnpackedType(0);
    
    if (row_idx < params.seq_len_q) {
        for (uint d = 0; d < headDimension; d++) {
            // Q shape: [Batch, NumHeads, SeqQ, HeadDim]
            q_local[d] = params.Q.eval(Coord(batch_idx, head_idx, row_idx, d), emptyInput);
        }
    }

    // 3. Loop over blocks of KV tokens
    uint num_kv_blocks = (params.seq_len_kv + blockSizeC - 1) / blockSizeC;
    for (uint j = 0; j < num_kv_blocks; j++) {
        
        // Collaborative load of K and V into Shared Memory
        uint kv_row_in_tile = tid;
        uint kv_row_global = j * blockSizeC + kv_row_in_tile;
        
        if (kv_row_global < params.seq_len_kv) {
            for (uint d = 0; d < headDimension; d++) {
                // K, V shape: [Batch, NumHeads, SeqKV, HeadDim]
                Coord kvCoord = Coord(batch_idx, head_idx, kv_row_global, d);
                s_K[kv_row_in_tile][d] = params.K.eval(kvCoord, emptyInput);
                s_V[kv_row_in_tile][d] = params.V.eval(kvCoord, emptyInput);
            }
        } else {
            // Out of bounds padding for incomplete tiles
            for (uint d = 0; d < headDimension; d++) {
                s_K[kv_row_in_tile][d] = T.UnpackedType(0);
                s_V[kv_row_in_tile][d] = T.UnpackedType(0);
            }
        }
        
        GroupMemoryBarrierWithGroupSync();

        // 4. Compute Attention scores for this KV tile
        if (row_idx < params.seq_len_q) {
            float m_curr = -1e30f;
            // Scores in T.UnpackedType, but max finding uses float for stability
            T.UnpackedType scores[blockSizeC];

            for (uint c = 0; c < blockSizeC; c++) {
                uint col_idx = j * blockSizeC + c;
                // Check bounds and causal mask (causal usually only applied in self-attention)
                if (col_idx >= params.seq_len_kv || (params.is_causal != 0 && row_idx < col_idx)) {
                    scores[c] = T.UnpackedType.fromFloat(-1e30f);
                } else {
                    T.UnpackedType sum = T.UnpackedType(0);
                    for (uint d = 0; d < headDimension; d++) {
                        sum = sum + q_local[d] * s_K[c][d];
                    }
                    scores[c] = sum * T.UnpackedType.fromFloat(params.scale);
                }
                m_curr = max(m_curr, scores[c].toFloat());
            }

            // 5. Online Softmax Rescale and update accumulators
            float m_new = max(m_prev, m_curr);
            float exp_prev = exp(m_prev - m_new);
            float exp_curr = exp(m_curr - m_new);
            
            // Rescale the existing accumulator to the new maximum ONCE per block
            T.UnpackedType scale_prev = T.UnpackedType.fromFloat(exp_prev);
            for (uint d = 0; d < headDimension; d++) {
                acc[d] = acc[d] * scale_prev;
            }

            float p_sum = 0.0f;
            for (uint c = 0; c < blockSizeC; c++) {
                float p_ij = exp(scores[c].toFloat() - m_curr);
                p_sum += p_ij;
                
                // P_ij is already relative to m_curr, so we scale it by exp_curr
                // to make it relative to the new global m_new.
                T.UnpackedType p_scaled = T.UnpackedType.fromFloat(p_ij * exp_curr); 
                
                for (uint d = 0; d < headDimension; d++) {
                    acc[d] = acc[d] + p_scaled * s_V[c][d];
                }
            }
            // Update the denominator
            l_prev = (l_prev * exp_prev) + (p_sum * exp_curr);
            m_prev = m_new;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 6. Final Normalization and Output Fusion
    if (row_idx < params.seq_len_q) {
        T.UnpackedType inv_l = T.UnpackedType.fromFloat(1.0f / l_prev);
        for (uint d = 0; d < headDimension; d++) {
            Coord outCoord = Coord(batch_idx, head_idx, row_idx, d);
            T.UnpackedType attentionVal = acc[d] * inv_l;
            
            Input<T> outInput = {};
            outInput.value = attentionVal;
            
            params.sink.store(outCoord, params.outFunc.eval(outCoord, outInput));
        }
    }
}

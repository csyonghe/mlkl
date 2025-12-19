implementing mlkl;

// Generalized Parameter Structure for Cross-Attention
struct FlashAttentionParams<TQ: IExpr, TK: IExpr, TV: IExpr, TSink:ISink,FOut: IExpr> {
    TQ Q;
    TK K;
    TV V;
    FOut outFunc;
    TSink sink;
    uint seq_len_q;
    uint seq_len_kv;
    uint num_heads;
    float scale;
    uint is_causal;
}

[numthreads(blockSizeR, 1, 1)]
void flashAttention2<let blockSizeR : int, let blockSizeC : int, let headDimension : int, TQ: IExpr, TK: IExpr, TV: IExpr, TSink:ISink, FOut: IExpr>(
    ConstantBuffer<FlashAttentionParams<TQ, TK, TV, TSink, FOut>, CDataLayout> params,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID)
{
    uint tid = groupThreadID.x;
    uint row_idx = groupID.x * blockSizeR + tid;
    uint head_idx = groupID.y;
    uint batch_idx = groupID.z;

    // SRAM for tiling
    static groupshared float s_K[blockSizeC][headDimension];
    static groupshared float s_V[blockSizeC][headDimension];

    // --- DECOUPLED STRIDE LOGIC ---
    uint q_head_stride = params.seq_len_q * headDimension;
    uint kv_head_stride = params.seq_len_kv * headDimension;
    
    uint q_batch_stride = params.num_heads * q_head_stride;
    uint kv_batch_stride = params.num_heads * kv_head_stride;

    uint q_global_offset = (batch_idx * q_batch_stride) + (head_idx * q_head_stride);
    uint kv_global_offset = (batch_idx * kv_batch_stride) + (head_idx * kv_head_stride);

    // 1. Online Softmax Accumulators
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float acc[headDimension];
    for (uint i = 0; i < headDimension; i++) acc[i] = 0.0f;

    // 2. Load Q (Static for the duration of this row block)
    float q_local[headDimension];
    if (row_idx < params.seq_len_q) {
        for (uint d = 0; d < headDimension; d++) {
            uint linearIdx = q_global_offset + (row_idx * headDimension) + d;
            q_local[d] = params.Q.eval(linearIdx, Input(0.0f));
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
                uint linearIdx = kv_global_offset + (kv_row_global * headDimension) + d;
                s_K[kv_row_in_tile][d] = params.K.eval(linearIdx, Input(0.0f));
                s_V[kv_row_in_tile][d] = params.V.eval(linearIdx, Input(0.0f));
            }
        } else {
            // Out of bounds padding for incomplete tiles
            for (uint d = 0; d < headDimension; d++) {
                s_K[kv_row_in_tile][d] = 0.0f;
                s_V[kv_row_in_tile][d] = 0.0f;
            }
        }
        
        GroupMemoryBarrierWithGroupSync();

        // 4. Compute Attention scores for this KV tile
        if (row_idx < params.seq_len_q) {
            float m_curr = -1e30f;
            float scores[blockSizeC];

            for (uint c = 0; c < blockSizeC; c++) {
                uint col_idx = j * blockSizeC + c;
                // Check bounds and causal mask (causal usually only applied in self-attention)
                if (col_idx >= params.seq_len_kv || (params.is_causal != 0 && row_idx < col_idx)) {
                    scores[c] = -1e30f;
                } else {
                    float sum = 0.0f;
                    for (uint d = 0; d < headDimension; d++) {
                        sum += q_local[d] * s_K[c][d];
                    }
                    scores[c] = sum * params.scale;
                }
                m_curr = max(m_curr, scores[c]);
            }

            // 5. Online Softmax Rescale and update accumulators
            float m_new = max(m_prev, m_curr);
            float exp_prev = exp(m_prev - m_new);
            float exp_curr = exp(m_curr - m_new);
            // Rescale the existing accumulator to the new maximum ONCE per block
            for (uint d = 0; d < headDimension; d++) {
                acc[d] *= exp_prev;
            }

            float p_sum = 0.0f;
            for (uint c = 0; c < blockSizeC; c++) {
                float p_ij = exp(scores[c] - m_curr);
                p_sum += p_ij;
                
                // P_ij is already relative to m_curr, so we scale it by exp_curr
                // to make it relative to the new global m_new.
                float p_scaled = p_ij * exp_curr; 
                
                for (uint d = 0; d < headDimension; d++) {
                    acc[d] += p_scaled * s_V[c][d];
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
        for (uint d = 0; d < headDimension; d++) {
            uint destIdx = q_global_offset + (row_idx * headDimension) + d;
            float attentionVal = acc[d] / l_prev;
            params.sink.store(Coord(batch_idx, head_idx, row_idx, d), params.outFunc.eval(destIdx, Input(attentionVal)));
        }
    }
}
implementing mlkl;

// ============================================================================
// Prototype: Highly-specialized Winograd F(4x4,3x3) conv2d (FP32, NHWC)
// ============================================================================
// Constraints (prototype fast path):
// - kernel=3, stride=1, padding=1, dilation=1
// - NHWC contiguous layout: [B,H,W,C]
// - inChannels % 8 == 0, outChannels % 4 == 0
// - input/output/bias pointers are 16-byte aligned (so float4 loads/stores are valid)
//
// Weights are expected pre-transformed to Winograd domain (6x6) and packed as:
//   weightsU[pos][ocVec][icVec][o] where:
//     pos   = 0..35 (6x6, row-major)
//     ocVec = outChannels/4
//     icVec = inChannels/4
//     o     = 0..3 (each is a float4 of 4 IC scalars for one output channel)
//
// Flattened as float4*:
//   idx = (((pos * ocVecCount + ocVec) * icVecCount + icVec) * 4 + o)
//
// This allows computing 4 output channels at once (packed in float4) using dot(float4,float4).

struct WinogradConvProtoParams
{
    float* input;      // [B,H,W,Cin] NHWC
    float* output;     // [B,H,W,Cout] NHWC (same H,W due to pad=1, stride=1)
    float* bias;       // [Cout]
    float4* weightsU;  // packed transformed weights (see above)

    int H;
    int W;
    int inChannels;
    int outChannels;
    int batchSize;
};

// 6-point Winograd transform used for F(4,3) (same as in existing convolution.slang)
// Applied on float4 so it transforms 4 channels at once.
[ForceInline]
static void winogradTransform6(float4 d0, float4 d1, float4 d2, float4 d3, float4 d4, float4 d5,
                               out float4 t0, out float4 t1, out float4 t2, out float4 t3, out float4 t4, out float4 t5)
{
    // Constants
    const float4 c2 = float4(2.0);
    const float4 c4 = float4(4.0);
    const float4 c5 = float4(5.0);

    // Matches the transform used in `winogradConvolution`:
    // row0:  4*d0 - 5*d2 + d4
    // row1: -4*d1 - 4*d2 + d3 + d4
    // row2:  4*d1 - 4*d2 - d3 + d4
    // row3: -2*d1 - d2 + 2*d3 + d4
    // row4:  2*d1 - d2 - 2*d3 + d4
    // row5:  4*d1 - 5*d3 + d5
    t0 = d0 * c4 - d2 * c5 + d4;
    t1 = -d1 * c4 - d2 * c4 + d3 + d4;
    t2 = d1 * c4 - d2 * c4 - d3 + d4;
    t3 = -d1 * c2 - d2 + d3 * c2 + d4;
    t4 = d1 * c2 - d2 - d3 * c2 + d4;
    t5 = d1 * c4 - d3 * c5 + d5;
}

[ForceInline]
static float4 dot4_to_float4(float4 v, float4 w0, float4 w1, float4 w2, float4 w3)
{
    // Return 4 dot products packed in float4: [dot(v,w0), dot(v,w1), dot(v,w2), dot(v,w3)]
    float d0 = dot(v, w0);
    float d1 = dot(v, w1);
    float d2 = dot(v, w2);
    float d3 = dot(v, w3);
    return float4(d0, d1, d2, d3);
}

// Prototype kernel:
// - 128 threads = 4 warps (y dimension), each warp computes 4 output channels (float4)
// - laneId 0..15 correspond to the 16 output points in the 4x4 tile
// - each active lane writes one float4 output
// ---------------------------------------------------------------------------
// v0: single 4x4 tile per CTA (kept for reference)
// ---------------------------------------------------------------------------
[numthreads(32, 4, 1)]
void winogradConvProtoTile4(
    ConstantBuffer<WinogradConvProtoParams, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    // Tunables (prototype v1)
    static const int TILE_O = 4;          // output tile width/height
    static const int TILE_I = 6;          // input tile width/height
    static const int TILE_OC = 16;        // per CTA
    static const int TILE_IC = 8;         // per K-iteration
    static const int ICVEC = TILE_IC / 4; // 2

    const int ocVecCount = params.outChannels / 4;
    const int icVecCount = params.inChannels / 4;

    // Decode groupId.z into (batch, ocTile)
    uint ocTiles = (uint(params.outChannels) + TILE_OC - 1) / TILE_OC;
    uint batchIdx = groupId.z / ocTiles;
    uint ocTileIdx = groupId.z % ocTiles;
    if (batchIdx >= uint(params.batchSize)) return;

    // Tile origin in output
    int tileY = int(groupId.y) * TILE_O;
    int tileX = int(groupId.x) * TILE_O;

    // Thread mapping
    uint laneId = groupThreadId.x; // 0..31
    uint warpId = groupThreadId.y; // 0..3 (4 warps)
    uint tid = warpId * 32 + laneId; // 0..127

    // Each warp computes one ocVec (4 output channels)
    int ocVec = int(ocTileIdx) * (TILE_OC / 4) + int(warpId);
    if (ocVec >= ocVecCount) return;

    // Active lanes are the 16 output positions
    bool activeLane = laneId < 16;
    int localY = int(laneId) >> 2;   // 0..3
    int localX = int(laneId) & 3;    // 0..3
    int outY = tileY + localY;
    int outX = tileX + localX;
    bool validOut = (uint)outY < (uint)params.H && (uint)outX < (uint)params.W;

    // Shared storage for the transformed input V for current TILE_IC chunk:
    // V[icVecLocal][6][6] as float4 (4 input channels).
    static groupshared float4 s_V[ICVEC][TILE_I][TILE_I];
    static groupshared float4 s_T[ICVEC][TILE_I][TILE_I];
    // Shared weights for current TILE_IC chunk:
    // s_W[pos][icvLocal][ocvLocal][o] -> float4 of 4 IC scalars for one output channel.
    // This avoids reloading the same weights 16x (once per active lane) in the compute loop.
    static groupshared float4 s_W[36][ICVEC][TILE_OC / 4][4];

    // Per-lane accumulators in Winograd domain for 4 output channels (float4).
    // Note: this is register heavy but correct; we'll reduce it later.
    float4 M[36];
    [unroll] for (int i = 0; i < 36; ++i) { M[i] = float4(0.0); }

    // Main loop over input channels in TILE_IC chunks
    for (int icBase = 0; icBase < params.inChannels; icBase += TILE_IC)
    {
        // ------------------------------------------------------------
        // (1) Load input patch (6x6xTILE_IC) as float4 into s_V (raw)
        // ------------------------------------------------------------
        // Input patch origin includes padding=1
        int inY0 = tileY - 1;
        int inX0 = tileX - 1;

        for (uint i = tid; i < uint(ICVEC * 36); i += 128)
        {
            int icvLocal = int(i / 36);
            int pos = int(i % 36);
            int sy = pos / TILE_I;
            int sx = pos % TILE_I;

            int iy = inY0 + sy;
            int ix = inX0 + sx;

            // Global icVec (4-channel pack)
            int icVecGlobal = (icBase / 4) + icvLocal;

            float4 v = float4(0.0);
            if ((uint)iy < (uint)params.H && (uint)ix < (uint)params.W)
            {
                // Base scalar index for NHWC, then reinterpret as float4 pack
                int baseScalar = ((int(batchIdx) * (params.H * params.W) + iy * params.W + ix) * params.inChannels);
                float4* in4 = (float4*)params.input;
                v = in4[(baseScalar / 4) + icVecGlobal];
            }
            s_V[icvLocal][sy][sx] = v;
        }

        GroupMemoryBarrierWithGroupSync();

        // ------------------------------------------------------------
        // (2) Winograd input transform: V = BT * d * B
        //     We do col-transform then row-transform, operating on float4.
        // ------------------------------------------------------------
        // Column transform: for each icvLocal and column.
        if (tid < uint(ICVEC * TILE_I))
        {
            int icvLocal = int(tid / TILE_I);
            int col = int(tid % TILE_I);

            float4 d0 = s_V[icvLocal][0][col];
            float4 d1 = s_V[icvLocal][1][col];
            float4 d2 = s_V[icvLocal][2][col];
            float4 d3 = s_V[icvLocal][3][col];
            float4 d4 = s_V[icvLocal][4][col];
            float4 d5 = s_V[icvLocal][5][col];

            float4 t0, t1, t2, t3, t4, t5;
            winogradTransform6(d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5);

            s_T[icvLocal][0][col] = t0;
            s_T[icvLocal][1][col] = t1;
            s_T[icvLocal][2][col] = t2;
            s_T[icvLocal][3][col] = t3;
            s_T[icvLocal][4][col] = t4;
            s_T[icvLocal][5][col] = t5;
        }

        GroupMemoryBarrierWithGroupSync();

        // Row transform: for each icvLocal and row.
        if (tid < uint(ICVEC * TILE_I))
        {
            int icvLocal = int(tid / TILE_I);
            int row = int(tid % TILE_I);

            float4 r0 = s_T[icvLocal][row][0];
            float4 r1 = s_T[icvLocal][row][1];
            float4 r2 = s_T[icvLocal][row][2];
            float4 r3 = s_T[icvLocal][row][3];
            float4 r4 = s_T[icvLocal][row][4];
            float4 r5 = s_T[icvLocal][row][5];

            float4 v0, v1, v2, v3, v4, v5;
            winogradTransform6(r0, r1, r2, r3, r4, r5, v0, v1, v2, v3, v4, v5);

            s_V[icvLocal][row][0] = v0;
            s_V[icvLocal][row][1] = v1;
            s_V[icvLocal][row][2] = v2;
            s_V[icvLocal][row][3] = v3;
            s_V[icvLocal][row][4] = v4;
            s_V[icvLocal][row][5] = v5;
        }

        GroupMemoryBarrierWithGroupSync();

        // ------------------------------------------------------------
        // (3) Load weights for this TILE_IC chunk into shared memory (once per CTA)
        // ------------------------------------------------------------
        // Total float4 entries: 36 * ICVEC * (TILE_OC/4) * 4 = 1152 (fits easily).
        for (uint i = tid; i < uint(36 * ICVEC * (TILE_OC / 4) * 4); i += 128)
        {
            int tmp = int(i);
            int o = tmp & 3; tmp >>= 2;
            int ocvLocal = tmp & ((TILE_OC / 4) - 1); tmp >>= 2; // TILE_OC/4 == 4
            int icvLocal = tmp & (ICVEC - 1); tmp >>= 1;         // ICVEC == 2
            int pos = tmp;                                       // 0..35

            int ocVecGlobal = int(ocTileIdx) * (TILE_OC / 4) + ocvLocal;
            int icVecGlobal = (icBase / 4) + icvLocal;

            float4 w = float4(0.0);
            if (ocVecGlobal < ocVecCount && icVecGlobal < icVecCount)
            {
                int base = (((pos * ocVecCount + ocVecGlobal) * icVecCount + icVecGlobal) * 4);
                w = params.weightsU[base + o];
            }
            s_W[pos][icvLocal][ocvLocal][o] = w;
        }

        GroupMemoryBarrierWithGroupSync();

        // ------------------------------------------------------------
        // (4) Winograd-domain multiply-accumulate: M += V ⊙ U
        //     Each active lane accumulates float4 (4 OCs).
        // ------------------------------------------------------------
        if (activeLane)
        {
            // Global icVec base for this TILE_IC chunk
            int icVecBase = (icBase / 4);

            [unroll]
            for (int pos = 0; pos < 36; ++pos)
            {
                int py = pos / TILE_I;
                int px = pos % TILE_I;

                // Accumulate over icVec within this chunk (2)
                [unroll]
                for (int icvLocal = 0; icvLocal < ICVEC; ++icvLocal)
                {
                    int icVecGlobal = icVecBase + icvLocal;
                    float4 v = s_V[icvLocal][py][px];

                    // Read shared weights (ocvLocal == warpId).
                    float4 w0 = s_W[pos][icvLocal][int(warpId)][0];
                    float4 w1 = s_W[pos][icvLocal][int(warpId)][1];
                    float4 w2 = s_W[pos][icvLocal][int(warpId)][2];
                    float4 w3 = s_W[pos][icvLocal][int(warpId)][3];

                    M[pos] += dot4_to_float4(v, w0, w1, w2, w3);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // ------------------------------------------------------------
    // (5) Output transform: Y = AT * M * A (compute only this lane's element)
    // ------------------------------------------------------------
    if (activeLane && validOut)
    {
        const float4 c2 = float4(2.0);
        const float4 c4 = float4(4.0);
        const float4 c8 = float4(8.0);

        // First stage (column): produce 6 values for this lane's output row (localY)
        float4 tcol[6];
        [unroll]
        for (int col = 0; col < 6; ++col)
        {
            float4 m0 = M[0 * 6 + col];
            float4 m1 = M[1 * 6 + col];
            float4 m2 = M[2 * 6 + col];
            float4 m3 = M[3 * 6 + col];
            float4 m4 = M[4 * 6 + col];
            float4 m5 = M[5 * 6 + col];

            if (localY == 0)      tcol[col] = m0 + m1 + m2 + m3 + m4;
            else if (localY == 1) tcol[col] = m1 - m2 + m3 * c2 - m4 * c2;
            else if (localY == 2) tcol[col] = m1 + m2 + m3 * c4 + m4 * c4;
            else                  tcol[col] = m1 - m2 + m3 * c8 - m4 * c8 + m5;
        }

        // Second stage (row): compute only this lane's output column (localX)
        float4 y;
        float4 t0 = tcol[0];
        float4 t1 = tcol[1];
        float4 t2 = tcol[2];
        float4 t3 = tcol[3];
        float4 t4 = tcol[4];
        float4 t5 = tcol[5];

        if (localX == 0)      y = t0 + t1 + t2 + t3 + t4;
        else if (localX == 1) y = t1 - t2 + t3 * c2 - t4 * c2;
        else if (localX == 2) y = t1 + t2 + t3 * c4 + t4 * c4;
        else                  y = t1 - t2 + t3 * c8 - t4 * c8 + t5;

        // Add bias (float4)
        float4* bias4 = (float4*)params.bias;
        y += bias4[ocVec];

        // Store float4 output
        float4* out4 = (float4*)params.output;
        int outVecStride = ocVecCount;
        int outIndex = ((int(batchIdx) * (params.H * params.W) + outY * params.W + outX) * outVecStride) + ocVec;
        // Aligned 16B store of float4 to global output.
        storeAligned<16>(out4 + outIndex, y);
    }
}

// ---------------------------------------------------------------------------
// v1: cuDNN-inspired CTA shape (single kernel, no separate transforms)
// ---------------------------------------------------------------------------
// cuDNN nonfused uses tilesPerCTA=4 for F(4x4,3x3), i.e. a CTA covers 4x4 tiles
// -> 16x16 output region, which requires an 18x18 input patch for pad=1.
//
// This kernel computes one ocVec (4 output channels) per CTA.
// Grid:
//   x = ceil(W / 16)
//   y = ceil(H / 16)
//   z = batchSize * (outChannels/4)
//
// Threads:
//   16x16 = 256 threads (maps naturally to output pixels in the 16x16 block)
[numthreads(16, 16, 1)]
void winogradConvProtoBlock16(
    ConstantBuffer<WinogradConvProtoParams, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    static const int TILE_O = 4;          // output tile is 4x4
    static const int TILE_I = 6;          // input tile is 6x6
    static const int TILES_PER_CTA = 4;   // 4x4 tiles per CTA -> 16x16 output
    static const int BLOCK_O = TILE_O * TILES_PER_CTA; // 16
    static const int BLOCK_I = BLOCK_O + 2; // 18 (pad=1)

    static const int TILE_IC = 8;
    static const int ICVEC = TILE_IC / 4; // 2
    static const int OCV_PER_CTA = 2;     // number of ocVec (float4 output channel groups) per CTA

    const int ocVecCount = params.outChannels / 4;
    const int icVecCount = params.inChannels / 4;

    // Decode groupId.z -> (batch, ocTile) where ocTile spans OCV_PER_CTA ocVecs
    const int ocTileCount = (ocVecCount + OCV_PER_CTA - 1) / OCV_PER_CTA;
    uint batchIdx = groupId.z / uint(ocTileCount);
    int ocTile = int(groupId.z % uint(ocTileCount));
    if (batchIdx >= uint(params.batchSize)) return;

    // Block origin in output
    int baseY = int(groupId.y) * BLOCK_O;
    int baseX = int(groupId.x) * BLOCK_O;

    // Thread -> output pixel within the 16x16 block
    int localY = int(groupThreadId.y); // 0..15
    int localX = int(groupThreadId.x); // 0..15
    int outY = baseY + localY;
    int outX = baseX + localX;
    bool validOut = (uint)outY < (uint)params.H && (uint)outX < (uint)params.W;

    // Mapping to tile/local-in-tile for output transform
    int tileY = localY >> 2;  // 0..3
    int tileX = localX >> 2;  // 0..3
    int tileIdx = tileY * TILES_PER_CTA + tileX; // 0..15
    int inTileY = localY & 3; // 0..3
    int inTileX = localX & 3; // 0..3

    // Thread linear id
    int tid = localY * 16 + localX; // 0..255

    // Per-thread tile mapping (fixed for this thread)
    // tileGroup: which 4x4 Winograd tile within the 16x16 block (0..15)
    // laneInTile: which output pixel within that 4x4 tile (0..15)
    int tileGroup = tileIdx;
    int laneInTile = inTileY * 4 + inTileX;

    // Origin of this tile in the (padded) input patch
    int tileInY = tileY * TILE_O;
    int tileInX = tileX * TILE_O;

    // Each lane owns 2 positions, plus 1 extra for lanes 0..3 (to cover 36 positions)
    int p0 = laneInTile * 2;
    int p1 = p0 + 1;
    int p2 = (laneInTile < 4) ? (32 + laneInTile) : -1;
    int posList[3] = {p0, p1, p2};

    // Shared input patch for current TILE_IC chunk (float4 over IC)
    static groupshared float4 s_in[ICVEC][BLOCK_I][BLOCK_I];
    // s_V[tile][icvLocal][row][col] = full transform (BT * d * B)
    static groupshared float4 s_V[16][ICVEC][TILE_I][TILE_I];
    // Shared weights for current TILE_IC chunk (pos-major, coalesced global reads):
    // s_W[pos][icvLocal][ocvLocal][o] where:
    //   ocvLocal in [0..OCV_PER_CTA) selects which ocVec within this CTA
    //   o selects which output channel within that ocVec (0..3)
    static groupshared float4 s_W[36][ICVEC][OCV_PER_CTA][4];
    // Shared Winograd-domain accumulators per tile:
    // s_M[tile][pos][ocvLocal] is float4 (4 output channels)
    static groupshared float4 s_M[16][36][OCV_PER_CTA];

    // Register accumulators (per-thread 2-3 Winograd positions, across all IC chunks)
    float4 accReg[OCV_PER_CTA][3];
    [unroll]
    for (int ocvLocal = 0; ocvLocal < OCV_PER_CTA; ++ocvLocal)
    {
        accReg[ocvLocal][0] = float4(0.0);
        accReg[ocvLocal][1] = float4(0.0);
        accReg[ocvLocal][2] = float4(0.0);
    }

    // Main loop over input channels in chunks of 8
    for (int icBase = 0; icBase < params.inChannels; icBase += TILE_IC)
    {
        // Synchronize here so the next iteration can safely overwrite s_in/s_V/s_W.
        if (icBase != 0)
        {
            GroupMemoryBarrierWithGroupSync();
        }

        // ------------------------------------------------------------
        // (1) Load 18x18 input patch for this block and TILE_IC chunk
        // ------------------------------------------------------------
        int inY0 = baseY - 1; // pad=1
        int inX0 = baseX - 1;

        for (uint i = uint(tid); i < uint(ICVEC * BLOCK_I * BLOCK_I); i += 256)
        {
            int icvLocal = int(i / (BLOCK_I * BLOCK_I));
            int pos = int(i % (BLOCK_I * BLOCK_I));
            int sy = pos / BLOCK_I;
            int sx = pos % BLOCK_I;

            int iy = inY0 + sy;
            int ix = inX0 + sx;
            int icVecGlobal = (icBase / 4) + icvLocal;

            float4 v = float4(0.0);
            if ((uint)iy < (uint)params.H && (uint)ix < (uint)params.W)
            {
                int baseScalar = ((int(batchIdx) * (params.H * params.W) + iy * params.W + ix) * params.inChannels);
                float4* in4 = (float4*)params.input;
                v = in4[(baseScalar / 4) + icVecGlobal];
            }
            s_in[icvLocal][sy][sx] = v;
        }

        // ------------------------------------------------------------
        // (2) Load weights for this TILE_IC chunk (pos-major)
        // ------------------------------------------------------------
        for (uint i = uint(tid); i < uint(36 * ICVEC * OCV_PER_CTA * 4); i += 256)
        {
            int tmp = int(i);
            int o = tmp & 3; tmp >>= 2;          // 0..3
            int ocvLocal = tmp % OCV_PER_CTA; tmp /= OCV_PER_CTA;
            int icvLocal = tmp & 1; tmp >>= 1;   // 0..1
            int pos = tmp;                       // 0..35

            int icVecGlobal = (icBase / 4) + icvLocal;
            int ocVec = ocTile * OCV_PER_CTA + ocvLocal;

            float4 w = float4(0.0);
            if (icVecGlobal < icVecCount && ocVec < ocVecCount)
            {
                int base = (((pos * ocVecCount + ocVec) * icVecCount + icVecGlobal) * 4);
                w = params.weightsU[base + o];
            }
            s_W[pos][icvLocal][ocvLocal][o] = w;
        }

        GroupMemoryBarrierWithGroupSync();

        // ------------------------------------------------------------
        // (3) Per-tile Winograd input transform (compute full 6x6 once, cuDNN-style)
        // ------------------------------------------------------------
        // Row-first transform into s_V (temporary), then column transform overwriting s_V.
        // This avoids a separate `s_Tcol` buffer.
        if (laneInTile < TILE_I)
        {
            int row = laneInTile; // 0..5
            [unroll]
            for (int icvLocal = 0; icvLocal < ICVEC; ++icvLocal)
            {
                float4 d0 = s_in[icvLocal][tileInY + row][tileInX + 0];
                float4 d1 = s_in[icvLocal][tileInY + row][tileInX + 1];
                float4 d2 = s_in[icvLocal][tileInY + row][tileInX + 2];
                float4 d3 = s_in[icvLocal][tileInY + row][tileInX + 3];
                float4 d4 = s_in[icvLocal][tileInY + row][tileInX + 4];
                float4 d5 = s_in[icvLocal][tileInY + row][tileInX + 5];

                float4 t0, t1, t2, t3, t4, t5;
                winogradTransform6(d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5);
                s_V[tileGroup][icvLocal][row][0] = t0;
                s_V[tileGroup][icvLocal][row][1] = t1;
                s_V[tileGroup][icvLocal][row][2] = t2;
                s_V[tileGroup][icvLocal][row][3] = t3;
                s_V[tileGroup][icvLocal][row][4] = t4;
                s_V[tileGroup][icvLocal][row][5] = t5;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Column transform overwriting s_V.
        if (laneInTile < TILE_I)
        {
            int col = laneInTile; // 0..5
            [unroll]
            for (int icvLocal = 0; icvLocal < ICVEC; ++icvLocal)
            {
                float4 r0 = s_V[tileGroup][icvLocal][0][col];
                float4 r1 = s_V[tileGroup][icvLocal][1][col];
                float4 r2 = s_V[tileGroup][icvLocal][2][col];
                float4 r3 = s_V[tileGroup][icvLocal][3][col];
                float4 r4 = s_V[tileGroup][icvLocal][4][col];
                float4 r5 = s_V[tileGroup][icvLocal][5][col];

                float4 v0, v1, v2, v3, v4, v5;
                winogradTransform6(r0, r1, r2, r3, r4, r5, v0, v1, v2, v3, v4, v5);
                s_V[tileGroup][icvLocal][0][col] = v0;
                s_V[tileGroup][icvLocal][1][col] = v1;
                s_V[tileGroup][icvLocal][2][col] = v2;
                s_V[tileGroup][icvLocal][3][col] = v3;
                s_V[tileGroup][icvLocal][4][col] = v4;
                s_V[tileGroup][icvLocal][5][col] = v5;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // ------------------------------------------------------------
        // (4) Winograd-domain implicit GEMM accumulate into registers
        // ------------------------------------------------------------
        [unroll]
        for (int pi = 0; pi < 3; ++pi)
        {
            int pos = posList[pi];
            if (pos < 0 || pos >= 36) continue;

            int py = pos / 6;
            int px = pos % 6;

            float4 v0 = s_V[tileGroup][0][py][px];
            float4 v1 = s_V[tileGroup][1][py][px];

            [unroll]
            for (int ocvLocal = 0; ocvLocal < OCV_PER_CTA; ++ocvLocal)
            {
                float4 acc = float4(0.0);

                // icvLocal = 0
                float4 w00 = s_W[pos][0][ocvLocal][0];
                float4 w01 = s_W[pos][0][ocvLocal][1];
                float4 w02 = s_W[pos][0][ocvLocal][2];
                float4 w03 = s_W[pos][0][ocvLocal][3];
                acc += dot4_to_float4(v0, w00, w01, w02, w03);

                // icvLocal = 1
                float4 w10 = s_W[pos][1][ocvLocal][0];
                float4 w11 = s_W[pos][1][ocvLocal][1];
                float4 w12 = s_W[pos][1][ocvLocal][2];
                float4 w13 = s_W[pos][1][ocvLocal][3];
                acc += dot4_to_float4(v1, w10, w11, w12, w13);

                accReg[ocvLocal][pi] += acc;
            }
        }
    }

    // ------------------------------------------------------------
    // Final sync: make sure all threads are done with the IC loop before writing/reading s_M.
    // ------------------------------------------------------------
    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------
    // (5) Write final M to shared once (after all IC chunks)
    // ------------------------------------------------------------
    [unroll]
    for (int pi = 0; pi < 3; ++pi)
    {
        int pos = posList[pi];
        if (pos < 0 || pos >= 36) continue;
        [unroll]
        for (int ocvLocal = 0; ocvLocal < OCV_PER_CTA; ++ocvLocal)
        {
            s_M[tileGroup][pos][ocvLocal] = accReg[ocvLocal][pi];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // ------------------------------------------------------------
    // (6) Output transform + store (tile-wise)
    // ------------------------------------------------------------
    // We overwrite `s_M` with tmp = AT * M (size 4x6) because M is no longer needed after this.
    const float4 c2 = float4(2.0);
    const float4 c4 = float4(4.0);
    const float4 c8 = float4(8.0);

    float4* out4 = (float4*)params.output;
    float4* bias4 = (float4*)params.bias;
    int outVecStride = ocVecCount;

    // Stage 1 (AT*M): lanes 0..5 compute one column each, for all ocvLocal.
    if (laneInTile < TILE_I)
    {
        int col = laneInTile;
        [unroll]
        for (int ocvLocal = 0; ocvLocal < OCV_PER_CTA; ++ocvLocal)
        {
            int ocVec = ocTile * OCV_PER_CTA + ocvLocal;
            if (ocVec >= ocVecCount) continue;

            float4 m0 = s_M[tileGroup][0 * 6 + col][ocvLocal];
            float4 m1 = s_M[tileGroup][1 * 6 + col][ocvLocal];
            float4 m2 = s_M[tileGroup][2 * 6 + col][ocvLocal];
            float4 m3 = s_M[tileGroup][3 * 6 + col][ocvLocal];
            float4 m4 = s_M[tileGroup][4 * 6 + col][ocvLocal];
            float4 m5 = s_M[tileGroup][5 * 6 + col][ocvLocal];

            // Store tmp rows into s_M[0..23]
            s_M[tileGroup][0 * 6 + col][ocvLocal] = m0 + m1 + m2 + m3 + m4;
            s_M[tileGroup][1 * 6 + col][ocvLocal] = m1 - m2 + m3 * c2 - m4 * c2;
            s_M[tileGroup][2 * 6 + col][ocvLocal] = m1 + m2 + m3 * c4 + m4 * c4;
            s_M[tileGroup][3 * 6 + col][ocvLocal] = m1 - m2 + m3 * c8 - m4 * c8 + m5;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Stage 2 ((AT*M)*A): each thread writes its output pixel (if valid).
    if (validOut)
    {
        int outBase = ((int(batchIdx) * (params.H * params.W) + outY * params.W + outX) * outVecStride);
        [unroll]
        for (int ocvLocal = 0; ocvLocal < OCV_PER_CTA; ++ocvLocal)
        {
            int ocVec = ocTile * OCV_PER_CTA + ocvLocal;
            if (ocVec >= ocVecCount) continue;

            float4 t0 = s_M[tileGroup][inTileY * 6 + 0][ocvLocal];
            float4 t1 = s_M[tileGroup][inTileY * 6 + 1][ocvLocal];
            float4 t2 = s_M[tileGroup][inTileY * 6 + 2][ocvLocal];
            float4 t3 = s_M[tileGroup][inTileY * 6 + 3][ocvLocal];
            float4 t4 = s_M[tileGroup][inTileY * 6 + 4][ocvLocal];
            float4 t5 = s_M[tileGroup][inTileY * 6 + 5][ocvLocal];

            float4 y;
            if (inTileX == 0)      y = t0 + t1 + t2 + t3 + t4;
            else if (inTileX == 1) y = t1 - t2 + t3 * c2 - t4 * c2;
            else if (inTileX == 2) y = t1 + t2 + t3 * c4 + t4 * c4;
            else                   y = t1 - t2 + t3 * c8 - t4 * c8 + t5;

            y += bias4[ocVec];
            out4[outBase + ocVec] = y;
        }
    }
}

// ============================================================================
// Winograd nonfused prototype (cuDNN-style pipeline: A/B/C buffers + GEMM)
// ============================================================================
// Layouts (matches cuDNN nonfused intent, but packed in float4):
// - V (Winograd A/data):  [tile][pos][icVec] float4   (icVec = C/4)
// - M (Winograd C/output):[tile][pos][ocVec] float4   (ocVec = K/4)
// - weightsU (Winograd B/filter): packed as in transformWeightsF43Packed:
//     weightsU[pos][ocVec][icVec][o] where each entry is float4 of 4 input-channel weights,
//     and `o` selects which output lane within ocVec (0..3).

struct WinogradNonFusedParams
{
    float* input;          // NHWC float
    float* output;         // NHWC float
    float* bias;           // float (outChannels)
    float4* weightsU;      // packed Winograd weights (see above)
    float4* V;             // [numTiles][36][icVecCount]
    float4* M;             // [numTiles][36][ocVecCount]

    int H;
    int W;
    int padding;
    int batchSize;
    int inChannels;
    int outChannels;

    int numTilesH;         // ceil(H/4)
    int numTilesW;         // ceil(W/4)
    int numTiles;          // batchSize * numTilesH * numTilesW
};

// ---------------------------------------------------------------------------
// Kernel 1: Input transform (produce V)
// Grid:
//   x = ceil(numTilesW / 4)   (each CTA covers 4 tiles in X => 16 output pixels)
//   y = ceil(numTilesH / 4)
//   z = batchSize * icVecCount
// Threads:
//   256 = 16 tiles * 16 lanes (reuse the same transform pattern as fused kernel)
// Notes:
// - We compute V for up to 4x4 tiles per CTA (tilesPerCTA=4 like cuDNN).
// - Each CTA handles exactly one icVec (float4 input channels).
// ---------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void winogradInputTransformBlock16(
    ConstantBuffer<WinogradNonFusedParams, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    static const int TILE_O = 4;
    static const int TILE_I = 6;
    static const int TILES_PER_CTA = 4;
    static const int BLOCK_O = TILE_O * TILES_PER_CTA; // 16
    static const int BLOCK_I = BLOCK_O + 2;            // 18 (pad=1)

    int icVecCount = params.inChannels / 4;
    int tid = int(groupThreadId.x); // 0..255

    int blockX = int(groupId.x);
    int blockY = int(groupId.y);

    int z = int(groupId.z);
    int batch = z / icVecCount;
    int icVec = z - batch * icVecCount;
    if ((uint)batch >= (uint)params.batchSize) return;

    // Tile block origin in tile-space
    int tileBlockX = blockX * TILES_PER_CTA;
    int tileBlockY = blockY * TILES_PER_CTA;

    // Output block origin in pixel-space
    int baseY = tileBlockY * TILE_O;
    int baseX = tileBlockX * TILE_O;

    // Per-thread tile mapping (within this CTA)
    int tileGroup = tid >> 4;  // 0..15
    int lane = tid & 15;       // 0..15
    int tileInBlockY = tileGroup >> 2; // 0..3
    int tileInBlockX = tileGroup & 3;  // 0..3

    int tileY = tileBlockY + tileInBlockY;
    int tileX = tileBlockX + tileInBlockX;

    bool validTile = (uint)tileY < (uint)params.numTilesH && (uint)tileX < (uint)params.numTilesW;
    int tileIdx = (batch * params.numTilesH + tileY) * params.numTilesW + tileX;

    // Shared staging for input patch (18x18) for this icVec.
    static groupshared float4 s_in[BLOCK_I][BLOCK_I];
    // Shared transformed output for this CTA: 16 tiles * 6x6
    static groupshared float4 s_V[16][TILE_I][TILE_I];

    int inY0 = baseY - params.padding;
    int inX0 = baseX - params.padding;

    // Cooperative load of 18x18 patch for this icVec.
    for (uint i = uint(tid); i < uint(BLOCK_I * BLOCK_I); i += 256)
    {
        int pos = int(i);
        int sy = pos / BLOCK_I;
        int sx = pos % BLOCK_I;
        int iy = inY0 + sy;
        int ix = inX0 + sx;

        float4 v = float4(0.0);
        if ((uint)iy < (uint)params.H && (uint)ix < (uint)params.W)
        {
            int baseScalar = ((batch * (params.H * params.W) + iy * params.W + ix) * params.inChannels);
            float4* in4 = (float4*)params.input;
            // Tell Slang this is 16B-aligned so it can generate a real 128-bit vector load.
            v = loadAligned<16>(in4 + (baseScalar / 4) + icVec);
        }
        s_in[sy][sx] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute tile-local input transform:
    // Row transform into s_V (temporary), then column transform overwriting s_V.
    int tileInY = tileInBlockY * TILE_O;
    int tileInX = tileInBlockX * TILE_O;

    // Row transform: use lanes 0..5 as "row"
    if (lane < TILE_I)
    {
        int row = lane;
        float4 d0 = s_in[tileInY + row][tileInX + 0];
        float4 d1 = s_in[tileInY + row][tileInX + 1];
        float4 d2 = s_in[tileInY + row][tileInX + 2];
        float4 d3 = s_in[tileInY + row][tileInX + 3];
        float4 d4 = s_in[tileInY + row][tileInX + 4];
        float4 d5 = s_in[tileInY + row][tileInX + 5];

        float4 t0, t1, t2, t3, t4, t5;
        winogradTransform6(d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5);
        s_V[tileGroup][row][0] = t0;
        s_V[tileGroup][row][1] = t1;
        s_V[tileGroup][row][2] = t2;
        s_V[tileGroup][row][3] = t3;
        s_V[tileGroup][row][4] = t4;
        s_V[tileGroup][row][5] = t5;
    }

    GroupMemoryBarrierWithGroupSync();

    // Column transform: use lanes 0..5 as "col"
    if (lane < TILE_I)
    {
        int col = lane;
        float4 r0 = s_V[tileGroup][0][col];
        float4 r1 = s_V[tileGroup][1][col];
        float4 r2 = s_V[tileGroup][2][col];
        float4 r3 = s_V[tileGroup][3][col];
        float4 r4 = s_V[tileGroup][4][col];
        float4 r5 = s_V[tileGroup][5][col];

        float4 v0, v1, v2, v3, v4, v5;
        winogradTransform6(r0, r1, r2, r3, r4, r5, v0, v1, v2, v3, v4, v5);
        s_V[tileGroup][0][col] = v0;
        s_V[tileGroup][1][col] = v1;
        s_V[tileGroup][2][col] = v2;
        s_V[tileGroup][3][col] = v3;
        s_V[tileGroup][4][col] = v4;
        s_V[tileGroup][5][col] = v5;
    }

    GroupMemoryBarrierWithGroupSync();

    // Store 2-3 positions per lane to global V.
    // Layout: V[(pos * icVecCount + icVec) * numTiles + tileIdx]
    if (!validTile) return;

    int p0 = lane * 2;
    int p1 = p0 + 1;
    int p2 = (lane < 4) ? (32 + lane) : -1;
    int posList[3] = {p0, p1, p2};

    [unroll]
    for (int pi = 0; pi < 3; ++pi)
    {
        int pos = posList[pi];
        if (pos < 0 || pos >= 36) continue;
        int py = pos / 6;
        int px = pos % 6;

        int outIndex = (pos * icVecCount + icVec) * params.numTiles + tileIdx;
        // Aligned 16B store of float4 to global memory (see Slang storeAligned docs).
        storeAligned<16>(params.V + outIndex, s_V[tileGroup][py][px]);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Winograd-domain GEMM (produce M = U (*) V)
// Grid:
//   x = ceil(numTiles / TILE_TILES)   (TILE_TILES = 32)
//   y = ceil(ocVecCount / TILE_OCV)   (TILE_OCV = 8)
//   z = 36 (pos)
// Threads:
//   256 = TILE_TILES * TILE_OCV
// Notes:
// - We stage V for a small icVec block into shared to reduce redundant loads across ocVec.
// ---------------------------------------------------------------------------
[numthreads(256, 1, 1)]
void winogradDomainGemmF43(
    ConstantBuffer<WinogradNonFusedParams, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    static const int TILE_TILES = 32; // compute 32 tiles per block
    static const int TILE_OCV = 16; // compute 16 ocVec per block (each thread computes 2 ocVec)
    static const int IC_STEP = 16; // number of icVec values staged per iteration (trade shared for fewer iterations/barriers)

    int ocVecCount = params.outChannels / 4;
    int icVecCount = params.inChannels / 4;

    int tid = int(groupThreadId.x); // 0..255
    int laneTile = tid & 31;               // 0..31 (tile lane within the block)
    int laneOCV = tid >> 5;                // 0..7 (each lane handles ocVec0 and ocVec1=ocVec0+8)

    int tileIdx0 = int(groupId.x) * TILE_TILES + laneTile;
    int ocVec0 = int(groupId.y) * TILE_OCV + laneOCV;
    int ocVec1 = ocVec0 + 8;
    int pos = int(groupId.z);

    if (pos >= 36) return;
    if ((uint)ocVec0 >= (uint)ocVecCount) return;

    // Shared staged V for IC_STEP consecutive icVec values for this tile-block.
    // Only laneOCV==0 loads V; other oc lanes reuse it (cuts V loads by 8x).
    // +4 padding to reduce shared-memory bank conflict patterns (cuDNN commonly pads shared strides).
    static groupshared float4 s_Vblk0[IC_STEP][32 + 4];
    // Shared staged weights for IC_STEP consecutive icVec values for this oc-block.
    // Only laneTile==0 loads weights; all tile lanes reuse (cuts W loads by 32x).
    static groupshared float4 s_Wblk[IC_STEP][TILE_OCV][4];

    float4 acc00 = float4(0.0);
    float4 acc01 = float4(0.0);

    for (int icBase = 0; icBase < icVecCount; icBase += IC_STEP)
    {
        // Load V for this tile-block (only laneOCV==0)
        if (laneOCV == 0)
        {
            for (int s = 0; s < IC_STEP; ++s)
            {
                int icVec = icBase + s;
                float4 v0 = float4(0.0);
                if (icVec < icVecCount)
                {
                    // V layout: [(pos * icVecCount + icVec) * numTiles + tileIdx]
                    if ((uint)tileIdx0 < (uint)params.numTiles)
                    {
                        int vIndex0 = (pos * icVecCount + icVec) * params.numTiles + tileIdx0;
                        v0 = loadAligned<16>(params.V + vIndex0);
                    }
                }
                s_Vblk0[s][laneTile] = v0;
            }
        }

        // Load weights for this oc-block (only laneTile==0)
        if (laneTile == 0)
        {
            for (int s = 0; s < IC_STEP; ++s)
            {
                int icVec = icBase + s;
                // ocVec0
                if (icVec < icVecCount && ocVec0 < ocVecCount)
                {
                    int wBase0 = (((pos * ocVecCount + ocVec0) * icVecCount + icVec) * 4);
                    s_Wblk[s][laneOCV][0] = loadAligned<16>(params.weightsU + wBase0 + 0);
                    s_Wblk[s][laneOCV][1] = loadAligned<16>(params.weightsU + wBase0 + 1);
                    s_Wblk[s][laneOCV][2] = loadAligned<16>(params.weightsU + wBase0 + 2);
                    s_Wblk[s][laneOCV][3] = loadAligned<16>(params.weightsU + wBase0 + 3);
                }
                else
                {
                    s_Wblk[s][laneOCV][0] = float4(0.0);
                    s_Wblk[s][laneOCV][1] = float4(0.0);
                    s_Wblk[s][laneOCV][2] = float4(0.0);
                    s_Wblk[s][laneOCV][3] = float4(0.0);
                }

                // ocVec1 (second half of the K tile)
                if (icVec < icVecCount && ocVec1 < ocVecCount)
                {
                    int wBase1 = (((pos * ocVecCount + ocVec1) * icVecCount + icVec) * 4);
                    s_Wblk[s][laneOCV + 8][0] = loadAligned<16>(params.weightsU + wBase1 + 0);
                    s_Wblk[s][laneOCV + 8][1] = loadAligned<16>(params.weightsU + wBase1 + 1);
                    s_Wblk[s][laneOCV + 8][2] = loadAligned<16>(params.weightsU + wBase1 + 2);
                    s_Wblk[s][laneOCV + 8][3] = loadAligned<16>(params.weightsU + wBase1 + 3);
                }
                else
                {
                    s_Wblk[s][laneOCV + 8][0] = float4(0.0);
                    s_Wblk[s][laneOCV + 8][1] = float4(0.0);
                    s_Wblk[s][laneOCV + 8][2] = float4(0.0);
                    s_Wblk[s][laneOCV + 8][3] = float4(0.0);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Compute for IC_STEP values.
        for (int s = 0; s < IC_STEP; ++s)
        {
            int icVec = icBase + s;
            if (icVec >= icVecCount) continue;

            float4 v0 = s_Vblk0[s][laneTile];
            // ocVec0
            float4 w00 = s_Wblk[s][laneOCV][0];
            float4 w01 = s_Wblk[s][laneOCV][1];
            float4 w02 = s_Wblk[s][laneOCV][2];
            float4 w03 = s_Wblk[s][laneOCV][3];
            acc00 += dot4_to_float4(v0, w00, w01, w02, w03);

            // ocVec1
            if (ocVec1 < ocVecCount)
            {
                float4 w10 = s_Wblk[s][laneOCV + 8][0];
                float4 w11 = s_Wblk[s][laneOCV + 8][1];
                float4 w12 = s_Wblk[s][laneOCV + 8][2];
                float4 w13 = s_Wblk[s][laneOCV + 8][3];
                acc01 += dot4_to_float4(v0, w10, w11, w12, w13);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // M layout: [(pos * ocVecCount + ocVec) * numTiles + tileIdx]
    if ((uint)tileIdx0 < (uint)params.numTiles)
    {
        int outIndex00 = (pos * ocVecCount + ocVec0) * params.numTiles + tileIdx0;
        storeAligned<16>(params.M + outIndex00, acc00);
        if (ocVec1 < ocVecCount)
        {
            int outIndex01 = (pos * ocVecCount + ocVec1) * params.numTiles + tileIdx0;
            storeAligned<16>(params.M + outIndex01, acc01);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Output transform (consume M, write output)
// Grid:
//   x = ceil(numTilesW / 4)  (one CTA per 16x16 output block)
//   y = ceil(numTilesH / 4)
//   z = batchSize * ocVecCount
// Threads:
//   16x16 = 256 threads (one per output pixel in the 16x16 block)
// ---------------------------------------------------------------------------
[numthreads(16, 16, 1)]
void winogradOutputTransformBlock16(
    ConstantBuffer<WinogradNonFusedParams, CDataLayout> params,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID)
{
    static const int TILE_O = 4;
    static const int TILE_I = 6;
    static const int TILES_PER_CTA = 4;
    static const int BLOCK_O = TILE_O * TILES_PER_CTA; // 16

    int ocVecCount = params.outChannels / 4;

    int blockX = int(groupId.x);
    int blockY = int(groupId.y);
    int z = int(groupId.z);
    int batch = z / ocVecCount;
    int ocVec = z - batch * ocVecCount;
    if ((uint)batch >= (uint)params.batchSize) return;

    int localY = int(groupThreadId.y); // 0..15
    int localX = int(groupThreadId.x); // 0..15
    int outY = blockY * BLOCK_O + localY;
    int outX = blockX * BLOCK_O + localX;
    bool validOut = (uint)outY < (uint)params.H && (uint)outX < (uint)params.W;

    int tileY = localY >> 2;
    int tileX = localX >> 2;
    int tileIdxInBlock = tileY * TILES_PER_CTA + tileX; // 0..15
    int inTileY = localY & 3;
    int inTileX = localX & 3;

    // Tile coordinates in global tile grid
    int tileGlobalY = blockY * TILES_PER_CTA + tileY;
    int tileGlobalX = blockX * TILES_PER_CTA + tileX;
    if ((uint)tileGlobalY >= (uint)params.numTilesH || (uint)tileGlobalX >= (uint)params.numTilesW) return;
    int tileIdx = (batch * params.numTilesH + tileGlobalY) * params.numTilesW + tileGlobalX;

    // Shared tmp = AT * M (4x6) for this tile, for this ocVec.
    static groupshared float4 s_tmp[16][4][6];

    // Lane within tile: 0..15
    int lane = inTileY * 4 + inTileX;

    const float4 c2 = float4(2.0);
    const float4 c4 = float4(4.0);
    const float4 c8 = float4(8.0);

    // Stage 1 (AT*M): use lanes 0..5 as column id.
    if (lane < TILE_I)
    {
        int col = lane;
        int stridePos = ocVecCount * params.numTiles;
        int base = ocVec * params.numTiles + tileIdx;

        // Load M with known 16B alignment (float4)
        float4 m0 = loadAligned<16>(params.M + ((0 * 6 + col) * stridePos + base));
        float4 m1 = loadAligned<16>(params.M + ((1 * 6 + col) * stridePos + base));
        float4 m2 = loadAligned<16>(params.M + ((2 * 6 + col) * stridePos + base));
        float4 m3 = loadAligned<16>(params.M + ((3 * 6 + col) * stridePos + base));
        float4 m4 = loadAligned<16>(params.M + ((4 * 6 + col) * stridePos + base));
        float4 m5 = loadAligned<16>(params.M + ((5 * 6 + col) * stridePos + base));

        s_tmp[tileIdxInBlock][0][col] = m0 + m1 + m2 + m3 + m4;
        s_tmp[tileIdxInBlock][1][col] = m1 - m2 + m3 * c2 - m4 * c2;
        s_tmp[tileIdxInBlock][2][col] = m1 + m2 + m3 * c4 + m4 * c4;
        s_tmp[tileIdxInBlock][3][col] = m1 - m2 + m3 * c8 - m4 * c8 + m5;
    }

    GroupMemoryBarrierWithGroupSync();

    if (validOut)
    {
        float4 t0 = s_tmp[tileIdxInBlock][inTileY][0];
        float4 t1 = s_tmp[tileIdxInBlock][inTileY][1];
        float4 t2 = s_tmp[tileIdxInBlock][inTileY][2];
        float4 t3 = s_tmp[tileIdxInBlock][inTileY][3];
        float4 t4 = s_tmp[tileIdxInBlock][inTileY][4];
        float4 t5 = s_tmp[tileIdxInBlock][inTileY][5];

        float4 y;
        if (inTileX == 0)      y = t0 + t1 + t2 + t3 + t4;
        else if (inTileX == 1) y = t1 - t2 + t3 * c2 - t4 * c2;
        else if (inTileX == 2) y = t1 + t2 + t3 * c4 + t4 * c4;
        else                   y = t1 - t2 + t3 * c8 - t4 * c8 + t5;

        // Bias
        float4* bias4 = (float4*)params.bias;
        y += loadAligned<16>(bias4 + ocVec);

        // Store float4 output
        float4* out4 = (float4*)params.output;
        int outBase = ((batch * (params.H * params.W) + outY * params.W + outX) * ocVecCount);
        // Aligned 16B store of float4 to global output.
        storeAligned<16>(out4 + outBase + ocVec, y);
    }
}


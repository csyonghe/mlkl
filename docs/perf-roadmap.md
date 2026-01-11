### Performance Roadmap: MLKL Stable Diffusion Inference

This document captures a staged roadmap to turn the current convolution prototypes (Winograd + implicit-GEMM) into production-quality kernels integrated with the engine’s `IExpr` / `ISink` system, build an auto-selection strategy, and then extend the same optimization approach to the next likely bottlenecks (linear + attention). Once FP32 is solid, the roadmap pivots to FP16 + Tensor Cores via Slang `CoopMatrix`.

---

### Goals and Success Criteria

- **Primary goal**: close the remaining performance gap vs PyTorch for Stable Diffusion inference while keeping correctness, maintainability, and debuggability.
- **Short-term success**: production-ready conv path that keeps the current prototype performance characteristics (aligned vector loads/stores, low overhead indexing, predictable scheduling).
- **Mid-term success**: auto algorithm selection + auto tile selection that is stable across SD layers and devices.
- **Long-term success**: FP16 Tensor Core path that significantly outperforms FP32 and becomes the default on supported GPUs.

---

### Phase 0 — Baselines, constraints, and instrumentation hardening

**Why**: all subsequent work needs stable measurement and guardrails.

- **Lock down benchmark configurations**
  - Stable SD conv layer shapes we care about (high-C and low/mid-C).
  - Microbenchmarks (`--profile-conv`, `--bench-conv2d`) kept reproducible (fixed seeds, warmup rules, pinned clocks if possible).
- **Expand perf instrumentation**
  - Keep/extend GPU timestamp breakdown support (per-kernel).
  - Add optional counters/derived metrics (GB/s, GFLOP/s, effective occupancy proxy where available).
- **Correctness gates**
  - Deterministic comparison mode for conv outputs (max abs, max rel, acceptable eps).
  - Spot-check across padding/border cases and non-multiple-of-4 channels (fallback path).

**Exit criteria**
- Benchmarks are stable enough to detect ~1–2% regressions reliably.
- Timestamp breakdown remains accurate when new kernels are added.

---

### Phase 1 — Productionize convolution: integrate Winograd + implicit GEMM with `IExpr` / `ISink`

**Objective**: preserve prototype performance while making the kernels usable by the full graph/fusion system.

#### 1.1 Decide “fast-path contracts” for production kernels

We currently rely on contracts like:
- **Alignment** for `loadAligned<16>()` / `storeAligned<16>()` to actually materialize 128-bit vector ops.
- **Channel divisibility** (`%4==0`) for the v2 kernels (Winograd-domain GEMM v2 and GEMM conv v2).
- **Layout expectations** (NHWC vs NCHW, packed weights layouts, intermediate buffers layout for Winograd).

**Tasks**
- Define explicit contracts per kernel entrypoint:
  - required alignment for input/output/weights/intermediates
  - supported layouts
  - supported `k/s/p/d` (Winograd is specialized: 3x3, s=1, p=1, d=1)
  - supported channel constraints and the fallback behavior
- Make those contracts enforceable at runtime:
  - validate and pick fallback if violated
  - optionally add debug asserts when a “fast kernel” is forced

#### 1.2 Adjust `IExpr` / `ISink` to preserve fast aligned vectorization

The prototypes bypass `IExpr`/`ISink` and operate on raw pointers. To keep performance after integration, we likely need to ensure the abstraction doesn’t hide:
- pointer alignment information
- “linear indexing” / reduced address math
- vector lane grouping (float4) across channels

**Candidate interface additions (directionally)**
- **Aligned pointer access**: allow an `IExpr` to expose alignment (e.g., “I can guarantee 16B alignment for base+offset for these offsets”).
- **Vectorized loads/stores**: allow `float4`/`uint4` pathways (or a generalized “vector width”).
- **Fast linear iterator**: provide a way for a producer/consumer pair to agree on a linearized iteration space for hot loops (avoid recomputing multi-dimensional coordinates repeatedly).
- **Optional “interior tile” capability**: allow the executor to request an interior-only dispatch region so kernels can skip bounds checks in the hot path.

**Deliverables**
- A minimal set of `IExpr`/`ISink` extensions (or “fast paths”) that let conv kernels:
  - keep `loadAligned<16>()` / `storeAligned<16>()`
  - keep coalesced memory patterns (especially weight packing)
  - avoid per-element index recomputation overhead

#### 1.3 Implement production entrypoints

**Winograd path (nonfused, 3-kernel pipeline)**
- Input transform kernel (write V)
- Winograd-domain GEMM v2 (consume V and packed weightsU; write M)
- Output transform kernel (consume M; write output)

**Implicit GEMM path**
- `gemmConvolution_v2` for `%4==0` channels (vectorized).
- `gemmConvolution` as general fallback (non-multiple-of-4 safe).

**Key requirements**
- No silent performance regression due to extra abstraction layers.
- Maintain the current packed weight layouts and their transforms.

**Exit criteria**
- End-to-end conv in the full engine uses production codepaths (not the unit-test-only prototype path).
- Perf matches prototypes within an acceptable tolerance (target: within ~5%, then iterate).

---

### Phase 2 — Auto algorithm selection + auto tile selection

**Objective**: select the best algorithm (Winograd vs implicit GEMM) and best tile configuration per layer, automatically.

#### 2.1 Algorithm selection logic

**Inputs**
- convolution parameters: `N, H, W, Cin, Cout, K, stride, pad, dilation, groups`
- tensor layout and alignment
- device properties (SM count, shared memory limits, subgroup size, etc.)

**Rules (initial)**
- **Winograd candidate** only when: 3x3, stride=1, dilation=1, pad=1 (or equivalent), and layout/packing requirements satisfied.
- Otherwise: **implicit GEMM**.
- For implicit GEMM:
  - prefer `%4==0` v2 kernel when applicable
  - else fallback to general kernel

#### 2.2 Tile selection logic

**Approach**
- Start with a small curated set of tile configs:
  - tuned for low/mid channel counts (4–64)
  - tuned for typical SD shapes
- Add an optional **on-device micro-autotune** mode:
  - run a tiny number of candidate configs once per device and cache results (persistent cache).
  - avoid expensive exhaustive search.

**Cache strategy**
- Key by: device + conv shape bucket + layout/stride/dilation categories + precision.

**Exit criteria**
- A selection system that is stable (no large variance run-to-run) and improves perf vs static choices.

---

### Phase 3 — Re-benchmark Stable Diffusion and optimize the next bottlenecks

**Objective**: after conv is fast and productionized, re-profile SD end-to-end and attack the top hotspots.

#### 3.1 End-to-end profiling pass

**Tasks**
- Run SD inference with:
  - GPU timestamps per kernel/operator
  - aggregated breakdown by op type (conv/linear/attention/norm/etc.)
- Identify:
  - top 3–5 kernels by total time
  - top 3–5 kernels by launch count (overhead risk)

#### 3.2 Likely next targets: linear and flash-attention-2

**Linear**
- Apply the same principles:
  - vectorized aligned loads/stores
  - packed weight layouts that make global reads coalesced
  - shared memory staging only when it actually reduces global traffic
  - keep register pressure controlled (don’t over-unroll by default)

**Attention (FlashAttention2-style)**
- Focus areas:
  - memory layout and tiling in Q/K/V
  - minimizing global reads/writes (keep softmax intermediates in registers/shared)
  - numerically stable softmax with minimal overhead
  - careful work partitioning to avoid under-occupancy

**Exit criteria**
- SD inference speed improves materially, and the bottleneck shifts (or the remaining gap becomes small).

---

### Phase 4 — FP16 + Tensor Cores via Slang `CoopMatrix`

**Objective**: once FP32 is robust and fast, add an FP16 path that is “extremely fast” on Tensor Core capable GPUs.

#### 4.1 Precision strategy
- Decide supported modes:
  - FP16 weights + FP16 activations (accum FP32) vs full FP16 accum
  - mixed-precision policies per operator (conv/linear/attention)
- Establish correctness tolerances and test coverage for FP16.

#### 4.2 Tensor Core kernels
- Implement Tensor Core versions of:
  - Winograd-domain GEMM (if beneficial in FP16; keep packing aligned with coop-matrix tile shapes)
  - implicit GEMM conv for low/mid channels where Winograd isn’t used
  - GEMM/linear kernels (likely the biggest win)

#### 4.3 Selection logic integration
- Extend Phase 2 selection:
  - choose FP16 Tensor Core kernels when device supports and inputs allow
  - fallback to FP32 otherwise

**Exit criteria**
- FP16 end-to-end SD path is correct and significantly faster than FP32 on supported GPUs.

---

### Cross-cutting engineering practices (apply to all phases)

- **Performance regression discipline**
  - keep perf benchmarks in CI or at least as a repeatable local script
  - capture baseline numbers before large refactors
- **Data layout discipline**
  - treat weight packing and alignment as first-class: it’s often the difference between “ok” and “great”
- **Don’t break the schedule**
  - kernel fusion is only good if it preserves the efficient execution schedule; otherwise it can regress sharply
- **Document contracts**
  - each “fast kernel” must clearly state alignment/layout/channel constraints and its fallback behavior


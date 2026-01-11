### Development Notes: Convolution Kernel Optimization Journey

This document captures the optimization work done so far on MLKL’s convolution path(s), the key experiments we ran, the results we observed, and the lessons that turned out to matter most.

---

### Goals & constraints

- **Primary goal**: close the performance gap vs high-performance GPU conv implementations for Stable Diffusion inference, focusing on the slowest common SD conv shape(s).
- **Constraint**: initially stay on **FP32 SIMT** (no Tensor Core / TF32 reliance) and optimize what we have.
- **Method**: make one change at a time, benchmark, and revert quickly if the change regresses.

---

### Baseline profiling observations (early)

Initial GPU profiling for the hot 3×3 conv showed classic “not compute-limited” symptoms for our baseline kernel:

- **0% Tensor Core usage**
- **High register pressure** (often ~100+ regs/thread)
- **Low occupancy / low active CTAs per SM**
- **Low SM throughput**

These symptoms strongly suggested we needed to focus on:
- reducing register pressure,
- improving global memory access patterns (coalescing, vectorization),
- and reducing overhead from bounds checks / index math / barriers in hot loops.

---

### GEMM-style convolution (`gemmConvolution`) experiments

#### Small wins that held up

- **Avoiding 64-bit index math in hot paths**: removing unnecessary `int64_t` address arithmetic in weight indexing produced a small but measurable speedup.
- **Compiler unrolling control**: adding `[loop]` hints to discourage aggressive unrolling reduced pressure and gave a small improvement in the hottest cases.

#### Changes that regressed (and why we reverted)

- “Fast-path” branches for full tiles (skipping bounds checks) looked sensible, but increased control-flow overhead and/or register pressure enough to cause a regression.
- Specialized code paths like `THREAD_OH==1 && THREAD_OW==1` also regressed (likely due to instruction scheduling and register allocation changes).

**Lesson**: for GPU kernels near a local optimum, small structural changes can easily shift register allocation and hurt occupancy even if they reduce nominal work.

---

### Winograd direction: why and how we approached it

For Stable Diffusion, a huge fraction of conv layers are:
- **3×3**, **stride=1**, **padding=1**, **dilation=1**

Winograd F(4,3) is a strong match here because it reduces multiply count for 3×3 stride-1 conv. A key SD-specific assumption that makes this especially attractive is that many of these convolutions run at **high channel counts** (e.g., hundreds to thousands of channels), where:
- the Winograd-domain GEMM is large enough to be efficient, and
- the transform overhead is relatively small compared to the total accumulation work.

We decided to prototype Winograd without touching the existing production conv pipeline until it proved faster and correct.

#### Prototyping strategy

- Start with a working, readable prototype to validate correctness.
- Split into a **nonfused 3-kernel pipeline** to measure time breakdown precisely:
  - input transform → Winograd-domain GEMM → output transform
- Use GPU timestamp profiling to confirm the bottleneck.

**Result**: the Winograd-domain GEMM dominated the time (often ~95%+), so the optimization focus was clear.

---

### High-impact Winograd GEMM breakthroughs

#### 1) Alignment-aware vector loads/stores mattered a lot

Converting hot float4 global loads/stores to Slang’s:
- `loadAligned<16>()`
- `storeAligned<16>()`

produced a large speedup. This validated that the compiler needed explicit alignment information to reliably generate 128-bit vector memory ops.

**Lesson**: alignment is not “nice to have” — it’s a core performance feature for bandwidth-bound kernels.

#### 2) Data layout / packing dominated performance (the “v2 packing” breakthrough)

The biggest single improvement came from changing how weights are packed so that global loads become coalesced for the kernel’s access pattern.

- With the original packing, threads loading a block of weights often walked memory with large strides, producing poor coalescing.
- With the v2 packing, weights for the CTA’s working set became contiguous in the dimension we load collectively, which drastically reduced memory transactions.

This took the Winograd nonfused path from ~**1.3 ms** down to ~**0.66–0.69 ms** on the hot 16×16, 2560→1280, batch=2 layer.

**Lesson**: when a kernel is memory-dominated, *packing/layout* is often worth more than any arithmetic micro-optimization.

---

### Fusion experiments: what we tried and what we learned

We explored fusing kernels to reduce intermediate global memory traffic.

- A fused (2+3) attempt (domain GEMM + output transform in one kernel) was **significantly slower** than the nonfused v2 pipeline.

Why this happened (high level):
- The fast Winograd GEMM kernel relies on a clean execution schedule (good coalescing, shared staging, predictable barriers).
- Naive fusion introduced large inner loops and synchronization/staging overhead that outweighed any savings from removing the `M` buffer.

**Lesson**: fusion can help, but only if it preserves (or improves) the kernel’s best properties; otherwise it can catastrophically regress.

We kept the fused kernel around as a reference, but the nonfused v2 pipeline remains the fast path today.

---

### Bringing Winograd lessons back to GEMM conv

We created a `gemmConvolution_v2` prototype that assumes:
- **NHWC float**
- **inChannels%4==0 and outChannels%4==0**
- **packed weights layout for coalesced access**
- **float4 staging and float4 accumulators**

This delivered a meaningful improvement on the hot config:
- baseline GEMM: ~**2.76 ms**
- GEMM v2 prototype: ~**1.98 ms**

**Lesson**: the same three pillars helped again:
- **vectorization + explicit alignment**
- **coalesced packing**
- **distributed cooperative loading**

Next step (planned): once validated, migrate the successful ideas back into the general `gemmConvolution` path (removing the %4 assumption by adding a remainder path), so it remains a robust fallback for odd channel counts.

---

### “Rules of thumb” that proved reliable

- **Always benchmark and revert quickly**: many “obvious” changes regress due to register allocation or scheduling effects.
- **Fix memory access first**:
  - coalescing > minor arithmetic changes
  - alignment hints matter for real vector ops
- **Measure per-kernel time** (not just end-to-end): it tells you exactly where to spend effort.
- **Be skeptical of fusion** unless you can keep the best parts of the schedule.
- **Prefer adding v2 entrypoints** while iterating: it keeps a known-good baseline for comparison and reduces integration risk.

---

### Current state (at the time of writing)

- Winograd nonfused with v2 packed weights is currently the fastest solution for the common SD 3×3 stride-1 case.
- GEMM-style convolution has improved via a v2 prototype but remains slower than Winograd for 3×3 stride-1 (which is expected due to Winograd’s algorithmic advantage).


"""
Benchmark PyTorch conv2d on CUDA for the SD-relevant config:
  N=2, H=W=16, Cin=2560, Cout=1280, K=3, stride=1, pad=1, dtype=float32

It also disables TF32 (so fp32 conv doesn't use TF32 tensor-core paths),
to be closer to "CUDA core FP32 FMA" apples-to-apples comparisons.

Usage examples:
  python tools/bench_pytorch_conv2d.py
  python tools/bench_pytorch_conv2d.py --iters 200 --warmup 50
  python tools/bench_pytorch_conv2d.py --channels-last
  python tools/bench_pytorch_conv2d.py --mlkl-ms 2.527

Notes:
  - PyTorch conv2d uses cuDNN by default. Disabling TF32 prevents TF32 math.
  - cuDNN may still pick different algorithms (implicit GEMM/FFT/etc). This script
    reports what PyTorch says about TF32, and benchmarks the chosen fast path.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ConvConfig:
    n: int = 2
    cin: int = 2560
    cout: int = 1280
    h: int = 16
    w: int = 16
    k: int = 3
    stride: int = 1
    pad: int = 1


def conv_gflops(cfg: ConvConfig, ms: float) -> float:
    out_h = (cfg.h + 2 * cfg.pad - cfg.k) // cfg.stride + 1
    out_w = (cfg.w + 2 * cfg.pad - cfg.k) // cfg.stride + 1
    # FLOPs = 2 * N * outH * outW * Cout * (Cin * K * K)
    flops = 2.0 * cfg.n * out_h * out_w * cfg.cout * (cfg.cin * cfg.k * cfg.k)
    return flops / (ms * 1e6)  # ms -> seconds: /1e3, FLOPs->GFLOPs: /1e9, combined /1e6


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("ERROR: Failed to import torch.")
        print("Install PyTorch with CUDA first, e.g. (example):")
        print("  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision")
        print("")
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200, help="Timed iterations")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    ap.add_argument("--channels-last", action="store_true", help="Use channels_last memory format (NHWC)")
    ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cudnn.benchmark (autotune)")
    ap.add_argument("--deterministic", action="store_true", help="Enable cudnn.deterministic (may be slower)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--mlkl-ms", type=float, default=None, help="Your MLKL time (ms) to compute a ratio")
    args = ap.parse_args()

    _require_torch()
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. Need a CUDA GPU + CUDA build of PyTorch.")
        return 2

    cfg = ConvConfig()

    # Disable TF32 so fp32 conv doesn't use TF32 tensor-core paths.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Also set matmul precision policy (mostly affects matmul, but harmless here).
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(args.deterministic)

    # Reduce noise.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    dtype = torch.float32

    # Create inputs/weights in NCHW as PyTorch expects.
    x = torch.randn((cfg.n, cfg.cin, cfg.h, cfg.w), device=device, dtype=dtype)
    w = torch.randn((cfg.cout, cfg.cin, cfg.k, cfg.k), device=device, dtype=dtype)
    b = torch.randn((cfg.cout,), device=device, dtype=dtype)

    if args.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        w = w.contiguous(memory_format=torch.channels_last)

    # Warm up + ensure kernels are compiled/tuned.
    torch.cuda.synchronize()
    y = None
    for _ in range(max(0, args.warmup)):
        y = F.conv2d(x, w, b, stride=cfg.stride, padding=cfg.pad)
    torch.cuda.synchronize()

    # Timing via CUDA events.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(max(1, args.iters)):
        y = F.conv2d(x, w, b, stride=cfg.stride, padding=cfg.pad)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / max(1, args.iters)

    # Prevent dead-code elimination concerns (shouldn't happen, but be safe).
    _ = float(y.sum().item())

    gflops = conv_gflops(cfg, avg_ms)

    print("=== PyTorch CUDA conv2d benchmark (FP32, TF32 disabled) ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    try:
        print(f"cuDNN:   {torch.backends.cudnn.version()}")
    except Exception:
        print("cuDNN:   (unknown)")
    print("")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    cc = torch.cuda.get_device_capability(0)
    print(f"SM:      {cc[0]}.{cc[1]}")
    print("")
    print(f"allow_tf32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
    print(f"allow_tf32 (cudnn):  {torch.backends.cudnn.allow_tf32}")
    print(f"cudnn.benchmark:     {torch.backends.cudnn.benchmark}")
    print(f"cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"memory_format:       {'channels_last' if args.channels_last else 'contiguous(NCHW)'}")
    print("")
    print(
        f"Config: N={cfg.n}, Cin={cfg.cin}, Cout={cfg.cout}, H=W={cfg.h}, "
        f"K={cfg.k}, stride={cfg.stride}, pad={cfg.pad}, dtype=float32"
    )
    print(f"Warmup: {args.warmup} iters, Timed: {args.iters} iters")
    print("")
    print(f"Avg time: {avg_ms:.3f} ms")
    print(f"GFLOPS:   {gflops:.1f}")

    if args.mlkl_ms is not None and args.mlkl_ms > 0:
        ratio = avg_ms / args.mlkl_ms
        print("")
        print(f"MLKL:     {args.mlkl_ms:.3f} ms (provided)")
        print(f"Ratio:    {ratio:.2f}x (PyTorch / MLKL)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


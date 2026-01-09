#!/usr/bin/env python3
"""
Benchmark PyTorch Stable Diffusion 1.5 with accurate GPU timing.

Usage:
    python generate_reference.py                # Run benchmark (default)
    python generate_reference.py --fp16         # Benchmark with FP16
    python generate_reference.py --no-xformers  # Benchmark without xformers (fair FP32 comparison)
    python generate_reference.py --no-benchmark # Just generate one image

Output:
    reference.png - PyTorch generated image for comparison
"""

import torch
import numpy as np
from PIL import Image
import argparse
import time

def benchmark_pytorch_sd(
    pipe,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    use_fp16: bool = False,
):
    """Benchmark Stable Diffusion with accurate GPU timing."""
    
    device = pipe.device
    
    print(f"\n{'='*60}")
    print("PyTorch Stable Diffusion Benchmark")
    print(f"{'='*60}")
    print(f"Steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Size: {width}x{height}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")
    print()
    
    # Warmup runs (not timed)
    print("Warming up...", flush=True)
    for i in range(warmup_runs):
        try:
            generator = torch.Generator(device=device).manual_seed(seed + i)
            with torch.no_grad():
                _ = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="pil",
                )
            torch.cuda.synchronize()
            print(f"  Warmup {i+1}/{warmup_runs} done", flush=True)
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    # Benchmark runs with CUDA events for accurate GPU timing
    print("\nRunning benchmark...", flush=True)
    
    times_total = []
    times_unet = []
    times_vae = []
    
    for i in range(benchmark_runs):
        generator = torch.Generator(device=device).manual_seed(seed + 100 + i)
        
        # Create CUDA events
        start_total = torch.cuda.Event(enable_timing=True)
        end_unet = torch.cuda.Event(enable_timing=True)
        end_total = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_total.record()
        
        with torch.no_grad():
            # Run pipeline - get latents (this includes UNet diffusion)
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )
        
        end_total.record()
        torch.cuda.synchronize()
        
        total_ms = start_total.elapsed_time(end_total)
        times_total.append(total_ms)
        
        print(f"  Run {i+1}/{benchmark_runs}: {total_ms:.1f} ms", flush=True)
    
    # Calculate statistics
    times_total = np.array(times_total)
    mean_ms = np.mean(times_total)
    std_ms = np.std(times_total)
    min_ms = np.min(times_total)
    max_ms = np.max(times_total)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total time (mean):    {mean_ms:.1f} ms")
    print(f"Total time (std):     {std_ms:.1f} ms")
    print(f"Total time (min):     {min_ms:.1f} ms")
    print(f"Total time (max):     {max_ms:.1f} ms")
    print(f"Time per step:        {mean_ms / num_inference_steps:.1f} ms/step")
    print(f"{'='*60}")
    
    return mean_ms


def main():
    parser = argparse.ArgumentParser(description="Generate SD reference image and benchmark")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmark, just generate one image")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--no-xformers", action="store_true", help="Disable xformers")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    args = parser.parse_args()
    
    # Check dependencies
    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler
    except ImportError:
        print("Please install diffusers: pip install diffusers transformers accelerate")
        return
    
    # Settings matching main.cpp
    prompt = "a beautiful sunset over mountains, digital art, highly detailed"
    seed = 1377  # Match C++ seed
    num_inference_steps = args.steps
    height = 512
    width = 512
    guidance_scale = 7.5
    
    print("=" * 60)
    print("Stable Diffusion 1.5 - PyTorch Reference")
    print("=" * 60)
    print(f"Prompt: \"{prompt}\"")
    print(f"Seed: {seed}")
    print(f"Steps: {num_inference_steps}")
    print(f"Size: {width}x{height}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
    print()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. GPU benchmarking requires CUDA.")
        return
    
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print()
    
    # Load pipeline
    print("Loading Stable Diffusion 1.5...")
    
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    pipe = pipe.to(device)
    
    # Enable optimizations
    if not args.no_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[OK] xformers memory efficient attention enabled")
        except Exception as e:
            print(f"[--] xformers not available: {e}")
            print("     (This is fine, will use default attention)")
    else:
        print("[--] xformers disabled by user")
    
    if args.compile:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("[OK] torch.compile enabled for UNet")
        except Exception as e:
            print(f"[--] torch.compile failed: {e}")
    
    print(f"Using device: {device}")
    print()
    
    if not args.no_benchmark:
        # Run benchmark
        benchmark_pytorch_sd(
            pipe=pipe,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
            use_fp16=args.fp16,
        )
    else:
        # Just generate one image
        print("Generating image...", flush=True)
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Time the generation
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
            )
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        print(f"\nGeneration time: {elapsed_ms:.1f} ms ({elapsed_ms/1000:.2f} sec)")
        print(f"Time per step: {elapsed_ms / num_inference_steps:.1f} ms/step")
        
        image = result.images[0]
        image.save("reference.png")
        print("Saved to reference.png")
        
        # Save reference latent
        gen2 = torch.Generator(device=device).manual_seed(seed)
        initial_latent = torch.randn(
            (1, 4, height // 8, width // 8),
            generator=gen2,
            device=device,
            dtype=torch_dtype,
        )
        latent_nhwc = initial_latent.float().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        latent_nhwc.astype(np.float32).tofile("test_data/reference_latent.bin")
        print(f"Saved test_data/reference_latent.bin (shape: {latent_nhwc.shape})")


if __name__ == "__main__":
    main()

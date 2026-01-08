#!/usr/bin/env python3
"""
Generate a reference image using PyTorch for comparison with MLKL output.

This uses the exact same settings as main.cpp:
- Prompt: "a beautiful sunset over mountains, digital art, highly detailed"
- Seed: 42
- Steps: 20
- Image size: 512x512
- Scheduler: DDIM

Usage:
    python generate_reference.py

Output:
    reference.png - PyTorch generated image for comparison
"""

import torch
import numpy as np
from PIL import Image

def main():
    # Check dependencies
    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler
    except ImportError:
        print("Please install diffusers: pip install diffusers transformers accelerate")
        return
    
    # Same settings as main.cpp
    prompt = "a beautiful sunset over mountains, digital art, highly detailed"
    seed = 42
    num_inference_steps = 20
    height = 512
    width = 512
    
    # Classifier-Free Guidance (CFG)
    # guidance_scale = 1.0 means no CFG (what we had before - blurry)
    # guidance_scale = 7.5 is typical for good results
    guidance_scale = 7.5
    
    print("=" * 60)
    print("Generating Reference Image")
    print("=" * 60)
    print(f"Prompt: \"{prompt}\"")
    print(f"Seed: {seed}")
    print(f"Steps: {num_inference_steps}")
    print(f"Size: {width}x{height}")
    print()
    
    # Load pipeline with same components
    print("Loading Stable Diffusion 1.5...")
    
    # Use DDIM scheduler to match our implementation
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
    )
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # Use CUDA if available
    device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Using xformers memory efficient attention")
        except:
            pass
    
    pipe = pipe.to(device)
    print(f"Using device: {device}", flush=True)
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image
    print(f"\nGenerating image with {num_inference_steps} steps...", flush=True)
    
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        print("Starting pipeline...", flush=True)
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )
        print("Pipeline finished!", flush=True)
        
        image = result.images[0]
        
        # Save to file
        image.save("reference.png")
        print("Saved to reference.png", flush=True)
        
        # Also save the initial latent used (for C++ to load)
        # Re-generate with same seed to get the latent
        gen2 = torch.Generator(device=device).manual_seed(seed)
        initial_latent = torch.randn((1, 4, height // 8, width // 8), generator=gen2, device=device, dtype=torch.float32)
        
        # Save in NHWC format (C++ uses NHWC)
        latent_nhwc = initial_latent.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        latent_nhwc.astype(np.float32).tofile("test_data/reference_latent.bin")
        print(f"Saved test_data/reference_latent.bin (shape: {latent_nhwc.shape})", flush=True)
        print(f"First 8 values: {latent_nhwc.flatten()[:8].tolist()}")
        
        # Display in window
        image.show()
        
        # Keep script alive
        input("Press Enter to exit...")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()

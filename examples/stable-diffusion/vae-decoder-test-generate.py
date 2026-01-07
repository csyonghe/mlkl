#!/usr/bin/env python3
"""
Generate test data for VAE decoder validation.

This script loads the SD 1.5 VAE from Hugging Face and generates:
1. Random latent input
2. Reference decoder output
3. Intermediate layer outputs for debugging

Usage:
    python vae-decoder-test-generate.py

Prerequisites:
    pip install torch diffusers safetensors
"""

import os
import torch
import numpy as np
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_tensor(tensor, path):
    """Save tensor as raw binary (float32)"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().float().numpy()
    tensor.astype(np.float32).tofile(path)
    print(f"Saved {path}: shape={tensor.shape}, dtype={tensor.dtype}")

def generate_test_data():
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        print("Please install diffusers: pip install diffusers")
        return

    output_dir = "test_data"
    ensure_dir(output_dir)
    
    print("Loading SD 1.5 VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    vae.eval()
    
    # Generate random latent input
    # SD 1.5 uses latent shape [B, 4, H/8, W/8]
    # For 512x512 output, latent is [1, 4, 64, 64]
    print("\nGenerating test latent...")
    torch.manual_seed(42)
    latent = torch.randn(1, 4, 64, 64)
    
    # Scale latent (VAE expects scaled latents)
    # The scaling factor for SD 1.5 VAE is 0.18215
    latent_scaled = latent / 0.18215
    
    # Save input (in NCHW format as SD uses)
    save_tensor(latent_scaled, f"{output_dir}/vae_latent_input_nchw.bin")
    
    # Also save in NHWC format (what our engine uses)
    latent_nhwc = latent_scaled.permute(0, 2, 3, 1).contiguous()
    save_tensor(latent_nhwc, f"{output_dir}/vae_latent_input.bin")
    
    # Run decoder
    print("\nRunning VAE decoder...")
    with torch.no_grad():
        decoded = vae.decode(latent_scaled).sample
    
    # Clamp to valid image range
    decoded = torch.clamp(decoded, -1, 1)
    
    # Save output in NCHW
    save_tensor(decoded, f"{output_dir}/vae_decoder_output_nchw.bin")
    
    # Save in NHWC (what our engine produces)
    decoded_nhwc = decoded.permute(0, 2, 3, 1).contiguous()
    save_tensor(decoded_nhwc, f"{output_dir}/vae_decoder_output.bin")
    
    # Generate intermediate outputs for debugging
    # These match the debug points in VAEDecoder::queueExecute
    print("\nGenerating intermediate outputs...")
    
    # Hook to capture intermediate activations
    intermediates = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            intermediates[name] = output.detach()
        return hook
    
    # Register hooks on key layers (matching C++ debug points)
    hooks = []
    hooks.append(vae.post_quant_conv.register_forward_hook(make_hook("01_post_quant_conv")))
    hooks.append(vae.decoder.conv_in.register_forward_hook(make_hook("02_conv_in")))
    hooks.append(vae.decoder.mid_block.resnets[0].register_forward_hook(make_hook("03_mid_resnet1")))
    hooks.append(vae.decoder.mid_block.attentions[0].register_forward_hook(make_hook("04_mid_attn")))
    hooks.append(vae.decoder.mid_block.resnets[1].register_forward_hook(make_hook("05_mid_resnet2")))
    
    for i, up_block in enumerate(vae.decoder.up_blocks):
        hooks.append(up_block.register_forward_hook(make_hook(f"06_up_block_{i}")))
    
    hooks.append(vae.decoder.conv_norm_out.register_forward_hook(make_hook("07_norm_out")))
    hooks.append(vae.decoder.conv_out.register_forward_hook(make_hook("08_conv_out")))
    
    # Run again with hooks
    with torch.no_grad():
        _ = vae.decode(latent_scaled)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Save intermediates with matching names
    for name, tensor in intermediates.items():
        # Save in NHWC format (matching C++ layout)
        if tensor.dim() == 4:
            tensor_nhwc = tensor.permute(0, 2, 3, 1).contiguous()
        else:
            tensor_nhwc = tensor
        save_tensor(tensor_nhwc, f"{output_dir}/ref_{name}.bin")
        
        # Also print first 8 values for quick comparison
        flat = tensor_nhwc.flatten()
        print(f"  REF {name}: shape={list(tensor_nhwc.shape)}, first 8: {flat[:8].tolist()}")
    
    print("\n=== Test data generation complete ===")
    print(f"Files saved to {output_dir}/")
    print("\nTo run validation:")
    print("  1. Build the stable-diffusion example")
    print("  2. Run: ./stable-diffusion")

def generate_small_test_data():
    """Generate small synthetic test data (no model required)"""
    output_dir = "test_data_small"
    ensure_dir(output_dir)
    
    print("Generating small synthetic test data...")
    
    # Small latent [1, 4, 4, 4] in NHWC
    np.random.seed(42)
    latent = np.random.randn(1, 4, 4, 4).astype(np.float32)
    save_tensor(latent, f"{output_dir}/small_latent_input.bin")
    
    print(f"Saved to {output_dir}/")

if __name__ == "__main__":
    import sys
    
    if "--small" in sys.argv:
        generate_small_test_data()
    else:
        generate_test_data()


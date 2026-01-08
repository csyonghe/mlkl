#!/usr/bin/env python3
"""
Generate test data for SD 1.5 UNet testing.

This script:
1. Downloads the SD 1.5 UNet model from HuggingFace
2. Exports weights to SafeTensors format
3. Generates random test inputs and reference outputs

Run from the examples/stable-diffusion directory:
    python unet-test-generate.py
"""

import os
import sys
import numpy as np
import torch

def ensure_dependencies():
    """Check and install required dependencies."""
    try:
        from diffusers import UNet2DConditionModel
        from safetensors.torch import save_file
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install diffusers safetensors transformers accelerate")
        from diffusers import UNet2DConditionModel
        from safetensors.torch import save_file
    return True

def main():
    ensure_dependencies()
    
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file
    
    # Create output directory
    os.makedirs("test_data", exist_ok=True)
    
    # ========================================================================
    # Download and export UNet model
    # ========================================================================
    print("Loading SD 1.5 UNet from HuggingFace...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float32
    )
    unet.eval()
    
    # Export weights to SafeTensors
    print("Exporting UNet weights to SafeTensors...")
    state_dict = unet.state_dict()
    
    # Convert to float32 for compatibility
    state_dict_f32 = {k: v.float() for k, v in state_dict.items()}
    
    save_file(state_dict_f32, "test_data/unet.safetensors")
    print(f"Saved test_data/unet.safetensors ({len(state_dict_f32)} tensors)")
    
    # Print some weight names for reference
    print("\nSample weight names:")
    for i, name in enumerate(sorted(state_dict_f32.keys())):
        if i < 20:
            shape = list(state_dict_f32[name].shape)
            print(f"  {name}: {shape}")
    print(f"  ... and {len(state_dict_f32) - 20} more")
    
    # ========================================================================
    # Generate test inputs
    # ========================================================================
    print("\nGenerating test inputs...")
    
    batch_size = 1
    height = 64
    width = 64
    latent_channels = 4
    seq_len = 77
    context_dim = 768
    timestep = 500
    
    # Random latent input (simulating VAE-encoded image)
    torch.manual_seed(42)
    latent = torch.randn(batch_size, latent_channels, height, width)
    
    # Random context (simulating CLIP text embeddings)
    context = torch.randn(batch_size, seq_len, context_dim)
    
    # Timestep
    timesteps = torch.tensor([timestep])
    
    # ========================================================================
    # Run reference inference
    # ========================================================================
    print(f"Running reference inference (timestep={timestep})...")
    
    with torch.no_grad():
        output = unet(
            latent,
            timesteps,
            encoder_hidden_states=context,
            return_dict=False
        )[0]
    
    print(f"  Input latent shape: {latent.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Output shape: {output.shape}")
    
    # ========================================================================
    # Convert to NHWC and save
    # ========================================================================
    print("\nSaving test data (NHWC format for C++ engine)...")
    
    # Convert NCHW → NHWC
    latent_nhwc = latent.permute(0, 2, 3, 1).contiguous()
    output_nhwc = output.permute(0, 2, 3, 1).contiguous()
    
    # Save as binary files
    latent_nhwc.numpy().astype(np.float32).tofile("test_data/unet_input_latent.bin")
    context.numpy().astype(np.float32).tofile("test_data/unet_input_context.bin")
    output_nhwc.numpy().astype(np.float32).tofile("test_data/unet_output.bin")
    
    print(f"  Saved test_data/unet_input_latent.bin: shape={tuple(latent_nhwc.shape)}")
    print(f"  Saved test_data/unet_input_context.bin: shape={tuple(context.shape)}")
    print(f"  Saved test_data/unet_output.bin: shape={tuple(output_nhwc.shape)}")
    
    # Print first few values for verification
    print("\nReference values:")
    print(f"  Input latent first 8: {latent_nhwc.flatten()[:8].tolist()}")
    print(f"  Context first 8: {context.flatten()[:8].tolist()}")
    print(f"  Output first 8: {output_nhwc.flatten()[:8].tolist()}")
    
    # ========================================================================
    # Save timestep info
    # ========================================================================
    with open("test_data/unet_timestep.txt", "w") as f:
        f.write(f"{timestep}\n")
    print(f"  Saved test_data/unet_timestep.txt: {timestep}")
    
    print("\n=== UNet test data generation complete ===")
    print("Files saved to test_data/")

if __name__ == "__main__":
    main()


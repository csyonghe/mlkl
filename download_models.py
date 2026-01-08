#!/usr/bin/env python3
"""
Download all model weights and generate test data for MLKL Stable Diffusion example.

This script downloads:
1. Model weights to models/ directory
2. Test data to test_data/ directory

Usage:
    python download_models.py [--models-only] [--test-data-only]

Requirements:
    pip install torch diffusers transformers safetensors huggingface_hub

Models downloaded (~5GB total):
    - CLIP text encoder (openai/clip-vit-large-patch14)
    - UNet (runwayml/stable-diffusion-v1-5)
    - VAE (stabilityai/sd-vae-ft-mse)
    - Tokenizer vocabulary and merges
"""

import os
import sys
import shutil
import argparse
import numpy as np
from pathlib import Path

# ============================================================================
# Dependency check
# ============================================================================

def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import diffusers
    except ImportError:
        missing.append("diffusers")
    
    try:
        import safetensors
    except ImportError:
        missing.append("safetensors")
    
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    
    return True


# ============================================================================
# Directory setup
# ============================================================================

def get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.resolve()

def ensure_dirs():
    """Create output directories."""
    repo_root = get_repo_root()
    
    models_dir = repo_root / "models"
    test_data_dir = repo_root / "test_data"
    
    models_dir.mkdir(exist_ok=True)
    test_data_dir.mkdir(exist_ok=True)
    
    return models_dir, test_data_dir


# ============================================================================
# Model downloads
# ============================================================================

def download_tokenizer(models_dir):
    """Download CLIP tokenizer vocab and merges."""
    from huggingface_hub import hf_hub_download
    
    print("\n=== Downloading CLIP Tokenizer ===")
    
    vocab_path = models_dir / "vocab.json"
    merges_path = models_dir / "merges.txt"
    
    repo_id = "openai/clip-vit-large-patch14"
    
    if not vocab_path.exists():
        print("  Downloading vocab.json...")
        downloaded = hf_hub_download(repo_id=repo_id, filename="vocab.json")
        shutil.copy(downloaded, vocab_path)
        print(f"  Saved: {vocab_path}")
    else:
        print(f"  vocab.json already exists")
    
    if not merges_path.exists():
        print("  Downloading merges.txt...")
        downloaded = hf_hub_download(repo_id=repo_id, filename="merges.txt")
        shutil.copy(downloaded, merges_path)
        print(f"  Saved: {merges_path}")
    else:
        print(f"  merges.txt already exists")


def download_clip(models_dir):
    """Download and export CLIP text encoder."""
    import torch
    from transformers import CLIPTextModel
    from safetensors.torch import save_file
    
    print("\n=== Downloading CLIP Text Encoder ===")
    
    output_path = models_dir / "clip.safetensors"
    
    if output_path.exists():
        print(f"  CLIP model already exists: {output_path}")
        return
    
    print("  Loading openai/clip-vit-large-patch14...")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    
    print(f"  Config: hidden_size={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}, "
          f"vocab={model.config.vocab_size}")
    
    # Export weights
    print("  Exporting to SafeTensors...")
    weights = {name: param.data.cpu().float() for name, param in model.named_parameters()}
    save_file(weights, str(output_path))
    
    print(f"  Saved: {output_path} ({len(weights)} tensors)")


def download_unet(models_dir):
    """Download and export UNet."""
    import torch
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file
    
    print("\n=== Downloading UNet ===")
    
    output_path = models_dir / "unet.safetensors"
    
    if output_path.exists():
        print(f"  UNet model already exists: {output_path}")
        return
    
    print("  Loading runwayml/stable-diffusion-v1-5/unet (this may take a while)...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float32
    )
    unet.eval()
    
    # Export weights
    print("  Exporting to SafeTensors...")
    weights = {k: v.float() for k, v in unet.state_dict().items()}
    save_file(weights, str(output_path))
    
    print(f"  Saved: {output_path} ({len(weights)} tensors)")


def download_vae(models_dir):
    """Download and export VAE decoder."""
    import torch
    from diffusers import AutoencoderKL
    from safetensors.torch import save_file
    
    print("\n=== Downloading VAE ===")
    
    output_path = models_dir / "vae.safetensors"
    
    if output_path.exists():
        print(f"  VAE model already exists: {output_path}")
        return
    
    print("  Loading stabilityai/sd-vae-ft-mse...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    vae.eval()
    
    # Export weights
    print("  Exporting to SafeTensors...")
    weights = {k: v.float() for k, v in vae.state_dict().items()}
    save_file(weights, str(output_path))
    
    print(f"  Saved: {output_path} ({len(weights)} tensors)")


# ============================================================================
# Test data generation
# ============================================================================

def save_binary(data, path):
    """Save numpy array as binary float32 file."""
    data = np.ascontiguousarray(data.astype(np.float32))
    data.tofile(path)


def generate_clip_test_data(test_data_dir):
    """Generate CLIP encoder test data (input/output for validation)."""
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    
    print("\n=== Generating CLIP Test Data ===")
    
    # Check if already exists
    if (test_data_dir / "clip_output.bin").exists():
        print("  CLIP test data already exists")
        return
    
    print("  Loading CLIP model...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    
    # Generate test input
    prompt = "a beautiful sunset over the mountains, digital art, highly detailed"
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"]
    
    # Save input tokens
    save_binary(input_ids.numpy(), test_data_dir / "clip_input_tokens.bin")
    print(f"  Saved: clip_input_tokens.bin (prompt: '{prompt[:50]}...')")
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state
    
    # Save output
    save_binary(hidden_states.numpy(), test_data_dir / "clip_output.bin")
    print(f"  Saved: clip_output.bin (shape: {list(hidden_states.shape)})")


def generate_unet_test_data(test_data_dir):
    """Generate UNet test data."""
    import torch
    from diffusers import UNet2DConditionModel
    
    print("\n=== Generating UNet Test Data ===")
    
    # Check if already exists
    if (test_data_dir / "unet_output.bin").exists():
        print("  UNet test data already exists")
        return
    
    print("  Loading UNet model...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float32
    )
    unet.eval()
    
    # Generate test inputs
    batch_size = 1
    height, width = 64, 64
    seq_len, context_dim = 77, 768
    timestep = 500
    
    torch.manual_seed(42)
    latent = torch.randn(batch_size, 4, height, width)
    context = torch.randn(batch_size, seq_len, context_dim)
    timesteps = torch.tensor([timestep])
    
    # Run inference
    print(f"  Running inference (timestep={timestep})...")
    with torch.no_grad():
        output = unet(latent, timesteps, encoder_hidden_states=context, return_dict=False)[0]
    
    # Convert to NHWC and save
    latent_nhwc = latent.permute(0, 2, 3, 1).contiguous()
    output_nhwc = output.permute(0, 2, 3, 1).contiguous()
    
    save_binary(latent_nhwc.numpy(), test_data_dir / "unet_input_latent.bin")
    save_binary(context.numpy(), test_data_dir / "unet_input_context.bin")
    save_binary(output_nhwc.numpy(), test_data_dir / "unet_output.bin")
    
    with open(test_data_dir / "unet_timestep.txt", "w") as f:
        f.write(f"{timestep}\n")
    
    print(f"  Saved: unet_input_latent.bin (shape: {list(latent_nhwc.shape)})")
    print(f"  Saved: unet_input_context.bin (shape: {list(context.shape)})")
    print(f"  Saved: unet_output.bin (shape: {list(output_nhwc.shape)})")
    print(f"  Saved: unet_timestep.txt ({timestep})")


def generate_vae_test_data(test_data_dir):
    """Generate VAE decoder test data."""
    import torch
    from diffusers import AutoencoderKL
    
    print("\n=== Generating VAE Test Data ===")
    
    # Check if already exists
    if (test_data_dir / "vae_decoder_output.bin").exists():
        print("  VAE test data already exists")
        return
    
    print("  Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    vae.eval()
    
    # Generate test input
    torch.manual_seed(42)
    latent = torch.randn(1, 4, 64, 64)
    latent_scaled = latent / 0.18215  # VAE scaling factor
    
    # Run inference
    print("  Running decoder...")
    with torch.no_grad():
        decoded = vae.decode(latent_scaled).sample
    decoded = torch.clamp(decoded, -1, 1)
    
    # Convert to NHWC and save
    latent_nhwc = latent_scaled.permute(0, 2, 3, 1).contiguous()
    decoded_nhwc = decoded.permute(0, 2, 3, 1).contiguous()
    
    save_binary(latent_nhwc.numpy(), test_data_dir / "vae_latent_input.bin")
    save_binary(decoded_nhwc.numpy(), test_data_dir / "vae_decoder_output.bin")
    
    print(f"  Saved: vae_latent_input.bin (shape: {list(latent_nhwc.shape)})")
    print(f"  Saved: vae_decoder_output.bin (shape: {list(decoded_nhwc.shape)})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download MLKL Stable Diffusion models and test data")
    parser.add_argument("--models-only", action="store_true", help="Only download models")
    parser.add_argument("--test-data-only", action="store_true", help="Only generate test data")
    parser.add_argument("--force", action="store_true", help="Re-download/regenerate even if files exist")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MLKL Stable Diffusion Model Downloader")
    print("=" * 60)
    
    check_dependencies()
    models_dir, test_data_dir = ensure_dirs()
    
    print(f"\nModels directory: {models_dir}")
    print(f"Test data directory: {test_data_dir}")
    
    if args.force:
        print("\n[Force mode: will overwrite existing files]")
        # Remove existing files
        if not args.test_data_only:
            for f in models_dir.glob("*"):
                f.unlink()
        if not args.models_only:
            for f in test_data_dir.glob("*"):
                f.unlink()
    
    # Download models
    if not args.test_data_only:
        download_tokenizer(models_dir)
        download_clip(models_dir)
        download_unet(models_dir)
        download_vae(models_dir)
    
    # Generate test data
    if not args.models_only:
        generate_clip_test_data(test_data_dir)
        generate_unet_test_data(test_data_dir)
        generate_vae_test_data(test_data_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    if not args.test_data_only:
        print(f"\nModel files in {models_dir}/:")
        for f in sorted(models_dir.glob("*")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    
    if not args.models_only:
        print(f"\nTest data in {test_data_dir}/:")
        for f in sorted(test_data_dir.glob("*")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("\nNext steps:")
    print("  1. Build the project: cmake --build build")
    print("  2. Run tests: ./build/bin/stable-diffusion --test")
    print("  3. Generate image: ./build/bin/stable-diffusion")


if __name__ == "__main__":
    main()

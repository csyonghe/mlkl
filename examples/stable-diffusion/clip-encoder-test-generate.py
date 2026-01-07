#!/usr/bin/env python3
"""
Generate test data for CLIP Text Encoder validation.

This script:
1. Downloads the CLIP text encoder from HuggingFace (or uses a local cache)
2. Exports the model weights to SafeTensors format
3. Generates sample input tokens and reference output

Usage:
    python clip-encoder-test-generate.py

Output files (in test_data/):
    - clip_text_model.safetensors: Model weights
    - clip_input_tokens.bin: Sample input (token IDs as float32)
    - clip_output.bin: Reference output from PyTorch (float32)
"""

import os
import numpy as np
import torch
from pathlib import Path

# Try to import transformers for CLIP
try:
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)

# Try to import safetensors
try:
    from safetensors.torch import save_file
except ImportError:
    print("Please install safetensors: pip install safetensors")
    exit(1)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_binary(data: np.ndarray, path: str):
    """Save numpy array as binary file."""
    # Ensure C-contiguous and float32
    data = np.ascontiguousarray(data.astype(np.float32))
    data.tofile(path)
    print(f"Saved {path}: shape={data.shape}, dtype={data.dtype}")


def main():
    output_dir = "test_data"
    ensure_dir(output_dir)
    
    # SD 1.5 uses "openai/clip-vit-large-patch14"
    model_name = "openai/clip-vit-large-patch14"
    
    print(f"Loading CLIP model: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    model.eval()
    
    # Print model config
    print(f"\nModel config:")
    print(f"  vocab_size: {model.config.vocab_size}")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_attention_heads: {model.config.num_attention_heads}")
    print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"  max_position_embeddings: {model.config.max_position_embeddings}")
    print(f"  intermediate_size: {model.config.intermediate_size}")
    
    # Export weights to SafeTensors
    print("\nExporting model weights...")
    weights_dict = {}
    for name, param in model.named_parameters():
        # Remove "text_model." prefix if present (we'll use our own prefix structure)
        clean_name = name
        weights_dict[clean_name] = param.data.cpu()
        print(f"  {clean_name}: {list(param.shape)}")
    
    safetensors_path = os.path.join(output_dir, "clip_text_model.safetensors")
    save_file(weights_dict, safetensors_path)
    print(f"\nSaved model to: {safetensors_path}")
    
    # Generate test input
    print("\nGenerating test input...")
    
    # A sample prompt
    prompt = "a beautiful sunset over the mountains, digital art, highly detailed"
    
    # Tokenize (max_length=77 is standard for SD)
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = tokens["input_ids"]  # [1, 77]
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs shape: {input_ids.shape}")
    print(f"First 10 tokens: {input_ids[0, :10].tolist()}")
    
    # Save input tokens as float32 (since our gather kernel expects float indices)
    input_tokens_path = os.path.join(output_dir, "clip_input_tokens.bin")
    save_binary(input_ids.numpy(), input_tokens_path)
    
    # Run the model
    print("\nRunning CLIP forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state  # [B, seqLen, hidden_size]
    
    print(f"Output shape: {hidden_states.shape}")
    print(f"Output first 8 values: {hidden_states[0, 0, :8].tolist()}")
    
    # Save reference output
    output_path = os.path.join(output_dir, "clip_output.bin")
    save_binary(hidden_states.numpy(), output_path)
    
    print("\n=== Test data generation complete ===")
    print(f"Files saved to {output_dir}/")
    
    # Check the actual weight names to determine the prefix
    first_name = list(weights_dict.keys())[0]
    print(f"\nFirst weight name: {first_name}")
    if first_name.startswith("text_model."):
        print("Expected weight prefix for loadParams: 'text_model.'")
    else:
        print("Expected weight prefix for loadParams: '' (empty string)")


if __name__ == "__main__":
    main()


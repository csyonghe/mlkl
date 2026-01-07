#!/usr/bin/env python3
"""
Generate SafeTensors test files for unit testing the SafeTensorsReader class.

Run with: python generate.py

This creates several .safetensors files with known values that can be verified
by the C++ unit tests.
"""

import torch
from safetensors.torch import save_file
from pathlib import Path

def main():
    output_dir = Path(__file__).parent
    
    # ==========================================================================
    # Test 1: Basic data types
    # ==========================================================================
    tensors_basic = {
        # Float32 tensors
        "f32_scalar": torch.tensor([42.0], dtype=torch.float32),
        "f32_1d": torch.arange(16, dtype=torch.float32),
        "f32_2d": torch.arange(24, dtype=torch.float32).reshape(4, 6),
        
        # Float16 tensors
        "f16_1d": torch.arange(16, dtype=torch.float16),
        "f16_2d": torch.arange(24, dtype=torch.float16).reshape(4, 6),
        
        # BFloat16 tensors
        "bf16_1d": torch.arange(16, dtype=torch.bfloat16),
        "bf16_2d": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6),
    }
    save_file(tensors_basic, output_dir / "test_basic_types.safetensors")
    print(f"Created: test_basic_types.safetensors ({len(tensors_basic)} tensors)")
    
    # ==========================================================================
    # Test 2: Linear layer weights (2D tensors)
    # ==========================================================================
    # PyTorch Linear stores weights as [OutFeatures, InFeatures]
    # Values: weight[out, in] = out * 100 + in
    in_features, out_features = 8, 4
    linear_weight = torch.zeros(out_features, in_features, dtype=torch.float32)
    for o in range(out_features):
        for i in range(in_features):
            linear_weight[o, i] = o * 100 + i
    
    linear_bias = torch.arange(out_features, dtype=torch.float32)
    
    tensors_linear = {
        "linear.weight": linear_weight,
        "linear.bias": linear_bias,
    }
    save_file(tensors_linear, output_dir / "test_linear.safetensors")
    print(f"Created: test_linear.safetensors ({len(tensors_linear)} tensors)")
    
    # ==========================================================================
    # Test 3: Conv2D weights (4D tensors) - [OutCh, InCh, Ky, Kx]
    # ==========================================================================
    # Values: weight[o, i, ky, kx] = o*1000 + i*100 + ky*10 + kx
    out_ch, in_ch, k = 2, 3, 3
    conv_weight = torch.zeros(out_ch, in_ch, k, k, dtype=torch.float32)
    for o in range(out_ch):
        for i in range(in_ch):
            for ky in range(k):
                for kx in range(k):
                    conv_weight[o, i, ky, kx] = o*1000 + i*100 + ky*10 + kx
    
    conv_bias = torch.arange(out_ch, dtype=torch.float32)
    
    tensors_conv = {
        "conv.weight": conv_weight,
        "conv.bias": conv_bias,
    }
    save_file(tensors_conv, output_dir / "test_conv2d.safetensors")
    print(f"Created: test_conv2d.safetensors ({len(tensors_conv)} tensors)")
    
    # ==========================================================================
    # Test 4: ConvTranspose2D weights (4D tensors) - [InCh, OutCh, Ky, Kx]
    # ==========================================================================
    # Note: PyTorch ConvTranspose2d has shape [InCh, OutCh, K, K] (different from Conv2d!)
    # Values: weight[i, o, ky, kx] = i*1000 + o*100 + ky*10 + kx
    in_ch, out_ch, k = 3, 2, 3
    tconv_weight = torch.zeros(in_ch, out_ch, k, k, dtype=torch.float32)
    for i in range(in_ch):
        for o in range(out_ch):
            for ky in range(k):
                for kx in range(k):
                    tconv_weight[i, o, ky, kx] = i*1000 + o*100 + ky*10 + kx
    
    tconv_bias = torch.arange(out_ch, dtype=torch.float32)
    
    tensors_tconv = {
        "tconv.weight": tconv_weight,
        "tconv.bias": tconv_bias,
    }
    save_file(tensors_tconv, output_dir / "test_transposed_conv2d.safetensors")
    print(f"Created: test_transposed_conv2d.safetensors ({len(tensors_tconv)} tensors)")
    
    # ==========================================================================
    # Test 5: GroupNorm / LayerNorm parameters
    # ==========================================================================
    num_channels = 32
    tensors_norm = {
        "norm.weight": torch.arange(1, num_channels + 1, dtype=torch.float32),  # gamma
        "norm.bias": torch.arange(num_channels, dtype=torch.float32) * 0.1,     # beta
    }
    save_file(tensors_norm, output_dir / "test_norm.safetensors")
    print(f"Created: test_norm.safetensors ({len(tensors_norm)} tensors)")
    
    # ==========================================================================
    # Test 6: Embedding weights (like CLIP token embeddings)
    # ==========================================================================
    vocab_size, embed_dim = 10, 8
    embed_weight = torch.zeros(vocab_size, embed_dim, dtype=torch.float32)
    for v in range(vocab_size):
        for e in range(embed_dim):
            embed_weight[v, e] = v * 100 + e
    
    tensors_embed = {
        "embed.weight": embed_weight,
    }
    save_file(tensors_embed, output_dir / "test_embedding.safetensors")
    print(f"Created: test_embedding.safetensors ({len(tensors_embed)} tensors)")
    
    # ==========================================================================
    # Test 7: Mixed precision file (F16 weights, F32 biases - common pattern)
    # ==========================================================================
    tensors_mixed = {
        "layer.weight": torch.arange(64, dtype=torch.float16).reshape(8, 8),
        "layer.bias": torch.arange(8, dtype=torch.float32),
    }
    save_file(tensors_mixed, output_dir / "test_mixed_precision.safetensors")
    print(f"Created: test_mixed_precision.safetensors ({len(tensors_mixed)} tensors)")
    
    # ==========================================================================
    # Test 8: Type conversion verification
    # ==========================================================================
    # Use specific values that can be exactly represented in F16
    # to verify conversions are accurate
    exact_values = torch.tensor([
        0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 
        0.125, 4.0, 8.0, 16.0, 0.0625, 100.0, 1000.0, 2048.0
    ], dtype=torch.float32)
    
    tensors_convert = {
        "exact_f32": exact_values,
        "exact_f16": exact_values.to(torch.float16),
        "exact_bf16": exact_values.to(torch.bfloat16),
    }
    save_file(tensors_convert, output_dir / "test_conversion.safetensors")
    print(f"Created: test_conversion.safetensors ({len(tensors_convert)} tensors)")
    
    # ==========================================================================
    # Test 9: Permutation verification with small tensor
    # ==========================================================================
    # Create [2, 3, 2, 2] tensor where value encodes position
    # This makes it easy to verify permutation correctness
    perm_test = torch.zeros(2, 3, 2, 2, dtype=torch.float32)
    for d0 in range(2):
        for d1 in range(3):
            for d2 in range(2):
                for d3 in range(2):
                    # Encode all 4 indices into the value
                    perm_test[d0, d1, d2, d3] = d0*1000 + d1*100 + d2*10 + d3
    
    tensors_perm = {
        "perm_test": perm_test,
    }
    save_file(tensors_perm, output_dir / "test_permutation.safetensors")
    print(f"Created: test_permutation.safetensors ({len(tensors_perm)} tensors)")
    
    # ==========================================================================
    # Print summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Test files generated successfully!")
    print("="*60)
    print("\nExpected test values:")
    print("\ntest_conv2d.safetensors 'conv.weight':")
    print(f"  Shape: {list(conv_weight.shape)}")
    print(f"  [0,0,0,0] = {conv_weight[0,0,0,0].item()}")
    print(f"  [1,2,2,1] = {conv_weight[1,2,2,1].item()}")
    
    print("\ntest_permutation.safetensors 'perm_test':")
    print(f"  Shape: {list(perm_test.shape)}")
    print(f"  After permutation [1,2,3,0] ([O,I,Ky,Kx] -> [I,Ky,Kx,O]):")
    print(f"    New shape: [3, 2, 2, 2]")
    print(f"    Value at new[i,ky,kx,o] should equal o*1000 + i*100 + ky*10 + kx")


if __name__ == "__main__":
    main()


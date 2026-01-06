"""
Dumps UNet model input and output for comparison with C++ implementation.
Creates reference files that test-unet-model.cpp can load and compare against.
"""

import torch
import numpy as np
from diffmodel import SimpleDiffusionUNet, DEVICE, IMG_SIZE, image_channels

def write_tensor_nhwc(filepath, tensor, name="tensor"):
    """
    Write a tensor to file in NHWC format (C++ convention).
    Input tensor is expected in NCHW format (PyTorch convention).
    """
    # Move to CPU and convert to numpy
    data = tensor.detach().cpu().numpy().astype(np.float32)
    
    # If 4D tensor (NCHW), permute to NHWC
    if len(data.shape) == 4:
        data = np.transpose(data, (0, 2, 3, 1))  # NCHW -> NHWC
        print(f"  {name}: NCHW {list(tensor.shape)} -> NHWC {list(data.shape)}")
    else:
        print(f"  {name}: shape {list(data.shape)}")
    
    # Ensure contiguous memory
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    data.tofile(filepath)
    print(f"  Saved to {filepath}")
    print(f"  Stats: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    return data


def main():
    print("=" * 60)
    print("UNet Reference Dumper")
    print("=" * 60)
    
    # Load the trained model
    print("\nLoading model from model_weights.pth...")
    model = SimpleDiffusionUNet().to(DEVICE)
    model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
    
    # Create a deterministic input (same as C++ uses)
    print("\nGenerating deterministic input...")
    torch.manual_seed(171)  # Same seed as C++ initImage
    input_image = torch.randn(1, image_channels, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Use a specific timestep
    timestep = 499  # First step in reverse diffusion (t=499 for step=49 in 50-step inference)
    timestep_tensor = torch.tensor([timestep], dtype=torch.long).to(DEVICE)
    
    print(f"\nInput image shape (NCHW): {list(input_image.shape)}")
    print(f"Timestep: {timestep}")
    
    # Run the model
    print("\nRunning UNet forward pass...")
    with torch.no_grad():
        output = model(input_image, timestep_tensor)
    
    print(f"Output shape (NCHW): {list(output.shape)}")
    
    # Save input and output in NHWC format for C++ comparison
    print("\n" + "-" * 40)
    print("Saving reference tensors (in NHWC format for C++):")
    print("-" * 40)
    
    write_tensor_nhwc("ref_input.bin", input_image, "Input")
    write_tensor_nhwc("ref_output.bin", output, "Output")
    
    # Also save timestep
    timestep_array = np.array([timestep], dtype=np.int32)
    timestep_array.tofile("ref_timestep.bin")
    print(f"\n  Timestep: {timestep} saved to ref_timestep.bin")
    
    # Save some intermediate values for deeper debugging
    print("\n" + "-" * 40)
    print("Computing intermediate values for debugging:")
    print("-" * 40)
    
    with torch.no_grad():
        # Time embedding - step by step for debugging
        # Step 1: Sinusoidal embedding (first layer of time_mlp)
        sinusoidal_emb = model.time_mlp[0](timestep_tensor)
        print(f"\nSinusoidal embedding shape: {list(sinusoidal_emb.shape)}")
        sinusoidal_np = sinusoidal_emb.detach().cpu().numpy().astype(np.float32)
        sinusoidal_np.tofile("ref_sinusoidal_embed.bin")
        print(f"  Stats: min={sinusoidal_np.min():.4f}, max={sinusoidal_np.max():.4f}, mean={sinusoidal_np.mean():.4f}")
        print(f"  Saved to ref_sinusoidal_embed.bin")
        
        # Step 2: After linear layer (before ReLU)
        after_linear = model.time_mlp[1](sinusoidal_emb)
        print(f"\nAfter linear (before ReLU) shape: {list(after_linear.shape)}")
        after_linear_np = after_linear.detach().cpu().numpy().astype(np.float32)
        after_linear_np.tofile("ref_time_embed_after_linear.bin")
        print(f"  Stats: min={after_linear_np.min():.4f}, max={after_linear_np.max():.4f}, mean={after_linear_np.mean():.4f}")
        print(f"  Saved to ref_time_embed_after_linear.bin")
        
        # Step 3: Full time_mlp output (after ReLU)
        t_emb = model.time_mlp(timestep_tensor)
        print(f"\nTime embedding (after ReLU) shape: {list(t_emb.shape)}")
        t_emb_np = t_emb.detach().cpu().numpy().astype(np.float32)
        t_emb_np.tofile("ref_time_embed.bin")
        print(f"  Stats: min={t_emb_np.min():.4f}, max={t_emb_np.max():.4f}, mean={t_emb_np.mean():.4f}")
        print(f"  Saved to ref_time_embed.bin")
        
        # Also dump the linear layer weights/biases for verification
        linear_layer = model.time_mlp[1]
        linear_weight = linear_layer.weight.detach().cpu().numpy().astype(np.float32)
        linear_bias = linear_layer.bias.detach().cpu().numpy().astype(np.float32)
        print(f"\nLinear layer weight shape: {list(linear_weight.shape)}")
        print(f"  Stats: min={linear_weight.min():.4f}, max={linear_weight.max():.4f}")
        linear_weight.tofile("ref_time_linear_weight.bin")
        print(f"Linear layer bias shape: {list(linear_bias.shape)}")
        print(f"  Stats: min={linear_bias.min():.4f}, max={linear_bias.max():.4f}")
        linear_bias.tofile("ref_time_linear_bias.bin")
        
        # Initial conv output
        x = model.conv0(input_image)
        write_tensor_nhwc("ref_after_conv0.bin", x, "After conv0")
        
        # Dump conv0 weights and biases
        conv0_weight = model.conv0.weight.detach().cpu().numpy().astype(np.float32)
        conv0_bias = model.conv0.bias.detach().cpu().numpy().astype(np.float32)
        print(f"\nConv0 weight shape (OIHW): {list(conv0_weight.shape)}")
        conv0_weight.tofile("ref_conv0_weight.bin")
        print(f"Conv0 bias shape: {list(conv0_bias.shape)}")
        conv0_bias.tofile("ref_conv0_bias.bin")
        
        # First downblock - step by step for debugging
        down0 = model.downs[0]
        
        # Step 1: conv1 + batchnorm + relu
        h = down0.relu(down0.bnorm1(down0.conv1(x)))
        write_tensor_nhwc("ref_down0_after_conv1.bin", h, "Down0 after conv1+bn+relu")
        
        # Step 2: time embedding linear projection + relu  
        time_proj = down0.relu(down0.time_mlp(t_emb))
        time_proj_np = time_proj.detach().cpu().numpy().astype(np.float32)
        time_proj_np.tofile("ref_down0_time_proj.bin")
        print(f"  Down0 time_proj shape: {list(time_proj.shape)}")
        print(f"  Stats: min={time_proj_np.min():.4f}, max={time_proj_np.max():.4f}")
        
        # Step 3: broadcast add (time_emb is [B, C], h is [B, H, W, C] in NHWC)
        # In PyTorch (NCHW): time_emb is [B, C, 1, 1], h is [B, C, H, W]
        time_emb_expanded = time_proj[(..., ) + (None, ) * 2]  # [B, C] -> [B, C, 1, 1]
        h_after_add = h + time_emb_expanded
        write_tensor_nhwc("ref_down0_after_add.bin", h_after_add, "Down0 after broadcast add")
        
        # Step 4: conv2 + batchnorm + relu
        h = down0.relu(down0.bnorm2(down0.conv2(h_after_add)))
        write_tensor_nhwc("ref_down0_after_conv2.bin", h, "Down0 after conv2+bn+relu")
        
        # Step 5: downsample transform
        x = down0.transform(h)
        write_tensor_nhwc("ref_after_down0.bin", x, "After down[0] (full block)")
        
        # Continue through all down blocks to get to up blocks
        for i in range(1, len(model.downs)):
            x = model.downs[i](x, t_emb)
        
        # Save state before first up block
        write_tensor_nhwc("ref_before_up0.bin", x, "Before up[0] (bottleneck)")
        print(f"  Bottleneck shape: {list(x.shape)}")
        
        # Save the skip connection for first up block (last residual = bottleneck output)
        # In PyTorch, residuals are: [down0_out, down1_out, down2_out, down3_out]
        # For up0, we concatenate x with down3_out (which equals x in this case!)
        residual_for_up0 = x  # Same as bottleneck
        
        # First up block step by step
        up0 = model.ups[0]
        
        # Concat (in PyTorch, channel dim is 1)
        x_concat = torch.cat((x, residual_for_up0), dim=1)
        write_tensor_nhwc("ref_up0_after_concat.bin", x_concat, "Up0 after concat")
        print(f"  After concat shape: {list(x_concat.shape)}")
        
        # up0 block
        h = up0.relu(up0.bnorm1(up0.conv1(x_concat)))
        write_tensor_nhwc("ref_up0_after_conv1.bin", h, "Up0 after conv1+bn+relu")
        
        time_proj = up0.relu(up0.time_mlp(t_emb))
        time_proj_np = time_proj.detach().cpu().numpy().astype(np.float32)
        time_proj_np.tofile("ref_up0_time_proj.bin")
        print(f"  Up0 time_proj shape: {list(time_proj.shape)}")
        
        time_emb_expanded = time_proj[(..., ) + (None, ) * 2]
        h_after_add = h + time_emb_expanded
        write_tensor_nhwc("ref_up0_after_add.bin", h_after_add, "Up0 after broadcast add")
        
        h = up0.relu(up0.bnorm2(up0.conv2(h_after_add)))
        write_tensor_nhwc("ref_up0_after_conv2.bin", h, "Up0 after conv2+bn+relu")
        
        x = up0.transform(h)
        write_tensor_nhwc("ref_after_up0.bin", x, "After up[0] (full block)")
    
    print("\n" + "=" * 60)
    print("Reference files created successfully!")
    print("Now run test-unet-model.cpp to compare with C++ output.")
    print("=" * 60)


if __name__ == "__main__":
    main()


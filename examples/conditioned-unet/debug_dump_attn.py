import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffmodel import ConditionalUNet

def write_bin(tensor, name):
    # Ensure we detach, move to CPU, and cast to float32
    data = tensor.detach().cpu().numpy().astype(np.float32)
    
    # Sanity check: verify the shape is what we expect for C++ (NHWC)
    # For a 1x1x32x32 input, NCHW=[1,1,32,32], NHWC=[1,32,32,1]
    # For Attn (downsampled): NCHW=[1,256,8,8], NHWC=[1,8,8,256]
    print(f"  -> Writing {name} | Shape: {data.shape} | First val: {data.flatten()[0]:.4f}")
    
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    with open(name, "wb") as f:
        data.tofile(f)

# Global storage
captured_data = {}

def hook_fn(module, args, output):
    # args[0]: Input X (NCHW in PyTorch)
    # args[1]: Context (Batch, Seq, Dim) - Usually already compatible
    # output: Result (NCHW in PyTorch)
    
    x_nchw = args[0]
    context = args[1]
    out_nchw = output
    
    print(f">> Hook Triggered! Input Shape: {x_nchw.shape}")

    # 1. Permute Input X: NCHW -> NHWC (0, 2, 3, 1)
    # This matches the memory layout the C++ Conv2D blocks produce.
    captured_data['input_x_nhwc'] = x_nchw.permute(0, 2, 3, 1)
    
    # 2. Context: [B, 1, 128]. No permutation needed (Batch, Seq, Feature is standard)
    captured_data['context'] = context
    
    # 3. Permute Output: NCHW -> NHWC (0, 2, 3, 1)
    # This allows direct comparison with the C++ Kernel output.
    captured_data['output_nhwc'] = out_nchw.permute(0, 2, 3, 1)

def main():
    device = "cpu"
    print("--- Generating CrossAttention Debug Data (NHWC Layout) ---")

    # 1. Load Model
    model = ConditionalUNet().to(device)
    weights_path = "model_weights_conditioned.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Run train.py first!")
        return
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Register Hook
    handle = model.mid_attn.register_forward_hook(hook_fn)

    # 3. Prepare Inputs
    torch.manual_seed(42)
    x = torch.randn(1, 1, 32, 32).to(device)
    t = torch.tensor([499]).long().to(device)
    label = torch.tensor([7]).long().to(device)

    # 4. Run Inference
    with torch.no_grad():
        model(x, t, label)

    handle.remove()

    # 5. Save Data
    if 'input_x_nhwc' in captured_data:
        write_bin(captured_data['input_x_nhwc'], "debug_dump/debug_attn_in_x_nhwc.bin")
        write_bin(captured_data['context'],      "debug_dump/debug_attn_in_context.bin")
        write_bin(captured_data['output_nhwc'],  "debug_dump/debug_attn_out_nhwc.bin")
        print("Done. Files saved for C++ testing.")
    else:
        print("Error: Hook was not triggered.")

if __name__ == "__main__":
    main()
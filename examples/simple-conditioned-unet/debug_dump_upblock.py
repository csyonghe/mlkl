import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffmodel import ConditionalUNet

def write_bin(tensor, name):
    # Permute 4D tensors NCHW -> NHWC
    if len(tensor.shape) == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    
    data = tensor.detach().cpu().numpy().astype(np.float32)
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    
    with open(name, "wb") as f:
        data.tofile(f)
    print(f"Saved {name} \t| Shape: {data.shape}")

# Storage for captured data
captured = {}

def hook_fn(module, args, output):
    # args is tuple: (x, t)
    # x: Spatial Input (Concatenated) [B, C, H, W]
    # t: Time Embedding [B, TimeDim]
    captured['input_x'] = args[0]
    captured['input_t'] = args[1]
    captured['output'] = output
    print(f">> Hook Triggered on {module.__class__.__name__}")

def main():
    device = "cpu"
    print("--- Generating UpBlock[0] Debug Data ---")

    # 1. Load Model
    model = ConditionalUNet().to(device)
    weights_path = "model_weights_conditioned.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Register Hook on first Up Block
    # model.ups is the ModuleList of up-blocks
    model.ups[0].register_forward_hook(hook_fn)

    # 3. Fixed Inputs
    torch.manual_seed(42)
    x = torch.randn(1, 1, 32, 32).to(device)
    t = torch.tensor([499]).long().to(device)
    label = torch.tensor([7]).long().to(device)

    # 4. Run Model
    with torch.no_grad():
        model(x, t, label)

    # 5. Save Data
    if 'input_x' in captured:
        # Input X (Concatenated tensor: 256+256=512 channels)
        write_bin(captured['input_x'], "debug_dump/debug_up0_in_x.bin")
        # Input T (Time Embedding Vector: 32 dim)
        write_bin(captured['input_t'], "debug_dump/debug_up0_in_t.bin")
        # Output (Upsampled tensor: 16x16, 128 channels)
        write_bin(captured['output'],  "debug_dump/debug_up0_out.bin")
    else:
        print("Error: Hook not triggered!")

if __name__ == "__main__":
    main()
import torch
import numpy as np
import os
import sys

# Ensure we can import diffmodel from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffmodel import ConditionalUNet

def write_bin(tensor, name):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    # Ensure contiguous C-order memory for binary compatibility
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    with open(name, "wb") as f:
        data.tofile(f)
    print(f"Saved {name} | Shape: {data.shape} | First val: {data.flatten()[0]:.4f}")

def main():
    device = "cpu" # Debug on CPU to avoid potential CUDA nondeterminism issues
    
    print("--- Generating Debug Data for C++ Verification ---")

    # 1. Load Model Structure
    model = ConditionalUNet().to(device)
    
    # 2. Load Weights (Must match the .bin file used in C++)
    weights_path = "model_weights_conditioned.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"ERROR: {weights_path} not found. Run train.py first!")
        return

    model.eval()

    # 3. Create Fixed Inputs
    # We use specific values to ensure the test is deterministic
    torch.manual_seed(42)
    
    # Batch=1, Channel=1, Size=32
    x = torch.randn(1, 1, 32, 32).to(device)
    
    # Time step 499 (The last step of a 500-step schedule)
    # C++ 'timeStep' corresponds to the index in the schedule
    t_val = 499
    t = torch.tensor([t_val]).long().to(device)
    
    # Label 7
    label_val = 7
    label = torch.tensor([label_val]).long().to(device)

    print(f"Running Inference with: t={t_val}, label={label_val}")

    # 4. Run Model
    with torch.no_grad():
        eps = model(x, t, label)

    # 5. Dump Inputs and Outputs to Binary
    write_bin(x, "debug_dump/debug_input_x.bin")
    write_bin(eps, "debug_dump/debug_output_eps.bin")
    
    print("Done. Move these .bin files to your C++ working directory if needed.")

if __name__ == "__main__":
    main()
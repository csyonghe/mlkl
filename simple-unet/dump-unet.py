import torch
import os
from diffmodel import *;
import numpy as np

def save_nhwc(tensor, filename):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    # [Batch, Channel, Height, Width] -> [Batch, Height, Width, Channel]
    if len(data.shape) == 4:
        data = data.transpose(0, 2, 3, 1)
    data = np.ascontiguousarray(data)
    data.tofile(filename)
    print(f"Saved {filename} | Shape: {data.shape}")

# 1. Load Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleDiffusionUNet().to(DEVICE)
# Checkpoint must match the one used in C++!
model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))
model.eval()

# 2. Setup Inputs for Step 495
# Use a fixed seed so this matches every time you run it
torch.manual_seed(42)

# Input Image (Pure Noise at start of inference)
img_size = 32
x_in = torch.randn(1, 1, img_size, img_size).to(DEVICE)

# Timestep
t_val = 495
t_in = torch.tensor([t_val]).long().to(DEVICE)

# 3. Run Model
print(f"--- Running Inference for t={t_val} ---")
with torch.no_grad():
    predicted_noise = model(x_in, t_in)
# 4. Save Dumps
os.makedirs("debug_dump", exist_ok=True)
save_nhwc(x_in, "debug_dump/step495_input.bin")

# --- CAPTURE BOTTLENECK SAFELY ---
# We use a dictionary to store the output from the hook
captured_tensors = {}

def get_capture_hook(name):
    def hook(model, input, output):
        captured_tensors[name] = output
    return hook

# Register hook on the last DownBlock to grab the bottleneck features
# This corresponds to the output of the Encoder
model.downs[-1].register_forward_hook(get_capture_hook("bottleneck_out"))

# 5. Run Model (End-to-End)
print(f"--- Running Inference for t={t_val} ---")
with torch.no_grad():
    # This runs the FULL model, triggering the hook automatically
    predicted_noise = model(x_in, t_in)

# 6. Save Outputs
save_nhwc(predicted_noise, "debug_dump/step495_output.bin")

if "bottleneck_out" in captured_tensors:
    # This allows you to verify if the Encoder half is perfect
    save_nhwc(captured_tensors["bottleneck_out"], "debug_dump/step495_bottleneck_out.bin")
    print("Saved debug_dump/step495_bottleneck_out.bin")
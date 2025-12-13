from diffmodel import *;
import os
import numpy as np
import time

# load model
model = SimpleDiffusionUNet().to(DEVICE)
model.load_state_dict(torch.load("model_weights.pth", map_location=DEVICE))

diffusion = DiffusionManager(noise_steps=NOISE_STEPS, img_size=IMG_SIZE, device=DEVICE)
# get the current system time to benchmark the inferencing performance
startTime = time.time()
result = diffusion.sample(model, n=1)
endTime = time.time()
print(f"Sampling Time: {endTime - startTime} seconds")

# Create debug folder
os.makedirs("debug_dump", exist_ok=True)

def save_debug_tensor(tensor, filename):
    # 1. Convert to NumPy
    data = tensor.detach().cpu().numpy().astype(np.float32)
    
    # 2. Handle Layouts for C++ Comparison
    if len(data.shape) == 4:
        # Image: [Batch, Channel, Height, Width] -> [Batch, Height, Width, Channel] (NHWC)
        # This matches your C++ "Pixel-by-Pixel" memory layout
        data = data.transpose(0, 2, 3, 1)
        
    # 3. Save
    # Ensure contiguous C-order before dumping raw bytes
    data = np.ascontiguousarray(data)
    data.tofile(f"debug_dump/{filename}.bin")
    print(f"Saved {filename:<40} | Shape: {data.shape}")

# --- HOOK FUNCTION ---
# This runs automatically every time a layer finishes
def get_activation_hook(layer_name):
    def hook(model, input, output):
        # input is a tuple (x,), we want input[0]
        save_debug_tensor(input[0], f"{layer_name}_input")
        save_debug_tensor(output,   f"{layer_name}_output")
    return hook

def get_relu_dump_hook(filename):
    def hook(model, input, output):
        # 'output' here is the result of BatchNorm(Conv(x))
        # We manually apply ReLU to match the 'Fused' C++ expectation
        activated = torch.relu(output)
        save_debug_tensor(activated, filename)
    return hook

# --- SETUP DEBUGGING ---
# 1. Register hooks on the layers you want to inspect
# You can add more layers here by looking at model.named_modules()
model.conv0.register_forward_hook(get_activation_hook("conv0"))
for i in range(0, 4):
    # Example: Inspecting the very first DownBlock's components
    # Block 0 -> Conv1
    model.downs[i].conv1.register_forward_hook(get_activation_hook(f"down{i}_conv1"))
    # Block 0 -> Time Embedding Linear Projection
    model.downs[i].time_mlp.register_forward_hook(get_relu_dump_hook(f"down{i}_time_proj_output"))
    # Block 0 -> Conv2
    model.downs[i].conv2.register_forward_hook(get_activation_hook(f"down{i}_conv2"))
    # Block 0 -> Downsample Transform
    model.downs[i].transform.register_forward_hook(get_activation_hook(f"down{i}_transform"))
    # Register on bnorm1, because: Output = ReLU(BN(Conv(x)))
    # This captures the state exactly where your C++ Conv2D kernel finishes.
    model.downs[i].bnorm1.register_forward_hook(get_relu_dump_hook(f"down{i}_conv1_fused_output"))

# --- DUMP UP-BLOCK 0 (Bottleneck/First Decoder Step) ---

# 1. Hook the INPUT to UpBlock0's first convolution
#    This verifies if your C++ Concatenation logic is correct.
def get_input_dump_hook(filename):
    def hook(model, input, output):
        # Input is a tuple (x, t), we want x
        save_debug_tensor(input[0], filename)
    return hook

# Register hook on the first layer of UpBlock 0
# The input to this layer IS the result of the concatenation
model.ups[0].conv1.register_forward_hook(get_input_dump_hook("up0_concat_output"))

# 2. Hook the OUTPUT of UpBlock0
#    Verifies the Conv -> BN -> ReLU -> TransposedConv chain
model.ups[0].transform.register_forward_hook(get_activation_hook("up0_output"))


# --- PREPARE INPUTS ---
# 2. Create a deterministic start state
torch.manual_seed(42)
# Create input X (Standard Normal)
x_debug = torch.randn(1, image_channels, IMG_SIZE, IMG_SIZE).to(DEVICE)
save_debug_tensor(x_debug, "initial_x_input")

# Create Timestep t (e.g., Step 495 roughly matches inference step 0 in a 100-step loop)
t_val = 495 
t_debug = torch.tensor([t_val]).long().to(DEVICE)
save_debug_tensor(model.time_mlp(t_debug), "global_time_embed_output")

# --- RUN ONE STEP ---
print("\n--- Running Single Debug Step ---")
model.eval()
with torch.no_grad():
    # This triggers all the hooks and saves files
    predicted_noise = model(x_debug, t_debug)

print("\nDebug dump complete in folder 'debug_dump/'")

# --- PARAMETER DUMPING HELPER ---
def dump_layer_params_grouped(layer, filename):
    print(f"Dumping Grouped Params to: {filename}")
    parts = []
    
    # --- 1. WEIGHTS ---
    if hasattr(layer, 'weight') and layer.weight is not None:
        w = layer.weight.detach()
        
        # Apply the layout optimization permutations
        # Resulting Layout: Row-Major / Contiguous C-Order
        if isinstance(layer, nn.Conv2d):
            # [Out, In, K, K] -> [In, K, K, Out]
            w = w.permute(1, 2, 3, 0)
            
        elif isinstance(layer, nn.ConvTranspose2d):
            # [In, Out, K, K] -> [In, K, K, Out]
            w = w.permute(0, 2, 3, 1)
            
        elif isinstance(layer, nn.Linear):
            # [Out, In] -> [In, Out]
            w = w.t()
            
        # Ensure float32 and flatten to 1D array
        w_flat = w.contiguous().cpu().numpy().astype(np.float32).flatten()
        parts.append(w_flat)
        print(f"  + Weights: {w_flat.size} elements")

    # --- 2. BIASES ---
    if hasattr(layer, 'bias') and layer.bias is not None:
        b_flat = layer.bias.detach().cpu().numpy().astype(np.float32).flatten()
        parts.append(b_flat)
        print(f"  + Bias:    {b_flat.size} elements")

    # --- 3. BATCHNORM STATS ---
    if isinstance(layer, nn.BatchNorm2d):
        # Order: Running Mean, Running Var
        # (Note: Weight/Bias were handled above as learnable params)
        mean = layer.running_mean.detach().cpu().numpy().astype(np.float32).flatten()
        var  = layer.running_var.detach().cpu().numpy().astype(np.float32).flatten()
        parts.append(mean)
        parts.append(var)
        print(f"  + Stats:   {mean.size} (Mean) + {var.size} (Var)")

    # --- 4. CONCATENATE & SAVE ---
    if len(parts) > 0:
        combined = np.concatenate(parts)
        combined.tofile(filename)
        print(f"  -> Saved {len(combined)} floats to {filename}")
    else:
        print(f"  [Warning] No parameters found for {filename}")

# --- USAGE EXAMPLE ---
os.makedirs("debug_dump", exist_ok=True)

# Dump specific layers for C++ Unit Tests
dump_layer_params_grouped(model.time_mlp[1], "debug_dump/global_time_linear1.bin")
dump_layer_params_grouped(model.conv0,       "debug_dump/conv0.bin")

# Dump Block 0 components
block0 = model.downs[0]
dump_layer_params_grouped(block0.conv1,      "debug_dump/down0_conv1.bin")
dump_layer_params_grouped(block0.bnorm1,     "debug_dump/down0_bn1.bin")
dump_layer_params_grouped(block0.time_mlp,   "debug_dump/down0_time_proj.bin")
dump_layer_params_grouped(block0.conv2,      "debug_dump/down0_conv2.bin")
dump_layer_params_grouped(block0.bnorm2,     "debug_dump/down0_bn2.bin")
dump_layer_params_grouped(block0.transform,  "debug_dump/down0_transform.bin")

# 3. Dump the Parameters for UpBlock0
blockUp0 = model.ups[0]
dump_layer_params_grouped(blockUp0.conv1,      "debug_dump/up0_conv1.bin")
dump_layer_params_grouped(blockUp0.bnorm1,     "debug_dump/up0_bn1.bin")
dump_layer_params_grouped(blockUp0.time_mlp,   "debug_dump/up0_time_proj.bin")
dump_layer_params_grouped(blockUp0.conv2,      "debug_dump/up0_conv2.bin")
dump_layer_params_grouped(blockUp0.bnorm2,     "debug_dump/up0_bn2.bin")
dump_layer_params_grouped(blockUp0.transform,  "debug_dump/up0_transform.bin")

print("\nParameter dump complete.")
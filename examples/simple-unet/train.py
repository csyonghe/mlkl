from diffmodel import *;

## Training

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Save to disk
import numpy as np
def write_tensor(f, tensor, name, info=""):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    
    # Ensure contiguous memory before writing (crucial for permuted tensors)
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
        
    data.tofile(f)
    print(f"   -> Wrote {name: <30} | Shape: {str(list(data.shape)): <20} | {info}")

def export_model_weights(model, filepath="model_weights.bin"):
    print(f"--- Exporting to {filepath} ---")
    
    with open(filepath, "wb") as f:
        # We iterate over named_modules to handle specific layer types contextually
        for name, module in model.named_modules():
            
            # Skip the top-level container (it has no name)
            if name == "": continue
            
            # --- 1. Standard Convolution ---
            if isinstance(module, nn.Conv2d):
                print(f"[Layer] {name} (Conv2d)")
                
                # Original: [Out, In, K, K]
                # Target:   [In, K, K, Out]
                # Permute:  (1, 2, 3, 0)
                w = module.weight
                w_permuted = w.permute(1, 2, 3, 0)
                
                write_tensor(f, w_permuted, "Weight", f"Permuted {list(w.shape)} -> {list(w_permuted.shape)}")
                
                if module.bias is not None:
                    write_tensor(f, module.bias, "Bias")

            # --- 2. Transposed Convolution ---
            elif isinstance(module, nn.ConvTranspose2d):
                print(f"[Layer] {name} (ConvTranspose2d)")
                
                # Original: [In, Out, K, K]
                # Target:   [In, K, K, Out]
                # Permute:  (0, 2, 3, 1)
                w = module.weight
                w_permuted = w.permute(0, 2, 3, 1)
                
                write_tensor(f, w_permuted, "Weight", f"Permuted {list(w.shape)} -> {list(w_permuted.shape)}")
                
                if module.bias is not None:
                    write_tensor(f, module.bias, "Bias")

            # --- 3. Linear (Dense) ---
            elif isinstance(module, nn.Linear):
                print(f"[Layer] {name} (Linear)")
                
                # Transpose [Out, In] -> [In, Out]
                w = module.weight
                w_transposed = w.t()
                
                write_tensor(f, w_transposed, "Weight", f"TRANSPOSED from {list(w.shape)}")
                
                if module.bias is not None:
                    write_tensor(f, module.bias, "Bias")

            # --- 4. Batch Normalization ---
            elif isinstance(module, nn.BatchNorm2d):
                print(f"[Layer] {name} (BatchNorm2d)")
                # Export all 4 statistics in a fixed order
                write_tensor(f, module.weight, "Weight (Gamma)")
                write_tensor(f, module.bias, "Bias (Beta)")
                write_tensor(f, module.running_mean, "Running Mean")
                write_tensor(f, module.running_var, "Running Var")

            # --- 5. Containers & Activations (Safe to Skip) ---
            # We explicitly list things we expect to skip to keep the logic clean.
            # SinusoidalPositionEmbeddings has no weights, so it falls here.
            elif isinstance(module, (nn.Sequential, nn.ModuleList, nn.ReLU, nn.SiLU, nn.Identity)):
                continue

            # --- 6. CATCH ALL / ERROR CHECK ---
            else:
                # Some custom classes (like Block, SimpleDiffusionUNet) act as containers.
                # We check if they have *direct* parameters.
                # recurse=False ensures we only check weights owned *directly* by this object,
                # not by its children (which we handle in the loop iterations above).
                
                direct_params = list(module.parameters(recurse=False))
                
                if len(direct_params) > 0:
                    # CRITICAL ERROR: We found a layer with weights that we didn't write!
                    raise ValueError(f"\n[ERROR] Unhandled Layer Type with parameters!\n"
                                     f"Layer: {name}\n"
                                     f"Type:  {type(module)}\n"
                                     f"Params found: {len(direct_params)}\n"
                                     f"Please add an 'elif' block to handle this layer type.")
                else:
                    # It's just a container (like 'Block'), safe to skip logging
                    pass

    print("--- Export Complete ---")

def trainModel():  
    # --- DATA SETUP ---
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Normalize to [-1, 1]
    ])
    # We use MNIST because it's built-in and fast
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # --- INIT MODELS ---
    # Note: You need to modify the U-Net code I gave you earlier
    # to accept "in_channels=1" (MNIST is grayscale) or change the dataset to CIFAR10.
    # For now, let's assume we use the U-Net from before but update the input conv to 1 channel.
    model = SimpleDiffusionUNet().to(DEVICE) 
    # HACK: If using the previous code, change self.conv0 = nn.Conv2d(1, ...) inside the class.

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    diffusion = DiffusionManager(noise_steps=NOISE_STEPS, img_size=IMG_SIZE, device=DEVICE)

    # --- TRAINING LOOP ---
    print("Starting training...")
    for epoch in range(3): # Train for 3 epochs
        for i, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # 1. Sample random timestamps
            t = torch.randint(low=1, high=NOISE_STEPS, size=(images.shape[0],)).to(DEVICE)
            
            # 2. Add noise to images
            x_t, noise = diffusion.noise_images(images, t)
            
            # 3. Predict noise
            predicted_noise = model(x_t, t)
            
            # 4. Calculate Loss & Backprop
            loss = loss_fn(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item()}")
    export_model_weights(model)
    torch.save(model.state_dict(), "model_weights.pth")
    
    # --- GENERATE A PREVIEW ---
    gen_imgs = diffusion.sample(model, n=1)

    # Plot
    plt.imshow(gen_imgs[0].permute(1, 2, 0).cpu().numpy().astype('uint8'), cmap="gray")
    plt.show()

trainModel()

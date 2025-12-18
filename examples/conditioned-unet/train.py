from diffmodel import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Lower noise steps/img size for faster local testing if needed
NOISE_STEPS = 500 
IMG_SIZE = 32
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 5

# --- HELPER FUNCTIONS ---

def write_tensor(f, tensor, name, info=""):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    # Ensure contiguous memory
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    data.tofile(f)
    print(f"   -> Wrote {name: <30} | Shape: {str(list(data.shape)): <20} | {info}")

def export_model_weights(model, filepath="model_weights_conditioned.bin"):
    print(f"--- Exporting to {filepath} ---")
    with open(filepath, "wb") as f:
        for name, module in model.named_modules():
            if name == "": continue
            
            # 1. Conv2d -> [In, K, K, Out] (Permute 1,2,3,0)
            if isinstance(module, nn.Conv2d):
                print(f"[Layer] {name} (Conv2d)")
                w = module.weight
                w_permuted = w.permute(1, 2, 3, 0) # [Out, In, K, K] -> [In, K, K, Out]
                write_tensor(f, w_permuted, "Weight", f"Permuted {list(w.shape)} -> {list(w_permuted.shape)}")
                if module.bias is not None:
                    write_tensor(f, module.bias, "Bias")

            # 2. ConvTranspose2d -> [In, K, K, Out] (Permute 0,2,3,1)
            elif isinstance(module, nn.ConvTranspose2d):
                print(f"[Layer] {name} (ConvTranspose2d)")
                w = module.weight
                w_permuted = w.permute(0, 2, 3, 1) # [In, Out, K, K] -> [In, K, K, Out]
                write_tensor(f, w_permuted, "Weight", f"Permuted {list(w.shape)} -> {list(w_permuted.shape)}")
                if module.bias is not None:
                    write_tensor(f, module.bias, "Bias")

            # 3. Linear -> [In, Out] (Transpose)
            # This handles the CrossAttention projections (to_q, to_k, etc.)
            elif isinstance(module, nn.Linear):
                print(f"[Layer] {name} (Linear)")
                w = module.weight
                w_transposed = w.t() # [Out, In] -> [In, Out]
                write_tensor(f, w_transposed, "Weight", f"Transposed {list(w.shape)} -> {list(w_transposed.shape)}")
                if module.bias is not None:
                    write_tensor(f, module.bias, "Bias")

            # 4. Embedding -> [Num, Dim] (Keep as is)
            elif isinstance(module, nn.Embedding):
                print(f"[Layer] {name} (Embedding)")
                # Embeddings are typically [NumEmbeddings, EmbeddingDim]
                # We write them as-is (Row Major)
                write_tensor(f, module.weight, "Weight", f"Raw {list(module.weight.shape)}")

            # 5. BatchNorm -> Gamma, Beta, Mean, Var
            elif isinstance(module, nn.BatchNorm2d):
                print(f"[Layer] {name} (BatchNorm2d)")
                write_tensor(f, module.weight, "Weight (Gamma)")
                write_tensor(f, module.bias, "Bias (Beta)")
                write_tensor(f, module.running_mean, "Running Mean")
                write_tensor(f, module.running_var, "Running Var")

            # 6. Skip containers
            elif isinstance(module, (nn.Sequential, nn.ModuleList, nn.ReLU, nn.SiLU, nn.Identity, 
                                     SinusoidalPositionEmbeddings, CrossAttention, ConditionalUNet, Block)):
                continue

            else:
                # Safety check for unhandled parameters
                direct_params = list(module.parameters(recurse=False))
                if len(direct_params) > 0:
                    print(f"[WARNING] Layer {name} of type {type(module)} has parameters but is not handled!")

    print("--- Export Complete ---")

# --- DIFFUSION MANAGER (Conditioned) ---
class DiffusionManager:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample(self, model, n, labels=None):
        print(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            
            # If no labels provided, generate random digits 0-9
            if labels is None:
                labels = torch.randint(0, 10, (n,)).to(self.device)
            else:
                labels = labels.to(self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Conditioned Forward Pass
                predicted_noise = model(x, t, labels)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x * 255, labels

# --- MAIN ---
def trainModel():
    # 1. Data
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model
    model = ConditionalUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    diffusion = DiffusionManager(noise_steps=NOISE_STEPS, img_size=IMG_SIZE, device=DEVICE)

    # 3. Train
    print("Starting training...")
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            t = torch.randint(low=1, high=NOISE_STEPS, size=(images.shape[0],)).to(DEVICE)
            x_t, noise = diffusion.noise_images(images, t)
            
            # Pass labels to model
            predicted_noise = model(x_t, t, labels)
            
            loss = loss_fn(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item()}")

    # 4. Export
    export_model_weights(model, "model_weights_conditioned.bin")
    torch.save(model.state_dict(), "model_weights_conditioned.pth")
    
    # 5. Preview (Generate one of each digit)
    test_labels = torch.arange(10) # 0, 1, ... 9
    gen_imgs, _ = diffusion.sample(model, n=10, labels=test_labels)

    # Plot
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(gen_imgs[i].permute(1, 2, 0).cpu().numpy().astype('uint8'), cmap="gray")
        axes[i].set_title(f"Digit {test_labels[i].item()}")
        axes[i].axis('off')
    plt.show()

if __name__ == '__main__':
    trainModel()
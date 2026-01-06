# This file defines a simple U-NET architecture to be used as the
# noise predictor in a diffusion model.

# See train.py for training code and sampling code.

# simple-unet.cpp contains the native C++ implementation of the same model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_STEPS = 500  # Lowered from 1000 for speed
IMG_SIZE = 32      # Lowered from 64 for speed
image_channels = 1


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(out_ch)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(out_ch)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.relu(self.bnorm1(self.conv1(x)))
        # Time Embedding Injection
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions so we can add to the image (Broadcasting)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.relu(self.bnorm2(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)
    
class SimpleDiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time Embedding Encoding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                                    for i in range(len(down_channels)-1)])
        
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                                        for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embed time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Residual connections cache
        residual_inputs = []

        # Downsampling (Encoder)
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        # Upsampling (Decoder)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels (Concatenation)
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)

        return self.output(x)

class DiffusionManager:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define the Beta Schedule (Linear)
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        # Alpha Bar (Cumulative Product) used for forward diffusion
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # FORWARD: Add noise to x_0 to get x_t
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    # REVERSE: Sample from noise to get x_0
    def sample(self, model, n):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # 1. Start with pure noise
            x = torch.randn((n, image_channels, self.img_size, self.img_size)).to(self.device)
            
            # 2. Loop backward from T to 0
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Predict noise
                predicted_noise = model(x, t)
                
                # Math for the reverse step (Langevin Dynamics)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        # Clamp pixels to valid image range [-1, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        return x * 255

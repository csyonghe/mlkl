import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal positional embedding for time.
    Same as your unconditional model.
    """
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

class CrossAttention(nn.Module):
    """
    Simple Cross-Attention Layer.
    This allows the image features (x) to attend to the context (class embedding).
    """
    def __init__(self, channel_dim, context_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channel_dim // num_heads) ** -0.5
        
        # Projections for Query (Image), Key (Context), and Value (Context)
        self.to_q = nn.Linear(channel_dim, channel_dim, bias=False)
        self.to_k = nn.Linear(context_dim, channel_dim, bias=False)
        self.to_v = nn.Linear(context_dim, channel_dim, bias=False)
        
        self.to_out = nn.Linear(channel_dim, channel_dim)

    def forward(self, x, context):
        """
        x: [Batch, Channels, Height, Width] (Image features)
        context: [Batch, Sequence_Length, Context_Dim] (Class/Text embedding)
        """
        b, c, h, w = x.shape
        
        # 1. Reshape image to sequence: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        
        # 2. Project Q, K, V
        # Q comes from the Image
        q = self.to_q(x_flat)  # [B, H*W, C]
        
        # K, V come from the Context (The digit embedding)
        k = self.to_k(context) # [B, 1, C]
        v = self.to_v(context) # [B, 1, C]

        # 3. Attention Score: Q @ K^T
        # [B, H*W, C] @ [B, C, 1] -> [B, H*W, 1]
        dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = dots.softmax(dim=-1)

        # 4. Weighted Sum: Score @ V
        # [B, H*W, 1] @ [B, 1, C] -> [B, H*W, C]
        out = torch.bmm(attn, v)
        
        # 5. Output projection and reshape back to image
        out = self.to_out(out)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        
        # Residual connection + Original Image features
        return x + out

class Block(nn.Module):
    """
    A standard ResNet-like block: Conv -> BN -> ReLU
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time Embedding Injection
        time_emb = self.relu(self.time_mlp(t))
        # Extend time_emb to match spatial dims [B, C, 1, 1]
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time embedding
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Downsample or Upsample
        return self.transform(h)

class ConditionalUNet(nn.Module):
    """
    The main U-Net architecture with Class Conditioning.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256)
        up_channels = (256, 128, 64)
        out_dim = 1
        time_emb_dim = 32
        
        # Number of classes (0-9 for MNIST) + 1 for null/unconditional
        self.num_classes = 10 
        self.context_dim = 128 # Dimension of the class embedding

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Class Embedding (The "Context")
        self.class_emb = nn.Embedding(self.num_classes, self.context_dim)

        # Initial Projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample Path
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim) \
            for i in range(len(down_channels)-1)
        ])
        
        # Bottleneck Cross-Attention
        # We place attention here at the lowest resolution (smallest spatial dim)
        # This is the most efficient place for attention.
        self.mid_attn = CrossAttention(down_channels[-1], self.context_dim)

        # Upsample Path
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, t, class_labels):
        """
        x: [B, 1, 32, 32] - Noisy Image
        t: [B] - Timestep
        class_labels: [B] - Int labels (0-9)
        """
        # Embed Time
        t = self.time_mlp(t)
        
        # Embed Context (Classes)
        # [B] -> [B, Context_Dim] -> [B, 1, Context_Dim] to act as sequence length 1
        context = self.class_emb(class_labels).unsqueeze(1) 

        # Initial Conv
        x = self.conv0(x)
        
        # Save residuals for skip connections
        residuals = []
        
        # Downsampling
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
            
        # Bottleneck Attention
        # We apply attention to the feature map 'x' using 'context'
        x = self.mid_attn(x, context)

        # Upsampling
        for up in self.ups:
            residual = residuals.pop()
            # Concatenate skip connection
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.output(x)

if __name__ == '__main__':
    # Simple test to verify dimensions
    model = ConditionalUNet()
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 100, (2,))
    labels = torch.randint(0, 10, (2,)) # Random digits 0-9
    
    out = model(x, t, labels)
    print(f"Input shape: {x.shape}")
    print(f"Context labels: {labels}")
    print(f"Output shape: {out.shape}")
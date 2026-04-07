import torch
import torch.nn as nn

class MediaLatentSpace(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MediaLatentSpace, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class MediaRefiningGate(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(MediaRefiningGate, self).__init__()
        # FIX: Project latent_dim (256) to output_dim (512) so they can be added
        self.projection = nn.Linear(latent_dim, output_dim)
        self.gate = nn.GRU(output_dim, output_dim, batch_first=True)
        self.style_embedding = nn.Parameter(torch.randn(1, 1, output_dim))

    def forward(self, latent_vec, omega):
        batch_size = latent_vec.size(0)
        
        # Align dimensions: (Batch, 256) -> (Batch, 512)
        projected_latent = self.projection(latent_vec).unsqueeze(1)
        style = self.style_embedding.repeat(batch_size, 1, 1)
        
        # Now both are (Batch, 1, 512), so we can fuse them
        fused = (omega * projected_latent) + ((1 - omega) * style)
        output, _ = self.gate(fused)
        return output.squeeze(1)

class OmniMediaTransformer(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=256):
        super(OmniMediaTransformer, self).__init__()
        self.latent_space = MediaLatentSpace(input_dim, latent_dim)
        
        # Defining heads with consistent output dimensions
        self.journalism_head = MediaRefiningGate(latent_dim, 512)
        self.fiction_head = MediaRefiningGate(latent_dim, 512)
        self.immersive_3d_head = MediaRefiningGate(latent_dim, 1024)

    def forward(self, x, media_type="journalism", verifiability=0.5):
        latent = self.latent_space(x)
        
        # Ensure omega is a tensor on the same device as the input
        omega = torch.tensor([verifiability], device=x.device)
        
        if media_type == "journalism":
            return self.journalism_head(latent, omega)
        elif media_type == "fiction":
            return self.fiction_head(latent, torch.tensor([0.2], device=x.device))
        elif media_type == "3D":
            return self.immersive_3d_head(latent, torch.tensor([0.9], device=x.device))

if __name__ == "__main__":
    model = OmniMediaTransformer()
    dummy_ugc = torch.randn(1, 1024) 
    
    # This will now run without the dimension mismatch error
    news_output = model(dummy_ugc, media_type="journalism", verifiability=0.9)
    print(f"Success! Journalism Output Shape: {news_output.shape}")

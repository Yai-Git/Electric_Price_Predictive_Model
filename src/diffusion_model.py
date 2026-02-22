import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset

# =============================================================================
# DATASET CLASS
# =============================================================================
class ElectricityPriceDataset(Dataset):
    """
    PyTorch Dataset for electricity price prediction with diffusion model.
    """
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# =============================================================================
# TIMESTEP EMBEDDING
# =============================================================================
def get_timestep_embedding(timesteps, embedding_dim, device='cpu'):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1D tensor of timesteps
        embedding_dim: Dimension of the embedding
        device: Device to create tensors on
    
    Returns:
        Tensor of shape (len(timesteps), embedding_dim)
    """
    assert embedding_dim % 2 == 0, "Embedding dimension must be even"
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    
    # Ensure timesteps is float for matmul
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    return emb

# =============================================================================
# DIFFUSION SCHEDULE
# =============================================================================
class DiffusionSchedule:
    """
    Precompute diffusion schedule parameters with LINEAR or COSINE schedule.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 device='cpu', schedule_type='cosine'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            # Cosine schedule from Nichol & Dhariwal (2021)
            s = 0.008
            t = torch.linspace(0, num_timesteps, num_timesteps + 1, device=device)
            alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.alphas_cumprod = alphas_cumprod[1:]  # Remove t=0
            
            # Compute alphas and betas
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            self.alphas = self.alphas_cumprod / self.alphas_cumprod_prev
            self.betas = 1 - self.alphas
        else:
            # Linear beta schedule
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Computed values for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Log calculation clipped for stability
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # Precompute sqrt values for forward diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

# =============================================================================
# MLP DENOISER MODEL
# =============================================================================
class DiffusionDenoiser(nn.Module):
    """
    MLP Denoiser for conditional diffusion model with Classifier-Free Guidance.
    """
    def __init__(self, n_features, time_emb_dim=64, hidden_dim=256, cfg_dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.time_emb_dim = time_emb_dim
        self.cfg_dropout = cfg_dropout
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Input dimension: noisy_price(1) + time_embedding(64) + features(n_features)
        input_dim = 1 + time_emb_dim + n_features
        
        # Main denoising network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)  # Output: predicted noise
        )
        
    def forward(self, noisy_x, timesteps, conditions, use_cfg=False):
        """
        Args:
            noisy_x: (batch_size, 1) - noisy price at timestep t
            timesteps: (batch_size,) - diffusion timesteps
            conditions: (batch_size, n_features) - conditioning features
            use_cfg: bool - whether to use classifier-free guidance
        """
        # Get timestep embedding
        time_emb = get_timestep_embedding(timesteps, self.time_emb_dim, device=noisy_x.device)
        time_emb = self.time_mlp(time_emb)
        
        if use_cfg and self.training:
            # During training with CFG: randomly drop out conditions
            # Create mask: 1 = keep condition, 0 = drop condition
            mask = torch.rand(conditions.shape[0], 1, device=conditions.device) > self.cfg_dropout
            conditions_masked = conditions * mask.float() # Ensure float mult
            
            # Concatenate: noisy_price + time_embedding + masked_features
            x = torch.cat([noisy_x, time_emb, conditions_masked], dim=-1)
            predicted_noise = self.network(x)
            return predicted_noise, conditions  # Return original conditions for loss (unused but consistent signature)
            
        elif use_cfg and not self.training:
            # During inference with CFG: get both conditional and unconditional
            # Unconditional (no conditions)
            x_uncond = torch.cat([noisy_x, time_emb, torch.zeros_like(conditions)], dim=-1)
            noise_uncond = self.network(x_uncond)
            
            # Conditional (with conditions)
            x_cond = torch.cat([noisy_x, time_emb, conditions], dim=-1)
            noise_cond = self.network(x_cond)
            
            return noise_uncond, noise_cond
            
        else:
            # Normal forward pass
            x = torch.cat([noisy_x, time_emb, conditions], dim=-1)
            predicted_noise = self.network(x)
            return predicted_noise

# =============================================================================
# FORWARD DIFFUSION & LOSS
# =============================================================================
def q_sample(x_start, t, noise=None, schedule=None):
    """
    Forward diffusion: add noise to data at timestep t.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Get schedule values for timesteps t
    sqrt_alpha_cumprod = schedule.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_cumprod = schedule.sqrt_one_minus_alphas_cumprod[t]
    
    # Reshape for broadcasting
    sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod[:, None]
    
    # Add noise
    x_noisy = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    return x_noisy


def compute_loss(model, x_start, conditions, schedule, use_cfg=True):
    """
    Compute diffusion training loss with optional Classifier-Free Guidance.
    """
    batch_size = x_start.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=x_start.device).long()
    
    # Sample noise
    noise = torch.randn_like(x_start)
    
    # Forward diffusion: add noise
    x_noisy = q_sample(x_start, t, noise, schedule)
    
    # Predict noise with model
    if use_cfg:
        # CFG mode: model returns (predicted_noise, original_conditions)
        predicted_noise, _ = model(x_noisy, t, conditions, use_cfg=True)
    else:
        # Standard mode
        predicted_noise = model(x_noisy, t, conditions, use_cfg=False)
    
    # MSE loss between actual and predicted noise
    loss = F.mse_loss(predicted_noise, noise)
    
    return loss


# =============================================================================
# SAMPLING (REVERSE DIFFUSION FOR INFERENCE) WITH CFG
# =============================================================================

@torch.no_grad()
def p_sample(model, x_t, t, conditions, schedule, cfg_scale=1.5):
    """
    Single reverse diffusion step: sample x_{t-1} given x_t.
    """
    # Get schedule values for timesteps t
    alpha_t = schedule.alphas[t]
    alpha_cumprod_t = schedule.alphas_cumprod[t]
    beta_t = schedule.betas[t]
    sqrt_one_minus_alpha_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t]
    
    # Reshape for broadcasting
    alpha_t = alpha_t[:, None]
    alpha_cumprod_t = alpha_cumprod_t[:, None]
    beta_t = beta_t[:, None]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[:, None]
    
    # Predict noise with CFG
    noise_uncond, noise_cond = model(x_t, t, conditions, use_cfg=True)
    
    # Apply classifier-free guidance
    # epsilon_final = epsilon_uncond + cfg_scale * (epsilon_cond - epsilon_uncond)
    predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
    
    # Compute mean of posterior
    mean = (1 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
    )
    
    # Add noise for t > 0
    if t[0] > 0:
        posterior_variance = schedule.posterior_variance[t]
        z = torch.randn_like(x_t)
        variance = posterior_variance[:, None] * z
    else:
        variance = 0
    
    # Sample from posterior
    x_prev = mean + variance
    
    return x_prev


@torch.no_grad()
def sample(model, conditions, schedule, num_steps=None, x_start=None, cfg_scale=1.5):
    """
    Full reverse diffusion: sample from noise to get prediction with CFG.
    """
    batch_size = conditions.shape[0]
    device = conditions.device
    
    if num_steps is None:
        num_steps = schedule.num_timesteps

    # Start from pure noise
    if x_start is None:
        x = torch.randn(batch_size, 1, device=device)
    else:
        x = x_start
    
    # Reverse diffusion: T -> 0
    timesteps = torch.arange(num_steps - 1, -1, -1, device=device)
    
    for t in timesteps:
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, conditions, schedule, cfg_scale)
    
    return x


@torch.no_grad()
def sample_with_uncertainty(model, conditions, schedule, n_samples=10, cfg_scale=1.5):
    """
    Generate multiple samples to estimate prediction uncertainty.
    """
    samples = []
    for _ in range(n_samples):
        sample_x = sample(model, conditions, schedule, cfg_scale=cfg_scale)
        samples.append(sample_x)
    
    samples = torch.cat(samples, dim=1)  # (batch_size, n_samples)
    
    return samples



"""
NoisyLinear layer implementation for NoisyNets (Fortunato et al., 2017).
Factorized Gaussian noise for efficient exploration in deep RL.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer with factorized Gaussian noise.
    
    Weights: W = μ_w + σ_w ⊙ ε_w
    Bias:    b = μ_b + σ_b ⊙ ε_b
    
    Factorized noise: ε_w = f(ε_out) ⊗ f(ε_in), where f(x) = sign(x) * sqrt(|x|)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        sigma_init: Initial value for σ parameters (default: 0.5)
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Trainable mean parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # Trainable std parameters
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not trained, resampled each step)
        self.register_buffer("eps_in", torch.zeros(1, in_features))
        self.register_buffer("eps_out", torch.zeros(out_features, 1))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize μ and σ parameters."""
        # μ init: uniform in [-1/√in, 1/√in]
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # σ init: constant = sigma_init / √in
        sigma_bound = self.sigma_init / math.sqrt(self.in_features)
        nn.init.constant_(self.weight_sigma, sigma_bound)
        nn.init.constant_(self.bias_sigma, sigma_bound)

    @torch.no_grad()
    def reset_noise(self):
        """Resample factorized Gaussian noise."""
        # Sample from standard normal
        eps_in = torch.randn(1, self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, 1, device=self.weight_mu.device)
        
        # Apply factorized transform: f(x) = sign(x) * sqrt(|x|)
        eps_in = eps_in.sign() * eps_in.abs().sqrt()
        eps_out = eps_out.sign() * eps_out.abs().sqrt()
        
        self.eps_in.copy_(eps_in)
        self.eps_out.copy_(eps_out)

    @torch.no_grad()
    def remove_noise(self):
        """Zero out noise for deterministic evaluation."""
        self.eps_in.zero_()
        self.eps_out.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with noisy parameters.
        
        Args:
            x: Input tensor (batch_size, in_features)
            
        Returns:
            Output tensor (batch_size, out_features)
        """
        # Construct factorized noise: ε_w = ε_out ⊗ ε_in
        eps_w = self.eps_out @ self.eps_in  # (out_features, in_features)
        
        # Noisy weight and bias
        weight = self.weight_mu + self.weight_sigma * eps_w
        bias = self.bias_mu + self.bias_sigma * self.eps_out.view(-1)
        
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, sigma_init={self.sigma_init}'

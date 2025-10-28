"""
NoisyQNetwork: Q-network with NoisyLinear layers for exploration.
Gen-5: Replace ε-greedy with parameter noise for sustained exploration.
"""
import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import QNetwork
from noisy_linear import NoisyLinear


class NoisyQNetwork(QNetwork):
    """
    Q-network with NoisyLinear layers in the final MLP head.
    
    Architecture:
        CNN features → MLP(256) → NoisyLinear(256, 256) → ReLU → NoisyLinear(256, n_actions)
    
    Key methods:
        - reset_noise(): Resample noise (call every training step)
        - remove_noise(): Zero noise for deterministic evaluation
    """
    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor,
        features_dim: int,
        net_arch=None,
        activation_fn=nn.ReLU,
        normalize_images: bool = True,
        sigma_init: float = 0.5,
    ):
        # Don't call super().__init__() - we'll build from scratch
        nn.Module.__init__(self)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        self.sigma_init = sigma_init
        
        action_dim = int(self.action_space.n)
        
        # Default architecture: [256, 256] with noisy final layers
        if net_arch is None:
            net_arch = [256, 256]
        
        # Build feature extraction (CNN) + first MLP layer(s)
        # Keep first layer(s) deterministic, only make final layer(s) noisy
        q_net = []
        last_layer_dim = self.features_dim
        
        # First layer: deterministic
        if len(net_arch) > 0:
            q_net.append(nn.Linear(last_layer_dim, net_arch[0]))
            q_net.append(activation_fn())
            last_layer_dim = net_arch[0]
        
        # Second-to-last layer: NoisyLinear
        if len(net_arch) > 1:
            q_net.append(NoisyLinear(last_layer_dim, net_arch[1], sigma_init=sigma_init))
            q_net.append(activation_fn())
            last_layer_dim = net_arch[1]
        
        # Final layer: NoisyLinear (to action logits)
        q_net.append(NoisyLinear(last_layer_dim, action_dim, sigma_init=sigma_init))
        
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN + noisy Q-head.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Q-values for each action
        """
        # Normalize images if needed (convert uint8 to float in [0, 1])
        if self.normalize_images:
            obs = obs.float() / 255.0
        
        features = self.features_extractor(obs)
        return self.q_net(features)

    def reset_noise(self):
        """Resample noise in all NoisyLinear layers (call every training step)."""
        for module in self.q_net:
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def remove_noise(self):
        """Remove noise for deterministic evaluation."""
        for module in self.q_net:
            if isinstance(module, NoisyLinear):
                module.remove_noise()

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Override to support deterministic flag."""
        if deterministic:
            self.remove_noise()
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)

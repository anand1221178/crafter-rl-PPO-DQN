"""
Custom Dueling DQN Policy for Crafter Environment

Implements Dueling Network Architecture (Wang et al., 2016) for Stable-Baselines3.
This architecture separates state value V(s) from action advantages A(s,a):
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
"""

import torch
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from gymnasium import spaces
from typing import List, Type, Optional


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network that separates value and advantage streams.
    
    Architecture:
        Input (CNN features) → Shared MLP → Split into:
            1. Value stream V(s): outputs single scalar
            2. Advantage stream A(s,a): outputs vector of advantages
        Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        
        if net_arch is None:
            net_arch = [256, 256]
        
        action_dim = int(self.action_space.n)
        
        # Build shared layers
        self.shared_net = nn.Sequential(*create_mlp(features_dim, -1, net_arch, activation_fn))
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        
        # Value stream: outputs V(s) - single scalar
        self.value_stream = nn.Sequential(
            nn.Linear(last_layer_dim, 128),
            activation_fn(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: outputs A(s,a) - one per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_layer_dim, 128),
            activation_fn(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values using dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        # Normalize images if needed (convert from uint8 [0,255] to float32 [0,1])
        if self.normalize_images:
            obs = obs.float() / 255.0
        
        # Extract features from observations
        features = self.features_extractor(obs)
        
        # Pass through shared layers
        shared_features = self.shared_net(features)
        
        # Compute value and advantages
        value = self.value_stream(shared_features)  # Shape: (batch, 1)
        advantages = self.advantage_stream(shared_features)  # Shape: (batch, n_actions)
        
        # Combine using dueling formula: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def set_training_mode(self, mode: bool) -> None:
        """
        Set the training mode for the network.
        Required by SB3's DQN implementation.
        """
        self.train(mode)
    
    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Predict action from Q-values.
        Required by SB3's DQN implementation.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return argmax(Q). If False, sample from distribution.
        
        Returns:
            Action tensor
        """
        q_values = self(obs)
        # Return action with highest Q-value
        return q_values.argmax(dim=1).reshape(-1)


class DuelingDQNPolicy(DQNPolicy):
    """
    Dueling DQN Policy that uses DuelingQNetwork instead of standard QNetwork.
    """
    
    def _build(self, lr_schedule):
        """
        Override _build to ensure features_extractor is created before make_q_net is called.
        """
        # Build features extractor manually
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        
        # Now build q_net (will use the features_extractor we just created)
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        # Set up optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )
    
    def make_q_net(self) -> DuelingQNetwork:
        """Create the Q-network with dueling architecture."""
        # Get features_dim from features_extractor
        features_dim = self.features_extractor.features_dim
        
        # Create dueling Q-network
        net_arch = self.net_arch
        activation_fn = self.activation_fn
        
        dueling_q_net = DuelingQNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=self.features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=self.normalize_images,
        )
        
        return dueling_q_net.to(self.device)

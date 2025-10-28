"""
NoisyDQNPolicy: Custom DQN policy with NoisyLinear layers.
Gen-5: Replaces ε-greedy exploration with parameter noise.
"""
import torch
from typing import Any, Dict, List, Optional, Type
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.dqn.policies import DQNPolicy
from noisy_qnetwork import NoisyQNetwork
import gymnasium as gym


class NoisyDQNPolicy(DQNPolicy):
    """
    DQN policy with NoisyLinear layers for exploration.
    
    Replaces ε-greedy with parameter noise in Q-network head.
    Noise is resampled every training step and removed for evaluation.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        sigma_init: float = 0.5,
    ):
        self.sigma_init = sigma_init
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> NoisyQNetwork:
        """Create NoisyQNetwork instead of regular QNetwork."""
        # Make sure we use the correctly parametrized features extractor
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        
        return NoisyQNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=net_args["features_extractor"],
            features_dim=net_args["features_dim"],
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            sigma_init=self.sigma_init,
        ).to(self.device)

    def reset_noise(self):
        """Resample noise in both online and target Q-networks."""
        self.q_net.reset_noise()
        self.q_net_target.reset_noise()

    def remove_noise(self):
        """Remove noise for deterministic evaluation."""
        self.q_net.remove_noise()
        self.q_net_target.remove_noise()

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Override to support deterministic prediction."""
        if deterministic:
            self.remove_noise()
        return self.q_net._predict(observation, deterministic=deterministic)

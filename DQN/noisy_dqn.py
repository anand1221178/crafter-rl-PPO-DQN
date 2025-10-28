"""
NoisyDQN: Custom DQN that resets noise every training step.
Gen-5: Integrates NoisyNets with proper noise management.
"""
from typing import Any, Dict, Optional, Type, Union
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from noisy_dqn_policy import NoisyDQNPolicy


class NoisyDQN(DQN):
    """
    DQN with NoisyLinear layers and proper noise management.
    
    Key changes:
    - Uses NoisyDQNPolicy by default
    - Resets noise after every training step
    - Removes noise for deterministic evaluation
    """
    
    def __init__(
        self,
        policy: Union[str, Type[NoisyDQNPolicy]] = NoisyDQNPolicy,
        env: Union[GymEnv, str, None] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.05,  # Quick decay to final eps
        exploration_initial_eps: float = 0.01,  # Minimal ε-greedy
        exploration_final_eps: float = 0.01,  # Noise does exploration
        max_grad_norm: float = 10.0,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        n_steps: int = 1,  # Gen-1: N-step returns support
    ):
        # Set default policy_kwargs for NoisyNets
        if policy_kwargs is None:
            policy_kwargs = {}
        if "sigma_init" not in policy_kwargs:
            policy_kwargs["sigma_init"] = 0.5  # Default sigma
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            n_steps=n_steps,  # Gen-1: N-step returns
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the DQN and reset noise after EACH gradient step.
        
        Ordering (per step):
          1. Optimizer step (gradient update)
          2. Target update (hard copy every target_update_interval steps)
          3. Reset noise on BOTH online and target networks
        
        This ensures fresh exploration noise for each update, with proper
        ordering for hard target updates (following Rainbow/NoisyNets papers).
        
        Args:
            gradient_steps: Number of gradient steps
            batch_size: Batch size for training
        """
        # We need to reset noise after EACH gradient step, not after all steps
        # So we call parent train with gradient_steps=1 in a loop
        for _ in range(gradient_steps):
            # Steps 1+2: optimizer step + target update (if interval reached)
            super().train(gradient_steps=1, batch_size=batch_size)
            
            # Step 3: Reset noise after update (both online and target nets)
            if hasattr(self.policy, "reset_noise"):
                self.policy.reset_noise()

    def predict(
        self,
        observation,
        state = None,
        episode_start = None,
        deterministic: bool = False,
    ):
        """
        Override predict to ensure noise is removed for deterministic predictions.
        
        Args:
            observation: Observation
            state: Hidden state (not used for DQN)
            episode_start: Episode start flag
            deterministic: Whether to use deterministic policy
            
        Returns:
            action, state
        """
        if deterministic and hasattr(self.policy, "remove_noise"):
            self.policy.remove_noise()
        
        return super().predict(observation, state, episode_start, deterministic)

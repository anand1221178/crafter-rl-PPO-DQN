"""
Prioritized Experience Replay (PER) Buffer for Stable-Baselines3

Implements prioritized sampling based on TD-error magnitude, allowing the agent
to learn more efficiently from important/surprising transitions.

Reference: Schaul et al. (2015) - "Prioritized Experience Replay"
https://arxiv.org/abs/1511.05952
"""

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Optional, Union, NamedTuple


class PrioritizedReplayBufferSamples(NamedTuple):
    """Extended ReplayBufferSamples with indices and weights for PER."""
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    indices: np.ndarray
    weights: th.Tensor


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to their TD-error:
        P(i) ∝ p_i^α
    where p_i is the priority (TD-error) and α controls prioritization strength.
    
    Uses importance sampling weights to correct for bias:
        w_i = (N * P(i))^(-β) / max_w
    where β anneals from β_start to β_end over training.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 1_000_000,
    ):
        """
        Args:
            alpha: Prioritization exponent (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_frames: Number of frames to anneal beta from start to end
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        
        # Priority storage
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0
        self.eps = 1e-6  # Small constant to avoid zero priorities
    
    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
    ) -> None:
        """Add transition with max priority (ensures new samples are seen)."""
        # Add to underlying buffer
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Set priority to max (new transitions are important)
        idx = (self.pos - 1) % self.buffer_size
        self.priorities[idx] = self.max_priority
    
    def _get_beta(self) -> float:
        """Linearly anneal beta from beta_start to beta_end."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def _get_samples(self, batch_inds: np.ndarray, env = None, weights: th.Tensor = None) -> PrioritizedReplayBufferSamples:
        """
        Override parent to return PrioritizedReplayBufferSamples with indices and weights.
        """
        # Get standard samples from parent
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env=env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env=env)

        # Create PrioritizedReplayBufferSamples
        data = PrioritizedReplayBufferSamples(
            observations=self.to_torch(self._normalize_obs(self.observations[batch_inds, 0, :], env=env)),
            actions=self.to_torch(self.actions[batch_inds, 0, :]),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(self.dones[batch_inds]),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds], env=env)),
            indices=batch_inds,
            weights=weights if weights is not None else th.ones(len(batch_inds), device=self.device),
        )
        
        # Increment frame counter for beta annealing
        self.frame += len(batch_inds)
        
        return data
    
    def sample(self, batch_size: int, env = None) -> PrioritizedReplayBufferSamples:
        """
        Sample batch with prioritized sampling + importance weights.
        
        Returns:
            PrioritizedReplayBufferSamples with indices and weights
        """
        # Get current buffer size
        upper_bound = self.buffer_size if self.full else self.pos
        
        assert upper_bound > 0, "Cannot sample from empty buffer"
        
        # Get priorities for valid samples
        priorities = self.priorities[:upper_bound]
        
        # Avoid division by zero
        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)
        
        # Compute sampling probabilities: P(i) = p_i^α / Σ p_j^α
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices according to priorities
        batch_inds = np.random.choice(upper_bound, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights: w_i = (N * P(i))^(-β)
        beta = self._get_beta()
        weights = (upper_bound * probs[batch_inds]) ** (-beta)
        weights /= weights.max()  # Normalize by max weight
        
        # Convert to torch tensor
        weights = th.as_tensor(weights, device=self.device, dtype=th.float32)
        
        # Use parent's _get_samples which properly handles everything including n-step
        return self._get_samples(batch_inds, env=env, weights=weights)
    
    def update_priorities(self, indices: np.ndarray, td_errors: th.Tensor) -> None:
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Indices of sampled transitions
            td_errors: TD-errors for those transitions (can be tensor or numpy)
        """
        if isinstance(td_errors, th.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        
        # Update priorities: p_i = |TD_error_i| + ε
        priorities = np.abs(td_errors) + self.eps
        self.priorities[indices] = priorities
        
        # Update max priority
        self.max_priority = max(self.max_priority, float(priorities.max()))

"""
Base Agent class for all RL algorithms.
This defines the interface that all agents must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    All agents (DrQ-v2, PPO, etc.) should inherit from this.
    """

    def __init__(self, observation_shape: tuple, num_actions: int, device: str = 'cpu'):
        """
        Initialize base agent.

        Args:
            observation_shape: Shape of observations (H, W, C)
            num_actions: Number of discrete actions
            device: Device to use ('cpu' or 'cuda')
        """
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.device = device
        self.training_step = 0

    @abstractmethod
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.

        TODO(human): Each agent implements its own action selection
        - DrQ-v2: Uses Q-values with epsilon-greedy
        - PPO: Samples from policy distribution
        - Your choice for course algorithm
        """
        pass

    @abstractmethod
    def store_experience(self, obs: np.ndarray, action: int, reward: float,
                        next_obs: np.ndarray, done: bool) -> None:
        """
        Store a transition for learning.

        TODO(human): Implementation depends on algorithm
        - DrQ-v2: Store in replay buffer
        - PPO: Store in rollout buffer
        """
        pass

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update the agent's policy/value functions.

        TODO(human): This is where the learning happens!
        Returns dict of metrics (loss, etc.) or None if not ready
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from disk."""
        pass
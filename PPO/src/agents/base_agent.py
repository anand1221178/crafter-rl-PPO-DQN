"""
Base Agent class for all RL algorithms.
"""
###UNUSED CLASS PLEASE IGNORE####
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional


class BaseAgent(ABC):

    def __init__(self, observation_shape: tuple, num_actions: int, device: str = 'cpu'):
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.device = device
        self.training_step = 0

    @abstractmethod
    def act(self, observation: np.ndarray, training: bool = True) -> int:

        pass

    @abstractmethod
    def store_experience(self, obs: np.ndarray, action: int, reward: float,
                        next_obs: np.ndarray, done: bool) -> None:

        pass

    @abstractmethod
    def update(self) -> Optional[Dict[str, float]]:

        pass

    @abstractmethod
    def save(self, path: str) -> None:

        pass

    @abstractmethod
    def load(self, path: str) -> None:

        pass
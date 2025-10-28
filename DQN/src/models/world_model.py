"""
World Model for Dyna-Q Agent

This module implements a tabular world model that learns environment dynamics
by storing (state, action) → (next_state, reward) transitions.

References:
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
    MIT Press. Chapter 8: Planning and Learning with Tabular Methods.
    Specifically Section 8.2 (Dyna: Integrated Planning, Acting, and Learning) and
    Figure 8.2 (Tabular Dyna-Q pseudocode), p. 164.

Algorithm:
    From Sutton & Barto (2018), Figure 8.2, Step (e):
    "Model(S,A) ← R, S' (assuming deterministic environment)"

    For high-dimensional states (like Crafter's 64×64×3 images), we use
    feature hashing to reduce dimensionality while maintaining lookup capability.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict
import hashlib


class WorldModel:
    """
    Tabular world model for Dyna-Q planning.

    Stores transitions as (state_hash, action) → (next_state, reward) pairs.
    For visual observations, uses feature-based hashing to handle high-dimensional inputs.

    Reference:
        Sutton & Barto (2018), Section 8.2, p. 160:
        "The model is a simple table that stores the most recent experience
        for each state-action pair."

    Args:
        capacity: Maximum number of transitions to store (default: 50000)
        state_dim: Dimension of state features for hashing (default: 256)
    """

    def __init__(self, capacity: int = 50000, state_dim: int = 256):
        self.capacity = capacity
        self.state_dim = state_dim

        # Model storage: (state_hash, action) → (next_state, reward)
        self.transitions: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

        # Track which states and actions have been visited
        self.visited_states: Set[int] = set()
        self.state_actions: Dict[int, Set[int]] = defaultdict(set)

        # Statistics for debugging
        self.num_updates = 0
        self.collision_count = 0

    def hash_state(self, state: np.ndarray) -> int:
        """
        Create a hash of the state for dictionary lookup.

        For high-dimensional visual states (like 64×64×3 images), we:
        1. Downsample the image to reduce dimensions
        2. Convert to bytes
        3. Use cryptographic hash for uniform distribution

        Reference:
            Adapted from standard practice in large-scale tabular methods.
            See Sutton & Barto (2018), Section 8.11, p. 184:
            "When the state space is large or continuous, the model can be
            represented using function approximation or feature-based hashing."

        Args:
            state: Observation array (typically 64×64×3 for Crafter)

        Returns:
            Integer hash of the state
        """
        # Downsample state to reduce hash collisions
        # For 64×64×3 images, downsample to 8×8×3 = 192 dimensions
        if len(state.shape) == 3 and state.shape[0] == 64:
            # Downsample by factor of 8 (64 → 8)
            downsampled = state[::8, ::8, :]
        else:
            downsampled = state

        # Convert to bytes and hash
        # Use quantization to reduce sensitivity to small pixel changes
        quantized = (downsampled / 32).astype(np.int8)  # 256 levels → 8 levels
        state_bytes = quantized.tobytes()

        # Use MD5 hash (fast, good distribution, cryptographic not needed)
        hash_int = int(hashlib.md5(state_bytes).hexdigest()[:16], 16)

        return hash_int

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray) -> None:
        """
        Update the world model with a new transition.

        Implements Step (e) from Sutton & Barto (2018), Figure 8.2:
        "Model(S,A) ← R, S'"

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Resulting next state

        Note:
            Assuming deterministic environment as per Sutton & Barto.
            Stochastic environments would require storing transition probabilities.
        """
        # Hash the state for efficient lookup
        state_hash = self.hash_state(state)

        # Check if we're at capacity (simple FIFO replacement)
        if len(self.transitions) >= self.capacity:
            # Remove oldest transition (first key)
            oldest_key = next(iter(self.transitions))
            del self.transitions[oldest_key]

        # Store transition (most recent experience for this state-action)
        key = (state_hash, action)
        self.transitions[key] = (next_state.copy(), reward)

        # Update visitation tracking
        self.visited_states.add(state_hash)
        self.state_actions[state_hash].add(action)

        self.num_updates += 1

    def sample_random_transition(self) -> Optional[Tuple[int, int, np.ndarray, float]]:
        """
        Sample a random transition from the model for planning.

        Implements Step (f) from Sutton & Barto (2018), Figure 8.2:
        "S ← random previously observed state
         A ← random action previously taken in S
         R, S' ← Model(S,A)"

        Returns:
            Tuple of (state_hash, action, next_state, reward), or None if model is empty

        Reference:
            Sutton & Barto (2018), p. 164:
            "Random-sample one-step tabular Q-planning randomly samples
            state-action pairs from among those that have been previously experienced."
        """
        if not self.transitions:
            return None

        # Sample random state-action pair
        # np.random.choice doesn't work with list of tuples, so use randint
        keys = list(self.transitions.keys())
        idx = np.random.randint(len(keys))
        key = keys[idx]

        state_hash, action = key
        next_state, reward = self.transitions[key]

        return state_hash, action, next_state, reward

    def sample_from_state(self, state: np.ndarray) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        Sample a transition starting from a specific state.

        Useful for more directed planning strategies.

        Args:
            state: State to sample transition from

        Returns:
            Tuple of (action, next_state, reward), or None if state not in model
        """
        state_hash = self.hash_state(state)

        if state_hash not in self.state_actions:
            return None

        # Get valid actions for this state
        valid_actions = list(self.state_actions[state_hash])
        if not valid_actions:
            return None

        # Sample random action
        action = np.random.choice(valid_actions)

        # Look up transition
        key = (state_hash, action)
        if key in self.transitions:
            next_state, reward = self.transitions[key]
            return action, next_state, reward

        return None

    def get_transition(self, state: np.ndarray, action: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Deterministic lookup of a specific transition.

        Args:
            state: State to query
            action: Action to query

        Returns:
            Tuple of (next_state, reward), or None if transition not in model
        """
        state_hash = self.hash_state(state)
        key = (state_hash, action)

        if key in self.transitions:
            return self.transitions[key]
        return None

    def get_statistics(self) -> Dict[str, int]:
        """
        Get model statistics for debugging and logging.

        Returns:
            Dictionary with model statistics
        """
        return {
            'num_transitions': len(self.transitions),
            'num_states': len(self.visited_states),
            'num_updates': self.num_updates,
            'capacity_utilization': len(self.transitions) / self.capacity,
            'avg_actions_per_state': (
                sum(len(actions) for actions in self.state_actions.values()) /
                max(len(self.state_actions), 1)
            )
        }

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.transitions)

    def __repr__(self) -> str:
        """String representation of the model."""
        stats = self.get_statistics()
        return (f"WorldModel(transitions={stats['num_transitions']}, "
                f"states={stats['num_states']}, "
                f"capacity={self.capacity})")

"""
Rollout Buffer for PPO (On-Policy Learning)

Unlike DQN's replay buffer (off-policy), PPO uses a rollout buffer that stores
trajectories collected with the current policy. After each update, the buffer
is cleared since PPO is on-policy.
"""

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """
    Storage for trajectories collected during PPO rollouts.

    PPO collects a fixed number of steps (e.g., 2048) from the environment,
    computes advantages using GAE, then performs multiple epochs of updates
    on this data before collecting new trajectories.

    Key differences from replay buffer:
    - Fixed size (n_steps), not capacity-based
    - Cleared after each update cycle
    - Stores additional info for PPO (log_probs, values)
    """

    def __init__(self, n_steps: int, observation_shape: Tuple[int, int, int],
                 num_actions: int, device: str = 'cpu'):
        """
        Initialize rollout buffer.

        Args:
            n_steps: Number of steps to collect per rollout (e.g., 2048)
            observation_shape: Shape of observations (C, H, W) - (3, 64, 64) for Crafter
            num_actions: Number of discrete actions (17 for Crafter)
            device: Device to store tensors on ('cpu' or 'cuda')
        """
        self.n_steps = n_steps
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.device = device

        # Storage arrays
        self.observations = np.zeros((n_steps,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)

        # Computed during finalize() after rollout collection
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        # Current position in buffer
        self.pos = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            value: float, log_prob: float) -> None:
        """
        Add a single transition to the buffer.

        Args:
            obs: Observation, shape (C, H, W)
            action: Action taken (integer)
            reward: Reward received
            done: Whether episode terminated
            value: State value V(s) from critic
            log_prob: Log probability of the action under current policy
        """
        if self.pos >= self.n_steps:
            raise RuntimeError(f"Buffer is full! Call reset() before adding more transitions.")

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1

        if self.pos == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99,
                                       gae_lambda: float = 0.95) -> None:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        GAE balances bias-variance tradeoff in advantage estimation:
        - λ=0: A = δ (one-step TD, low variance but high bias)
        - λ=1: A = Σ(γ^t * r_t) - V(s) (Monte Carlo, high variance but low bias)
        - λ=0.95: Good middle ground (recommended default)

        Formula:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t = δ_t + γ * λ * A_{t+1} * (1 - done)

        Args:
            last_value: Value of the final state after rollout (for bootstrapping)
            gamma: Discount factor (default 0.99)
            gae_lambda: GAE lambda parameter (default 0.95)
        """
        if not self.full:
            raise RuntimeError("Cannot compute advantages until buffer is full!")

        # Compute advantages using GAE
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # Bootstrap from last_value (value of state after rollout)
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # TD error: δ = r + γ*V(s') - V(s)
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]

            # GAE: A = δ + γ*λ*A_next
            last_gae_lam = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae_lam
            self.advantages[t] = last_gae_lam

        # Returns are advantages + values
        # This is the target for the value function: V_target = A + V
        self.returns = self.advantages + self.values

    def get(self) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Get all data from the buffer as PyTorch tensors.

        Yields a single batch containing all collected transitions.
        Used for creating minibatches during PPO update.

        Returns:
            Generator yielding tuple of:
                - observations: (n_steps, C, H, W)
                - actions: (n_steps,)
                - log_probs: (n_steps,)
                - advantages: (n_steps,)
                - returns: (n_steps,)
                - values: (n_steps,)
        """
        if not self.full:
            raise RuntimeError("Cannot get data until buffer is full!")

        # Convert to torch tensors
        obs_tensor = torch.from_numpy(self.observations).to(self.device)
        actions_tensor = torch.from_numpy(self.actions).to(self.device)
        log_probs_tensor = torch.from_numpy(self.log_probs).to(self.device)
        advantages_tensor = torch.from_numpy(self.advantages).to(self.device)
        returns_tensor = torch.from_numpy(self.returns).to(self.device)
        values_tensor = torch.from_numpy(self.values).to(self.device)

        yield (obs_tensor, actions_tensor, log_probs_tensor,
               advantages_tensor, returns_tensor, values_tensor)

    def reset(self) -> None:
        """
        Reset the buffer for next rollout collection.

        PPO is on-policy, so we discard old data after each update cycle.
        """
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return self.pos if not self.full else self.n_steps


# Helper function for creating minibatches during PPO update
def create_minibatches(batch_size: int, *arrays) -> Generator[Tuple, None, None]:
    """
    Create random minibatches from arrays for multiple epochs of training.

    PPO performs multiple epochs (e.g., 10) of updates on collected data.
    Each epoch shuffles the data and creates minibatches.

    Args:
        batch_size: Size of each minibatch
        *arrays: Arrays to batch (all must have same length)

    Yields:
        Tuples of minibatches for each array
    """
    n_samples = len(arrays[0])
    indices = np.random.permutation(n_samples)

    # Generate minibatches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield tuple(arr[batch_indices] for arr in arrays)


if __name__ == "__main__":
    """Test the rollout buffer."""
    print("Testing RolloutBuffer...")

    # Create buffer
    buffer = RolloutBuffer(n_steps=4, observation_shape=(3, 64, 64), num_actions=17)

    # Add some dummy transitions
    for i in range(4):
        obs = np.random.randn(3, 64, 64).astype(np.float32)
        action = np.random.randint(0, 17)
        reward = np.random.randn()
        done = i == 3  # Last one is terminal
        value = np.random.randn()
        log_prob = np.random.randn()

        buffer.add(obs, action, reward, done, value, log_prob)

    print(f"Buffer size: {len(buffer)}/{buffer.n_steps}")
    print(f"Buffer full: {buffer.full}")

    # Compute advantages
    last_value = 0.0  # Terminal state has value 0
    buffer.compute_returns_and_advantages(last_value)

    print(f"Advantages: {buffer.advantages}")
    print(f"Returns: {buffer.returns}")

    # Get data
    for batch in buffer.get():
        obs, actions, log_probs, advantages, returns, values = batch
        print(f"\nBatch shapes:")
        print(f"  Observations: {obs.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Advantages: {advantages.shape}")

    print("\nRolloutBuffer test passed!")

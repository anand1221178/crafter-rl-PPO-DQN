#This is just a storage class

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:


    def __init__(self, n_steps: int, observation_shape: Tuple[int, int, int], num_actions: int, device: str = 'cpu'):
        self.n_steps = n_steps
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.device = device

        #  arrays for storage
        self.observations = np.zeros((n_steps,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)

        # Computed during finalize
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        # Current pos in buffer
        self.pos = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,value: float, log_prob: float) -> None:
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

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99,gae_lambda: float = 0.95) -> None:
        if not self.full:
            raise RuntimeError("Cannot compute advantages until buffer is full!")

        #  advantages using GAE
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:

                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # TD error
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]

            # GAE
            last_gae_lam = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae_lam
            self.advantages[t] = last_gae_lam

        # V_target = A + V
        self.returns = self.advantages + self.values

    def get(self) -> Generator[Tuple[torch.Tensor, ...], None, None]:

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

        self.pos = 0
        self.full = False

    def __len__(self) -> int:

        return self.pos if not self.full else self.n_steps


def create_minibatches(batch_size: int, *arrays) -> Generator[Tuple, None, None]:
    n_samples = len(arrays[0])
    indices = np.random.permutation(n_samples)

    # Generate minibatches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield tuple(arr[batch_indices] for arr in arrays)


if __name__ == "__main__":

    print("Testing RolloutBuffer...")


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

    #  advantages
    last_value = 0.0 
    buffer.compute_returns_and_advantages(last_value)

    print(f"Advantages: {buffer.advantages}")
    print(f"Returns: {buffer.returns}")


    for batch in buffer.get():
        obs, actions, log_probs, advantages, returns, values = batch
        print(f"\nBatch shapes:")
        print(f"  Observations: {obs.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Advantages: {advantages.shape}")

    print("\nRolloutBuffer test passed!")

"""  The replay buffer is a circular data structure that stores past experiences for training. DrQ-v2
  needs this to break temporal correlations in the data - by sampling random batches of old
  experiences, the agent learns more stably than if it only used the most recent transitions.
  ─────────────────────────────────────────────────"""

"""
  Replay Buffer for DrQ-v2 Agent

  This module implements an efficient circular buffer for storing and sampling
  transitions in reinforcement learning. The buffer stores raw experiences and
  enables random batch sampling for training.
"""

import numpy as np
from typing import Tuple,Dict

class ReplayBuffer:
    """Circular buffer to store transitions and smaple random batches"""
    """Crucial for off-policy algos
    1) Will decorrelate seqential experiences (breaks temporal correlation)
    2) Enables us to reuse past experiences multiple times (sample efficient)
    3) Provides stable training by mixing old and new experiences.
    """

    def __init__ (self, capacity: int = 100_000, observation_shape: Tuple[int,int,int] = (64,64,3), device: str='cpu'):
        """
          Initialize the replay buffer with pre-allocated numpy arrays.
          
          Args:
              capacity: Maximum number of transitions to store
              observation_shape: Shape of observations (H, W, C) for Crafter
              device: Device for tensor operations (we'll convert when sampling)
          """
        self.capacity = capacity
        self.device = device
        self.observation_shape = observation_shape

        #Preallocate memory -> from paper to improve efficiency
        self.observations = np.zeros((capacity, *observation_shape), dtype = np.uint8)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)

        #Actions as discrete-> crafter has 17 actions
        self.actions = np.zeros(capacity, dtype=np.uint64)

        #rewards and done flags
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        #circular buffer pointers
        self.idx = 0 #current position
        self.size = 0 # current number of stored transitions
        self.full = False #Whether buffer has been filled yet or not.

    def add(self, obs: np.ndarray, action: int, reward:float, next_obs: np.ndarray, done:bool) -> None:
        """
          Add a single transition to the buffer.
          
          This overwrites old data when the buffer is full since it is a circle duh.
          
          Args:
              obs: Current observation (64x64x3 pciture)
              action: Action taken 0-16, 17 total
              reward: Reward received
              next_obs: Next observation after taking action
              done: Whether episode ended
          """
        #Transitions at current point
        self.observations[self.idx] =obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = done

        #Update the buffer ptrs
        self.idx = (self.idx + 1) % self.capacity #source : GeeksforGeeks circular ptr update

        #Tracking transitions:
        if not self.full:
            self.size+=1
            if self.size == self.capacity:
                self.full = True
                print(f"Replay buffer filled to capactity ({self.capacity ,} transitions)")
            
    def sample(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
          Sample a random batch of transitions for training.
          
          Random sampling is key for breaking correlations in sequential data.
          
          Args:
              batch_size: Number of transitions to sample
              
          Returns:
              Dictionary containing batch of transitions with keys:
              - 'obs': Current observations (batch_size, 64, 64, 3)
              - 'action': Actions taken (batch_size,)
              - 'reward': Rewards received (batch_size,)
              - 'next_obs': Next observations (batch_size, 64, 64, 3)
        """

        #Make sure we dont smaple more than we store
        if batch_size > self.size:
            raise ValueError(f"Batch size ({batch_size}) larger than buffer size ({self.size})")
        
        #Use np choice for random sampling
        indices = np.random.choice(self.size, batch_size, replace=False)

        batch ={
            'obs' : self.observations[indices],
            'action' : self.actions[indices],
            'reward' : self.rewards[indices],
            'next_obs' : self.next_observations[indices],
            'done' : self.dones[indices]
        }

        return batch
    
    def sample_with_augmentation(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
          Sample a batch and prepare it for augmentation.
          
          This method will be extended in Improvement 1 to apply data augmentation.
          For now, it just samples normally but converts to float32 for neural networks.
          
          Args:
              batch_size: Number of transitions to sample
              
          Returns:
              Batch dictionary with observations as float32 in [0, 1] range
          """
        batch = self.sample(batch_size)
        # Convert images from uint8 [0, 255] to float32 [0, 1]
          # Neural networks work better with normalized inputs
        batch['obs'] = batch['obs'].astype(np.float32) / 255.0
        batch['next_obs'] = batch['next_obs'].astype(np.float32) / 255.0

        # TODO(Improvement 1): Add data augmentation here
          # - Random crops (84x84 -> 64x64)
          # - Color jittering (brightness/contrast)

        return batch

    def __len__(self) -> int:
        return self.size
        
    def is_ready(self, min_size: int = 1000) -> bool:
        """
        Check if buffer has enough samples to start training.

        We typically wait for some minimum number of transitions before
        starting training to ensure diverse initial batches.

        Args:
            min_size: Minimum transitions needed before training
            
        Returns:
            True if buffer has at least min_size transitions
        """
        return self.size >= min_size
        
    def get_stats(self) -> Dict[str, float]:
        """
        Get buffer statistics for logging.

        Returns:
            Dictionary with buffer metrics
        """
        return {
        'buffer_size': self.size,
        'buffer_capacity': self.capacity,
        'buffer_idx': self.idx,
        'buffer_full': self.full,
        'avg_reward': np.mean(self.rewards[:self.size]) if self.size > 0 else 0.0,
        'done_ratio': np.mean(self.dones[:self.size]) if self.size > 0 else 0.0}

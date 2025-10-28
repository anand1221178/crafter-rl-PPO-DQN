"""
DQN with Prioritized Experience Replay

Extends Stable-Baselines3's DQN to support priority updates in PER buffer.
"""

import torch as th
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from per_buffer import PrioritizedReplayBuffer


class DQN_PER(DQN):
    """
    DQN with Prioritized Experience Replay support.
    
    Automatically updates priorities after computing TD-errors during training.
    """
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Update policy using gradient descent (with priority updates for PER).
        
        This overrides the parent train() to add priority updates after computing TD-errors.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            # Compute TD-error (for priority updates)
            td_errors = target_q_values - current_q_values
            
            # Compute Huber loss (less sensitive to outliers)
            loss = th.nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            
            # Apply importance sampling weights if using PER
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and hasattr(replay_data, 'weights'):
                loss = (loss * replay_data.weights.unsqueeze(1)).mean()
            else:
                loss = loss.mean()
            
            losses.append(loss.item())
            
            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Update priorities in PER buffer
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and hasattr(replay_data, 'indices'):
                self.replay_buffer.update_priorities(replay_data.indices, td_errors.squeeze())
        
        # Increase update counter
        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

"""
PPO (Proximal Policy Optimization) Agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.utils.networks import ActorCritic
from src.utils.rollout_buffer import RolloutBuffer, create_minibatches


class PPOAgent(BaseAgent):

    def __init__(self,
                 observation_shape: tuple = (3, 64, 64),
                 num_actions: int = 17,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # Network hyperparameters
                 hidden_dim: int = 512,
                 lr: float = 3e-4,
                 # PPO hyperparameters
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_clip: float = 0.2,
                 entropy_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 normalize_advantages: bool = True,
                 dual_clip: Optional[float] = None):
        """
        agent params - this is from the internet!

        Args:
            observation_shape: Shape of observations (C, H, W)
            num_actions: Number of discrete actions
            device: Device to use ('cpu' or 'cuda')
            hidden_dim: Hidden dimension for actor/critic heads
            lr: Learning rate (same for actor and critic)
            n_steps: Steps per rollout collection
            batch_size: Minibatch size for updates
            n_epochs: Number of epochs to train on each rollout
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_clip: Value function clipping parameter
            entropy_coef: Entropy bonus coefficient
            vf_coef: Value function loss coefficient
            max_grad_norm: Gradient clipping threshold
            normalize_advantages: Whether to normalize advantages
            dual_clip: Dual-clip coefficient (None to disable, typical: 3.0)
        """
        super().__init__(observation_shape, num_actions, device)

        # Hyperparameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.dual_clip = dual_clip

        # create newroks 
        self.policy = ActorCritic(num_actions=num_actions, hidden_dim=hidden_dim).to(device)

        # TODO: find best optimisor - done (not sure if adam best but only one that worked)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # rollouts
        self.rollout_buffer = RolloutBuffer(
            n_steps=n_steps,
            observation_shape=observation_shape,
            num_actions=num_actions,
            device=device
        )

        self.training_step = 0
        self.n_updates = 0

    def act(self, observation: np.ndarray, training: bool = True) -> int:

        # preprocess forst 
        obs = self._preprocess_obs(observation)

        # action from policy
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs, deterministic=not training)

        
        self._last_obs = obs.cpu().numpy().squeeze(0)  # Remove batch dim
        self._last_action = action.item()
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()

        return self._last_action

    def store_experience(self, obs: np.ndarray, action: int, reward: float,next_obs: np.ndarray, done: bool) -> None:

        #Rollout buffer addition here
        self.rollout_buffer.add(
            obs=self._last_obs,
            action=self._last_action,
            reward=reward,
            done=done,
            value=self._last_value,
            log_prob=self._last_log_prob
        )

        self.training_step += 1

    def update(self) -> Optional[Dict[str, float]]:
        #TODO: update ppo when roloout buffer is full
        if not self.rollout_buffer.full:
            return None

        #TODO: Add here for ppo update logic of bootstrap if full or previous - DONE
        last_obs = self._preprocess_obs(self.rollout_buffer.observations[-1])
        with torch.no_grad():
            last_value = self.policy.get_value(last_obs).item()

        #TODO: compute GAE ad
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

   
        for batch in self.rollout_buffer.get():
            obs, actions, old_log_probs, advantages, returns, old_values = batch

            #From paper -> normalise for stability
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            #gather lots of epochs for update
            all_policy_losses = []
            all_value_losses = []
            all_entropies = []
            all_clip_fractions = []

            for epoch in range(self.n_epochs):
                # minibatch
                for minibatch in create_minibatches(
                    self.batch_size, obs, actions, old_log_probs,
                    advantages, returns, old_values
                ):
                    mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_old_values = minibatch

                    # Evaluate actions from current policy
                    new_log_probs, new_values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)

                    # TODO: Criiical respahe!
                    new_values = new_values.squeeze(-1)

                    # PPO stuff below

                    
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages

                    if self.dual_clip is not None:

                        surr3 = torch.max(self.dual_clip * mb_advantages, surr2)
                        policy_loss = -torch.min(surr1, surr3).mean()
                    else:

                        policy_loss = -torch.min(surr1, surr2).mean()


                    if self.value_clip is not None:
                        value_pred_clipped = mb_old_values + torch.clamp(
                            new_values - mb_old_values,
                            -self.value_clip,
                            self.value_clip
                        )
                        value_loss_unclipped = (new_values - mb_returns).pow(2)
                        value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()


                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()


                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropies.append(entropy_loss.item())


                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    all_clip_fractions.append(clip_fraction)


        self.rollout_buffer.reset()
        self.n_updates += 1

        # Return metrics
        return {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy': np.mean(all_entropies),
            'clip_fraction': np.mean(all_clip_fractions),
            'n_updates': self.n_updates
        }

    def save(self, path: str) -> None:
        """store agent on disk"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'n_updates': self.n_updates,
            'hyperparameters': {
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_clip': self.value_clip,
                'entropy_coef': self.entropy_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
            }
        }, save_path)

    def load(self, path: str) -> None:
        """get agent on disk"""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.n_updates = checkpoint.get('n_updates', 0)

        print(f"Loaded PPO agent from {path}")
        print(f"  Training step: {self.training_step}")
        print(f"  Updates: {self.n_updates}")

    def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:

        # accoutn for  different input formats
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0

        # HWC -> CHW
        if obs.shape[-1] == 3:
            obs = obs.transpose(2, 0, 1)

        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]

        # Cto tensor
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        return obs_tensor


if __name__ == "__main__":

    print("Testing PPO Agent...")


    agent = PPOAgent(
        observation_shape=(3, 64, 64),
        num_actions=17,
        device='cpu',
        n_steps=8  # Small for testing
    )

    print(f"Agent created successfully!")
    print(f"  Device: {agent.device}")
    print(f"  Buffer size: {agent.n_steps}")
    print(f"  Policy parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    # Test action selection
    dummy_obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    action = agent.act(dummy_obs)
    print(f"\nTest action: {action}")

    print("\nPPO Agent test passed!")

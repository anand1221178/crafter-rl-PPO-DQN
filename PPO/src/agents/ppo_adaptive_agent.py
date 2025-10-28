import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from src.agents.ppo_agent import PPOAgent


class PPOAdaptiveAgent(PPOAgent):
    """
    PPO agent with adaptive exploration mechanisms.
    """

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
                 entropy_coef_start: float = 0.05,  # Start high 
                 entropy_coef_end: float = 0.001,   # End low 
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 normalize_advantages: bool = True,
                 # Adaptive exploration parameters
                 target_kl: float = 0.015,          # Target KL divergence
                 kl_early_stop: bool = True,        # Stop epoch if KL too high
                 adaptive_clip: bool = True,        # Adjust clip epsilon based on KL
                 total_timesteps: int = 1_000_000): # For entropy decay schedule
        # Initialize base PPO agent
        super().__init__(
            observation_shape=observation_shape,
            num_actions=num_actions,
            device=device,
            hidden_dim=hidden_dim,
            lr=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            value_clip=value_clip,
            entropy_coef=entropy_coef_start,  # Will be updated dynamically
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantages=normalize_advantages
        )

        # Adaptive exploration parameters
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.target_kl = target_kl
        self.kl_early_stop = kl_early_stop
        self.adaptive_clip = adaptive_clip
        self.total_timesteps = total_timesteps

        # Initial clip epsilon 
        self.initial_clip_epsilon = clip_epsilon
        self.clip_epsilon = clip_epsilon

        # Tracking
        self.kl_divergences = []

    def _update_entropy_coef(self):
        progress = min(1.0, self.training_step / self.total_timesteps)
        self.entropy_coef = self.entropy_coef_start + progress * (
            self.entropy_coef_end - self.entropy_coef_start
        )

    def _compute_kl_divergence(self, old_log_probs: torch.Tensor,new_log_probs: torch.Tensor) -> float:
        """
        KL(old || new) = E[log(old) - log(new)]
        """
        kl = (old_log_probs - new_log_probs).mean().item()
        return kl

    def _adapt_clip_epsilon(self, kl_div: float):
        """
        Adapt clip_epsilon based on KL divergence.
        """
        if not self.adaptive_clip:
            return

        if kl_div > self.target_kl * 1.5:
            # Policy changing too fast need to be more conservatice
            self.clip_epsilon = max(0.05, self.clip_epsilon * 0.9)
        elif kl_div < self.target_kl * 0.5:
            #too slow update faster
            self.clip_epsilon = min(self.initial_clip_epsilon, self.clip_epsilon * 1.1)

    def update(self) -> Optional[Dict[str, float]]:

        if not self.rollout_buffer.full:
            return None


        self._update_entropy_coef()

        # value for bootstrping
        last_obs = self._preprocess_obs(self.rollout_buffer.observations[-1])
        with torch.no_grad():
            last_value = self.policy.get_value(last_obs).item()

        #GAE lol
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # Pull data from buffer
        for batch in self.rollout_buffer.get():
            obs, actions, old_log_probs, advantages, returns, old_values = batch

            # normalise
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # epochs of updates
            all_policy_losses = []
            all_value_losses = []
            all_entropies = []
            all_clip_fractions = []
            all_kl_divs = []

            for epoch in range(self.n_epochs):
                # minibatches
                from src.utils.rollout_buffer import create_minibatches
                for minibatch in create_minibatches(
                    self.batch_size, obs, actions, old_log_probs,
                    advantages, returns, old_values
                ):
                    mb_obs, mb_actions, mb_old_log_probs, mb_advantages, mb_returns, mb_old_values = minibatch

                    # check how well actions are doing 
                    new_log_probs, new_values, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)

                    # need to respahe
                    new_values = new_values.squeeze(-1)

                    # find KL divergence
                    kl_div = self._compute_kl_divergence(mb_old_log_probs, new_log_probs)
                    all_kl_divs.append(kl_div)

                    # stop early if kl too hgigh
                    if self.kl_early_stop and kl_div > self.target_kl * 2.0:
                        print(f"  Early stopping at epoch {epoch+1}/{self.n_epochs} (KL={kl_div:.4f} > {self.target_kl*2.0:.4f})")
                        break

                    #ppo stuff below

                    # 1 - policy loss
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 2 - value loss - need to clip this so we dont have massive updates - resolved
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

                    # TODO: Add bonus here - Resolved
                    entropy_loss = entropy.mean()

                    #entire functions loss
                    loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

                    # TODO: Apply built in optimiser - done
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    #TODO: add tracking before training run!
                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropies.append(entropy_loss.item())

      
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    all_clip_fractions.append(clip_fraction)

                if self.kl_early_stop and kl_div > self.target_kl * 2.0:
                    break

        #TODO: add adpative clip epsilon - done
        if all_kl_divs:
            mean_kl = np.mean(all_kl_divs)
            self._adapt_clip_epsilon(mean_kl)
            self.kl_divergences.append(mean_kl)

        #reset buffer for next
        self.rollout_buffer.reset()
        self.n_updates += 1

        
        return {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy': np.mean(all_entropies),
            'clip_fraction': np.mean(all_clip_fractions),
            'kl_divergence': mean_kl if all_kl_divs else 0.0,
            'entropy_coef': self.entropy_coef, 
            'clip_epsilon': self.clip_epsilon,  
            'n_updates': self.n_updates
        }


if __name__ == "__main__":

    #Simple test to see if works
    print("Testing PPOAdaptiveAgent...")


    agent = PPOAdaptiveAgent(
        observation_shape=(3, 64, 64),
        num_actions=17,
        device='cpu',
        n_steps=8,  
        entropy_coef_start=0.05,
        entropy_coef_end=0.001,
        total_timesteps=1000
    )

    print(f"Agent created successfully!")
    print(f"  Initial entropy coef: {agent.entropy_coef:.4f}")
    print(f"  Initial clip epsilon: {agent.clip_epsilon:.4f}")
    print(f"  Target KL: {agent.target_kl:.4f}")


    agent.training_step = 500  # Halfway
    agent._update_entropy_coef()
    print(f"\nAfter 500 steps (50% progress):")
    print(f"  Entropy coef: {agent.entropy_coef:.4f}")

    agent.training_step = 1000  # End
    agent._update_entropy_coef()
    print(f"\nAfter 1000 steps (100% progress):")
    print(f"  Entropy coef: {agent.entropy_coef:.4f}")

    print("\nPPOAdaptiveAgent test passed!")

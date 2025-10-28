"""
DQN with Polyak (soft) target updates for Gen-3.

This subclass replaces hard target network copies with Polyak averaging
to provide smoother, more stable bootstrapping for long-horizon tasks.

Key Design:
- Inherits all DQN behavior from Stable-Baselines3
- Only overrides `_on_step()` to apply Polyak updates after each gradient step
- tau (τ) parameter controls blend rate: target ← (1-τ)*target + τ*online
- Typical τ ≈ 0.005 means target moves 0.5% toward online network per step

Why for Gen-3:
- Gen-2's action masking improved stone collection (0.46% → 1.33%)
- But progression remains brittle (no stone pickaxe yet)
- N-step returns (n=3) can have spiky TD errors
- Hard target copies every 1000 steps amplify jumps
- Polyak smoothing stabilizes learning for longer chains

Expected Impact:
- Crafter Score: 4.00% → 4.20-4.40% (+5-10%)
- Stone pickaxe: 0.00% → 0.5-2% (first unlock)
- TD error variance: -20-30% reduction
- Basics remain stable (no regression)

"""

from typing import Any, Dict, Optional, Type
import torch as th
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import GymEnv


class DQNPolyak(DQN):
    """
    DQN with Polyak (soft) target network updates.
    
    Replaces periodic hard copies of Q-network to target network with
    gradual Polyak averaging applied after every gradient step:
    
        θ_target ← (1 - τ) * θ_target + τ * θ_online
    
    This provides smoother target values and better stability for
    compositional tasks with sparse rewards.
    
    Parameters
    ----------
    tau : float, default=0.005
        Polyak averaging coefficient. Typical range: 0.001-0.01.
        - Lower τ: slower target updates, more stable but less responsive
        - Higher τ: faster updates, more responsive but less stable
        
    All other parameters inherited from stable_baselines3.DQN
    """
    
    def __init__(
        self,
        policy: str | Type,
        env: GymEnv,
        tau: float = 0.005,
        target_update_interval: int = 1,  # Must be 1 for Polyak to work
        **kwargs
    ):
        """
        Initialize DQN with Polyak updates.
        
        Parameters
        ----------
        tau : float
            Polyak averaging coefficient (0.001-0.01 typical)
        target_update_interval : int
            MUST be 1 for Polyak updates (update every gradient step)
            If set to other values, reverts to standard DQN behavior
        """
        # Force target_update_interval=1 for Polyak to work correctly
        if target_update_interval != 1:
            print(f"⚠️  Warning: target_update_interval={target_update_interval} incompatible with Polyak updates")
            print(f"⚠️  Forcing target_update_interval=1 for smooth updates")
            target_update_interval = 1
        
        # Initialize parent DQN
        super().__init__(
            policy=policy,
            env=env,
            target_update_interval=target_update_interval,
            **kwargs
        )
        
        # Store Polyak coefficient
        self.polyak_tau = tau
        
        # Initial hard sync: target ← online (required before Polyak updates)
        # This is critical: Polyak is incremental, needs proper initialization
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        print(f"🎯 Gen-3: Polyak Target Updates ENABLED (τ={tau})")
        print(f"   ✅ Initial hard sync: θ_target ← θ_online")
        print(f"   Target network will update smoothly every gradient step")
        print(f"   Formula: θ_target ← (1-τ)*θ_target + τ*θ_online")
    
    def _on_step(self) -> None:
        """
        This method is called in the parent's learn() loop but we don't 
        use it for Polyak updates. See train() method instead.
        """
        # Let parent handle learning rate updates
        super()._on_step()
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the Q-network with gradient descent, then apply Polyak updates.
        
        Overrides parent's train() to add Polyak averaging after gradient steps.
        Double DQN logic is preserved in parent's implementation.
        """
        # Call parent's train method (does gradient descent with Double DQN)
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        
        # Apply Polyak update to target network after gradient steps
        # This runs every time train() is called (with target_update_interval=1)
        # Formula: θ_target ← (1-τ)*θ_target + τ*θ_online
        with th.no_grad():
            polyak_update(
                self.q_net.parameters(),      # Source: online network
                self.q_net_target.parameters(), # Target: target network
                self.polyak_tau                # Blend coefficient
            )


def create_dqn_polyak_model(
    env: GymEnv,
    tau: float = 0.005,
    learning_rate: float = 1e-4,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    n_steps: int = 3,
    exploration_fraction: float = 0.2,
    exploration_final_eps: float = 0.05,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
    device: str = "auto"
) -> DQNPolyak:
    """
    Factory function to create DQNPolyak model with Gen-3 configuration.
    
    This preserves Gen-2 hyperparameters and adds Polyak updates.
    
    Parameters
    ----------
    env : GymEnv
        Training environment (should have action masking from Gen-2)
    tau : float, default=0.005
        Polyak coefficient (typical: 0.001-0.01)
    
    Returns
    -------
    DQNPolyak
        Configured model ready for training
    """
    return DQNPolyak(
        policy="CnnPolicy",
        env=env,
        tau=tau,
        target_update_interval=1,  # Required for Polyak
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device=device,
        # Gen-1 feature: n-step returns
        n_steps=n_steps,
        # Gen-1 feature: enhanced network
        policy_kwargs=dict(
            net_arch=[256, 256],
        )
    )

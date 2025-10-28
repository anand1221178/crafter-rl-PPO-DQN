"""
Reward Shaping Wrapper for Crafter Environment (Gen-4)

Provides dense reward signal for foundational milestones.
Implements linear decay over training.

NOTE: This approach failed to improve performance in experiments.
"""

import gymnasium as gym
import numpy as np


def linear_decay(progress, start=0.0, stop=1.0):
    """
    Linear decay schedule from 1.0 to 0.0.
    
    Args:
        progress: Training progress in [0,1]
        start: Progress value where decay begins (default: 0.0)
        stop: Progress value where decay ends (default: 1.0)
    
    Returns:
        Scale factor in [0, 1] (1.0 before start, 0.0 after stop)
    """
    if progress <= start:
        return 1.0
    if progress >= stop:
        return 0.0
    t = (progress - start) / (stop - start)
    return max(0.0, 1.0 - t)


class ShapingWrapper(gym.Wrapper):
    """
    Train-time reward shaping for foundational milestones in Crafter.
    
    Key Features:
    - Awards each milestone ONCE per episode
    - Decays shaping over training progress (40%→90% of training)
    - Caps total shaping per episode to prevent reward hacking
    - Tracks global step count for decay schedule
    
    CRITICAL: NEVER use this wrapper for evaluation!
    Evaluation should use raw environment with standard Crafter rewards only.
    
    Calibration (from Gen-1 results):
    - Median return: 3.1
    - Cap per episode: 0.62 (20% of median)
    - Milestones sum to ≤ cap_per_ep
    
    Example usage:
        MILESTONES = {
            "collect_wood": 0.155,       # 25% of cap (foundational)
            "place_table": 0.155,        # 25% of cap (unlocks tools)
            "make_wood_pickaxe": 0.186,  # 30% of cap (progression)
            "collect_coal": 0.124,       # 20% of cap (advanced)
        }
        train_env = ShapingWrapper(raw_env, milestones=MILESTONES, 
                                   cap_per_ep=0.62, total_steps=1_000_000)
    """
    
    def __init__(
        self,
        env,
        milestones: dict,
        cap_per_ep: float,
        total_steps: int,
        decay_start: float = 0.4,
        decay_stop: float = 0.9
    ):
        """
        Args:
            env: Base Crafter environment
            milestones: Dict mapping achievement names to base bonus values
            cap_per_ep: Maximum total shaping reward per episode
            total_steps: Total training steps (for computing progress)
            decay_start: Progress fraction where decay begins (default: 0.4)
            decay_stop: Progress fraction where decay ends (default: 0.9)
        """
        super().__init__(env)
        self.milestones = milestones
        self.cap_per_ep = float(cap_per_ep)
        self.total_steps = int(total_steps)
        self.decay_start = decay_start
        self.decay_stop = decay_stop
        
        # Episode-level state
        self.seen = None
        self.shaping_total = None
        
        # Global training progress
        self._global_steps = 0
        
        self._reset_episode_state()
    
    def _reset_episode_state(self):
        """Reset per-episode tracking variables."""
        self.seen = {k: False for k in self.milestones}
        self.shaping_total = 0.0
    
    def reset(self, **kwargs):
        """Reset environment and episode-level shaping state."""
        self._reset_episode_state()
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """
        Take environment step and add shaping rewards for milestone achievements.
        
        Returns:
            obs, reward (with shaping), terminated, truncated, info (with shaping metadata)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_steps += 1
        
        # Compute training progress and decay scale
        progress = min(1.0, self._global_steps / max(1, self.total_steps))
        scale = linear_decay(progress, start=self.decay_start, stop=self.decay_stop)
        
        shaped = 0.0
        
        # Crafter exposes achievements in info["achievements"] as {name: bool}
        ach = info.get("achievements", {})
        
        for milestone_name, base_bonus in self.milestones.items():
            # Check if achievement just unlocked this step
            flag = ach.get(milestone_name, False)
            hit = (bool(flag) and not self.seen[milestone_name])
            
            if hit and self.shaping_total < self.cap_per_ep and scale > 0.0:
                # Award scaled bonus, respecting cap
                bonus = min(base_bonus * scale, self.cap_per_ep - self.shaping_total)
                shaped += bonus
                self.seen[milestone_name] = True
                self.shaping_total += bonus
        
        # Add shaping metadata to info for logging/analysis
        if shaped != 0.0:
            info = dict(info)  # Make mutable copy
            info["shaping_bonus_step"] = shaped
            info["shaping_progress"] = progress
            info["shaping_scale"] = scale
        
        return obs, reward + shaped, terminated, truncated, info


# ========================================
# Gen-4 Calibrated Milestones
# ========================================

# Based on Gen-1 results:
# - Median return: 3.1
# - Cap: 0.62 (20% of median)
# - Decay: 40%→90% of training (full bonus early, fades out late)

CAP_PER_EPISODE = 0.62

MILESTONES = {
    "collect_wood": 0.155,       # 0.25 * cap (25%) - Foundational
    "place_table": 0.155,        # 0.25 * cap (25%) - Unlocks tools
    "make_wood_pickaxe": 0.186,  # 0.30 * cap (30%) - Forces progression
    "collect_coal": 0.124,       # 0.20 * cap (20%) - Advanced resource
}

# Verify sum ≤ cap
assert sum(MILESTONES.values()) <= CAP_PER_EPISODE, "Milestones exceed cap!"

print(f"✅ Reward Shaping Configuration:")
print(f"  Cap per episode: {CAP_PER_EPISODE:.3f}")
print(f"  Total possible shaping: {sum(MILESTONES.values()):.3f}")
print(f"  Decay schedule: 40%→90% of training")
print(f"  Milestones:")
for name, bonus in MILESTONES.items():
    print(f"    - {name}: +{bonus:.3f}")


class StonePotentialShapingWrapper(gym.Wrapper):
    """
    Potential-based shaping focused on the stone progression chain.

    Potential stages (Φ):
        0 → baseline (no tools)
        1 → first wood pickaxe crafted
        2 → first stone collected
        3 → first stone pickaxe crafted

    Shaping term applied per step:
        F(s, s') = λ * scale(progress) * (γ Φ(s') - Φ(s))

    Properties:
    - Policy invariant (potential-based)
    - Train-only (never use for evaluation!)
    - Linear decay 30%→70% of training so shaping fades out
    - λ chosen so max shaping ≤ ~10% of Gen-1 median return
    """

    def __init__(
        self,
        env,
        total_steps: int,
        lambda_scale: float,
        discount: float,
        decay_start: float,
        decay_stop: float,
    ):
        super().__init__(env)
        self.total_steps = max(1, int(total_steps))
        self.lambda_scale = float(lambda_scale)
        self.discount = float(discount)
        self.decay_start = float(decay_start)
        self.decay_stop = float(decay_stop)

        # Training progress tracker
        self._global_steps = 0

        # Current potential value Φ(s)
        self._phi = 0.0

    def reset(self, **kwargs):
        self._phi = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _potential(self, achievements) -> float:
        """Map achievements dict to potential value Φ(s)."""
        phi = 0.0
        if achievements.get('make_wood_pickaxe', False):
            phi = max(phi, 1.0)
        if achievements.get('collect_stone', False):
            phi = max(phi, 2.0)
        if achievements.get('make_stone_pickaxe', False):
            phi = max(phi, 3.0)
        return phi

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_steps += 1

        achievements = info.get('achievements', {}) or {}
        next_phi = self._potential(achievements)

        progress = min(1.0, self._global_steps / self.total_steps)
        scale = linear_decay(progress, start=self.decay_start, stop=self.decay_stop)

        shaped = 0.0
        if scale > 0.0:
            shaped = self.lambda_scale * scale * (
                self.discount * next_phi - self._phi
            )

        # Update state AFTER computing shaping term
        self._phi = next_phi

        if shaped != 0.0:
            info = dict(info)
            info['stone_shaping_bonus'] = shaped
            info['stone_shaping_progress'] = progress
            info['stone_shaping_scale'] = scale
            info['stone_phi'] = self._phi

        return obs, reward + shaped, terminated, truncated, info


# Stone-chain shaping defaults
STONE_POTENTIAL_LAMBDA = 0.10
STONE_POTENTIAL_DECAY_START = 0.30
STONE_POTENTIAL_DECAY_STOP = 0.70

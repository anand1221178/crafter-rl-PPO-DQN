"""
FrameStack Wrapper for Crafter
Stacks last N frames along channel dimension for temporal context.
Addresses partial observability by giving CNN access to recent history.
"""

import numpy as np
import gymnasium as gym
from collections import deque


class ChannelFrameStack(gym.Wrapper):
    """
    Stack last N frames along the channel dimension (HWC format).
    
    Transforms observation from (H, W, C) to (H, W, C*N) by concatenating
    the last N frames along the channel axis. This gives the CNN temporal
    context without requiring recurrent layers.
    
    Args:
        env: Environment to wrap
        num_frames: Number of frames to stack (default: 4)
    """
    
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
        # Update observation space to reflect stacked channels
        old_space = env.observation_space
        assert len(old_space.shape) == 3, "Expected (H, W, C) observation space"
        
        h, w, c = old_space.shape
        new_shape = (h, w, c * num_frames)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=old_space.dtype
        )
        
        print(f"🎞️ FrameStack: {old_space.shape} → {new_shape}")
    
    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        
        # Fill deque with copies of initial frame
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        
        return self._get_stacked_obs(), info
    
    def step(self, action):
        """Take step and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add new frame to deque (oldest automatically removed)
        self.frames.append(obs)
        
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self):
        """Concatenate frames along channel dimension."""
        # Stack along axis=2 (channel axis for HWC format)
        stacked = np.concatenate(list(self.frames), axis=2)
        return stacked.astype(self.observation_space.dtype)


class FrameStackLogger(gym.Wrapper):
    """
    Logs framestack-related metrics for analysis.
    Tracks temporal patterns that might help with stone chain.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_steps = 0
        self.time_to_table = None
        self.time_to_first_stone = None
        self.time_to_wood_pickaxe = None
    
    def reset(self, **kwargs):
        # Save previous episode metrics
        if self.episode_steps > 0:
            info = {
                'episode_steps': self.episode_steps,
                'time_to_table': self.time_to_table,
                'time_to_first_stone': self.time_to_first_stone,
                'time_to_wood_pickaxe': self.time_to_wood_pickaxe,
            }
            # Log if we have any progression
            if self.time_to_first_stone or self.time_to_table:
                pass  # Could log to file here
        
        # Reset for new episode
        self.episode_steps = 0
        self.time_to_table = None
        self.time_to_first_stone = None
        self.time_to_wood_pickaxe = None
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        # Track first occurrences
        if 'inventory' in info:
            inv = info['inventory']
            
            if self.time_to_table is None and inv.get('table', 0) > 0:
                self.time_to_table = self.episode_steps
            
            if self.time_to_first_stone is None and inv.get('stone', 0) > 0:
                self.time_to_first_stone = self.episode_steps
            
            if self.time_to_wood_pickaxe is None and inv.get('wood_pickaxe', 0) > 0:
                self.time_to_wood_pickaxe = self.episode_steps
        
        return obs, reward, terminated, truncated, info


def test_framestack():
    """Test that FrameStack works correctly with Crafter."""
    import crafter
    from shimmy import GymV21CompatibilityV0
    
    # create test environment
    env = crafter.Env()
    env = GymV21CompatibilityV0(env=env)
    
    print("Original observation space:", env.observation_space.shape)
    
    # apply FrameStack
    env = ChannelFrameStack(env, num_frames=4)
    print("Stacked observation space:", env.observation_space.shape)
    
    # test reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")
    
    # Test step
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"Step observation shape: {obs.shape}")
    
    # Verify channel stacking
    h, w, c = env.observation_space.shape
    assert c == 12, f"Expected 12 channels (3*4), got {c}"
    
    print("\n✅ FrameStack test passed!")


if __name__ == "__main__":
    test_framestack()

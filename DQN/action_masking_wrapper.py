import gymnasium as gym
import numpy as np

ACTION_NAMES = [
    'Noop', 'Move Left', 'Move Right', 'Move Up', 'Move Down',
    'Do', 'Sleep', 'Place Stone', 'Place Table', 'Place Furnace',
    'Place Plant', 'Make Wood Pickaxe', 'Make Stone Pickaxe',
    'Make Iron Pickaxe', 'Make Wood Sword', 'Make Stone Sword',
    'Make Iron Sword'
]


class ActionMaskingWrapper(gym.Wrapper):
    """
    Inventory-aware action masking wrapper.
    
    Checks if actions are valid based on inventory before executing.
    Invalid actions are remapped to either NOOP or random valid action.
    """

    def __init__(self, env, fallback_mode='noop', seed=None):
        """
        Args:
            env: Base environment
            fallback_mode: 'noop' or 'random' fallback for invalid actions
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        self.fallback_mode = fallback_mode
        self.noop_action = 0
        self.seed_val = seed
        self._rng = np.random.default_rng(seed)
        
        # Episode counters
        self.invalid_actions = 0
        self.fallback_count = 0
        self.total_actions = 0
        self._last_info = {}
        self._last_valid_count = 0
        self.episode_idx = 0

    def reset(self, **kwargs):
        self.invalid_actions = 0
        self.fallback_count = 0
        self.total_actions = 0
        self._last_valid_count = 0
        
        obs, info = self.env.reset(**kwargs)
        self._last_info = info if isinstance(info, dict) else {}
        self.episode_idx += 1
        
        return obs, info

    def _inventory(self):
        """Extract inventory from last observed info."""
        inv = self._last_info.get('inventory')
        if isinstance(inv, dict):
            return inv
        return None

    def _get_valid_actions(self, info):
        """Return list of valid action IDs based on inventory."""
        inv = self._inventory()
        if inv is None:
            valid = list(range(17))
            self._last_valid_count = len(valid)
            return valid
        
        valid = []
        
        # Helper function
        def has(item, count=1):
            try:
                return int(inv.get(item, 0)) >= count
            except Exception:
                return False
        
        # Movement, noop, do, sleep
        valid.extend([0, 1, 2, 3, 4, 5, 6])
        
        # Crafting actions
        if has('stone', 1):
            valid.append(7)  # Place Stone
        if has('wood', 1):
            valid.append(8)  # Place Table
        if has('stone', 1):
            valid.append(9)  # Place Furnace
        if has('sapling', 1):
            valid.append(10)  # Place Plant
        if has('wood', 1):
            valid.append(11)  # Make Wood Pickaxe
        if has('stone', 1) and has('wood', 1):
            valid.append(12)  # Make Stone Pickaxe
        if has('iron', 1):
            valid.append(13)  # Make Iron Pickaxe
        if has('wood', 1):
            valid.append(14)  # Make Wood Sword
        if has('stone', 1):
            valid.append(15)  # Make Stone Sword
        if has('iron', 1):
            valid.append(16)  # Make Iron Sword
        
        self._last_valid_count = len(valid)
        return valid

    def _sample_valid_action(self, info):
        """Sample a valid action based on fallback_mode."""
        valid = self._get_valid_actions(info)
        
        if self.fallback_mode == 'random':
            # Prefer non-NOOP actions
            valid_non_noop = [a for a in valid if a != self.noop_action]
            
            if len(valid_non_noop) > 0:
                action = int(self._rng.choice(valid_non_noop))
                self.fallback_count += 1
                return action
            elif len(valid) > 0:
                return self.noop_action
            else:
                return self.noop_action
        else:
            return self.noop_action

    def step(self, action):
        """Step with action masking and fallback."""
        self.total_actions += 1
        
        prev_info = self._last_info if self._last_info else {}
        valid_actions = self._get_valid_actions(prev_info)
        is_valid = (action in valid_actions)
        
        if not is_valid:
            self.invalid_actions += 1
            if self.fallback_mode == 'random':
                valid_non_noop = [a for a in valid_actions if a != self.noop_action]
                if len(valid_non_noop) > 0:
                    action = int(self._rng.choice(valid_non_noop))
                    self.fallback_count += 1
                else:
                    action = self.noop_action
            else:
                action = self.noop_action
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info if isinstance(info, dict) else {}
        
        # Log stats
        info = dict(info) if isinstance(info, dict) else {}
        info["invalid_action_count_ep"] = self.invalid_actions
        info["invalid_action_rate_ep"] = self.invalid_actions / max(1, self.total_actions)
        info["fallback_count_ep"] = self.fallback_count
        info["valid_action_count"] = self._last_valid_count
        
        return obs, reward, terminated, truncated, info

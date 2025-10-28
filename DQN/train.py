import argparse
import os
from datetime import datetime
import gym as old_gym
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register

# Import DQN components from wrappers folder
from wrappers.dueling_dqn import DuelingDQNPolicy
from wrappers.per_buffer import PrioritizedReplayBuffer
from wrappers.dqn_per import DQN_PER
from wrappers.shaping_wrapper import (
    ShapingWrapper,
    MILESTONES,
    CAP_PER_EPISODE,
    StonePotentialShapingWrapper,
    STONE_POTENTIAL_LAMBDA,
    STONE_POTENTIAL_DECAY_START,
    STONE_POTENTIAL_DECAY_STOP,
)
from wrappers.action_masking_wrapper import ActionMaskingWrapper
from wrappers.framestack_wrapper import ChannelFrameStack, FrameStackLogger
from wrappers.icm_lite_wrapper import IcmLiteWrapper
from wrappers.noisy_dqn import NoisyDQN
from wrappers.noisy_dqn_policy import NoisyDQNPolicy

# Import custom agents
# from src.agents.dynaq_agent import DynaQAgent  # Dyna-Q (external algorithm) - commented out for now

# Additional imports for training
import torch
import numpy as np
import time
from collections import defaultdict

parser = argparse.ArgumentParser(description='Train RL agents on Crafter environment')
parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn', 'dynaq'],
                   default='ppo', help='Algorithm to train (ppo, dqn, or dynaq)')
parser.add_argument('--outdir', default='logdir/crafter')
parser.add_argument('--steps', type=float, default=1e6, help='Training steps (default: 1M)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--eval_freq', type=int, default=50000, help='Evaluation frequency')
# Gen-4: Reward shaping flag
parser.add_argument('--reward-shaping', action='store_true', help='Enable reward shaping (Gen-4)')
# Gen-3b: Stone-chain potential shaping flag
parser.add_argument('--stone-shaping', action='store_true', help='Enable stone-chain potential shaping (Gen-3)')
# Gen-2: Action masking flag
parser.add_argument('--action-masking', action='store_true', help='Enable inventory-aware action masking (Gen-2)')
# Gen-3c: Action masking fallback mode
parser.add_argument('--mask-fallback', type=str, choices=['noop', 'random'], default='noop',
                   help='Fallback mode for invalid actions: noop (Gen-2) or random (Gen-3c, default: noop)')
# Gen-3: Polyak soft target updates flag
parser.add_argument('--polyak', action='store_true', help='Enable Polyak soft target updates (Gen-3)')
parser.add_argument('--tau', type=float, default=0.005, help='Polyak coefficient for soft updates (Gen-3, default: 0.005)')
# Gen-3b: FrameStack for temporal context
parser.add_argument('--framestack', action='store_true', help='Enable FrameStack(4) for temporal context (Gen-3b)')
parser.add_argument('--num-frames', type=int, default=4, help='Number of frames to stack (default: 4)')
# Gen-4: ICM-lite curiosity
parser.add_argument('--icm', action='store_true', help='Enable ICM-lite curiosity wrapper (Gen-4, TRAIN ONLY)')
parser.add_argument('--icm-beta', type=float, default=0.05, help='ICM intrinsic reward scale (default: 0.05)')
parser.add_argument('--icm-cap', type=float, default=0.31, help='ICM per-episode cap (default: 0.31, ~10%% median)')
parser.add_argument('--icm-decay-start', type=float, default=0.20, help='ICM decay start progress (default: 0.20)')
parser.add_argument('--icm-decay-stop', type=float, default=0.60, help='ICM decay stop progress (default: 0.60)')
# Gen-5: NoisyNets for sustained exploration
parser.add_argument('--noisyNets', action='store_true', help='Enable NoisyNets (Gen-5, replaces ε-greedy with parameter noise)')
parser.add_argument('--sigma-init', type=float, default=0.5, help='Initial sigma for NoisyLinear layers (default: 0.5)')
# Dyna-Q specific arguments
parser.add_argument('--planning_steps', type=int, default=5, help='Planning steps per real step (Dyna-Q only)')
parser.add_argument('--prioritized', action='store_true', help='Use prioritized sweeping (Dyna-Q only)')
parser.add_argument('--exploration_bonus', type=float, default=0.0, help='Exploration bonus κ for Dyna-Q+')
args = parser.parse_args()

if args.reward_shaping and args.stone_shaping:
    raise ValueError("Reward shaping and stone-chain shaping cannot be enabled simultaneously")

# Simple wrapper to handle Gym API differences
class CrafterWrapper(gym.Env):
    """Wrapper to make Crafter compatible with Gymnasium/SB3."""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Convert old gym spaces to gymnasium spaces
        from gymnasium import spaces as gym_spaces
        
        # Action space: Discrete(17)
        self.action_space = gym_spaces.Discrete(env.action_space.n)
        
        # Observation space: Box(0, 255, (64, 64, 3), uint8)
        obs_shape = env.observation_space.shape
        self.observation_space = gym_spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )


class RandomShiftAugmentation(gym.Wrapper):
    """
    🎨 IMPROVEMENT 2: Data Augmentation (DrQ-style)
    
    Applies random translation (shifts) to observations to improve sample efficiency
    and generalization. Based on "Data-Efficient RL with Self-Predictive Representations"
    (Schwarzer et al., 2021) and "Image Augmentation is All You Need" (Kostrikov et al., 2020).
    
    How it works:
    1. Pad observation by `pad` pixels on all sides (using edge replication)
    2. Randomly crop back to original size (64×64)
    3. Effect: Random shifts up/down/left/right by up to `pad` pixels
    
    Why this helps Crafter:
    - Agent learns spatial invariance (resources look same if shifted slightly)
    - Better generalization (sees more diverse views of same situation)
    - Regularization effect (prevents overfitting to specific pixel positions)
    - Proven to improve sample efficiency by 20-40% in pixel-based RL
    
    Args:
        env: The environment to wrap
        pad: Number of pixels to pad (default: 4, effective range: 0-8 pixel shifts)
    """
    
    def __init__(self, env, pad=4):
        super().__init__(env)
        self.pad = pad
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info
    
    def _augment(self, obs):
        """Apply random shift augmentation to a single observation."""
        # Pad image (replicate edge pixels to avoid black borders)
        padded = np.pad(
            obs,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='edge'
        )
        
        # Random crop back to original size
        h, w = obs.shape[:2]
        top = np.random.randint(0, 2 * self.pad + 1)
        left = np.random.randint(0, 2 * self.pad + 1)
        
        cropped = padded[top:top+h, left:left+w]
        return cropped


class CrafterWrapperBase(gym.Env):
    """Base wrapper to make Crafter compatible with Gymnasium/SB3."""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Convert old gym spaces to gymnasium spaces
        from gymnasium import spaces as gym_spaces
        
        # Action space: Discrete(17)
        self.action_space = gym_spaces.Discrete(env.action_space.n)
        
        # Observation space: Box(0, 255, (64, 64, 3), uint8)
        obs_shape = env.observation_space.shape
        self.observation_space = gym_spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )
    
    def reset(self, seed=None, options=None):
        # Gymnasium style reset (returns obs, info)
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        # Always return just observation (not tuple)
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Ensure done is a Python bool (handle different types)
        if hasattr(done, 'item'):  # NumPy scalar
            done = bool(done.item())
        else:
            done = bool(done)
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        truncated = False  # Crafter doesn't have truncation
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()


# Alias for backward compatibility
CrafterWrapper = CrafterWrapperBase


# Setup environment - bypass Gym entirely for cleaner setup
# Create output directory with algorithm and timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"{args.outdir}_{args.algorithm}_{timestamp}"
os.makedirs(outdir, exist_ok=True)

# Create Crafter environment directly
base_env = crafter.Env()

# Add recording wrapper for Crafter metrics
recorded_env = crafter.Recorder(base_env, outdir, save_stats=True, save_video=False, save_episode=False)

# Apply our simple wrapper to handle API differences
env = CrafterWrapper(recorded_env)

# Gen-2: Apply action masking wrapper FIRST (needs raw info with inventory)
if args.action_masking:
    if args.mask_fallback == 'random':
        print("\n✨ Gen-3c: Action Masking with RANDOM-VALID fallback")
        print("  Invalid actions → uniform sample from valid actions")
        print("  Reduces sticky loops, maintains exploration")
    else:
        print("\n✨ Gen-2: Action Masking with NOOP fallback")
        print("  Invalid actions → NOOP (original Gen-2 behavior)")
    
    env = ActionMaskingWrapper(env, fallback_mode=args.mask_fallback, seed=args.seed)

# Gen-3b: Apply FrameStack for temporal context (AFTER masking, only changes obs)
if args.framestack:
    print(f"\n✨ Gen-3b: FrameStack ENABLED (num_frames={args.num_frames})")
    print("  Provides temporal context for partial observability")
    print(f"  Observation: (64,64,3) → (64,64,{3*args.num_frames})")
    env = ChannelFrameStack(env, num_frames=args.num_frames)
    env = FrameStackLogger(env)

# Gen-4: Apply reward shaping wrapper if enabled (TRAIN ONLY!)
if args.reward_shaping:
    print(f"\n✨ Gen-4: Reward Shaping ENABLED")
    print(f"  Milestones: {list(MILESTONES.keys())}")
    print(f"  Cap per episode: {CAP_PER_EPISODE:.3f}")
    print(f"  Decay: 40%→90% of training")
    env = ShapingWrapper(
        env,
        milestones=MILESTONES,
        cap_per_ep=CAP_PER_EPISODE,
        total_steps=int(args.steps),
        decay_start=0.4,
        decay_stop=0.9
    )

# Gen-3b: Apply stone-chain potential shaping (TRAIN ONLY!)
if args.stone_shaping:
    print(f"\n✨ Gen-3: Stone-Chain Potential Shaping ENABLED")
    print("  Potential stages: wood pickaxe → stone → stone pickaxe")
    print(f"  λ (scale): {STONE_POTENTIAL_LAMBDA:.3f}")
    print(f"  Decay: {STONE_POTENTIAL_DECAY_START*100:.0f}%→{STONE_POTENTIAL_DECAY_STOP*100:.0f}% of training")
    env = StonePotentialShapingWrapper(
        env,
        total_steps=int(args.steps),
        lambda_scale=STONE_POTENTIAL_LAMBDA,
        discount=0.99,
        decay_start=STONE_POTENTIAL_DECAY_START,
        decay_stop=STONE_POTENTIAL_DECAY_STOP,
    )

# Gen-4: Apply ICM-lite curiosity wrapper (TRAIN ONLY!)
if args.icm:
    print(f"\n✨ Gen-4: ICM-lite Curiosity ENABLED")
    print(f"  Intrinsic reward: cap={args.icm_cap:.3f} (~10% Gen-1 median), beta={args.icm_beta:.3f}")
    print(f"  Decay: {args.icm_decay_start*100:.0f}%→{args.icm_decay_stop*100:.0f}% of training")
    print("  Components: Forward model + inverse model (action-relevant features)")
    print("  ⚠️  TRAIN ONLY: ICM OFF at evaluation for clean Crafter metrics")
    
    # Determine input channels based on framestack
    in_channels = 3 * args.num_frames if args.framestack else 3
    
    env = IcmLiteWrapper(
        env,
        n_actions=env.action_space.n,
        total_steps=int(args.steps),
        beta=args.icm_beta,
        cap_per_ep=args.icm_cap,
        decay_start=args.icm_decay_start,
        decay_stop=args.icm_decay_stop,
        use_inverse=True,
        lr=1e-3,
        max_grad_norm=5.0,
        emb_dim=128,
        ebar_clip=3.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        enabled=True,
        optimize=True,
        seed=args.seed,
        in_channels=in_channels
    )

# NOTE: Data Augmentation (Gen-2 attempt) was removed after it FAILED
# It caused -14% regression by breaking spatial learning in Crafter
# See: logdir/Gen-2_DataAug_FAILED_20251024_113010/GENERATION_SUMMARY.md

print(f"Training {args.algorithm.upper()} for {int(args.steps):,} steps")
print(f"Output directory: {outdir}")

# TODO: Choose algorithm based on argument
if args.algorithm == 'ppo':
    # Partner's work: Stable-baselines PPO
    print("🚀 Starting PPO training...")

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create PPO model with device specification
    model = stable_baselines3.PPO(
        'CnnPolicy',
        env,
        verbose=1,
        device=device,
        # Optimized hyperparameters for Crafter
        learning_rate=3e-4,
        n_steps=2048,  # Steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=outdir
    )

    print(f"\n📊 PPO Configuration:")
    print(f"  Total steps: {int(args.steps):,}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Device: {device}")
    print(f"\n" + "="*50)
    print("Starting training loop...")
    print("="*50 + "\n")

    # Train the model
    model.learn(total_timesteps=int(args.steps))

    # Save the final model
    model_path = os.path.join(outdir, 'ppo_final.zip')
    model.save(model_path)

    print(f"\n" + "="*50)
    print("PPO Training Complete!")
    print(f"  Model saved: {model_path}")
    print(f"  Results saved: {outdir}")
    print("="*50)
elif args.algorithm == 'dqn':
    # DQN with iterative improvements
    print("🚀 Starting DQN training...")

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Gen-3: Use built-in Polyak updates (SB3 2.7.0+ has tau parameter)
    # tau=1.0 (default) = hard updates, tau<1.0 = soft (Polyak) updates
    # Gen-5 OVERRIDE: Force hard updates for NoisyNets (better target alignment)
    if args.noisyNets:
        tau_value = 1.0  # Hard updates for NoisyNets
        target_update_freq = 2000  # Moderate periodic hard copies
    else:
        tau_value = args.tau if args.polyak else 1.0
        target_update_freq = 1 if args.polyak else 1000  # Polyak needs freq=1
    
    # Gen-5: NoisyNets - replace ε-greedy with parameter noise
    if args.noisyNets:
        print("🔊 Gen-5: NoisyNets enabled - using parameter noise for exploration")
        model = NoisyDQN(
            NoisyDQNPolicy,
            env,
            verbose=1,
            device=device,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            # Minimal ε-greedy (noise does exploration)
            exploration_initial_eps=0.01,
            exploration_final_eps=0.01,
            exploration_fraction=0.05,
            tensorboard_log=outdir,
            # Gen-5: HARD target updates (tau=1.0, periodic copies every 2000 steps)
            # Better than Polyak for NoisyNets: faster target alignment, fewer interactions
            tau=tau_value,  # 1.0 (hard updates)
            target_update_interval=target_update_freq,  # 2000 (moderate periodic)
            # Gen-1: N-Step Returns
            n_steps=3,
            # Gen-5: NoisyLinear sigma initialization
            policy_kwargs=dict(
                sigma_init=args.sigma_init,
            )
        )
    else:
        model = stable_baselines3.DQN(
            'CnnPolicy',
            env,
            verbose=1,
            device=device,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.75,
            exploration_final_eps=0.05,
            tensorboard_log=outdir,
            # Gen-3: Polyak soft target updates (built-in SB3 support)
            tau=tau_value,  # 0.005 for Polyak, 1.0 for hard updates
            target_update_interval=target_update_freq,  # 1 for Polyak, 1000 for hard
            # Gen-1: N-Step Returns
            n_steps=3,
            # Gen-1: Enhanced architecture
            policy_kwargs=dict(
                net_arch=[256, 256],
            )
        )

    print(f"\n📊 DQN Configuration:")
    print(f"  Total steps: {int(args.steps):,}")
    print(f"  Device: {device}")
    print(f"  ✅ Gen-1 Base: N-step Returns (n=3) + Uniform Sampling")
    if args.noisyNets:
        print(f"  ✅ Gen-5: NoisyNets - Parameter noise exploration (σ_init={args.sigma_init})")
        print(f"     ε-greedy: {0.01}→{0.01} (minimal, noise dominates)")
        print(f"     Architecture: CNN → MLP(256) → NoisyLinear(256,256) → NoisyLinear(256,{env.action_space.n})")
        print(f"  ✅ Gen-5: HARD Target Updates (τ=1.0, interval={target_update_freq})")
        print(f"     Periodic hard copies every {target_update_freq} steps (better for NoisyNets)")
    if args.action_masking:
        if args.mask_fallback == 'random':
            print(f"  ✅ Gen-3c: Action Masking with RANDOM-VALID fallback")
            print(f"     Invalid actions → uniform sample from valid set")
        else:
            print(f"  ✅ Gen-2: Action Masking with NOOP fallback")
            print(f"     Invalid actions → NOOP (original behavior)")
    if args.framestack:
        print(f"  ✅ Gen-3b: FrameStack - ACTIVE (num_frames={args.num_frames}, channels={3*args.num_frames})")
        print(f"     Provides temporal context for partial observability")
    if args.polyak:
        print(f"  ✅ Gen-3: Polyak Updates - ACTIVE (τ={tau_value}, update_freq={target_update_freq})")
        print(f"     Formula: θ_target ← (1-τ)*θ_target + τ*θ_online every gradient step")
    else:
        print(f"  ⚠️  Gen-3: Hard Updates - Using standard (τ=1.0, update_freq=1000)")
    if args.stone_shaping:
        print(
            f"  ✅ Gen-3b: Stone Potential Shaping - ACTIVE (λ={STONE_POTENTIAL_LAMBDA}, "
            f"decay {STONE_POTENTIAL_DECAY_START:.2f}→{STONE_POTENTIAL_DECAY_STOP:.2f})"
        )
    if args.reward_shaping:
        print(f"  ✨ Gen-4: Reward Shaping - ACTIVE (milestone bonuses)")
    print(f"  Architecture: CNN → [256×256] MLP")
    print(f"\n" + "="*50)
    print("Starting training loop...")
    print("="*50 + "\n")

    # Train the model
    model.learn(total_timesteps=int(args.steps))

    # Save the final model
    model_path = os.path.join(outdir, 'dqn_final.zip')
    model.save(model_path)

    print(f"\n" + "="*50)
    print("DQN Training Complete!")
    print(f"  Model saved: {model_path}")
    print(f"  Results saved: {outdir}")
    print("="*50)

elif args.algorithm == 'dynaq':
    # Partner's work: Dyna-Q (model-based RL)
    # TODO: Partner needs to implement DynaQAgent in src/agents/dynaq_agent.py
    print("❌ Dyna-Q not yet implemented. This is your partner's algorithm.")
    print("For now, use --algorithm dqn or --algorithm ppo")
    raise NotImplementedError("Dyna-Q implementation required in src/agents/dynaq_agent.py")

else:
    raise ValueError(f"Unknown algorithm: {args.algorithm}")
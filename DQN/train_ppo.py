"""
Training script for custom PPO implementation on Crafter.

This script trains the PPO agent defined in src/agents/ppo_agent.py
from scratch (no stable-baselines3).
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import torch
import crafter

from src.agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Train custom PPO on Crafter')

    # Training
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps (default: 1M)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')

    # PPO hyperparameters
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy coefficient')

    # Logging & checkpoints
    parser.add_argument('--outdir', type=str, default='logs/ppo',
                       help='Output directory')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log_freq', type=int, default=2048,
                       help='Log metrics every N steps')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("PPO Training on Crafter")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Total steps: {args.steps:,}")
    print(f"Rollout size: {args.n_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs per update: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"GAE lambda: {args.gae_lambda}")
    print(f"Clip epsilon: {args.clip_epsilon}")
    print(f"Entropy coef: {args.entropy_coef}")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"ppo_baseline_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Create environment with Crafter recorder
    base_env = crafter.Env()
    env = crafter.Recorder(
        base_env,
        str(outdir),
        save_stats=True,
        save_video=False,
        save_episode=False
    )

    print(f"Environment created: {env}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")

    # Create PPO agent
    agent = PPOAgent(
        observation_shape=(3, 64, 64),
        num_actions=env.action_space.n,
        device=device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
    )

    print(f"\nAgent created:")
    print(f"  Parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    start_time = time.time()

    # Metrics tracking
    all_episode_rewards = []
    all_episode_lengths = []

    for step in range(args.steps):
        # Select action
        action = agent.act(obs, training=True)

        # Take step
        next_obs, reward, done, info = env.step(action)

        # Store transition
        agent.store_experience(obs, action, reward, next_obs, done)

        # Update tracking
        episode_reward += reward
        episode_length += 1
        obs = next_obs

        # Handle episode end
        if done:
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            episode_count += 1

            obs = env.reset()
            episode_reward = 0
            episode_length = 0

        # PPO update (when rollout buffer is full)
        metrics = agent.update()

        # Logging
        if metrics is not None:
            elapsed_time = time.time() - start_time
            fps = step / elapsed_time if elapsed_time > 0 else 0

            print(f"\nStep {step:,}/{args.steps:,} ({step/args.steps*100:.1f}%)")
            print(f"  Time: {elapsed_time/60:.1f}m | FPS: {fps:.1f}")
            print(f"  Updates: {metrics['n_updates']}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Clip fraction: {metrics['clip_fraction']:.3f}")

            if all_episode_rewards:
                recent_rewards = all_episode_rewards[-10:]
                recent_lengths = all_episode_lengths[-10:]
                print(f"  Episodes: {episode_count}")
                print(f"  Reward (last 10): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                print(f"  Length (last 10): {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}")

        # Save checkpoint
        if (step + 1) % args.save_freq == 0:
            checkpoint_path = outdir / f"checkpoint_{step+1}.pt"
            agent.save(str(checkpoint_path))
            print(f"\n✓ Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = outdir / "ppo_baseline_final.pt"
    agent.save(str(final_path))
    print(f"\n✓ Training complete! Final model saved: {final_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {args.steps:,}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    if all_episode_rewards:
        print(f"Mean episode reward: {np.mean(all_episode_rewards):.2f}")
        print(f"Mean episode length: {np.mean(all_episode_lengths):.1f}")
        print(f"Best episode reward: {np.max(all_episode_rewards):.2f}")

    env.close()


if __name__ == '__main__':
    main()



import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import torch
import crafter

from src.agents.ppo_agent import PPOAgent
from src.modules.icm import ICMModule


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO + ICM + Dual-Clip on Crafter')

    # Training
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps (default: 1M)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use (auto will detect MPS on Apple Silicon)')

    # PPO hyperparameters (from Improvement 2)
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='PPO learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                       help='Entropy coefficient')

    # DualClip PPO
    parser.add_argument('--dual_clip', type=float, default=3.0,
                       help='Dual-clip coefficient (3.0 is standard)')

    # ICM hyperparameters
    parser.add_argument('--icm_feature_dim', type=int, default=512,
                       help='ICM feature dimension')
    parser.add_argument('--icm_lr_inverse', type=float, default=1e-3,
                       help='ICM inverse model learning rate')
    parser.add_argument('--icm_lr_forward', type=float, default=1e-3,
                       help='ICM forward model learning rate')
    parser.add_argument('--icm_beta', type=float, default=0.2,
                       help='ICM intrinsic reward weight')
    parser.add_argument('--icm_lambda', type=float, default=0.1,
                       help='ICM inverse model loss weight')

    # Logging
    parser.add_argument('--outdir', type=str, default='logs/ppo_icm_dualclip',
                       help='Output directory')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save checkpoint every N steps')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("PPO + ICM + Dual-Clip Training on Crafter")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Total steps: {args.steps:,}")
    print("\n[PPO Hyperparameters]")
    print(f"  Learning rate: {args.lr}")
    print(f"  Entropy coef: {args.entropy_coef}")
    print(f"  Clip epsilon: {args.clip_epsilon}")
    print(f"  Dual-clip coefficient: {args.dual_clip} ⭐ NEW!")
    print("\n[ICM Hyperparameters]")
    print(f"  Beta (intrinsic weight): {args.icm_beta}")
    print(f"  Forward/Inverse LR: {args.icm_lr_forward}")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"ppo_icm_dualclip_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")

    # Create environment
    base_env = crafter.Env()
    env = crafter.Recorder(
        base_env,
        str(outdir),
        save_stats=True,
        save_video=False,
        save_episode=False
    )

    print(f"Environment created: {env}")

    # Create PPO agent with dual-clip
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
        dual_clip=args.dual_clip  # Enable dual-clip!
    )

    # Create ICM module
    icm = ICMModule(
        feature_dim=args.icm_feature_dim,
        num_actions=env.action_space.n,
        inverse_lr=args.icm_lr_inverse,
        forward_lr=args.icm_lr_forward,
        beta=args.icm_beta,
        lambda_inverse=args.icm_lambda,
        device=device
    )

    print(f"\nPPO Agent: {sum(p.numel() for p in agent.policy.parameters()):,} params")
    print(f"ICM Module: {sum(p.numel() for p in icm.parameters()):,} params")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    obs = env.reset()
    episode_reward = 0
    episode_extrinsic = 0
    episode_intrinsic = 0
    episode_length = 0
    episode_count = 0
    start_time = time.time()

    # Metrics tracking
    all_episode_rewards = []
    all_episode_extrinsic = []
    all_episode_intrinsic = []
    all_episode_lengths = []

    # ICM rewards log
    icm_rewards_file = outdir / 'icm_rewards.jsonl'
    icm_rewards_log = open(icm_rewards_file, 'w')

    for step in range(args.steps):
        # Select action
        action = agent.act(obs, training=True)

        # Take step
        next_obs, extrinsic_reward, done, info = env.step(action)

        # Compute intrinsic reward
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        next_obs_tensor = torch.from_numpy(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)

        intrinsic_reward_tensor = icm.compute_intrinsic_reward(obs_tensor, next_obs_tensor, action_tensor)
        intrinsic_reward = intrinsic_reward_tensor.item()

        # Combine rewards
        combined_reward_tensor = icm.get_combined_reward(
            torch.tensor([extrinsic_reward], device=device),
            intrinsic_reward_tensor
        )
        combined_reward = combined_reward_tensor.item()

        # Store transition
        agent.store_experience(obs, action, combined_reward, next_obs, done)

        # Update tracking
        episode_reward += combined_reward
        episode_extrinsic += extrinsic_reward
        episode_intrinsic += intrinsic_reward
        episode_length += 1
        obs = next_obs

        # Episode end
        if done:
            all_episode_rewards.append(episode_reward)
            all_episode_extrinsic.append(episode_extrinsic)
            all_episode_intrinsic.append(episode_intrinsic)
            all_episode_lengths.append(episode_length)
            episode_count += 1

            # Log to file
            icm_rewards_log.write(json.dumps({
                'episode': episode_count,
                'combined_reward': float(episode_reward),
                'extrinsic_reward': float(episode_extrinsic),
                'intrinsic_reward': float(episode_intrinsic),
                'episode_length': int(episode_length)
            }) + '\n')
            icm_rewards_log.flush()

            obs = env.reset()
            episode_reward = 0
            episode_extrinsic = 0
            episode_intrinsic = 0
            episode_length = 0

        # PPO update
        metrics = agent.update()

        # ICM update
        if step > 0 and step % args.batch_size == 0:
            buffer = agent.rollout_buffer
            if buffer.pos > args.batch_size:
                indices = np.random.choice(buffer.pos, size=args.batch_size, replace=False)
                obs_batch = torch.from_numpy(buffer.observations[indices]).to(device)
                next_indices = (indices + 1) % buffer.pos
                next_obs_batch = torch.from_numpy(buffer.observations[next_indices]).to(device)
                actions_batch = torch.from_numpy(buffer.actions[indices]).long().to(device)

                icm.update(obs_batch, next_obs_batch, actions_batch)

        # Logging
        if metrics is not None:
            elapsed_time = time.time() - start_time
            fps = step / elapsed_time if elapsed_time > 0 else 0

            print(f"\nStep {step:,}/{args.steps:,} ({step/args.steps*100:.1f}%)")
            print(f"  Time: {elapsed_time/60:.1f}m | FPS: {fps:.1f}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")

            if all_episode_rewards:
                recent_combined = all_episode_rewards[-10:]
                recent_extrinsic = all_episode_extrinsic[-10:]
                recent_intrinsic = all_episode_intrinsic[-10:]

                print(f"  Episodes: {episode_count}")
                print(f"  Combined reward: {np.mean(recent_combined):.2f}")
                print(f"  Extrinsic reward: {np.mean(recent_extrinsic):.2f}")
                print(f"  Intrinsic reward: {np.mean(recent_intrinsic):.2f}")

        # Save checkpoint
        if (step + 1) % args.save_freq == 0:
            checkpoint_path = outdir / f"checkpoint_{step+1}.pt"
            agent.save(str(checkpoint_path))
            print(f"\n Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = outdir / "ppo_icm_dualclip_final.pt"
    agent.save(str(final_path))

    icm_final_path = outdir / "icm_final.pt"
    torch.save({
        'feature_encoder': icm.feature_encoder.state_dict(),
        'inverse_model': icm.inverse_model.state_dict(),
        'forward_model': icm.forward_model.state_dict(),
    }, icm_final_path)

    print(f"\n Training complete!")
    print(f" PPO model saved: {final_path}")
    print(f" ICM model saved: {icm_final_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {args.steps:,}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")

    if all_episode_rewards:
        print(f"\nCombined reward: {np.mean(all_episode_rewards):.2f} (±{np.std(all_episode_rewards):.2f})")
        print(f"Extrinsic reward: {np.mean(all_episode_extrinsic):.2f} (±{np.std(all_episode_extrinsic):.2f})")
        print(f"Intrinsic reward: {np.mean(all_episode_intrinsic):.2f} (±{np.std(all_episode_intrinsic):.2f})")

    icm_rewards_log.close()
    print(f"\n ICM rewards logged to: {icm_rewards_file}")

    env.close()


if __name__ == '__main__':
    main()

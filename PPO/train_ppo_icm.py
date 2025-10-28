
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
    parser = argparse.ArgumentParser(description='Train PPO + ICM on Crafter')

    # Training
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps (default: 1M)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use (auto will detect MPS on Apple Silicon)')

    # PPO hyperparameters (from Improvement 1)
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='PPO learning rate (default: 5e-4 from Improvement 1)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                       help='Entropy coefficient (default: 0.001 from Improvement 1)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for actor/critic heads (default: 512)')

    # ICM hyperparameters
    parser.add_argument('--icm_feature_dim', type=int, default=512,
                       help='ICM feature dimension (default: 512)')
    parser.add_argument('--icm_lr_inverse', type=float, default=1e-3,
                       help='ICM inverse model learning rate')
    parser.add_argument('--icm_lr_forward', type=float, default=1e-3,
                       help='ICM forward model learning rate')
    parser.add_argument('--icm_beta', type=float, default=0.2,
                       help='ICM intrinsic reward weight (0.2 = 20% curiosity, 80% extrinsic)')
    parser.add_argument('--icm_lambda', type=float, default=0.1,
                       help='ICM inverse model loss weight')

    # Logging & checkpoints
    parser.add_argument('--outdir', type=str, default='logs/ppo_icm',
                       help='Output directory')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log_freq', type=int, default=2048,
                       help='Log metrics every N steps')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device (with MPS support for Apple Silicon)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("PPO + ICM Training on Crafter")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Total steps: {args.steps:,}")
    print("\n[PPO Hyperparameters]")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Rollout size: {args.n_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs per update: {args.n_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  GAE lambda: {args.gae_lambda}")
    print(f"  Clip epsilon: {args.clip_epsilon}")
    print(f"  Entropy coef: {args.entropy_coef}")
    print("\n[ICM Hyperparameters]")
    print(f"  Feature dim: {args.icm_feature_dim}")
    print(f"  Inverse LR: {args.icm_lr_inverse}")
    print(f"  Forward LR: {args.icm_lr_forward}")
    print(f"  Beta (intrinsic weight): {args.icm_beta} ({args.icm_beta*100:.0f}% curiosity, {(1-args.icm_beta)*100:.0f}% extrinsic)")
    print(f"  Lambda (inverse weight): {args.icm_lambda}")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / f"ppo_icm_{timestamp}"
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
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
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

    print(f"\nPPO Agent created:")
    print(f"  Parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")
    print(f"\nICM Module created:")
    print(f"  Feature encoder: {sum(p.numel() for p in icm.feature_encoder.parameters()):,} params")
    print(f"  Inverse model: {sum(p.numel() for p in icm.inverse_model.parameters()):,} params")
    print(f"  Forward model: {sum(p.numel() for p in icm.forward_model.parameters()):,} params")
    print(f"  Total ICM params: {sum(p.numel() for p in icm.parameters()):,}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    obs = env.reset()
    episode_reward = 0
    episode_extrinsic_reward = 0
    episode_intrinsic_reward = 0
    episode_length = 0
    episode_count = 0
    start_time = time.time()

    # Metrics tracking
    all_episode_rewards = []
    all_episode_extrinsic_rewards = []
    all_episode_intrinsic_rewards = []
    all_episode_lengths = []

    # ICM metrics tracking
    icm_inverse_losses = []
    icm_forward_losses = []
    icm_total_losses = []

    # Create ICM rewards log file
    icm_rewards_file = outdir / 'icm_rewards.jsonl'
    icm_rewards_log = open(icm_rewards_file, 'w')

    for step in range(args.steps):
        # Select action
        action = agent.act(obs, training=True)

        # Take step
        next_obs, extrinsic_reward, done, info = env.step(action)

        # Compute intrinsic reward from ICM
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        next_obs_tensor = torch.from_numpy(next_obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)

        intrinsic_reward_tensor = icm.compute_intrinsic_reward(obs_tensor, next_obs_tensor, action_tensor)
        intrinsic_reward = intrinsic_reward_tensor.item()

        # Combine rewards (weighted by beta)
        combined_reward_tensor = icm.get_combined_reward(
            torch.tensor([extrinsic_reward], device=device),
            intrinsic_reward_tensor
        )
        combined_reward = combined_reward_tensor.item()

        # Store transition (with combined reward for PPO)
        agent.store_experience(obs, action, combined_reward, next_obs, done)

        # Update tracking
        episode_reward += combined_reward
        episode_extrinsic_reward += extrinsic_reward
        episode_intrinsic_reward += intrinsic_reward
        episode_length += 1
        obs = next_obs

        # Handle episode end
        if done:
            all_episode_rewards.append(episode_reward)
            all_episode_extrinsic_rewards.append(episode_extrinsic_reward)
            all_episode_intrinsic_rewards.append(episode_intrinsic_reward)
            all_episode_lengths.append(episode_length)
            episode_count += 1

            # Log ICM rewards to file
            icm_rewards_log.write(json.dumps({
                'episode': episode_count,
                'combined_reward': float(episode_reward),
                'extrinsic_reward': float(episode_extrinsic_reward),
                'intrinsic_reward': float(episode_intrinsic_reward),
                'episode_length': int(episode_length)
            }) + '\n')
            icm_rewards_log.flush()

            obs = env.reset()
            episode_reward = 0
            episode_extrinsic_reward = 0
            episode_intrinsic_reward = 0
            episode_length = 0

        # PPO update (when rollout buffer is full)
        metrics = agent.update()

        # ICM update (every step, using recent experience)
        # Sample recent transitions from buffer for ICM update
        if step > 0 and step % args.batch_size == 0:
            # Get batch of transitions from PPO's buffer
            # We'll use the current rollout data for ICM training
            buffer = agent.rollout_buffer
            if buffer.pos > args.batch_size:
                # Sample indices
                indices = np.random.choice(buffer.pos, size=args.batch_size, replace=False)

                # Get observations, next observations, and actions
                obs_batch = torch.from_numpy(buffer.observations[indices]).to(device)
                # For next_obs, we need to shift indices by 1 (careful about buffer wraparound)
                next_indices = (indices + 1) % buffer.pos
                next_obs_batch = torch.from_numpy(buffer.observations[next_indices]).to(device)
                actions_batch = torch.from_numpy(buffer.actions[indices]).long().to(device)

                # Update ICM
                inv_loss, fwd_loss, total_loss = icm.update(obs_batch, next_obs_batch, actions_batch)
                icm_inverse_losses.append(inv_loss)
                icm_forward_losses.append(fwd_loss)
                icm_total_losses.append(total_loss)

        # Logging
        if metrics is not None:
            elapsed_time = time.time() - start_time
            fps = step / elapsed_time if elapsed_time > 0 else 0

            print(f"\nStep {step:,}/{args.steps:,} ({step/args.steps*100:.1f}%)")
            print(f"  Time: {elapsed_time/60:.1f}m | FPS: {fps:.1f}")
            print(f"  Updates: {metrics['n_updates']}")

            print(f"\n  [PPO Metrics]")
            print(f"    Policy loss: {metrics['policy_loss']:.4f}")
            print(f"    Value loss: {metrics['value_loss']:.4f}")
            print(f"    Entropy: {metrics['entropy']:.4f}")
            print(f"    Clip fraction: {metrics['clip_fraction']:.3f}")

            if icm_inverse_losses:
                print(f"\n  [ICM Metrics]")
                print(f"    Inverse loss: {np.mean(icm_inverse_losses[-10:]):.4f}")
                print(f"    Forward loss: {np.mean(icm_forward_losses[-10:]):.4f}")
                print(f"    Total loss: {np.mean(icm_total_losses[-10:]):.4f}")

            if all_episode_rewards:
                recent_rewards = all_episode_rewards[-10:]
                recent_extrinsic = all_episode_extrinsic_rewards[-10:]
                recent_intrinsic = all_episode_intrinsic_rewards[-10:]
                recent_lengths = all_episode_lengths[-10:]

                print(f"\n  [Episode Stats (last 10)]")
                print(f"    Episodes: {episode_count}")
                print(f"    Combined reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                print(f"    Extrinsic reward: {np.mean(recent_extrinsic):.2f} ± {np.std(recent_extrinsic):.2f}")
                print(f"    Intrinsic reward: {np.mean(recent_intrinsic):.2f} ± {np.std(recent_intrinsic):.2f}")
                print(f"    Episode length: {np.mean(recent_lengths):.1f} ± {np.std(recent_lengths):.1f}")

        # Save checkpoint
        if (step + 1) % args.save_freq == 0:
            checkpoint_path = outdir / f"checkpoint_{step+1}.pt"
            agent.save(str(checkpoint_path))

            # Save ICM separately
            icm_checkpoint_path = outdir / f"icm_checkpoint_{step+1}.pt"
            torch.save({
                'feature_encoder': icm.feature_encoder.state_dict(),
                'inverse_model': icm.inverse_model.state_dict(),
                'forward_model': icm.forward_model.state_dict(),
                'inverse_optimizer': icm.inverse_optimizer.state_dict(),
                'forward_optimizer': icm.forward_optimizer.state_dict(),
            }, icm_checkpoint_path)

            print(f"\n Saved checkpoint: {checkpoint_path}")
            print(f" Saved ICM checkpoint: {icm_checkpoint_path}")

    # Final save
    final_path = outdir / "ppo_icm_final.pt"
    agent.save(str(final_path))

    # Save ICM final
    icm_final_path = outdir / "icm_final.pt"
    torch.save({
        'feature_encoder': icm.feature_encoder.state_dict(),
        'inverse_model': icm.inverse_model.state_dict(),
        'forward_model': icm.forward_model.state_dict(),
        'inverse_optimizer': icm.inverse_optimizer.state_dict(),
        'forward_optimizer': icm.forward_optimizer.state_dict(),
    }, icm_final_path)

    print(f"\n Training complete!")
    print(f" PPO model saved: {final_path}")
    print(f" ICM model saved: {icm_final_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {args.steps:,}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    if all_episode_rewards:
        print(f"\nCombined reward: {np.mean(all_episode_rewards):.2f} (±{np.std(all_episode_rewards):.2f})")
        print(f"Extrinsic reward: {np.mean(all_episode_extrinsic_rewards):.2f} (±{np.std(all_episode_extrinsic_rewards):.2f})")
        print(f"Intrinsic reward: {np.mean(all_episode_intrinsic_rewards):.2f} (±{np.std(all_episode_intrinsic_rewards):.2f})")
        print(f"Mean episode length: {np.mean(all_episode_lengths):.1f}")
        print(f"Best combined reward: {np.max(all_episode_rewards):.2f}")
        print(f"Best extrinsic reward: {np.max(all_episode_extrinsic_rewards):.2f}")

    # Close ICM rewards log
    icm_rewards_log.close()
    print(f"\n ICM rewards logged to: {icm_rewards_file}")

    env.close()


if __name__ == '__main__':
    main()

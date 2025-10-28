"""
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_training_logs(log_dir):
    """
    Load training logs from the output directory.

    The training script prints episode-level statistics that we need to parse.
    We'll look for the stats.jsonl file from Crafter's recorder.
    """
    log_dir = Path(log_dir)
    stats_file = log_dir / 'stats.jsonl'

    if not stats_file.exists():
        print(f"Warning: {stats_file} not found")
        return None

    episodes = []
    with open(stats_file, 'r') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    return episodes


def parse_stdout_logs(log_dir):
    """
    Parse stdout logs to extract intrinsic/extrinsic reward breakdown.

    The training script prints episode stats including:
    - Combined reward
    - Extrinsic reward
    - Intrinsic reward

    We need to capture these from the console output or save them separately.
    """
    # For now, we'll create a simple tracking mechanism
    # In practice, these would be saved during training
    pass


def create_reward_tracking_file(log_dir):
    """
    Create a simple JSON file to track rewards during training.

    This should be called from the training script to save:
    - episode_number
    - combined_reward
    - extrinsic_reward
    - intrinsic_reward
    - episode_length
    """
    log_dir = Path(log_dir)
    rewards_file = log_dir / 'icm_rewards.jsonl'
    return rewards_file


def plot_icm_rewards(log_dir, outdir=None, window=10):
    """
    Plot intrinsic vs extrinsic rewards over training.

    Args:
        log_dir: Directory containing training logs
        outdir: Output directory for plots (default: log_dir)
        window: Moving average window size
    """
    log_dir = Path(log_dir)
    if outdir is None:
        outdir = log_dir
    else:
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load ICM reward logs
    rewards_file = log_dir / 'icm_rewards.jsonl'
    if not rewards_file.exists():
        print(f"Error: {rewards_file} not found")
        print("Make sure the training script saves ICM reward logs!")
        return

    # Parse rewards
    episodes = []
    combined_rewards = []
    extrinsic_rewards = []
    intrinsic_rewards = []
    episode_lengths = []

    with open(rewards_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                episodes.append(data['episode'])
                combined_rewards.append(data['combined_reward'])
                extrinsic_rewards.append(data['extrinsic_reward'])
                intrinsic_rewards.append(data['intrinsic_reward'])
                episode_lengths.append(data['episode_length'])

    episodes = np.array(episodes)
    combined_rewards = np.array(combined_rewards)
    extrinsic_rewards = np.array(extrinsic_rewards)
    intrinsic_rewards = np.array(intrinsic_rewards)
    episode_lengths = np.array(episode_lengths)

    # Compute moving averages
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    ma_combined = moving_average(combined_rewards, window)
    ma_extrinsic = moving_average(extrinsic_rewards, window)
    ma_intrinsic = moving_average(intrinsic_rewards, window)
    ma_episodes = episodes[window-1:]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ICM Training Analysis: Intrinsic vs Extrinsic Rewards', fontsize=16, fontweight='bold')

    # Plot 1: Combined vs Components
    ax = axes[0, 0]
    ax.plot(ma_episodes, ma_combined, label='Combined', linewidth=2, color='purple', alpha=0.8)
    ax.plot(ma_episodes, ma_extrinsic, label='Extrinsic (Environment)', linewidth=2, color='blue', alpha=0.8)
    ax.plot(ma_episodes, ma_intrinsic, label='Intrinsic (Curiosity)', linewidth=2, color='orange', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Reward (MA-{window})')
    ax.set_title('Reward Components Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Intrinsic/Extrinsic Ratio
    ax = axes[0, 1]
    ratio = ma_intrinsic / (ma_extrinsic + 1e-8)  # Avoid division by zero
    ax.plot(ma_episodes, ratio, linewidth=2, color='green', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal contribution')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Intrinsic / Extrinsic Ratio')
    ax.set_title('Curiosity vs Achievement Balance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Intrinsic Reward Decay
    ax = axes[1, 0]
    ax.plot(ma_episodes, ma_intrinsic, linewidth=2, color='orange', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Intrinsic Reward (MA-{window})')
    ax.set_title('Curiosity Signal Decay (Learning Progress)')
    ax.grid(True, alpha=0.3)

    # Add annotation
    if len(ma_intrinsic) > 0:
        early_avg = np.mean(ma_intrinsic[:len(ma_intrinsic)//4]) if len(ma_intrinsic) >= 4 else ma_intrinsic[0]
        late_avg = np.mean(ma_intrinsic[-len(ma_intrinsic)//4:]) if len(ma_intrinsic) >= 4 else ma_intrinsic[-1]
        decay_pct = ((early_avg - late_avg) / early_avg * 100) if early_avg > 0 else 0
        ax.text(0.05, 0.95, f'Decay: {decay_pct:.1f}%\nEarly: {early_avg:.3f}\nLate: {late_avg:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Extrinsic Reward Growth
    ax = axes[1, 1]
    ax.plot(ma_episodes, ma_extrinsic, linewidth=2, color='blue', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Extrinsic Reward (MA-{window})')
    ax.set_title('Achievement Progress (Main Objective)')
    ax.grid(True, alpha=0.3)

    # Add annotation
    if len(ma_extrinsic) > 0:
        early_avg = np.mean(ma_extrinsic[:len(ma_extrinsic)//4]) if len(ma_extrinsic) >= 4 else ma_extrinsic[0]
        late_avg = np.mean(ma_extrinsic[-len(ma_extrinsic)//4:]) if len(ma_extrinsic) >= 4 else ma_extrinsic[-1]
        growth_pct = ((late_avg - early_avg) / (early_avg + 1e-8) * 100)
        ax.text(0.05, 0.95, f'Growth: {growth_pct:.1f}%\nEarly: {early_avg:.3f}\nLate: {late_avg:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Save plot
    plot_path = outdir / 'icm_reward_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("ICM Reward Statistics")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print(f"\nCombined reward: {np.mean(combined_rewards):.3f} ± {np.std(combined_rewards):.3f}")
    print(f"Extrinsic reward: {np.mean(extrinsic_rewards):.3f} ± {np.std(extrinsic_rewards):.3f}")
    print(f"Intrinsic reward: {np.mean(intrinsic_rewards):.3f} ± {np.std(intrinsic_rewards):.3f}")
    print(f"\nIntrinsic/Extrinsic ratio: {np.mean(intrinsic_rewards) / (np.mean(extrinsic_rewards) + 1e-8):.3f}")

    # Show plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ICM reward analysis')
    parser.add_argument('log_dir', type=str, help='Directory containing training logs')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--window', type=int, default=10, help='Moving average window')

    args = parser.parse_args()
    plot_icm_rewards(args.log_dir, args.outdir, args.window)

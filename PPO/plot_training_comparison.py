#!/usr/bin/env python3


import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_stats(stats_path: Path) -> Dict[str, List]:
    """
    Load training statistics from stats.jsonl file.

    Args:
        stats_path: Path to stats.jsonl file

    Returns:
        Dictionary with lists of training metrics
    """
    stats = {
        'episode': [],
        'reward': [],
        'length': [],
    }

    if not stats_path.exists():
        print(f"Warning: {stats_path} not found")
        return stats

    episode_num = 0
    with open(stats_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                stats['episode'].append(episode_num)
                stats['reward'].append(data.get('reward', 0))
                stats['length'].append(data.get('length', 0))
                episode_num += 1

    return stats


def smooth_curve(data: List[float], window: int = 10) -> np.ndarray:
    """
    Smooth a curve using moving average.

    Args:
        data: List of values to smooth
        window: Window size for moving average

    Returns:
        Smoothed numpy array
    """
    if len(data) < window:
        return np.array(data)

    data_array = np.array(data)
    smoothed = np.convolve(data_array, np.ones(window)/window, mode='valid')
    return smoothed


def plot_comparison(experiments: Dict[str, Path], output_dir: Path):
    """
    Create comparison plots for multiple experiments.

    Args:
        experiments: Dictionary mapping experiment names to their log directories
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up nice plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Training Comparison Across Experiments', fontsize=16, fontweight='bold')

    # Load all experiments
    all_stats = {}
    for name, log_dir in experiments.items():
        stats_file = log_dir / 'stats.jsonl'
        stats = load_stats(stats_file)
        if stats['episode']:  # Only add if data exists
            all_stats[name] = stats
            print(f"✓ Loaded {name}: {len(stats['episode'])} episodes")
        else:
            print(f"✗ Skipped {name}: No data found")

    if not all_stats:
        print("ERROR: No training data found!")
        return

    # Plot 1: Episode Reward over Episodes
    ax1 = axes[0, 0]
    for idx, (name, stats) in enumerate(all_stats.items()):
        color = colors[idx % len(colors)]
        episodes = stats['episode']
        rewards = stats['reward']

        # Plot raw data with transparency
        ax1.plot(episodes, rewards, alpha=0.2, color=color)

        # Plot smoothed curve
        if len(rewards) > 100:
            smoothed = smooth_curve(rewards, window=100)
            smooth_episodes = episodes[49:-50]  # Adjust for convolution edge effects
            ax1.plot(smooth_episodes, smoothed, label=name, linewidth=2, color=color)
        else:
            ax1.plot(episodes, rewards, label=name, linewidth=2, color=color)

    ax1.set_xlabel('Episode Number', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Episode Reward Over Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Episode Length over Episodes
    ax2 = axes[0, 1]
    for idx, (name, stats) in enumerate(all_stats.items()):
        color = colors[idx % len(colors)]
        episodes = stats['episode']
        lengths = stats['length']

        # Plot smoothed curve
        if len(lengths) > 100:
            smoothed = smooth_curve(lengths, window=100)
            smooth_episodes = episodes[49:-50]
            ax2.plot(smooth_episodes, smoothed, label=name, linewidth=2, color=color)
        else:
            ax2.plot(episodes, lengths, label=name, linewidth=2, color=color)

    ax2.set_xlabel('Episode Number', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length Over Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Reward over Episodes
    ax3 = axes[1, 0]
    for idx, (name, stats) in enumerate(all_stats.items()):
        color = colors[idx % len(colors)]
        episodes = stats['episode']
        rewards = stats['reward']
        cumulative = np.cumsum(rewards)

        ax3.plot(episodes, cumulative, label=name, linewidth=2, color=color)

    ax3.set_xlabel('Episode Number', fontsize=12)
    ax3.set_ylabel('Cumulative Reward', fontsize=12)
    ax3.set_title('Cumulative Reward Over Training', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics (Bar Chart)
    ax4 = axes[1, 1]
    names = list(all_stats.keys())
    mean_rewards = [np.mean(stats['reward']) for stats in all_stats.values()]

    bars = ax4.barh(names, mean_rewards, color=colors[:len(names)])
    ax4.set_xlabel('Mean Episode Reward', fontsize=12)
    ax4.set_title('Mean Reward Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for bar, value in zip(bars, mean_rewards):
        ax4.text(value, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}', va='center', ha='left', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_path = output_dir / 'training_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: {output_path}")

    # Create summary statistics table
    print("\n" + "="*80)
    print("TRAINING SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Experiment':<30} {'Mean Reward':<15} {'Final Reward':<15} {'Episodes':<10}")
    print("-"*80)

    for name, stats in all_stats.items():
        mean_reward = np.mean(stats['reward'])
        final_reward = np.mean(stats['reward'][-10:]) if len(stats['reward']) >= 10 else np.mean(stats['reward'])
        num_episodes = len(stats['reward'])
        print(f"{name:<30} {mean_reward:>14.2f} {final_reward:>14.2f} {num_episodes:>9}")

    print("="*80)

    plt.show()


def main():
    """Main function to create training comparison plots."""
    parser = argparse.ArgumentParser(description='Plot training statistics comparison')
    parser.add_argument('--output', type=str, default='plots/training_comparison',
                        help='Output directory for plots')
    args = parser.parse_args()

    # Define experiments to compare
    base_dir = Path('logs')
    experiments = {
        'Baseline (lr=3e-4)': base_dir / 'ppo_m4' / 'ppo_baseline_20251008_133158',
        'lr=1e-4': base_dir / 'ppo_lr1e4' / 'ppo_baseline_20251009_201112',
        'lr=1e-4 + entropy=0.015': base_dir / 'ppo_lr1e4_ent015' / 'ppo_baseline_20251009_225337',
        'Batch size 128': base_dir / 'ppo_batch128' / 'ppo_baseline_20251009_142511',
        'GAE λ=0.98': base_dir / 'ppo_gae098' / 'ppo_baseline_20251010_144829',
        'GAE λ=0.97': base_dir / 'ppo_gae097' / 'ppo_baseline_20251010_171915',
        'Deeper CNN (4 layers)': base_dir / 'ppo_deep_cnn' / 'ppo_baseline_20251010_225541',
    }

    # Check which experiments exist
    existing_experiments = {}
    for name, path in experiments.items():
        if path.exists():
            existing_experiments[name] = path
        else:
            print(f"Warning: {name} not found at {path}")

    if not existing_experiments:
        print("ERROR: No experiment directories found!")
        print(f"Searched in: {base_dir}")
        return

    print(f"\nFound {len(existing_experiments)} experiments to compare:")
    for name in existing_experiments.keys():
        print(f"  ✓ {name}")

    # Create comparison plots
    output_dir = Path(args.output)
    plot_comparison(existing_experiments, output_dir)


if __name__ == '__main__':
    main()

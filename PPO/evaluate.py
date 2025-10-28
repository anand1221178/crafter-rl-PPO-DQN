#!/usr/bin/env python3
"""
Unified Crafter Model Evaluation Script

This script combines all analysis functionality from the Crafter repository
to evaluate trained models and generate comprehensive performance reports.

Usage:
    python evaluate.py --model_path models/ppo_model.zip --algorithm ppo --episodes 100
    python evaluate.py --logdir logdir/crafter_dqn_20251005_180000/ --algorithm dqn --episodes 100
    python evaluate.py --logdir logdir/crafter_dynaq_20251005_180000/ --algorithm dynaq --episodes 100
"""

import argparse
import collections
import json
import os
import pathlib
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gym as old_gym
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register

# Import stable baselines for PPO evaluation
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: Stable Baselines3 not available. PPO evaluation may not work.")

# Import custom agents
try:
    from src.agents.dynaq_agent import DynaQAgent
    HAS_DYNAQ = True
except ImportError:
    HAS_DYNAQ = False

from src.agents.ppo_agent import PPOAgent
import torch


class CrafterEvaluator:
    """Unified evaluator for Crafter agents with comprehensive analysis."""

    def __init__(self, algorithm='ppo', episodes=100, budget=1e6):
        self.algorithm = algorithm
        self.episodes = episodes
        self.budget = int(budget)
        self.setup_environment()

    def setup_environment(self):
        """Setup Crafter environment for evaluation (direct, no wrappers)."""
        # Use Crafter directly without Gym wrappers to avoid API conflicts
        self.env = crafter.Env()

    def load_model(self, model_path):
        """Load trained model based on algorithm type."""
        if self.algorithm == 'ppo':
            # Try custom PPO first (our implementation)
            if model_path.endswith('.pt'):
                device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

                # Load checkpoint to infer hidden_dim from model architecture
                checkpoint = torch.load(model_path, map_location=device)
                # Extract hidden_dim from actor.0.weight shape: [hidden_dim, 4096]
                hidden_dim = checkpoint['policy_state_dict']['actor.0.weight'].shape[0]

                agent = PPOAgent(
                    observation_shape=(3, 64, 64),
                    num_actions=17,
                    device=device,
                    hidden_dim=hidden_dim
                )
                agent.load(model_path)
                return agent
            # Otherwise try Stable-Baselines3 PPO
            elif HAS_SB3:
                return PPO.load(model_path)
            else:
                raise ImportError("Neither custom PPO checkpoint (.pt) nor Stable Baselines3 available")
        elif self.algorithm == 'dqn':
            if not HAS_SB3:
                raise ImportError("Stable Baselines3 required for DQN evaluation")
            from stable_baselines3 import DQN
            return DQN.load(model_path)
        elif self.algorithm == 'dynaq':
            if not HAS_DYNAQ:
                raise ImportError("DynaQAgent not available")
            # Load Dyna-Q agent
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            agent = DynaQAgent(
                observation_shape=(64, 64, 3),
                num_actions=17,
                device=device
            )
            agent.load(model_path)
            return agent
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def evaluate_model(self, model_path, outdir):
        """Run evaluation episodes and collect metrics."""
        print(f"\n🎯 Evaluating {self.algorithm.upper()} model...")
        print(f"Model: {model_path}")
        print(f"Episodes: {self.episodes}")
        print(f"Output: {outdir}")

        # Load model
        model = self.load_model(model_path)

        # Create output directory with recording
        os.makedirs(outdir, exist_ok=True)
        env = crafter.Recorder(
            self.env,
            outdir,
            save_stats=True,
            save_video=False,
            save_episode=False
        )

        # Run evaluation episodes
        print(f"\nRunning {self.episodes} evaluation episodes...")
        for episode in range(self.episodes):
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Get action based on model type
                if hasattr(model, 'predict'):
                    # Stable-Baselines3 models (PPO, DQN)
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Custom agents (PPOAgent, DynaQAgent)
                    action = model.act(obs, training=False)

                # Handle both Gym APIs (4-tuple vs 5-tuple)
                step_result = env.step(action)
                if len(step_result) == 5:
                    # New Gymnasium API: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    # Old Gym API: (obs, reward, done, info)
                    obs, reward, done, info = step_result

                episode_reward += reward

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{self.episodes} completed")

        env.close()
        print("✅ Evaluation complete!")
        return outdir

    def load_stats(self, filename, budget):
        """Load statistics from Crafter stats.jsonl file."""
        steps = 0
        rewards = []
        lengths = []
        achievements = collections.defaultdict(list)

        for line in filename.read_text().split('\n'):
            if not line.strip():
                continue
            episode = json.loads(line)
            steps += episode['length']
            if steps > budget:
                break
            lengths.append(episode['length'])
            for key, value in episode.items():
                if key.startswith('achievement_'):
                    achievements[key].append(value)
            unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
            health = -0.9
            rewards.append(unlocks + health)
        return rewards, lengths, achievements

    def compute_success_rates(self, runs, budget=1e6):
        """Compute achievement success rates from runs."""
        methods = sorted(set(run['method'] for run in runs))
        seeds = sorted(set(run['seed'] for run in runs))
        tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))
        percents = np.empty((len(methods), len(seeds), len(tasks)))
        percents[:] = np.nan

        for run in runs:
            episodes = (np.array(run['xs']) <= budget).sum()
            i = methods.index(run['method'])
            j = seeds.index(run['seed'])
            for key, values in run.items():
                if key in tasks:
                    k = tasks.index(key)
                    percent = 100 * (np.array(values[:episodes]) >= 1).mean()
                    percents[i, j, k] = percent
        return percents, methods, seeds, tasks

    def compute_scores(self, percents):
        """Compute geometric mean scores with 1% offset."""
        assert (0 <= percents).all() and (percents <= 100).all()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
        return scores

    def analyze_results(self, logdir):
        """Analyze results from evaluation run."""
        logdir = pathlib.Path(logdir)
        stats_file = logdir / 'stats.jsonl'

        if not stats_file.exists():
            raise FileNotFoundError(f"No stats.jsonl found in {logdir}")

        print(f"\n📊 Analyzing results from {logdir.name}...")

        # Load statistics
        rewards, lengths, achievements = self.load_stats(stats_file, self.budget)

        # Create run data structure
        runs = [dict(
            method=self.algorithm,
            seed='0',
            xs=np.cumsum(lengths).tolist(),
            reward=rewards,
            length=lengths,
            **achievements,
        )]

        # Compute metrics
        percents, methods, seeds, tasks = self.compute_success_rates(runs, self.budget)
        scores = self.compute_scores(percents)

        # Print summary
        episodes = np.array([len(run['length']) for run in runs])
        avg_rewards = np.array([np.mean(run['reward']) for run in runs])
        avg_lengths = np.array([np.mean(run['length']) for run in runs])

        print("\n" + "="*50)
        print(f"📈 EVALUATION RESULTS - {self.algorithm.upper()}")
        print("="*50)
        print(f"Crafter Score:    {np.mean(scores):8.2f} ± {np.std(scores):.2f}%")
        print(f"Average Reward:   {np.mean(avg_rewards):8.2f} ± {np.std(avg_rewards):.2f}")
        print(f"Average Length:   {np.mean(avg_lengths):8.2f} ± {np.std(avg_lengths):.2f}")
        print(f"Total Episodes:   {np.mean(episodes):8.0f}")
        print("-"*50)

        # Achievement breakdown
        print("🏆 ACHIEVEMENT UNLOCK RATES:")
        print("-"*50)
        for task, percent in zip(tasks, np.squeeze(percents).T):
            name = task[len('achievement_'):].replace('_', ' ').title()
            print(f"{name:<20}  {np.mean(percent):6.2f}%")

        return {
            'score': float(np.mean(scores)),
            'reward': float(np.mean(avg_rewards)),
            'length': float(np.mean(avg_lengths)),
            'episodes': float(np.mean(episodes)),
            'achievements': {task: float(np.mean(percent)) for task, percent in zip(tasks, np.squeeze(percents).T)}
        }

    def generate_plots(self, logdir, results):
        """Generate evaluation plots."""
        logdir = pathlib.Path(logdir)
        plots_dir = logdir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        print(f"\n📊 Generating plots in {plots_dir}...")

        # Achievement success rate plot
        tasks = list(results['achievements'].keys())
        rates = list(results['achievements'].values())

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))
        bars = ax.bar(range(len(tasks)), rates, color=colors)

        ax.set_title(f'{self.algorithm.upper()} Achievement Success Rates')
        ax.set_xlabel('Achievement')
        ax.set_ylabel('Success Rate (%)')
        ax.set_xticks(range(len(tasks)))
        task_names = [task[len('achievement_'):].replace('_', ' ').title() for task in tasks]
        ax.set_xticklabels(task_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{rate:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = plots_dir / 'achievement_rates.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📈 Saved {plot_path}")

        # Summary metrics plot
        metrics = ['Score', 'Avg Reward', 'Avg Length']
        values = [results['score'], results['reward'], results['length']]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(f'{self.algorithm.upper()} Summary Metrics')
        ax.set_ylabel('Value')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                   f'{value:.1f}', ha='center', va='center',
                   color='white', fontweight='bold')

        plt.tight_layout()
        plot_path = plots_dir / 'summary_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved {plot_path}")

    def save_report(self, logdir, results):
        """Save detailed evaluation report."""
        logdir = pathlib.Path(logdir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            'algorithm': self.algorithm,
            'evaluation_date': timestamp,
            'episodes_evaluated': self.episodes,
            'budget': self.budget,
            'results': results,
            'summary': {
                'crafter_score': results['score'],
                'geometric_mean_achievements': results['score'],
                'average_reward_per_episode': results['reward'],
                'average_episode_length': results['length'],
                'total_episodes': results['episodes']
            }
        }

        # Save JSON report
        report_path = logdir / f'evaluation_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Saved detailed report: {report_path}")

        # Save human-readable summary
        summary_path = logdir / f'evaluation_summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write(f"CRAFTER EVALUATION REPORT - {self.algorithm.upper()}\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("="*60 + "\n\n")

            f.write("SUMMARY METRICS:\n")
            f.write(f"  Crafter Score:      {results['score']:8.2f}%\n")
            f.write(f"  Average Reward:     {results['reward']:8.2f}\n")
            f.write(f"  Average Length:     {results['length']:8.2f}\n")
            f.write(f"  Episodes Evaluated: {results['episodes']:8.0f}\n\n")

            f.write("ACHIEVEMENT UNLOCK RATES:\n")
            for task, rate in results['achievements'].items():
                name = task[len('achievement_'):].replace('_', ' ').title()
                f.write(f"  {name:<20} {rate:6.2f}%\n")

        print(f"📝 Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Crafter RL agents')
    parser.add_argument('--model_path', type=str, help='Path to trained model file (.zip)')
    parser.add_argument('--logdir', type=str, help='Path to training logdir (alternative to model_path)')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn', 'dynaq'],
                       required=True, help='Algorithm type')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--budget', type=float, default=1e6,
                       help='Step budget for analysis')
    parser.add_argument('--outdir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')

    args = parser.parse_args()

    # Validate arguments
    if not args.model_path and not args.logdir:
        parser.error("Either --model_path or --logdir must be provided")

    # Create evaluator
    evaluator = CrafterEvaluator(
        algorithm=args.algorithm,
        episodes=args.episodes,
        budget=args.budget
    )

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"{args.outdir}_{args.algorithm}_{timestamp}"

    try:
        if args.model_path:
            # Evaluate from model file
            evaluation_dir = evaluator.evaluate_model(args.model_path, outdir)
            results = evaluator.analyze_results(evaluation_dir)
        else:
            # Analyze existing logdir
            results = evaluator.analyze_results(args.logdir)
            evaluation_dir = args.logdir

        # Generate plots and reports
        evaluator.generate_plots(evaluation_dir, results)
        evaluator.save_report(evaluation_dir, results)

        print(f"\n✅ Evaluation complete! Results saved to {evaluation_dir}")
        print(f"🏆 Final Crafter Score: {results['score']:.2f}%")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
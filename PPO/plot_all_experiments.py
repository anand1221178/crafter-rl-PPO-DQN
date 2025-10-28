"""

Creates comprehensive visualizations showing:
1. Crafter score progression across all evaluations
2. Achievement rates comparison
3. Failed vs successful attempts
4. Training time analysis


"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_evaluation_results(result_dir):
    """Load evaluation results from JSON file."""
    result_path = Path(result_dir)

    # Find the evaluation report JSON
    json_files = list(result_path.glob('evaluation_report_*.json'))
    if not json_files:
        raise FileNotFoundError(f"No evaluation report found in {result_dir}")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    return data


def plot_1_crafter_score_progression(successful_evals, failed_evals, output_dir):
    """
    Figure 1: Main progression plot showing all attempts.
    Shows the improvement trajectory with successful vs failed experiments.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Successful evaluations
    success_names = [e['name'] for e in successful_evals]
    success_scores = [e['score'] for e in successful_evals]
    success_x = list(range(len(success_scores)))

    # Plot successful progression line
    ax.plot(success_x, success_scores, 'o-', linewidth=2.5, markersize=10,
            color='#2ecc71', label='Successful Evaluations', zorder=3)

    # Add value labels on successful points
    for i, (x, y) in enumerate(zip(success_x, success_scores)):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontweight='bold',
                   fontsize=10, color='#27ae60')

    # Failed attempts as scatter points
    if failed_evals:
        failed_names = [e['name'] for e in failed_evals]
        failed_scores = [e['score'] for e in failed_evals]
        # Position failed attempts between successful ones
        failed_x = [i + 0.5 for i in range(len(failed_scores))]

        ax.scatter(failed_x, failed_scores, s=100, marker='x',
                  color='#e74c3c', linewidths=2.5,
                  label='Failed Attempts', zorder=2, alpha=0.7)

        # Add labels for failed attempts
        for i, (x, y, name) in enumerate(zip(failed_x, failed_scores, failed_names)):
            ax.annotate(name, (x, y), textcoords="offset points",
                       xytext=(0, -15), ha='center', fontsize=7,
                       color='#c0392b', style='italic')

    # Styling
    ax.set_xlabel('Evaluation Number', fontweight='bold')
    ax.set_ylabel('Crafter Score (%)', fontweight='bold')
    ax.set_title('PPO Agent Performance Progression on Crafter\n(Successful Evaluations and Failed Attempts)',
                fontweight='bold', pad=20)
    ax.set_xticks(success_x)
    ax.set_xticklabels([f'Eval {i+1}' for i in success_x])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.95)

    # Add improvement annotations
    for i in range(len(success_scores) - 1):
        improvement = ((success_scores[i+1] - success_scores[i]) / success_scores[i]) * 100
        mid_x = success_x[i] + 0.5
        mid_y = (success_scores[i] + success_scores[i+1]) / 2
        ax.annotate(f'+{improvement:.1f}%', (mid_x, mid_y),
                   fontsize=8, color='#16a085', style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#16a085', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_score_progression.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_score_progression.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig1_score_progression.png/pdf")
    plt.close()


def plot_2_achievement_comparison(evaluations, output_dir):
    """
    Figure 2: Achievement rates comparison across all successful evaluations.
    Grouped bar chart showing key achievements.
    """
    # Key achievements to highlight
    key_achievements = [
        'achievement_collect_wood', 'achievement_collect_sapling', 'achievement_place_table',
        'achievement_make_wood_pickaxe', 'achievement_make_wood_sword',
        'achievement_defeat_zombie', 'achievement_collect_coal', 'achievement_defeat_skeleton'
    ]

    achievement_labels = [
        'Wood', 'Sapling', 'Table',
        'Wood Pickaxe', 'Wood Sword',
        'Zombie', 'Coal', 'Skeleton'
    ]

    # Prepare data
    n_evals = len(evaluations)
    n_achievements = len(key_achievements)

    data = []
    for eval_data in evaluations:
        achievements = eval_data['data']['results']['achievements']
        rates = [achievements.get(ach, 0) for ach in key_achievements]
        data.append(rates)

    data = np.array(data).T  # Shape: (n_achievements, n_evals)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_achievements)
    width = 0.2
    colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']

    for i, (eval_data, color) in enumerate(zip(evaluations, colors)):
        offset = width * (i - n_evals/2 + 0.5)
        bars = ax.bar(x + offset, data[:, i], width, label=eval_data['name'],
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Achievement Type', fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Achievement Unlock Rates Across Evaluations', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(achievement_labels, rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_achievement_comparison.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_achievement_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig2_achievement_comparison.png/pdf")
    plt.close()


def plot_3_summary_metrics(evaluations, output_dir):
    """
    Figure 3: Summary metrics comparison (score, reward, episode length).
    Three subplots showing key performance indicators.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    names = [e['name'] for e in evaluations]
    scores = [e['score'] for e in evaluations]
    rewards = [e['data']['results']['reward'] for e in evaluations]
    lengths = [e['data']['results']['length'] for e in evaluations]

    colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']
    x = np.arange(len(names))

    # Plot 1: Crafter Score
    bars1 = axes[0].bar(x, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Crafter Score (%)', fontweight='bold')
    axes[0].set_title('Crafter Score', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    for bar, score in zip(bars1, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Average Reward
    bars2 = axes[1].bar(x, rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Average Reward', fontweight='bold')
    axes[1].set_title('Episode Reward', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    for bar, reward in zip(bars2, rewards):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{reward:.1f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Episode Length
    bars3 = axes[2].bar(x, lengths, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[2].set_ylabel('Average Episode Length', fontweight='bold')
    axes[2].set_title('Survival Time', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
    for bar, length in zip(bars3, lengths):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{length:.0f}', ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Summary Performance Metrics', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_summary_metrics.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_summary_metrics.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig3_summary_metrics.png/pdf")
    plt.close()


def plot_4_failed_attempts_analysis(failed_evals, output_dir):
    """
    Figure 4: Analysis of failed attempts showing why different strategies didn't work.
    """
    if not failed_evals:
        print("⚠ No failed attempts to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    names = [e['name'] for e in failed_evals]
    scores = [e['score'] for e in failed_evals]

    # Sort by score
    sorted_indices = np.argsort(scores)
    names = [names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    # Color code by score (red = bad, yellow = close)
    colors = ['#e74c3c' if s < 7.0 else '#f39c12' if s < 8.2 else '#f1c40f'
              for s in scores]

    y = np.arange(len(names))
    bars = ax.barh(y, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add baseline reference line
    baseline_score = 8.27
    ax.axvline(baseline_score, color='#2ecc71', linestyle='--', linewidth=2,
              label=f'Baseline to Beat ({baseline_score}%)', zorder=0)

    # Add value labels
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}%', va='center', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Crafter Score (%)', fontweight='bold')
    ax.set_title('Failed Improvement Attempts\n(Strategies That Did Not Beat Baseline)',
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_failed_attempts.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_failed_attempts.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig4_failed_attempts.png/pdf")
    plt.close()


def plot_5_improvement_breakdown(evaluations, output_dir):
    """
    Figure 5: Stacked improvement showing contribution of each evaluation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [e['name'] for e in evaluations]
    scores = [e['score'] for e in evaluations]

    # Calculate incremental improvements
    baseline = scores[0]
    improvements = [0]  # First eval is baseline
    for i in range(1, len(scores)):
        improvement = scores[i] - scores[i-1]
        improvements.append(improvement)

    # Create stacked bar
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#2ecc71']
    bottom = 0

    for i, (name, improvement, color) in enumerate(zip(names, improvements, colors)):
        if i == 0:
            # Baseline
            bar = ax.bar(0, scores[0], color=color, alpha=0.8,
                        edgecolor='black', linewidth=1, label=name)
            ax.text(0, scores[0]/2, f'{scores[0]:.2f}%',
                   ha='center', va='center', fontweight='bold', fontsize=12)
            bottom = scores[0]
        else:
            # Improvements
            bar = ax.bar(0, improvement, bottom=bottom, color=color, alpha=0.8,
                        edgecolor='black', linewidth=1, label=name)
            if improvement > 0.1:
                ax.text(0, bottom + improvement/2,
                       f'+{improvement:.2f}%\n({name})',
                       ha='center', va='center', fontweight='bold', fontsize=9)
            bottom += improvement

    # Add total at top
    ax.text(0, bottom + 0.3, f'Total: {bottom:.2f}%',
           ha='center', va='bottom', fontweight='bold', fontsize=13,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))

    # Calculate total improvement percentage
    total_improvement = ((scores[-1] - scores[0]) / scores[0]) * 100
    ax.text(0, -0.5, f'Total Improvement: +{total_improvement:.1f}%',
           ha='center', va='top', fontweight='bold', fontsize=11,
           color='#27ae60')

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, bottom + 1)
    ax.set_xticks([])
    ax.set_ylabel('Crafter Score (%)', fontweight='bold')
    ax.set_title('Cumulative Improvement Breakdown\n(Contribution of Each Evaluation)',
                fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_improvement_breakdown.png', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_improvement_breakdown.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig5_improvement_breakdown.png/pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate all experiment comparison plots')
    parser.add_argument('--output_dir', type=str, default='results/paper_figures',
                       help='Output directory for figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication-Quality Figures for IEEE Report")
    print("=" * 60)

    # Define all experiments (manual configuration for now)
    # You can modify these paths to match your actual result directories

    successful_evaluations = [
        {
            'name': 'Eval 1\n(Weak Baseline)',
            'path': 'results/eval1_baseline_ppo_20251012_120926',
            'score': 5.08,
            'data': None
        },
        {
            'name': 'Improvement 1\n(Hyperparams)',
            'path': 'results/improvement1_hyperparams_ppo_20251012_094504',
            'score': 7.10,
            'data': None
        },
        {
            'name': 'Improvement 2\n(ICM)',
            'path': 'results/improvement2_icm_logged_20251012_ppo_20251012_194436',
            'score': 8.27,
            'data': None
        },
        {
            'name': 'Improvement 3\n(Large Net + Low β)',
            'path': 'results/improvement3_combo_20251022_ppo_20251022_225842',
            'score': 8.61,
            'data': None
        }
    ]

    failed_evaluations = [
        {'name': 'RND v1', 'score': 7.08},
        {'name': 'RND v2', 'score': 5.78},
        {'name': 'RND v3', 'score': 6.11},
        {'name': 'Conservative LR', 'score': 8.11},
        {'name': 'High β=0.3', 'score': 6.38},
        {'name': 'High Entropy 0.005', 'score': 4.86},
        {'name': 'High Entropy 0.002', 'score': 6.08},
        {'name': 'Dual-Clip', 'score': 5.79},
        {'name': 'Extended 1.5M', 'score': 7.52},
        {'name': 'Large Net Only', 'score': 8.36},
    ]

    # Try to load detailed data from evaluation reports
    print("\nLoading evaluation data...")
    for eval_config in successful_evaluations:
        try:
            data = load_evaluation_results(eval_config['path'])
            eval_config['data'] = data
            print(f"✓ Loaded: {eval_config['name'].strip()}")
        except Exception as e:
            print(f"⚠ Could not load {eval_config['name'].strip()}: {e}")
            # Use placeholder data if file not found
            eval_config['data'] = {
                'results': {
                    'score': eval_config['score'],
                    'reward': 0,
                    'length': 0,
                    'achievements': {}
                }
            }

    # Generate all figures
    print("\n" + "=" * 60)
    print("Generating Figures...")
    print("=" * 60)

    print("\n[1/5] Score Progression...")
    plot_1_crafter_score_progression(successful_evaluations, failed_evaluations, output_dir)

    print("\n[2/5] Achievement Comparison...")
    plot_2_achievement_comparison(successful_evaluations, output_dir)

    print("\n[3/5] Summary Metrics...")
    plot_3_summary_metrics(successful_evaluations, output_dir)

    print("\n[4/5] Failed Attempts Analysis...")
    plot_4_failed_attempts_analysis(failed_evaluations, output_dir)

    print("\n[5/5] Improvement Breakdown...")
    plot_5_improvement_breakdown(successful_evaluations, output_dir)

    print("\n" + "=" * 60)
    print(f"✅ All figures saved to: {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - fig1_score_progression.png/pdf")
    print("  - fig2_achievement_comparison.png/pdf")
    print("  - fig3_summary_metrics.png/pdf")
    print("  - fig4_failed_attempts.png/pdf")
    print("  - fig5_improvement_breakdown.png/pdf")
    print("\nReady for IEEE format report! 📊")


if __name__ == '__main__':
    main()

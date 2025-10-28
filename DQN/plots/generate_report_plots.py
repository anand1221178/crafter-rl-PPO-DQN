"""
Generate all plots for DQN report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Output directory
OUTPUT_DIR = Path("report_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# RAINBOW COLOR PALETTE
RAINBOW_COLORS = {
    'red': '#E74C3C',        # Muted red
    'orange': '#E67E22',     # Muted orange
    'green': '#27AE60',      # Muted green
    'blue': '#3498DB',       # Muted blue
    'indigo': '#5B2C6F',     # Muted indigo
    'violet': '#8E44AD',     # Muted violet
}

RAINBOW_SEQUENCE = ['#E74C3C', '#E67E22', '#27AE60', '#3498DB', '#5B2C6F', '#8E44AD']

SUCCESS_COLOR = '#27AE60'   
FAIL_COLOR = '#E74C3C'     

# ============================================================================
# DATA DEFINITIONS
# ============================================================================

# Successful generations (in chronological order)
SUCCESSFUL_GENS = [
    {
        "name": "Evaluation 1 (Baseline)",
        "short_name": "Eval 1\n(Baseline)",
        "legend_name": "Evaluation 1 (Baseline)",
        "score": 2.80,  
        "reward": 4.0,  
        "length": 170, 
        "eval_num": 1,
        "achievements": {
            "collect_wood": 85.0,
            "place_table": 55.0,
            "make_wood_pickaxe": 8.0,
            "collect_stone": 0.5,
            "collect_coal": 0.0,
            "defeat_zombie": 35.0,
            "wake_up": 88.0,
            "collect_drink": 25.0,
        }
    },
    {
        "name": "Improvement 1 (N-Step Returns)",
        "short_name": "Improvement 1\n(N-Step)",
        "legend_name": "Improvement 1 (N-Step)",
        "score": 3.53,  
        "reward": 4.5,  
        "length": 180,  
        "eval_num": 2,
        "achievements": {
            "collect_wood": 88.0,
            "place_table": 62.0,
            "make_wood_pickaxe": 12.0,
            "collect_stone": 0.8,
            "collect_coal": 0.0,
            "defeat_zombie": 48.0,
            "wake_up": 90.0,
            "collect_drink": 32.0,
        }
    },
    {
        "name": "Improvement 2 (Action Masking)",
        "short_name": "Improvement 2\n(Action Masking)",
        "legend_name": "Improvement 2 (Action Masking)",
        "score": 4.00,
        "reward": 5.0, 
        "length": 185,  
        "eval_num": 3,
        "achievements": {
            "collect_wood": 89.0,
            "place_table": 68.0,
            "make_wood_pickaxe": 14.0,
            "collect_stone": 1.0,
            "collect_coal": 0.0,
            "defeat_zombie": 52.0,
            "wake_up": 92.0,
            "collect_drink": 38.0,
        }
    },
    {
        "name": "Improvement 3 (Random-Valid Fallback)",
        "short_name": "Improvement 3\n(Random-Valid)",
        "legend_name": "Improvement 3 (Random-Valid)",
        "score": 4.33,
        "reward": 5.12,
        "length": 191,
        "eval_num": 4,
        "achievements": {
            "collect_wood": 90.6,
            "place_table": 71.8,
            "make_wood_pickaxe": 14.2,
            "collect_stone": 1.0,
            "collect_coal": 0.0,
            "defeat_zombie": 54.0,
            "wake_up": 97.4,
            "collect_drink": 54.2,
        }
    },
    {
        "name": "Improvement 4 (ICM Curiosity)",
        "short_name": "Improvement 4\n(ICM)",
        "legend_name": "Improvement 4 (ICM)",
        "score": 4.38,
        "reward": 5.15,
        "length": 192,
        "eval_num": 5,
        "achievements": {
            "collect_wood": 92.4,
            "place_table": 66.4,
            "make_wood_pickaxe": 14.6,
            "collect_stone": 0.6,
            "collect_coal": 0.2,
            "defeat_zombie": 56.6,
            "wake_up": 90.4,
            "collect_drink": 30.4,
        }
    },
    {
        "name": "Improvement 5 (NoisyNets)",
        "short_name": "Improvement 5\n(NoisyNets)",
        "legend_name": "Improvement 5 (NoisyNets)",
        "score": 5.93,
        "reward": 5.47,
        "length": 196.44,
        "eval_num": 6,
        "achievements": {
            "collect_wood": 90.4,
            "place_table": 74.0,
            "make_wood_pickaxe": 21.2,
            "collect_stone": 2.4,
            "collect_coal": 0.0,
            "defeat_zombie": 55.6,
            "wake_up": 97.8,
            "collect_drink": 42.2,
        }
    },
]

# Failed attempts
FAILED_ATTEMPTS = [
    {"name": "Data Augmentation", "short": "DataAug", "score": 2.65, "tried_after_eval": 2, "tried_after_name": "Improvement 1"},
    {"name": "Dueling DQN", "short": "Dueling", "score": 2.90, "tried_after_eval": 2, "tried_after_name": "Improvement 1"},
    {"name": "PER", "short": "PER", "score": 3.20, "tried_after_eval": 3, "tried_after_name": "Improvement 2"},
    {"name": "Polyak Updates", "short": "Polyak", "score": 3.35, "tried_after_eval": 3, "tried_after_name": "Improvement 2"},
    {"name": "Reward Shaping", "short": "Reward-Shaping", "score": 3.65, "tried_after_eval": 4, "tried_after_name": "Improvement 3"},
    {"name": "Frame Stacking", "short": "FrameStack", "score": 3.54, "tried_after_eval": 4, "tried_after_name": "Improvement 3"},
    {"name": "Stone Shaping", "short": "Stone-Shaping", "score": 3.82, "tried_after_eval": 4, "tried_after_name": "Improvement 3"},
]

# Key achievements
KEY_ACHIEVEMENTS = [
    "collect_wood",
    "place_table", 
    "make_wood_pickaxe",
    "collect_stone",
    "collect_coal",
    "defeat_zombie",
    "wake_up",
    "collect_drink",
]

ACHIEVEMENT_LABELS = {
    "collect_wood": "Wood",
    "place_table": "Table",
    "make_wood_pickaxe": "Wood Pickaxe",
    "collect_stone": "Stone",
    "collect_coal": "Coal",
    "defeat_zombie": "Zombie",
    "wake_up": "Wake Up",
    "collect_drink": "Drink",
}


# ============================================================================
# FIG. D1 — PERFORMANCE PROGRESSION (SUCCESS VS FAILURES)
# ============================================================================

def plot_performance_progression():
    """Green dots for successes, red X's for failures."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    eval_nums = [gen["eval_num"] for gen in SUCCESSFUL_GENS]
    scores = [gen["score"] for gen in SUCCESSFUL_GENS]
    
    ax.plot(eval_nums, scores, 'o-', color=SUCCESS_COLOR, linewidth=3, 
            markersize=14, label='Successful Evaluations', zorder=3, markeredgewidth=2, markeredgecolor='white')
    
    for i in range(len(SUCCESSFUL_GENS) - 1):
        curr_gen = SUCCESSFUL_GENS[i]
        next_gen = SUCCESSFUL_GENS[i + 1]
        
        improvement = ((next_gen["score"] - curr_gen["score"]) / curr_gen["score"]) * 100
        
        mid_x = (curr_gen["eval_num"] + next_gen["eval_num"]) / 2
        mid_y = (curr_gen["score"] + next_gen["score"]) / 2 + 0.2
        
        ax.annotate(f'+{improvement:.1f}%', 
                   xy=(mid_x, mid_y), 
                   fontsize=11, 
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=RAINBOW_COLORS['green'], alpha=0.9, edgecolor='white', linewidth=2))
    
    for gen in SUCCESSFUL_GENS:
        ax.annotate(f'{gen["score"]:.2f}%', 
                   xy=(gen["eval_num"], gen["score"]),
                   xytext=(0, 12),
                   textcoords='offset points',
                   ha='center',
                   fontsize=10,
                   fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=SUCCESS_COLOR, alpha=0.9))
    
    failed_eval_map = {
        2: [2.3, 2.6],  # After Improvement 1
        3: [3.3, 3.6],  # After Improvement 2
        4: [4.3, 4.5, 4.7],  # After Improvement 3
    }
    
    for failed in FAILED_ATTEMPTS:
        after_eval = failed["tried_after_eval"]
        positions = failed_eval_map.get(after_eval, [3.5])
        x_pos = positions[0] if positions else 3.5
        failed_eval_map[after_eval] = positions[1:] if len(positions) > 1 else [after_eval + 0.4]
        
        ax.scatter(x_pos, failed["score"], marker='x', s=400, 
                  color=FAIL_COLOR, linewidth=4, label='_nolegend_', zorder=2)
        
        ax.annotate(failed["short"], 
                   xy=(x_pos, failed["score"]),
                   xytext=(10, -5),
                   textcoords='offset points',
                   fontsize=9,
                   color=FAIL_COLOR,
                   fontweight='bold')
    
    ax.scatter([], [], marker='x', s=400, color=FAIL_COLOR, 
              linewidth=4, label='Failed Attempts')
    
    ax.set_xlabel('Evaluation Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Crafter Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('DQN Agent Performance Progression on Crafter\n(Successful Evaluations and Failed Attempts)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(2.0, 9.0)
    
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_d1_performance_progression.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Fig. D1 saved: Performance progression")


# ============================================================================
# FIG. D2 — ACHIEVEMENT UNLOCK RATES (SIDE-BY-SIDE BARS)
# ============================================================================

def plot_achievement_bars():
    """Bar groups for key achievements across checkpoints."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    n_achievements = len(KEY_ACHIEVEMENTS)
    n_gens = len(SUCCESSFUL_GENS)
    
    x = np.arange(n_achievements)
    width = 0.14
    
    colors = RAINBOW_SEQUENCE  # Red, Orange, Green, Blue, Indigo, Violet
    
    for i, gen in enumerate(SUCCESSFUL_GENS):
        values = [gen["achievements"][ach] for ach in KEY_ACHIEVEMENTS]
        offset = (i - n_gens/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=gen["legend_name"], 
               color=colors[i], alpha=0.9, edgecolor='white', linewidth=1.5)
    
    ax.set_xlabel('Achievement Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Achievement Unlock Rates Across Evaluations', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([ACHIEVEMENT_LABELS[ach] for ach in KEY_ACHIEVEMENTS], 
                       rotation=15, ha='right', fontsize=11)
    ax.legend(loc='upper left', ncol=2, fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 105)
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_d2_achievement_rates.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Fig. D2 saved: Achievement unlock rates")


# ============================================================================
# FIG. D3 — SUMMARY PERFORMANCE METRICS (3 MINI BAR GRAPHS)
# ============================================================================

def plot_summary_metrics():
    """Three side-by-side bar charts: Score, Reward, Length."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    gen_labels = [gen["legend_name"] for gen in SUCCESSFUL_GENS]
    scores = [gen["score"] for gen in SUCCESSFUL_GENS]
    rewards = [gen["reward"] for gen in SUCCESSFUL_GENS]
    lengths = [gen["length"] for gen in SUCCESSFUL_GENS]
    
    colors = RAINBOW_SEQUENCE  # Red, Orange, Green, Blue, Indigo, Violet
    
    # Plot 1: Crafter Score
    bars1 = axes[0].bar(range(len(gen_labels)), scores, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Crafter Score (%)', fontweight='bold', fontsize=12)
    axes[0].set_title('Crafter Score', fontweight='bold', fontsize=14)
    axes[0].set_xticks(range(len(gen_labels)))
    axes[0].set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].set_facecolor('#FAFAFA')
    
    for i, (bar, val) in enumerate(zip(bars1, scores)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Episode Reward
    bars2 = axes[1].bar(range(len(gen_labels)), rewards, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    axes[1].set_ylabel('Average Reward', fontweight='bold', fontsize=12)
    axes[1].set_title('Episode Reward', fontweight='bold', fontsize=14)
    axes[1].set_xticks(range(len(gen_labels)))
    axes[1].set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].set_facecolor('#FAFAFA')
    
    for i, (bar, val) in enumerate(zip(bars2, rewards)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Survival Time
    bars3 = axes[2].bar(range(len(gen_labels)), lengths, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    axes[2].set_ylabel('Average Episode Length', fontweight='bold', fontsize=12)
    axes[2].set_title('Survival Time', fontweight='bold', fontsize=14)
    axes[2].set_xticks(range(len(gen_labels)))
    axes[2].set_xticklabels(gen_labels, rotation=45, ha='right', fontsize=9)
    axes[2].set_facecolor('#FAFAFA')
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars3, lengths)):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.suptitle('Summary Performance Metrics', fontsize=17, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_d3_summary_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Fig. D3 saved: Summary metrics")


# ============================================================================
# FIG. D4 — FAILED IMPROVEMENT ATTEMPTS
# ============================================================================

def plot_failed_attempts():
    """Horizontal bars for failed methods with gradient based on badness."""
    fig, ax = plt.subplots(figsize=(13, 8))
    
    baseline = 4.33
    
    sorted_failures = sorted(FAILED_ATTEMPTS, key=lambda x: x["score"])
    
    names_with_context = [f"{f['name']}\n(After {f['tried_after_name']})" for f in sorted_failures]
    scores = [f["score"] for f in sorted_failures]
    
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    
    colors_gradient = ['#C0392B', '#E74C3C', '#E67E22', '#F39C12', '#F1C40F', '#F4D03F', '#F7DC6F']
    n_colors = len(sorted_failures)
    color_indices = np.linspace(0, len(colors_gradient)-1, n_colors)
    colors = [colors_gradient[int(i)] for i in color_indices]
    
    y_pos = np.arange(len(names_with_context))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    
    ax.axvline(baseline, color=SUCCESS_COLOR, linestyle='--', linewidth=3, 
               label=f'Baseline to Beat ({baseline}%)', zorder=3)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.20, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_with_context, fontsize=11)
    ax.set_xlabel('Crafter Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Failed Improvement Attempts\n(Strategies That Did Not Beat Baseline)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_xlim(0, 9.5)
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_d4_failed_attempts.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Fig. D4 saved: Failed attempts")


# ============================================================================
# FIG. D5 — CUMULATIVE IMPROVEMENT (WATERFALL/STACKED)
# ============================================================================

def plot_cumulative_improvement():
    """Stacked bar showing contribution of each generation."""
    fig, ax = plt.subplots(figsize=(13, 9))
    
    baseline_score = SUCCESSFUL_GENS[0]["score"]
    total_score = SUCCESSFUL_GENS[-1]["score"]
    
    deltas = [baseline_score]
    labels = [SUCCESSFUL_GENS[0]["legend_name"]]
    
    for i in range(1, len(SUCCESSFUL_GENS)):
        delta = SUCCESSFUL_GENS[i]["score"] - SUCCESSFUL_GENS[i-1]["score"]
        deltas.append(delta)
        labels.append(SUCCESSFUL_GENS[i]["legend_name"])
    
    colors = RAINBOW_SEQUENCE  # Red, Orange, Green, Blue, Indigo, Violet
    
    bottom = 0
    for i, (delta, label, color) in enumerate(zip(deltas, labels, colors)):
        ax.bar(0, delta, bottom=bottom, color=color, alpha=0.9, 
               label=label, width=0.7, edgecolor='white', linewidth=2)
        
        text_y = bottom + delta/2
        
        if i == 0:
            ax.text(0, text_y, f'{delta:.2f}%', ha='center', va='center',
                   fontsize=15, fontweight='bold', color='white')
        else:
            ax.text(0, text_y, f'+{delta:.2f}%', ha='center', va='center',
                   fontsize=13, fontweight='bold', color='white')
            
            ax.text(0.45, text_y, label, ha='left', va='center',
                   fontsize=10, fontweight='bold', color=color)
        
        bottom += delta
    
    ax.text(0, total_score + 0.4, f'Total: {total_score:.2f}%', 
           ha='center', va='bottom', fontsize=17, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor=RAINBOW_COLORS['orange'], 
                    alpha=0.9, edgecolor='white', linewidth=2))
    
    total_improvement = ((total_score - baseline_score) / baseline_score) * 100
    ax.text(0, -0.6, f'Total Improvement: +{total_improvement:.1f}%',
           ha='center', va='top', fontsize=14, fontweight='bold',
           color=SUCCESS_COLOR)
    
    ax.set_xlim(-0.6, 1.2)
    ax.set_ylim(-1, total_score + 1.2)
    ax.set_ylabel('Crafter Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cumulative Improvement Breakdown\n(Contribution of Each Evaluation)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_d5_cumulative_improvement.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Fig. D5 saved: Cumulative improvement")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating DQN Report Plots")
    print("="*60 + "\n")
    
    # Generate all core plots
    plot_performance_progression()    # Fig. D1
    plot_achievement_bars()           # Fig. D2
    plot_summary_metrics()            # Fig. D3
    plot_failed_attempts()            # Fig. D4
    plot_cumulative_improvement()     # Fig. D5
    
    print("\n" + "="*60)
    print(f"✅ All plots saved to: {OUTPUT_DIR}/")
    print("="*60 + "\n")
    
    print("\nGenerated plots:")
    print("  1. fig_d1_performance_progression.png")
    print("  2. fig_d2_achievement_rates.png")
    print("  3. fig_d3_summary_metrics.png")
    print("  4. fig_d4_failed_attempts.png")
    print("  5. fig_d5_cumulative_improvement.png")
    print("\nNext steps:")
    print("  - Review plots for accuracy")
    print("  - Run optional plots (D6-D9) if time allows")


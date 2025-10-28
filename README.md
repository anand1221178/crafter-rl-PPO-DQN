# Crafter RL Project 🎮

Teaching AI agents to survive in a 2D Minecraft-like world. Spoiler: it's harder than it looks.

## What We Did

Built two different reinforcement learning agents from scratch to play **Crafter** - a survival game where you need to gather wood, craft tools, fight zombies, and generally not die. The catch? You only get rewarded for achievements like "craft a pickaxe" or "defeat a zombie," which happens maybe 1% of the time. The other 99% of the time, the agent has no idea if it's doing well or just wandering around aimlessly.

**Final Results:**
- **PPO (Proximal Policy Optimization)**: 8.61% Crafter Score → Won 🏆
- **DQN (Deep Q-Network)**: 5.93% Crafter Score → Solid effort

(For context, random agent gets ~0.5%. Human experts get ~50%. So we're... better than random!)

## The Journey

### PPO Implementation (Anand)
Started at 5.08%, ended at 8.61% through:
1. **Better hyperparameters** - Turns out the internet knows things (+39.8% improvement)
2. **Curiosity-driven exploration (ICM)** - Gave the agent "bonus points" for discovering new things (+16.5%)
3. **Bigger brain** - Increased network size so it could remember more complex strategies (+4.1%)

Also tried 7 things that completely failed (RND curiosity, too much entropy, training for too long, etc.) but documenting failures is science, right?

### DQN Implementation (Mikyle)
Went from 2.80% to 5.93% through:
1. **n-step returns** - Look ahead more than one step (+26.1%)
2. **Action masking** - Stop trying impossible actions like "craft sword" when you have no wood (+13.3%)
3. **Random-valid fallback** - If you can't do that action, do something useful instead of nothing (+8.3%)
4. **ICM-lite curiosity** - Conservative exploration bonus, train-time only (+1.2%)
5. **NoisyNets** - Parameter noise for sustained exploration (+35.4%)

DQN was clever with inventory-aware masking but couldn't explore as well as PPO's full curiosity system.

## Key Takeaway

**Exploration beats efficiency** for sparse-reward environments like Crafter. PPO's curiosity-driven approach found rare achievements (coal, skeletons) that DQN missed, even though DQN was more sample-efficient on paper. Sometimes you just gotta explore weird stuff to discover the good stuff.

## Project Structure

```
crafter-rl-project/
├── src/
│   ├── agents/          # PPO agent implementation
│   ├── modules/         # ICM curiosity module
│   └── utils/           # Networks, replay buffers, GAE
├── train_ppo_icm.py     # Main PPO training script
├── evaluate.py          # Evaluation (100 episodes)
├── report/              # IEEE paper (8 pages)
│   └── main.tex
├── results/             # Evaluation results & figures
├── logs/                # Training logs (19GB, gitignored)
└── sweep_results/       # Hyperparameter sweep experiments
```

## Quick Start

### Setup
```bash
conda activate crafter  # or create env from requirements
```

### Train PPO
```bash
# Best config (8.61% result)
python train_ppo_icm.py \
    --steps 1000000 \
    --lr 5e-4 \
    --entropy_coef 0.001 \
    --icm_beta 0.15 \
    --hidden_dim 1024 \
    --outdir logs/my_run
```

### Evaluate
```bash
python evaluate.py \
    --model_path logs/my_run/*/ppo_icm_final.pt \
    --algorithm ppo \
    --episodes 100 \
    --outdir results/my_eval
```

### Hyperparameter Sweep (Cluster)
```bash
# Generate 20 random configs
python hyperparam_sweep.py --mode random --n_samples 20 --cluster slurm

# Submit to cluster
sbatch sweep_results/submit_sweep.sh

# Analyze results
python analyze_sweep.py sweep_results/ --top_n 10
```

## Files You Actually Care About

- **`CLAUDE.md`** - Full experiment log with all results
- **`report/main.tex`** - IEEE paper with everything documented
- **`train_ppo_icm.py`** - The thing that actually trains the agent
- **`src/modules/icm.py`** - Curiosity module (the secret sauce)
- **`plot_all_experiments.py`** - Generate all the pretty figures

## What We Learned

1. **Curiosity works** - ICM intrinsic rewards help discover rare achievements
2. **Balance matters** - Too much curiosity (β=0.3) = curiosity trap. Too little = miss rare states
3. **Literature > guessing** - Standard hyperparameters beat our initial guesses by a lot
4. **More ≠ better** - Training for 1.5M steps was worse than 1M steps
5. **Training metrics lie** - Best eval model (8.61%) had lower training reward than worse models

## Things That Failed (So You Don't Have To Try)

- Random Network Distillation (RND) - Created curiosity trap, agent just explored forever
- High entropy (0.005) - Policy stayed too random, never converged
- Extended training (1.5M steps) - PPO degraded after 1M
- Dual-clip PPO - Made training unstable with curiosity
- Conservative learning rate (1e-4) - Too slow for 1M step budget

## Hardware

Trained on M4 MacBook Pro with MPS (Apple Silicon GPU):
- Training time: ~2.5 hours per 1M steps
- Speed: ~135 FPS
- Total experiments: 11 runs, ~30 hours total

## Team

- **Anand Patel** - PPO + ICM implementation
- **Mikyle Singh** - DQN + action masking implementation

## Papers We Referenced

1. **PPO**: Schulman et al. 2017
2. **ICM**: Pathak et al. 2017 (Curiosity-driven Exploration)
3. **Crafter**: Hafner 2021
4. **NoisyNets**: Fortunato et al. 2018
5. **Rainbow DQN**: Hessel et al. 2018




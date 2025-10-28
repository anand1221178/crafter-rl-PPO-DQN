# Crafter DQN - Complete Documentation# Crafter DQN Project# Crafter RL Project



Deep Q-Network (DQN) implementation for the Crafter survival game environment, featuring iterative improvements through 6 generations of development achieving **5.93% Crafter Score**.



## Project OverviewA Deep Q-Network (DQN) implementation for the Crafter survival game environment, featuring iterative improvements through 5 generations of development. This project achieved a **5.93% Crafter Score** through advanced exploration techniques.> 



This project systematically improves DQN performance on the Crafter benchmark through iterative experimentation. Starting from a baseline DQN, we tested multiple improvement techniques and achieved significant gains through advanced exploration strategies.> **Contains:** Complete generation history, training commands, results, and evaluation protocol



### Generation Results## Project Overview



| Generation | Score | Key Feature | Improvement |## Project Overview

|------------|-------|-------------|-------------|

| Gen-0 | 2.80% | Baseline DQN | Baseline |This project implements and iteratively improves DQN for the Crafter benchmark environment, progressing from a baseline implementation to advanced exploration techniques including NoisyNets. The final model (Gen-5) achieved significant performance gains through parameter-space exploration.Implementation of reinforcement learning agents for the Crafter survival game environment. This project implements two RL algorithms:

| Gen-1 | ~3.5% | N-step returns (n=3) | - |

| Gen-2 | ~4.0% | Reward shaping | - |1. **Course Algorithm**: DQN (Deep Q-Network) - Model-free value-based RL

| Gen-3c | 4.33% | Action masking + random fallback | +54.6% vs Gen-0 |

| Gen-4 | 4.38% | ICM curiosity | +1.2% vs Gen-3c || Generation | Score | Key Feature | Improvement |Each algorithm will undergo iterative improvements to optimize performance in the challenging Crafter survival environment.

| **Gen-5** | **5.93%** | **NoisyNets exploration** | **+35.4% vs Gen-4** ✅ |

|------------|-------|-------------|-------------|

**Total Improvement**: 111.8% improvement from baseline (2.80% → 5.93%)

| Gen-0 | 2.80% | Baseline DQN | - |## Assignment Details

## Environment Details

- **Environment**: Crafter (CrafterPartial-v1)

- **Observation**: 64×64×3 RGB images| Gen-2 | ~3.X% | Reward shaping | - |- **Team Size**: 2 members

- **Action Space**: 17 discrete actions

- **Achievements**: 22 hierarchical achievements| Gen-3c | 4.33% | Action masking (random-valid) | - |- **Environment**: Crafter with partial observability (CrafterPartial-v1)

- **Episode Length**: Maximum 10,000 steps

- **Reward**: Sparse (+1 per achievement unlock)| Gen-4 | 4.38% | ICM curiosity | +1.2% |- **Observation Space**: 64x64 RGB images

- **Primary Metric**: Crafter Score (geometric mean of achievement unlock rates)

| **Gen-5** | **5.93%** | **NoisyNets** | **+35.4%** ✅ |- **Action Space**: 17 discrete actions

## Project Structure



```

crafter-rl/**Total Improvement**: Gen-5 achieved **111.8% improvement** over baseline (2.80% → 5.93%)## Project Structure

├── train.py                        # Main DQN training script

├── evaluate.py                     # Evaluation script

├── visualize_agent.py              # Agent visualization

├── crafter_env.yaml                # Conda environment specification## Project Structure```

├── README.md                       # Main project README

├── README_DQN.md                   # This file (DQN documentation)crafter-rl-project/

│

├── logdir/                         # All generation results```├── crafter_env.yaml            # Conda environment specification

│   ├── Gen-0_Baseline/             # 2.80% - Baseline DQN

│   ├── Gen-1_NStep/                # N-step returnscrafter-rl/├── README_DQN.md               # DQN comprehensive documentation

│   ├── Gen-2_ActionMasking/        # Reward shaping

│   ├── Gen-3c_RandomValid/         # 4.33% - Action masking├── train.py                        # Main training script├── train.py                    # Unified training script (PPO, DQN)

│   ├── Gen-4_ICMlite/              # 4.38% - ICM curiosity

│   ├── Gen-5_NoisyNets/            # 5.93% - NoisyNets ✅ BEST├── evaluate.py                     # Evaluation script├── evaluate.py                 # Comprehensive evaluation script

│   │   ├── dqn_final.zip           # Trained model

│   │   ├── GEN-5_SUMMARY.md        # Detailed results├── visualize_agent.py              # Agent visualization├── visualize_agent.py          # Agent visualization

│   │   └── stats.jsonl             # Episode statistics

│   │├── train_ppo.py                    # PPO training (baseline comparison)├── train_ppo.sbatch            # Cluster training script (PPO)

│   └── Failed Experiments/

│       ├── Gen-2_DataAug_FAILED/├── crafter_env.yaml                # Conda environment specification├── train_dqn.sbatch            # Cluster training script (DQN)

│       ├── Gen-2_DuelingDQN_FAILED/

│       ├── Gen-3_PER_FAILED/├── README.md                       # This file├── wrappers/                   # DQN components and wrappers

│       ├── Gen-3_Polyak_FAILED/

│       ├── Gen-3b_FrameStack_FAILED/││   ├── action_masking_wrapper.py

│       ├── Gen-3_StoneShaping_FAILED/

│       └── Gen-4_RewardShaping_FAILED/├── logdir/                         # Training results and saved models│   ├── icm_lite_wrapper.py

│

├── wrappers/                       # DQN components│   ├── Gen-0_Baseline/             # 2.80% - Initial DQN│   ├── noisy_dqn.py

│   ├── action_masking_wrapper.py   # Invalid action masking

│   ├── icm_lite_wrapper.py         # Intrinsic curiosity│   ├── Gen-1_NStep/                # N-step returns (n=3)│   └── ...

│   ├── noisy_linear.py             # NoisyNet layer

│   ├── noisy_qnetwork.py           # Noisy Q-network│   ├── Gen-2_ActionMasking/        # Reward shaping experiments├── plots/                      # Plot generation scripts

│   ├── noisy_dqn_policy.py         # NoisyNet policy

│   ├── noisy_dqn.py                # NoisyNet DQN algorithm│   ├── Gen-3c_RandomValid/         # 4.33% - Action masking with random fallback│   ├── generate_report_plots.py

│   ├── shaping_wrapper.py          # Reward shaping

│   ├── dqn_per.py                  # PER implementation (failed)│   ├── Gen-4_ICMlite/              # 4.38% - Intrinsic curiosity│   └── generate_optional_plots.py

│   ├── dqn_polyak.py               # Polyak averaging (failed)

│   ├── dueling_dqn.py              # Dueling DQN (failed)│   ├── Gen-5_NoisyNets/            # 5.93% - NoisyNets (BEST) ✅├── src/

│   └── framestack_wrapper.py       # Frame stacking (failed)

││   │   ├── dqn_final.zip           # Trained model│   ├── agents/

├── plots/                          # Visualization

│   ├── generate_report_plots.py│   │   ├── GEN-5_SUMMARY.md        # Detailed results│   │   ├── base_agent.py       # Abstract base class for all agents

│   └── generate_optional_plots.py

││   │   └── ...│   │   └── dynaq_agent.py      # Dyna-Q implementation (to be implemented)

└── train_dqn.sbatch                # Cluster training script

```│   ││   ├── models/



## Setup Instructions│   └── Failed_Experiments/         # Documented failures│   │   ├── world_model.py      # Environment dynamics model (to be implemented)



### Environment Setup│       ├── Gen-2_DataAug_FAILED/│   │   └── prioritized_sweeping.py  # Priority queue for planning (to be implemented)



```bash│       ├── Gen-2_DuelingDQN_FAILED/│   ├── utils/

# Clone repository

git clone <your-repo-url>│       ├── Gen-3_PER_FAILED/│   │   ├── networks.py         # Q-network (CNN for 64x64 RGB)

cd crafter-rl

│       ├── Gen-3_Polyak_FAILED/│   │   └── replay_buffer.py    # Experience replay

# Create conda environment

conda env create -f crafter_env.yaml│       ├── Gen-3b_FrameStack_FAILED/│   └── evaluation/

conda activate crafter_env

│       ├── Gen-3_StoneShaping_FAILED/│       ├── plot_reward.py      # Reward plotting utilities

# Verify installation

python -c "import torch; print(f'PyTorch: {torch.__version__}')"│       └── Gen-4_RewardShaping_FAILED/│       ├── plot_scores.py      # Achievement score plotting

python -c "import crafter; print('Crafter installed')"

python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"││       └── read_metrics.py     # Metrics reading from Crafter logs

```

├── wrappers/                       # DQN components and environment wrappers├── models/                     # Saved model checkpoints

## Training Commands

│   ├── action_masking_wrapper.py   # Invalid action handling├── results/                    # Training results and evaluations

### Gen-0: Baseline DQN (2.80%)

│   ├── icm_lite_wrapper.py         # Intrinsic curiosity module├── logdir/                     # Training logs (Crafter metrics)

```bash

python train.py --algorithm dqn --steps 1000000 --seed 42│   ├── noisy_linear.py             # NoisyNet layer implementation└── README.md                   # This file

```

│   ├── noisy_qnetwork.py           # Q-network with noisy layers```

**Configuration**:

- Standard DQN with epsilon-greedy exploration│   ├── noisy_dqn_policy.py         # NoisyNet DQN policy

- Experience replay buffer (100k transitions)

- Target network updates every 10,000 steps│   ├── noisy_dqn.py                # NoisyNet DQN algorithm## Key Metrics to Track

- Learning rate: 1e-4

│   ├── shaping_wrapper.py          # Reward shaping utilities1. **Achievement Unlock Rate**: Percentage of times each of 22 achievements is unlocked

### Gen-1: N-Step Returns

│   └── ...2. **Geometric Mean (Crafter Score)**: Overall score combining all achievements

```bash

python train.py --algorithm dqn --steps 1000000 --seed 42 --n-step 3│3. **Survival Time**: Average timesteps survived per episode

```

├── plots/                          # Visualization scripts4. **Cumulative Reward**: Total reward per episode

**Improvement**: N-step returns (n=3) for better credit assignment

│   ├── generate_report_plots.py    # Main plotting script

### Gen-2: Reward Shaping

│   └── generate_optional_plots.py  # Additional visualizations## Implementation Pipeline

```bash

python train.py --algorithm dqn --steps 1000000 --seed 42 \│

  --n-step 3 \

  --reward-shaping└── train_*.sbatch                  # Cluster training scripts### For Each Algorithm:

```

```1. **Base Implementation** → Evaluate (Eval 1)

**Improvement**: Milestone-based reward shaping to guide early learning

2. **Improvement 1** → Evaluate (Eval 2)

### Gen-3c: Action Masking (4.33%)

## Setup Instructions3. **Improvement 2** → Evaluate (Eval 3)

```bash

python train.py --algorithm dqn --steps 1000000 --seed 42 \4. **Final Comparison** between both algorithms

  --n-step 3 \

  --reward-shaping \### Quick Setup (Recommended)

  --action-masking \

  --mask-fallback random## Algorithms

```

```bash

**Breakthrough**: Action masking with random-valid fallback

- Masks invalid actions (e.g., place table without wood)# Clone repository### Course Algorithm: DQN (Deep Q-Network)

- Falls back to random valid action instead of zeros

- Reduced sticky action loops, maintained explorationgit clone <your-repo-url>Model-free value-based RL using Stable-Baselines3 implementation:

- **+54.6% improvement** over baseline

cd crafter-rl- Q-learning with neural network function approximation

### Gen-4: ICM Lite (4.38%)

- Experience replay for sample efficiency

```bash

python train.py --algorithm dqn --steps 1000000 --seed 42 \# Create conda environment- Target network for stable training

  --action-masking \

  --mask-fallback random \conda env create -f crafter_env.yaml- Epsilon-greedy exploration

  --icm-lite \

  --icm-beta 0.2 \conda activate crafter_env

  --icm-lr 1e-4

```# Verify installation**Integrated Planning and Learning** - Classic model-based RL algorithm:



**Addition**: Intrinsic Curiosity Module (ICM)python -c "import torch; print(f'PyTorch: {torch.__version__}')"- Learns environment dynamics (world model)

- Forward/inverse model for curiosity-driven exploration

- Intrinsic reward weight: 0.2python -c "import crafter; print('Crafter: installed')"- Combines real experience with simulated planning

- Marginal improvement (+1.2%)

python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"- Sample efficient: 1 real experience → N planning updates

### Gen-5: NoisyNets (5.93%) ✅ BEST

```- Three phases of improvement:

```bash

python train.py --algorithm dqn --steps 1000000 --seed 42 \  1. **Eval 1**: Baseline Dyna-Q (random planning)

  --action-masking \

  --mask-fallback random \### Manual Setup (Alternative)  2. **Eval 2**: + Prioritized Sweeping (focused planning)

  --noisyNets \

  --sigma-init 0.5  3. **Eval 3**: + Dyna-Q+ (exploration bonuses)

```

```bash

**Major Breakthrough**: NoisyNets for parameter-space exploration

- Factorized Gaussian noise in network parameters# Create environment**Why Model-Based RL for Crafter?**

- Replaces epsilon-greedy as primary exploration

- Noise reset after each gradient stepconda create -n crafter_env python=3.10 -y- Sparse rewards benefit from planning (simulate rare experiences)

- **+35.4% improvement** over Gen-4

- **+111.8% improvement** over baselineconda activate crafter_env- Multi-step reasoning (chop tree → wood → stick → sword)



**Architecture**:- Sample efficient learning from limited interactions

```

CNN → MLP(256) → NoisyLinear(256,256) → ReLU → NoisyLinear(256,17)# Install PyTorch (adjust for your CUDA version)

Noise: μ + σ ⊙ ε, where f(ε) = sign(ε) * sqrt(|ε|)

σ_init: 0.5conda install pytorch=2.0.0 -c pytorch -y## Setup Instructions

```



**Key Results**:

- Training time: 3.08 hours (50% faster than Gen-4)# Install dependencies### Quick Setup (Recommended)

- Wood pickaxe: 21.2% (+45% vs Gen-4)

- Stone collection: 2.4% (+300% vs Gen-4)pip install stable-baselines3 crafter gymnasium shimmy wandb imageio imageio-ffmpeg

- Wake up: 97.8% (recovered from Gen-4 regression)

```**1. Create conda environment from YAML:**

## Evaluation Commands

```bash

### Quick Evaluation (10 episodes)

## Training# Clone repository

```bash

python evaluate.py \git clone <your-repo-url>

  --algorithm dqn \

  --model_path "logdir/Gen-5_NoisyNets/dqn_final.zip" \### Training Commandscd crafter-rl-project

  --episodes 10 \

  --action-masking \

  --mask-fallback random

```All successful generations can be reproduced with the following commands:# Create environment



### Full Evaluation (200-500 episodes, recommended)conda env create -f crafter_env.yaml



```bash#### Gen-0: Baseline DQN (2.80%)conda activate crafter_env

python evaluate.py \

  --algorithm dqn \```bash```

  --model_path "logdir/Gen-5_NoisyNets/dqn_final.zip" \

  --episodes 200 \python train.py --algorithm dqn --steps 1000000 --seed 42

  --action-masking \

  --mask-fallback random```**2. Verify installation:**

```

```bash

### Evaluate Other Generations

#### Gen-1: N-Step Returnspython -c "import torch; print(f'PyTorch: {torch.__version__}')"

**Gen-4 (ICM)**:

```bash```bashpython -c "import crafter; print('Crafter: installed')"

python evaluate.py \

  --algorithm dqn \python train.py --algorithm dqn --steps 1000000 --seed 42 --n-step 3python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"

  --model_path "logdir/Gen-4_ICMlite/dqn_final.zip" \

  --episodes 200 \``````

  --action-masking \

  --mask-fallback random \

  --icm-lite

```#### Gen-2: Reward Shaping### Manual Setup (Alternative)



**Gen-3c (Action Masking)**:```bash

```bash

python evaluate.py \python train.py --algorithm dqn --steps 1000000 --seed 42 --n-step 3 --reward-shapingIf you prefer manual installation:

  --algorithm dqn \

  --model_path "logdir/Gen-3c_RandomValid/dqn_final.zip" \```

  --episodes 200 \

  --action-masking \```bash

  --mask-fallback random

```#### Gen-3c: Action Masking with Random Fallback (4.33%)# Create environment



**Gen-0 (Baseline)**:```bashconda create -n crafter_env python=3.10 -y

```bash

python evaluate.py \python train.py --algorithm dqn --steps 1000000 --seed 42 \conda activate crafter_env

  --algorithm dqn \

  --model_path "logdir/Gen-0_Baseline/dqn_final.zip" \  --n-step 3 \

  --episodes 200

```  --reward-shaping \# Install PyTorch (adjust for your CUDA version)



### Evaluation Outputs  --action-masking \conda install pytorch=2.8.0 -c pytorch -y



Evaluation generates:  --mask-fallback random

- **JSON report**: Detailed metrics, achievement rates

- **Text summary**: Human-readable results```# Install dependencies

- **Plots**:

  - `achievement_rates.png` - 22 achievement unlock ratespip install stable-baselines3 crafter gymnasium shimmy wandb imageio imageio-ffmpeg

  - `summary_metrics.png` - Crafter score, reward, length

#### Gen-4: ICM Lite (4.38%)```

**Key Metrics**:

- **Crafter Score**: Geometric mean of achievement unlock rates (primary)```bash

- **Average Reward**: Mean episode reward

- **Average Length**: Mean episode lengthpython train.py --algorithm dqn --steps 1000000 --seed 42 \## 🧪 Local Testing Commands

- **Achievement Rates**: Individual unlock percentages

  --action-masking \

## Training Options Reference

  --mask-fallback random \### Create Environment

### Common Arguments

  --icm-lite \```bash

```bash

--algorithm dqn              # Algorithm (always dqn for this project)  --icm-beta 0.2 \# Create conda environment locally

--steps 1000000             # Training steps (default: 1M)

--seed 42                   # Random seed (default: 42)  --icm-lr 1e-4conda env create -f crafter_env.yaml

--eval_freq 50000           # Evaluation frequency (default: 50k)

``````conda activate crafter_env



### N-Step Returns (Gen-1+)```



```bash#### Gen-5: NoisyNets (5.93%) ✅ BEST

--n-step 3                  # N-step returns (default: 3)

``````bash### Training Algorithms Locally



### Reward Shaping (Gen-2+)python train.py --algorithm dqn --steps 1000000 --seed 42 \```bash



```bash  --action-masking \# Quick PPO test (10K steps, ~5 minutes)

--reward-shaping            # Enable milestone-based reward shaping

```  --mask-fallback random \python train.py --algorithm ppo --steps 10000



### Action Masking (Gen-3c+)  --noisyNets \



```bash  --sigma-init 0.5# Quick DQN test (10K steps, ~5 minutes)

--action-masking            # Enable invalid action masking

--mask-fallback random      # Fallback: 'random' or 'zeros' (default: random)```python train.py --algorithm dqn --steps 10000

```



### ICM Curiosity (Gen-4)

### Training Options# Dyna-Q test (when implemented)

```bash

--icm-lite                  # Enable intrinsic curiosity modulepython train.py --algorithm dynaq --steps 10000 --planning_steps 5

--icm-beta 0.2              # Intrinsic reward weight (default: 0.2)

--icm-lr 1e-4               # ICM learning rate (default: 1e-4)Key arguments for `train.py`:

```

# Full training (1M steps, ~4-8 hours on GPU)

### NoisyNets (Gen-5)

```python train.py --algorithm dqn --steps 1000000

```bash

--noisyNets                 # Enable NoisyNets exploration--algorithm {ppo,dqn}       Algorithm to train (default: dqn)```

--sigma-init 0.5            # Noise std initialization (default: 0.5)

```--steps STEPS               Training steps (default: 1000000)



## Generation Details--seed SEED                 Random seed (default: 42)### Evaluating Models



### Gen-5: NoisyNets (5.93%) - Final Best--eval_freq FREQ            Evaluation frequency (default: 50000)```bash



**Core Innovation**: Parameter-space exploration via NoisyNets# Comprehensive evaluation (100 episodes, detailed analysis)



**Configuration**:# N-Step Returns (Gen-1+)python evaluate.py \

```yaml

Network:--n-step N                  N-step returns (default: 3)    --model_path models/dqn_final.zip \

  type: CNN + NoisyMLP

  features: [32, 64, 64]    --algorithm dqn \

  mlp_size: 256

  noisy_layers: 2 (output layers)# Reward Shaping (Gen-2+)    --episodes 100



Training:--reward-shaping            Enable reward shaping

  steps: 1,000,000

  learning_rate: 1e-4# Quick test (10 episodes, basic metrics)

  batch_size: 32

  gamma: 0.99# Action Masking (Gen-3c+)python test_model.py models/dqn_final.zip dqn 10

  buffer_size: 100,000

  learning_starts: 50,000--action-masking            Enable action masking



Target Updates:--mask-fallback {zeros,random}  Invalid action fallback (default: random)# Evaluate from training logdir (analyze existing stats.jsonl)

  type: hard

  tau: 1.0python evaluate.py \

  interval: 2000

# ICM Curiosity (Gen-4)    --logdir logdir/crafter_dqn_20251005_180000/ \

Exploration:

  type: NoisyNets--icm-lite                  Enable intrinsic curiosity module    --algorithm dqn \

  sigma_init: 0.5

  epsilon: 0.01 (minimal)--icm-beta BETA             ICM intrinsic reward weight (default: 0.2)    --episodes 100



Action Masking:--icm-lr LR                 ICM learning rate (default: 1e-4)```

  enabled: true

  fallback: random-valid

```

# NoisyNets (Gen-5)### Evaluation Outputs

**Achievements** (500 episodes):

- Wake Up: 97.8%--noisyNets                 Enable NoisyNets explorationEvaluation generates:

- Collect Sapling: 91.2%

- Place Plant: 90.2%--sigma-init SIGMA          NoisyNet noise std initialization (default: 0.5)- 📊 Crafter Score (geometric mean of achievements)

- Collect Wood: 90.4%

- Place Table: 74.0%```- 📈 Achievement unlock rates (all 22 achievements)

- Defeat Zombie: 55.6%

- Eat Cow: 43.2%- 🎯 Average reward and episode length

- Collect Drink: 42.2%

- Make Wood Sword: 27.4%### Training Outputs- 📊 Plots (achievement rates, summary metrics)

- Make Wood Pickaxe: 21.2% ✅

- Collect Stone: 2.4% ✅- 📄 JSON + text reports

- Place Stone: 1.4% ✅ (first time)

Training creates a timestamped directory in `logdir/` with:

### Gen-4: ICM Lite (4.38%)

- `dqn_final.zip` - Trained model (Stable-Baselines3 format)## 🚀 Cluster Training Commands

**Core Innovation**: Curiosity-driven exploration

- `stats.jsonl` - Episode statistics (Crafter format)

**Configuration**:

- ICM intrinsic reward weight: 0.2- `DQN_1/` - TensorBoard logs### Submit Training Jobs

- Forward/inverse model for state prediction

- Marginal improvement (+1.2%)- `*.md` - Generation summary and results

- Validated ICM's limited impact in sparse rewards

```bash

### Gen-3c: Action Masking (4.33%)

## Evaluation# Submit DQN training (partner's course algorithm)

**Core Innovation**: Invalid action handling with random fallback

sbatch train_dqn.sbatch

**Configuration**:

- Masks invalid actions (e.g., place without resources)### Evaluation Commands

- Random-valid fallback maintains exploration

- Reduced sticky loops# Submit Dyna-Q training (external algorithm)

- Major improvement (+54.6% vs baseline)

Evaluate any generation using:sbatch train_dynaq.sbatch

### Gen-0-2: Foundation



- **Gen-0**: Baseline DQN (2.80%)

- **Gen-1**: + N-step returns (n=3)#### Quick Evaluation (10 episodes)# Submit PPO training (baseline comparison)

- **Gen-2**: + Reward shaping

```bashsbatch train_ppo.sbatch

## Failed Experiments

python evaluate.py \```

The following techniques were tested but did not improve performance:

  --algorithm dqn \

### 1. Data Augmentation (Gen-2)

- **Technique**: Random crops, color jitter on observations  --model_path "logdir/Gen-5_NoisyNets/dqn_final.zip" \### Monitor Jobs

- **Result**: No improvement, added training time

- **Reason**: Sparse rewards dominate; augmentation doesn't help exploration  --episodes 10 \



### 2. Dueling DQN (Gen-2)  --action-masking \```bash

- **Technique**: Separate value and advantage streams

- **Result**: Similar performance to vanilla DQN  --mask-fallback random# Check job status

- **Reason**: No significant advantage in this environment

```squeue -u $USER

### 3. Prioritized Experience Replay (Gen-3)

- **Technique**: Priority sampling based on TD error

- **Result**: Added complexity without gains

- **Reason**: Uniform sampling sufficient; priority didn't help rare achievements#### Full Evaluation (200-500 episodes, recommended for reporting)# View live logs



### 4. Polyak Averaging (Gen-3)```bashtail -f logs/dqn_<job_id>.out

- **Technique**: Soft target updates (τ=0.005)

- **Result**: Unstable training, worse performancepython evaluate.py \tail -f logs/dynaq_<job_id>.out

- **Reason**: Hard updates work better with action masking

  --algorithm dqn \

### 5. Frame Stacking (Gen-3b)

- **Technique**: Stack last 4 frames for temporal info  --model_path "logdir/Gen-5_NoisyNets/dqn_final.zip" \# Cancel job

- **Result**: No improvement in exploration

- **Reason**: Single frame sufficient; temporal info doesn't help skill discovery  --episodes 200 \scancel <job_id>



### 6. Stone-Specific Reward Shaping (Gen-3)  --action-masking \```

- **Technique**: Extra rewards for stone-related actions

- **Result**: Hurt overall performance  --mask-fallback random

- **Reason**: Too narrow focus; agent ignored other achievements

```### Training Options

### 7. Enhanced Reward Shaping (Gen-4)

- **Technique**: Stronger shaping with ICM

- **Result**: Interfered with ICM learning

- **Reason**: Shaping and curiosity conflict#### Gen-4 ICM EvaluationAll training scripts support the following arguments:



All failed experiments documented in `logdir/` folders with analysis.```bash



## Technical Configurationpython evaluate.py \```bash



### DQN Base Configuration  --algorithm dqn \python train.py [OPTIONS]



```yaml  --model_path "logdir/Gen-4_ICMlite/dqn_final.zip" \

Algorithm: DQN (Stable-Baselines3)

Network: CNN + MLP  --episodes 200 \Options:

  CNN: 3 conv layers [32, 64, 64 filters]

  MLP: 256 hidden units  --action-masking \  --algorithm {ppo,dqn,dynaq}  Algorithm to train

Learning Rate: 1e-4

Batch Size: 32  --mask-fallback random \  --steps STEPS                Training steps (default: 1M)

Gamma: 0.99

Buffer Size: 100,000  --icm-lite  --seed SEED                  Random seed (default: 42)

Learning Starts: 50,000

Target Update Interval: 10,000 (Gen-0-4)```  --eval_freq FREQ             Evaluation frequency (default: 50K)

Target Update Interval: 2,000 (Gen-5, hard updates)

Gradient Steps: 1

```

#### Gen-3c Action Masking Evaluation  # Dyna-Q specific:

### Hardware & Performance

```bash  --planning_steps N           Planning steps per real step (default: 5)

**Requirements**:

- GPU: CUDA-enabled NVIDIA GPU (recommended)python evaluate.py \  --prioritized                Use prioritized sweeping

- RAM: 8GB minimum, 16GB recommended

- Storage: ~5GB per generation  --algorithm dqn \  --exploration_bonus KAPPA    Exploration bonus for Dyna-Q+ (default: 0.0)



**Training Time** (1M steps):  --model_path "logdir/Gen-3c_RandomValid/dqn_final.zip" \```

- GPU (CUDA): 3-6 hours

- CPU: 30-60 hours  --episodes 200 \



**Evaluation Time** (200 episodes):  --action-masking \## Training Results

- ~30-60 minutes

  --mask-fallback random

## Visualization

```Results are saved in timestamped directories:

### Generate Plots

```

```bash

# Standard evaluation plots### Evaluation Optionslogdir/crafter_{algorithm}_{timestamp}/

python plots/generate_report_plots.py logdir/Gen-5_NoisyNets

├── stats.jsonl              # Episode statistics (Crafter format)

# Optional plots (learning curves, etc.)

python plots/generate_optional_plots.py logdir/Gen-5_NoisyNetsKey arguments for `evaluate.py`:├── {algorithm}_final.zip    # Final model checkpoint (SB3 format)

```

└── {algorithm}_final.pt     # Final model checkpoint (PyTorch format)

### Visualize Agent

``````

```bash

# Watch agent play (saves video)--algorithm {ppo,dqn}       Algorithm used (default: dqn)

python visualize_agent.py \

  --model_path "logdir/Gen-5_NoisyNets/dqn_final.zip" \--model_path PATH           Path to trained model (.zip file)## Implementation Status

  --episodes 5 \

  --action-masking \--episodes N                Number of evaluation episodes (default: 100)

  --mask-fallback random

```--seed SEED                 Random seed (default: 42)### ✅ Completed



## Cluster Training- [x] Project setup and conda environment



For SLURM cluster environments:# Must match training configuration:- [x] Unified training script (PPO, DQN)



```bash--action-masking            Enable if model trained with action masking- [x] Evaluation infrastructure

# Submit training job

sbatch train_dqn.sbatch--mask-fallback {zeros,random}  Must match training fallback- [x] Cluster training scripts (conda-based)



# Monitor job--icm-lite                  Enable if model trained with ICM- [x] DQN iterative improvements (6 generations)

squeue -u $USER

tail -f logs/dqn_*.out--noisyNets                 Enable if model trained with NoisyNets (auto-detected)- [x] Documentation (README_DQN.md with comprehensive guide)



# Cancel job```

scancel <job_id>

```### 🚧 In Progress



**Cluster Configuration**:### Evaluation Outputs- [ ] Dyna-Q agent implementation (`src/agents/dynaq_agent.py`)

- Partition: bigbatch

- GPUs: 1 GPU per job- [ ] World model (`src/models/world_model.py`)

- CPUs: 16 cores

- Time limit: 24 hoursEvaluation generates:- [ ] Prioritized sweeping (`src/models/prioritized_sweeping.py`)

- Environment: Conda (auto-created from YAML)

- **JSON report** - Detailed metrics and achievement rates

## Key Findings

- **Text summary** - Human-readable results### 📋 Planned

### What Worked ✅

- **Plots**:- [ ] Baseline Dyna-Q training (Eval 1)

1. **NoisyNets** (+35% improvement)

   - Most effective exploration technique  - `achievement_rates.png` - Achievement unlock rates (22 achievements)- [ ] Prioritized sweeping improvement (Eval 2)

   - Parameter noise > action noise for skill discovery

   - Stable training, fast convergence  - `summary_metrics.png` - Crafter score, reward, episode length- [ ] Dyna-Q+ with exploration bonuses (Eval 3)



2. **Action Masking** (+55% vs baseline)- [ ] Final comparison (DQN vs Dyna-Q)

   - Random-valid fallback critical

   - Reduced invalid actions, maintained exploration**Key Metrics**:- [ ] Report writing

   - Foundation for all later improvements

- **Crafter Score** - Geometric mean of achievement unlock rates (primary metric)

3. **Hard Target Updates** (Gen-5)

   - τ=1.0, interval=2000 stable with NoisyNets- **Average Reward** - Mean episode reward## Expected Results

   - 50% faster training than soft updates

- **Average Length** - Mean episode length

4. **N-Step Returns**

   - Improved credit assignment- **Achievement Rates** - Unlock percentage for each of 22 achievements| Evaluation | Algorithm | Target Score | Key Feature |

   - Better for sparse rewards

|------------|-----------|--------------|-------------|

### What Didn't Work ❌

## Generation Details| **Eval 1** | Baseline Dyna-Q | 0.5-2% | Planning (5 steps/real step) |

1. **ICM Curiosity** (+1.2% only)

   - Minimal impact in sparse reward setting| **Eval 2** | + Prioritized Sweeping | 3-8% | Focused planning |

   - Curiosity doesn't solve exploration bottlenecks

### Gen-5: NoisyNets (5.93%) ✅ CURRENT BEST| **Eval 3** | + Exploration Bonus | 8-15% | Directed exploration |

2. **Reward Shaping**

   - Helped early learning but limited late-game

   - Can interfere with other techniques (ICM)

**Core Innovation**: NoisyNets (Factorized Gaussian Noise) for parameter-space exploration**Sample Efficiency Analysis:**

3. **PER/Dueling/Data Aug/Frame Stack**

   - Added complexity without clear benefits- **DQN (model-free)**: ~800K-1M steps to achieve 1% score

   - Not addressing core exploration problem

**Architecture**:- **Dyna-Q (model-based)**: ~200K-400K steps to achieve 1% score (2-5× faster)

### Bottlenecks 🚧

```

1. **Stone Pickaxe** (0%)

   - Only 2.4% stone collectionCNN → MLP(256) → NoisyLinear(256,256) → ReLU → NoisyLinear(256,17)## Technical Notes

   - Need 10-20% stone rate for reliable crafting

Noise: μ + σ ⊙ ε, where f(ε) = sign(ε) * sqrt(|ε|)

2. **Late-Game Progression** (Coal/Iron/Diamond all 0%)

   - Tech tree blocking prevents progressionσ_init: 0.5### Crafter Environment

   - Requires different approach (hierarchical RL?)

```- **Direct API**: Uses `crafter.Env()` directly (bypasses Gym/Gymnasium)

3. **Tech Tree Dependencies**

   - Linear progression required- **Wrapper**: `CrafterWrapper` in train.py handles API normalization

   - Can't skip tiers

**Key Results**:- **Recorder**: `crafter.Recorder` automatically logs stats.jsonl

## Future Directions

- **Crafter Score**: 5.93% (+35.4% vs Gen-4)- **Observations**: 64×64×3 RGB numpy arrays

If continuing development:

- **Wood Pickaxe**: 21.2% (+45% vs Gen-4's 14.6%)- **Actions**: 17 discrete actions (0-16)

1. **Hierarchical RL**

   - High-level goal planning- **Stone Collection**: 2.4% (+300% vs Gen-4's 0.6%)

   - Low-level skill execution

   - Could break stone pickaxe barrier- **Training Time**: 3.08 hours (50% faster than Gen-4)### Conda vs Pip



2. **Curriculum Learning**This project uses **conda** for reproducible environments:

   - Phase 1: Master wood tier

   - Phase 2: Stone tier**Breakthrough Achievements**:- ✅ Consistent Python version (3.10)

   - Phase 3: Coal/iron

- Wake Up: 97.8% (recovered from Gen-4's 90.4%)- ✅ Compatible PyTorch + CUDA on cluster

3. **Hybrid Exploration**

   - NoisyNets + state coverage bonus- Collect Drink: 42.2% (recovered from Gen-4's 30.4%)- ✅ Faster package resolution

   - May reach rare achievements

- Make Wood Pickaxe: 21.2% (new milestone)- ✅ Better dependency isolation

4. **Longer Training**

   - 5M-10M steps- Collect Stone: 2.4% (4x improvement)

   - May improve late-game

- Place Stone: 1.4% (first time achieved)### Cluster Configuration

5. **Multi-Task Learning**

   - Auxiliary prediction tasksScripts are configured for:

   - Better sample efficiency

### Gen-4: ICM Lite (4.38%)- **Partition**: `bigbatch`

## References

- **GPUs**: 1 GPU per job (CUDA 12.6)

### Papers

**Core Innovation**: Intrinsic Curiosity Module (ICM) for curiosity-driven exploration- **CPUs**: 16 cores

- **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"

- **NoisyNets**: Fortunato et al. (2017) - "Noisy Networks for Exploration"- **Time**: 24 hours

- **Crafter**: Hafner (2021) - "Benchmarking the Spectrum of Agent Capabilities"

- **ICM**: Pathak et al. (2017) - "Curiosity-driven Exploration by Self-supervised Prediction"**Key Results**:- **Conda**: Auto-creates environment from YAML



### Code- **Crafter Score**: 4.38% (+1.2% vs Gen-3c)



- **Crafter**: https://github.com/danijar/crafter- Marginal improvement, validated ICM's limited impact in sparse reward environments## Team Members

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

- **PyTorch**: https://pytorch.org/- **Anand Patel** (Student #: _TO_BE_FILLED_)



## Citation### Gen-3c: Action Masking - Random Valid (4.33%)  - Role: Dyna-Q Implementation (External Algorithm)



```- **Partner Name** (Student #: _TO_BE_FILLED_)

Crafter DQN Project (2024-2025)

Achieved 5.93% Crafter Score through iterative DQN improvements**Core Innovation**: Invalid action masking with random-valid fallback  - Role: DQN Implementation (Course Algorithm)

Best technique: NoisyNets (Fortunato et al. 2017)

Environment: Crafter (Hafner, 2021)

```

**Key Results**:## References

## License

- **Crafter Score**: 4.33%

This project is for educational purposes as part of COMS4061A coursework.

- Reduced sticky action loops, maintained exploration### Papers

---

1. **Dyna-Q**: Sutton & Barto 1996/2018 - Reinforcement Learning: An Introduction (Chapter 8)

**Last Updated**: October 2024  

**Status**: Complete (Gen-5 final deliverable)  ### Gen-0-2: Foundation Generations2. **Prioritized Sweeping**: Moore & Atkeson 1993

**Best Model**: `logdir/Gen-5_NoisyNets/dqn_final.zip` (5.93% Crafter Score)

3. **Crafter Benchmark**: Hafner 2021 - https://arxiv.org/abs/2109.06780

- **Gen-0**: Baseline DQN (2.80%)

- **Gen-1**: N-step returns (n=3)### Code

- **Gen-2**: Reward shaping experiments- **Crafter GitHub**: https://github.com/danijar/crafter

- **Skeleton Code**: https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git

## Technical Details- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/



### Crafter Environment## License

This project is for educational purposes as part of COMS4061A/COMS7071A coursework.

- **Observation**: 64×64×3 RGB images
- **Action Space**: 17 discrete actions
- **Achievements**: 22 hierarchical achievements (basic → wood → stone → iron → diamond)
- **Episode Length**: Maximum 10,000 steps
- **Reward**: Sparse (+1 per achievement unlock)

### DQN Configuration (Gen-5)

```yaml
Network:
  type: CNN + NoisyMLP
  features: [32, 64, 64]
  mlp_size: 256
  noisy_layers: 2 (last two layers)
  
Training:
  steps: 1,000,000
  learning_rate: 1e-4
  batch_size: 32
  gamma: 0.99
  buffer_size: 100,000
  learning_starts: 50,000
  
Target Updates:
  type: hard
  tau: 1.0
  interval: 2000
  
Exploration:
  type: NoisyNets
  sigma_init: 0.5
  epsilon: 0.01 (minimal, noise dominates)
  
Action Masking:
  enabled: true
  fallback: random-valid
```

### Hardware Requirements

- **GPU**: Recommended (CUDA-enabled NVIDIA GPU)
- **CPU Training**: Possible but ~10x slower
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~5GB per generation (models + logs)

### Training Time

- **GPU (CUDA)**: ~3-6 hours for 1M steps
- **CPU**: ~30-60 hours for 1M steps
- **Evaluation (200 episodes)**: ~30-60 minutes

## Failed Experiments

The following approaches were tested but failed to improve performance:

1. **Data Augmentation** - Image augmentation didn't help sparse rewards
2. **Dueling DQN** - No significant advantage over vanilla DQN
3. **Prioritized Experience Replay (PER)** - Added complexity without gains
4. **Polyak Averaging** - Soft target updates unstable
5. **Frame Stacking** - Temporal information didn't improve exploration
6. **Stone-Specific Reward Shaping** - Too narrow, hurt overall performance
7. **Enhanced Reward Shaping (Gen-4)** - Interfered with ICM learning

All failed experiments are documented in `logdir/` with analysis.

## Visualization

### Generate Plots

```bash
# Generate standard plots for a generation
python plots/generate_report_plots.py logdir/Gen-5_NoisyNets

# Generate optional plots (learning curves, etc.)
python plots/generate_optional_plots.py logdir/Gen-5_NoisyNets
```

### Visualize Agent Playing

```bash
# Watch agent play (saves video)
python visualize_agent.py \
  --model_path "logdir/Gen-5_NoisyNets/dqn_final.zip" \
  --episodes 5 \
  --action-masking \
  --mask-fallback random
```

## Cluster Training

For cluster environments (SLURM):

```bash
# Submit Gen-5 training job
sbatch train_dqn.sbatch

# Monitor job
squeue -u $USER
tail -f logs/dqn_*.out

# Cancel job
scancel <job_id>
```

## Key Findings

### What Worked ✅

1. **NoisyNets** - Most effective exploration technique (+35% improvement)
2. **Hard Target Updates** - Stable with NoisyNets, 50% faster training
3. **Action Masking** - Reduced invalid actions, maintained exploration
4. **N-Step Returns** - Improved credit assignment

### What Didn't Work ❌

1. **ICM Curiosity** - Minimal impact (+1.2%) in sparse reward setting
2. **Reward Shaping** - Helped early but limited late-game exploration
3. **PER/Dueling/Data Aug** - Added complexity without clear benefits

### Bottlenecks 🚧

1. **Stone Pickaxe** - Only 2.4% stone collection, 0% stone pickaxe crafting
2. **Late-Game Progression** - Coal/iron/diamond achievements at 0%
3. **Tech Tree Dependency** - Blocking prevents access to higher tiers

## Future Directions

Potential improvements if continuing development:

1. **Hierarchical RL** - High-level goals → low-level skills
2. **Curriculum Learning** - Progressive difficulty (wood → stone → iron)
3. **Hybrid Exploration** - NoisyNets + state coverage bonus
4. **Longer Training** - 5M-10M steps for rare achievements
5. **Multi-Task Learning** - Auxiliary prediction tasks

## Citation

If using this code or referencing these results:

```
Crafter DQN Project (2025)
Achieved 5.93% Crafter Score through iterative DQN improvements
Final technique: NoisyNets (Fortunato et al. 2017)
Environment: Crafter (Hafner, 2021)
```

### References

- **NoisyNets**: Fortunato et al. (2017) - "Noisy Networks for Exploration"
- **Crafter**: Hafner (2021) - "Benchmarking the Spectrum of Agent Capabilities"
- **DQN**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- **ICM**: Pathak et al. (2017) - "Curiosity-driven Exploration"

## License

This project is for educational purposes as part of COMS4061A/COMS7071A coursework.

## Acknowledgments

- Crafter environment by Danijar Hafner
- Stable-Baselines3 library
- PyTorch and CUDA for GPU acceleration

---

**Last Updated**: 27 October 2025  
**Status**: Complete (Gen-5 final deliverable)  
**Best Model**: `logdir/Gen-5_NoisyNets/dqn_final.zip` (5.93% Crafter Score)

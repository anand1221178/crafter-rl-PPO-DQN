# Generation 3: Prioritized Experience Replay (PER) - FAILED ❌

**Training Date:** [DATE REMOVED]  
**Status:** FAILED (Regression: -28.0% vs Gen-1)  
**Crafter Score:** 2.54% (Gen-1: 3.53%, Gen-0: 2.80%)

---

## 📋 Motivation

After two architectural failures (Data Augmentation Gen-2a: -14%, Dueling DQN Gen-2b: -18%), we pivoted to an **algorithmic improvement** that directly addresses sampling efficiency: **Prioritized Experience Replay (PER)**.

The hypothesis was that Crafter's sparse rewards and rare achievements (e.g., coal collection, tool crafting) would benefit from prioritized sampling of high-TD-error transitions, allowing the agent to learn faster from important/surprising events.

**Key Innovation:** PER changes **which transitions we learn from**, not the network architecture. This avoids the capacity/complexity issues that plagued Gen-2.

**Reference:** Schaul et al. (2015) - "Prioritized Experience Replay" (https://arxiv.org/abs/1511.05952)

---

## 🔧 Implementation Details

### Custom PER Buffer for SB3
**Files:** `per_buffer.py`, `dqn_per.py`

#### 1. PrioritizedReplayBuffer (per_buffer.py)
Extends SB3's `ReplayBuffer` with priority-based sampling:

```python
class PrioritizedReplayBuffer(ReplayBuffer):
    - Priority storage: priorities[buffer_size]
    - Priority assignment: p_i = |TD_error_i| + ε
    - Sampling probability: P(i) ∝ p_i^α (α=0.6)
    - Importance weights: w_i = (N·P(i))^(-β) / max(w)
    - Beta annealing: β: 0.4 → 1.0 over 1M steps
    - New samples: Assigned max priority (ensures visibility)
```

**Key hyperparameters:**
- **α = 0.6:** Prioritization strength (0=uniform, 1=full prioritization)
- **β_start = 0.4:** Initial importance sampling correction
- **β_end = 1.0:** Final correction (full bias correction)
- **β_frames = 1,000,000:** Linear annealing over training

#### 2. DQN_PER Wrapper (dqn_per.py)
Extends SB3's DQN to update priorities during training:

```python
class DQN_PER(DQN):
    def train():
        - Compute TD-errors: δ = Q_target - Q_current
        - Apply importance weights to loss: loss * w_i
        - Update priorities: buffer.update_priorities(indices, δ)
```

**Critical Integration:** SB3's default DQN doesn't call `update_priorities()`, so this wrapper is essential for PER to function.

#### 3. Implementation Quality
Our implementation **surpasses ** in several ways:
- ✅ **Automatic beta annealing** ('s had a broken stub that never updated)
- ✅ **No duplicate sampling** (`replace=False` for batch diversity)
- ✅ **Proper SB3 integration** (correctly overrides `_get_samples()`)
- ✅ **Full documentation** and error handling

---

## 🎯 Training Configuration

**Environment:** CrafterPartial-v1 (64×64×3 RGB, 17 actions)  
**Algorithm:** DQN with PER + N-Step Returns  
**Training Steps:** 1,000,000  
**Seed:** 42  
**Hardware:** RTX 3070 Ti (8GB VRAM)  
**Framework:** Stable-Baselines3 2.7.0, PyTorch 2.5.1, CUDA 12.1  
**Environment:** crafter_env (conda)

**Key Settings:**
- **Architecture:** Standard CnnPolicy → [256×256] MLP (Gen-1 baseline)
- **N-Step Returns:** n=3 (kept from successful Gen-1)
- **PER Alpha:** 0.6 (prioritization strength)
- **PER Beta:** 0.4 → 1.0 (importance sampling correction)
- **Buffer Size:** 100,000 transitions
- **Batch Size:** 32
- **Learning Rate:** 1e-4

---

## 📊 Results Analysis

### Performance Metrics
| Metric | Gen-3 PER | Gen-1 N-Step | Gen-0 Baseline | vs Gen-1 | vs Gen-0 |
|--------|-----------|--------------|----------------|----------|----------|
| **Crafter Score** | **2.54%** | **3.53%** | **2.80%** | **-28.0%** ❌ | **-9.3%** ❌ |
| Avg Reward | 2.36 | 3.37 | 3.19 | -30.0% | -26.0% |
| Avg Episode Length | 179.08 | 173.72 | 171.05 | +3.1% | +4.7% |
| Total Episodes | 5,584 | 5,760 | 5,964 | -3.1% | -6.4% |

### Achievement Breakdown: Catastrophic Regressions

**Core Survival Skills (Massive Collapse):**
| Achievement | Gen-3 PER | Gen-1 | Gen-0 | vs Gen-1 | Status |
|-------------|-----------|-------|-------|----------|--------|
| Collect Wood | 45.43% | 63.47% | 57.05% | **-28.4%** | 🔴 Severe |
| Place Table | 19.29% | 26.96% | 27.70% | **-28.5%** | 🔴 Severe |
| Collect Sapling | 73.76% | 95.83% | 94.59% | **-23.0%** | 🔴 Severe |
| Place Plant | 65.31% | 94.21% | 94.85% | **-30.7%** | 🔴 Critical |
| Wake Up | 75.70% | 97.51% | 97.92% | **-22.4%** | 🔴 Severe |

**Tool Crafting (Near Zero):**
| Achievement | Gen-3 PER | Gen-1 | Gen-0 | vs Gen-1 | Status |
|-------------|-----------|-------|-------|----------|--------|
| Make Wood Pickaxe | 0.84% | 0.23% | 0.12% | +265% | ⚠️ Still negligible |
| Make Wood Sword | 1.02% | 0.27% | 0.07% | +278% | ⚠️ Still negligible |

**Resource Collection (Complete Failure):**
| Achievement | Gen-3 PER | Gen-1 | Gen-0 | vs Gen-1 | Status |
|-------------|-----------|-------|-------|----------|--------|
| Collect Stone | 0.04% | 0.02% | 0.03% | +100% | ⚠️ Still ~0% |
| Collect Coal | 0.00% | 0.00% | 0.00% | - | ❌ Never achieved |

---

## 🔍 Failure Analysis

### Why Did PER Fail So Badly?

#### 1. **Sampling Bias Disrupted Foundational Learning**
PER prioritizes **surprising** transitions (high TD-error), but in Crafter:
- Early-game actions (collect wood, place plant) have **low TD-error** once learned
- These foundational skills become **under-sampled** as training progresses
- Agent forgets basics while chasing rare high-error transitions
- Result: **Catastrophic forgetting** of core survival behaviors

**Evidence:**
- Place Plant: 94.21% (Gen-1) → 65.31% (Gen-3) = **-30.7% collapse**
- Collect Sapling: 95.83% → 73.76% = **-23.0% regression**
- These are the most fundamental skills; their collapse indicates sampling pathology

#### 2. **Importance Sampling Weights Failed to Correct Bias**
The theory: IS weights $w_i = (N \cdot P(i))^{-\beta}$ correct for sampling bias.

**In practice:**
- Beta annealing (0.4 → 1.0) reduces correction early in training
- Early stages are **critical** for learning foundational skills
- By the time β → 1.0 (full correction), agent has already learned bad policy
- Weighting loss differently ≠ fixing catastrophic forgetting

#### 3. **N-Step Returns + PER = Compounding Sample Inefficiency**
Gen-1's success with n=3 relied on **uniform sampling** of recent experience.

PER + N-Step interaction:
- N-step targets compute TD-errors over 3-step sequences
- PER prioritizes entire sequences based on aggregate error
- **Problem:** Low-error foundational sequences get starved
- **Worse:** N-step blurs which transition in sequence caused error

**Hypothesis:** PER + N-step creates a **temporal credit assignment problem** that hurts stable learning.

#### 4. **Crafter's Sparse Rewards Break PER's Assumptions**
PER was designed for Atari (dense rewards, clear value gradients).

**Crafter characteristics that break PER:**
- **Milestone rewards:** Sparse, episodic (only on achievement unlock)
- **Compositional skills:** Must master A before B is possible
- **Delayed consequences:** Forgetting wood collection → can't build table → can't progress
- **Reward homogeneity:** Most transitions have reward=0 or small survival rewards

**Result:** TD-errors are **noisy signals** in Crafter, not reliable importance indicators.

#### 5. **PER Hyperparameters Were Suboptimal**
Our settings (α=0.6, β: 0.4→1.0) are standard for Atari, but may be too aggressive for Crafter:
- **α=0.6:** May over-prioritize outliers in sparse reward setting
- **β_start=0.4:** Insufficient bias correction during critical early learning
- **No priority decay:** Old high-priority samples can dominate indefinitely

**Better settings might be:**
- α=0.3-0.4 (gentler prioritization)
- β_start=0.6-0.7 (stronger early correction)
- Priority decay: p_i *= 0.99 per step (age out old priorities)

---

## 💡 Lessons Learned

### What Failed
❌ PER caused **catastrophic forgetting** of foundational skills  
❌ Sampling bias more harmful than uniform replay in sparse rewards  
❌ IS weights insufficient to prevent collapse of early-learned behaviors  
❌ N-step + PER interaction created credit assignment issues  
❌ Standard PER hyperparameters (Atari-tuned) unsuitable for Crafter  

### Key Insights

#### 1. **Sampling Strategy is Critical in Compositional Domains**
Crafter requires **stable mastery** of foundational skills (sequence: wood → table → tools → resources).
- **PER prioritizes** novel/surprising transitions
- **Uniform sampling** maintains access to foundational skills
- **Result:** Uniform sampling is more stable for compositional learning

#### 2. **Not All Improvements Transfer Across Domains**
- PER: +25-50% in Atari (dense rewards, single-task)
- PER: -28% in Crafter (sparse rewards, multi-task, compositional)
- **Lesson:** "State-of-the-art" in one domain ≠ universal improvement

#### 3. **Architecture vs Sampling Trade-offs**
- Gen-2 (Dueling): Architecture change, sampling unchanged → -18% (bad)
- Gen-3 (PER): Sampling change, architecture unchanged → -28% (worse!)
- **Surprising:** Changing sampling harmed more than changing architecture

#### 4. **Importance Sampling is Not a Silver Bullet**
IS weights theoretically unbias PER, but in practice:
- Early training (low β) allows bias to corrupt learning
- Catastrophic forgetting happens before full correction (β=1.0)
- **Takeaway:** IS theory assumes convergence; doesn't prevent transient failures

---

## 🎓 Technical Takeaways

### For Future Crafter Improvements

#### ✅ **What to Try Next (Based on 3 Failures)**

**1. Reward Shaping (Gen-4 Candidate)**
- **Why:** Directly address sparse reward problem without sampling/architecture changes
- **How:** Add small milestone bonuses for foundational skills
  - +0.2 for first wood collection
  - +0.2 for first table placement
  - +0.3 for first pickaxe crafted
- **Cap:** Max 30% of episode return from shaping
- **Advantage:** Denser reward signal → better TD-errors → stable learning
- **Risk:** Low (shaping used successfully in robotics, curriculum learning)

**2. Episodic Memory / Hindsight Experience Replay**
- **Why:** Store successful skill sequences, replay strategically
- **How:** Track achievement unlocks, oversample successful episodes
- **Advantage:** Learn from rare successes without forgetting common skills
- **Risk:** Medium (implementation complexity)

**3. Longer Training (1.5-2M steps)**
- **Why:** All 3 failures may be **under-training** vs architecture/sampling
- **How:** Run Gen-1 (n-step=3, uniform sampling) for 2M steps
- **Advantage:** Simple, no new failure modes
- **Risk:** Low (just more compute)

**4. Simpler Changes: Target Update Frequency**
- **Why:** More frequent target updates → faster learning of value function
- **How:** target_update_interval: 10,000 → 5,000
- **Advantage:** Minimal change, well-understood
- **Risk:** Very low

#### ❌ **What to Avoid (Learned from Gen-2 & Gen-3)**

- **Complex architectures** (Dueling, Rainbow, C51): Hurt more than help
- **Aggressive sampling changes** (PER with α>0.5): Causes forgetting
- **Data augmentation** (Random crops, color jitter): Breaks spatial learning
- **Multi-objective learning** without careful balancing

---

## 📊 Detailed Comparison: All Generations

| Generation | Crafter Score | vs Baseline | Key Change | Status |
|------------|---------------|-------------|------------|--------|
| **Gen-0** | 2.80% | - | Baseline DQN | ✅ Success |
| **Gen-1** | **3.53%** | **+26.1%** | N-Step Returns (n=3) | ✅ **Best** |
| Gen-2a | 3.03% | +8.2% | Data Augmentation | ❌ Failed (-14% vs Gen-1) |
| Gen-2b | 2.88% | +2.9% | Dueling DQN | ❌ Failed (-18% vs Gen-1) |
| **Gen-3** | **2.54%** | **-9.3%** | PER (α=0.6, β: 0.4→1.0) | ❌ **Worst** |

**Ranking:** Gen-1 > Gen-2a > Gen-0 > Gen-2b > Gen-3

---

## 📁 Results Location

**Folder:** `logdir/Gen-3_PER_FAILED_20251025_015149/`
- `dqn_final.zip` - Trained model weights
- `stats.jsonl` - Episode-by-episode achievements (5,584 episodes)
- `DQN_1/` - TensorBoard training logs
- `plots/` - Achievement and metric visualizations
- `evaluation_report_*.json` - Detailed evaluation metrics
- `evaluation_summary_*.txt` - Text summary of results

---

## ⏭️ Next Steps: Gen-4 Strategy

### Recommended: Reward Shaping

**Rationale:**
- 3 failures (Data Aug, Dueling, PER) all modified **how agent learns**
- Common failure mode: Collapse of foundational skills
- **Root cause:** Sparse rewards provide insufficient gradient for early-game skills
- **Solution:** Make foundational skills more rewarding

**Implementation:**
```python
class ShapingWrapper(gym.Wrapper):
    MILESTONES = {
        "collect_wood": 0.2,      # First wood → enables table
        "place_table": 0.2,       # Table → enables tools
        "make_wood_pickaxe": 0.3, # Pickaxe → enables stone/coal
        "collect_coal": 0.3,      # Coal → enables furnace
    }
    # Award bonus ONCE per episode per milestone
    # Cap total shaping: ≤30% of typical survival return
```

**Expected Outcome:**
- +15-25% Crafter Score improvement over Gen-1
- Stable foundational skills (wood >70%, table >35%)
- First breakthrough past pickaxe crafting bottleneck

### Alternative: Just Train Longer
- Re-run Gen-1 (n_steps=3, uniform sampling) for 2M steps
- Hypothesis: 1M steps insufficient for convergence
- Low risk, high potential payoff

---

## 🏁 Conclusion

**Generation 3 (PER) is a FAILURE.**

Despite being a theoretically sound and well-implemented algorithm (superior to ), PER caused **catastrophic forgetting** of foundational skills due to sampling bias incompatible with Crafter's compositional, sparse-reward structure.

**Key Learning:** State-of-the-art algorithms from dense-reward domains (Atari) can actively harm performance in sparse, compositional environments. Uniform sampling + longer training may outperform sophisticated sampling strategies in these settings.

**Recommendation:** Pivot to **Reward Shaping (Gen-4)** to address the root cause (sparse rewards) rather than symptoms (sampling, architecture).

---

**Status:** FAILED - Worst performance of all generations (-28% vs Gen-1)  
**Recommendation:** Do not use this model. Revert to Gen-1 baseline + add reward shaping for Gen-4.

# Generation 2: Dueling DQN Architecture - FAILED ❌

**Training Date:** [DATE REMOVED]  
**Status:** FAILED (Regression: -18.4% vs Gen-1)  
**Crafter Score:** 2.88% (Gen-1: 3.53%, Gen-0: 2.80%)

---

## 📋 Motivation

After the Data Augmentation approach failed (Gen-2 attempt 1), we pivoted to an **architectural improvement**: implementing a **Dueling DQN** architecture. The Dueling architecture separates the Q-value estimation into two streams:

1. **Value Stream V(s)**: Estimates the value of being in state s
2. **Advantage Stream A(s,a)**: Estimates the advantage of each action in state s

The Q-values are then computed as: **Q(s,a) = V(s) + (A(s,a) - mean(A))**

This decomposition allows the network to learn which states are valuable without needing to learn the effect of each action for every state, potentially leading to:
- Better generalization across actions
- More stable value estimation
- Improved sample efficiency in sparse reward environments

---

## 🔧 Implementation Details

### Custom SB3 Integration Challenge
Stable-Baselines3 v2.7.0 **does not have built-in Dueling DQN support**, requiring a completely custom implementation:

**File:** `dueling_dqn.py`

#### 1. DuelingQNetwork (nn.Module)
Custom PyTorch module implementing the dueling architecture:
```python
class DuelingQNetwork(nn.Module):
    - Shared CNN feature extractor (from DQN policy)
    - Shared MLP: [256 → 256] with ReLU
    - Value stream: [256 → 128 → 1] 
    - Advantage stream: [256 → 128 → 17 actions]
    - Aggregation: Q = V + (A - mean(A))
```

#### 2. DuelingDQNPolicy (DQNPolicy)
Custom policy class inheriting from SB3's DQNPolicy:
- Overrides `_build()` method to control initialization order
- Manually creates `features_extractor` before Q-network
- Uses custom `DuelingQNetwork` instead of standard Q-network

#### 3. Debugging Iterations
Multiple fixes required for SB3 compatibility:
- **Issue 1:** `features_extractor` was None during `make_q_net()` call
  - **Fix:** Override `_build()` to create features_extractor first
- **Issue 2:** Missing `set_training_mode()` method
  - **Fix:** Added method calling `self.train(mode)`
- **Issue 3:** Missing `_predict()` method
  - **Fix:** Added method returning `argmax(Q-values)`
- **Issue 4:** Dtype mismatch (Byte vs Float) in forward pass
  - **Fix:** Added normalization `obs.float() / 255.0` in forward()

---

## 🎯 Training Configuration

**Environment:** CrafterPartial-v1 (64×64×3 RGB, 17 actions)  
**Algorithm:** DQN with custom Dueling architecture  
**Training Steps:** 1,000,000  
**Seed:** 42  
**N-Step Returns:** 3 (carried over from Gen-1)  
**Hardware:** RTX 3070 Ti (8GB VRAM)  
**Framework:** Stable-Baselines3 2.7.0, PyTorch 2.5.1, CUDA 12.1  
**Environment:** crafter_env (conda)

---

## 📊 Results Analysis

### Performance Metrics
| Metric | Gen-2 Dueling | Gen-1 N-Step | Gen-0 Baseline | vs Gen-1 | vs Gen-0 |
|--------|---------------|--------------|----------------|----------|----------|
| **Crafter Score** | **2.88%** | **3.53%** | **2.80%** | **-18.4%** ❌ | **+2.9%** ✓ |
| Avg Reward | 2.97 | 3.37 | 3.19 | -11.9% | -6.9% |
| Avg Episode Length | 178.46 | 173.72 | 171.05 | +2.7% | +4.3% |

### Achievement Breakdown
**Improvements vs Gen-1:**
- Defeat Skeleton: 0.02% → 0.23% (+1050%)
- Defeat Zombie: 0.02% → 8.55% (+42650%)
- Eat Cow: 0.04% → 5.85% (+14525%)
- Make Wood Pickaxe: 0.23% → 0.93% (+304%)
- Make Wood Sword: 0.27% → 1.21% (+348%)

**Regressions vs Gen-1:**
- Collect Sapling: 95.83% → 90.38% (-5.7%)
- Collect Wood: 63.47% → 46.92% (-26.1%)
- Place Plant: 94.21% → 87.17% (-7.5%)
- Place Table: 26.96% → 17.03% (-36.8%)
- Wake Up: 97.51% → 95.45% (-2.1%)

---

## 🔍 Failure Analysis

### Why Did Dueling DQN Fail?

1. **Catastrophic Regression in Core Skills**
   - **Major drop in wood collection (-26%)** - fundamental early-game action
   - **Table placement collapsed (-37%)** - critical for progression
   - Agent learned more complex behaviors (combat) at expense of basics

2. **Value/Advantage Separation Mismatch**
   - Crafter has **sparse, milestone-based rewards**
   - V(s) may struggle to estimate state values without clear reward signals
   - A(s,a) learned combat advantages but failed at exploration/gathering

3. **Overcomplicated Architecture**
   - Added **50% more parameters** to Q-network (V + A streams)
   - May need more training steps (>1M) for convergence
   - Dueling benefits diminish in environments with action-conditional rewards

4. **Training Dynamics**
   - Combat achievements (skeleton/zombie/cow) jumped dramatically
   - Suggests network focused on high-reward late-game actions
   - Lost foundational behaviors that enable reaching late-game

5. **Environment Characteristics**
   - Crafter's **compositional nature** (sequence of subgoals) may not align with V/A decomposition
   - Standard DQN's unified Q-value may be more suitable for sequential dependencies

---

## 💡 Lessons Learned

### What Worked
✅ Successfully implemented custom Dueling architecture in SB3  
✅ Combat behavior learning significantly improved  
✅ Agent attempted more diverse actions  

### What Failed
❌ Core gathering/crafting skills regressed  
❌ Overall performance dropped 18.4%  
❌ V/A separation didn't benefit sparse reward structure  

### Key Insights
1. **Architectural complexity ≠ Better performance** - Need to match architecture to environment structure
2. **Crafter rewards sequential skills** - Dueling's state-value decomposition may hinder this
3. **Foundation before specialization** - Combat skills useless without gathering/crafting basics
4. **SB3 customization is feasible** but requires deep understanding of internal mechanisms

---

## 🎓 Technical Takeaways

### For Future Improvements
- **Prioritized Experience Replay (PER)** may be more suitable - focuses on important transitions without architectural changes
- **Reward shaping** could guide learning toward foundational skills before complex behaviors
- **Curriculum learning** might prevent regression in basic skills while learning advanced ones

### For Dueling DQN Specifically
- May need **2M+ steps** for convergence in complex environments
- Consider **separate replay buffers** for different skill types
- Could benefit from **auxiliary tasks** to stabilize V(s) learning

---

## 📁 Results Location

**Folder:** `logdir/Gen-2_DuelingDQN_FAILED_20251024_211723/`
- `dqn_final.zip` - Trained model weights
- `stats.jsonl` - Episode-by-episode achievements
- `DQN_1/` - TensorBoard training logs
- `plots/` - Achievement and metric visualizations
- `evaluation_report_*.json` - Detailed evaluation metrics

---

## ⏭️ Next Steps

Given this failure, we'll proceed with:
1. **Gen-3: Prioritized Experience Replay (PER)** - Focus on important transitions, well-supported in SB3
2. **Gen-4: Reward Shaping** - Guide agent toward foundational skills with milestone bonuses

Both approaches avoid architectural complexity while directly addressing learning efficiency and exploration.

---

**Status:** FAILED - Regression in Crafter Score (-18.4%)  
**Recommendation:** Do not use this model. Revert to Gen-1 N-Step baseline for further improvements.

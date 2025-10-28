# Gen-2: Data Augmentation (DrQ-style Random Shifts)

## 🎯 Generation Overview
- **Generation:** 2
- **Status:** ❌ FAILED - Regression from Gen-1
- **Date:** [DATE REMOVED]
- **Training Duration:** ~6 hours

---

## 📊 Configuration

### **Algorithm:** N-Step DQN + Data Augmentation

**Key Changes from Gen-1:**
```python
# NEW: Data Augmentation Wrapper
class RandomShiftAugmentation(gym.Wrapper):
    def __init__(self, env, pad=4):
        # Pad 4 pixels on each side
        # Random crop back to 64×64
        # Effect: Random shifts up/down/left/right
```

**Applied to environment:**
```python
env = CrafterWrapper(recorded_env)
env = RandomShiftAugmentation(env, pad=4)  # NEW
```

**Kept from Gen-1:**
```python
n_steps = 3
policy_kwargs = dict(net_arch=[256, 256])
# All other hyperparameters unchanged
```

---

## 🎯 Motivation (Based on Gen-1 Results)

### **Problem Identified in Gen-1:**
- Tool crafting improved to 5% but plateaued
- Agent may be overfitting to specific pixel positions
- Sample efficiency could be improved
- Learning curves showed early plateau

### **Why Data Augmentation?**

**Theory:** Random shifts improve sample efficiency and generalization in pixel-based RL.

**Based on:**
- "Data-Efficient RL with Self-Predictive Representations" (Schwarzer et al., 2021)
- "Image Augmentation is All You Need" (Kostrikov et al., 2020)
- DrQ (Data-Regularized Q-learning) from Atari domain

**How it works:**
1. Pad observation by 4 pixels (edge replication)
2. Random crop back to 64×64
3. Effect: Agent sees same scene from slightly different "camera positions"

**Expected Benefits:**
1. Spatial invariance: Resources look same if shifted slightly
2. Better generalization: More diverse views of same situation
3. Regularization: Prevents overfitting to exact pixel positions
4. Sample efficiency: Proven +20-40% in Atari games

**Expected Improvements:**
- Crafter Score: 3.53% → 4.2-4.8% (+19-36%)
- Stone collection: 0.46% → 1.5-2.5%
- Better generalization to unseen situations

---

## 📈 Results

### **Core Metrics:**
- **Crafter Score:** **3.03%** (was 3.53%) → **-14.2% REGRESSION** ❌
- **Average Reward:** 2.98 (was 3.40) → **-12.4% regression** ❌
- **Average Episode Length:** 176.30 (was 178.27) → **-1.1% regression** ❌
- **Total Episodes:** 5,672

### **Achievement Comparison (vs Gen-1 - Previous Best):**

**❌ Major Regressions:**
- **Place Table: 49.40% → 21.83% (-56%)** 💥 CATASTROPHIC
- **Make Wood Pickaxe: 4.96% → 2.63% (-47%)** 💥 DESTROYED OUR BEST WIN
- **Collect Wood: 73.72% → 51.66% (-30%)** ❌
- Make Wood Sword: 4.80% → 2.03% (-58%)
- Eat Cow: 6.51% → 4.57% (-30%)

**✓ Minor Gains:**
- Collect Sapling: 82.74% → 86.53% (+4.6%)
- Place Plant: 78.50% → 82.51% (+5.1%)
- Collect Drink: 28.40% → 36.18% (+27%, but still below Gen-0's 42.84%)

**→ No Change:**
- Wake Up: 94.78% → 94.43% (minimal)
- Defeat Zombie: 5.31% → 5.11% (minimal)

---

## 🔍 Analysis

### **Why Data Augmentation FAILED:**

**1. Spatial Information is Critical in Crafter**
- ❌ Unlike Atari where pixel positions don't matter
- ❌ Crafter has fixed resource spawn locations:
  - Trees in specific biomes
  - Water sources have consistent positions
  - Stone deposits in particular areas
- ❌ Random shifts broke positional learning
- ❌ Agent couldn't learn "where things are"

**2. Regularization Too Strong**
- Data augmentation prevents overfitting
- But Crafter REQUIRES "overfitting" to its spatial structure
- Agent needs to memorize resource locations
- Seeing shifted views confused Q-learning

**3. Sample Efficiency Actually Decreased**
- Every transition treated as "novel" due to shifts
- Agent couldn't consolidate learning from similar states
- Replay buffer filled with pseudo-unique transitions
- Worse learning, not better!

**4. Tool Crafting Destroyed**
- Wood pickaxe: 4.96% → 2.63% (-47%)
- This was Gen-1's biggest achievement, completely lost
- Precise positioning needed for crafting table interaction
- Random shifts made table locations unpredictable

### **Fundamental Domain Mismatch:**

**Where DrQ Works:** Atari games (spatially-invariant)
- Breakout: Ball position doesn't matter absolutely
- Pong: Paddle can be anywhere
- Space Invaders: Enemy positions relative

**Where DrQ Fails:** Crafter (spatially-aware)
- Resource locations are meaningful
- Navigating requires absolute position awareness
- Crafting needs precise positioning
- Procedural generation doesn't mean spatial invariance!

---

## 💡 Key Insights Learned

### **Scientific Value - Negative Result:**

This is a **valuable failure** that demonstrates:

1. ✅ **Not all Atari techniques transfer to 3D/procedural worlds**
2. ✅ **Spatial reasoning requires consistent observations**
3. ✅ **"Seeing more variety" can hurt learning**
4. ✅ **Domain characteristics matter for augmentation choice**

### **What We Lost:**
- Tool crafting gains from Gen-1 (completely reversed)
- Stable resource gathering behavior
- Efficient table placement ability
- Overall performance dropped below Gen-1

### **What We Kept:**
- Still above Gen-0 baseline (2.80% → 3.03%, +8%)
- Basic gathering skills intact
- N-step returns still helping (vs pure vanilla DQN)

---

## 🔄 Lessons for Next Generation

### **Decision Point:**
❌ **Data Augmentation is NOT the right approach for Crafter**

### **Why Gen-1 Remains Superior:**
- N-step returns address actual bottleneck (credit assignment)
- No unintended side effects on spatial learning
- Clean improvement without trade-offs

### **Proposed Gen-3 Improvement:** Dueling DQN Architecture
**Pivot Strategy:** Focus on architecture, not observations

**Why Dueling Will Work Better:**
- Doesn't modify observations (keeps spatial info)
- Separates state value from action advantages
- Helps with "waiting" states (nighttime, inventory management)
- Expected: 3.53% → 4.5-5.0% (+27-42%)

**Alternative Considered:**
- Prioritized Experience Replay (learn more from rare events)
- Reward Shaping (explicit guidance for tech tree)

---

## 📁 Files Generated

- **Model:** `dqn_final.zip` (NOT recommended for use)
- **Plots:** `plots/achievement_rates.png`, `plots/summary_metrics.png`
- **Reports:**
  - `evaluation_report_20251024_173123.json`
  - `evaluation_summary_20251024_173123.txt`
- **Training Logs:** `stats.jsonl` (5,672 episodes)

---

## 🎓 Conclusion

**Failed attempt - valuable learning experience.** Data augmentation, while successful in Atari, is counterproductive for Crafter's spatially-aware environment. This negative result validates the importance of domain-appropriate technique selection. Gen-1 (N-Step Returns) remains our best model.

**Next Step:** Pivot to Dueling DQN architecture to improve value estimation without disrupting spatial learning.

---

**Classification:** ❌ FAILED ATTEMPT
- Did not beat previous best (3.53%)
- Regression in key metrics
- Evidence that technique doesn't transfer to this domain

---

**Generated:** [DATE REMOVED]

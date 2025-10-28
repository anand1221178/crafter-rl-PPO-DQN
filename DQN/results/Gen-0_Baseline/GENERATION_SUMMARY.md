# Gen-0: Baseline (Vanilla DQN)

## 🎯 Generation Overview
- **Generation:** 0 (Baseline)
- **Status:** ✅ SUCCESS - Baseline Established
- **Date:** [DATE REMOVED]
- **Training Duration:** ~3 hours

---

## 📊 Configuration

### **Algorithm:** Vanilla DQN (Standard Implementation)

**Hyperparameters:**
```python
learning_rate = 1e-4
buffer_size = 100000
learning_starts = 1000
batch_size = 32
gamma = 0.99
exploration_fraction = 0.75
exploration_final_eps = 0.05
n_steps = 1  # Standard 1-step TD
```

**Network Architecture:**
- CNN Policy (SB3 default)
- Single Q-value head
- No dueling architecture
- Standard fully-connected layers

---

## 🎯 Motivation

**Purpose:** Establish baseline performance for Crafter environment
- Use standard DQN implementation from Stable-Baselines3
- No modifications or enhancements
- Measure raw performance on sparse-reward environment

**Research Questions:**
1. Can vanilla DQN learn basic survival in Crafter?
2. What achievements can be unlocked with standard 1-step TD?
3. Where does credit assignment fail?

---

## 📈 Results

### **Core Metrics:**
- **Crafter Score:** **2.80%**
- **Average Reward:** 2.54
- **Average Episode Length:** 178.09 steps (~3 minutes survival)
- **Total Episodes:** 5,614

### **Achievement Unlock Rates:**

**✅ Strong Achievements (>50%):**
- Wake Up: 71.29%
- Collect Sapling: 68.90%
- Place Plant: 60.19%
- Collect Wood: 58.27%

**⚠️ Moderate Achievements (20-50%):**
- Collect Drink: 42.84%
- Place Table: 31.15%

**❌ Weak Achievements (1-10%):**
- Eat Cow: 4.17%
- Make Wood Pickaxe: 2.33%
- Make Wood Sword: 2.30%
- Defeat Zombie: 2.03%

**❌ Failed Achievements (<1%):**
- Collect Stone: 0.12%
- All advanced tech tree: 0.00%

---

## 🔍 Analysis

### **What Worked:**
1. ✅ **Basic Survival:** Agent learned sleep cycle (71% wake up)
2. ✅ **Simple Gathering:** Can collect wood (58%) and saplings (69%)
3. ✅ **Basic Placement:** Learned to place plants (60%) and tables (31%)
4. ✅ **Short-term Planning:** 2-3 step sequences successful

### **What Failed:**
1. ❌ **Tool Crafting:** Only 2.33% wood pickaxe (requires 6-step sequence)
2. ❌ **Resource Mining:** Cannot mine stone (0.12%) or coal (0.02%)
3. ❌ **Combat:** Poor performance against zombies/skeletons (<2%)
4. ❌ **Advanced Tech:** 0% on all iron/stone/furnace achievements
5. ❌ **Long-term Planning:** Cannot execute multi-step crafting chains

### **Root Causes:**
- **Sparse Rewards:** No credit for intermediate steps in crafting chains
- **Short Credit Assignment:** 1-step TD insufficient for long sequences
- **Poor Value Estimation:** Can't distinguish states where actions don't immediately matter
- **Exploration vs Exploitation:** 75% exploration helps gathering but hurts task completion

---

## 💡 Insights for Next Generation

### **Key Bottleneck:** Credit Assignment
- Wood pickaxe requires 6+ steps: move → chop → collect → craft table → craft stick → craft pickaxe
- 1-step TD only assigns credit to final action
- Agent can't learn intermediate states have value

### **Proposed Gen-1 Improvement:** N-Step Returns
- Use n-step TD targets (n=3 or n=5)
- Better credit assignment for multi-step sequences
- Expected improvement: +20-40% Crafter Score
- Target: 10-15% wood pickaxe unlock rate

---

## 📁 Files Generated

- **Model:** `dqn_final.zip`
- **Plots:** `plots/achievement_rates.png`, `plots/summary_metrics.png`
- **Reports:** 
  - `evaluation_report_20251014_182701.json`
  - `evaluation_summary_20251014_182701.txt`
- **Training Logs:** `stats.jsonl` (5,614 episodes)

---

## 🎓 Conclusion

**Baseline established successfully.** Vanilla DQN can learn basic survival and resource gathering but fails at multi-step tool crafting and advanced tech tree progression. Clear evidence that credit assignment is the primary bottleneck.

**Next Step:** Implement n-step returns to improve credit assignment over longer horizons.

---



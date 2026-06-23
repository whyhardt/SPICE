# Weber 2024 - SPICE Models

Quick reference for working with Weber et al. (2024) laser-shield tracking task models.

## Task Overview

Participants control a circular shield to catch a moving laser beam. The laser position updates trial-by-trial, requiring continuous tracking and learning. Binary feedback: laser caught or missed.

**Key signals:**
- `shield[t]`: shield position at trial t (sin/cos encoding)
- `laser[t]`: laser position at trial t (sin/cos encoding)
- `laser_caught[t]`: binary catch outcome (0 or 1)
- `prediction_error[t]`: laser[t] - belief[t]

## Models

### 1. Simple Dynamic LR Model ([spice_weber2024.py](spice_weber2024.py))

**Architecture:**
```python
logits = (1-alpha) * shield + alpha * belief
alpha = sigmoid(lr_value)
```

**Modules:**
- `belief_update_caught/missed`: learns f(PE) → belief update
- `lr_update_caught/missed`: learns dynamic learning rate (context-dependent)

**Gated Output:** Alpha interpolates between current shield position and internal belief. Alpha is applied at OUTPUT level → guaranteed gradient flow to LR modules.

**Use case:** Baseline model with single adaptive learning rate.

---

### 2. Changepoint Model ([spice_weber2024_changepoint.py](spice_weber2024_changepoint.py))

**Architecture (Bruckner-style):**
```python
logits = (1-alpha) * shield + alpha * belief
alpha = cp_prob * alpha_cp + (1-cp_prob) * alpha_var

# Update/decay pattern based on p(CP) > 0.5 threshold
mask_cp = (cp_prob > 0.5).float()
```

**Modules:**
- `changepoint_detection`: learns p(CP) from PE magnitude
- `changepoint_lr_update/decay`: high LR for changepoints
  - Update fires when `p(CP) > 0.5`
  - Decay fires when `p(CP) ≤ 0.5`
  - Receives `laser_caught` as input
- `uncertainty_lr_update/decay`: baseline LR for stable periods
  - Update fires when `p(CP) ≤ 0.5`
  - Decay fires when `p(CP) > 0.5`
  - Receives `laser_caught` as input
- `belief_update_caught/missed`: learns f(PE) → belief update

**Use case:** Dual learning rates weighted by changepoint probability. More complex, captures switch between exploration (CP) and exploitation (stable).

---

## Model Comparison & Findings

**Best Model: Simple Dynamic LR** ([spice_weber2024.py](spice_weber2024.py))
- Outperforms the changepoint model on this task
- Single adaptive LR is sufficient - may encode both uncertainty and changepoint-like dynamics
- Cleaner, more parsimonious architecture

**Changepoint Model:** Underperforms
- CP detection module learns simple decay (not complex changepoint detection)
- Suggests task may not have strong changepoint structure, or single LR is flexible enough

**Key Architectural Insight: Gated Output Architecture**
- **Problem:** Gradient flow issues when passing one module's output as input to another
- **Solution:** Apply weighting/gating at output level instead of inside modules
- **Benefit:** Clean gradient flow + modules learn independently + interpretable

```python
# ✓ Gated output (working):
alpha = sigmoid(lr_value)
logits = (1-alpha) * shield + alpha * belief

# ✗ Multiplicative input (gradient issues):
belief_update(inputs=(alpha * PE,))
```

---

## Key Files

### Model Definitions
- **[spice_weber2024.py](spice_weber2024.py)** - Simple dynamic LR model
- **[spice_weber2024_changepoint.py](spice_weber2024_changepoint.py)** - Changepoint model with dual LRs

### Data & Training
- **Data:** `weinhardt2026/studies/weber2024/data/weber2024.csv`
- **Training script:** `weinhardt2026/run.py` (see usage below)
- **Loss function:** `clamped_angular_mse` from [benchmarking_weber2024.py](benchmarking_weber2024.py)

### Analysis & Plotting
- **[weber2024.py](weber2024.py)** - Main analysis notebook
  - Loads fitted models
  - Extracts SPICE states (belief, LR, CP dynamics)
  - Plots trial-by-trial dynamics
  - Handles both simple LR and changepoint models automatically
- **[benchmarking_weber2024.py](benchmarking_weber2024.py)** - Data loading, loss functions, environment

---

## Quick Start

### 1. Train & Analyze

Run **[weber2024.py](weber2024.py)** to train and analyze models:

```bash
python weinhardt2026/studies/weber2024/weber2024.py
```

**What it does:**
- Trains model (configurable: `'spice_weber2024'` or `'spice_weber2024_changepoint'`)
- Saves to: `results/[model_name]/`
- Extracts RNN and SPICE states (belief, LR, CP dynamics)
- Plots trial-by-trial dynamics
- Compares RNN vs SPICE predictions

**To modify training:**
Edit parameters in the script:

*Quick testing:*
- `epochs=100-200`
- `ensemble_size=1-3`
- `truncate_dataset=True` (small subset)

*Full training runs:*
- `epochs=500-1000`
- `ensemble_size=10`
- `sindy_weight=0.01`
- `truncate_dataset=False` (whole dataset)
- `compile_forward=True` (faster)
- `learning_rate=0.01`

### 2. Modify Models

**To change architecture:**
1. Edit CONFIG in model file (e.g., add/remove modules)
2. Update forward() method if needed
3. Run training to verify changes work

**To add new control signals:**
1. Add to `library_setup` in CONFIG
2. Pass as `inputs=(...)` in `call_module()`
3. Ensure signal exists in `additional_inputs`

**For inspiration:**
- Check **[spice_bruckner2025.py](../bruckner2025/spice_bruckner2025.py)** for:
  - Dual learning rate patterns (uncertainty + changepoint)
  - Update/decay module structure
  - PE-based changepoint detection
  - Composite alpha formulations

---

## Architecture Notes

### Gated Output (Critical for Gradient Flow)

Both models use **gated output** instead of multiplicative alpha inside belief update:

```python
# ❌ OLD (vanishing gradients):
belief_update(inputs=(alpha * PE,))

# ✓ NEW (clean gradients):
belief_update(inputs=(PE,))
logits = (1-alpha) * shield + alpha * belief
```

**Why:** Alpha appears directly in output computation → gradients flow cleanly to LR modules. Belief module learns f(PE) without constraints → SINDy can discover any functional form.

### Initialization

- `belief_value`: 0 (initialized to first laser observation in forward pass)
- `lr_value`: 3 → sigmoid(3) ≈ 0.95 (high initial alpha)
- `changepoint_value`: 0 → sigmoid(0) = 0.5
- `changepoint_lr_value`: 3
- `uncertainty_lr_value`: 3

High LR initialization ensures gradients flow from the start.

---

## Common Issues

**"LR modules not learning"**
- Check if LR states are changing during training (print initial vs final values)
- Ensure alpha is in output computation (gated output architecture)
- Verify LR initialization is high (lr_value=3)

---

## Contact

Questions? Check:
- Main SPICE docs: `/docs/`
- CLAUDE.md for architecture guidelines
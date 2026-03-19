# Guidelines for Polynomial-Amenable SPICE Architectures

## Motivation

When fitting SPICE models, the SPICE-RNN alone (`sindy_weight=0`) can match GRU-level performance, but full SPICE training (`sindy_weight > 0`) typically loses 2–3% accuracy. This gap arises because the SINDy regularization pushes the RNN dynamics into a polynomial subspace, and not all GRU-learned dynamics are polynomial-expressible.

This document provides guidelines for designing model architectures that minimize this gap by making the RNN dynamics naturally amenable to polynomial approximation.

## The Core Tension

The SINDy update rule is:

```
h_{t+1} = h_t + Σ_j c_j · φ_j(h_t, u_t)
```

where `φ_j` are polynomial basis functions of the current state `h_t` and control inputs `u_t`.

The GRU update rule is:

```
r = sigmoid(W_ir·x + W_hr·h)           ← reset gate
z = sigmoid(W_iz·x + W_hz·h)           ← update gate
n = tanh(W_in·x + r·(W_hn·h))          ← candidate
h_{t+1} = (1 - z)·n + z·h              ← convex combination
```

Even with `hidden_size=1`, the GRU has three capabilities that polynomials fundamentally struggle with:

1. **Sigmoid saturation** — gates become approximately binary (0 or 1), creating hard mode-switching. Polynomials need infinite degree to approximate step functions.
2. **Bounded outputs** — tanh/sigmoid confine values to [-1,1] / [0,1]. Polynomials diverge.
3. **Data-dependent convex combination** — `(1-z)·n + z·h` is a soft IF-THEN: when `z ≈ 1`, state is preserved; when `z ≈ 0`, state is replaced. This conditional behavior is the GRU's core power.

The architecture guidelines below aim to make these three capabilities *unnecessary* for the task at hand.

---

## Guidelines

### 1. Externalize Gating Through Action Masks and Separate Modules

The single biggest source of the accuracy gap: when the GRU must internally decide *whether* to update, it uses sigmoid gating — exactly what polynomials can't do. Move this decision into the architecture.

```python
# BAD — GRU must learn internal gating to ignore irrelevant trials
self.call_module('value', key_state='value', action_mask=None,
                 inputs=(reward, action, ...))

# GOOD — architecture handles the condition, GRU only learns the update magnitude
self.call_module('value_chosen', key_state='value',
                 action_mask=action, inputs=(reward,))
self.call_module('value_unchosen', key_state='value',
                 action_mask=1-action, inputs=(reward,))
```

**Rule of thumb:** If a module receives an input that acts as a binary selector (action flag, exit flag), it is probably better as two separate modules with action masks.

### 2. Fewer Inputs per Module — Target 1–3 Control Signals

The polynomial library size grows combinatorially, but the real problem isn't library size — it's that more inputs give the GRU more dimensions to create complex nonlinear interactions that polynomials can't match.

| Control inputs | + state | Deg 1 terms | Deg 2 terms | GRU effective params |
|----------------|---------|-------------|-------------|----------------------|
| 1              | 2       | 3           | 6           | ~15                  |
| 3              | 4       | 5           | 15          | ~30                  |
| 8              | 9       | 10          | 55          | ~60                  |

With 8 inputs, the GRU can learn to attend selectively to different inputs in different regimes (via sigmoid gates) — a soft attention mechanism that polynomials cannot replicate.

### 3. Precompute Non-Polynomial Transforms in the Forward Pass

If you need exponential decay, running averages, min/max, or clipping — compute them explicitly in `forward()` rather than asking the GRU to learn them.

```python
# These belong in forward(), NOT inside a GRU module:
dreward = reward - prev_reward                     # explicit differencing
average_reward = cum_reward / (time + 1e-8)        # explicit averaging
n_harvests = where(harvested, n + 1, 0)            # explicit counting
```

**Any transform that you can write as a closed-form expression should NOT go through a GRU module** — it should be a `self.state[...]` update in `forward()`.

### 4. Separate Memory States for Separate Cognitive Functions

One state variable tracking everything forces the GRU to learn a multiplexed encoding — different aspects of cognition packed into one scalar. Polynomials struggle with multiplexed signals because different regimes need different update rules (which requires gating).

```python
# BAD — one value must simultaneously track reward, depletion, and choice
memory_state = {'value': 0}

# GOOD — each state has simple, polynomial-friendly dynamics
memory_state = {
    'value_reward': 0,      # tracks reward accumulation
    'value_depletion': 0,   # tracks reward decline
    'value_tenure': 0,      # tracks time pressure
}
```

Each separate state can have a simpler update rule (e.g., `V_{t+1} = V_t + α·R` for reward, `D_{t+1} = D_t + β·ΔR` for depletion).

### 5. Match Polynomial Degree to the Mechanism

Some cognitive mechanisms are inherently multiplicative:

- **Rescorla-Wagner:** `V + α·(R - V) = (1-α)·V + α·R` — degree 1 suffices
- **Prediction error scaling:** `α · δ · V` — needs degree 2 (or 3 if α is learned)
- **Interaction effects:** `reward · state` — needs degree 2

If a module only needs additive updates, use `polynomial_degree=1` to reduce the search space. If it needs multiplicative interactions, use `polynomial_degree=2`. This can be set per-module via `setup_module(..., polynomial_degree=2)`.

### 6. Keep Input Values in a Range Where the GRU Stays Linear

When inputs are small (roughly [-1, 1]):
- sigmoid(x) ≈ 0.5 + 0.25·x (approximately linear)
- tanh(x) ≈ x (approximately linear)

In this regime, the GRU is approximately polynomial and the SINDy gap shrinks. When inputs are large, sigmoid saturates → hard gating → non-polynomial behavior.

```python
# Center and scale inputs:
spice_signals.rewards += 0.8980                     # center rewards around 0
n_harvests_scaled = n_harvests / max_harvests        # scale counts to [0, 1]
reward_normalized = reward / max_reward              # normalize to [-1, 1]
```

### 7. Remove Redundant Inputs the GRU Will Learn to Gate Out

If a module receives an input irrelevant to its function, the GRU learns sigmoid → 0 (ignore gate) for that input dimension. This sigmoid-based ignoring is exactly what polynomials can't do — a polynomial of an irrelevant input just adds noise terms.

**Test:** After fitting with `sindy_weight=0`, check input gradients. If `∂h/∂input_i ≈ 0` consistently, that input should be removed from the module.

### 8. Prefer Additive Logit Composition Over Internal Complexity

```python
# GOOD — simple modules, complexity through composition
logits = state['value_reward'] + state['value_depletion'] + state['value_tenure']

# BAD — one module must internally compute a complex mapping
logits = state['value']  # where value must encode reward + depletion + tenure
```

Additive composition in logit space means each module only needs to learn one aspect of the decision. The softmax in cross-entropy handles the nonlinear combination.

---

## Diagnostic Framework

These guidelines can be empirically validated from fitted models using the following diagnostics.

### Diagnostic 1: Polynomial Adequacy Test (R²)

**The single most informative diagnostic.** After fitting SPICE-RNN (`sindy_weight=0`), extract `(h_t, u_t) → h_{t+1}` trajectories per module and fit an ordinary polynomial regression post-hoc.

```python
def polynomial_adequacy_test(model, data, degree=2):
    """For each module, measure how well polynomials fit the learned dynamics."""
    # 1. Run forward pass, collect (h_t, inputs_t) and h_{t+1} at each call_module
    # 2. Build polynomial features from (h_t, inputs_t)
    # 3. Fit OLS: h_{t+1} - h_t ~ polynomial(h_t, inputs_t)
    # 4. Report R² per module
```

| R² value | Interpretation |
|----------|----------------|
| > 0.95   | Architecture is SINDy-friendly for this module |
| 0.90–0.95 | Marginal — consider minor restructuring |
| < 0.90   | Architecture needs restructuring for this module |
| < 0.80   | Fundamental expressiveness mismatch |

### Diagnostic 2: GRU Gate Saturation Analysis

Extract the gate values (`r`, `z`) from a fitted SPICE-RNN (`sindy_weight=0`) during inference. Gate values near 0 or 1 indicate hard switching that polynomials can't replicate.

```python
def analyze_gate_saturation(model, data):
    """Check if GRU gates are saturated (bimodal) or linear (unimodal)."""
    # Register hooks on each EnsembleGRUModule
    # Collect z (update gate) values across all timesteps
    # Plot histogram of gate values per module
```

| Gate distribution | Interpretation |
|-------------------|----------------|
| Unimodal, peak near 0.5 | Good — GRU is in linear regime, SINDy-friendly |
| Bimodal, peaks near 0 and 1 | Bad — GRU uses hard mode-switching, split this module |
| Unimodal, peak near 0 or 1 | The gate is effectively constant — module may be simplified |

### Diagnostic 3: Residual Pattern Analysis

After fitting full SPICE (with SINDy), compute `h_next_rnn - h_next_sindy` per trial and analyze when residuals are largest.

```python
def residual_analysis(model, data):
    """Identify when SINDy fails to approximate the RNN."""
    # 1. Forward pass with both RNN and SINDy predictions
    # 2. Compute residual per timestep per module
    # 3. Correlate residual magnitude with:
    #    - Input values (large inputs → sigmoid saturation?)
    #    - State values (boundary effects?)
    #    - Action types (conditional behavior?)
    #    - Trial position (early vs late in sequence?)
```

**If residuals correlate with a binary variable** (e.g., large after "exit" actions), that binary condition should be externalized as an action mask or separate module.

### Diagnostic 4: Input Sensitivity Analysis

```python
def input_sensitivity(model, data, module_name):
    """Gradient-based input importance for each module."""
    # For each call_module, compute |∂h_next/∂input_i| averaged over data
    # Inputs with near-zero gradients are candidates for removal
    # Inputs with highly variable gradients suggest regime-dependent usage
```

| Gradient pattern | Interpretation |
|------------------|----------------|
| Stable, moderate | Good — polynomial-like, input contributes consistently |
| Near-zero | Remove this input from the module |
| Bimodal / regime-dependent | GRU uses sigmoid gating to selectively attend — split module |

### Diagnostic 5: State Trajectory Comparison

Plot state trajectories from three models for the same participant/session:

1. **SPICE-RNN** (`sindy_weight=0`)
2. **SPICE-RNN** (`sindy_weight > 0`)
3. **SPICE** (SINDy equations only)

Where (1) and (2) diverge reveals what SINDy regularization is constraining. Where (2) and (3) diverge reveals the remaining polynomial approximation gap.

---

## Practical Workflow

1. **Design initial architecture** following guidelines 1–8
2. **Fit SPICE-RNN** with `sindy_weight=0`
3. **Run Diagnostic 1** (polynomial adequacy R²) per module
4. For modules with R² < 0.90:
   - Run Diagnostic 2 (gate saturation) — if bimodal, split the module
   - Run Diagnostic 4 (input sensitivity) — remove low-gradient inputs
   - Run Diagnostic 3 (residual analysis) — check if residuals correlate with binary conditions
5. **Restructure architecture** based on diagnostics
6. **Repeat** until all modules have R² > 0.95
7. **Fit full SPICE** with `sindy_weight > 0`

---

## Summary

| Guideline | Why it helps | Diagnostic to validate |
|-----------|-------------|------------------------|
| Externalize gating via action masks | Removes need for sigmoid gating | Gate saturation (bimodal → split) |
| 1–3 control signals per module | Less room for complex nonlinear interactions | Input sensitivity (remove dead inputs) |
| Precompute transforms in forward() | Keeps GRU dynamics simple | R² test (should increase) |
| Separate memory states | Avoids multiplexed encoding | R² per state (each should be high) |
| Match polynomial degree to mechanism | Avoids underfitting multiplicative terms | R² at deg 1 vs deg 2 |
| Keep inputs in [-1, 1] range | GRU stays in linear sigmoid/tanh regime | Gate saturation histogram |
| Remove irrelevant inputs | Prevents learned gating behavior | Input sensitivity (gradient ≈ 0) |
| Additive logit composition | Each module learns one simple aspect | Per-module R² |

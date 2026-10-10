# Guidelines for Polynomial-Amenable SPICE Architectures

## Motivation

SINDy regularization pushes each RNN submodule's dynamics into the space spanned by its polynomial candidate library. Dynamics the RNN can learn but a sparse polynomial cannot express cost predictive accuracy, interpretability, or both. This document collects guidelines for designing `BaseModel` subclasses whose submodule dynamics are naturally amenable to polynomial approximation, and whose discovered equations stay short and identifiable.

## The Core Tension

The SINDy update rule is:

```
h_{t+1} = h_t + Σ_j ξ_j · φ_j(h_t, u_t)
```

where `φ_j` are polynomial basis functions of the current state `h_t` and the control signals `u_t`.

A flexible recurrent network can learn things that polynomials fundamentally struggle with. A GRU, for example, updates its state as:

```
r = sigmoid(W_ir·x + W_hr·h)           ← reset gate
z = sigmoid(W_iz·x + W_hz·h)           ← update gate
n = tanh(W_in·x + r·(W_hn·h))          ← candidate
h_{t+1} = (1 - z)·n + z·h              ← convex combination
```

Even with `hidden_size=1`, three of its capabilities are hard for polynomials:

1. **Sigmoid saturation.** Gates become approximately binary (0 or 1), creating hard mode switching. Polynomials need infinite degree to approximate step functions.
2. **Bounded outputs.** tanh/sigmoid confine values to [-1, 1] or [0, 1]. Polynomials diverge.
3. **Data-dependent convex combination.** `(1-z)·n + z·h` is a soft IF-THEN: when `z ≈ 1` the state is preserved, when `z ≈ 0` it is replaced.

The SPICE submodules avoid explicit gates on the dynamics (a GELU residual update, see [training.md](training.md#basemodel-spiceresourcesmodelpy)). However, any sufficiently flexible network can learn the same kind of switching as a function of its **time-varying inputs** (control signals and state). The guidelines below make that switching unnecessary.

This concern does **not** apply to the individual-level gates of the SPICE submodules. These gates are computed from the participant embedding, a time-invariant attribute. They select which group-level features a participant uses, but they never switch behavior from trial to trial. Within one participant they are constants, and their effect shows up in the per-participant SINDy coefficients (which terms are present), which is exactly the structural individual difference SPICE is designed to reveal.

---

## Guidelines

### 1. Externalize Gating Through Action Masks and Separate Modules

The single biggest source of non-polynomial dynamics is a module that must decide internally *whether* to update. Move this decision into the architecture.

```python
# BAD — the module must learn internal gating to ignore irrelevant items
self.call_module('value', key_state='value', action_mask=None,
                 inputs=(reward, action))

# GOOD — the architecture handles the condition, the module only learns the update
self.call_module('value_chosen', key_state='value',
                 action_mask=action, inputs=(reward,))
self.call_module('value_not_chosen', key_state='value',
                 action_mask=1 - action)
```

**Rule of thumb:** if a module receives an input that acts as a binary selector (action flag, exit flag, catch flag), split it into separate modules with complementary action masks.

Splitting also keeps the final equations **short and comprehensible**. With chosen and unchosen mechanisms in separate modules, each equation describes one mechanism (e.g. learning from reward for the chosen option, forgetting for the unchosen one) instead of one long equation with selector interaction terms.

**Each split pair needs an anchor.** At least one module of a pair should receive an informative control signal (e.g. the reward for `value_reward_chosen`). If both modules receive no control signals (e.g. `value_choice_chosen` and `value_choice_not_chosen` with only their own state), both equations consist only of intercepts and self-terms. The intercepts of the two modules can then shift against each other (and against the initial value) without changing the predicted choices, which severely limits model identification. In that case, merge the pair into **one module that receives the selector as its input**:

```python
# Choice perseveration: no informative signal besides the choice itself -> one module, choice as anchor
self.call_module('value_choice', key_state='value_choice',
                 inputs=spice_signals.actions[trial])
```

With a binary selector `c` and a degree-2 library, the update `ξ_0 + ξ_1·h + ξ_2·c + ξ_3·h·c` is still exactly two affine branches (one per value of `c`), so nothing is lost relative to the split, and the choice itself anchors the equation.

### 2. Keep the Candidate Library Small

The number of candidate terms grows combinatorially with the number of control signals and the polynomial degree. A large library makes the sparse regression harder to identify and the resulting equations harder to read. More inputs also give the RNN more room for nonlinear interactions that polynomials cannot match.

| Control signals | + state | Degree-1 terms | Degree-2 terms |
|-----------------|---------|----------------|----------------|
| 1               | 2       | 3              | 6              |
| 3               | 4       | 5              | 15             |
| 8               | 9       | 10             | 55             |

As a suggestion, aim for 1–3 control signals per module. When a module genuinely needs more, keep the library small by other means:
- **Lower the polynomial degree** (e.g. `--polynomial_degree 1` for kolff2025, where each module receives 7 signals).
- **Prune structurally redundant terms** before training (`preprocess_coefficients` in the study models, via `sindy_coefficients_prior_mask`). Squares of binary indicators duplicate the linear term (`x^2 = x`). Products of two signals from a mutually exclusive group vanish (`x_i · x_j = 0`, e.g. one-hot act indicators, or `is_adjacent`/`is_opposite` in eckstein2026).

### 3. Precompute Non-Polynomial Transforms in the Forward Pass

If you need differencing, averaging, counting, circular distances or clipping, compute them explicitly in `forward()` rather than asking an RNN module to learn them.

```python
# These belong in forward(), NOT inside an RNN module:
prediction_error = outcome[trial] - self.state['belief_value']   # explicit differencing
average_reward = cum_reward / (time + 1e-8)                      # explicit averaging
n_harvests = torch.where(harvested, n_harvests + 1, 0)           # explicit counting
```

**Any transform you can write as a closed-form expression should NOT go through an RNN module.** Compute it in `forward()`, either as a local variable or as a `self.state[...]` buffer update.

The same holds for the output side. A fixed, closed-form read-out of learned states (e.g. a sigmoid-bounded learning rate that interpolates between the current position and a belief, as in bruckner2025 and weber2024) belongs in `forward()`, so that the modules only learn the simple state dynamics underneath.

### 4. Separate Memory States for Separate Cognitive Functions

One state variable tracking everything forces the RNN to learn a multiplexed encoding: different aspects of cognition packed into one scalar. Polynomials struggle with multiplexed signals because different regimes need different update rules (which requires gating).

```python
# BAD — one value must simultaneously track reward, recency and choice
memory_state = {'value': 0}

# GOOD — each state has simple, polynomial-friendly dynamics
memory_state = {
    'value_wm_reward': 0,   # fast, one-shot reward memory
    'value_reward': 0,      # slow, incremental reward learning
    'value_choice': 0,      # choice perseveration
}
```

Each separate state can then have a simple update rule.

### 5. Keep Input Values in [-1, 1]

When inputs are small, the RNN's nonlinearities operate in their near-linear regime and the learned dynamics stay close to polynomial. Large inputs push them into saturation, i.e. hard switching.

```python
# Normalize and center inputs in forward() (or in the data preparation):
dreward_tasks = spice_signals.additional_inputs['difference'] / MAX_VALUE_DIFFERENCE   # task values in [0, 10] -> difference in [-1, 1]
blocks = spice_signals.blocks / n_blocks                                               # block index -> [0, 1]
```

Rewards are normalized by `SpiceDataset.normalize_rewards()`. Additional inputs are not normalized automatically, so scale them yourself, using the task's theoretical range rather than the observed range so that simulated behavior stays in range too.

### 6. Remove Redundant Inputs

If a module receives an input irrelevant to its function, the RNN has to learn to ignore it, and the polynomial of an irrelevant input only adds noise terms.

**Test:** after fitting with `sindy_weight=0`, check input gradients. If `∂h/∂input_i ≈ 0` consistently, remove that input from the module.

Loading an additional input without feeding it to any module is harmless. It is simply never part of a library.

### 7. Prefer Additive Logit Composition Over Internal Complexity

```python
# GOOD — simple modules, complexity through composition
spice_signals.logits[trial] = self.state['value_wm_reward'] + self.state['value_reward'] + self.state['value_choice']

# BAD — one module must internally compute a complex mapping
spice_signals.logits[trial] = self.state['value']  # where value must encode everything
```

Additive composition in logit space means each module only needs to learn one aspect of the decision. The softmax in the cross-entropy loss handles the nonlinear combination.

### 8. Use Stateless Modules for Instantaneous Mappings

If a mechanism maps the current input to a value without memory (e.g. the effect of the latest reward in working memory, the sensitivity to an instantaneous task-value difference, the perceptual certainty for a contrast difference), register the module with `include_state=False`. The module then never sees its own state, its output replaces the state instead of incrementing it, and its library contains no self-terms.

### 9. Fix What Is Not Identified

Logits are invariant to a shift common to all options, so some quantities cannot be identified from choices alone. Fix them in the architecture instead of letting the fit drift:
- **Reference item.** If only some items need a value (e.g. the harvest item in bustamante2023, or the `none` item in kolff2025), leave the other item unwritten at 0 as the softmax reference.
- **Initial values.** Use fixed initial values (`0`) by default. Use learnable per-participant initial values (`None` in `memory_state`) only where an individual resting level is meaningful and not already absorbed by an intercept.
- **Module cleanup.** After the refit, `module_cleanup` removes terms without influence on the predictions (e.g. constants that shift all options equally). An architecture that produces many such terms is a sign that a reference point is missing.

---

## Diagnostic Framework

These guidelines can be validated empirically on fitted models with the following diagnostics.

### Diagnostic 1: Polynomial Adequacy Test (R²)

**The single most informative diagnostic.** After fitting SPICE-RNN (`sindy_weight=0`), extract the `(h_t, u_t) → h_{t+1}` trajectories per module and fit an ordinary polynomial regression post hoc.

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
| 0.90–0.95 | Marginal, consider minor restructuring |
| < 0.90   | Architecture needs restructuring for this module |
| < 0.80   | Fundamental expressiveness mismatch |

### Diagnostic 2: Residual Pattern Analysis

After fitting full SPICE (with SINDy), compute `h_next_rnn - h_next_sindy` per trial and analyze when the residuals are largest.

```python
def residual_analysis(model, data):
    """Identify when SINDy fails to approximate the RNN."""
    # 1. Forward pass with both RNN and SINDy predictions
    # 2. Compute residual per timestep per module
    # 3. Correlate residual magnitude with:
    #    - Input values (large inputs → saturation?)
    #    - State values (boundary effects?)
    #    - Action types (conditional behavior?)
    #    - Trial position (early vs late in sequence?)
```

**If residuals correlate with a binary variable** (e.g. large after "exit" actions), that binary condition should be externalized as an action mask or separate module (Guideline 1).

### Diagnostic 3: Input Sensitivity Analysis

```python
def input_sensitivity(model, data, module_name):
    """Gradient-based input importance for each module."""
    # For each call_module, compute |∂h_next/∂input_i| averaged over data
    # Inputs with near-zero gradients are candidates for removal
    # Inputs with highly variable gradients suggest regime-dependent usage
```

| Gradient pattern | Interpretation |
|------------------|----------------|
| Stable, moderate | Good: polynomial-like, the input contributes consistently |
| Near-zero | Remove this input from the module |
| Bimodal / regime-dependent | The module switches on this input: split the module |

### Diagnostic 4: State Trajectory Comparison

Plot state trajectories from three models for the same participant/session:

1. **SPICE-RNN** (`sindy_weight=0`)
2. **SPICE-RNN** (`sindy_weight > 0`)
3. **SPICE** (SINDy equations only)

Where (1) and (2) diverge reveals what the SINDy regularization constrains. Where (2) and (3) diverge reveals the remaining polynomial approximation gap.

---

## Practical Workflow

1. **Design the initial architecture** following Guidelines 1–9.
2. **Fit SPICE-RNN** with `sindy_weight=0`.
3. **Run Diagnostic 1** (polynomial adequacy R²) per module.
4. For modules with R² < 0.90:
   - Run Diagnostic 3 (input sensitivity): remove low-gradient inputs, split modules with regime-dependent inputs.
   - Run Diagnostic 2 (residual analysis): check whether residuals correlate with binary conditions.
5. **Restructure the architecture** based on the diagnostics.
6. **Repeat** until all modules have R² > 0.95.
7. **Fit full SPICE** with `sindy_weight > 0`.

---

## Summary

| Guideline | Why it helps | Diagnostic to validate |
|-----------|-------------|------------------------|
| 1. Externalize gating via action masks, with an anchor per split pair | Removes the need for learned switching, keeps equations short, avoids drifting intercepts | Residuals vs. binary conditions, input sensitivity (bimodal → split) |
| 2. Keep the candidate library small | Better identified sparse regression, readable equations | R² test, library size |
| 3. Precompute transforms and read-outs in `forward()` | Keeps module dynamics simple | R² test (should increase) |
| 4. Separate memory states | Avoids multiplexed encoding | R² per state (each should be high) |
| 5. Keep inputs in [-1, 1] | Nonlinearities stay in their near-linear regime | Residuals vs. input magnitude |
| 6. Remove redundant inputs | Prevents learned ignoring and noise terms | Input sensitivity (gradient ≈ 0) |
| 7. Additive logit composition | Each module learns one simple aspect | Per-module R² |
| 8. Stateless modules for instantaneous mappings | No spurious self-dynamics | — |
| 9. Fix what is not identified | Stable, comparable coefficients | Terms removed by module cleanup |

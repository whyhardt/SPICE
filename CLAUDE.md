# SPICE - Sparse and Interpretable Cognitive Equations

## Project Overview

SPICE is a framework for discovering symbolic cognitive mechanisms from behavioral data. It combines Recurrent Neural Networks (RNNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to extract interpretable mathematical equations that describe latent cognitive processes.

## Abstract

Discovering computational models that explain human cognition and behavior remains  a  central  goal  of  cognitive  science,  yet the  reliance  on  hand-crafted equations  limits  the  range  of  cognitive  mechanisms  that  can  be  uncovered. We introduce SPICE (Sparse and Interpretable Cognitive Equations), a framework that automates the discovery of mechanistically interpretable cognitive models directly from behavioral data. SPICE fits recurrent neural networks to capture latent cognitive dynamics and then applies sparse equation discovery to extract concise mathematical expressions describing those dynamics. Theory-guided priors make the approach data- and compute-efficient, while a hierarchical design reveals individual differences in the algorithmic structure of cognitive dynamics rather than in parameters alone. In simulations, SPICE accurately recovered the structure  and  parameters  of  known  reinforcement  learning  models. Applied to human behavior in a two-armed bandit task, it uncovered new equations that outperformed existing models and revealed structural alterations in reinforcement learning mechanisms among participants with depression, such as a loss of nonlinear exploration dynamics regulating behavioral flexibility. This approach provides systematic insights into structural individual differences in cognitive mechanisms and establishes a foundation for automated discovery of interpretable behavioral models.

### Core Methodology

1. **RNN Training**: A task-specific RNN learns to predict human behavior, implicitly capturing latent cognitive mechanisms in disentangled submodules
2. **SINDy Regularization**: During training, SINDy equations act as regularizers (similar to SINDy-SHRED), pushing submodule dynamics toward spaces amenable to SINDy candidate terms
3. **Equation Discovery**: SINDy approximates the fitted dynamics in each disentangled submodule, yielding interpretable symbolic equations

## Repository Structure

```
SPICE/
├── spice/                              # Core framework (backend / pip package)
│   ├── resources/
│   │   ├── estimator.py                # SpiceEstimator — scikit-learn compatible wrapper
│   │   ├── rnn.py                      # BaseRNN — core RNN + SINDy architecture
│   │   ├── spice_utils.py              # SpiceConfig, SpiceDataset, SpiceSignals
│   │   ├── spice_training.py           # Two-stage training pipeline
│   │   └── sindy_differentiable.py     # Differentiable SINDy polynomial library
│   ├── precoded/                       # Pre-built cognitive model architectures
│   │   ├── rescorlawagner.py           # Rescorla-Wagner learning model
│   │   ├── choice.py                   # Choice perseveration
│   │   ├── forgetting.py               # Forgetting mechanisms
│   │   ├── learningrate.py             # Dynamic learning rates
│   │   ├── interaction.py              # Interaction effects
│   │   ├── embedding.py                # Participant embeddings
│   │   ├── ddm.py                      # Drift Diffusion Model (within-trial dynamics)
│   │   ├── workingmemory.py            # Working memory with reward/choice buffers
│   │   └── workingmemory_*.py          # Working memory variants
│   └── utils/
│       ├── convert_dataset.py          # CSV ↔ SpiceDataset conversion pipeline
│       └── plotting.py                 # Visualization utilities
│
├── weinhardt2025/                      # Paper-specific code (fitting, benchmarking, analyses)
│   ├── run.py                          # Main entry point for training
│   ├── aux/                            # Jupyter notebooks per dataset
│   ├── data/                           # 16 benchmark datasets
│   ├── params/                         # Pre-trained model parameters
│   ├── benchmarking/                   # Baseline comparisons (Q-learning, GRU, etc.)
│   └── analysis/                       # Post-hoc analysis pipelines
│
├── docs/                               # Documentation and tutorials
├── pyproject.toml                      # Package config (autospice v0.2.0, Python >=3.11)
├── setup.py                            # Installation
└── requirements.txt                    # Core dependencies
```

## Tech Stack

- **Language**: Python 3.11+
- **ML Framework**: PyTorch (2.7+)
- **API Style**: Scikit-learn estimator interface
- **Package Name**: `autospice` (pip installable)

## Commands

```bash
pip install autospice                    # Install from PyPI
pip install -e .                         # Install locally in editable mode
python weinhardt2025/run.py              # Fit SPICE model to dataset
```

---

## Key Classes and Concepts

### BaseRNN (`spice/resources/rnn.py`)

Core neural network architecture. All task-specific models subclass this.

**Key components:**
- `submodules_rnn` — `ModuleDict` of GRU-based submodules, each learning one cognitive mechanism
- `sindy_coefficients` — Learnable `Dict[module_name, Tensor]` with shape `(E, P, X, n_terms)`
- `sindy_coefficients_presence` — Binary masks for active coefficients (same shape)
- `sindy_candidate_terms` / `sindy_degree_weights` — Library of basis functions and complexity penalties
- `sindy_cutoff_patience_counters` — Patience counters for threshold-based pruning
- `use_sindy` — Boolean toggle: `True` = equation mode, `False` = RNN mode

**Key methods:**

#### `setup_module()` — Register a GRU submodule with its SINDy configuration

```python
setup_module(
    key_module: str,          # Module name (must match a key in SpiceConfig.library_setup)
    input_size: int,          # Number of input features (control signals, excluding own state)
    dropout: float = 0.,      # Dropout rate in the GRU module
    polynomial_degree: int = None,  # SINDy library degree (None → use model default)
    include_bias: bool = True,      # Include constant term '1' in SINDy library
    include_state: bool = True,     # Include own state variable in SINDy library
    interaction_only: bool = False, # Only keep interaction terms (exclude pure polynomials like x^2)
)
```

Creates an `EnsembleGRUModule` and initializes SINDy coefficients + candidate library for this module. The `input_size` should account for the control signals plus any embeddings that will be concatenated at call time. The SINDy library is constructed from all inputs + the module's own state (if `include_state=True`), up to `polynomial_degree`.

#### `call_module()` — Forward pass through a submodule (RNN or SINDy path)

```python
call_module(
    key_module: str,                          # Module to execute (must exist in submodules_rnn)
    key_state: Optional[str] = None,          # Memory state key to update (in self.state)
    action_mask: torch.Tensor = None,         # Binary mask [W,E,B,I]: 1=update, 0=keep previous
    inputs: Union[Tensor, Tuple[Tensor]] = None,  # Control signals, each broadcastable to [W,E,B,I]
    participant_embedding: torch.Tensor = None,    # Learned embeddings [E,B,emb_dim]
    participant_index: torch.Tensor = None,        # Participant IDs [E,B] for SINDy coefficient indexing
    experiment_embedding: torch.Tensor = None,     # Experiment embeddings [E,B,emb_dim]
    experiment_index: torch.Tensor = None,         # Experiment IDs [E,B] for SINDy coefficient indexing
    activation_rnn: Callable = None,               # Optional activation on RNN output (e.g. torch.relu)
) -> torch.Tensor  # [W,E,B,I] updated state
```

Executes the module's forward pass. Broadcasts and concatenates inputs + embeddings to `[W,E,B,I,features]`, runs through GRU (or SINDy equation if `use_sindy=True`), applies `action_mask` (only masked items updated, unmasked retain previous state), clips to `[-10, 10]`, and updates `self.state[key_state]`. During training, also computes SINDy loss (MSE between RNN and SINDy predictions) accumulated in `model.sindy_loss`.

**Other methods:**
- `setup_embedding()` — Create participant/experiment embeddings
- `init_forward_pass()` — Promote input to canonical 5D shape, extract signals
- `post_forward_pass()` — Reshape outputs back to batch-first format
- `forward_sindy()` — Compute state update using sparse polynomial equations
- `compute_sindy_loss_for_module()` — MSE loss between RNN and SINDy predictions
- `sindy_ridge_solve()` — Direct least-squares coefficient solving
- `sindy_coefficient_pruning()` — Hard thresholding with patience

**Ensemble support:** `EnsembleLinear`, `EnsembleEmbedding`, `EnsembleGRUModule` — vectorized computation across ensemble members.

### SpiceConfig (`spice/resources/spice_utils.py`)

Architecture specification for a SPICE model. This is the central configuration that defines the cognitive architecture — which submodules exist, what they receive as input, and how they contribute to behavior.

```python
SpiceConfig(
    library_setup: Dict[str, Iterable[str]],
    memory_state: Union[List[str], Dict[str, float]],
    states_in_logit: List[str] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `library_setup` | `Dict[str, Iterable[str]]` | *required* | Maps each RNN submodule name to its input control signals (excluding self-references — the module's own state is added automatically). Module names must match keys used in `setup_module()`. Control signals are features from the input data (e.g., `'reward'`, `'choice'`). |
| `memory_state` | `Dict[str, float]` or `List[str]` | *required* | Defines the model's latent memory state variables and their initial values. If a list, all initial values default to `0.0`. Each key names a state variable that can be updated by `call_module()` via `key_state`. |
| `states_in_logit` | `List[str]` | `None` (= all states) | Subset of `memory_state` keys that feed into the output logits. If `None`, all memory states are used. Use this to exclude auxiliary states (e.g., working memory buffers) from the action-selection computation. |

**Derived attributes** (computed from inputs):
- `control_signals` — `tuple` of all unique input signal names across all modules
- `modules` — `tuple` of all module names (keys of `library_setup`)
- `all_features` — `tuple` of `modules + control_signals` (complete feature list)

**Example:**
```python
CONFIG = SpiceConfig(
    library_setup={
        'value_chosen':   ('reward',),       # learns value update for chosen action
        'value_unchosen': ('reward',),       # learns value update for unchosen action
        'choice':         (),                # learns choice perseveration (no control signals)
    },
    memory_state={
        'value': 0.5,     # initial action value
        'choice_value': 0.0,  # initial choice perseveration value
    },
    states_in_logit=['value', 'choice_value'],
)
```

### SpiceDataset (`spice/resources/spice_utils.py`)

PyTorch Dataset with auto-promotion (2D→3D→4D), NaN-based padding, and optional sequence splitting.

**Objective: Next-action prediction.** `xs[t]` contains the observation at trial `t` and `ys[t] = action[t+1]` (the next action, one-hot encoded). This is the standard cognitive modeling objective — predicting what the participant will do next given their history.

**Shape:** `(sessions, outer_ts, within_ts, features)`

**Feature columns in xs:** `[actions (one-hot, n_actions cols), rewards (one-hot, n_actions cols, optional), additional_inputs, time_trial, trials, block, experiment_id, participant_id]`

**Feature columns in ys:** `[next_action (one-hot, n_actions cols)]`

### SpiceEstimator (`spice/resources/estimator.py`)

Main user-facing class implementing sklearn's estimator interface.

**Methods:** `fit(data, targets, data_test, target_test)`, `predict(conditions)` → `(rnn_pred, spice_pred)`, `save_spice(path)`, `load_spice(path)`, `get_sindy_coefficients()`, `get_participant_embeddings()`, `print_spice_model()`, `count_sindy_coefficients()`, `get_modules()`, `get_candidate_terms()`

**Constructor arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Model specification** ||||
| `spice_class` | `BaseRNN` | *required* | RNN class (precoded or custom subclass of BaseRNN) |
| `spice_config` | `SpiceConfig` | *required* | Architecture configuration |
| `kwargs_rnn_class` | `dict` | `{}` | Extra keyword arguments forwarded to `spice_class.__init__()` |
| **Data/environment** ||||
| `n_actions` | `int` | `2` | Number of observable actions |
| `n_items` | `int` | `None` | Number of internal item representations (defaults to `n_actions`) |
| `n_participants` | `int` | `1` | Number of participants in the dataset |
| `n_experiments` | `int` | `1` | Number of experiments |
| `n_reward_features` | `int` | `None` | Number of reward feature columns (auto-detected if `None`) |
| **RNN training** ||||
| `epochs` | `int` | `1` | Total training epochs |
| `warmup_steps` | `int` | `0` | Epochs of exponential SINDy weight warmup (no pruning during warmup) |
| `learning_rate` | `float` | `1e-2` | Learning rate for RNN parameters |
| `batch_size` | `int` | `None` | Training batch size (`None` = auto-detect max via GPU probing) |
| `n_steps_per_call` | `int` | `None` | BPTT truncation length (`None` = full sequence) |
| `ensemble_size` | `int` | `1` | Number of independent RNN ensemble members |
| `dropout` | `float` | `0.` | Dropout rate in GRU modules |
| `l2_rnn` | `float` | `0` | L2 weight decay for RNN parameters |
| `convergence_threshold` | `float` | `0` | Early stopping threshold (0 = disabled) |
| `scheduler` | `bool` | `False` | Enable ReduceOnPlateauWithRestarts LR scheduler |
| `bagging` | `bool` | `False` | Whether to use bagging |
| `loss_fn` | `callable` | `cross_entropy_loss` | Behavioral loss function `(prediction, target) → scalar` |
| `device` | `torch.device` | `cpu` | Compute device |
| **SPICE / SINDy** ||||
| `use_sindy` | `bool` | `False` | Enable SINDy integration |
| `sindy_weight` | `float` | `0.1` | Lambda for SINDy regularization loss |
| `sindy_alpha` | `float` | `1e-4` | Degree-weighted L1 penalty strength |
| `sindy_library_polynomial_degree` | `int` | `1` | Max polynomial degree for SINDy candidate library |
| `sindy_pruning_frequency` | `int` | `1` | Epochs between pruning events |
| `sindy_threshold_pruning` | `float` | `None` | Minimum effect size delta for CI test (`None` = disabled) |
| `sindy_ensemble_pruning` | `float` | `None` | Confidence level for ensemble CI test (e.g. `0.05`; primary pruning mechanism) |
| `sindy_population_pruning` | `float` | `None` | Cross-participant presence threshold 0-1 (`None` = disabled) |
| `sindy_reconditioning_epochs` | `int` | `3` | Pure SINDy SGD epochs after ridge recalibration to warm-start the optimizer (`0` = disable) |
| **Output / misc** ||||
| `verbose` | `bool` | `False` | Print training progress |
| `keep_log` | `bool` | `False` | Keep full training log (vs. live terminal update) |
| `save_path_spice` | `str` | `None` | Auto-save .pkl path after training |
| `compiled_forward` | `bool` | `True` | Use `@torch.compile` for forward loops |

### Differentiable SINDy (`spice/resources/sindy_differentiable.py`)

PyTorch-based polynomial library computation for end-to-end gradient flow. Functions: `compute_library_size()`, `get_library_feature_names()`, `compute_polynomial_library()`, `get_polynomial_degree_from_term()`.

---

## Dimension Conventions

| Symbol | Meaning | Notes |
|--------|---------|-------|
| T | Outer timesteps (trials) | |
| W | Within-trial timesteps | Typically 1; >1 for DDM |
| E | Ensemble members | |
| B | Batch (sessions) | sessions = participants x experiments x blocks |
| F | Features | |
| I | Items | Internal value representations; defaults to `n_actions` but can differ (see below) |
| A | Actions | Observable action space (one-hot) |
| P | Participants | |
| X | Experiments | |
| C | Candidate terms | SINDy library size |

**Items vs. Actions:** Items and actions can be decoupled. `n_items` is the number of latent value representations the model maintains internally (state shape uses I). `n_actions` is the observable action space (logits shape uses A). By default `n_items = n_actions`, but they can differ — e.g., in a two-armed bandit with multiple symbol pairs, items might be contrast-specific values (low vs. high) while actions are position-specific (left vs. right). See `aux/ganesh2024a_choice_position.ipynb` for an example.

**Canonical internal shapes:**
- Input: `(T, W, E, B, F)` — after `init_forward_pass()` promotes from batch-first `(B, T, W, F)`
- State: `(W, E, B, I)`
- Logits: `(T, W, E, B, A)`
- SINDy coefficients: `(E, P, X, C)`

---

## Data Pipeline

### CSV → SpiceDataset (`spice/utils/convert_dataset.py`)

```
Raw CSV (participant, experiment, block, choice, [reward], [additional_inputs])
    ↓  csv_to_dataset()
    ↓  - Map categorical columns to numeric IDs
    ↓  - One-hot encode choices → n_actions columns
    ↓  - Promote rewards to action-aligned structure → n_actions columns (one per action)
    ↓  - Normalize rewards to [-1, 1] or [0, 1]
    ↓  - Build metadata columns (last 5: time_trial, trials, block, experiment_id, participant_id)
    ↓  - Shift: xs[t] = observation at trial t, ys[t] = action[t+1]
    ↓
SpiceDataset: shape (sessions, outer_ts, within_ts=1, features)
```

**Rewards are optional** (`df_feedback=None` to omit). When provided, rewards are promoted to one reward column per action:
- **Partial feedback** (single reward column): reward is placed in the column of the chosen action; unchosen columns are NaN
- **Full/counterfactual feedback** (multiple reward columns): each column maps directly to its corresponding action index

**Splitting utilities:** `split_data_along_timedim()`, `split_data_along_sessiondim()`, `reshape_data_along_participantdim()`

---

## Two-Stage Training Pipeline (`spice/resources/spice_training.py`)

Main function: `fit_spice()`

### Stage 1: Joint RNN-SINDy Training

The loss is composed in `_run_batch_training()`:
```python
loss = loss_fn(ys_pred, ys_step)                                          # behavioral loss
loss = loss + sindy_weight * model.sindy_loss                             # SINDy regularization
loss = loss + model.compute_weighted_coefficient_penalty(sindy_alpha, norm=1)  # degree-weighted L1 penalty
```

- `loss_fn(ys_pred, ys_step)` — behavioral prediction loss (cross-entropy by default, custom loss supported)
- `sindy_weight * model.sindy_loss` — MSE between RNN and SINDy predictions (computed inside `call_module` during forward pass). Only added when `sindy_weight > 0`.
- `compute_weighted_coefficient_penalty(sindy_alpha, norm=1)` — L1 penalty on SINDy coefficients, weighted by term degree (higher-degree terms penalized more). Only added when `sindy_alpha > 0`.

**Epoch flow:**
1. **Warmup**: For the first `n_warmup_steps` epochs, `sindy_weight` is exponentially scaled up from ~0 to its full value. No pruning during warmup.
2. **Batch training**: Random session batches → forward pass → loss → backprop. RNN and SINDy coefficients use separate optimizer param groups (SINDy lr=0.01 fixed, RNN lr configurable with optional scheduler).
3. **Validation**: If test data provided, evaluate both RNN-only and SINDy loss.
4. **Pruning** (every `sindy_pruning_frequency` epochs, after warmup):
   - **Primary mechanism** (`sindy_ensemble_pruning`): Minimum-effect CI test across ensemble members — a term survives iff `|mean| - t_crit * SE > delta` where `delta = sindy_threshold_pruning`.
   - **Fallback** (no ensemble test): Per-member hard thresholding with patience accumulation.
   - **Population filter** (`sindy_population_pruning`): Optional cross-participant consistency check.
   - Terms must fail **2 consecutive pruning events** before permanent removal (patience counter resets on success).
5. **LR scheduler**: `ReduceOnPlateauWithRestarts` — reduces RNN lr by 0.1× on plateau, restarts when hitting min_lr.
6. **Convergence check**: Exponentially smoothed loss change vs. threshold.

### Stage 2: Final SINDy Refit
- Freeze RNN weights
- Refit SINDy coefficients via SGD on vectorized hidden states

### Custom Loss Functions

A custom loss function can be passed via `loss_fn` to both `SpiceEstimator` and `fit_spice()`. It must accept `(prediction, target)` and return a scalar tensor.

**Default: `cross_entropy_loss`** (`spice/resources/spice_training.py`):
```python
def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n_actions = target.shape[-1]
    prediction = prediction.reshape(-1, n_actions)                    # flatten to (N, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)       # one-hot → class index
    return torch.nn.functional.cross_entropy(prediction, target)
```
- **Input `prediction`**: raw logits, any shape `(..., n_actions)` — reshaped to 2D internally
- **Input `target`**: one-hot encoded next-actions, same shape as prediction — converted to class indices via `argmax`
- **Output**: scalar cross-entropy loss

**NaN masking in `_run_batch_training()`:** Before the loss function is called, NaN-padded trials (from variable-length sessions) are masked out:
```python
mask = ~torch.isnan(xs_step[..., :model.n_actions].sum(dim=(-1)))
ys_pred = ys_pred[mask]
ys_step = ys_step[mask]
```
So the loss function only receives valid (non-padded) predictions and targets.

---

## Precoded Model Pattern

All task-specific models follow this pattern:

```python
class MyModel(BaseRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.participant_embedding = self.setup_embedding(...)
        self.setup_module(key_module='module_A', input_size=X, ...)
        self.setup_module(key_module='module_B', input_size=X, ...)
        self.setup_module(key_module='module_C', input_size=X, ...)

    def forward(self, inputs, prev_state=None, batch_first=False):
        spice_signals = self.init_forward_pass(...)
        embeddings = self.participant_embedding(spice_signals.participants_ids)

        for trial in spice_signals.trials:
            # this module could update the item values which were selected
            self.call_module(
                key_module='module_A',
                key_state='some_state_value',
                action_mask=spice_signals.actions[trial],
                inputs=(...),
                participant_index=spice_signals.participants_ids,
                participant_embedding=embeddings,
                # can also add experiment specific information to capture participant-specific information across different experiments
                experiment_index=None,
                experiment_embedding=None,
            )
            # this module could update the item values which were NOT selected
            self.call_module(
                key_module='module_B',
                key_state='some_state_value',
                action_mask=1-spice_signals.actions[trial],
                inputs=(...),
                participant_index=spice_signals.participants_ids,
                participant_embedding=embeddings,
            )
            # this module could update the another item value in the memory state (also applicable for both values at the same time)
            self.call_module(
                key_module='module_C',
                key_state='another_state_value',
                action_mask=None,
                inputs=(...),
                participant_index=spice_signals.participants_ids,
                participant_embedding=embeddings,
            )
            
            spice_signals.logits[timestep] = self.state['some_state_value'] + self.state['another_state_value']

        spice_signals = self.post_forward_pass(...)
        return spice_signals.logits, self.get_state()
```

**Available precoded models:** Rescorla-Wagner, Choice Perseveration, Forgetting, Learning Rate, Interaction, Embedding, DDM, Working Memory (+ variants).

---

## Coding Style

### Philosophy
- **Highly modular** over compact — prefer small, composable pieces
- **Slim over backward-compatible** — delete dead code rather than shimming it; no `_deprecated_*` wrappers or re-exports
- **General backend, specific frontends** — `BaseRNN` and training infrastructure stay task-agnostic; all task-specific logic lives in `class MyModel(BaseRNN)` subclasses

### Conventions
- snake_case for functions/variables, PascalCase for classes
- Descriptive variable names (e.g., `n_participants` over `n`)
- Follow existing PyTorch conventions (`.forward()`, `torch.nn.Module`, device handling)
- Maintain sklearn estimator API compatibility for `SpiceEstimator`
- Type hints where present in existing code
- `@torch.compile` for performance-critical forward loops

### File Organization
- Core framework code → `spice/`
- Research/paper code → `weinhardt2025/`
- New utility functions → `spice/utils/`
- New model architectures → `spice/precoded/`

### Designing Polynomial-Amenable Architectures

Full guidelines: [`docs/guidelines_polynomial_amenable_architectures.md`](docs/guidelines_polynomial_amenable_architectures.md)

When designing `BaseRNN` subclasses for SPICE, the architecture determines how well SINDy polynomials can approximate the learned GRU dynamics. The GRU's sigmoid gates, tanh activations, and data-dependent convex combinations are fundamentally non-polynomial — the goal is to make these capabilities *unnecessary* through architecture design:

1. **Externalize gating via action masks** — If a module receives a binary selector (action flag, exit flag), split it into separate modules with explicit `action_mask` arguments instead of relying on the GRU to learn internal gating.
2. **1–3 control signals per module** — More inputs give the GRU more dimensions for complex nonlinear interactions that polynomials can't match. Keep modules focused.
3. **Precompute non-polynomial transforms in `forward()`** — Differencing, running averages, counting, clipping — anything expressible in closed form belongs in `forward()`, not inside a GRU module.
4. **Separate memory states per cognitive function** — One state tracking multiple functions forces multiplexed encoding that requires GRU gating. Use separate `key_state` entries (e.g., `value_reward`, `value_depletion`, `value_tenure`).
5. **Match polynomial degree to mechanism** — Use `polynomial_degree=1` for additive updates (e.g., Rescorla-Wagner), `polynomial_degree=2` for multiplicative interactions (e.g., prediction error scaling).
6. **Keep inputs in [-1, 1]** — Small inputs keep sigmoid ≈ linear and tanh ≈ identity, making GRU dynamics approximately polynomial.
7. **Remove redundant inputs** — Irrelevant inputs force the GRU to learn sigmoid-based ignoring, which polynomials can't replicate. Validate with input gradient analysis after fitting with `sindy_weight=0`.
8. **Additive logit composition** — Combine simple module outputs (`logits = state['a'] + state['b'] + state['c']`) rather than packing complexity into one module.

---

## Common Pitfalls

- **`tensor.to(device)` returns a new tensor** — must reassign (`x = x.to(device)`)
- **Non-parameter tensors** need `register_buffer()` to auto-move with model
- **Hard masking logits with large negative values** can catastrophically inflate CE loss even when only a few targets are masked
- **Overlapping action masks** in sequential `call_module` calls: second call overwrites first for shared items
- **Shape mismatches**: Ensure `xs` has shape `(batch, timesteps, features)` and `ys` has shape `(batch, timesteps, actions)`
- **One-hot encoding**: Actions and rewards must be one-hot encoded, not integer indices
- **SINDy convergence**: If coefficients don't converge, adjust `sindy_weight` or increase epochs
- **Patience tuning**: Too low → premature elimination; too high → delayed sparsification
- **Degree weights**: Overly aggressive penalties may suppress necessary nonlinear terms

## Documentation

Full documentation: https://whyhardt.github.io/SPICE/

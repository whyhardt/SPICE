# SPICE - Sparse and Interpretable Cognitive Equations

## Project Overview

SPICE is a framework for discovering symbolic cognitive mechanisms from behavioral data. It combines Recurrent Neural Networks (RNNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to extract interpretable mathematical equations that describe latent cognitive processes.

## Abstract

Discovering computational models that explain human cognition and behavior remains  a  central  goal  of  cognitive  science,  yet the  reliance  on  hand-crafted equations  limits  the  range  of  cognitive  mechanisms  that  can  be  uncovered. We introduce SPICE (Sparse and Interpretable Cognitive Equations), a framework that automates the discovery of mechanistically interpretable cognitive models directly from behavioral data. SPICE fits recurrent neural networks to capture latent cognitive dynamics and then applies sparse equation discovery to extract concise mathematical expressions describing those dynamics. Theory-guided priors make the approach data- and compute-efficient, while a hierarchical design reveals individual differences in the algorithmic structure of cognitive dynamics rather than in parameters alone. In simulations, SPICE accurately recovered the structure  and  parameters  of  known  reinforcement  learning  models. Applied to human behavior in a two-armed bandit task, it uncovered new equations that outperformed existing models and revealed structural alterations in reinforcement learning mechanisms among participants with depression, such as a loss of nonlinear exploration dynamics regulating behavioral flexibility. This approach provides systematic insights into structural individual differences in cognitive mechanisms and establishes a foundation for automated discovery of interpretable behavioral models.

### Core Methodology

1. **Polynomial RNN Training**: A task-specific polynomial RNN learns to predict human behavior, with each submodule implementing an exact polynomial update whose coefficients can be analytically extracted
2. **Sparse Pruning**: During training, minimum-effect CI tests prune insignificant polynomial terms, yielding sparse interpretable equations
3. **Equation Discovery**: The unfolded polynomial coefficients directly give the discovered equations — no separate SINDy fitting step needed

## Repository Structure

```
SPICE/
├── spice/                              # Core framework (backend / pip package)
│   ├── resources/
│   │   ├── estimator.py                # SpiceEstimator — scikit-learn compatible wrapper
│   │   ├── model.py                    # BaseModel — core RNN + SINDy architecture
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
├── weinhardt2026/                      # Paper-specific code (fitting, benchmarking, analyses)
│   ├── run.py                          # Main entry point for training
│   ├── studies/                        # Self-contained study directories
│   │   ├── synthetic/                  # Synthetic parameter recovery
│   │   ├── braun2018/                  # Each study has: notebook, benchmarking script,
│   │   ├── bustamante2023/             #   data/, params/, results/
│   │   ├── castro2025/
│   │   ├── dezfouli2019/
│   │   ├── ganesh2024a/
│   │   ├── huang2026/
│   │   ├── hwang2026/
│   │   └── archive/                    # Inactive studies
│   ├── analysis/                       # Cross-study analysis pipelines
│   └── utils/                          # Shared utilities (benchmarking, bandits, etc.)
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
python weinhardt2026/run.py              # Fit SPICE model to dataset
```

---

## Key Classes and Concepts

### BaseModel (`spice/resources/model.py`)

Core neural network architecture. All task-specific models subclass this.

**Key components:**
- `submodules_rnn` — `ModuleDict` of polynomial RNN submodules, each learning one cognitive mechanism
- `sindy_coefficients_presence` — Binary masks for active polynomial terms, shape `(E, P, X, n_terms)`
- `sindy_candidate_terms` — Library term names per module
- `sindy_pruning_patience_counters` — Patience counters for threshold-based pruning

**Key methods:**

#### `setup_module()` — Register an RNN submodule with its SINDy configuration

```python
setup_module(
    key_module: str,          # Module name (must match a key in SpiceConfig.library_setup)
    input_size: int,          # Number of RNN input features (control signals + embeddings; excludes own state)
    dropout: float = 0.,      # Dropout rate in the RNN module
    polynomial_degree: int = None,  # SINDy library degree (None → use model default)
    include_bias: bool = True,      # Include constant term '1' in SINDy library
    include_state: bool = True,     # Include own state variable in SINDy library
    interaction_only: bool = False, # Only keep interaction terms (exclude pure polynomials like x^2)
)
```

Creates an `EnsembleRNNModule` and initializes SINDy coefficients + candidate library for this module. The `input_size` should account for the control signals plus any embeddings that will be concatenated at call time. **Important:** `input_size` defines the RNN's input dimension, NOT the SINDy library features. The SINDy library is constructed only from the control signals (defined in `SpiceConfig.library_setup`) + the module's own state (if `include_state=True`), up to `polynomial_degree`. Embeddings are NOT included in the SINDy library — they feed the RNN only, while participant variation in SINDy is handled via per-participant coefficients.

#### `call_module()` — Forward pass through a submodule (polynomial path)

```python
call_module(
    key_module: str,                          # Module to execute (must exist in submodules_rnn)
    key_state: Optional[str] = None,          # Memory state key to update (in self.state)
    action_mask: torch.Tensor = None,         # Binary mask [W,E,B,I]: 1=update, 0=keep previous
    inputs: Union[Tensor, Tuple[Tensor]] = None,  # Control signals, each broadcastable to [W,E,B,I]
    participant_embedding: torch.Tensor = None,    # Learned embeddings [E,B,emb_dim]
    participant_index: torch.Tensor = None,        # Participant IDs [E,B] for coefficient indexing
    experiment_embedding: torch.Tensor = None,     # Experiment embeddings [E,B,emb_dim]
    experiment_index: torch.Tensor = None,         # Experiment IDs [E,B] for coefficient indexing
) -> torch.Tensor  # [W,E,B,I] updated state
```

Executes the module's polynomial forward pass. Broadcasts and concatenates inputs + embeddings to `[W,E,B,I,features]`, unfolds polynomial coefficients from RNN weights, applies sparsity mask from `sindy_coefficients_presence`, computes gated polynomial update, applies `action_mask` (only masked items updated), clips to `[-10, 10]`, and updates `self.state[key_state]`.

**Other methods:**
- `setup_embedding()` — Create participant/experiment embeddings
- `init_forward_pass()` — Promote input to canonical 5D shape, extract signals
- `post_forward_pass()` — Reshape outputs back to batch-first format
- `get_sindy_coefficients()` — Return effective polynomial coefficients (gate absorbed)
- `sindy_coefficient_pruning()` — Hard thresholding with patience
- `get_spice_model_string()` — Print discovered equations

**Ensemble support:** `EnsembleLinear`, `EnsembleEmbedding`, `EnsembleRNNModule` — vectorized computation across ensemble members.

**`EnsembleRNNModule` architecture:** Uses a gated polynomial update:
1. Concatenate input with current state: `x_t = concat(h, x[t])`
2. Multilinear polynomial projection: `gi = Π_d(W_d @ x_t + b_d) / √D` (product of D linear forms = exact degree-D polynomial)
3. Candidate computation: `n = W_out @ gi + b_out`
4. Gated update: `h = (1-α)*h + α_n*n` where `α = sigmoid(damping_coefficient)`, `α_n = α` by default

The `damping_coefficient` is a scalar `nn.Parameter` shared across all ensemble members.
`scale_candidate = True` (default): `α_n = α`; set to `False` for `α_n = 1` (unscaled candidate).

**Effective coefficients:** `unfold_polynomial_coefficients()` analytically expands the multilinear product into monomial-basis coefficients. All reporting uses effective coefficients with the gate absorbed: `c_eff = α_n * θ` for non-state terms, `c_eff = (1-α) + α_n * θ` for the state self-term.

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
| `spice_class` | `BaseModel` | *required* | RNN class (precoded or custom subclass of BaseModel) |
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
| `dropout` | `float` | `0.` | Dropout rate in RNN modules |
| `l2_rnn` | `float` | `0` | L2 weight decay for RNN parameters |
| `convergence_threshold` | `float` | `0` | Early stopping threshold (0 = disabled) |
| `scheduler` | `bool` | `False` | Enable ReduceOnPlateauWithRestarts LR scheduler |
| `bagging` | `bool` | `False` | Whether to use bagging |
| `loss_fn` | `callable` | `cross_entropy_loss` | Behavioral loss function `(prediction, target) → scalar` |
| `device` | `torch.device` | `cpu` | Compute device |
| **Polynomial pruning** ||||
| `sindy_library_polynomial_degree` | `int` | `1` | Max polynomial degree for candidate library |
| `sindy_pruning_frequency` | `int` | `1` | Epochs between pruning events |
| `sindy_threshold_pruning` | `float` | `None` | Minimum effect size delta for CI test (`None` = disabled) |
| `sindy_ensemble_pruning` | `float` | `None` | Confidence level for ensemble CI test (e.g. `0.05`; primary pruning mechanism) |
| `sindy_population_pruning` | `float` | `None` | Cross-participant presence threshold 0-1 (`None` = disabled) |
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

**Items vs. Actions:** Items and actions can be decoupled. `n_items` is the number of latent value representations the model maintains internally (state shape uses I). `n_actions` is the observable action space (logits shape uses A). By default `n_items = n_actions`, but they can differ — e.g., in a two-armed bandit with multiple symbol pairs, items might be contrast-specific values (low vs. high) while actions are position-specific (left vs. right). See `weinhardt2026/studies/ganesh2024a/ganesh2024a.ipynb` for an example.

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

## Training Pipeline (`spice/resources/spice_training.py`)

Main function: `fit_spice()`

### Single-Stage Polynomial RNN Training

The loss in `_run_batch_training()` is purely behavioral:
```python
loss = loss_fn(ys_pred, ys_step)  # behavioral loss only
```

Regularization is handled by AdamW's L2 weight decay, which implicitly penalizes higher-degree polynomial terms more.

**Epoch flow:**
1. **Warmup**: No pruning during the first `n_warmup_steps` epochs.
2. **Batch training**: Random session batches → polynomial forward pass → behavioral loss → backprop with gradient clipping (max_norm=1.0).
3. **Validation**: Single forward pass on test data.
4. **Pruning** (every `sindy_pruning_frequency` epochs, after warmup):
   - Polynomial coefficients are unfolded from RNN weights and scaled by α_n (effective coefficients).
   - **Primary mechanism** (`sindy_ensemble_pruning`): Minimum-effect CI test across ensemble members — a term survives iff `|mean| - t_crit * SE > delta` where `delta = sindy_threshold_pruning`.
   - **Fallback** (no ensemble test): Per-member hard thresholding with patience accumulation.
   - **Population filter** (`sindy_population_pruning`): Optional cross-participant consistency check.
   - Terms must fail **2 consecutive pruning events** before permanent removal (patience counter resets on success).
5. **LR scheduler**: `ReduceOnPlateauWithRestarts` — reduces RNN lr by 0.1× on plateau, restarts when hitting min_lr.
6. **Convergence check**: Exponentially smoothed loss change vs. threshold.

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
class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.participant_embedding = self.setup_embedding(...)
        self.setup_module(key_module='module_A', input_size=X, ...)
        self.setup_module(key_module='module_B', input_size=X, ...)
        self.setup_module(key_module='module_C', input_size=X, ...)

    def forward(self, inputs, prev_state=None):
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
- **General backend, specific frontends** — `BaseModel` and training infrastructure stay task-agnostic; all task-specific logic lives in `class MyModel(BaseModel)` subclasses

### Conventions
- snake_case for functions/variables, PascalCase for classes
- Descriptive variable names (e.g., `n_participants` over `n`)
- Follow existing PyTorch conventions (`.forward()`, `torch.nn.Module`, device handling)
- Maintain sklearn estimator API compatibility for `SpiceEstimator`
- Type hints where present in existing code
- `@torch.compile` for performance-critical forward loops

### File Organization
- Core framework code → `spice/`
- Research/paper code → `weinhardt2026/`
- New utility functions → `spice/utils/`
- New model architectures → `spice/precoded/`

### Designing Polynomial-Amenable Architectures

Full guidelines: [`docs/guidelines_polynomial_amenable_architectures.md`](docs/guidelines_polynomial_amenable_architectures.md)

When designing `BaseModel` subclasses for SPICE, the architecture determines how well SINDy polynomials can approximate the learned RNN dynamics. The RNN submodules use a residual architecture (GELU projection + additive update) that is inherently more polynomial-amenable than traditional GRUs, but good architecture design still matters:

1. **Externalize gating via action masks** — If a module receives a binary selector (action flag, exit flag), split it into separate modules with explicit `action_mask` arguments instead of relying on the RNN to learn internal gating.
2. **1–3 control signals per module** — More inputs give the RNN more dimensions for complex nonlinear interactions that polynomials can't match. Keep modules focused.
3. **Precompute non-polynomial transforms in `forward()`** — Differencing, running averages, counting, clipping — anything expressible in closed form belongs in `forward()`, not inside an RNN module.
4. **Separate memory states per cognitive function** — One state tracking multiple functions forces multiplexed encoding. Use separate `key_state` entries (e.g., `value_reward`, `value_depletion`, `value_tenure`).
5. **Match polynomial degree to mechanism** — Use `polynomial_degree=1` for additive updates (e.g., Rescorla-Wagner), `polynomial_degree=2` for multiplicative interactions (e.g., prediction error scaling).
6. **Keep inputs in [-1, 1]** — Small inputs keep the RNN dynamics in a regime amenable to polynomial approximation.
7. **Remove redundant inputs** — Irrelevant inputs force the RNN to learn to ignore them, wasting capacity. Validate with input gradient analysis after fitting with `sindy_weight=0`.
8. **Additive logit composition** — Combine simple module outputs (`logits = state['a'] + state['b'] + state['c']`) rather than packing complexity into one module.

---

## Common Pitfalls

- **`tensor.to(device)` returns a new tensor** — must reassign (`x = x.to(device)`)
- **Non-parameter tensors** need `register_buffer()` to auto-move with model
- **Hard masking logits with large negative values** can catastrophically inflate CE loss even when only a few targets are masked
- **Overlapping action masks** in sequential `call_module` calls: second call overwrites first for shared items
- **Shape mismatches**: Ensure `xs` has shape `(batch, timesteps, features)` and `ys` has shape `(batch, timesteps, actions)`
- **One-hot encoding**: Actions and rewards must be one-hot encoded, not integer indices
- **Patience tuning**: Too low → premature elimination; too high → delayed sparsification

## Generative Benchmarking (`weinhardt2026/utils/task.py`)

Generative benchmarking simulates new behavioral data by running a fitted model through the original task environment. This produces synthetic datasets that can be compared against the original human data.

### Architecture

```
task.py                          # Shared infrastructure
├── Env (base class)             # Abstract task environment
└── generate_behavior()          # Batched trial-by-trial generation loop

studies/<study>/benchmarking_<study>.py  # Per-study file
├── get_dataset()                # Load & split data
├── BenchmarkModel (nn.Module)   # Hand-coded cognitive model (e.g. GQLModel)
├── Environment<Study>(Env)      # Study-specific reward mechanics
└── generate_behavior()          # Thin wrapper → calls shared _generate_behavior
```

### Env Base Class

All task environments subclass `Env` and implement batched `reset()` + `step()`:

```python
class Env:
    def __init__(self, n_actions: int, n_participants: int, n_blocks: int):
        ...

    @property
    def n_sessions(self) -> int:
        return self.n_participants * self.n_blocks

    def reset(self, block_ids: torch.Tensor, participant_ids: torch.Tensor = None) -> None:
        """Set up per-session environment state from dataset metadata."""
        ...

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One trial for all sessions in parallel.
        Args:    action: (n_sessions,) integer action indices.
        Returns: (reward, terminated) — both (n_sessions,) tensors.
        """
        ...
```

### Shared `generate_behavior()` Flow

1. Extract `block_ids` and `participant_ids` from dataset metadata (`xs[:, 0, 0, -3]` and `xs[:, 0, 0, -1]`)
2. Call `environment.reset(block_ids, participant_ids)`
3. Resolve model: `SpiceEstimator` → unwrap `.model`; raw `nn.Module` → use directly
4. For each trial `t` in `range(n_trials)`:
   - `reward, _ = environment.step(action_idx)` — environment gives reward for current action
   - Build observation: one-hot action + partial-feedback reward (NaN for unchosen) + metadata from original dataset
   - Forward pass: `rnn(obs, state)` for `BaseModel`, `rnn(obs, state)` otherwise
   - Normalize logits: 5D `(E,B,T,W,A)` → mean over ensemble → 4D → extract `(B, A)`
   - Sample next action: `multinomial(softmax(logits))`
5. Restore NaN padding for variable-length sessions (matching original dataset structure)

### Study-Specific Environment Pattern

```python
class EnvironmentMyStudy(Env):
    REWARD_PROBS = torch.tensor([...])  # Task-specific reward structure

    def __init__(self, n_actions, n_participants, n_blocks):
        super().__init__(n_actions, n_participants, n_blocks)

    def reset(self, block_ids, participant_ids=None):
        # Map block IDs to per-session reward parameters
        self.session_reward_probs = self.REWARD_PROBS[block_ids]

    def step(self, action):
        # Sample rewards based on task mechanics
        probs = self.session_reward_probs[torch.arange(len(action)), action]
        reward = torch.bernoulli(probs)
        return reward, torch.zeros(len(action), dtype=torch.bool)
```

### Study-Specific `generate_behavior` Wrapper

```python
def generate_behavior(model, path_data=None, dataset=None, save_dataset=None):
    if dataset is None:
        dataset, _, _ = get_dataset(path_data=path_data)
    environment = EnvironmentMyStudy(
        n_actions=dataset.n_actions,
        n_participants=dataset.n_participants,
        n_blocks=N_BLOCKS,
    )
    return _generate_behavior(dataset=dataset, model=model, environment=environment, save_dataset=save_dataset)
```

### Dataset Metadata Conventions

Block and participant IDs in dataset metadata (`xs[..., -3]` and `xs[..., -1]`):
- **Participant IDs**: 0-indexed (remapped by `csv_to_dataset`)
- **Block IDs**: Kept as-is from the CSV (typically 1-indexed). Environment `reset()` receives these raw values — index reward tables accordingly.

---

## Documentation

Full documentation: https://whyhardt.github.io/SPICE/

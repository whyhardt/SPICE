# SPICE Training Mechanisms

Internals of `BaseModel`, `SpiceConfig`, `SpiceDataset`, `SpiceEstimator`, and the training pipeline, verified against the current code in `spice/resources/`. See [CLAUDE.md](../CLAUDE.md) for the project overview and where this fits in the repo.

---

## BaseModel (`spice/resources/model.py`)

Core neural network architecture. All task-specific models subclass this.

```python
BaseModel(
    spice_config: SpiceConfig,
    n_actions,
    n_participants: int = 1,
    n_experiments: int = 1,
    n_items: int = None,
    n_reward_features: int = None,
    ensemble_size: int = 1,
    embedding_size: int = 32,
    dropout: float = 0.,
    use_sindy: bool = False,
    sindy_polynomial_degree: int = 1,
    sindy_alpha: float = 1e-4,
    fit_sindy: bool = True,
    device=torch.device('cpu'),
    compiled_forward=True,
    batch_first: bool = True,
    **kwargs,
)
```

**Key components:**
- `submodules_rnn` — `ModuleDict` of RNN submodules (residual architecture), each learning one cognitive mechanism
- `submodules_eq` — registry for hard-coded (non-learned) equation modules, an alternative to `submodules_rnn` for mechanisms you want to specify by hand instead of fit
- `sindy_coefficients` — Learnable `Dict[module_name, Tensor]` with shape `(E, P, X, n_terms)`
- `sindy_coefficients_presence` — Binary masks for active coefficients (same shape); flipped by pruning
- `sindy_coefficients_prior_mask` — Binary masks for theory-driven exclusions (e.g. forcing `binary^2 = 0` for one-hot features). **Never** reset by pruning, unlike `sindy_coefficients_presence`
- `sindy_pruning_patience_counters` — Per-`(E,P,X,term)` counters; a term must accumulate 2 consecutive failed pruning checks before removal
- `sindy_candidate_terms` / `sindy_degree_weights` — Library of basis functions and complexity penalties (degree weighting is currently disabled — see Known Inert Code below)
- `use_sindy` — Boolean toggle: `True` = equation mode, `False` = RNN mode
- `fit_sindy` — Independent of `use_sindy`: controls whether `call_module` computes the SINDy fit/regularization losses during training at all
- `ridge_mode` — Internal flag toggled by the training pipeline during Stage 2 ridge solves; when `True`, `call_module` bypasses normal RNN/SINDy branching and solves SINDy coefficients directly against the RNN's own predictions
- `sindy_loss_reg` / `sindy_loss_fit` — Two **decoupled** scalar loss accumulators computed per forward pass. `sindy_loss_reg` has gradients flowing only to RNN params (SINDy side detached) — this is what `sindy_weight` scales to regularize the RNN toward SINDy-discoverable dynamics. `sindy_loss_fit` has gradients flowing only to SINDy coefficients (RNN side detached) and is fit independently of `sindy_weight`.
- `sindy_norm` — `1` (L1, default) or `2` (L2) penalty norm used by `compute_weighted_coefficient_penalty`
- `learnable_initial_values` — For any `memory_state` entry whose initial value is `None` in `SpiceConfig`, a per-`(ensemble, participant)` learnable scalar parameter instead of a fixed constant; applied in `init_forward_pass` only when no `prev_state` is passed
- `embedding_fusion` — How multiple registered embeddings (participant + experiment, etc.) are combined; `torch.cat` by default, or a learned `EnsembleEmbeddingFusion` layer once more than one embedding is registered

### `setup_module()` — Register an RNN submodule with its SINDy configuration

```python
setup_module(
    key_module: str,          # Module name (must match a key in SpiceConfig.library_setup)
    input_size: int = None,   # Number of RNN input features (control signals + embeddings; excludes own state)
    embedding_size: int = None,     # Per-module override of the model-level embedding_size
    dropout: float = None,          # Dropout rate in the RNN module
    polynomial_degree: int = None,  # SINDy library degree (None → use model default)
    include_bias: bool = True,      # Include constant term '1' in SINDy library
    include_state: bool = True,     # Include own state variable in SINDy library AND feed it back as an RNN input
    interaction_only: bool = False, # Only keep interaction terms (exclude pure polynomials like x^2)
    dt: float = 1.,                  # Physical time step this module's update represents
    within_trial_timesteps: bool = False,  # Whether dynamics evolve over W (within-trial) rather than T (across-trial)
)
```

Creates an `EnsembleRNNModule` and initializes SINDy coefficients + candidate library for this module. The `input_size` should account for the control signals plus any embeddings that will be concatenated at call time. **Important:** `input_size` defines the RNN's input dimension, NOT the SINDy library features. The SINDy library is constructed only from the control signals (defined in `SpiceConfig.library_setup`) + the module's own state (if `include_state=True`), up to `polynomial_degree`. Embeddings are NOT included in the SINDy library — they feed the RNN only, while participant variation in SINDy is handled via per-participant coefficients.

- `include_state=False` makes the module stateless: its own previous state is zeroed every step (not just at t=0), so it's purely a function of external inputs.
- `dt` scales both the RNN residual update (`h = h + dt * n`) and the SINDy update/ridge-solve target (`(h_next - h_current) / dt`), so discovered coefficients are per-unit-time rates rather than per-step deltas. Use `dt < 1` for modules meant to represent continuous/physical time.
- `within_trial_timesteps` tells Stage 2's SINDy refit which axis (`W` or `T`) this module's dynamics roll out along when "shooting" multi-step windows — set `True` for evidence-accumulation/DDM-style modules that evolve within a trial, `False` for the common case of one update per trial.

### `call_module()` — Forward pass through a submodule (RNN or SINDy path)

```python
call_module(
    key_module: str,                          # Module to execute (in submodules_rnn or submodules_eq)
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

Broadcasts and concatenates inputs + fused embeddings to `[W,E,B,I,features]`. For `submodules_rnn` entries: runs the RNN branch when `not use_sindy or ridge_mode` (with the ridge-mode variant additionally solving SINDy coefficients directly against the RNN's prediction), and the SINDy branch (looping per within-trial timestep, calling `forward_sindy`) when `use_sindy`. For `submodules_eq` entries, calls the hard-coded equation on the last within-trial step only. Computes `sindy_loss_reg`/`sindy_loss_fit` whenever `fit_sindy and training and not use_sindy` (i.e. normal RNN training with SINDy fitting enabled). Clips output to `[-10, 10]`, applies `action_mask` (only masked items updated, unmasked retain previous state), and writes back into `self.state[key_state]` if provided.

**Other methods:**
- `setup_embedding()` — Create participant/experiment embeddings (and, once more than one is registered, upgrade `embedding_fusion` to a learned layer)
- `init_forward_pass()` / `post_forward_pass()` — Canonicalize input to `(T,W,E,B,F)`, extract control signals/actions/metadata into a `SpiceSignals` container; reshape logits back to batch-first on the way out
- `init_state()` / `set_state()` / `get_state(detach=False)` — Hidden-state management, shape `(within_ts, ensemble, batch, n_items)` per state key
- `forward_sindy()` — Compute state update using sparse polynomial equations
- `compute_sindy_loss_for_module()` — Computes `sindy_loss_reg`/`sindy_loss_fit` for a module
- `sindy_ridge_solve()` — Accumulates per-`(participant, experiment)` normal equations via scatter-add and solves in closed form (float64, ridge-penalized); used by Stage 2 refit and `ridge_mode`
- `sindy_coefficient_pruning(patience=1, n_terms_pruning=None)` / `sindy_coefficient_patience(threshold)` — Patience-gated hard thresholding fallback (used when ensemble pruning is unavailable)
- `compute_weighted_coefficient_penalty(sindy_alpha)` — Degree-weighted L1/L2 penalty on coefficients
- `print(participant_id=0, experiment_id=0)` / `get_spice_model_string(...)` — Human-readable equations per module, e.g. `"value[t+1] = 0.92 value[t] + 0.14 reward[t]"`
- `get_modules()` / `get_candidate_terms(key_module=None)` / `get_sindy_coefficients(key_module=None, aggregate=False)` / `count_sindy_coefficients()` — Introspection (see [SpiceEstimator](#spiceestimator-spiceresourcesestimatorpy))
- `eval(use_sindy=True)` / `train(mode=True, use_sindy=False)` — Override `nn.Module`'s eval/train to also set `self.use_sindy` (eval defaults to SINDy-mode-on, train defaults to SINDy-mode-off)
- `to(device)` — Also moves the sindy loss tensors, presence/prior masks, degree weights, and patience counters, since these are plain dict/Tensor attributes rather than registered buffers

**Ensemble support:** `EnsembleLinear`, `EnsembleEmbedding`, `EnsembleRNNModule` — vectorized computation across ensemble members.

**`EnsembleRNNModule` architecture:** Despite being referred to as a "GRU" in some comments, it is **not** a GRU — there are no reset/update gates. It's a single GELU-MLP residual update, `dt`-scaled:
```
x_t = concat(inputs[t], h_in)                                       # h_in = h if include_state else 0
gi  = dropout(GELU(W_linear @ x_t + b_linear))                      # proj_size = 8 + input_size + embedding_size
n   = W_n @ gi + b_n
h   = h + dt * n                                                    # residual update
```
No sigmoid/tanh gates — the architecture is inherently more polynomial-amenable than a traditional GRU.

### Known Inert Code

A few attributes/parameters exist in the model but are not currently wired into the active forward path — present for a planned feature or an experiment that didn't pan out, not active behavior:
- `sindy_damping_raw` (per-module learnable damping, `sigmoid(raw) ∈ (0,1)`) — the multiplication into `forward_sindy` is commented out.
- `weight_out_scale` in `EnsembleRNNModule` — bounded-tanh rescaling of the output is commented out.
- Degree-weighting in `compute_weighted_coefficient_penalty` — currently overridden to uniform weights (`degree_weights = torch.ones_like(...)`, marked `TODO: REMOVE IF NOT HELPING` in the code).

Do not describe these as active behavior in code you write against this model; if you need them, they'll need to be re-enabled and tested first.

---

## SpiceConfig (`spice/resources/spice_utils.py`)

Architecture specification for a SPICE model. This is the central configuration that defines the cognitive architecture — which submodules exist, what they receive as input, and how they contribute to behavior.

```python
SpiceConfig(
    library_setup: Dict[str, Iterable[str]],
    memory_state: Union[List[str], Dict[str, float]],
    states_in_logit: List[str] = None,
    additional_inputs: List[str] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `library_setup` | `Dict[str, Iterable[str]]` | *required* | Maps each RNN submodule name to its input control signals (excluding self-references — the module's own state is added automatically). Module names must match keys used in `setup_module()`. Control signals are features from the input data (e.g., `'reward'`, `'choice'`). |
| `memory_state` | `Dict[str, float]` or `List[str]` | *required* | Defines the model's latent memory state variables and their initial values. If a list, all initial values default to `0.0`. A value of `None` creates a learnable per-participant initial value instead of a fixed constant (see `learnable_initial_values` above). Each key names a state variable that can be updated by `call_module()` via `key_state`. |
| `states_in_logit` | `List[str]` | `None` (= all states) | Subset of `memory_state` keys that feed into the output logits. If `None`, all memory states are used. Use this to exclude auxiliary states (e.g., working memory buffers) from the action-selection computation. |
| `additional_inputs` | `List[str]` | `None` | Extra named control-signal columns beyond the standard actions/rewards (e.g. task-specific metadata like dominance rank). Sliced out in `init_forward_pass()` and exposed via `spice_signals.additional_inputs[name]`. |

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

---

## SpiceDataset (`spice/resources/spice_utils.py`)

```python
SpiceDataset(
    xs, ys,
    n_reward_features: int = None,
    sequence_length: int = None,
    stride: int = 1,
    device=None,
    continuous_action: bool = False,
)
```

PyTorch Dataset with auto-promotion (2D→3D→4D: unsqueezes a session dim, then a within-trial dim), NaN-based padding, and optional sequence splitting via `sequence_length`/`stride` (BPTT-truncation windowing along the outer-trial axis).

**Objective: Next-action prediction.** `xs[t]` contains the observation at trial `t` and `ys[t] = action[t+1]` (the next action, one-hot encoded). This is the standard cognitive modeling objective — predicting what the participant will do next given their history.

**Shape:** `(sessions, outer_ts, within_ts, features)`

**Feature columns in xs:** `[actions (one-hot, n_actions cols), rewards (one-hot, n_actions cols, optional), additional_inputs, time_trial, trials, block, experiment_id, participant_id]`

**Feature columns in ys:** `[next_action (one-hot, n_actions cols)]`

**`continuous_action=True`** stores raw (non-one-hot) action values for continuous/multivariate action spaces, and skips argmax-based handling elsewhere in the pipeline.

**Other methods:** `normalize_column(index, range=None)` / `normalize_rewards(range=None)` — in-place min-max normalization of xs feature columns (auto-detects `(0,1)`, `(-1,0)`, or `(-1,1)` range from the column's sign).

**`SpiceSignals`** — the plain container `init_forward_pass()` fills and hands to `forward()`: `participant_ids, experiment_ids, blocks, actions, feedback, trials, time_trial, additional_inputs, logits, sindy_loss_timesteps, mask_valid_trials`.

---

## SpiceEstimator (`spice/resources/estimator.py`)

Main user-facing class implementing sklearn's estimator interface.

**Methods:** `fit(data, targets, data_test, target_test)`, `predict(conditions)` → single prediction array (ensemble mean in RNN mode, member-0 in SINDy mode — **not** a `(rnn_pred, spice_pred)` tuple), `save_spice(path)`, `load_spice(path)`, `get_sindy_coefficients()`, `get_participant_embeddings()`, `print_spice_model()`, `count_sindy_coefficients()`, `get_modules()`, `get_candidate_terms()`

**Constructor arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Model specification** ||||
| `spice_class` | `BaseModel` | *required* | RNN class (precoded or custom subclass of BaseModel) |
| `spice_config` | `SpiceConfig` | *required* | Architecture configuration |
| `kwargs_spice_class` | `dict` | `{}` | Extra keyword arguments forwarded to `spice_class.__init__()` |
| **Data/environment** ||||
| `n_actions` | `int` | `2` | Number of observable actions |
| `n_items` | `int` | `None` | Number of internal item representations (defaults to `n_actions`) |
| `n_participants` | `int` | `1` | Number of participants in the dataset |
| `n_experiments` | `int` | `1` | Number of experiments |
| `n_reward_features` | `int` | `None` | Number of reward feature columns (auto-detected if `None`) |
| **RNN training** ||||
| `epochs` | `int` | `1` | Total Stage-1 training epochs |
| `warmup_steps` | `int` | `0` | Epochs of exponential SINDy weight warmup (no pruning during warmup) |
| `learning_rate` | `float` | `1e-2` | Learning rate for RNN parameters |
| `batch_size` | `int` | `None` | Training batch size (`None` = auto-detect max via GPU probing) |
| `n_steps_per_call` | `int` | `None` | BPTT truncation length (`None` = full sequence) |
| `ensemble_size` | `int` | `10` | Number of independent RNN ensemble members |
| `embedding_size` | `int` | `8` | Participant/experiment embedding dimensionality |
| `dropout` | `float` | `0.1` | Dropout rate in RNN modules |
| `l2_rnn` | `float` | `0` | L2 weight decay for RNN parameters |
| `convergence_threshold` | `float` | `0` | Early stopping threshold (0 = disabled) |
| `bagging` | `bool` | `False` | Whether to use bagging |
| `loss_fn` | `callable` | `cross_entropy_loss` | Behavioral loss function `(prediction, target) → scalar` |
| `loss_fn_kwargs` | `dict` | `{'label_smoothing': 0.01}` | Extra kwargs forwarded to `loss_fn` |
| `device` | `torch.device` | `cpu` | Compute device |
| **SPICE / SINDy** ||||
| `use_sindy` | `bool` | `False` | Enable SINDy integration (forward predictions use equations instead of the RNN) |
| `sindy_weight` | `float` | `0.01` | Lambda for SINDy regularization loss (`sindy_loss_reg`) |
| `sindy_alpha` | `float` | `1e-4` | Degree-weighted L1 penalty strength (also used as ridge alpha in Stage 2) |
| `sindy_library_polynomial_degree` | `int` | `2` | Max polynomial degree for SINDy candidate library |
| `sindy_pruning_frequency` | `int` | `100` | Epochs between pruning events |
| `sindy_threshold_pruning` | `float` | `0.01` | Minimum `\|coefficient\|` for a member to count as supporting a term in the ensemble ratio test (`None` disables) |
| `sindy_ensemble_pruning` | `float` | `0.5` | Minimum fraction of ensemble members that must exceed `sindy_threshold_pruning` for a term to survive (ensemble ratio test); primary pruning mechanism |
| `sindy_pruning_terms` | `int` | `None` | Max terms pruned per event, across all modules (`None` = auto-computed so pruning can reach zero terms within the available epochs) |
| `sindy_reconditioning_epochs` | `int` | `3` | Pure SINDy SGD epochs after ridge recalibration (currently unused — see [Two-Stage Training Pipeline](#two-stage-training-pipeline-spiceresourcesspice_trainingpy)) |
| `sindy_refit` | `bool` | `True` | Enable Stage 2 (SINDy refit on frozen RNN parameters). If `False`, the estimator returns whatever Stage 1 produced. |
| `sindy_ridge` | `bool` | `True` | Use closed-form ridge regression to initialize Stage 2 coefficients (falls back to SGD on failure/non-finite loss) |
| `sindy_shooting_steps` | `int` | `100` | Multi-step "shooting" rollout horizon for Stage 2 (`1` = one-step-ahead; larger values penalize compounding error over more trials) |
| **Output / misc** ||||
| `verbose` | `bool` | `False` | Print training progress |
| `keep_log` | `bool` | `False` | Keep full training log (vs. live terminal update) |
| `save_path_spice` | `str` | `None` | Auto-save .pkl path after Stage 1 (as `<path>_stage1.pkl`) and after training completes |
| `compiled_forward` | `bool` | `True` | Use `@torch.compile` for forward loops |

`predict()` returns a single array: the ensemble mean in RNN mode, or ensemble member 0 in SINDy mode (all members are fit toward consensus targets in Stage 2, so any one member is representative).


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

```python
csv_to_dataset(
    file: Union[str, pd.DataFrame],
    df_participant_id: str = 'participant',
    df_block: str = 'block',
    df_experiment_id: str = 'experiment',
    df_choice: Union[str, Iterable[str]] = 'choice',
    df_feedback: Optional[Union[str, Iterable[str]]] = 'reward',
    df_time: Optional[str] = None,
    df_trial: Optional[str] = None,
    additional_inputs: Optional[Union[str, Iterable[str]]] = None,
    device=None,
    sequence_length: int = None,
    timeshift_additional_inputs: Optional[Iterable[int]] = None,
    remove_failed_trials: bool = True,
    continuous_action: bool = False,
) -> SpiceDataset
```

```
Raw CSV (participant, experiment, block, choice, [reward], [additional_inputs])
    ↓  csv_to_dataset()
    ↓  - Map categorical columns to numeric IDs (string choices → sorted-alphabetical codes, with a warning)
    ↓  - One-hot encode choices → n_actions columns (or store raw values if continuous_action=True)
    ↓  - Promote rewards to action-aligned structure → n_actions columns (one per action)
    ↓  - Normalize rewards to [-1, 1] or [0, 1]
    ↓  - Build metadata columns (last 5: time_trial, trials, block, experiment_id, participant_id)
    ↓  - Shift: xs[t] = observation at trial t, ys[t] = action[t+1]
    ↓
SpiceDataset: shape (sessions, outer_ts, within_ts, features)
```

**Rewards are optional** (`df_feedback=None` to omit). Full vs. partial feedback is inferred structurally:
- **Partial feedback** (single reward column, discrete action): reward is placed in the column of the chosen action; unchosen columns are NaN
- **Full/counterfactual feedback** (`len(df_feedback) > 1`, or `continuous_action=True`): each feedback column maps directly to its corresponding action column, no NaN masking of alternatives

**Within-trial sequences:** passing `df_trial`/`df_time` switches to a code path that groups rows by an outer-trial identifier and produces a true 4D `(sessions, outer_ts, within_ts, features)` dataset (e.g. for DDM/evidence-accumulation models). Without them, `within_ts` is always 1.

**`timeshift_additional_inputs`** applies a per-column shift (`-1`/`0`/`1`) to additional-input columns along the trial dimension; a `'-'` entry also truncates the last trial from both `xs`/`ys`.

**Round-trip export:** `dataset_to_csv()` reconstructs a tabular CSV from a `SpiceDataset`, auto-detecting full vs. partial feedback and reward-column naming.

**Splitting utilities** (also in `spice/utils/convert_dataset.py`, not `spice_utils.py`): `split_data_along_timedim(dataset, split_ratio, device)`, `split_data_along_blockdim(dataset, test_blocks=None, device)`, `reshape_data_along_participantdim(dataset, device)`.

**Dataset metadata conventions:** block and participant IDs in metadata (`xs[..., -3]` and `xs[..., -1]`) — participant IDs are 0-indexed (remapped by `csv_to_dataset`); block IDs are kept as-is from the CSV (typically 1-indexed).

---

## Two-Stage Training Pipeline (`spice/resources/spice_training.py`)

Main function: `fit_spice()`, called by `SpiceEstimator.fit()`. Returns `model.eval(use_sindy=True), optimizer`.

### Stage 1: Joint RNN-SINDy Training (`_run_joint_training`, runs when `epochs > 0`)

The loss is composed in `_run_batch_training()` (called each iteration on a session batch):
```python
loss = loss_fn(ys_pred, ys_step, **loss_fn_kwargs)                        # behavioral loss
loss = loss + sindy_weight * model.sindy_loss_reg                         # SINDy regularization (RNN-side gradient only)
loss = loss + model.compute_weighted_coefficient_penalty(sindy_alpha)     # degree-weighted L1 penalty on SINDy coefficients
```
`sindy_weight * model.sindy_loss_reg` is only added when `sindy_weight > 0` and `model.sindy_loss_reg != 0`; the penalty term only when `sindy_weight > 0 and sindy_alpha > 0`.

**Epoch flow (matching the actual code):**
1. **SINDy weight warmup**: for the first `n_warmup_steps` epochs, the effective SINDy weight is scaled by an exponential warmup curve (`_setup_warmup_scaler`) instead of applied at full strength immediately.
2. **Batching**: sessions are randomly batched (5D tensors `(ensemble, sessions, trials, within_ts, features)`); each batch runs through `_run_batch_training`, accumulating the average training loss for the epoch.
3. **LR scheduling**: `torch.optim.lr_scheduler.ReduceLROnPlateau` (`mode='min', factor=0.5, patience=50`) steps on the epoch's training loss. RNN params can decay to `min_lr=1e-5`; SINDy params are effectively pinned at their base LR (`0.01`).
4. **Validation** (only if `dataset_test` is given): one no-grad forward pass in RNN mode (`loss_test_rnn`) and, if `sindy_weight > 0`, one in SINDy mode (`loss_test_sindy`).
5. **Pruning** (only if `sindy_weight > 0` and `sindy_pruning_frequency` is set, and only past warmup): fires when the epoch count hits `sindy_pruning_frequency` (or on the first call). Uses `_ensemble_pruning` (the ensemble ratio test, `_ensemble_ratio_test`: a term survives iff at least `sindy_ensemble_pruning` fraction of ensemble members have `|coefficient| > sindy_threshold_pruning`) when `ensemble_size > 1` and `sindy_ensemble_pruning` is set; otherwise falls back to per-member hard thresholding (`model.sindy_coefficient_pruning`). A term must fail **2 consecutive** pruning checks before permanent removal (patience counter resets on success). Pruning only masks/zeroes coefficients — it does **not** currently ridge-recalibrate them within Stage 1 (that code path is present but commented out).
6. **Convergence check**: an exponentially-smoothed estimate of `|Δloss|` (recency factor 0.5) is compared against `convergence_threshold`; training stops early once below threshold.
7. Auto batch-size probing: the whole batch loop is wrapped in a retry loop that halves `batch_size` on CUDA OOM.
8. `KeyboardInterrupt` is caught, allowing manual early stop while still returning a usable model.
9. After Stage 1, if a save path is configured, an intermediate checkpoint is saved as `<path>_stage1.pkl` — explicitly before Stage 2 overwrites the coefficients.

### Stage 2: Final SINDy Refit (`_run_sindy_training`, runs only when `sindy_refit=True`)

Freezes RNN weights and refits SINDy coefficients via multi-step "shooting" rather than one-step-ahead fitting, on CPU. Two sub-stages:

**Stage 2.1 — Sparsity discovery** (only if a pruning criterion is configured): resets all presence masks to fully active (respecting `sindy_coefficients_prior_mask`), re-initializes coefficients, then fits with genuine one-step (`K=1`) shooting — every within/across-trial transition is flattened into its own independent pseudo-session — combined with pruning and the L1 penalty. Starts from a closed-form ridge solve (`_ridge_solve_sindy`) when it produces a finite loss; falls back to random init + SGD otherwise.

**Stage 2.2 — Coefficient estimation**: freezes the sparsity pattern discovered in 2.1 (or the original pattern if 2.1 didn't run), re-initializes coefficients within that support, and fits via multi-step shooting with `K = sindy_shooting_steps` (default 100, clamped to the number of trials) — a window of `K` consecutive trials is rolled out autoregressively and the loss is the MSE against the RNN's own recorded state trajectory at each step, so this stage penalizes compounding error over the rollout, not just per-step error. Starts from a ridge solve (`sindy_ridge=True`) evaluated under the K-step rollout loss; falls back to SGD if ridge fails or produces a non-finite K-step loss (a ridge solve can be well-posed in isolation but unstable once rolled out — e.g. a self-coefficient that blows up over many steps).

If `E > 1` (ensemble), Stage 2.2 re-computes state trajectories on the full (non-bootstrapped) dataset with per-ensemble-member targets before fitting, so all members converge toward a consensus.

`sindy_reconditioning_epochs` is accepted by `fit_spice`/`SpiceEstimator` but not currently used by the Stage 2 code path described above — it is a leftover constructor parameter from an earlier design and has no effect.

### Custom Loss Functions

A custom loss function can be passed via `loss_fn` to both `SpiceEstimator` and `fit_spice()`. It must accept `(prediction, target, **loss_fn_kwargs)` and return a scalar tensor.

**Default: `cross_entropy_loss`** (`spice/resources/spice_training.py`):
```python
def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor, label_smoothing=0.) -> torch.Tensor:
    n_actions = target.shape[-1]
    prediction = prediction.reshape(-1, n_actions)                    # flatten to (N, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)       # one-hot → class index
    return torch.nn.functional.cross_entropy(prediction, target, label_smoothing=label_smoothing)
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

**Available precoded models** (`spice/precoded/`): Rescorla-Wagner (`rescorlawagner.py`), Choice Perseveration (`choice.py`), Forgetting (`forgetting.py`), Learning Rate (`learningrate.py`), Interaction (`interaction.py`), Embedding (`embedding.py`), DDM (`ddm.py`), Working Memory + variants (`workingmemory.py`, `workingmemory_counterfactual.py`, `workingmemory_multiitem.py`).

---

## Designing Polynomial-Amenable Architectures

Full guidelines: [`docs/guidelines_polynomial_amenable_architectures.md`](guidelines_polynomial_amenable_architectures.md)

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
- **One-hot encoding**: Actions and rewards must be one-hot encoded, not integer indices (unless `continuous_action=True`)
- **SINDy convergence**: If coefficients don't converge, adjust `sindy_weight` or increase epochs
- **Patience tuning**: Too low → premature elimination; too high → delayed sparsification
- **`SpiceEstimator.predict()` returns a single array**, not `(rnn_pred, spice_pred)` — check `estimator.model.use_sindy` (or call `estimator.eval(use_sindy=...)` first) to control which mode it reflects
- **`sindy_refit=False`** skips Stage 2 entirely — coefficients returned are whatever Stage 1's joint training converged to, not a dedicated refit
- Some attributes exist but are currently inert (see [Known Inert Code](#known-inert-code)) — don't assume every constructor parameter or model attribute is on the active path; check before relying on it

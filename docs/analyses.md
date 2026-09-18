# SPICE Analyses

Two layers of downstream analysis sit on top of a fitted `SpiceEstimator`:

1. **Generative benchmarking** (`weinhardt2026/utils/task.py`) — simulate new behavior by running the fitted model through the task environment, for comparison against real data.
2. **Cross-study analysis pipelines** (`weinhardt2026/analysis/`) — model evaluation, morphing, coefficient-level statistics, clustering, compression. These operate on a fitted `SpiceEstimator` (and optionally its generated behavior) and are shared across all studies in `weinhardt2026/studies/`.

See [training.md](training.md) for the model/training internals these analyses consume, and [studies.md](studies.md) for how individual studies wire them together.

---

## Generative Benchmarking (`weinhardt2026/utils/task.py`)

Generative benchmarking simulates new behavioral data by running a fitted model through the original task environment. This produces synthetic datasets that can be compared against the original human data — the basis for the generative-comparison and behavioral-clustering analyses below.

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

**Result**: a synthetic `SpiceDataset` with the same shape/metadata as the input, but actions (and derived rewards) drawn from the fitted model instead of the human participant. Feed this into `analysis_generative_comparison.py` or `analysis_behavioral_clustering.py` below.

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

---

## Cross-Study Analysis Pipelines (`weinhardt2026/analysis/`)

All analyses below are study-agnostic functions that take a fitted `SpiceEstimator` (loaded from a `.pkl` checkpoint or passed in-memory) plus a `SpiceDataset`, and write CSVs/plots to an `output_dir`. Each study wires these into its own data paths from `weinhardt2026/studies/<study>/<study>.py` — there are no per-study analysis runner scripts; anything reusable belongs here instead.

Analyses that only read coefficients do not need the training data. `weinhardt2026/utils/checkpoints.py` loads a checkpoint on its own:

```python
load_estimator(model_path, spice_class, spice_config, n_actions=2,
               n_participants=None, polynomial_degree=2, model_kwargs=None)
```

inferring `ensemble_size` and `n_participants` from the saved tensors. It also provides `select_lowest_bic_checkpoint(hpscan_csv, params_dir)`, `participant_label_map(data_path, label_col)` (SPICE's 0-based participant index → a label column) and `log_trials_per_participant(data_path)` (the data-volume covariate that presence regressions adjust for).

### Model Evaluation — `analysis_model_evaluation.py`

Quantifies how well a fitted SPICE model (RNN and/or SINDy-equation form) predicts held-out behavior, and compares against a hand-coded benchmark model and/or a plain GRU baseline.

```python
analysis_model_evaluation(
    dataset, spice_model=None, benchmark_model=None, gru_model=None,
    output_dir=None, trial_filter=None, n_actions_random_baseline=None,
)
```

- Computes per-trial **log-likelihood**, **BIC**, **AIC**, and **ΔBIC vs. random-choice baseline**.
- Information criteria are computed **per (participant, experiment) group** and reported as mean ± std across groups (`grouped_information_criteria`) — a single dataset-pooled BIC is not a fair comparison across models with different parameter-sharing structure (e.g. shared vs. per-participant coefficients).
- `trial_filter` lets you restrict evaluation to a subset of trials (e.g. excluding "wait" responses in a DDM-style task); pair with `n_actions_random_baseline` to correct the random baseline's action count.
- `analysis_model_evaluation_mse` — parallel pipeline for continuous-output (regression) models: computes MSE-based metrics instead of choice likelihoods.
- **Result**: a DataFrame/CSV of per-model (SPICE-RNN, SPICE-SINDy, benchmark, GRU) log-likelihood, BIC, AIC, ΔBIC — the standard model-comparison table used to argue SPICE fits at least as well as hand-crafted models.

### One-Step-Ahead Diagnosis — `analysis_sindy_onestepahead.py`

Diagnoses *why* the fitted SINDy equations underperform the RNN at test time: per-step approximation error vs. autoregressive error accumulation.

```python
analysis_sindy_onestepahead(dataset, spice_model) -> (summary_df, per_trial_df)
```

Compares three evaluation modes: **SPICE-RNN** (autoregressive, ensemble mean), **SPICE-SINDy (autoregressive)** (equations feed their own past predictions forward), **SPICE-SINDy (one-step-ahead)** (equations receive the RNN's true state at each trial). If one-step-ahead ≈ RNN but autoregressive ≪ RNN → the equations are locally accurate but their errors compound over trials. If one-step-ahead ≪ RNN → the equations poorly approximate the RNN even locally.

### Model Morphing — `analysis_morphing.py`

Traces how SINDy equation coefficients change along a continuous behavioral/embedding axis (e.g. from healthy to depressed, or low to high reward rate), rather than treating participant groups as discrete.

```python
run_morphing(
    estimator, dataset, metric_values, n_steps=20, morphing_range_sd=1.0,
    n_pruning_rounds=20, pruning_threshold=0.05, save_dir=None,
) -> dict  # {'member_results': [...]}
```

1. `_find_morphing_direction` — finds the direction in each ensemble member's participant-embedding space that best predicts `metric_values` (e.g. via `LinearRegression`).
2. `_create_morphed_dataset` / `_create_morphed_estimator` — steps `n_steps` points along that direction, refitting SINDy coefficients at each point with a fast **ridge → prune → ridge → prune → ... → ridge** cycle (closed-form solves, no SGD — orders of magnitude faster than full retraining).
3. Aggregates across ensemble members (each has its own RNN/embedding space, so morphing directions are found independently per member) into mean ± SE coefficient curves.
4. `get_morphed_coefficients(result)` extracts the coefficient trajectories for plotting.

`analysis_morphing(...)` wraps all of the above and is the entry point a study calls:

```python
analysis_morphing(
    estimator, dataset, metric_values, output_dir,
    environment=None,           # a weinhardt2026.utils.task.Env; enables validation
    n_steps=20, morphing_range_sd=1.0, save_dir=None,
    n_validation_runs=5, prefix='morphing',
)
```

It writes `<prefix>_coefficients.npz` and, when an `environment` is given, calls `validate_morphing()` — which generates behavior from the morphed model at each step and checks that the metric actually follows the axis (monotonicity count + step/metric correlation), writing `<prefix>_validation.npz`. The environment is the only task-specific piece, so it is passed in rather than imported.
- **Result**: for each SINDy term, a curve of its coefficient value as a function of position along the morphing axis — showing e.g. a nonlinear exploration term's coefficient collapsing toward zero as depression severity increases. This is how SPICE reveals *structural* (not just parametric) individual differences.

### Generative Comparison — `analysis_generative_comparison.py`

Compares distributions of behavioral metrics (from `generate_behavior`) between real data and one or more generative models.

```python
compute_generative_comparison(
    all_metrics: dict[str, dict[str, np.ndarray]],  # {model_name: {metric_name: array}}, must include 'real'
    participant_ids: dict[str, np.ndarray],
    output_dir=None,
) -> (df_similarity, df_spearman)
```

- `df_similarity` — `1 - normalized Wasserstein distance` per metric per model: how closely a model's *distribution* of a metric (e.g. average reward, switch rate) matches real participants.
- `df_spearman` — Spearman rank correlation per metric per model: whether a model preserves *individual differences* (which participants score high/low), not just the population distribution.
- **Result**: a two-table summary distinguishing "gets the population right" from "gets individuals right" — a model can match one without the other.

### Coefficient Distributions — `analysis_coefficients_distributions.py`

Population-level view of the fitted SINDy coefficients: which terms are present, how consistent they are across the ensemble, and how they compare across experiments.

```python
analysis_coefficients_distributions(
    spice_model=None, model_path=None, model_module=None, model_class=None,
    model_config=None, dataset=None, output_dir="analysis_coefficient_distributions",
    max_participants_strip=30, cluster_heatmap=True,
) -> (coeff_df, presence_df, ensemble_consistency_df)
```

Produces: ensemble-spread plots and CV heatmaps (`compute_ensemble_consistency`, `plot_ensemble_spread`, `plot_ensemble_cv_heatmap`) showing how stable each coefficient is across ensemble members; violin plots of coefficient distributions across participants (`plot_coefficient_violins`); presence-rate bar charts (`plot_presence_rate_bar`) — what fraction of participants retain each term after pruning; experiment-comparison plots (`plot_experiment_comparison`); and a sparsity heatmap (`plot_sparsity_heatmap`) of which terms are active for which participants.

### Individual-Differences Regression — `analysis_coefficients_individuals.py`

Tests whether individual SINDy coefficients relate to an external criterion (diagnosis, questionnaire score, task performance).

```python
analysis_coefficients_individuals(
    path_data, criterion, analysis,      # analysis: "disc" (discrete/odds-ratio) or "cont" (continuous)
    reference=None, spice_model=None, path_model=None, ...,
    output_dir=None,
)
```

- **Discrete** (`run_discrete`): logistic regression of term *presence* on group membership (e.g. patient vs. control), reported as odds ratios with forest plots (`_plot_forest`, `_plot_odds_ratios`) and per-group presence rates (`_plot_presence_rates`).
- **Continuous** (`run_continuous`): regresses coefficient *magnitude* on a continuous criterion, with beta-coefficient bar plots and fitted logistic curves; `jonckheere_terpstra` tests for monotonic trend across ordered groups.
- **Result**: statistical evidence for *which mechanisms* (equation terms) differ between groups or scale with a trait — the individual-level counterpart to the morphing analysis above.

### Coefficient Clustering — `analysis_coefficient_dendrogram.py`

Groups the candidate terms *within each submodule* by how their per-participant coefficients covary across the population, drawing one dendrogram per module. Terms that move together (or in exact opposition, e.g. `c_reward = -c_value` = a learning rate) form a **tie** that can be read as a single cognitive mechanism rather than as free parameters.

```python
analysis_coefficient_dendrogram(
    output_dir, spice_model=None, model_path=None, spice_class=None, spice_config=None,
    mode="coefficients",          # or "presence"
    metric="abs",                 # 1-|r| (sign-flipped ties join early) or "signed" (1-r)
    linkage_method="average", distance_threshold=0.5,
    min_active_fraction=0.05, min_overlap=30, pairwise_complete=True,
    n_bootstrap=0, bootstrap_seed=0, prefix=None,
)
```

Two axes, deliberately kept separate:

- **parametric** (`mode="coefficients"`) — given both terms are present, are their magnitudes locked? This is the axis that counts as model reduction.
- **structural** (`mode="presence"`) — do the terms switch on and off together across participants (Jaccard distance on the presence masks)? An individual difference in equation *form*.

A pruned term is stored as an exact zero, so correlating full columns mixes the two axes: two terms absent in the *same* participants correlate strongly without their magnitudes being related. `pairwise_complete=True` (the default) therefore correlates each pair **only over participants in which both terms are present**. Pairs with fewer than `min_overlap` such participants cannot be judged and are set to `r = 0`; the count of untestable pairs is printed per module and is worth reading alongside the tree. `--full-columns` restores the old, mixed behaviour.

With `n_bootstrap > 0`, participants are resampled with replacement and reclustered, giving per-cluster **Hennig clusterwise Jaccard** (mean best-match overlap across draws; ≥0.75 conventionally "stable") plus the weakest member pair's co-assignment probability, and a term-by-term co-assignment heatmap. These are *reproducibility* measures, not p-values — a cluster driven by estimation-noise correlation resamples just as stably as a real one.

- **Outputs**: `dendrogram_*.png` (one panel per module), `dendrogram_heatmaps_*.png` (similarity, in leaf order), `dendrogram_clusters_*.csv`, and with bootstrapping `dendrogram_stability_*.csv` + `dendrogram_coassignment_*.png`.

### Coefficient Beta Effects — `analysis_coefficient_betas.py`

Standardized effect of each SINDy coefficient on a per-participant behavioral criterion.

```python
analysis_coefficient_betas(
    data_path, output_dir, spice_model=None, model_path=None, ...,
    criterion_col="mean_reward", participant_col="participant", criterion_aggregate="mean",
    alpha=0.05, adjust_for_volume=True, include_singletons=True,
    distance_threshold=0.5, min_active_fraction=0.05, prefix=None,
)
```

Two Benjamini-Hochberg-corrected families, each corrected across all modules:

1. **single coefficients** — one univariate model per candidate term. This is a pre-specified regressor set and is the family to report.
2. **clusters** — the same against the cluster scores from the dendrogram above, where a cluster score is the sign-aligned mean of its z-scored members. With `include_singletons=True` (default) terms that joined no cluster enter as themselves, so the regressors partition the kept terms.

Every row also carries `beta_adjusted` / `p_adjusted`, the same effect with `log_n_trials` partialled out. **Check these**: a criterion like "reward per block" can be dominated by how many trials a participant played rather than by performance, in which case the adjusted column collapses and a per-trial reward rate (`criterion_col="reward"`, `criterion_aggregate="mean"`) is the better criterion.

A joint model over all cluster scores is also written (`betas_clusters_joint_*.csv`), giving partial betas — each regressor's contribution beyond the others. It is only interpretable with enough participants per regressor and moderate collinearity; with singletons included the regressor count rises and partial estimates can flip sign relative to their univariate values, which is suppression, not mechanism.

Two caveats on the cluster family: the clusters are derived from the same participants the regression then uses, so their q-values are optimistic in a way the FDR correction does not capture (a split-half — discover on one half, test on the other — fixes this); and every model here is cross-sectional across participants, so a coefficient predicting performance does not establish that changing it would change performance. On the studies checked so far the cluster betas have not exceeded their own best member's effect, so the single-term family alone is usually sufficient for inference, with the clustering serving interpretation.

- **Outputs**: `betas_terms_*.csv`, `betas_clusters_*.csv`, `betas_clusters_joint_*.csv`, and forest plots (filled markers = significant after FDR).

### Deeper Look at One Checkpoint — `analysis_bestbic.py`

Four analyses that read a single fitted checkpoint (typically the lowest-BIC one from the sparsity scan, via `select_lowest_bic_checkpoint`) and describe how its discovered equations vary across participants.

```python
analysis_embedding_dynamics(estimator, dataset, output_dir, results_dir,
                            label_map=None, module_symbols=None, n_select=3)
analysis_metric_fingerprint(estimator, output_dir, results_dir, metrics_csv,
                            data_path, metric='avg_reward', label_map=None)
analysis_group_betas(estimator, output_dir, results_dir, data_path,
                     criterion='diag', reference='Control')
analysis_metric_betas(estimator, output_dir, results_dir, data_path,
                      metrics_csv, metric='avg_reward')
```

1. **Embedding extremes** — the participants furthest apart in participant-embedding space, with their choice / value-update dynamics and their equations rendered as LaTeX (`format_equations_latex`; `module_symbols` maps a module name to its LaTeX symbol, and modules it omits are printed by name).
2. **Fingerprint** — structural (presence) and parametric (magnitude) coefficient differences ordered by a behavioral metric, adjusted for log data volume.
3. **Group betas** — presence regression of every coefficient on a discrete grouping column, wrapping `analysis_coefficients_individuals` and dropping near-constant terms.
4. **Metric betas** — the same against a continuous participant-level metric, attaching it to the behavioural CSV as a participant-constant column first.

### Archived — coefficient compression and mechanism-level differences

`analysis_coefficient_compression.py`, `analysis_mechanism_individuals.py`, `analysis_concepts.py` and `analysis_coefficient_ties.py` have moved to `weinhardt2026/analysis/archive/`, together with the core modules they wrapped (`sindy_compression.py`, `sindy_concepts.py`, `sindy_ties.py`, now under `archive/core/`). They compressed per-participant coefficients into a small number of "mechanisms" via per-module NMF and re-tested group differences at the mechanism level. `analysis_coefficient_dendrogram.py` above covers the same question — which coefficients belong together — without the factorization step.

### Behavioral Clustering — `analysis_behavioral_clustering.py`

Tests whether clusters found in raw behavior (e.g. average reward, switch rate, from `generate_behavior`) align with structural differences in the fitted equations.

```python
analysis_behavioral_clustering(
    spice_model, path_behavioral_metrics, n_clusters=3, output_dir='results',
) -> dict  # {'labels', 'linkage', 'centroids', 'nearest', 'equation_tests', 'alignment_ari', 'df_metrics'}
```

1. Loads per-participant behavioral metrics CSV (produced by a generative-behavior analysis).
2. Hierarchical clustering (`linkage`, `fcluster`, Ward's method) on standardized behavioral metrics.
3. Extracts equation features (`_extract_equation_features`: coefficients + presence per participant).
4. Tests whether equation structure differs across behavioral clusters (`_test_equation_differences`: Kruskal-Wallis / Mann-Whitney) and reports `adjusted_rand_score` alignment between behavioral clusters and any independently-known grouping.
- **Result**: evidence for (or against) the claim that behaviorally-defined subgroups correspond to structurally distinct equations, not just parameter shifts.

### Reward-History Kernel — `analysis_reward_history_kernel.py`

A classic behavioral-analysis baseline, independent of any fitted SPICE model: logistic regression of "stay with previous action" on reward history.

```python
compute_reward_history_kernel(dataset, max_lag=6) -> dict  # 'lags', 'coef_own', 'coef_other', 'se_own', 'se_other', + statsmodels result
```

For each trial, regresses `stay = 1[action[t] == action[t-1]]` jointly on reward history for the previously-chosen action (`reward_own`, lags 1..max_lag) and the other action (`reward_other`, lags 2..max_lag; lag-1 is structurally collinear with the intercept and dropped). Works on both real and model-generated datasets — run once on human data and once on `generate_behavior` output to check whether a fitted model reproduces the reward-history kernel shape.

### Parameter Recovery — `analysis_parameter_recovery.py`

Synthetic-data sanity check: simulate a known ground-truth model (e.g. `QLearning` from `weinhardt2026/studies/synthetic/`), fit SPICE to the simulated data, and verify the recovered SINDy coefficients match the ground-truth parameters. Handles term collapsing for binary signals (`signal^1 == signal^2`, e.g. binary reward/choice) when comparing fitted vs. true coefficients.

### Sparsity Hyperparameter Scan — `analysis_sparsity_hpscan.py`

```python
analysis_sparsity_hpscan(
    pkl_pattern, spice_class, spice_config, n_actions, data_path, test_blocks,
    polynomial_degree=2, model_kwargs=None, device=None,
) -> pd.DataFrame
```

Evaluates a batch of checkpoints from a pruning-threshold × pruning-test hyperparameter sweep (glob-matched `.pkl` files, e.g. `params_array/spice_dezfouli2019_*_*.pkl`) on held-out `test_blocks`, returning a summary table of predictive performance vs. sparsity level — used to pick `sindy_threshold_pruning` / `sindy_ensemble_pruning` for a study.

---

## Typical Analysis Sequence for a New Study

1. Fit `SpiceEstimator` on training data, evaluate on held-out data with `analysis_model_evaluation.py`.
2. Run `analysis_sindy_onestepahead.py` if SINDy test performance lags the RNN, to see whether it's a per-step or accumulation problem.
3. Inspect population-level equation structure with `analysis_coefficients_distributions.py`.
4. Relate coefficients to external criteria: `analysis_coefficients_individuals.py` (discrete groups or continuous traits), `analysis_coefficient_betas.py` for standardized effects of each term on a behavioral criterion, and/or `analysis_morphing.py` for a continuous structural trajectory.
5. Generate synthetic behavior (`generate_behavior`) and validate it against real data with `analysis_generative_comparison.py` and `compute_reward_history_kernel`.
6. Read the fitted equations as a small number of mechanisms with `analysis_coefficient_dendrogram.py` (one dendrogram per module, bootstrap stability), or check behavioral-cluster/equation alignment with `analysis_behavioral_clustering.py`.
7. For the selected checkpoint, run the four `analysis_bestbic.py` analyses (embedding extremes, fingerprint, group and metric betas) to describe how the equations vary across participants.

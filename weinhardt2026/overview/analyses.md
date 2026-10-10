# SPICE Analyses

Two layers of downstream analysis sit on top of a fitted `SpiceEstimator`:

1. **Generative benchmarking** (`weinhardt2026/utils/task.py`) — simulate new behavior by running the fitted model through the task environment, for comparison against real data.
2. **Cross-study analysis pipelines** (`weinhardt2026/analysis/`) — model evaluation, morphing, coefficient-level statistics, clustering. These operate on a fitted `SpiceEstimator` (and optionally its generated behavior) and are shared across all studies in `weinhardt2026/studies/`.

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
- **Continuous** (`run_continuous`): logistic regression of term *presence* on the standardized criterion, adjusted for log data volume, with profile-likelihood CIs, forest plots and fitted logistic curves; `jonckheere_terpstra` tests for monotonic trend across ordered groups.
- **Result**: statistical evidence for *which mechanisms* (equation terms) differ between groups or scale with a trait — the individual-level counterpart to the morphing analysis above.

### Coefficient Clustering — `analysis_coefficient_dendrogram.py`

Groups the candidate terms *within each submodule* and draws one dendrogram per module, so a fitted model can be read as a handful of co-occurring groups rather than as dozens of separate terms.

```python
analysis_coefficient_dendrogram(
    output_dir, spice_model=None, model_path=None, spice_class=None, spice_config=None,
    mode="coefficients",          # or "presence"
    metric="abs",                 # coefficients mode: 1-|r| or "signed" (1-r)
    linkage_method="average", distance_threshold=0.5,
    min_active_fraction=0.05, min_overlap=30, pairwise_complete=True,
    n_permutations=0, grouping_alpha=0.05, permutation_seed=0,
    split_signs=True,             # presence mode: split every term by sign
    n_bootstrap=0, bootstrap_seed=0, prefix=None,
)
```

Two modes, one per axis:

- **structural** (`mode="presence"`) — do the terms switch on and off together across participants? Distance is **1 − phi**, where phi is the correlation between two terms' 0/1 presence columns. Phi measures co-occurrence *beyond what the terms' frequencies imply* and counts shared absence as much as shared presence. (Jaccard is deliberately not used: two terms that are each present in most participants overlap heavily even when entirely independent, so Jaccard merges common terms rather than co-occurring ones.) Terms that exclude each other have phi < 0 and are pushed apart. This is the mode the beta analysis builds on.

  **Signs are part of the structure.** With `split_signs=True` (default), every term is first replaced by two pseudo-terms, `term (+)` and `term (-)` (`split_by_sign`): a pseudo-term is present in a participant who has the term *with that sign*. A term that points one way in some participants and the other way in others describes two different mechanisms — e.g. a choice bias that makes a participant repeat vs one that makes them switch — so a group like `{1 (-), choice[t-3] (+)}` says which terms switch on together *and in which direction*. The two signs of one term are never present together, so their phi is negative and they never group. A sign nobody carries is dropped; a rare one (below `min_active_fraction`) falls to the activity filter. The trade-off: splitting reveals effects that cancel out when both signs are pooled, but costs power for terms whose direction does not matter to the question, and makes rare terms rarer.
- **parametric** (`mode="coefficients"`) — given both terms are present, do their magnitudes move together? A pruned term is stored as an exact zero, so correlating full columns would mix this with the structural question. `pairwise_complete=True` (default) correlates each pair **only over participants in which both terms are present**; pairs with fewer than `min_overlap` such participants are set to `r = 0` and counted as untestable.

**Significance-based grouping** (`n_permutations > 0`). Instead of a fixed `distance_threshold`, every merge gets a permutation p-value (`merge_significance`):

1. Shuffle each term's column across participants independently. This keeps how common every term is but destroys any link between terms.
2. Rebuild the tree from the shuffled data and record its tightest merge — the strongest chance pair anywhere in the module.
3. Repeat `n_permutations` times (10,000 resolves p-values down to 1e-4 in ~13 s per model).

A real merge's p-value is the share of shuffled trees whose tightest merge was at least as tight, `p = (1 + #{null ≤ h}) / (N + 1)`. Because every merge is compared against the strongest chance pair in the whole module, this controls the **family-wise error rate per module** — no further correction is needed within a module; it is conservative for looser merges higher up the tree. Merge heights only increase up the tree, so "p < α" is exactly "height below a critical distance": the cut for α is the k-th smallest null tightest merge, `k = ⌈α(N+1) − 1⌉` (the 500th / 100th / 10th of 10,000 for α = .05 / .01 / .001). The **groups are the significant merges**: the tree is cut just below the `grouping_alpha` critical distance. Cuts differ per module, since more terms (more pairs for luck to work with) and rarer terms both produce tighter chance merges.

In the figure, each merge — the bracket joining its two children — is coloured by significance tier (dark = p < .001, mid = p < .01, light = p < .05, grey = n.s.), and the three critical distances are drawn as dashed lines in the matching tier colours. Tiers are defined once (`STAR_TIERS`) and shared by stars, colours and cut lines.

Significant co-occurrence means the terms switch on together *more than chance* — not that participants usually have all of them. Read the per-merge **all** (share with every term present) next to **any** (share with at least one): a large gap marks a loose group.

With `n_bootstrap > 0`, participants are resampled with replacement and reclustered at the module's cut, giving per-cluster **Hennig clusterwise Jaccard** (mean best-match overlap of cluster memberships across draws; ≥0.75 conventionally "stable") plus the weakest member pair's co-assignment probability. These are *reproducibility* measures, not p-values.

- **Outputs**: `dendrogram_*.png` (one panel per module), `dendrogram_heatmaps_*.png` (similarity, in leaf order), `dendrogram_clusters_*.csv` (term → group, with group all/any activity), with permutations `dendrogram_merges_*.csv` (one row per merge — every grouping level — with distance, p, stars, all, any), and with bootstrapping `dendrogram_stability_*.csv` + `dendrogram_coassignment_*.png`.

### Structural Beta Effects — `analysis_coefficient_betas.py`

Does the **presence** of a term — or of a group of terms that switch on together — relate to a per-participant behavioral metric? SPICE's claim is about structural individual differences, so this tests presence only, never coefficient magnitude.

```python
analysis_coefficient_betas(
    data_path, output_dir, spice_model=None, model_path=None, ...,
    criterion_col="reward",
    criterion_type="continuous",  # or "discrete" (e.g. a diagnosis column)
    reference=None,               # discrete: compare every group against this one
    comparisons=None,             # discrete: explicit [(group, reference), ...] pairs
    participant_col="participant", criterion_aggregate="mean",
    min_active_fraction=0.05, n_permutations=10000, grouping_alpha=0.05,
    split_signs=True, invariant_bounds=(0.05, 0.95), alpha=0.05, prefix=None,
)
```

1. **Groups** come from the presence dendrogram with significance-based grouping and sign splitting (above). A group counts as present in a participant only when *all* of its terms are, with their signs. The grouping never looks at the criterion, so it is computed once on all participants and shared by every contrast.
2. **Every node inside every group** — the group, its subgroups, down to its single terms — plus every ungrouped term is tested: logistic regression of presence on the criterion. Nodes present in fewer than 5% or more than 95% of the contrast's participants carry no structural variation and are marked invariant instead of tested. Fits that separate perfectly, fail to converge, or — for a group comparison — have a group in which nobody (or everybody) carries the node are marked not estimable: the odds ratio is infinite there, and statsmodels only flags *complete* separation, so the 2x2 cells are checked directly.
3. **Benjamini-Hochberg** runs once over all tested nodes. The family is fixed up front: counting only the nodes the rule below visits would make the family shrink exactly when a group looks significant, so groups that look good would face a milder correction.
4. **Break-up rule** (`select_reported`): each branch starts at its largest significant group. If that group's effect is significant it is reported and its subgroups are not; if it is n.s., invariant or not estimable, it is broken into its two children, down to single terms. The reported nodes partition the terms. The rule only decides what is *reported*, never how strict the correction is.

**Continuous vs discrete criteria.** A continuous criterion (e.g. reward rate) is one contrast over all participants, predictor = z(criterion); beta is the log-odds of presence per SD. A discrete criterion (e.g. diagnosis) is compared pair by pair — every group against `reference`, or the pairs in `comparisons` — each contrast over the participants of its two groups only, predictor = 1 for the group and 0 for the reference; beta is the log odds ratio of presence in the group vs the reference. **Each contrast is its own analysis**, with its own FDR family and its own break-up, so a group can hold in one contrast and break up in another; as in `analysis_coefficients_individuals`, there is no correction across contrasts. A discrete criterion is read as each participant's first value, whatever `criterion_aggregate` says.

Larger groups tend to break up for two reasons: members whose relation to the metric differs cancel out once all of them are required, and requiring every term shrinks the group toward its rarest member.

**Metric choice.** Term survival depends on data volume through pruning, so a criterion that scales with how much a participant played (e.g. rewards per *block*) produces spurious presence effects. The analysis does not adjust for trial count; it checks on every run — Pearson correlation with log trial count for a continuous criterion, Kruskal-Wallis of log trial count across groups for a discrete one — and warns when they are related. Use a metric that does not scale with trial count, such as reward rate (`criterion_col="reward"`, `criterion_aggregate="mean"`).

Every model here is cross-sectional across participants: a term whose presence predicts performance does not establish that adding it would change performance.

- **Outputs**: per contrast, `betas_structural_<prefix>_<criterion>.{csv,png}` for a continuous criterion and `betas_structural_<prefix>_<criterion>_<group>_vs_<reference>.{csv,png}` for each discrete contrast, so separate runs never overwrite each other. The CSV lists every node (contrast, group, parent, children, depth, all/any, beta, OR, p, q, status, `reported`, `broken_up`); the forest plot shows the reported, tested leaves (filled markers = significant after FDR).

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

`analysis_coefficient_compression.py`, `analysis_mechanism_individuals.py`, `analysis_concepts.py` and `analysis_coefficient_ties.py` have moved to `weinhardt2026/analysis/archive/`, together with the core modules they wrapped (`sindy_compression.py`, `sindy_concepts.py`, `sindy_ties.py`, now under `archive/core/`). They compressed per-participant coefficients into a small number of "mechanisms" via per-module NMF and re-tested group differences at the mechanism level. `analysis_coefficient_dendrogram.py` above covers the same question — which coefficients belong together — without a factorization step, and is used for interpretation only.

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
4. Relate coefficients to external criteria: `analysis_coefficients_individuals.py` (discrete groups or continuous traits), `analysis_coefficient_betas.py` for structural effects of co-occurring groups and single terms on a behavioral metric, and/or `analysis_morphing.py` for a continuous structural trajectory.
5. Generate synthetic behavior (`generate_behavior`) and validate it against real data with `analysis_generative_comparison.py` and `compute_reward_history_kernel`.
6. Read the fitted equations as a small number of co-occurring groups with `analysis_coefficient_dendrogram.py` (`mode="presence"`, `n_permutations=10000`), or check behavioral-cluster/equation alignment with `analysis_behavioral_clustering.py`.
7. For the selected checkpoint, run the four `analysis_bestbic.py` analyses (embedding extremes, fingerprint, group and metric betas) to describe how the equations vary across participants.

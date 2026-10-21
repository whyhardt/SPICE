# SPICE Analyses

Everything the paper reports is produced from the **final models** (`weinhardt2026/studies/<study>/params/`, ten seeds per model kind; `weinhardt2026/utils/final_models.py`) by the scripts below, and written to

- `weinhardt2026/results/<item>/` — results across studies, and
- `weinhardt2026/studies/<study>/results/<item>/` — results per study.

[`weinhardt2026/results/README.md`](../weinhardt2026/results/README.md) is the index of what lives where.

| Paper item | Script | Output |
|---|---|---|
| Model evaluation | `figures/figure_model_evaluation.py` | `results/model_evaluation/`; per study `results/model_evaluation/` |
| Parameter recovery | `analysis/analysis_parameter_recovery.py` | `results/parameter_recovery/` |
| Discovered mechanisms | `figures/figure_mechanisms.py --study <study>` | per study `results/mechanisms/` |
| Beta effects | `figures/figure_mechanisms.py --study <study>` | per study `results/beta_effects/` |
| Equation stability | `figures/figure_equation_stability.py all` | per study `results/equation_stability/` |
| Model identification | `confusion_matrix/` pipeline | `results/model_identification/` |
| Posterior predictive checks | `analysis/analysis_posterior_predictive.py all` | per study `results/posterior_predictive/` |
| Equation tables | `analysis/analysis_equation_tables.py all` | per study `results/equations/` |
| Morphing | `analysis/analysis_morphing.py <study>` | per study `results/morphing/` |
| Hyperparameter grid search | `analysis/analysis_grid_search.py`, `analysis/analysis_grid_mechanisms.py` | dezfouli2019 `results/grid_search/` |

Paths are relative to `weinhardt2026/`; scripts run from the repository root. The study run scripts (`studies/<study>/<study>.py`) only fit or load the models and print a quick model evaluation of the best seed; they write no results.

See [training.md](training.md) for the model/training internals these analyses consume, and [studies.md](studies.md) for the studies.

---

## Generative Benchmarking (`weinhardt2026/utils/task.py`)

Generative benchmarking simulates new behavioral data by running a fitted model through the original task environment. This produces synthetic datasets that can be compared against the original human data — the basis for the posterior predictive checks and the model-identification datasets below.

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
3. Resolve the model's device (`model.device` for a `SpiceEstimator`, else its parameters); the model is called as `model(obs, state)`
4. Draw the first action uniformly at random (the model has seen no trial yet)
5. For each trial `t` in `range(n_trials)`:
   - `reward, _ = environment.step(action_idx)` — environment gives reward for current action
   - Build observation: one-hot action + partial-feedback reward (NaN for unchosen) + metadata from original dataset
   - Forward pass: `rnn(obs, state)` for `BaseModel`, `rnn(obs, state)` otherwise
   - Normalize logits: 5D `(E,B,T,W,A)` → mean over ensemble → 4D → extract `(B, A)`
   - Sample next action: `multinomial(softmax(logits))`
6. Restore NaN padding for variable-length sessions (matching original dataset structure)

`use_sindy` (SPICE-EQ vs SPICE-RNN) is set by the caller. Wrappers that unwrap a `SpiceEstimator` (bruckner2025, weber2024) must not call `BaseModel.eval()` on it: its default `use_sindy=True` would silently turn every SPICE-RNN run into SPICE-EQ.

**Replay what the data record.** Where the CSV stores the outcome of every option on every trial, the environment replays it instead of simulating a reward process, so a model is tested on the trials the participant actually faced: ganesh2024a (`reward_left`/`reward_right`; the contrast–reward contingency changes per block and is not stored), eckstein2026 (`payout_1..4`, scaled like `normalize_rewards`), bruckner2025 (outcomes, change points) and weber2024 (laser trajectories). dezfouli2019, braun2018 and bustamante2023 simulate the task's reward process (dezfouli2019: Bernoulli rewards with each session's own probabilities, `p_reward_0`/`p_reward_1` from the block condition of the authors' raw data, see [studies.md](studies.md#dezfouli2019--two-armed-bandit-in-depression--bipolar-disorder); the block number does not determine them). kolff2025 replays the partner's acts and the timing of the focal ape's acts and is closed-loop only on the modelled decision (groom or not; the non-grooming act is the real one or, where the real act was grooming, drawn from the observed non-grooming acts).

**Result**: a synthetic `SpiceDataset` with the same shape/metadata as the input, but actions (and derived rewards) drawn from the fitted model instead of the human participant. The posterior predictive checks and the confusion matrix reach it through each study's `simulation_<study>.py` (see [Posterior Predictive Checks](#posterior-predictive-checks--analysis_posterior_predictivepy)).

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

## Paper Analyses (`weinhardt2026/analysis/`, `weinhardt2026/figures/`)

Shared loading: `utils/final_models.py` — `final_checkpoints(study, kind)`, `best_checkpoint(study, kind)` (highest training likelihood; SPICE: of its best member), `load_best`, `model_setup(study, kind)` (path → model loader and training data, kolff2025 included) and `seed_matrices(study)` (per final SPICE seed and module: coefficients, presence = ensemble majority, terms; cached in `results/mechanisms/matrices.pkl`, kolff2025 in `results/equation_stability/`). `utils/study_models.py` — simulation modules (`load_study_module`), `EnsembleMember`, per-member training likelihoods (`member_train_loglik`, `select_spice_members`). `utils/checkpoints.py` loads a single checkpoint without the training data (`load_estimator(model_path, spice_class, spice_config, n_actions=2, n_participants=None, polynomial_degree=2, model_kwargs=None)`).

### Model Evaluation — `analysis_model_evaluation.py`

Quantifies how well a fitted SPICE model (RNN and SINDy-equation form) fits the training data and predicts held-out behavior, and compares against a hand-coded benchmark model and/or a plain GRU baseline. Both splits are evaluated in one call and reported in one table.

```python
analysis_model_evaluation(
    dataset_train, dataset_test, spice_model=None, benchmark_model=None, gru_model=None,
    spice_selection='best-on-train', verbose=False, output_dir=None, trial_filter=None,
    benchmark_group_model=False,
)
```

- **Training split**: per-trial likelihood, NLL, **AIC** and **BIC**. **Test split**: per-trial likelihood and NLL only — held-out likelihood already measures generalization, so an information criterion on it would charge for parsimony twice.
- **Number of parameters** per model: `n_parameters` (mean ± std over the training split's (participant, experiment) groups; SPICE-EQ counts the active coefficients) and `n_parameters (total)`, the parameter count of the joint model.
- AIC/BIC treat the **whole training split as one model** (`joint_information_criteria`): the NLL is summed over all N trials of all participants, and K = `n_parameters (total)` counts every participant's own parameters — per-participant parameters summed over participants (SPICE-EQ: over its distinct (participant, experiment) coefficient sets; benchmark: `count_parameters()` × participants), shared parameters once (GRU, SPICE-RNN, a SPICE group model with `n_participants=1`, or the benchmark with `benchmark_group_model=True`). AIC = 2 NLL + 2K, BIC = 2 NLL + K log N.
- AIC/BIC are reported only for the interpretable models (Benchmark, SPICE-EQ); GRU and SPICE-RNN get NaN (printed as `-`).
- `spice_selection` sets how the SPICE ensemble members enter the scores (SPICE-EQ and SPICE-RNN separately):
  - `'best-on-train'` (default, used by every study): each member is scored on its own; the member with the highest training likelihood is reported, on train **and** test.
  - `'averaged-model'`: each member is scored on its own (its own logits and coefficient count); the scores are averaged over members.
  - `'averaged-logit'`: the members' logits are averaged at every time step and that averaged model is scored (SPICE-EQ counts the terms active in any member).
- When SPICE and the benchmark are both evaluated, the benchmark's row reports a per-participant sign test against SPICE-EQ for each split (`Train …` / `Test …` columns, `participant_win_counts`).
- `trial_filter` lets you restrict evaluation to a subset of trials (applied to both splits, e.g. excluding "wait" responses in a DDM-style task).
- The printed table is transposed (models as columns), numbers rounded to 3 decimals (integers without decimals), NaN as `-`; the returned DataFrame and the CSV keep full precision.
- `analysis_model_evaluation_mse` — parallel pipeline for continuous-output (regression) models with the same `dataset_train`/`dataset_test`/`spice_selection` interface: MSE, RMSE, MAE, R² on both splits, AIC/BIC on the training split from a joint Gaussian likelihood (one MLE variance over all trials; same parameter counting; Benchmark and SPICE-EQ only); `best-on-train` selects the member with the lowest training MSE. Writes `model_evaluation_mse.csv` and a plot of the test metrics.
- **Result**: a DataFrame (saved as `model_evaluation.csv` in `output_dir`) with one row per model (Benchmark, GRU, SPICE-RNN, SPICE-EQ) and columns `n_parameters`, `n_parameters (total)`, `Train Trial Lik.`, `Train NLL`, `AIC`, `BIC`, `Test Trial Lik.`, `Test NLL` (plus stds) — the standard model-comparison table used to argue SPICE fits at least as well as hand-crafted models.

### Model Comparison Figure — `weinhardt2026/figures/figure_model_evaluation.py`

The paper's model comparison: one column per study (discrete studies first, kolff2025 included, then bruckner2025 and weber2024), one bar per model (GRU, Benchmark, SPICE-EQ, SPICE-RNN; bar = mean over the final seeds of `weinhardt2026/utils/final_models.py`, ± SD, white dots = seeds). Data and SPICE estimators are rebuilt as `run.py` built them (`STUDIES` registry), so every model is scored on the split the SPICE checkpoints were trained on.

```bash
python weinhardt2026/figures/figure_model_evaluation.py              # cached evaluation + plot
python weinhardt2026/figures/figure_model_evaluation.py --recompute  # re-evaluate (new checkpoints)
```

Rows:
1. **Held-out prediction** (test blocks): discrete → average trial likelihood exp(−NLL / n_trials); continuous → one-step MSE per trial in the response's units (bruckner2025: bucket position, px²; weber2024: angular shield error, deg², with the movement-speed clamping of `spice_weber2024`).
2. **Posterior predictive check**: read per (model, seed) from the study's `results/posterior_predictive/ppc_scores_instances.csv` (see [Posterior Predictive Checks](#posterior-predictive-checks--analysis_posterior_predictivepy)); discrete → nRMSE, continuous → rollout MSE. Read at plot time, so re-running the checks only needs a re-plot, not `--recompute`.
3. **AIC / trial** and 4. **BIC / trial** (training blocks, Benchmark and SPICE-EQ only): per (participant, experiment) group, divided by the group's trials, averaged over groups (`grouped_information_criteria`: `aic_per_trial_mean`, `bic_per_trial_mean`). SPICE-EQ counts each group's active coefficients; a SPICE model with one experiment slot (weber2024, `n_experiments=1`) shares it across experiments. Continuous studies use a Gaussian likelihood with per-group MLE variance.

Within a row, the columns of one kind share the y-axis span (the largest range among them), so equal differences look equal: all discrete studies in every row, the continuous studies only in AIC/BIC (their MSEs differ in unit; `SHARED_SPAN`). Bars rise from the bottom of the zoomed axis, not from 0.

**kolff2025** is not in the `STUDIES` registry (`GROUP_STUDIES`, `evaluate_group_study`): its models are group models (one parameter set for all apes), scored only where the ape acts next on the held-out dyads of fold 0; AIC/BIC per trial on the pooled training data, since the parameters are shared. The benchmark (logistic regression) is fit in closed form, so it is one fit (seed 0).

`spice_estimator(spice_module, dataset_train, **overrides)` builds a study's `SpiceEstimator` as `run.py` does (used by `load_spice`, and by the morphing analysis to build estimators with other participant counts).

**Outputs**: `weinhardt2026/results/model_evaluation/` — `figure_model_evaluation.png/.pdf`, `model_evaluation_summary.csv` (mean and SD over seeds, including `ppc`); per study `studies/<study>/results/model_evaluation/model_evaluation.csv` (per model and seed; the cache, re-used unless `--recompute`).

### Parameter Recovery — `analysis_parameter_recovery.py`

Synthetic-data check: Q-learning ground-truth models (`studies/synthetic/`, `benchmarking_qlearning.QLearning`) with every combination of the mechanisms Reward, Asymmetry, Forgetting and Choice simulate datasets of 32–512 participants (8 datasets per size, `utils/generate_synthetic_datasets.py`); SPICE is fit to each with 10 seeds (`slurm_jobs/parameter_recovery.sh`). A term counts as recovered if it is active in more than half of the seeds; binary signals are collapsed (`signal^1 == signal^2`). Writes to `weinhardt2026/results/parameter_recovery/`: `*_metrics_by_mechanism.csv` (per dataset size × ground-truth model: n_fits, TP/TN/FP/FN rates, accuracy, **precision, recall, F1**, F2), `*_metrics.npz` (also the metrics binned by number of parameters, and true vs. fitted coefficients), figures of the classification metrics (overall and by ground-truth model), confusion rates, per-term recovery and true vs. fitted coefficients per dataset size (each also as `_clean` without annotations).

### Discovered Mechanisms — `figures/figure_mechanisms.py`

```bash
python weinhardt2026/figures/figure_mechanisms.py --study dezfouli2019
```

Per final SPICE seed, the presence dendrogram of `analysis_coefficient_dendrogram` (terms split by sign, distance 1 − phi, per-module permutation test, FWER) gives that seed's significant groups; `analysis_grid_mechanisms.core_mechanisms` keeps, per family of overlapping groups, the largest term set that at least `VOTES = 5` of the 10 seeds share. Figure, one panel per module: consensus tree (average linkage on the seed-averaged 1 − phi distance of the terms kept in ≥ `VOTES` seeds; modules with fewer than two such terms have no tree), recurring groups coloured with their merge distance per seed (mean ± SD) and the per-seed critical distance at p = .05. Not for kolff2025 (one group model: no individual differences to cluster).

**Outputs** in `studies/<study>/results/mechanisms/`: `figure_<study>_mechanisms.png/.pdf`, `core_mechanisms_distances_votes5.csv` (group, support in seeds, merge distances), caches `matrices.pkl` and `seed_clusters.pkl`. dezfouli2019 additionally has `trajectories/` (`figures/figure_dezfouli2019_trajectories.py`: value trajectories and equations of three participants chosen in embedding space).

### Beta Effects — `figures/figure_mechanisms.py`

The same run tests the structural beta effects of the recurring mechanisms plus all other signed terms (`analysis_coefficient_betas.analysis_mechanism_betas`, presence = carried in ≥ `VOTES` seeds) for the criteria in `BETA_CRITERIA`, on the study's data with its reward column (`utils/participant_metrics.study_data`, `REWARD_COLUMNS`; reward rate = that column averaged over the participant's trials, or summed per block and averaged over blocks):

| Study | Criterion |
|---|---|
| dezfouli2019 | reward rate; diagnosis (Depression vs Control, Bipolar vs Control, Bipolar vs Depression) |
| ganesh2024a, eckstein2026 | reward rate (`reward`) |
| bustamante2023 | reward per second: every round is a fixed 240 s window, so round total / 240 s, averaged over rounds (travel time and missed responses lower it) |
| braun2018 | RT switch cost: mean RT on switch minus repeat trials (RTs 200–3000 ms); a participant who never switches has none and is left out |
| weber2024 | reward rate = catch rate (`laser_caught`) |
| bruckner2025 | reward rate (`trial_reward` = coin value `r_t` if caught); age group, every pair of children (code 1, 7–11 y), younger adults (3, 20–28 y) and older adults (4, 61–76 y) |

`--merged` (opt-in) adds the merged-group contrasts of `MERGED_CRITERIA` (`criterion_map` maps several codes onto one group): dezfouli2019 Patients (Depression + Bipolar) vs Control, bruckner2025 adults (younger + older) vs children.

```bash
python weinhardt2026/figures/figure_mechanisms.py --study dezfouli2019 --merged
```

A continuous criterion that is undefined (NaN) for a participant leaves that participant out of the contrast. The study-specific criteria are defined in `utils/participant_metrics.py`.

**braun2018 reward rate** (morphing default; switch rate was set aside as a criterion as circular — terms that drive switching are necessarily more common in participants who switch more). The points are not in the CSV, but a block ends when they reach 500 (reconstructed from the task values — both start at 5 per block, `current`/`other` flag the ±1 changes caused by the previous choice, `difference` is the display before the choice — the gap-free blocks total 495–508 points; participants are assumed correct on every trial). So the block's points are the target, and the rate is 500 / the block's summed response times (all trials; excluded trials, ~11% of the rows, at the block's mean RT), averaged over the participant's blocks. Summing RTs instead of using the block duration leaves out the response-stimulus interval (200 or 1100 ms, 6 blocks each).

`get_criterion` takes the data as a CSV path or DataFrame and, with `block_col`, sums the criterion per block before averaging over blocks. **Outputs** in `studies/<study>/results/beta_effects/`: `betas_structural_<study>_mechanisms_votes5_<criterion>[_<group>_vs_<reference>].csv` (every node) and `.png` (significant reported leaves only; no plot when none is significant).

### Equation Stability — `figures/figure_equation_stability.py`

```bash
python weinhardt2026/figures/figure_equation_stability.py all
```

How reproducibly the final seeds identify the same equations (all eight studies). Per module and term: **agreement** = mean over participants of max(f, 1 − f), f = share of seeds in which the participant carries the term (1 = all seeds agree, 0.5 = maximal disagreement); next to it, per seed, the mean ± SD across participants of the term's coefficient (0 where absent). Terms no seed ever selects are omitted from the figure. **Outputs** in `studies/<study>/results/equation_stability/`: `equation_stability.png/.pdf`, `equation_stability.csv` (agreement, presence rate, coefficient mean and SD over seeds).

### Posterior Predictive Checks — `analysis_posterior_predictive.py`

One script for the whole check: simulates every final model in the study's task, summarizes the simulated and the real behavior with study-specific statistics, scores each model, and saves figures and numbers per study. The model comparison figure reads its overall scores (second row).

```bash
python weinhardt2026/analysis/analysis_posterior_predictive.py all
python weinhardt2026/analysis/analysis_posterior_predictive.py braun2018 dezfouli2019 [--n_runs 100] [--recompute]
```

Simulation and loading come from the study's `studies/<study>/simulation_<study>.py` (`load_dataset`, `load_models`, `generate`; interface and shared helpers in `weinhardt2026/utils/study_models.py`); all blocks are simulated. Models: every final seed of GRU, Benchmark and SPICE (per seed the ensemble member with the highest training likelihood, simulated as SPICE-EQ and SPICE-RNN); a single fit without seeds (kolff2025's logistic-regression benchmark) is one instance.

**Discrete studies** (dezfouli2019, ganesh2024a, eckstein2026, bustamante2023, braun2018, kolff2025) — every instance runs the experiment `N_RUNS = 100` times, run r with `torch.manual_seed(r)` (the same seeds for every model). Every statistic is a ratio of sums over a participant's sessions (e.g. switches / decisions in a cell); a conditional probability needs `MIN_COUNT = 5` decisions of the participant. Statistics are grouped into panels: a single statistic, or a grid of cells (bins from the real data, capturing `COVERAGE = 95%` of the decisions; integer counts binned one per value up to 4, log-spaced above; saved to `ppc_bins.json`).

| Study | Panels |
|---|---|
| braun2018 | P(Switch \| point difference × repeats of the current task); P(Switch \| current constant/decrease × other constant/increase). The inputs of trial t condition the decision at t + 1. |
| bustamante2023 | P(Exit \| harvests in the patch × last reward, 5 exit-threshold bins); mean exit reward; reward rate (reward / time, harvest and travel durations) |
| dezfouli2019, ganesh2024a, eckstein2026 | P(Best Arm); P(Switch \| Lose); P(Stay \| Win); reward rate (reward / trial); learning curve (P(Best Arm) in 5 bins of block progress). Best arm: dezfouli2019 the session's higher reward probability, eckstein2026 the highest payout on the trial, ganesh2024a the side the block's contingency favors. Win = reward > 0.5. |
| eckstein2026 | + cyclic sequences per block (4 consecutive choices visiting all arms in one rotational direction) |
| ganesh2024a | + P(Best Arm \| \|contrast difference\|, 5 quantile bins) |
| kolff2025 | P(Groom) overall and after partner solicitation / preparation / grooming and own grooming; own acts before the first groom (group models, see `studies/kolff2025/simulation_kolff2025.py`) |

Score: **nRMSE** per statistic and instance = RMSE across participants between the real value and the instance's value pooled over runs, divided by SD(real) (≥ 10 participants). It is paired, so it penalizes both a shifted population and scrambled individuals; even the true model stays above 0 (real values are finite-trial estimates), and a model without individual differences lands near 1. A grid's score is the mean over its cells, the instance's overall score the mean over panels (`panel = all`); reported as mean and SD over instances.

**Continuous studies** (bruckner2025, weber2024) — the models predict deterministically, so every instance makes one autonomous rollout (its predictions are its responses); score: per-trial MSE to the real responses (bucket position, px²; angular shield error, deg²), the same units as the figure's one-step MSE.

**Outputs** in `studies/<study>/results/posterior_predictive/`:
- `ppc_scalars.png/.pdf` (single statistics) and `ppc_<panel>.png/.pdf` per grid (one small panel per cell): violins in the order GRU, Benchmark, Real, SPICE-EQ, SPICE-RNN over participants × runs × instances (Real: participants); participants as dots (models: pooled over instances and runs) for Benchmark, Real, SPICE-EQ and SPICE-RNN, joined in that order; each model's nRMSE against Real above its violin — for grids in every cell, with the mean over cells per model in the title. Continuous: `ppc_rollout.png/.pdf`, per-participant rollout MSE.
- `ppc_scores_instances.csv` (per model, instance, seed, panel, statistic), `ppc_scores.csv` (mean and SD over instances), `ppc_participants.csv` (per model, participant and statistic), `ppc_bins.json`.
- `cache_runs<N>_bins<hash>/` (simulated counts per model and instance) and `cache_rollout.csv`; re-used unless `--recompute`. The bins are part of the cache key, so changing them re-simulates.

Runtime: about 1 s per simulated run, i.e. ~45 min per discrete study with 100 runs × 10 seeds × 4 models on a laptop GPU.

### Equation Tables — `analysis_equation_tables.py`

```bash
python weinhardt2026/analysis/analysis_equation_tables.py all
```

**Outputs** in `studies/<study>/results/equations/`:
- `term_presence.csv` — per (participant, module, term): `presence` = share of the final seeds in which the participant carries the term (within a seed: ensemble majority), `present` = presence ≥ 0.5, with the participant's ID from the data.
- `example_participants.csv` / `.txt` — the equations of three participants chosen in the RNN's participant-embedding space: the two furthest apart and the most average one (closest to the centroid); distances averaged over every seed and ensemble member, each distance matrix normalized by its mean. Coefficients of the best final seed's best member; the module's own state is reported as its full coefficient (+1, stored as deviation from identity). kolff2025: the single group equation.

### Model Morphing — `analysis_morphing.py`

```bash
python weinhardt2026/analysis/analysis_morphing.py dezfouli2019                      # reward rate
python weinhardt2026/analysis/analysis_morphing.py braun2018 --metric rt --participant_col subject
```

Traces how the equations change along a continuous behavioral axis, for any study with participant embeddings (not kolff2025). From the best final seed: per ensemble member (each has its own embedding space), the direction in embedding space that best predicts the metric (linear regression; metric: the participant's reward rate by default, `utils/participant_metrics.reward_rate`, or `--metric` — any data column averaged per participant); every participant's embedding moved to `N_STEPS = 20` positions along it (projection mean ± 1 SD), giving P × 20 virtual participants that keep their real training data — all members in one estimator, the RNN frozen. The equations of the virtual participants are then refit with **Stage 2 of training** (`spice_training._run_sindy_training`: sparsity discovery and coefficient estimation with multi-step shooting, which optimizes the equations for autonomous rollouts) and the module cleanup, with the final models' hyperparameters (pruning threshold, ensemble ratio and alpha from the checkpoint name, `run.py` defaults otherwise). Per step and member the mean coefficient and inclusion probability over participants; aggregated over members (mean ± SE). For discrete studies with rewards, the morphed SPICE-EQ is simulated in the task (`simulation_<study>.generate`; `VirtualParticipants` lets the environment keep the real participant IDs) and the mean reward per step reported. The refit costs about a training run's Stage 2 on P × 20 participants: run it on the cluster (`--n_steps`, `--epochs` shrink it for a test). **Outputs** in `studies/<study>/results/morphing/`: `morphing_<metric>.csv`, `morphing_<metric>.png/.pdf`, `morphing_<metric>_validation.csv` (`<metric>` = `reward_rate` by default).

On the cluster: `sbatch --export=ALL,STUDY=<study>[,METRIC=<column>,PARTICIPANT_COL=<column>] slurm_jobs/morphing.sh`.

### Coefficient Clustering (library) — `analysis_coefficient_dendrogram.py`

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

**Continuous vs discrete criteria.** A continuous criterion (e.g. reward rate) is one contrast over all participants, predictor = z(criterion); beta is the log-odds of presence per SD. A discrete criterion (e.g. diagnosis) is compared pair by pair — every group against `reference`, or the pairs in `comparisons` — each contrast over the participants of its two groups only, predictor = 1 for the group and 0 for the reference; beta is the log odds ratio of presence in the group vs the reference. **Each contrast is its own analysis**, with its own FDR family and its own break-up, so a group can hold in one contrast and break up in another; there is no correction across contrasts. A discrete criterion is read as each participant's first value, whatever `criterion_aggregate` says.

Larger groups tend to break up for two reasons: members whose relation to the metric differs cancel out once all of them are required, and requiring every term shrinks the group toward its rarest member.

**Metric choice.** Term survival depends on data volume through pruning, so a criterion that scales with how much a participant played (e.g. rewards per *block*) produces spurious presence effects. The analysis does not adjust for trial count; it checks on every run — Pearson correlation with log trial count for a continuous criterion, Kruskal-Wallis of log trial count across groups for a discrete one — and warns when they are related. Use a metric that does not scale with trial count, such as reward rate (`criterion_col="reward"`, `criterion_aggregate="mean"`).

Every model here is cross-sectional across participants: a term whose presence predicts performance does not establish that adding it would change performance.

**Mechanism betas** (`analysis_mechanism_betas`, used by `figure_mechanisms.py`): the leaves are the mechanisms recurring across the final seeds plus every other signed term; a participant carries a node if at least `votes` seeds give it to them. `criterion_map` relabels the criterion before the contrasts (names groups, or merges them by mapping several codes onto one label).

- **Outputs**: per contrast, `betas_structural_<prefix>_<criterion>.{csv,png}` for a continuous criterion and `betas_structural_<prefix>_<criterion>_<group>_vs_<reference>.{csv,png}` for each discrete contrast, so separate runs never overwrite each other. The CSV lists every node (contrast, group, parent, children, depth, all/any, beta, OR, p, q, status, `reported`, `broken_up`); the forest plot shows the reported, tested leaves (filled markers = significant after FDR).

### Model Identification (Confusion Matrix) — `weinhardt2026/confusion_matrix/`

Checks that SPICE and the benchmark each identify data simulated by themselves: both simulate one dataset per study, both are refit to both datasets with new seeds (10–19; the final models use 0–9), and the winner per participant gives a 2×2 confusion matrix per study. Laid out like `benchmark_grid_search/` (scripts, `data/`, `runs/<study>/`); all eight studies, kolff2025 included.

1. **Simulate** — `python weinhardt2026/confusion_matrix/generate_datasets.py [--studies ...]`. Per study and generator, the lower-median seed by training likelihood (SPICE: each seed's best ensemble member; simulated as SPICE-EQ) generates one dataset with the study's `simulation_<study>.generate`. kolff2025's benchmark is its single closed-form fit. Continuous studies (bruckner2025, weber2024) simulate from the Gaussian likelihood the models are scored with: prediction + N(0, σ²), σ the generator's MLE residual s.d. per participant on the training data, passed as `noise_std` to the study's `generate_behavior` and added inside the loop (weber2024: angular noise after the speed clamp; the benchmark responds with its expectation instead of a drawn move). Each dataset is the study's CSV with the simulated columns overwritten in raw units (`SIMULATED_COLUMNS`), so both models are fit exactly as on the real data; a round trip through the study's loading pipeline must reproduce the simulation. Writes `data/<study>_sim<generator>.csv`, `data/generators.csv` (generator and refit configuration per dataset) and `data/selection.csv`.
2. **Fit SPICE on the cluster** — `sbatch slurm_jobs/confusion_matrix.sh` (one array task per study × generator × seed, `run.py` with the study's final grid-search hyperparameters; `DATA_DIR` / `OUT_DIR` redirect a rerun to other folders).
3. **Fit the benchmarks locally** — `python weinhardt2026/confusion_matrix/fit_benchmarks.py` (`fit_confusion_matrix.fit_run` for every study × generator × seed through the setups of `benchmark_grid_search.py`, which take the simulated CSV as `path_data`; kolff2025: one closed-form fit). Fits land in `runs/<study>/<fitter>_<study>_sim<generator>_rs<seed>.{pkl,json}`; existing ones are skipped.
4. **Confusion matrices** — `python weinhardt2026/confusion_matrix/analysis_confusion_matrix.py [--cached]`. Every fit is scored on its dataset (SPICE: per seed the member with the highest SPICE-EQ training likelihood) by held-out log-likelihood per trial on the test blocks (also for SPICE-RNN), and AIC and BIC on the training blocks for the benchmark and SPICE-EQ (parameters per participant; kolff2025's group models on the pooled data). Per participant, the model with the better score averaged over seeds wins; the spread pairs the k-th SPICE with the k-th benchmark seed.

**Outputs** in `weinhardt2026/results/model_identification/`: `confusion_matrix_{loglik,aic,bic}.csv` (rows study × simulating model, columns fitted model: P(wins) and its SD over seed pairs), `figure_confusion_matrix_{loglik,aic,bic}.png/.pdf` (one 2×2 matrix per study, diagonal = correct identification), `confusion_matrix.csv` (long format, with score differences and the inversion matrix P(generator | winner)) and `confusion_scores.csv` (per fit and participant; `--cached` re-aggregates it).

Simulating a stateful model trial by trial must give the same predictions as one pass over the session — the bruckner2025 benchmark ignored its state until 2026-10-17, and both its generative comparison and its confusion dataset had to be redone.

### Hyperparameter Grid Search — `analysis_grid_search.py`, `analysis_grid_mechanisms.py`

The SPICE grid search (`slurm_jobs/grid_search.sh`, dezfouli2019): `analysis_grid_search.py` scores every checkpoint (in-sample BIC/AIC and trial likelihood, held-out trial likelihood) and dissects the cell table (variance decomposition per hyperparameter, sparsity mediation); `analysis_grid_mechanisms.py` measures mechanism recovery across seeds in every cell. Outputs in `studies/dezfouli2019/results/grid_search/`. The benchmark and GRU grid searches live in `weinhardt2026/benchmark_grid_search/`.

### Checkpoint Scoring (library) — `analysis_sparsity_hpscan.py`

```python
analysis_sparsity_hpscan(
    pkl_pattern, spice_class, spice_config, n_actions, data_path, test_blocks,
    polynomial_degree=2, model_kwargs=None, device=None,
) -> pd.DataFrame
```

Evaluates a batch of checkpoints from a pruning-threshold × pruning-test hyperparameter sweep (glob-matched `.pkl` files, e.g. `params_array/spice_dezfouli2019_*_*.pkl`) on held-out `test_blocks`, returning a summary table of predictive performance vs. sparsity level — its `evaluate_checkpoint` / `prepare_split` score every checkpoint of the grid search below.

---

### One-Step-Ahead Diagnosis — `analysis_sindy_onestepahead.py`

Called by `weinhardt2026/run.py` after every fit. Diagnoses *why* the fitted SINDy equations underperform the RNN at test time: per-step approximation error vs. autoregressive error accumulation.

```python
analysis_sindy_onestepahead(dataset, spice_model) -> (summary_df, per_trial_df)
```

Compares three evaluation modes: **SPICE-RNN** (autoregressive, ensemble mean), **SPICE-SINDy (autoregressive)** (equations feed their own past predictions forward), **SPICE-SINDy (one-step-ahead)** (equations receive the RNN's true state at each trial). If one-step-ahead ≈ RNN but autoregressive ≪ RNN → the equations are locally accurate but their errors compound over trials. If one-step-ahead ≪ RNN → the equations poorly approximate the RNN even locally.

### Retired analyses

Removed on 2026-10-21 and kept only in the git history: coefficient distributions, individual-differences regression on single checkpoints, behavioral clustering, reward-history kernel, the lowest-BIC deep dive, the stage-2.2 behavioral refit, the earlier generative comparison, coefficient compression/concepts/ties, the prototype figure scripts (figures 2–5) and the rare-event prediction (replaced by the posterior predictive checks).

---

## Typical Analysis Sequence for a New Study

1. Fit the final models (grid search, `slurm_jobs/grid_search.sh`; benchmark and GRU: `benchmark_grid_search/`) into `studies/<study>/params/`; add the study to the `STUDIES` registry of `figures/figure_model_evaluation.py` and write `studies/<study>/simulation_<study>.py`.
2. Model evaluation: `figures/figure_model_evaluation.py --studies <study>`.
3. Posterior predictive checks: add the study's statistics to `STUDY_SPECS` in `analysis_posterior_predictive.py`, run it, re-plot the model evaluation figure.
4. Mechanisms and beta effects: add the criteria to `BETA_CRITERIA` and run `figures/figure_mechanisms.py --study <study>`.
5. Equation stability and equation tables: `figures/figure_equation_stability.py <study>`, `analysis/analysis_equation_tables.py <study>`; optionally `analysis/analysis_morphing.py <study>`.
6. Model identification: the confusion-matrix pipeline (`weinhardt2026/confusion_matrix/`).

# SPICE Studies

`weinhardt2026/studies/` holds self-contained per-study directories: each combines a task/data-loading module, a SPICE model config (`spice_<study>.py`), a hand-coded benchmark model, and a `benchmarking_<study>.py` file wiring generative benchmarking (see [analyses.md](analyses.md)). Every study directory typically has `data/`, `params/`, `results/`, and either a notebook or a `<study>.py` driver script.

Conventions and infrastructure shared across studies are described in [training.md](training.md) (model/config/estimator internals) and [analyses.md](analyses.md) (evaluation, morphing, coefficient statistics, generative comparison). This file covers what's specific to each study: the task paradigm, population, and benchmark model.

---

## Active Studies

### `braun2018` — Cognitive-control / task-switching effort
Participants choose which of two tasks to perform on each trial (`df_choice='transcode'`) under a reward and switch/repeat structure. SPICE models separate reward, control, and fatigue value modules split by repeat vs. switch trials, plus a fatigue module driven by block number (control becomes more costly the longer it is exerted). Effort-based task selection and switch costs.

### `bruckner2025` — Helicopter / predictive-tracking task
Participants track a hidden target position (helicopter/coin drop) and adjust a "bucket" under changing volatility and stochasticity. Models changepoint detection and uncertainty-driven adaptive learning rates, plus an anchoring-bias correction module for bucket displacement between trials. Benchmarked against a `RationalResourceModel` (bounded-rational, resource-rational learning-rate model).

### `bustamante2023` — Patch-foraging task
Participants decide to harvest a depleting resource patch or exit to a new one. SPICE models reward/depletion tracking per patch state plus a "continuation" (stay/leave) module. Benchmarked against a Marginal Value Theorem model (`MarginalValueTheoremModel`) — the classic optimal-foraging economics baseline.

### `dezfouli2019` — Working-memory-mediated bandit learning
Bandit-style value-learning task modeled with working-memory buffers: reward/choice history over 3 lagged timesteps (t-1..t-3) feeds separate chosen/not-chosen value modules via `spice.precoded.workingmemory`. Benchmarked against a `GQLModel` (generalized Q-learning). Models multi-timescale reward learning with short-term dependence on recent reward/choice sequences.

### `eckstein2026` — Directed exploration in reward learning
Reward-learning bandit task modeling both value learning (per-environment/chosen/unchosen reward tracking with a running-mean baseline) and directed exploration (separate positive/negative value-difference "exploration" modules), plus choice perseveration and spatial-attention biases (adjacent/opposite action bias). Benchmarked against `Castro2025Model` and `RWForgettingChoiceModel` (Rescorla-Wagner with forgetting). Associated with the "MindRL Challenge 2026" (`mindrl_challenge_2026.ipynb`).

### `ganesh2024a` — Confidence-weighted reward learning
Perceptual decision-making combined with reward learning: a `perception_certainty` module maps signed contrast difference to a confidence/certainty signal (sigmoid), gating separate chosen/unchosen reward-learning and choice-persistence value modules. Benchmarked against a `BayesianModel`. Demonstrates the items-vs-actions decoupling (see [training.md](training.md#dimension-conventions)).

### `groman2018` — *Planned, not yet implemented*
Reference material only (`groman2018.pdf` — Groman et al., *Neuropsychopharmacology* 2018, on reinforcement-based decision making in rats under chronic methamphetamine exposure). No `groman2018.py`, `spice_groman2018.py`, or `benchmarking_groman2018.py` exists yet; `data/`, `params/`, `results/`, `figures/` are placeholders.

### `huang2026` — Collaborative visuospatial foraging
A collaborative tile game between a self participant and a partner. Models tile-value dynamics from information loss (decay when unvisited), information gain (decrement on visiting), recency of self/partner visits, choice stickiness, and movement perseverance. Benchmarked against a 7-parameter `InformationForagingModel`.

### `kolff2025` — Primate social behavior
Models a 5-action repertoire (action/grooming/gesture/scratching/waiting) as a function of a partner's simultaneous behavior and dominance-rank difference (`rank_diff`), with separate per-dimension "partner influence" and "own-action transition" modules plus persistence (repeat/switch) modules. Additional inputs include per-individual dominance rank. Benchmarked against a `ConditionalFrequencyModel`.

### `weber2024` — Belief updating under volatility (angular state space)
A laser-shield tracking/belief-updating task ("catch"/"miss" a laser with a shield) under volatility and stochasticity: changepoint + relative-uncertainty learning-rate modules gate a circular belief-position update via sin/cos, plus a dynamic learning-rate state. Same predictive-inference family as `bruckner2025` (Nassar-style changepoint/volatility belief updating), but with an angular (not linear) state space — evaluated with circular/angular MSE loss (`clamped_angular_mse`). Benchmarked against a `ChangePointModel`.

---

## `synthetic` — Parameter Recovery Testbed

Not a real-data study. Contains `benchmarking_qlearning.py` and `synthetic_<Np>p_<seed>_<idx>.csv` datasets simulating Q-learning agents at varying population sizes (32/64/128/256/512 participants). Used to validate that SPICE recovers known ground-truth cognitive parameters/dynamics — see `analysis_parameter_recovery.py` in [analyses.md](analyses.md).

---

## `archive` — Inactive Studies

`augustat2025`, `eckstein2022`, `rtify2024`, `sugawara2021`, `ting2023` — earlier or superseded study directories, kept for reference but not maintained against the current SPICE API.

---

## Adding a New Study

1. Create `weinhardt2026/studies/<study>/` with `data/`, `params/`, `results/`.
2. Write a data-loading module (`get_dataset()`) using `csv_to_dataset()` (see [training.md](training.md#data-pipeline)).
3. Define `spice_<study>.py`: a `BaseModel` subclass + `SpiceConfig` following the [precoded model pattern](training.md#precoded-model-pattern), applying the [polynomial-amenable architecture guidelines](training.md#designing-polynomial-amenable-architectures).
4. Write a hand-coded benchmark model (for `analysis_model_evaluation.py` comparison) and an `Environment<Study>(Env)` + `generate_behavior()` wrapper (for generative benchmarking — see [analyses.md](analyses.md#generative-benchmarking-weinhardt2026utilstaskpy)).
5. Fit with `SpiceEstimator`, then run the [typical analysis sequence](analyses.md#typical-analysis-sequence-for-a-new-study).

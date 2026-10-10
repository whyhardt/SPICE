# SPICE Studies

`weinhardt2026/studies/` holds self-contained per-study directories: each combines a task/data-loading module, a SPICE model config (`spice_<study>.py`), a hand-coded benchmark model, and a `benchmarking_<study>.py` file wiring generative benchmarking (see [analyses.md](analyses.md)). Every study directory typically has `data/`, `params/`, `results/`, and either a notebook or a `<study>.py` driver script.

Conventions and infrastructure shared across studies are described in [training.md](training.md) (model/config/estimator internals) and [analyses.md](analyses.md) (evaluation, morphing, coefficient statistics, generative comparison). This file covers what's specific to each study: the task paradigm, population, and benchmark model.

---

## Active Studies

The task battery of the paper. Full architectures, benchmark equations and dataset details: [model_descriptions.md](model_descriptions.md). The study-to-model mapping used for the paper runs is in `slurm_jobs/spice_stability_studies.sh`.

**Shared RL architecture.** The bandit-style studies (`synthetic`, `dezfouli2019`, `eckstein2026`, `ganesh2024a`, `bustamante2023`) share one architecture of up to three value states, summed into the logits:
- a working-memory reward value (stateless chosen update, decaying unchosen update);
- an incremental reward value (chosen / not-chosen modules);
- a single merged choice-perseveration module that receives the choice indicator.

Each study only adds its task-specific deviations.

### `braun2018` — Cognitive-control / task-switching effort
Reward-based voluntary task switching. On each trial, participants choose which of two tasks to perform, and each task carries a point value (0–10). The selected task's value tends to fall and the other task's value tends to rise, so switching earns more points but costs cognitive control. Choices are modeled as repeat vs. switch (`df_choice='transcode'`). The model has three states with repeat/switch modules each:
- stateless reward modules on the signed value difference;
- a stateful control-cost state;
- stateless fatigue modules driven by the normalized block number (control becomes more costly the longer it is exerted).

Benchmarked against an `ExpectedValueControl` model.

### `bruckner2025` — Helicopter / predictive-inference task
Participants place a bucket to catch bags dropped from a hidden helicopter whose position changes at change points. On some trials the helicopter is visible. Coins in the bags are of high or low value, and in the *push* condition the bucket is displaced between trials (anchoring). Participants are children, younger adults and older adults. The model has:
- belief updates split by catch/miss;
- changepoint and uncertainty learning-rate states, routed by an externalized |PE| > threshold mask;
- a stateless anchoring module.

These combine into a gated output $(1-\alpha) b_t + \alpha \hat\mu_t + \text{anchor}$, trained with MSE. Benchmarked against the Reduced Bayesian Model (class `RationalResourceModel`, Bruckner et al., 2025).

### `bustamante2023` — Patch-foraging task
Participants harvest a depleting tree or exit to travel to a new one (the Marginal Value Theorem setting). This is the shared RL architecture reduced to working memory + choice perseveration. Only the harvest item carries values, the exit item is a fixed 0 reference, and harvest/exit take the roles of chosen/not chosen. Benchmarked against a Marginal Value Theorem model (`MarginalValueTheoremModel`).

### `dezfouli2019` — Two-armed bandit in depression / bipolar disorder
A two-armed bandit with sparse, block-wise fixed Bernoulli rewards. The participants are healthy controls and patients with depression or bipolar disorder (`diag`). The model is the full shared RL architecture without deviations (`spice_dezfouli2019.py`). Benchmarked against a `GQLModel` (generalized Q-learning).

### `eckstein2026` — Restless four-armed bandit
A large-scale (862 participants) four-armed bandit with drifting, mean-reverting payouts and the options arranged in a circle. The model is the full shared RL architecture plus a spatial attention-bias module (`is_adjacent`/`is_opposite` relative to the chosen option). Benchmarked against `BestRLModel`, the winning Q-learning model of Eckstein et al. (2026) with update bias, forgetting toward Q_init and perseveration. An older, larger architecture is kept in `spice_eckstein2026_original.py`.

### `ganesh2024a` — Confidence-weighted reward learning
Perceptual decision-making combined with reward learning. Participants choose between two Gabor patches, and the reward contingency is tied to the low- vs. high-contrast stimulus, so the reward-relevant item must first be perceived. This is the full shared RL architecture in item space (low/high contrast) rather than action space (left/right), which demonstrates the items-vs-actions decoupling (see [training.md](training.md#dimension-conventions)).
- **Perception module.** A stateless `perception_certainty` module maps |contrast difference| to a sigmoid certainty.
- **Certainty inputs.** The current trial's certainty is an extra input to the reward and working-memory modules. The choice module also gets the next trial's certainty.
- **Item-to-action mapping.** At decision time, the summed item-space logits are mixed with their flipped counterpart, weighted by the next trial's certainty (range 0.5–1). They are then mapped to left/right by the sign of the next trial's contrast difference.
- **Data preparation.** `prepare_dataset` adds the next trial's contrast difference as an extra input and drops the last trial of each session.

Benchmarked against a normative `BayesianModel` (Ganesh et al., 2024).

### `kolff2025` — Chimpanzee grooming negotiation
Video-coded dyadic grooming interactions of wild chimpanzees. Each interaction is modeled from each ape's perspective, so apes are participants and interactions are blocks. The raw behaviors are recoded into Groom + five Negotiation categories (see [kolff2025_preprocessing.md](kolff2025_preprocessing.md)).

The paper model is the **TOBETO** model (`spice_kolff2025_tobeto.py`). It predicts the focal ape's next act as `none` / to-be-groomed signal / to-groom signal / groom. Three latent drives are updated from both apes' last act and the centered dominance-rank difference. The SINDy library is linear (degree 1). Benchmarked against a lag-1 conditional frequency table (`ConditionalTobetoModel`). The grooming-expectation variant (`*_groom.py`) and the original 5-action model (`spice_kolff2025.py`) are not part of the paper.

### `weber2024` — Belief updating under volatility (angular state space)
A laser-shield tracking task on a circle. Laser beams scatter around a hidden mean that jumps at change points, under crossed volatility × stochasticity conditions. The model has belief-update and certainty modules split by catch/miss (externalized gating), plus a gated output $(1-c)\,s_t + c\,\hat\mu_t$ in sin/cos space. It is the same predictive-inference family as `bruckner2025` (Nassar-style changepoint belief updating), but with an angular state space, and is evaluated with a movement-clamped angular MSE loss (`clamped_angular_mse`). Benchmarked against a Bayesian `ChangePointModel` (Weber et al., 2024).

---

## `synthetic` — Parameter Recovery Testbed

Not a real-data study. Contains `benchmarking_qlearning.py` (ground-truth `QLearning` agents implemented as a `BaseModel` with fixed SINDy coefficients) and datasets generated by `weinhardt2026/utils/generate_synthetic_datasets.py` at 32/64/128/256/512 participants. The current `synthetic_balanced_<Np>_<iteration>_<idx>.csv` datasets sample each participant's model type (combination of reward, asymmetry, forgetting and choice mechanisms) uniformly before sampling parameters. Fitted with `spice.precoded.choice`, the shared RL architecture without working memory. Used to validate that SPICE recovers known ground-truth cognitive mechanisms. See `analysis_parameter_recovery.py` in [analyses.md](analyses.md) and `slurm_jobs/spice_parameter_recovery.sh`.

---

## `archive` — Inactive Studies

`augustat2025`, `eckstein2022`, `groman2018`, `huang2026`, `rtify2024`, `rtify2024_W`, `sugawara2021` and `ting2023` are earlier, superseded or no longer pursued study directories. They are kept for reference but are not maintained against the current SPICE API and are not part of the paper's task battery.

---

## Adding a New Study

1. Create `weinhardt2026/studies/<study>/` with `data/`, `params/`, `results/`.
2. Write a data-loading module (`get_dataset()`) using `csv_to_dataset()` (see [training.md](training.md#data-pipeline)).
3. Define `spice_<study>.py`: a `BaseModel` subclass + `SpiceConfig` following the [precoded model pattern](training.md#precoded-model-pattern), applying the [polynomial-amenable architecture guidelines](training.md#designing-polynomial-amenable-architectures).
4. Write a hand-coded benchmark model (for `analysis_model_evaluation.py` comparison) and an `Environment<Study>(Env)` + `generate_behavior()` wrapper (for generative benchmarking — see [analyses.md](analyses.md#generative-benchmarking-weinhardt2026utilstaskpy)).
5. Fit with `SpiceEstimator`, then run the [typical analysis sequence](analyses.md#typical-analysis-sequence-for-a-new-study).

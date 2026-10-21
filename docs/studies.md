# SPICE Studies

`weinhardt2026/studies/` holds self-contained per-study directories: a SPICE model config (`spice_<study>.py`), the data loading, hand-coded benchmark model and task environment (`benchmarking_<study>.py`), the simulation interface for the posterior predictive checks and the confusion matrix (`simulation_<study>.py`: `load_dataset`, `load_models`, `generate`), and a slim `<study>.py` run script that fits or loads SPICE, the benchmark and the GRU and prints a quick model evaluation. Every study directory has `data/`, `params/` (the final models, 10 seeds per kind) and `results/<item>/` (see [analyses.md](analyses.md)).

Conventions and infrastructure shared across studies are described in [training.md](training.md) (model/config/estimator internals) and [analyses.md](analyses.md) (the paper analyses and the results layout). This file covers what's specific to each study: the task paradigm, population, and benchmark model.

---

## Active Studies

The task battery of the paper. Full architectures, benchmark equations and dataset details: [weinhardt2026/model_descriptions.md](../weinhardt2026/model_descriptions.md). The study-to-model mapping used for the paper runs is in `slurm_jobs/grid_search.sh`.

**Shared RL architecture.** The bandit-style studies (`synthetic`, `dezfouli2019`, `eckstein2026`, `ganesh2024a`, `bustamante2023`) share one architecture of up to three value states, summed into the logits:
- a working-memory reward value (stateless chosen update, decaying unchosen update);
- an incremental reward value (chosen / not-chosen modules);
- a single merged choice-perseveration module that receives the choice indicator.

Each study only adds its task-specific deviations.

### `braun2018` — Cognitive-control / task-switching effort
Reward-based voluntary task switching. On each trial, participants choose which of two tasks to perform, and each task carries a point value (0–10). The selected task's value tends to fall and the other task's value tends to rise, so switching earns more points but costs cognitive control. A block ends when the points reach 500 (the CSV holds no points; the reconstructed totals of the gap-free blocks are 495–508), so block lengths vary in trials and time; the response-stimulus interval (200 or 1100 ms) alternates by block, 6 blocks each. Choices are modeled as repeat vs. switch (`df_choice='transcode'`). The model has three states with repeat/switch modules each:
- stateless reward modules on the signed value difference;
- a stateful control-cost state;
- stateless fatigue modules driven by the normalized block number (control becomes more costly the longer it is exerted).

Benchmarked against an `ExpectedValueControl` model.

### `bruckner2025` — Helicopter / predictive-inference task
Participants place a bucket to catch bags dropped from a hidden helicopter whose position changes at change points. On some trials the helicopter is visible. Coins in the bags are of high or low value, and in the *push* condition the bucket is displaced between trials (anchoring). Participants are children, younger adults and older adults (`age_group` codes 1, 3 and 4: 7–11, 20–28 and 61–76 years; code 2, adolescents, does not occur — see `data_raw/participants.tsv`). The model has:
- belief updates split by catch/miss;
- changepoint and uncertainty learning-rate states, routed by an externalized |PE| > threshold mask;
- a stateless anchoring module.

These combine into a gated output $(1-\alpha) b_t + \alpha \hat\mu_t + \text{anchor}$, trained with MSE. Benchmarked against the Reduced Bayesian Model (class `RationalResourceModel`, Bruckner et al., 2025).

### `bustamante2023` — Patch-foraging task
Participants harvest a depleting tree or exit to travel to a new one (the Marginal Value Theorem setting). This is the shared RL architecture reduced to the incremental reward value + choice perseveration, plus a stateless leave-threshold module driven by the block's effort condition (travel task: button pressing or MSIT x effort level: low or high; inputs `is_msit`, `is_high`). Its terms mirror the paper's per-condition leave thresholds, so the `is_high` coefficient is a per-participant effort cost of travel. Only the harvest item carries values, the exit item is a fixed 0 reference, and harvest/exit take the roles of chosen/not chosen. Benchmarked against the Marginal Value Theorem model with effort costs of the original paper (`EffortCostMVTModel`, Bustamante et al. 2023, Eqs. 2-5): a fixed leave threshold per effort condition (travel task x effort level), β plus four thresholds per participant.

### `dezfouli2019` — Two-armed bandit in depression / bipolar disorder
A two-armed bandit with sparse, block-wise fixed Bernoulli rewards. The participants are healthy controls and patients with depression or bipolar disorder (`diag`). The model is the full shared RL architecture without deviations (`spice_dezfouli2019.py`). Benchmarked against a `GQLModel` (generalized Q-learning).
- **Reward probabilities per block.** In each of a participant's 12 blocks one response pays with probability 0.25, 0.125 or 0.08 (four blocks each) and the other with 0.05. The published data (`data/dezfouli2019_original.csv`, the authors' `for_plos.csv`) does not record which block had which probabilities (the authors' `data_for_plos.R` deletes the block `condition`), and the block number is only the presentation order: each participant has every condition twice, in varying positions. The authors' own simulation code (`sims.py`) assumed a fixed pair per block number, which does not hold in their data.
  - **Raw data:** `data_raw/choices_diagno.csv`, the authors' raw trial data, unpacked unchanged from `data/BD/choices_diagno.csv.zip` of github.com/adezfouli/rnn_beh (Dezfouli et al., 2019, PLoS Comput Biol 15(6): e1006903; Apache License 2.0). SHA-256 of the csv: `80d8d04f59694019a983cf67b649a5f4a35e2d4120465f84a401d906bcc26379`. Columns used: `ID` (subject, not anonymized), `trial` (= block), `key` (R1 = right = action 0, R2 = left = action 1), `choice` (R1 = better response, `best_action` in the published data), `reward`, `condition`; symptom scores (`Mania`, `Depression`) and response times are not used.
  - **Mapping:** condition 1–2 → better response 0.25, 3–4 → 0.125, 5–6 → 0.08; the other response 0.05 (empirical reward rates 0.247 / 0.122 / 0.082 and 0.055).
  - **Processing:** `add_reward_probabilities()` (`benchmarking_dezfouli2019.py`) matches the anonymized participants to the raw ids by their complete trial sequences (block, key, reward, better response; unique for all 101) and writes `condition`, `p_reward_0`, `p_reward_1` into `data/dezfouli2019.csv` (reproducible from `data/dezfouli2019_original.csv` and the raw csv). Simulations (`EnvironmentDezfouli2019`) look the probabilities up per (participant, block); fitting and one-step evaluation use the observed rewards.
  - **To be stated in the SI** (data section of dezfouli2019), see [model_descriptions.md](../weinhardt2026/model_descriptions.md#34-dezfouli-2019-two-armed-bandit-clinical).

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
Video-coded dyadic grooming interactions of wild chimpanzees (41 apes, two communities). The raw behaviors are recoded into Groom + five Negotiation categories (see `kolff2025/data/PREPROCESSING.md`). Every interaction is read twice, from the approached ape's and from the approacher's perspective (`benchmarking_kolff2025.build_dataframe`), so apes are participants and interactions are blocks. The target is whether the focal ape's next act is grooming, scored only on events where it acts next (target mask packed into `ys` by `spice_kolff2025.prepare_dataset`, masked loss `LOSS_FN`).

The paper model (`spice_kolff2025.py`) has four modules with their own states, split by who acts and what is done: negotiation received / given and grooming received / given. Inputs are the act categories only (no rank or role), the SINDy library is linear (degree 1), and it is fitted as one **group model** (`n_participants=1`): with about 110 targets per ape, individual models overfit. The test set holds out whole dyads (fold 0 of `dyad_folds`, 5 folds balanced by targets). Benchmarked against a lag-1 logistic regression on the current event's acts (`LogisticRegressionModel`, group model) and a GRU with the same inputs (`ModelInputGRU`). Generative benchmarking is closed-loop only on that decision: the partner's acts and the timing of the ape's acts are replayed (`benchmarking_kolff2025.generate_behavior`), and the metrics are conditional grooming probabilities (after partner solicitation / preparation / grooming, after own grooming) and the number of own acts before the first grooming. Earlier variants (TOBETO, ID2-only, mirrored with context inputs) are in the git history (removed 2026-10-21).

### `weber2024` — Belief updating under volatility (angular state space)
A laser-shield tracking task on a circle. Laser beams scatter around a hidden mean that jumps at change points, under crossed volatility × stochasticity conditions. The model has belief-update and certainty modules split by catch/miss (externalized gating), plus a gated output $(1-c)\,s_t + c\,\hat\mu_t$ in sin/cos space. It is the same predictive-inference family as `bruckner2025` (Nassar-style changepoint belief updating), but with an angular state space, and is evaluated with a movement-clamped angular MSE loss (`clamped_angular_mse`). Benchmarked against a Bayesian `ChangePointModel` (Weber et al., 2024).

---

## Train/Test Splits and Hyperparameters

**Block split.** Each study holds out whole blocks (`split_data_along_blockdim`), aiming for 25% of the data in the test set (allowed range 20–30%, depending on the number of blocks). Test blocks come from the middle of the session so that early learning effects and late fatigue effects stay in the training data. Taking one of the two middle blocks as the start, further test blocks are spread evenly toward the beginning and the end. Where the two directions cannot be balanced exactly, the split leaves more room at the end, so fewer early blocks are held out (learning is the bigger concern than fatigue in these simple tasks).

**One test set.** Only one test set is used per study. The benchmark and GRU hyperparameters are selected on it: the configuration with the highest median test likelihood across seeds is kept. This overfits their hyperparameters to the test set and gives them an advantage over SPICE, so the comparison is conservative.

| Study | Blocks | Test blocks | Test share | Notes |
|---|---|---|---|---|
| `dezfouli2019` | 1–12 | 3, 6, 9 | 26% | |
| `braun2018` | 1–12 | 3, 6, 9 | 25% | |
| `ganesh2024a` | 1–12 | 3, 6, 9 | 25% | |
| `bustamante2023` | 1–8 | 3, 6 | 25% | |
| `eckstein2026` | 1–5 | 2 | 21% | 76 of 862 participants have only blocks 1–3. Block 2 is used because the simple task leaves little task learning to exclude. |
| `weber2024` | 0–15 (0-indexed) | 3, 5, 9, 12 | 25% | Stochasticity is fixed per block and volatility is counterbalanced per participant. These blocks hold out every volatility × stochasticity cell once per participant. |
| `bruckner2025` | 1–4 | 2, 4 | 50% | Exception: blocks 1–2 are no-push and 3–4 push (same order for everyone). One block per condition is held out so both conditions are tested, skipping the first block of each for learning and calibration. With 4 short blocks, fatigue is not a concern. |
| `kolff2025` | interactions | fold 0 of `dyad_folds` | 20% of interactions (18% of targets) | Exception: interactions have no fixed order or timing, so there are no learning or fatigue effects to avoid. Whole dyads are held out instead, testing whether the mechanism carries over to new pairings. |

The test blocks are set in `slurm_jobs/grid_search.sh` (`TEST_BLOCKS`) and in each study's driver script.

**SPICE hyperparameters.** All studies use one configuration (the defaults in `slurm_jobs/grid_search.sh`), with these deviations:
- **`bruckner2025`, `weber2024`:** gate and feature penalties are 0. These studies use MSE, which is much smaller than cross-entropy in absolute terms, so the penalties would otherwise dominate the loss.
- **`kolff2025`:** only a group model is fitted, so participant-level feature selection is redundant and gate and feature penalties are 0. `sindy_weight=0.01` instead adds a sparsifying pull on the dynamics. The model also uses a linear SINDy library (`--polynomial_degree 1`).

---

## `synthetic` — Parameter Recovery Testbed

Not a real-data study. Contains `benchmarking_qlearning.py` (ground-truth `QLearning` agents implemented as a `BaseModel` with fixed SINDy coefficients) and datasets generated by `weinhardt2026/utils/generate_synthetic_datasets.py` at 32/64/128/256/512 participants. The current `synthetic_balanced_<Np>_<iteration>_<idx>.csv` datasets sample each participant's model type (combination of reward, asymmetry, forgetting and choice mechanisms) uniformly before sampling parameters. Fitted with `spice.precoded.choice`, the shared RL architecture without working memory. Used to validate that SPICE recovers known ground-truth cognitive mechanisms. See `analysis_parameter_recovery.py` in [analyses.md](analyses.md) and `slurm_jobs/spice_parameter_recovery.sh`.

---

## Inactive Studies — branch `more-studies`

`augustat2025`, `eckstein2022`, `groman2018`, `huang2026`, `rtify2024`, `rtify2024_W`, `sugawara2021` and `ting2023` are earlier, superseded or no longer pursued studies, not part of the paper's task battery and not maintained against the current SPICE API. Their code lives in `weinhardt2026/studies/archive/` on the branch `more-studies`; their data and fitted parameters (untracked) were moved to `~/spice_more_studies_data/` (same layout below `weinhardt2026/studies/`).

---

## Adding a New Study

1. Create `weinhardt2026/studies/<study>/` with `data/`, `params/`, `results/`.
2. Write a data-loading module (`get_dataset()`) using `csv_to_dataset()` (see [training.md](training.md#data-pipeline)).
3. Define `spice_<study>.py`: a `BaseModel` subclass + `SpiceConfig` following the [precoded model pattern](training.md#precoded-model-pattern), applying the [polynomial-amenable architecture guidelines](training.md#designing-polynomial-amenable-architectures).
4. Write a hand-coded benchmark model (for the model evaluation) and an `Environment<Study>(Env)` + `generate_behavior()` wrapper (for generative benchmarking — see [analyses.md](analyses.md#generative-benchmarking-weinhardt2026utilstaskpy)), and `simulation_<study>.py` (`load_dataset`, `load_models`, `generate`; see `weinhardt2026/utils/study_models.py`).
5. Choose test blocks following the [split strategy](#traintest-splits-and-hyperparameters) and add the study to `slurm_jobs/grid_search.sh`.
6. Fit with `SpiceEstimator`, then run the [typical analysis sequence](analyses.md#typical-analysis-sequence-for-a-new-study).

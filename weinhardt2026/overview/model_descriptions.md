# Model Descriptions for SPICE Paper

This document provides full architectural descriptions of all SPICE models, benchmark models, and the GRU baseline used across the benchmark studies. It is intended as input for writing the methods/supplementary sections of the SPICE paper.

The study-to-model mapping follows the cluster runs in `slurm_jobs/grid_search.sh` (synthetic data: `slurm_jobs/parameter_recovery.sh`); each section names the module that defines the model.

---

## 1. Shared SPICE Architecture

Each SPICE model specifies a set of RNN submodules with their control-signal inputs, a set of latent memory states, and the trial-by-trial computation that orchestrates how submodules update states and produce the output (action logits, or a continuous prediction). This section describes what all study models share; Section 3 lists what is study-specific.

### 1.1. RNN Submodules

Every submodule $m$ is a single-unit recurrent cell that updates one scalar state per item. For item $i$ of participant $p$, with control signals $x_t \in \mathbb{R}^{n_x}$ and current state $h_t$:

$$y_t = \text{GELU}\big(W_f [x_t; h_t] + b_f\big) \in \mathbb{R}^{P}, \qquad P = 8 + n_x + d_e$$
$$g_p = \text{HardSigmoid}(U e_p) \in [0, 1]^{P}, \qquad \gamma_p = \text{HardSigmoid}(u^\top e_p) \in [0, 1]$$
$$h_{t+1} = h_t + \Delta t \cdot \gamma_p \cdot w_n^\top (y_t \odot g_p)$$

- **Group-level features** $y_t$ ($W_f$, $b_f$, $w_n$) are shared by all participants.
- **Individual-level gates** come from a learned participant embedding $e_p \in \mathbb{R}^{d_e}$. The feature gates $g_p$ switch single group-level features on or off for a participant. The module gate $\gamma_p$ switches the whole module on or off. Gates use a hard sigmoid in the forward pass, so they reach exactly 0 or 1, and a sigmoid gradient in the backward pass, so closed gates can reopen.
- **Residual update.** The module outputs an increment to the state ($\Delta t = 1$ for all trial-level modules).
- **Stateless modules** (`include_state=False`) never see their own state, and their output replaces the state rather than incrementing it: $h_{t+1} = \gamma_p \, w_n^\top (y_t \odot g_p)$ with $h_t \equiv 0$. They implement instantaneous mappings from control signals.
- **Action masks.** `call_module(..., action_mask=M)` updates only items with $M_i = 1$; all other items keep their state. Splitting a mechanism into, for example, a *chosen* and a *not-chosen* module is done through complementary masks rather than through the RNN inputs (externalized gating).
- **Memory-state initial values** are fixed constants in the config, or learnable per participant when the config sets `None`.

The participant embedding is the only source of individual differences in the RNN. It enters only through the gates, not as an input feature. Dropout (0.1) is applied to the group-level features and the feature gates, but not to the module gate.

### 1.2. SINDy Library and Structural Pruning

Each module has a SINDy library of polynomial candidate terms (degree 2 by default) over its own state and its control signals, with per-participant coefficients. Before training, every study model zeroes the library terms that are structurally redundant (`preprocess_coefficients`):
- squares of binary indicators ($x^2 = x$), e.g. `reward[t]` in reward-binary tasks, `choice[t]`, `catch`, `v_t`;
- products of two signals from a mutually exclusive group ($x_i x_j = 0$), e.g. `is_adjacent`/`is_opposite` in eckstein2026.

### 1.3. Training

Two-stage training as described in the main methods and [docs/training.md](../docs/training.md):
1. **Stage 1, joint RNN + SINDy training.** The objective is
$$\mathcal{L} = \mathcal{L}_{\text{beh}} + \lambda_{\text{sindy}}\,\mathcal{L}_{\text{sindy}} + \alpha\,\lVert\xi\rVert_1 + \lambda_{\text{gate}}\,\mathcal{L}_{\text{gate}} + \lambda_{\text{feature}}\,\lVert\theta_{\text{group}}\rVert_2^2$$
   - $\mathcal{L}_{\text{beh}}$: behavioral loss (cross-entropy by default).
   - $\mathcal{L}_{\text{sindy}}$: squared difference between each module's RNN state update and the SINDy prediction from the same state. Its gradients flow into both the RNN, which is pulled toward polynomial dynamics, and the SINDy coefficients, which are fitted jointly. $\lambda_{\text{sindy}}$ (`sindy_weight`) is ramped up during warmup.
   - $\alpha\lVert\xi\rVert_1$: L1 sparsity penalty on the SINDy coefficients (`sindy_alpha`).
   - $\mathcal{L}_{\text{gate}}$: L1 on the individual-level feature and module gates (`gate_penalty`), which pushes participants to switch features and modules off.
   - $\lambda_{\text{feature}}\lVert\theta_{\text{group}}\rVert_2^2$: L2 on the group-level RNN weights $W_f, b_f, w_n$ (`feature_penalty`). It counterbalances the gate penalty. Without it, shrinking gates could be compensated by growing group-level weights, so the gates would not truly switch features off.

   The feature penalty is implemented as decoupled weight decay (AdamW).
2. **Stage 2, SINDy refit** on the frozen RNN: sparsity discovery with pruning, then coefficient estimation on the fixed sparsity pattern.
3. **Module cleanup.** After the refit, every active SINDy term whose gradient on the (centered) predicted logits is numerically zero is removed per ensemble member and participant. This removes, for example, decay terms on states that never leave their initial value, and constants that shift all options equally.

**Default hyperparameters of the paper runs** (`slurm_jobs/grid_search.sh` defaults, which override the `weinhardt2026/run.py` defaults where noted). One configuration is used for all studies, with the per-study deviations listed below the table:

| Hyperparameter | Value |
|---|---|
| Embedding size $d_e$ | 8 |
| Ensemble size | 10 |
| Dropout | 0.1 |
| SINDy polynomial degree | 2 (1 for kolff2025) |
| `sindy_weight` | 0 (run.py default 0.01) |
| `sindy_alpha` | 0.0005 (run.py default 0.0001) |
| `gate_penalty` | 0.01 (run.py default 0) |
| `feature_penalty` | 0.0001 (run.py default 0) |
| Pruning threshold | 0.05 (run.py default 0.01) |
| Ensemble ratio test (`pruning_test`) | 0.5 |
| Behavioral loss | Cross-entropy with label smoothing 0.01 (study-specific losses noted below) |
| Seeds | 10 per study |

Per-study deviations:
- **bruckner2025, weber2024:** `gate_penalty` = `feature_penalty` = 0. Their MSE loss is much smaller in absolute terms than cross-entropy, so the penalties would otherwise dominate the loss.
- **kolff2025:** `gate_penalty` = `feature_penalty` = 0 and `sindy_weight` = 0.01. Only a group model is fitted, so no participant-level feature selection is needed; the SINDy weight instead adds a sparsifying pull on the dynamics.

**Train/test splits.** Each study holds out whole blocks, aiming for 25% of the data in the test set (allowed range 20–30%, depending on the number of blocks). Test blocks come from the middle of the session, so that early learning effects and late fatigue effects stay in the training data: starting from one of the two middle blocks, further test blocks are spread evenly toward the beginning and the end, leaving more room at the end where the two directions cannot be balanced exactly. Only one test set is used per study. The hyperparameters of the benchmark models and the GRU are selected on it (the configuration with the highest median test likelihood across seeds), which overfits them to the test set and gives them an advantage over SPICE. Exceptions are noted in the study sections: bruckner2025 holds out one block per condition (50%), and kolff2025 holds out whole dyads instead of blocks (20% of interactions), because the interactions have no fixed order or timing.

### 1.4. Shared Reinforcement-Learning Architecture (Bandit Family)

Five studies use one shared RL architecture: synthetic, dezfouli2019, eckstein2026, ganesh2024a and bustamante2023. Their study sections in Section 3 only list how they deviate from it.

**Submodules.** Up to three value states, each updated by one or two modules:

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|---|---|---|---|---|---|
| `value_wm_reward_chosen` | $r_t$ | `value_wm_reward` | chosen | False | Working-memory reward value: overwritten with a function of the latest reward |
| `value_wm_reward_not_chosen` | — | `value_wm_reward` | unchosen | True | Decay of working-memory values of unchosen items |
| `value_reward_chosen` | $r_t$ | `value_reward` | chosen | True | Incremental reward learning for the chosen item |
| `value_reward_not_chosen` | — | `value_reward` | unchosen | True | Forgetting / counterfactual drift for unchosen items |
| `value_choice` | $c_t$ (choice indicator) | `value_choice` | all items | True | Choice perseveration |

- **Working memory** (`value_wm_reward`) captures fast, one-shot reward effects. The chosen item's value is replaced by a function of the latest reward each time it is chosen, and decays while unchosen.
- **Reward value** (`value_reward`) captures slow, incremental learning: the chosen item's value moves with the reward, and unchosen values can be forgotten.
- **Choice value** (`value_choice`) captures perseveration. It is a single module for all items that receives the choice indicator $c_{t,i} \in \{0, 1\}$ as input, so one equation covers both chosen and unchosen items.

**Memory states:** all initialized at 0.

**Logit computation:** the states are summed:
$$\text{logits}_t = V^{\text{wm}}_t + V^{\text{reward}}_t + V^{\text{choice}}_t$$

**SINDy library:** degree 2. Rewards (when binary) and choice indicators are binary, so their squared terms are pruned (Section 1.2). With the remaining terms, the reward module can express asymmetric learning rates through the interaction $V \cdot r$.

**Study variants:**

| Study | Working memory | Reward value | Choice value | Deviations |
|---|---|---|---|---|
| Synthetic | — | ✓ | ✓ | Ground-truth-compatible subset (`spice.precoded.choice`) |
| Dezfouli 2019 | ✓ | ✓ | ✓ | None (the full shared architecture) |
| Eckstein 2026 | ✓ | ✓ | ✓ | 4 arms; additional spatial attention-bias module |
| Ganesh 2024a | — | ✓ | ✓ | Values in contrast item space; perceptual-certainty module whose output is an extra input to all value modules; certainty-weighted item-to-action mapping |
| Bustamante 2023 | ✓ | ✓ | ✓ | Only the harvest item carries values; harvest/exit take the roles of chosen/unchosen; exit item is a fixed 0 reference; working-memory module keeps its state; stateless leave-threshold module on the effort condition |

---

## 2. GRU Baseline Architecture

The GRU (Gated Recurrent Unit) baseline (`weinhardt2026/utils/benchmarking_gru.py`) serves as a black-box recurrent neural network benchmark across all studies.

**Input processing.** At each trial $t$, the model receives a concatenation of the one-hot encoded action $a_{t} \in \{0,1\}^A$, the reward vector $r_{t} \in \mathbb{R}^{n_{\text{reward}}}$, and any additional task-specific inputs $x_t^{\text{add}} \in \mathbb{R}^{n_{\text{add}}}$. The concatenated input is projected through a linear layer to a hidden representation of size $d_h$:

$$z_t = W_{\text{in}} [a_t; r_t; x_t^{\text{add}}] + b_{\text{in}}$$

**Recurrent computation.** The projected input $z_t$ is passed through a standard GRU cell with hidden state $h_t \in \mathbb{R}^{d_h}$:

$$\hat{r}_t = \sigma(W_{ir} z_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})$$
$$\hat{z}_t = \sigma(W_{iz} z_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})$$
$$\hat{n}_t = \tanh(W_{in} z_t + b_{in} + \hat{r}_t \odot (W_{hn} h_{t-1} + b_{hn}))$$
$$h_t = (1 - \hat{z}_t) \odot \hat{n}_t + \hat{z}_t \odot h_{t-1}$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication.

**Output.** The hidden state is projected through a linear output layer to produce action logits:

$$\text{logits}_t = W_{\text{out}} h_t + b_{\text{out}} \in \mathbb{R}^A$$

Action probabilities are obtained via softmax: $p(a_{t+1} | h_t) = \text{softmax}(\text{logits}_t)$. For continuous-output studies the output is used directly as the prediction and trained with the study's loss function.

**Hyperparameters.** Dropout is applied after the input projection and after the GRU output. Trained on the same loss function as the SPICE model of the study, with the optimizer and LR schedule of Section 2.1. The GRU has no participant embedding: one network is shared by all participants. Hidden size, dropout, learning rate and LR scheduler are grid-searched per study (`slurm_jobs/benchmark_grid_search.sh`: $d_h \in \{8, 16, 32\}$, dropout $\in \{0, 0.1, 0.25, 0.5\}$, lr $\in \{0.01, 0.001\}$, scheduler on/off; Section 1.3 describes the selection). The final configurations (`studies/<study>/params/gru_*`):

| Study | $d_h$ | Dropout | lr | Scheduler |
|---|---|---|---|---|
| Braun 2018 | 32 | 0.1 | 0.001 | ✓ |
| Bruckner 2025 | 32 | 0 | 0.01 | — |
| Bustamante 2023 | 16 | 0.1 | 0.01 | ✓ |
| Dezfouli 2019 | 32 | 0.5 | 0.01 | ✓ |
| Eckstein 2026 | 32 | 0.1 | 0.01 | — |
| Ganesh 2024a | 8 | 0.1 | 0.01 | ✓ |
| Kolff 2025 | 8 | 0.25 | 0.01 | ✓ |
| Weber 2024 | 32 | 0 | 0.01 | — |

Kolff 2025 uses a variant with the same inputs as its SPICE model (`ModelInputGRU` in `benchmarking_kolff2025.py`).

**Total parameters.** The input dimension is $d_{\text{in}} = A + n_{\text{reward}} + n_{\text{add}}$, and the model has $O(d_h^2 + d_h \cdot d_{\text{in}})$ parameters shared across all participants.

### 2.1. Optimizer and Learning-Rate Schedule (GRU and Benchmark Models)

The GRU and every gradient-trained benchmark model are fitted by the same loop (`training` in `weinhardt2026/utils/benchmarking_gru.py`, called by `weinhardt2026/analysis/grid_search/benchmark_grid_search.py`):

| Setting | Value |
|---|---|
| Optimizer | `torch.optim.Adam` over all model parameters, PyTorch defaults otherwise ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, no weight decay) |
| Learning rate | grid-searched, $\{0.01, 0.001\}$ |
| Epochs | 1000 (`EPOCHS` in `slurm_jobs/benchmark_grid_search.sh`) |
| Batch | full batch (all sessions of the training set), so one optimizer step per epoch |
| LR scheduler | grid-searched on/off; when on, `torch.optim.lr_scheduler.ReduceLROnPlateau` with `mode='min'`, `factor=0.5`, `patience=50`, `min_lr=1e-5`, stepped once per epoch on the **test loss** |
| Seeds | 10 per grid point |
| Initialization | GRU: PyTorch default. Benchmarks: their default parameter values, each unconstrained parameter jittered by $\mathcal{N}(0, 0.5)$ per seed (`--init_std`), since they would otherwise train deterministically |

The scheduler steps on the held-out loss, which, like the selection of the grid point (Section 1.3), uses the test set and favors the GRU and the benchmarks over SPICE. There is no early stopping: the model after the last epoch is kept. kolff2025's benchmark is a closed-form logistic regression (Section 3) and is not trained by this loop.

Final configurations of the benchmark models (`studies/<study>/params/benchmark_*`; the GRU's are in the table above):

| Study | lr | Scheduler |
|---|---|---|
| Braun 2018 | 0.01 | — |
| Bruckner 2025 | 0.01 | — |
| Bustamante 2023 | 0.01 | ✓ |
| Dezfouli 2019 | 0.01 | — |
| Eckstein 2026 | 0.01 | — |
| Ganesh 2024a | 0.01 | ✓ |
| Weber 2024 | 0.01 | — |

---

## 3. Study-Specific Models


### 3.1. Synthetic Study: Q-Learning Recovery

**Task.** A simulated two-armed bandit used for parameter-recovery validation (`weinhardt2026/utils/generate_synthetic_datasets.py`). On every trial a simulated agent chooses one of two arms and receives a binary reward from the chosen arm only (partial feedback). The reward probabilities of the arms drift over trials as random walks with step size $\sigma = 0.2$, so the agent has to keep learning throughout a block. The agents are the ground-truth `QLearning` models described below. Each one combines up to four mechanisms: reward learning, asymmetric learning from rewards vs. omissions, forgetting of the unchosen arm's value, and choice perseveration.

**Data.**
- $A = 2$; 4 blocks × 100 trials per participant.
- Dataset sizes of 32, 64, 128, 256 and 512 participants, 8 independent iterations per size (seed 42).
- Balanced sampling (`sampling='balanced'`): each participant's *model type* (set of active mechanisms) is first drawn uniformly from 9 types, then the parameters are drawn conditional on that type. The mechanisms are Reward ($\beta_r > 0$), Asymmetry ($\alpha_{\text{penalty}} \neq \alpha_{\text{reward}}$), Forgetting ($f > 0$) and Choice ($\beta_c > 0$, $\alpha_c > 0$). Asymmetry and Forgetting require Reward, which leaves the 9 types: R, C, R+A, R+F, R+C, R+A+F, R+A+C, R+F+C, R+A+F+C. Parameter means: $\beta_r = 3$, $\beta_c = 1$, $\alpha_{\text{reward}} = \alpha_{\text{penalty}} = 0.5$, $f = 0.2$, $\alpha_c = 0.5$. Values below the threshold 0.2 are set to zero.
- The stability runs use `synthetic_balanced_256p_0_0.csv` without a held-out block split.

#### SPICE Model (`spice.precoded.choice`)

The shared RL architecture (Section 1.4) **without the working-memory state**: the modules `value_reward_chosen`, `value_reward_not_chosen` and `value_choice` on the states `value_reward` and `value_choice`, with $\text{logits}_t = V^{\text{reward}}_t + V^{\text{choice}}_t$. This is exactly the structure of the generating agents, so every ground-truth mechanism maps onto one module and one or two library terms.

**Ground-truth equations** (`QLearning` in `studies/synthetic/benchmarking_qlearning.py`, which is itself a `BaseModel` whose SINDy coefficients are set from the parameters). With binary rewards ($r^2 = r$):
- Chosen value: $\Delta V^{\text{reward}}_{\text{chosen}} = -\alpha_{\text{penalty}} V + \beta_r \alpha_{\text{reward}} \, r + (\alpha_{\text{penalty}} - \alpha_{\text{reward}}) \, V r$
- Unchosen value: $\Delta V^{\text{reward}}_{\text{unchosen}} = -f \, V$
- Choice perseveration: $\Delta V^{\text{choice}} = -\alpha_c V^{\text{choice}} + \beta_c \alpha_c \, c$

#### Benchmark Model

None (ground-truth recovery study).


### 3.2. Braun 2018: Reward-Based Voluntary Task Switching

**Task.** Reward-based voluntary task switching (rVTS; Braun & Arrington, 2018). On each trial, participants freely choose which of two identification tasks to perform on the upcoming stimulus. Each task carries a point value (integer, 0-10) that is displayed before the choice and earned for a correct response. Both values start at 5 at the beginning of a block. After every selection, the value of the selected task decreases by 1 with probability 0.5 (floor 0), and the value of the unselected task increases by 1 with probability 0.5 (ceiling 10). Repeating the same task therefore gradually makes the other task more lucrative. Switching to it, however, carries a cognitive switch cost. The paradigm thus pits reward maximization against the effort of exerting cognitive control, and it tests whether control becomes more costly the longer it is exerted (a rising indifference point over the experiment).

The choice is modeled in relative terms, repeat vs. switch (`transcode`; 59% repeats), rather than as the identity of the task. The model receives the signed value difference between the other and the current task (`difference`, observed range −3 to 3), and indicators of whether the current task's value just decreased (`current`) and the other task's value just increased (`other`). The trials differ in response-stimulus interval (200 ms or 1100 ms). Blocks are time-limited, so the number of trials per block varies (5–274, median 61).

**Data.**
- $A = 2$ (repeat=0, switch=1), no reward column; 63 participants, 12 blocks each, 346–1548 trials per participant (median 760).
- Additional inputs are shifted by one trial (`timeshift_additional_inputs = -1`), so the model sees the point values that are shown before the next choice.
- Held-out test blocks: 3, 6, 9.

#### SPICE Model (`studies/braun2018/spice_braun2018.py`)

The model uses 6 submodules operating on 3 memory states. Each state has a repeat item (written by the `*_repeat` module) and a switch item (written by the `*_switch` module):

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|-----------|----------------|---------------|-------------|-----------------|-------------|
| `reward_repeat` | $-\Delta r_{\text{tasks}}$ | `value_reward` | repeat item | False | Instantaneous reward sensitivity for repeating |
| `reward_switch` | $\Delta r_{\text{tasks}}$ | `value_reward` | switch item | False | Instantaneous reward sensitivity for switching |
| `task_repeat` | `repeat` | `value_control` | repeat item | True | Evolving cognitive-control cost for repeating |
| `task_switch` | `repeat` | `value_control` | switch item | True | Evolving cognitive-control cost for switching |
| `fatigue_repeat` | $b / B$ | `value_fatigue` | repeat item | False | Fatigue effect on repeat tendency |
| `fatigue_switch` | $b / B$ | `value_fatigue` | switch item | False | Fatigue effect on switch tendency |

**Memory states:** `value_reward`, `value_control`, `value_fatigue`, all with learnable per-participant initial values (`None` in the config).

**Control signal preprocessing:**
- $\Delta r_{\text{tasks}}$ (`difference`): the other task's value minus the current task's value, divided by 10 (the maximum possible difference, so the input lies in $[-1, 1]$) and negated for the repeat item. This provides a relative reward signal.
- `repeat`: 1 if the previous action was repeat, 0 otherwise (binary; squared terms are pruned).
- $b/B$: block number normalized by the total number of blocks ($B = 12$).

**Logit computation:**
$$\text{logits}_t = V^{\text{reward}}_t + V^{\text{control}}_t + V^{\text{fatigue}}_t$$

The reward and fatigue modules are stateless and map their control signals to values instantaneously. Only `value_control` evolves across trials.

**SINDy library degree:** 2.

#### Benchmark Model: Expected Value of Control (EVC)

**Reference:** Shenhav, Botvinick & Cohen (2013); adapted for Braun & Arrington (2018).

The Expected Value of Control (EVC) model computes the value of each action (repeat or switch) as the expected reward minus the cognitive effort cost, where effort cost increases with accumulated fatigue. The action space is $A = 2$ (repeat, switch).

**Parameters (per participant, 3 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\beta_{\text{reward}}$ | $> 0$ (softplus, clamp [0.001, 20]) | Sensitivity to the value difference |
| $\beta_{\text{cost}}$ | $\geq 0$ (softplus, clamp [0, 20]) | Base control cost of a switch (indifference point) |
| $\beta_{\text{fatigue}}$ | $\geq 0$ (softplus, clamp [0, 20]) | Additional switch cost per unit of normalized time |

**Decision rule.** Actions are coded relative to the previous task, $a \in \{\text{repeat}=0, \text{switch}=1\}$. Only the difference $\text{EVC}(\text{switch}) - \text{EVC}(\text{repeat})$ is identifiable under a softmax, so the repeat logit is fixed at 0:

$$\text{EVC}(\text{repeat}) = 0, \qquad \text{EVC}(\text{switch}) = \beta_{\text{reward}} \cdot \Delta V - (\beta_{\text{cost}} + \beta_{\text{fatigue}} \cdot t_{\text{norm}})$$

where $\Delta V$ is the point value of the other task minus that of the current task (shown before the choice), and $t_{\text{norm}} = b / B$ is the normalized block position (1-indexed block number divided by the total number of blocks $B = 12$).

**Note:** The model is stateless. It computes action values from instantaneous task features without maintaining internal state across trials.


### 3.3. Bustamante 2023: Patch Foraging

**Task.** A foraging task (Bustamante et al., 2023) that implements the classic patch-leaving problem of optimal foraging theory. Participants harvest apples from trees. On each decision they either **harvest** the current tree, which takes 2 s and yields a reward, or **exit** it, which yields no reward and starts a travel period of ≈8.3 s that ends at a fresh tree. A new tree's first reward is drawn from $\mathcal{N}(15, 1)$, clipped to $[0.5, 20]$. Every harvest depletes the tree multiplicatively by a factor drawn from $\text{Beta}(14.91, 2.03)$ (mean 0.88), with a floor of 0.5. Each round lasts a fixed 240 s, so time spent travelling is time not spent harvesting, and the optimal policy leaves a tree once its reward falls below the environment's average reward rate (Marginal Value Theorem).

**Data.**
- $A = 2$ (harvest=0, exit=1); rewards are given only for harvests (partial feedback) and are min-max normalized over the received rewards (an exit counts as 0; the largest harvest is 18.6 apples).
- 250 participants (Prolific), 8 rounds each (`overall_round`; 40–112 decisions per round, mean 82), 340–864 decisions per participant.
- Each round has one effort condition: travel task (physical: rapid key pressing; cognitive: MSIT) × effort level (low, high), each tested in two rounds. `prepare_dataframe` codes it as the binary inputs `is_msit` and `is_high`; participants knew the round's condition from the background colour.
- Additional inputs: `harvest_duration` (2 s) and `travel_duration` (8.33 s), both constant, and `is_msit`, `is_high`.
- Held-out test blocks (rounds): 3, 6.

#### SPICE Model (`studies/bustamante2023/spice_bustamante2023.py`)

The shared RL architecture (Section 1.4) with these deviations:
- **All three value states plus a leave threshold.** Working memory (`value_wm_reward_chosen`, `value_wm_reward_not_chosen`), the incremental reward value (`value_reward_chosen`, `value_reward_not_chosen`) and choice perseveration (`value_choice`), plus a leave threshold (`leave_threshold`).
- **Working memory keeps its state.** Unlike the shared architecture, `value_wm_reward_chosen` is not stateless: it sees the harvest value it updates, like `value_reward_chosen`. Rewards are continuous here, so their squared terms stay in the library.
- **Only the harvest item carries values.** The exit item stays at 0 as the reference point, which mirrors the structure of the Marginal Value Theorem.
- **Harvest/exit replace chosen/unchosen.** The `*_chosen` module updates the harvest value after a harvest (input: the harvest reward), and the `*_not_chosen` module updates it after an exit (a new patch).
- **Choice value** is restricted to the harvest item and receives the harvest indicator $\mathbb{1}[\text{harvested}]$. It acts as a continuation tendency.
- **Leave threshold.** A stateless module (reset every trial, harvest item only) on the round's effort condition, `leave_threshold` $= f(\texttt{is\_msit}, \texttt{is\_high})$. With two binary inputs, its library reduces to
  $$\theta = b_0 + b_1\,\texttt{is\_msit} + b_2\,\texttt{is\_high} + b_3\,\texttt{is\_msit}\cdot\texttt{is\_high},$$
  the four per-condition thresholds of the benchmark's factorial design (Bustamante et al., 2023, Eqs. 2–5): $b_0$ is the threshold for low-effort key pressing, $b_1$ the shift for the cognitive travel task, $b_2$ the extra cost of high-effort travel, and $b_3$ how that extra cost differs between cognitive and physical effort. Through the participant embedding, every participant has their own effort costs; terms without support are pruned like any other.

$$\text{logits}_t = \big[V^{\text{wm}}_t + V^{\text{reward}}_t + V^{\text{choice}}_t - \theta_t, \; 0\big]$$

Because the harvest reward depletes within a patch, the working-memory and reward values track the expected next harvest reward, and the leave threshold sets the round's exit point, as the MVT's comparison of the expected reward with a per-condition threshold. The constant inputs `harvest_duration` and `travel_duration` are loaded but not used.

**GRU baseline.** Sees all four additional inputs (the constant durations scaled to $[0, 1]$, `is_msit`, `is_high`), the same information as SPICE and the benchmark.

#### Benchmark Model: Marginal Value Theorem with effort costs

**Reference:** Bustamante et al. (2023), Eqs. 2–5 (`EffortCostMVTModel`).

The harvest decision compares the expected reward of the next harvest with a leave threshold that is fixed per effort condition: the travel task (button pressing or MSIT) × effort level (low or high), each tested in two blocks. The threshold is the average reward rate net of the effort cost of travelling, so costlier travel lowers it and prolongs the stay in a tree. The action space is $A = 2$ (harvest=0, exit=1).

**Parameters (per participant, 5 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\beta$ | $> 0$ (exp) | Inverse temperature |
| $\rho_k$, $k = 1..4$ | unconstrained | Leave threshold per effort condition |

**Decision rule.**

$$P(\text{harvest}) = \frac{1}{1 + \exp[-\beta(R_e - \rho_k)]}, \qquad \rho_k = \frac{\sum r - n_{\text{travel}} \cdot c_k}{T}$$

$R_e = 0.88 \cdot \tfrac{1}{2}(r_t + r_{t-1})$, or $0.88 \cdot r_t$ if trial $t-1$ was a travel (the rule behind the authors' `expectedReward` column; 0.88 = mean depletion). On arrival at a fresh tree, $R_e$ is the expected yield of a fresh tree (15 apples). The decision for trial $t+1$ uses the reward of trial $t$.

**Deviations from the paper.**
- Fitted by maximum likelihood, the effort costs $c_k$ enter only through $\rho_k$, whose other terms are fixed per participant and condition. The model is therefore fitted with one free threshold per condition, which is equivalent; the costs follow as $c_k = (\sum r - \rho_k T)/n_{\text{travel}}$. This also avoids the block totals in $\rho_k$, which would be future information for trial-by-trial prediction.
- Every participant is fitted individually, like the other benchmarks; the original model was fitted hierarchically (Stan).
- On arrival at a fresh tree, the authors' `expectedReward` averages the travel's 0 into $R_e$, which makes the model predict leaving there while participants harvest 95% of the time; fitted that way, the thresholds collapse and leave decisions are barely discriminated (held-out trial likelihood 0.733 vs. 0.784).

### 3.4. Eckstein 2026: Drifting Multi-Armed Bandit

**Task.** A large-scale restless four-armed bandit (Eckstein et al., 2026). On each trial participants choose one of 4 options, arranged in a circle on the screen, and see the points (0–100) earned from the chosen option only. The options' mean payouts drift independently as mean-reverting Gaussian random walks:
$$\mu_{t,i} \sim \mathcal{N}\big(\lambda \mu_{t-1,i} + (1-\lambda) \cdot 50, \; \sigma_{\text{drift}}\big), \qquad \lambda = 0.9836, \; \sigma_{\text{drift}} = 2.8,$$
and the observed payout is noisy: $r_{t,i} \sim \mathcal{N}(\mu_{t,i}, \sigma_{\text{obs}})$ with $\sigma_{\text{obs}} = 4$. Because the best option changes over time, participants have to trade off exploiting the currently best option against exploring options whose values may have drifted upward. The circular layout makes spatial structure in switching (to the adjacent or the opposite option) a candidate behavioral mechanism.

**Data.**
- $A = 4$; payouts normalized to $[0, 1]$ (steps of 0.01); the payouts of all four options (`payout_1..4`) are recorded but only the chosen one is given to the models.
- 862 participants; blocks of 150 trials; 786 participants completed 5 blocks (750 trials) and 76 completed 3 blocks (450 trials).
- Held-out test block: 2 (76 of 862 participants have only blocks 1–3; the task is simple, so little task learning needs to be excluded).

#### SPICE Model (`studies/eckstein2026/spice_eckstein2026.py`)

The full shared RL architecture (Section 1.4) over 4 arms, with one addition:

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|-----------|----------------|---------------|-------------|-----------------|-------------|
| `bias_attention` | `is_adjacent`, `is_opposite` | `bias_attention` (initial: 0) | all items | True | Spatial attention bias by circular distance to the chosen arm |

`is_adjacent` and `is_opposite` are binary indicators of each arm's circular distance to the chosen arm (distance 1 and distance 2 for 4 arms). They are mutually exclusive, so their product term is pruned along with their squares.

$$\text{logits}_t = V^{\text{wm}}_t + V^{\text{reward}}_t + V^{\text{choice}}_t + B^{\text{attention}}_t$$

#### Benchmark Model: Best RL (Eckstein et al., 2026)

**Reference:** Eckstein et al. (2026), *Nature Human Behaviour* 10:972–987, "Q-learning model architectures". Implemented as `BestRLModel` in `benchmarking_eckstein2026.py`.

The winner of the systematic comparison of 48 Q-learning variants in the original study. It has a fixed learning rate; the Pearce-Hall variable learning rate belongs to other variants and is not part of this model. The action space is $A = 4$. Raw parameters are clipped to $[-5, 5]$ before transformation.

**Parameters (per participant, 6 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\alpha$ | $(0.01, 0.99)$ sigmoid | Learning rate |
| $\beta$ | $(0, 20)$ softplus | Inverse temperature |
| $f$ | $(0, 0.99)$ sigmoid | Forgetting rate toward $Q_{\text{init}}$ |
| $\kappa$ | $(-1, 1)$ tanh | Perseveration bonus on the previous choice |
| $b$ | $(-1, 1)$ tanh | Linear update bias (added to match the flexibility of the original study's neural-network models) |
| $Q_{\text{init}}$ | unconstrained | Initial value of all arms |

**Trial-by-trial update**, in this order:

1. Value update of the chosen arm only, with update bias (their Eq. 2):
$$Q(a_t) \leftarrow Q(a_t) + \alpha \big(r_t - Q(a_t)\big) + b$$
2. Forgetting of all arm values toward the initial value (their Eq. 5):
$$Q(a) \leftarrow (1 - f) \, Q(a) + f \, Q_{\text{init}} \quad \forall a$$
3. Perseveration (their Eq. 4): $c(a) = \kappa$ for the arm just chosen, 0 otherwise.
4. Additive choice rule with a single inverse temperature (their Eq. 3):
$$p(a_{t+1}) = \text{softmax}\big(\beta \, (Q + c)\big)$$


### 3.5. Dezfouli 2019: Two-Armed Bandit (Clinical)

**Task.** A two-armed bandit with a sparse reward schedule (Dezfouli et al., 2019), used to compare learning and choice mechanisms between healthy participants and patients with depression or bipolar disorder. In each block, participants repeatedly choose between two keys with binary rewards. One response paid with probability 0.25, 0.125 or 0.08 (four blocks each per participant) and the other with 0.05; which response was better and the order of the conditions varied across participants and blocks. Rewards are rare and the contrast between arms is small in most blocks. Choices therefore depend strongly on reward history, choice history (perseveration) and their interaction, which is what the original study identified as differing between groups.

**Data.**
- $A = 2$; binary rewards for the chosen arm only.
- 101 participants: 34 healthy controls, 34 with depression, 33 with bipolar disorder (`diag`).
- 12 blocks per participant with a variable number of trials (6–202 per block, median 108); 398–2021 trials per participant.
- Held-out test blocks: 3, 6, 9.

**Reward probabilities used for simulations (to be reported in the SI).** The published data do not record which reward probabilities a block had, and the block number is only the presentation order (each participant has every condition twice, in varying positions). The authors' raw data (github.com/adezfouli/rnn_beh, `data/BD/choices_diagno.csv.zip`, Apache License 2.0; the unpacked csv is kept unchanged as `studies/dezfouli2019/data_raw/choices_diagno.csv`) contain the condition of every block. Conditions 1–2, 3–4 and 5–6 give the better response a reward probability of 0.25, 0.125 and 0.08; the other response pays 0.05 (empirical reward rates 0.247 / 0.122 / 0.082 and 0.055). The better response is the one marked as best in the data. The published, anonymized data were matched to the raw data by each participant's complete trial sequence (block, key, reward, better response), which identifies all 101 participants uniquely, and the condition was transferred to every block (`add_reward_probabilities`; looked up per participant and block by `EnvironmentDezfouli2019`). These probabilities are used only where the task is simulated (generative comparison, rollout evaluation); fitting and one-step evaluation use the observed rewards. Note for the SI: the authors' own simulation code assumed a fixed probability pair per block number, which does not hold in their data. Before the raw data were found, the probabilities had been reconstructed from the published data alone (better response from the best-response indicator, level by an exact binomial posterior over all level assignments consistent with four blocks per level); this reconstruction matched the raw-data conditions in 99.3% of the 1,212 blocks.

#### SPICE Model (`studies/dezfouli2019/spice_dezfouli2019.py`)

The full shared RL architecture (Section 1.4) without deviations: 5 submodules on 3 memory states, with binary rewards.

#### Benchmark Model: Generalized Q-Learning (GQL)

**Reference:** Dezfouli et al. (2019). Implemented as `GQLModel` in `benchmarking_dezfouli2019.py`.

The Generalized Q-Learning model extends standard Q-learning by maintaining $d$-dimensional Q-values and choice history vectors per action, plus an interaction matrix that captures how choice history modulates value sensitivity. In the implementation, $d = 2$. The action space is $A = 2$.

**Parameters (per participant, per dimension $d$):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\phi_d$ | $(0.01, 0.99)$ sigmoid | Learning rate for Q-values in dimension $d$ |
| $\chi_d$ | $(0.01, 0.99)$ sigmoid | Learning rate for choice history in dimension $d$ |
| $\beta_d$ | $(0.1, 10)$ softplus | Q-value weight in dimension $d$ |
| $\kappa_d$ | $(-10, 10)$ | Choice history weight in dimension $d$ |
| $C_{d,d'}$ | $(-10, 10)$ | $d \times d$ interaction matrix between history and Q-values |

**Total parameters per participant:** $4d + d^2 = 12$ (with $d = 2$).

**Update rules.** At each trial, given chosen action $a_t$ and reward $r_t$:

$$Q_{a_t,d} \leftarrow (1 - \phi_d) \cdot Q_{a_t,d} + \phi_d \cdot r_t$$
$$Q_{a \neq a_t,d} \leftarrow (1 - \phi_d) \cdot Q_{a \neq a_t,d}$$
$$H_{a_t,d} \leftarrow (1 - \chi_d) \cdot H_{a_t,d} + \chi_d$$
$$H_{a \neq a_t,d} \leftarrow (1 - \chi_d) \cdot H_{a \neq a_t,d}$$

**Action values:** The value of each action $a$ combines weighted Q-values, weighted choice history, and their interaction:

$$V_a = \sum_d \beta_d \cdot Q_{a,d} + \sum_d \kappa_d \cdot H_{a,d} + H_a^\top C \, Q_a$$

where $H_a, Q_a \in \mathbb{R}^d$ are the history and Q-value vectors for action $a$, and $C \in \mathbb{R}^{d \times d}$ is the interaction matrix. Action probabilities are computed via softmax over $V_a$.


### 3.6. Ganesh 2024a: Perceptual Contrast Bandit

**Task.** A perceptual-uncertainty bandit (Ganesh et al., 2024) that combines perceptual decision making with reward learning. On each trial, two Gabor patches of different contrast appear on the left and right, and participants choose one side. The reward does not depend on the side itself but on the latent perceptual state, i.e. which patch has the higher contrast. Choosing the side that matches the rewarded state pays a binary reward with probability $\mu = 0.75$, and choosing the other side with probability $1 - \mu$. Participants have to learn this contingency (whether the high- or the low-contrast patch is the "good" one) while their perception of which patch has higher contrast is itself uncertain. The signed contrast difference varies continuously from trial to trial ($|\Delta c| \le 0.45$), so some trials are perceptually easy and others are close to chance. Normatively, a participant should learn less from the outcomes of trials on which they could not tell the patches apart, i.e. weight learning by perceptual confidence.

**Data.**
- $A = 2$ (left, right); binary rewards for the chosen side only (65% of trials rewarded).
- 98 participants, 12 blocks × 25 trials = 300 trials each.
- Additional input `contrast_difference` (signed: left minus right); `prepare_dataset` adds the next trial's value and drops each session's last trial.
- Held-out test blocks: 3, 6, 9.
- Simulations replay each trial's recorded contrast difference and both sides' outcomes (`reward_left`, `reward_right`; `get_reward_schedule`), because the block's contingency is not stored in the data.

#### SPICE Model (`studies/ganesh2024a/spice_ganesh2024a.py`)

The shared RL architecture (Section 1.4) with these deviations:
- **No working-memory state.** The incremental reward value (`value_reward_chosen`, `value_reward_not_chosen`) and choice perseveration (`value_choice`), plus a perception module.
- **Item space.** Values live in a contrast-based item space (low-contrast=0, high-contrast=1) instead of the left/right action space (see below).
- **Perception module.** A stateless module maps the contrast difference magnitude to a perceptual certainty:

| Submodule | Control signals | State updated | `include_state` | Description |
|-----------|----------------|---------------|-----------------|-------------|
| `perception_certainty` | $\lvert\Delta c\rvert$ | — (output passed through a sigmoid) | False | Perceptual certainty from contrast difference magnitude |

- **Certainty as an extra input.** Both reward modules additionally receive the current certainty $\text{cert}_t$ (`value_reward_chosen`: $r_t$, $\text{cert}_t$; `value_reward_not_chosen`: $\text{cert}_t$). The choice module receives the item-space choice indicator $c^{\text{item}}_t$ together with $\text{cert}_t$ and the next trial's certainty $\text{cert}_{t+1}$.

**Key design feature: item-space/action-space decoupling.** The model represents values in item space (low-contrast, high-contrast) rather than action space (left, right), because the stimulus-to-position mapping changes every trial:
1. Actions are remapped from position space to item space using the sign of the contrast difference: when $\Delta c \leq 0$, left=low and right=high; when $\Delta c > 0$, left=high and right=low.
2. Learning updates operate in item space with these deterministic masks; certainty enters as an input, so updates can be attenuated when the assignment is unreliable.
3. At decision time, item-space logits are mapped back to action space using the perceptual certainty for the next trial's contrast.

**Perceptual certainty.** $\text{cert}_t = \sigma(f(\lvert\Delta c_t\rvert))$ and $\text{cert}_{t+1} = \sigma(f(\lvert\Delta c_{t+1}\rvert))$, the same module evaluated on the current and on the next trial's contrast difference.

**Logit computation (soft item-to-action mapping):**
$$V^{\text{item}}_t = V^{\text{reward}}_t + V^{\text{choice}}_t$$
$$\text{cert}'_{t+1} = \text{cert}_{t+1} / 2 + 0.5 \quad \text{(rescaled to [0.5, 1.0])}$$
$$V^{\text{mixed}}_t = \text{cert}'_{t+1} \cdot V^{\text{item}}_t + (1 - \text{cert}'_{t+1}) \cdot \text{flip}(V^{\text{item}}_t)$$
$$\text{logits}_t = \begin{cases} V^{\text{mixed}}_t & \text{if } \Delta c_{t+1} < 0 \\ \text{flip}(V^{\text{mixed}}_t) & \text{if } \Delta c_{t+1} \geq 0 \end{cases}$$

When perceptual certainty is high, item values map cleanly to action values; when certainty is low, the mapping is more uniform.

**SINDy library degree:** 2.

#### Benchmark Model: Bayesian Belief-Update Model

**Reference:** Ganesh et al. (2024), normative agent model. Implemented as `BayesianModel` in `benchmarking_ganesh2024a.py`.

A Bayesian observer that maintains a discretized belief distribution over the contingency parameter $\mu \in [0, 1]$, which links the latent perceptual state to reward probability. The model performs exact posterior updating given its perceptual noise model. The action space is $A = 2$ (left, right).

**Parameters (per participant, 2 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\beta$ | $(1, 25)$ softplus + 1 | Inverse softmax temperature |
| $\sigma$ | $(0.01, 0.1)$ sigmoid-scaled | Perceptual noise (std. dev. of noisy observation) |

**Generative model:**
1. $\mu \sim P(\mu)$: prior over contingency parameter (initially uniform over $N = 100$ grid points)
2. $s \sim \text{Bernoulli}(\mu)$: latent state determining which side is rewarded
3. $o_t | s \sim \mathcal{N}(\Delta c, \sigma^2)$: noisy perceptual observation of contrast difference
4. $P(r = 1 | a = s) = \mu$; $P(r = 1 | a \neq s) = 1 - \mu$

**Perceptual belief computation.** Given observed contrast difference $\Delta c$ and perceptual noise $\sigma$, the model computes the probability that each state is true using a truncated normal model:

$$\pi_0 = P(\text{state}=0 | o_t) = \frac{\Phi(0; \Delta c, \sigma) - \Phi(-\kappa_{\max}; \Delta c, \sigma)}{\Phi(\kappa_{\max}; \Delta c, \sigma) - \Phi(-\kappa_{\max}; \Delta c, \sigma)}$$

where $\Phi$ is the normal CDF and $\kappa_{\max} = 0.1$ is the maximum contrast difference.

**Bayesian belief update.** After observing reward $r_t$ for action $a_t$, the posterior over $\mu$ is updated multiplicatively:

$$q_0 = \pi_1 \cdot r + \pi_0 \cdot (1 - r)$$
$$q_1 = (2r - 1)(\pi_0 - \pi_1)$$
$$P(\mu | \text{history}) \propto P(\mu | \text{history}_{t-1}) \cdot (q_1 \cdot \mu + q_0)$$

where $r$ is recoded to reflect the state-0/action-0 contingency (flipped when action=1 is chosen).

**Action value computation.** Given the expected value of $\mu$: $E[\mu] = \sum_i P(\mu_i) \cdot \mu_i$, and the perceptual beliefs $\pi_0^{\text{next}}, \pi_1^{\text{next}}$ for the next trial's contrast:

$$V_{a=0} = (\pi_0^{\text{next}} - \pi_1^{\text{next}}) \cdot E[\mu] + \pi_1^{\text{next}}$$
$$V_{a=1} = (\pi_1^{\text{next}} - \pi_0^{\text{next}}) \cdot E[\mu] + \pi_0^{\text{next}}$$
$$\text{logits} = \beta \cdot [V_{a=0}, V_{a=1}]$$


### 3.7. Kolff 2025: Chimpanzee Grooming Negotiation

**Task.** An observational dataset, not an experiment: video-coded grooming interactions between pairs of wild chimpanzees from two communities (CE and WE). Grooming is preceded by a *negotiation* phase, in which the two apes exchange behaviors that may lead to grooming and decide who grooms whom. Each interaction is coded as a sequence of events; in each event, either ape (or occasionally both) performs a behavioral element. The raw elements (6334 events, 311 interactions, 41 apes) are recoded into one *Groom* category and five *Negotiation* categories (agreed with the primatologists; see [studies/kolff2025/data/PREPROCESSING.md](studies/kolff2025/data/PREPROCESSING.md)):
- **Self_Reposition**: the ape changes its own posture without presenting a body part.
- **Reposition_Body**: the ape moves the partner into a position for grooming (`touch`, `grab-pull limb`, `push`, `touch hold`).
- **Grooming_Process**: `maintain contact`, i.e. a hand kept on the partner between grooming bouts.
- **Grooming_Solicitation**: visual or tactile requests to be groomed (`present_*`, `raise *`, `extend *`, `kiss`, `hold`).
- **Directed_Scratch**: a scratch directed at a body part, as a request.

Every interaction is modeled from the perspective of each of its two apes in turn (the *focal* ape and its *partner*), so each ape is a participant and each (interaction, perspective) pair is a block; both perspectives of an interaction share its id and land on the same side of a split. The question is which acts of both apes build up the focal ape's readiness to groom.

#### SPICE Model: Received and Given Acts (`studies/kolff2025/spice_kolff2025.py`)

The model predicts whether the focal ape's next act is grooming. A target exists only at events where the focal ape acts next; all other events are input steps only (target mask packed into `ys` by `prepare_dataset`, masked cross-entropy `masked_cross_entropy`).

**Data** (`data/kolff2025.csv`, built by `benchmarking_kolff2025.build_dataframe`): 41 apes, 311 interactions, 11810 perspective-events, 5599 targets (median 129 per ape, range 1–312). P(groom) over targets is 0.41 (0.25 for the approached ape, 0.49 for the approacher). Held-out test set: whole dyads, fold 0 of 5 folds balanced by targets (`dyad_folds`, seed 0): 62 interactions, 1011 targets (18%), i.e. pairings the model was never fit on.

**Readout** ($A = 2$): `not groom` (reference, logit 0) and `groom`.

| Submodule | Control signals | State written | Description |
|-----------|----------------|---------------|-------------|
| `negotiation_received` | partner's Grooming_Solicitation, Self_Reposition, Directed_Scratch | `value_negotiation_received` | Negotiation by the partner |
| `negotiation_given` | own Grooming_Solicitation, Self_Reposition, Directed_Scratch | `value_negotiation_given` | Negotiation by the focal ape |
| `grooming_received` | partner's Reposition_Body, Grooming_Process, Groom | `value_grooming_received` | Preparation and grooming by the partner |
| `grooming_given` | own Reposition_Body, Grooming_Process, Groom | `value_grooming_given` | Preparation and grooming by the focal ape |

Control signals are one-hot indicators of the category each ape performs at the current event (all zero if it does not act). An ape performs at most one category per event, so products of two categories of the same module are structurally zero and pruned, as are squared indicators. Each module updates its own state, which it uses in its update; all four write only the `groom` item. Rank and role are not inputs: on held-out dyads they added no predictive information beyond the act sequence.

**Logit computation:** $\text{logit}^{\text{groom}}_t = \sum_{m} \text{value}_{m,t}$, $\text{logit}^{\text{not groom}}_t = 0$.

**SINDy library degree:** 1 (linear: per module $\{1, \text{value}_m, 3 \text{ act indicators}\}$, 20 candidate terms in total).

**Group model.** One model is fitted for all apes (`n_participants=1`). With about 110 targets per ape, individual models (participant embeddings, and per-ape benchmarks alike) overfit and fail on held-out dyads. Gate and feature penalties are off: in a group model the gates are global constants and duplicate SINDy's sparsity.

#### Benchmark Model: Lag-1 Logistic Regression (`LogisticRegressionModel`)

A memoryless group model: $\text{logit}\, P(\text{groom}_{t+1}) = w^\top a_t + b$, where $a_t$ are the 12 act indicators of the current event (the partner's and the focal ape's category), i.e. the same inputs as SPICE without memory beyond the current event. Fitted on the scored targets (near-unpenalized, $C = 10^4$): 13 parameters shared by all apes. Whatever SPICE gains over it is attributable to latent dynamics. No generative benchmarking: the models predict only whether the next act is grooming, not which act follows, so they cannot generate interaction sequences.


### 3.8. Bruckner 2025: Helicopter Task (Predictive Inference)

**Task.** A predictive-inference ("helicopter") task from a lifespan study of learning (Bruckner, Nassar, Li & Eppinger, 2025, follow-up experiment). A hidden helicopter drops bags at positions $x_t \sim \mathcal{N}(\mu_t, \sigma^2)$ on a horizontal screen of 300 pixels ($\sigma = 17.5$ pixels in the data's `sigma` column). On every trial, participants place a bucket at $b_t$ to catch the next bag. The bag counts as caught when $|b_t - x_t| \le \sigma/2$ (27% of trials). The helicopter position $\mu_t$ stays fixed for a while and then jumps to a new location at change points ($c_t$, about 10% of trials). Participants therefore have to update strongly after surprising outcomes that signal a change point, and weakly after outcomes that are just noise. On about 10% of trials the helicopter is visible ($v_t = 1$) and reveals $\mu_t$ directly. Each bag contains a coin of high (gold, $r_t = 1$) or low (stone, $r_t = 0.25$) value, which tests whether reward value biases the learning rate.

The experiment has two conditions. In *noPush* the bucket stays where the participant left it. In *push* the bucket is displaced to a random starting position $z_{t+1}$ before the next prediction, and participants have to move it back. This measures anchoring: an incomplete correction toward the displaced position, which the original study links to resource-rational (effort-saving) computations.

**Data.**
- Continuous scalar output ($A = 1$): the next bucket position $b_{t+1}$, trained with MSE; all positions normalized to $[0, 1]$ by dividing by 300.
- 90 participants in three age groups (31 children, 25 younger adults, 34 older adults), each completing both conditions: 2 blocks of 100 trials in noPush (blocks 1–2) and 2 in push (blocks 3–4), i.e. 400 trials per participant. Condition is coded as `experiment`.
- Held-out test blocks: 2, 4 (the second block of each condition: both conditions are tested, the first block of each is left for learning and calibration; 50% of the data, an exception to the 20–30% rule).

#### SPICE Model (`studies/bruckner2025/spice_bruckner2025.py`)

A delta-rule model with a learned learning rate and an anchoring term: 2 submodules on 2 memory states. The learning rate carries all learning dynamics (surprise, catch, visibility), so its equation is the only place where the gain of the update lives; there is no separate belief module whose gain could trade off against it.

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|-----------|----------------|---------------|-------------|-----------------|-------------|
| `learning_rate_update` | $\lvert\delta_t\rvert$, `catch`, $v_t$ | `learning_rate_value` | — | True | Learning-rate dynamics, updated every trial |
| `anchor_update` | $y_t$ | `anchor_value` | — | False | Anchoring correction for bucket displacement |

**Memory states:**
- `learning_rate_value` (initial: 0): $\alpha_t = \sigma(\text{learning\_rate\_value}_t)$, initially 0.5.
- `anchor_value` (initial: 0): per-trial anchoring correction, reset to 0 every trial.

**Control signal preprocessing** (`prepare_dataframe`, `forward`):
- Prediction error relative to the current bucket position: $\delta_t = x_t - b_t$ (subjective, as in the benchmark). The module receives $\min(\lvert\delta_t\rvert / 0.5, 1)$, which spreads the PE magnitude (mostly < 0.2 in screen units) over $[0, 1]$.
- Anchor shift: $y_t = \text{clip}\big((z_{t+1} - b_t) / 0.3, -1, 1\big)$, the displacement between the next trial's initial bucket position and the current bucket position. It is 0 in the noPush condition.
- `catch` $\in \{0, 1\}$: whether the bag was caught; $v_t \in \{0, 1\}$: whether the helicopter is visible. Both are binary, so their squared terms are pruned.
- The true helicopter position $\mu_t$ is masked to NaN when $v_t = 0$; participants can only observe it when visible.
- The coin value $r_t$, `sigma` and $c_t$ are loaded but not used by the SPICE model.

**Output (delta rule + anchoring):**
$$\text{target}_t = \begin{cases} \mu_t & \text{if } v_t = 1 \\ x_t & \text{otherwise} \end{cases}$$
$$\hat{b}_{t+1} = b_t + \alpha_t \cdot (\text{target}_t - b_t) + V^{\text{anchor}}_t$$

The predicted next bucket position moves from the current bucket position toward the outcome (or toward the helicopter when it is visible) by the learned learning rate, plus the anchoring correction.

**SINDy library degree:** 2.

#### Benchmark Model: Reduced Bayesian Model (RBM)

**Reference:** Bruckner et al. (2025), Eqs. 4-14. Implemented as `RationalResourceModel` in `benchmarking_bruckner2025.py` (a reimplementation of `AlAgentRbm.py` and the bucket bias of `al_task_agent_int_rbm.py` from the original codebase).

A reduced Bayesian model that maintains a belief about the helicopter's position and an uncertainty estimate that modulates learning rate. The belief is reset to the participant's actual prediction $b_t$ at each trial (subjective prediction errors), while uncertainty dynamics carry forward across trials. On catch trials (helicopter visible), the belief is additionally updated with the observed helicopter position. The output is a continuous scalar prediction ($A = 1$).

**Parameters (per participant, 5 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $h$ | $(0, 1)$ sigmoid | Hazard rate: prior probability of a change point |
| $s$ | $(0, 1)$ sigmoid | Surprise sensitivity: modulates change-point detection |
| $u$ | unconstrained (exponentiated, clamped at $u \le 10$) | Uncertainty underestimation: $\exp(u)$ divides posterior uncertainty |
| $q$ | unconstrained | Reward bias: learning rate increase for high-value trials |
| $d$ | unconstrained | Bucket bias: anchoring on the pushed bucket position |

**Fixed task parameters:**
- $\sigma = 15$ pixels: outcome noise standard deviation
- $\sigma_H = 15$ pixels: helicopter cue noise standard deviation (used on catch trials)

(Note: `SIGMA = 15 / 300` is hard-coded in `benchmarking_bruckner2025.py`, while the data's `sigma` column is 17.5 pixels. To be checked.)

**State variables (per session):**
- $\hat{\sigma}^2_t$: estimation uncertainty (initialized to $100 / 300^2$)
- $\tau_t$: relative uncertainty (initialized to 0.5)

**Trial-by-trial update (Eqs. 4-14):**

1. **Prediction error** (subjective, using participant's actual bucket position):
$$\delta_t = x_t - b_t$$

2. **Change-point probability** (Eq. 8):
$$\sigma^2_{\text{total}} = \hat{\sigma}^2_t + \sigma^2$$
$$\ell_t = \mathcal{N}(\delta_t; 0, \sigma^2_{\text{total}})$$
$$\omega_t = \frac{h}{\ell_t^s \cdot (1 - h) + h}$$

3. **Learning rate with reward bias** (Eq. 7):
$$\alpha_t = \text{clamp}(\omega_t + \tau_t - \tau_t \cdot \omega_t + q \cdot \mathbb{1}[r_t \geq 1], \; 0, \; 1)$$

where $r_t$ is the coin value and $\mathbb{1}[r_t \geq 1]$ is a binary indicator for high-value trials.

4. **Belief update** (Eqs. 4-5):
$$\mu_{t+1} = b_t + \alpha_t \cdot \delta_t$$

5. **Catch trial update** (Eqs. 11-14, applied only when helicopter is visible, $v_t = 1$):
$$w_t = \frac{\hat{\sigma}^2_t}{\hat{\sigma}^2_t + \sigma_H^2} \quad \text{(Eq. 12)}$$
$$\mu_{t+1} = (1 - w_t) \cdot \mu_{t+1} + w_t \cdot \mu_H \quad \text{(Eq. 11)}$$
$$C = \frac{1}{1/\hat{\sigma}^2_t + 1/\sigma_H^2} \quad \text{(Eq. 14)}$$
$$\tau_t = \frac{C}{C + \sigma^2} \quad \text{(Eq. 13)}$$

where $\mu_H$ is the true helicopter position revealed on catch trials and $w_t$ weights the helicopter cue against the model's own estimate based on their relative uncertainties.

6. **Predicted next position** (Eq. 4, with the bucket bias):
$$\hat{b}_{t+1} = \text{clamp}(\mu_{t+1} + d \cdot y_t, 0, 1), \qquad y_t = z_{t+1} - b_t$$

where $y_t$ is the push, set to 0 on the first trial of a block (as in the original).

7. **Uncertainty update** (Eq. 9):
$$\hat{\sigma}^2_{t+1} = \frac{\omega_t \cdot \sigma^2 + (1 - \omega_t) \cdot \tau_t \cdot \sigma^2 + \omega_t(1 - \omega_t)(\delta_t(1 - \tau_t))^2}{\exp(u)}$$

8. **Relative uncertainty update** (Eq. 10):
$$\tau_{t+1} = \frac{\hat{\sigma}^2_{t+1}}{\hat{\sigma}^2_{t+1} + \sigma^2}$$

The model dynamically adjusts its learning rate via change-point detection: when a change point is detected (high $\omega_t$), the learning rate increases to rapidly incorporate the new outcome; during stable periods (low $\omega_t$), the learning rate decreases for more precise estimation. The uncertainty underestimation parameter $u$ captures individual differences in confidence calibration.


### 3.9. Weber 2024: Laser Tracking (Shield Movement)

**Task.** A continuous circular tracking task (Weber et al., 2024), a circular analog of the helicopter task (Bruckner 2025). Participants control a shield on a circular track (0°–360°) around a central source that fires laser beams. Each beam's position is drawn around a hidden mean, and the mean jumps by ±20°, ±30° or ±40° at change points. The shield is moved continuously with button presses at a fixed speed of 1°/frame. Between two beams there are 1–90 frames (median 17), so the shield cannot always reach the believed laser position in time. A beam is caught if it lands within ±10° of the shield's center at the beam's onset, which happens on 35% of beams.

Two factors are crossed within participants:
- **Volatility** (change-point rate): about 3.3% of beams in low-volatility blocks vs. about 9% in high-volatility blocks.
- **Stochasticity** (observation noise): the s.d. of beams around the mean is about 10° in low-stochasticity blocks vs. about 20° in high-stochasticity blocks.

The participant has to infer whether a deviating beam is noise or a change point, and adjust their learning rate under both kinds of uncertainty.

The model works with events rather than frames. Each event is one laser beam, and the model predicts the shield position at the next beam. This is a continuous 2D prediction in $(\sin, \cos)$ space, trained with a custom clamped angular MSE loss instead of cross-entropy.

**Data.**
- 30 participants; 4 conditions (`experiment` 0–3 = volatility × stochasticity: low/low, low/high, high/low, high/high), 4 blocks per condition, i.e. 16 blocks of 516–565 events (8590 events per participant).
- Additional inputs: `laser_caught` (scored on the beam's onset frame, so it does not contain the participant's subsequent movement), `volatility`, `stochasticity` (binary), `trial_duration_frames` (inter-beam interval), `trueMean` (diagnostics only, never a model input).
- Held-out test blocks: 3, 5, 9, 12 (0-indexed; every volatility x stochasticity cell once per participant).
- SPICE is fitted with `n_experiments=1`: the conditions enter as module inputs, so all conditions share one coefficient set.

**Data representation.** Shield and laser positions are encoded as $(\sin\theta, \cos\theta)$ pairs (`prepare_dataframe`): the shield position is the action, the laser position the outcome ($A = 2$). Inside the model they are converted back to angles in degrees, so all updates are rotation-invariant and stay on the circle.

#### SPICE Model (`studies/weber2024/spice_weber2024.py`)

An angular delta-rule model (analogous to Bruckner 2025): a learned learning rate on a latent belief, and a learned movement gain that carries the shield toward it. 2 submodules on 3 memory states.

| Submodule | Control signals | State updated | `include_state` | Description |
|-----------|----------------|---------------|-----------------|-------------|
| `learning_rate_update` | $\lvert\delta_t\rvert$, `caught`, `volatility`, `stochasticity` | `learning_rate_value` | True | Learning-rate dynamics, updated every beam |
| `movement_update` | $\lvert d_t\rvert$ | `movement_value` | False | Movement gain toward the belief, given the belief-shield distance |

**Memory states:**
- `belief_value`: belief about the laser mean in degrees, initialized to the first beam of a block. Not written by a module: updated by the hardcoded delta rule below.
- `learning_rate_value` (initial: 0): $\alpha_t = \sigma(\text{learning\_rate\_value}_t)$, initially 0.5.
- `movement_value` (initial: 0): per-beam movement gain $g_t = \sigma(\text{movement\_value}_t)$, reset to 0 every beam.

**Control signal preprocessing:**
- Prediction error: $\delta_t = \text{wrap}(\theta^{\text{laser}}_t - \hat{\mu}_t)$, the shortest arc from the belief to the beam, wrapped to $[-180°, 180°)$. The module receives $\min(\lvert\delta_t\rvert / 90°, 1)$.
- Belief-shield distance: $d_t = \text{wrap}(\hat{\mu}_{t+1} - \theta^{\text{shield}}_t)$, computed after the belief update. The module receives $\min(\lvert d_t\rvert / 90°, 1)$.
- `caught`, `volatility`, `stochasticity` $\in \{0, 1\}$; their squared terms are pruned.

**Belief update and output:**
$$\hat{\mu}_{t+1} = \big(\hat{\mu}_t + \alpha_t \cdot \delta_t\big) \bmod 360°$$
$$\hat{\theta}^{\text{shield}}_{t+1} = \theta^{\text{shield}}_t + g_t \cdot d_t$$

The output is $(\sin, \cos)$ of $\hat{\theta}^{\text{shield}}_{t+1}$: the shield moves part of the way from its current position toward the updated belief.

**Custom loss function: Clamped Angular MSE** (`clamped_angular_mse`). Because the shield has a finite movement speed, the predicted position may be physically unreachable within the inter-beam interval. The loss clamps the predicted movement:

$$\Delta = \hat{s}_{t+1} - s_t, \quad f = \min\left(\frac{v \cdot \Delta t \cdot (\pi / 180)}{||\Delta|| + \epsilon}, 1\right)$$
$$\hat{s}^{\text{clamped}}_{t+1} = s_t + f \cdot \Delta$$
$$\mathcal{L} = \text{MSE}(\hat{s}^{\text{clamped}}_{t+1}, s^{\text{actual}}_{t+1})$$

where $s_t$ is the current shield position in $(\sin, \cos)$, $v = 1$ °/frame and $\Delta t$ is the inter-beam interval in frames. `prepare_dataset` packs the targets as $[s^{\text{actual}}_{t+1}, s_t, \Delta t]$ so the loss has the current position and timing information.

**SINDy library degree:** 2.

#### Benchmark Model: Bayesian Changepoint Observer

**Reference:** Foucault et al. (2026), *eLife* 15:RP110137, changepoint model with variance inference and expected-catch response. Implemented as `ChangePointModel` in `benchmarking_weber2024.py` (grid step `h_step = 2`°).

Every beam runs three computations on a discretized circular grid ($[0°, 360°)$ in 2° steps):
1. **Inference.** Posterior over the latent state (laser mean × observation-noise level) given the beam, with a circular normal likelihood per noise level.
2. **Prediction.** Propagation through the changepoint transition: $\text{prior}_{t+1} = (1 - p_{\text{cp}}) \cdot \text{posterior} + p_{\text{cp}} \cdot \text{uniform}$ over the mean, keeping the noise level (the level is constant within a block). As in the paper, a changepoint redraws the mean uniformly from the circle: the task's actual jump sizes are unknown to participants.
3. **Response.** The expected catch probability of every shield location, $P(\lvert\text{next beam} - \text{location}\rvert \le 10°)$, is the circular convolution of the predictive distribution with a catch kernel (computed via FFT). The location with the highest expected catch is selected (soft argmax, `choice_sharpness = 50`, to stay differentiable) and approached along the shortest arc, as far as reachable within the inter-beam interval. The shield moves with response probability $\text{HalfNormalCDF}(\lvert\text{distance}\rvert; \sigma_{\text{resp}})$.

The output is the expected next shield position in $(\sin, \cos)$, mixing staying and moving by the response probability, fitted with MSE. In generative simulations the move is drawn instead.

**Parameters (per participant, 4 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $p_{\text{cp}}$ | $(0, 1)$ sigmoid | Assumed changepoint probability per beam |
| $\sigma_{\text{resp}}$ | $> 0$ (exp) | S.d. (degrees) of the response-probability CDF |
| $\sigma_{\text{obs},1}, \sigma_{\text{obs},2}$ | $> 0$ (exp) | Observation-noise s.d. (degrees) of the two noise levels |

**Deviations from the paper.** One update per beam, with the response clamped to the reachable arc; the paper's fixed hazard rate ($1/16$) and response s.d. (8°) are fitted per participant; the argmax over locations is a soft argmax.

---

## 4. Summary Table

| Study | Task | $A$ | SPICE model | Modules | States | SINDy Degree | Benchmark Model | Benchmark Params/Participant |
|-------|------|-----|-------------|---------|--------|--------------|----------------|------------------------------|
| Synthetic | Q-learning recovery | 2 | `spice.precoded.choice` | 3 | 2 | 2 | — (ground truth) | — |
| Braun 2018 | Voluntary task switching | 2 | `spice_braun2018` | 6 | 3 | 2 | Expected Value of Control | 3 |
| Bustamante 2023 | Patch foraging | 2 | `spice_bustamante2023` | 6 | 4 | 2 | MVT with effort costs (Bustamante 2023) | 5 |
| Eckstein 2026 | Drifting 4-armed bandit | 4 | `spice_eckstein2026` | 6 | 4 | 2 | Best RL (Eckstein 2026) | 6 |
| Dezfouli 2019 | Two-armed bandit (clinical) | 2 | `spice_dezfouli2019` | 5 | 3 | 2 | Generalized Q-Learning | 12 |
| Ganesh 2024a | Perceptual contrast bandit | 2 | `spice_ganesh2024a` | 4 | 2 | 2 | Bayesian Belief-Update | 2 |
| Kolff 2025 | Chimpanzee grooming negotiation (group model) | 2 | `spice_kolff2025` | 4 | 4 | 1 | Lag-1 logistic regression | 13 (shared) |
| Bruckner 2025 | Helicopter task (predictive inference) | 1 (continuous) | `spice_bruckner2025` | 2 | 2 | 2 | Reduced Bayesian Model | 5 |
| Weber 2024 | Laser tracking (shield movement) | 2 (continuous, sin/cos) | `spice_weber2024` | 2 | 3 (belief not module-written) | 2 | Bayesian Changepoint Observer (Foucault 2026) | 4 |

The GRU baseline (Section 2) is fitted for every study in addition to the study-specific benchmark.

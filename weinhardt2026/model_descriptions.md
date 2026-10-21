# Model Descriptions for SPICE Paper

This document provides full architectural descriptions of all SPICE models, benchmark models, and the GRU baseline used across the benchmark studies. It is intended as input for writing the methods/supplementary sections of the SPICE paper.

The study-to-model mapping follows the cluster runs in `slurm_jobs/spice_stability_studies.sh` (synthetic data: `slurm_jobs/spice_parameter_recovery.sh`); each section names the module that defines the model.

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
| Ganesh 2024a | ✓ | ✓ | ✓ | Values in contrast item space; perceptual-certainty module whose output is an extra input to all value modules; certainty-weighted item-to-action mapping |
| Bustamante 2023 | — | ✓ | ✓ | Only the harvest item carries values; harvest/exit take the roles of chosen/unchosen; exit item is a fixed 0 reference; stateless leave-threshold module on the effort condition |

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

**Hyperparameters.** Hidden size $d_h = 16$, dropout 0.1 applied after the input projection and after the GRU output. Trained with Adam on the same loss function as the SPICE model of the study.

**Total parameters.** The input dimension is $d_{\text{in}} = A + n_{\text{reward}} + n_{\text{add}}$, and the model has $O(d_h^2 + d_h \cdot d_{\text{in}})$ parameters shared across all participants.

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

**Parameters (per participant, 5 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\beta_{\text{reward}}$ | $> 0$ (softplus, clamp [0.001, 20]) | Sensitivity to point values |
| $\beta_{\text{cost}}$ | $\geq 0$ (softplus, clamp [0, 20]) | Base effort cost of switching |
| $\beta_{\text{fatigue}}$ | $\geq 0$ (softplus, clamp [0, 20]) | Additional effort cost per unit of normalized time |
| $\text{bias}_a$ | unconstrained (clamp [-10, 10]) | Per-action intercept ($A = 2$ values per participant) |

**Decision rule.** For action $a \in \{\text{repeat}=0, \text{switch}=1\}$:

$$\text{EVC}(a) = \beta_{\text{reward}} \cdot V_a - (\beta_{\text{cost}} + \beta_{\text{fatigue}} \cdot t_{\text{norm}}) \cdot \mathbb{1}[a = \text{switch}] + \text{bias}_a$$

where $V_a$ is the observed point value for action $a$, $t_{\text{norm}} = b / B$ is the normalized block position (block index divided by total blocks $B = 12$), and $\mathbb{1}[a = \text{switch}]$ is the switch indicator.

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
- **No working-memory state.** The incremental reward value (`value_reward_chosen`, `value_reward_not_chosen`) and choice perseveration (`value_choice`), plus a leave threshold (`leave_threshold`).
- **Only the harvest item carries values.** The exit item stays at 0 as the reference point, which mirrors the structure of the Marginal Value Theorem.
- **Harvest/exit replace chosen/unchosen.** The `*_chosen` module updates the harvest value after a harvest (input: the harvest reward), and the `*_not_chosen` module updates it after an exit (a new patch).
- **Choice value** is restricted to the harvest item and receives the harvest indicator $\mathbb{1}[\text{harvested}]$. It acts as a continuation tendency.
- **Leave threshold.** A stateless module (reset every trial, harvest item only) on the round's effort condition, `leave_threshold` $= f(\texttt{is\_msit}, \texttt{is\_high})$. With two binary inputs, its library reduces to
  $$\theta = b_0 + b_1\,\texttt{is\_msit} + b_2\,\texttt{is\_high} + b_3\,\texttt{is\_msit}\cdot\texttt{is\_high},$$
  the four per-condition thresholds of the benchmark's factorial design (Bustamante et al., 2023, Eqs. 2–5): $b_0$ is the threshold for low-effort key pressing, $b_1$ the shift for the cognitive travel task, $b_2$ the extra cost of high-effort travel, and $b_3$ how that extra cost differs between cognitive and physical effort. Through the participant embedding, every participant has their own effort costs; terms without support are pruned like any other.

$$\text{logits}_t = \big[V^{\text{reward}}_t + V^{\text{choice}}_t - \theta_t, \; 0\big]$$

Because the harvest reward depletes within a patch, the reward value tracks the expected next harvest reward, and the leave threshold sets the round's exit point, as the MVT's comparison of the expected reward with a per-condition threshold. The constant inputs `harvest_duration` and `travel_duration` are loaded but not used.

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

### 3.4. Dezfouli 2019: Two-Armed Bandit (Clinical)

**Task and data.** Dezfouli et al. (2019): 101 participants (healthy controls and patients with depression or bipolar disorder) performed 12 blocks of a two-armed bandit with binary rewards. In each block one response paid with probability 0.25, 0.125 or 0.08 (four blocks each per participant) and the other with 0.05; which response was better and the order of the conditions varied across participants and blocks.

**SPICE model and benchmark.** The full shared RL architecture without deviations (Section 1.4); benchmark: Generalized Q-Learning (Dezfouli et al., 2019).

**Reward probabilities used for simulations (to be reported in the SI).** The published data do not record which reward probabilities a block had, and the block number is only the presentation order (each participant has every condition twice, in varying positions). The authors' raw data (github.com/adezfouli/rnn_beh, `data/BD/choices_diagno.csv.zip`, Apache License 2.0; the unpacked csv is kept unchanged as `studies/dezfouli2019/data_raw/choices_diagno.csv`) contain the condition of every block. Conditions 1–2, 3–4 and 5–6 give the better response a reward probability of 0.25, 0.125 and 0.08; the other response pays 0.05 (empirical reward rates 0.247 / 0.122 / 0.082 and 0.055). The better response is the one marked as best in the data. The published, anonymized data were matched to the raw data by each participant's complete trial sequence (block, key, reward, better response), which identifies all 101 participants uniquely, and the condition was transferred to every block. These probabilities are used only where the task is simulated (generative comparison, rollout evaluation); fitting and one-step evaluation use the observed rewards. Note for the SI: the authors' own simulation code assumed a fixed probability pair per block number, which does not hold in their data. Before the raw data were found, the probabilities had been reconstructed from the published data alone (better response from the best-response indicator, level by an exact binomial posterior over all level assignments consistent with four blocks per level); this reconstruction matched the raw-data conditions in 99.3% of the 1,212 blocks.

## 4. Summary Table

| Study | Task | $A$ | SPICE model | Modules | States | SINDy Degree | Benchmark Model | Benchmark Params/Participant |
|-------|------|-----|-------------|---------|--------|--------------|----------------|------------------------------|
| Synthetic | Q-learning recovery | 2 | `spice.precoded.choice` | 3 | 2 | 2 | — (ground truth) | — |
| Braun 2018 | Voluntary task switching | 2 | `spice_braun2018` | 6 | 3 | 2 | Expected Value of Control | 5 |
| Bustamante 2023 | Patch foraging | 2 | `spice_bustamante2023` | 4 | 3 | 2 | MVT with effort costs (Bustamante 2023) | 5 |
| Eckstein 2026 | Drifting 4-armed bandit | 4 | `spice_eckstein2026` | 6 | 4 | 2 | Best RL (Eckstein 2026) | 6 |
| Dezfouli 2019 | Two-armed bandit (clinical) | 2 | `spice_dezfouli2019` | 5 | 3 | 2 | Generalized Q-Learning | 12 |
| Ganesh 2024a | Perceptual contrast bandit | 2 | `spice_ganesh2024a` | 6 | 3 | 2 | Bayesian Belief-Update | 2 |
| Kolff 2025 | Chimpanzee grooming negotiation (group model) | 2 | `spice_kolff2025` | 4 | 4 | 1 | Lag-1 logistic regression | 13 (shared) |
| Bruckner 2025 | Helicopter task (predictive inference) | 1 (continuous) | `spice_bruckner2025` | 7 | 4 | 2 | Reduced Bayesian Model | 4 |
| Weber 2024 | Laser tracking (shield movement) | 2 (continuous, sin/cos) | `spice_weber2024` | 4 | 2 | 2 | Bayesian Changepoint Model | 2 |

The GRU baseline (Section 2) is fitted for every study in addition to the study-specific benchmark.

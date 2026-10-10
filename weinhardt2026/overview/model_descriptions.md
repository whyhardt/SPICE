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

Two-stage training as described in the main methods and [training.md](training.md):
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

**Default hyperparameters of the paper runs** (`weinhardt2026/run.py` defaults, overridden by the stability script where noted):

| Hyperparameter | Value |
|---|---|
| Embedding size $d_e$ | 8 |
| Ensemble size | 10 |
| Dropout | 0.1 |
| SINDy polynomial degree | 2 (1 for kolff2025) |
| `sindy_weight` | 0.01 |
| `sindy_alpha` | 0.0001 |
| `gate_penalty` | 0 |
| `feature_penalty` | 0.0001 (stability script; run.py default 0) |
| Pruning threshold | 0.01 |
| Behavioral loss | Cross-entropy with label smoothing 0.01 (study-specific losses noted below) |
| Seeds | 10 per study (stability runs) |

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
| Bustamante 2023 | ✓ | — | ✓ | Only the harvest item carries values; harvest/exit take the roles of chosen/unchosen; exit item is a fixed 0 reference |

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
- $A = 2$ (harvest=0, exit=1); rewards are given only for harvests (partial feedback) and are normalized by the maximum reward 20.
- 250 participants (Prolific), 8 rounds each (`overall_round`; 40–112 decisions per round, mean 82), 340–864 decisions per participant.
- Additional inputs `harvest_duration` (2 s) and `travel_duration` (8.33 s) are constant.
- Held-out test blocks (rounds): 3, 6.

#### SPICE Model (`studies/bustamante2023/spice_bustamante2023.py`)

The shared RL architecture (Section 1.4) with these deviations:
- **No reward-value state.** Only working memory (`value_wm_reward_chosen`, `value_wm_reward_not_chosen`) and choice perseveration (`value_choice`).
- **Only the harvest item carries values.** The exit item stays at 0 as the reference point, which mirrors the structure of the Marginal Value Theorem.
- **Harvest/exit replace chosen/unchosen.** The `*_chosen` module updates the harvest value after a harvest (input: the harvest reward), and the `*_not_chosen` module updates it after an exit (a new patch).
- **Choice value** is restricted to the harvest item and receives the harvest indicator $\mathbb{1}[\text{harvested}]$. It acts as a continuation tendency.

$$\text{logits}_t = \big[V^{\text{wm}}_t + V^{\text{choice}}_t, \; 0\big]$$

Because the harvest reward depletes within a patch, the working-memory value tracks the expected next harvest reward. The comparison against the fixed exit reference corresponds to the MVT's comparison against the environment's average reward rate, which is absorbed into the learned intercepts. The additional inputs `harvest_duration` and `travel_duration` are loaded but not used.

#### Benchmark Model: Marginal Value Theorem (MVT)

**Reference:** Constantino & Daw (2015); applied to Bustamante et al. (2023).

The Marginal Value Theorem predicts that a forager should leave a patch when the instantaneous gain rate drops to the average gain rate in the environment. The implementation follows the learning model from Constantino & Daw (2015, Table 2). The action space is $A = 2$ (harvest=0, exit=1).

**Parameters (per participant, 3-5 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\alpha_{\text{env}}$ | $(0.01, 0.99)$ (sigmoid) | Learning rate for environmental gain rate |
| $\beta$ | $(0.1, 10)$ (softplus) | Inverse temperature for decision softmax |
| $c$ | $(-10, 10)$ | Intercept/bias for stay vs. leave decision |
| $\kappa$ | $(0.001, 1)$ (softplus, optional) | Within-patch depletion rate |
| $g_0$ | $(0.1, 20)$ (softplus, optional) | Baseline gain expectation for new patches |

**State variables (per session):**
- `cumulative_reward`: total reward accumulated in current patch
- `n_harvests`: number of harvest trials in current patch
- `time_in_patch`: total time spent in current patch
- `env_reward_rate` ($\rho$): estimated average gain rate in the environment
- `current_tree_state` ($s_i$): expected reward from next harvest in current patch

**Decision rule.** The probability of harvesting (staying) is:

$$P(\text{harvest}) = \frac{1}{1 + \exp[-c - \beta(\kappa \cdot s_i - \rho \cdot \tau_h)]}$$

where $s_i$ is the current tree state (expected next reward), $\rho$ is the estimated environmental reward rate, and $\tau_h$ is the harvest duration.

**Environment reward rate learning.** After each action with reward $r_i$ taking time $\tau_i$:

$$\delta_i = \frac{r_i}{\tau_i} - \rho_i$$
$$\rho_{i+1} = \rho_i + [1 - (1 - \alpha_{\text{env}})^{\tau_i}] \cdot \delta_i$$

The effective learning rate $1 - (1 - \alpha)^{\tau}$ increases with action duration, accounting for the fact that longer experiences carry more information.

**Patch state transitions:**
- After harvest: $s_{i+1} = r_i$ (update to observed reward), cumulative statistics increment
- After exit: all patch statistics reset to 0, $s_0 = g_0$ (new patch expectation)


### 3.4. Eckstein 2026: Drifting Multi-Armed Bandit

**Task.** A large-scale restless four-armed bandit (Eckstein et al., 2026). On each trial participants choose one of 4 options, arranged in a circle on the screen, and see the points (0–100) earned from the chosen option only. The options' mean payouts drift independently as mean-reverting Gaussian random walks:
$$\mu_{t,i} \sim \mathcal{N}\big(\lambda \mu_{t-1,i} + (1-\lambda) \cdot 50, \; \sigma_{\text{drift}}\big), \qquad \lambda = 0.9836, \; \sigma_{\text{drift}} = 2.8,$$
and the observed payout is noisy: $r_{t,i} \sim \mathcal{N}(\mu_{t,i}, \sigma_{\text{obs}})$ with $\sigma_{\text{obs}} = 4$. Because the best option changes over time, participants have to trade off exploiting the currently best option against exploring options whose values may have drifted upward. The circular layout makes spatial structure in switching (to the adjacent or the opposite option) a candidate behavioral mechanism.

**Data.**
- $A = 4$; payouts normalized to $[0, 1]$ (steps of 0.01); the payouts of all four options (`payout_1..4`) are recorded but only the chosen one is given to the models.
- 862 participants; blocks of 150 trials; 786 participants completed 5 blocks (750 trials) and 76 completed 3 blocks (450 trials).
- Held-out test block: 2.

#### SPICE Model (`studies/eckstein2026/spice_eckstein2026.py`)

The full shared RL architecture (Section 1.4) over 4 arms, with one addition:

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `bias_attention` | `is_adjacent`, `is_opposite` | `bias_attention` (initial: 0) | all items | Spatial attention bias by circular distance to the chosen arm |

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


### 3.5. Dezfouli 2019: Two-Armed Bandit with Depression

**Task.** A two-armed bandit with a sparse reward schedule (Dezfouli et al., 2019), used to compare learning and choice mechanisms between healthy participants and patients with depression or bipolar disorder. In each block, participants repeatedly choose between two keys. Each key pays a binary reward with a fixed Bernoulli probability for the whole block, and one key is always better than the other. The 12 blocks cover four contrasts, each presented with both key assignments: 0.25 vs. 0.05, 0.125 vs. 0.05, 0.08 vs. 0.05 and 0.05 vs. 0.08 (block-by-block mapping in `EnvironmentDezfouli2019`). Rewards are rare and the contrast between arms is small in most blocks. Choices therefore depend strongly on reward history, choice history (perseveration) and their interaction, which is what the original study identified as differing between groups.

**Data.**
- $A = 2$; binary rewards for the chosen arm only.
- 101 participants: 34 healthy controls, 34 with depression, 33 with bipolar disorder (`diag`).
- 12 blocks per participant with a variable number of trials (6–202 per block, median 108); 398–2021 trials per participant.
- Held-out test blocks: 3, 6, 9.

#### SPICE Model (`studies/dezfouli2019/spice_dezfouli2019.py`)

The full shared RL architecture (Section 1.4) without deviations: 5 submodules on 3 memory states, with binary rewards.

#### Benchmark Model: Generalized Q-Learning (GQL)

**Reference:** Dezfouli et al. (2019).

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

#### SPICE Model (`studies/ganesh2024a/spice_ganesh2024a.py`)

The full shared RL architecture (Section 1.4) with these deviations:
- **Item space.** Values live in a contrast-based item space (low-contrast=0, high-contrast=1) instead of the left/right action space (see below).
- **Perception module.** A stateless module maps the contrast difference magnitude to a perceptual certainty:

| Submodule | Control signals | State updated | `include_state` | Description |
|-----------|----------------|---------------|-----------------|-------------|
| `perception_certainty` | $\lvert\Delta c\rvert$ | — (output passed through a sigmoid) | False | Perceptual certainty from contrast difference magnitude |

- **Certainty as an extra input.** All reward and working-memory modules additionally receive the current certainty $\text{cert}_t$. The choice module receives the item-space choice indicator $c^{\text{item}}_t$ together with $\text{cert}_t$ and the next trial's certainty $\text{cert}_{t+1}$.

**Key design feature: item-space/action-space decoupling.** The model represents values in item space (low-contrast, high-contrast) rather than action space (left, right), because the stimulus-to-position mapping changes every trial:
1. Actions are remapped from position space to item space using the sign of the contrast difference: when $\Delta c \leq 0$, left=low and right=high; when $\Delta c > 0$, left=high and right=low.
2. Learning updates operate in item space with these deterministic masks; certainty enters as an input, so updates can be attenuated when the assignment is unreliable.
3. At decision time, item-space logits are mapped back to action space using the perceptual certainty for the next trial's contrast.

**Perceptual certainty.** $\text{cert}_t = \sigma(f(\lvert\Delta c_t\rvert))$ and $\text{cert}_{t+1} = \sigma(f(\lvert\Delta c_{t+1}\rvert))$, the same module evaluated on the current and on the next trial's contrast difference.

**Data preparation.** `prepare_dataset` appends the next trial's contrast difference $\Delta c_{t+1}$ as an additional input and drops the last trial of each session.

**Logit computation (soft item-to-action mapping):**
$$V^{\text{item}}_t = V^{\text{wm}}_t + V^{\text{reward}}_t + V^{\text{choice}}_t$$
$$\text{cert}'_{t+1} = \text{cert}_{t+1} / 2 + 0.5 \quad \text{(rescaled to [0.5, 1.0])}$$
$$V^{\text{mixed}}_t = \text{cert}'_{t+1} \cdot V^{\text{item}}_t + (1 - \text{cert}'_{t+1}) \cdot \text{flip}(V^{\text{item}}_t)$$
$$\text{logits}_t = \begin{cases} V^{\text{mixed}}_t & \text{if } \Delta c_{t+1} < 0 \\ \text{flip}(V^{\text{mixed}}_t) & \text{if } \Delta c_{t+1} \geq 0 \end{cases}$$

When perceptual certainty is high, item values map cleanly to action values; when certainty is low, the mapping is more uniform.

**SINDy library degree:** 2.

#### Benchmark Model: Bayesian Belief-Update Model

**Reference:** Ganesh et al. (2024), normative agent model.

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

**Task.** An observational dataset, not an experiment: video-coded grooming interactions between pairs of wild chimpanzees from two communities (CE and WE). Grooming is preceded by a *negotiation* phase, in which the two apes exchange behaviors that may lead to grooming and decide who grooms whom. Each interaction is coded as a sequence of events; in each event, either ape (or occasionally both) performs a behavioral element. The raw elements (6334 events, 311 interactions, 41 apes) are recoded into one *Groom* category and five *Negotiation* categories (agreed with the primatologists; see [kolff2025_preprocessing.md](kolff2025_preprocessing.md)):
- **Self_Reposition**: the ape changes its own posture without presenting a body part.
- **Reposition_Body**: the ape moves the partner into a position for grooming (`touch`, `grab-pull limb`, `push`, `touch hold`).
- **Grooming_Process**: `maintain contact`, i.e. a hand kept on the partner between grooming bouts.
- **Grooming_Solicitation**: visual or tactile requests to be groomed (`present_*`, `raise *`, `extend *`, `kiss`, `hold`).
- **Directed_Scratch**: a scratch directed at a body part, as a request.

Every interaction is modeled from the perspective of each of its two apes in turn (the *focal* ape and its *partner*), so each ape is a participant and each (interaction, perspective) pair is a block. Each ape's dominance rank is normalized within its community. `rank_diff` is the rank difference to the partner, and `rank_diff_centered` is that difference minus the ape's mean over its partners, which makes it orthogonal to the ape's own rank. The question is how the focal ape's next behavior depends on what both apes just did and on their dominance relation.

#### SPICE Model: TOBETO Drives (`studies/kolff2025/spice_kolff2025_tobeto.py`)

The model predicts the focal ape's own next act, grouped by the grooming role it signals: "to be groomed" (TOBE) or "to groom" (TO). Only bouts with at least 2 events are kept.

**Data** (`data/kolff2025_tobeto.csv`): 41 apes, 299 interactions, 11786 perspective-events (2–716 per ape). Held-out test set: the 64 interactions of 31 held-out dyads (20% of dyads, seed 42).

**Readout** ($A = 4$, one softmax): `none` (the focal ape does not act next; reference class), `tobegroomed` (Grooming_Solicitation, Self_Reposition, Directed_Scratch), `togroom` (Reposition_Body, Grooming_Process), `groom`.

| Submodule | Control signals | State item written | Description |
|-----------|----------------|--------------------|-------------|
| `drive_tobegroomed` | 7 signals (below) | `drive[tobegroomed]` | Drive to be groomed |
| `drive_togroom_signal` | 7 signals | `drive[togroom]` | Drive to groom, expressed as to-groom signals |
| `drive_togroom_groom` | 7 signals | `drive[groom]` | Drive to groom, expressed as grooming itself |

Control signals: one-hot indicators of the focal ape's and the partner's last act (`own_tobegroomed`, `own_togroom`, `own_groom`, `partner_tobegroomed`, `partner_togroom`, `partner_groom`; "no act" is the all-zero reference) and `rank_diff` (= `rank_diff_centered`, constant within an interaction, so it acts as a dyad-specific shift of the resting drive). The drive to groom is split into two modules because grooming persists over consecutive events while signals do not; a shared equation would fix their ratio.

**Memory states:** `drive` (4 items, learnable per-ape initial value). The `none` item is never written and serves as the softmax reference.

**Logit computation:** $\text{logits}_t = \text{drive}_t$.

**SINDy library degree:** 1 (linear: per module $\{1, \text{drive}, 6 \text{ act indicators}, \text{rank\_diff}\}$).

**Evaluation** is reported on all events and, with `filter_own_acts`, restricted to events where the focal ape itself acts next.

#### Benchmark Model: Lag-1 Conditional Frequency (`ConditionalTobetoModel`)

A memoryless lag-1 lookup table per ape, $P(\text{own act}_{t+1} \mid \text{own act}_t, \text{partner act}_t)$, estimated by counting with add-one smoothing. Because exactly one ape acts per event, 2 × 3 (own, partner) cells carry data, each with 3 free probabilities: 18 parameters per ape. Whatever SPICE gains over this table is attributable to latent dynamics.

### 3.8. Bruckner 2025: Helicopter Task (Predictive Inference)

**Task.** A predictive-inference ("helicopter") task from a lifespan study of learning (Bruckner, Nassar, Li & Eppinger, 2025, follow-up experiment). A hidden helicopter drops bags at positions $x_t \sim \mathcal{N}(\mu_t, \sigma^2)$ on a horizontal screen of 300 pixels ($\sigma = 17.5$ pixels in the data's `sigma` column). On every trial, participants place a bucket at $b_t$ to catch the next bag. The bag counts as caught when $|b_t - x_t| \le \sigma/2$ (27% of trials). The helicopter position $\mu_t$ stays fixed for a while and then jumps to a new location at change points ($c_t$, about 10% of trials). Participants therefore have to update strongly after surprising outcomes that signal a change point, and weakly after outcomes that are just noise. On about 10% of trials the helicopter is visible ($v_t = 1$) and reveals $\mu_t$ directly. Each bag contains a coin of high (gold, $r_t = 1$) or low (stone, $r_t = 0.25$) value, which tests whether reward value biases the learning rate.

The experiment has two conditions. In *noPush* the bucket stays where the participant left it. In *push* the bucket is displaced to a random starting position $z_{t+1}$ before the next prediction, and participants have to move it back. This measures anchoring: an incomplete correction toward the displaced position, which the original study links to resource-rational (effort-saving) computations.

**Data.**
- Continuous scalar output ($A = 1$): the next bucket position $b_{t+1}$, trained with MSE; all positions normalized to $[0, 1]$ by dividing by 300.
- 90 participants in three age groups (31 children, 25 younger adults, 34 older adults), each completing both conditions: 2 blocks of 100 trials in noPush (blocks 1–2) and 2 in push (blocks 3–4), i.e. 400 trials per participant. Condition is coded as `experiment`.
- Held-out test blocks: 1, 2.

#### SPICE Model (`studies/bruckner2025/spice_bruckner2025.py`)

7 submodules on 4 memory states. A dual learning-rate architecture inspired by the Reduced Bayesian Model: the prediction-error magnitude decides whether a changepoint-driven or an uncertainty-driven learning rate governs the update. Belief updates are split by whether the bag was caught, and an anchoring module captures the bias from bucket displacement.

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|-----------|----------------|---------------|-------------|-----------------|-------------|
| `changepoint_update` | $v_t$ | `changepoint_value` | $m^{\text{big}}_t$ | True | Changepoint learning-rate update on large PE |
| `changepoint_decay` | `catch`, $v_t$ | `changepoint_value` | $1 - m^{\text{big}}_t$ | True | Changepoint learning-rate decay on small PE |
| `uncertainty_update` | `catch`, $v_t$ | `uncertainty_value` | $1 - m^{\text{big}}_t$ | True | Uncertainty learning-rate update on small PE |
| `uncertainty_decay` | $v_t$ | `uncertainty_value` | $m^{\text{big}}_t$ | True | Uncertainty learning-rate decay on large PE |
| `belief_update_catch` | $\delta_t$ | `belief_value` | `catch` $\cdot (1 - v_t)$ | True | Belief update after a caught bag |
| `belief_update_miss` | $\delta_t$ | `belief_value` | $(1-$`catch`$) (1 - v_t)$ | True | Belief update after a missed bag |
| `anchor_update` | $y_t$ | `anchor_value` | — | False | Anchoring correction for bucket displacement |

**Memory states:**
- `belief_value`: internal estimate of the helicopter position. It is set to the participant's bucket position $b_t$ at the start of every trial (subjective prediction errors, as in the benchmark)
- `changepoint_value` (initial: 0): $\omega_t = \sigma(\text{changepoint\_value})$ is the changepoint learning rate
- `uncertainty_value` (initial: 0): $\tau_t = \sigma(\text{uncertainty\_value})$ is the uncertainty-driven learning rate
- `anchor_value`: per-trial anchoring correction, reset to 0 each trial

**Control signal preprocessing** (`prepare_dataframe`):
- Prediction error: $\delta_t = x_t - b_t$.
- PE magnitude mask: $m^{\text{big}}_t = \mathbb{1}[|\delta_t| > 3 \cdot \text{sigma}_t / 2]$, three bucket half-widths (`sigma` is an observable task variable, not a latent parameter). This externalized binary gate routes trials to the changepoint or the uncertainty pathway, mirroring the change-point detection of the Bayesian benchmark.
- Anchor shift: $y_t = z_{t+1} - b_t$, the displacement between the next trial's initial bucket position and the current bucket position.
- `catch` $\in \{0, 1\}$: whether the bag was caught; $v_t \in \{0, 1\}$: whether the helicopter is visible. Both are binary, so their squared terms are pruned.
- The true helicopter position $\mu_t$ is masked to NaN when $v_t = 0$; participants can only observe it when visible.

**Visible trials.** When $v_t = 1$, the belief modules are masked out and the belief is set to the true helicopter position: $\hat{\mu}_t = \mu_t$.

**Output (gated + anchoring):**
$$\alpha_t = m^{\text{big}}_t \cdot \omega_t + (1 - m^{\text{big}}_t) \cdot \tau_t$$
$$\hat{b}_{t+1} = (1 - \alpha_t) \cdot b_t + \alpha_t \cdot \hat{\mu}_t + V^{\text{anchor}}_t$$

The predicted next bucket position interpolates between the current bucket position and the internal belief, plus the anchoring correction.

**SINDy library degree:** 2.

#### Benchmark Model: Reduced Bayesian Model (RBM)

**Reference:** Bruckner et al. (2025), Eqs. 4-14. Implemented as `RationalResourceModel` in `benchmarking_bruckner2025.py` (a reimplementation of `AlAgentRbm.py` from the original codebase).

A reduced Bayesian model that maintains a belief about the helicopter's position and an uncertainty estimate that modulates learning rate. The belief is reset to the participant's actual prediction $b_t$ at each trial (subjective prediction errors), while uncertainty dynamics carry forward across trials. On catch trials (helicopter visible), the belief is additionally updated with the observed helicopter position. The output is a continuous scalar prediction ($A = 1$).

**Parameters (per participant, 4 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $h$ | $(0, 1)$ sigmoid | Hazard rate: prior probability of a change point |
| $s$ | $(0, 1)$ sigmoid | Surprise sensitivity: modulates change-point detection |
| $u$ | unconstrained (exponentiated, clamped at $u \le 10$) | Uncertainty underestimation: $\exp(u)$ divides posterior uncertainty |
| $q$ | unconstrained | Reward bias: learning rate increase for high-value trials |

**Fixed task parameters:**
- $\sigma = 15$ pixels: outcome noise standard deviation
- $\sigma_H = 15$ pixels: helicopter cue noise standard deviation (used on catch trials)

(Note: `SIGMA = 15 / 300` is hard-coded in `benchmarking_bruckner2025.py`, while the data's `sigma` column, used by the SPICE model's PE mask, is 17.5 pixels. To be checked.)

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

6. **Predicted next position** (Eq. 4):
$$\hat{b}_{t+1} = \text{clamp}(\mu_{t+1}, 0, 1)$$

7. **Uncertainty update** (Eq. 9):
$$\hat{\sigma}^2_{t+1} = \frac{\omega_t \cdot \sigma^2 + (1 - \omega_t) \cdot \tau_t \cdot \sigma^2 + \omega_t(1 - \omega_t)(\delta_t(1 - \tau_t))^2}{\exp(u)}$$

8. **Relative uncertainty update** (Eq. 10):
$$\tau_{t+1} = \frac{\hat{\sigma}^2_{t+1}}{\hat{\sigma}^2_{t+1} + \sigma^2}$$

The model dynamically adjusts its learning rate via change-point detection: when a change point is detected (high $\omega_t$), the learning rate increases to rapidly incorporate the new outcome; during stable periods (low $\omega_t$), the learning rate decreases for more precise estimation. The uncertainty underestimation parameter $u$ captures individual differences in confidence calibration.


### 3.9. Weber 2024: Laser Tracking (Shield Movement)

**Task.** A continuous circular tracking task (Weber et al., 2024), a circular analog of the helicopter task (Bruckner 2025). Participants control a shield on a circular track (0°–360°) around a central source that fires laser beams. Each beam's position is drawn around a hidden mean, and the mean jumps by ±20°, ±30° or ±40° at change points. The shield is moved continuously with button presses at a fixed speed of 1°/frame. Between two beams there are 1–90 frames (median 17), so the shield cannot always reach the believed laser position in time. A beam is caught if it lands within ±10° of the shield's center, which happens on 40% of beams.

Two factors are crossed within participants:
- **Volatility** (change-point rate): about 3.3% of beams in low-volatility blocks vs. about 9% in high-volatility blocks.
- **Stochasticity** (observation noise): the s.d. of beams around the mean is about 10° in low-stochasticity blocks vs. about 20° in high-stochasticity blocks.

The participant has to infer whether a deviating beam is noise or a change point, and adjust their learning rate under both kinds of uncertainty.

The model works with events rather than frames. Each event is one laser beam, and the model predicts the shield position at the next beam. This is a continuous 2D prediction in $(\sin, \cos)$ space, trained with a custom clamped angular MSE loss instead of cross-entropy.

**Data.**
- 30 participants; 4 conditions (`experiment` 0–3 = volatility × stochasticity: low/low, low/high, high/low, high/high), 4 blocks per condition, i.e. 16 blocks of 516–565 events (8590 events per participant).
- Additional inputs: `laser_caught`, `volatility`, `stochasticity`, `trial_duration_frames` (inter-beam interval), `trueMean` (diagnostics only).
- Held-out test blocks: 2, 5, 9, 14.

**Data representation.** All angular positions are encoded as $(\sin\theta, \cos\theta)$ pairs to handle the circular geometry (`prepare_dataframe`). The data is event-based: each trial corresponds to one laser beam event, regardless of the inter-beam time interval. Actions (shield position) and rewards (laser position) both have 2 components ($A = 2$, items $I = 2$).

#### SPICE Model (`studies/weber2024/spice_weber2024.py`)

4 submodules on 2 memory states. Belief and certainty updates are split by catch outcome through externalized binary gating via the `action_mask` mechanism. A hardcoded gated output translates the internal belief into a shield position prediction.

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `certainty_update_caught` | — | `certainty_raw` | `laser_caught` | Certainty dynamics when the shield catches the laser |
| `certainty_update_missed` | — | `certainty_raw` | $1 -$ `laser_caught` | Certainty dynamics when the shield misses the laser |
| `belief_update_caught` | $\delta_t$ | `belief_value` | `laser_caught` | Belief update after a catch |
| `belief_update_missed` | $\delta_t$ | `belief_value` | $1 -$ `laser_caught` | Belief update after a miss |

**Memory states:**
- `belief_value`: internal belief about the laser mean, as $(\sin, \cos)$ over the 2 items; initialized to the first observed laser position
- `certainty_raw` (initial: 0): $\sigma(\text{certainty\_raw}) = \alpha_t \in [0, 1]$ is the weight on the belief

**Control signal preprocessing:**
- Prediction error: $\delta_t = (\sin\theta^{\text{laser}}_t, \cos\theta^{\text{laser}}_t) - \hat{\mu}_t$, component-wise in $(\sin, \cos)$ space, computed before the belief update.
- Catch mask: `laser_caught` $\in \{0, 1\}$. It is not an RNN input; it routes each trial to the catch or miss modules.

The certainty modules have no control signals: their dynamics depend only on their own state, separately for catch and miss trials.

**Output (gated):**
$$\alpha_t = \sigma(\text{certainty\_raw}_t)$$
$$\hat{s}_{t+1} = (1 - \alpha_t) \cdot s_t + \alpha_t \cdot \hat{\mu}_{t}$$

where $s_t = (\sin\theta^{\text{shield}}_t, \cos\theta^{\text{shield}}_t)$ is the current shield position and $\hat{\mu}_{t}$ is the belief after this trial's update.

**Custom loss function: Clamped Angular MSE.** Because the shield has a finite movement speed, the raw prediction may be physically unreachable within the inter-beam interval. The loss clamps the predicted movement:

$$\Delta = \hat{s}_{t+1} - s_t, \quad f = \min\left(\frac{v \cdot \Delta t \cdot (\pi / 180)}{||\Delta|| + \epsilon}, 1\right)$$
$$\hat{s}^{\text{clamped}}_{t+1} = s_t + f \cdot \Delta$$
$$\mathcal{L} = \text{MSE}(\hat{s}^{\text{clamped}}_{t+1}, s^{\text{actual}}_{t+1})$$

where $v = 1$ °/frame and $\Delta t$ is the inter-beam interval in frames. `prepare_dataset` packs the targets as $[s^{\text{actual}}_{t+1}, s_t, \Delta t]$ so the loss has the current position and timing information.

**SINDy library degree:** 2.

**Additional inputs:** `laser_caught`, `volatility`, `stochasticity`, `trial_duration_frames`, `trueMean` (ground-truth laser mean, for diagnostic plotting only).

#### Benchmark Model: Bayesian Changepoint Model

**Reference:** Weber et al. (2024), "change-point model without variance inference". Implemented as `ChangePointModel` in `benchmarking_weber2024.py`.

Bayesian inference on a discretized circular state space ($[0°, 360°)$ in 2° steps) under changepoint dynamics with fixed observation noise.

**Parameters (per participant, 2 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $p_{\text{cp}}$ | $(0.001, 0.5)$ sigmoid | Changepoint probability per trial |
| $\sigma_{\text{obs}}$ | $[1°, 60°]$ softplus | Observation noise s.d. of the circular normal likelihood |

**Fixed:** allowed changepoint magnitudes $\pm 20°, \pm 30°, \pm 40°$; uniform prior at block start.

**Trial-by-trial update:**
1. Posterior over the laser mean via Bayes' rule with a circular normal likelihood of the observed laser position.
2. Belief = circular mean of the posterior, mapped to $(\sin, \cos)$.
3. Prediction = belief, clamped to the movement reachable from the current shield position (same clamping as the SPICE loss).
4. Propagation: $\text{prior}_{t+1} = (1 - p_{\text{cp}}) \cdot \text{posterior} + p_{\text{cp}} \cdot M_{\text{cp}} \, \text{posterior}$, where $M_{\text{cp}}$ redistributes mass uniformly to states at the allowed change distances.

---

## 4. Summary Table

| Study | Task | $A$ | SPICE model | Modules | States | SINDy Degree | Benchmark Model | Benchmark Params/Participant |
|-------|------|-----|-------------|---------|--------|--------------|----------------|------------------------------|
| Synthetic | Q-learning recovery | 2 | `spice.precoded.choice` | 3 | 2 | 2 | — (ground truth) | — |
| Braun 2018 | Voluntary task switching | 2 | `spice_braun2018` | 6 | 3 | 2 | Expected Value of Control | 5 |
| Bustamante 2023 | Patch foraging | 2 | `spice_bustamante2023` | 3 | 2 | 2 | Marginal Value Theorem | 3-5 |
| Eckstein 2026 | Drifting 4-armed bandit | 4 | `spice_eckstein2026` | 6 | 4 | 2 | Best RL (Eckstein 2026) | 6 |
| Dezfouli 2019 | Two-armed bandit (clinical) | 2 | `spice_dezfouli2019` | 5 | 3 | 2 | Generalized Q-Learning | 12 |
| Ganesh 2024a | Perceptual contrast bandit | 2 | `spice_ganesh2024a` | 6 | 3 | 2 | Bayesian Belief-Update | 2 |
| Kolff 2025 | Chimpanzee grooming negotiation (TOBETO) | 4 | `spice_kolff2025_tobeto` | 3 | 1 | 1 | Lag-1 conditional frequency | 18 |
| Bruckner 2025 | Helicopter task (predictive inference) | 1 (continuous) | `spice_bruckner2025` | 7 | 4 | 2 | Reduced Bayesian Model | 4 |
| Weber 2024 | Laser tracking (shield movement) | 2 (continuous, sin/cos) | `spice_weber2024` | 4 | 2 | 2 | Bayesian Changepoint Model | 2 |

The GRU baseline (Section 2) is fitted for every study in addition to the study-specific benchmark.

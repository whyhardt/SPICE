# Model Descriptions for SPICE Paper

This document provides full architectural descriptions of all SPICE models, benchmark models, and the GRU baseline used across the benchmark studies. It is intended as input for writing the methods/supplementary sections of the SPICE paper.

---

## 1. GRU Baseline Architecture

The GRU (Gated Recurrent Unit) baseline serves as a black-box recurrent neural network benchmark across all studies. It uses a standard PyTorch GRU cell with the following architecture:

**Input processing.** At each trial $t$, the model receives a concatenation of the one-hot encoded previous action $a_{t} \in \{0,1\}^A$, the reward vector $r_{t} \in \mathbb{R}^{n_{\text{reward}}}$, and any additional task-specific inputs $x_t^{\text{add}} \in \mathbb{R}^{n_{\text{add}}}$. The concatenated input $[a_t; r_t; x_t^{\text{add}}]$ is projected through a linear layer to a hidden representation of size $d_h$:

$$z_t = W_{\text{in}} [a_t; r_t; x_t^{\text{add}}] + b_{\text{in}}$$

**Recurrent computation.** The projected input $z_t$ is passed through a standard GRU cell with hidden state $h_t \in \mathbb{R}^{d_h}$:

$$\hat{r}_t = \sigma(W_{ir} z_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})$$
$$\hat{z}_t = \sigma(W_{iz} z_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})$$
$$\hat{n}_t = \tanh(W_{in} z_t + b_{in} + \hat{r}_t \odot (W_{hn} h_{t-1} + b_{hn}))$$
$$h_t = (1 - \hat{z}_t) \odot \hat{n}_t + \hat{z}_t \odot h_{t-1}$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication. The GRU has three gates: a reset gate $\hat{r}_t$ that controls how much past information to forget, an update gate $\hat{z}_t$ that controls how much of the candidate state to incorporate, and a candidate new state $\hat{n}_t$.

**Output.** The hidden state is projected through a linear output layer to produce action logits:

$$\text{logits}_t = W_{\text{out}} h_t + b_{\text{out}} \in \mathbb{R}^A$$

Action probabilities are obtained via softmax: $p(a_{t+1} | h_t) = \text{softmax}(\text{logits}_t)$.

**Hyperparameters.** The GRU uses a hidden size of $d_h = 16$ and dropout rate of 0.1 applied after the input projection and after the GRU output. The model is trained with Adam optimizer using cross-entropy loss, identical to the loss function used for SPICE models.

**Total parameters.** The GRU's parameter count scales with the hidden size and input dimension. For a given study, the input dimension is $A + n_{\text{reward}} + n_{\text{add}}$, and the model has $O(d_h^2 + d_h \cdot d_{\text{in}})$ parameters shared across all participants.

---

## 2. Study-Specific Models

Each SPICE model specifies a set of RNN submodules with their control signal inputs, a set of latent memory states, and the trial-by-trial computation that orchestrates how submodules update states and produce action logits. All SPICE models use the residual RNN submodule architecture and the two-stage training pipeline described in the main methods section.

All SPICE models use a learned participant embedding $e_p \in \mathbb{R}^{d_e}$ to capture individual differences in the RNN dynamics. The embedding is concatenated with the control signal inputs to each submodule. Individual differences in the SINDy equations are captured via per-participant SINDy coefficients.


### 2.1. Synthetic Study: Q-Learning Recovery

**Task.** Synthetic data generated from known Q-learning models with varying parameters (learning rates, forgetting, choice perseveration). Used for parameter recovery validation. The action space is $A = 2$.

#### SPICE Model

The model uses 4 submodules operating on 2 memory states:

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `value_reward_chosen` | $r_t$ | `value_reward` | chosen action | Reward learning for chosen action |
| `value_reward_not_chosen` | — | `value_reward` | unchosen action | Forgetting/decay for unchosen action |
| `value_choice_chosen` | — | `value_choice` | chosen action | Choice perseveration for chosen action |
| `value_choice_not_chosen` | — | `value_choice` | unchosen action | Choice decay for unchosen action |

**Memory states:** `value_reward` (initial: 0), `value_choice` (initial: 0).

**Logit computation:**
$$\text{logits}_t = V^{\text{reward}}_t + V^{\text{choice}}_t$$

**SINDy library degree:** 2 (quadratic). This allows recovery of asymmetric learning rates via the interaction term $V \cdot r$.

**Ground truth equations.** The model is designed to recover the standard asymmetric Rescorla-Wagner with forgetting and choice perseveration:
- Chosen value update: $\Delta V^{\text{reward}}_{\text{chosen}} = -\alpha_{\text{penalty}} \cdot V^{\text{reward}} + \alpha_{\text{reward}} \cdot r + (\alpha_{\text{penalty}} - \alpha_{\text{reward}}) \cdot V^{\text{reward}} \cdot r$
- Unchosen value update: $\Delta V^{\text{reward}}_{\text{unchosen}} = -f \cdot V^{\text{reward}}$ (forgetting toward initial value)
- Choice persistence: $\Delta V^{\text{choice}}_{\text{chosen}} = \alpha_c \cdot \beta_c - \alpha_c \cdot V^{\text{choice}}$; $\Delta V^{\text{choice}}_{\text{unchosen}} = -\alpha_c \cdot V^{\text{choice}}$

#### Benchmark Model

No benchmark model for this study (synthetic data generation only).


### 2.2. Braun 2018: Reward-Based Voluntary Task Switching

**Task.** Participants choose between two cognitive tasks (repeat or switch) in each trial. Each task has an associated point value (integer, 0-10) that evolves stochastically: the selected task's value decreases with probability 0.5, while the unselected task's value increases with probability 0.5. The paradigm measures the tradeoff between reward maximization and cognitive effort avoidance. The action space is $A = 2$ (repeat, switch).

#### SPICE Model

The model uses 6 submodules operating on 3 memory states. Modules are split by action to enable separate dynamics for repeating vs. switching:

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|-----------|----------------|---------------|-------------|-----------------|-------------|
| `reward_repeat` | $\Delta r_{\text{tasks}}$ | — (stateless) | repeat only | False | Instantaneous reward sensitivity for repeating |
| `reward_switch` | $\Delta r_{\text{tasks}}$ | — (stateless) | switch only | False | Instantaneous reward sensitivity for switching |
| `task_repeat` | `repeat` | `value_control` | repeat only | True | Evolving cognitive control cost for repeating |
| `task_switch` | `repeat` | `value_control` | switch only | True | Evolving cognitive control cost for switching |
| `fatigue_repeat` | $b / B$ | — (stateless) | repeat only | False | Fatigue effect on repeat tendency |
| `fatigue_switch` | $b / B$ | — (stateless) | switch only | False | Fatigue effect on switch tendency |

where $\Delta r_{\text{tasks}}$ is the signed difference in task values (other minus current), `repeat` is 1 if the previous action was repeat and 0 otherwise, and $b / B$ is the normalized block number (block index divided by total blocks, $B=12$).

**Memory states:** `value_reward` (initial: 0), `value_control` (initial: 0), `value_fatigue` (initial: 0).

**Control signal preprocessing:**
- `dreward_tasks`: The difference between the other task's value and the current task's value from the previous trial, negated for the repeat mask and kept as-is for the switch mask. This provides a relative reward signal.
- `repeat`: A binary indicator derived from the previous action (1 if repeat, 0 if switch), broadcast across items.
- `block`: Block number normalized by total blocks ($b/12$).

**Logit computation:**
$$\text{logits}_t = V^{\text{reward,repeat}}_t + V^{\text{reward,switch}}_t + V^{\text{control}}_t + V^{\text{fatigue,repeat}}_t + V^{\text{fatigue,switch}}_t$$

Note that the reward and fatigue modules are stateless (`include_state=False`), meaning they produce instantaneous mappings from their control signals rather than maintaining evolving state. Only `value_control` evolves across trials.

**SINDy library degree:** 2 (default).

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

**Note:** The model is stateless — it computes action values from instantaneous task features without maintaining internal state across trials.


### 2.3. Bustamante 2023: Patch Foraging

**Task.** Participants forage in a sequence of resource patches. On each trial they choose to harvest (stay in the current patch and receive a depleting reward) or exit (travel to a new patch with no reward). The task implements the classic patch-leaving problem from optimal foraging theory. The action space is $A = 2$ (harvest=0, exit=1).

#### SPICE Model

The model uses 6 submodules operating on 6 memory states (4 value states + 2 auxiliary buffers). Only the harvest action's value is updated; the exit value is kept at 0 as a reference point, mirroring the structure of the Marginal Value Theorem.

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `reward_environment` | $r_t$ | `value_reward_environment` | harvested | Global environment reward rate learning |
| `reward_patch_harvest` | $r_t$ | `value_reward_patch` | harvested | Current patch reward valuation |
| `reward_patch_exit` | — | `value_reward_patch` | exited | Patch value reset upon exit |
| `depletion_patch_harvest` | $\Delta r_t$ | `value_depletion_patch` | harvested | Within-patch depletion signal |
| `depletion_patch_exit` | — | `value_depletion_patch` | exited | Depletion reset upon exit |
| `continuation_patch` | $a_{t-1}$ | `value_continuation_patch` | all items | Action-dependent continuation effect |

**Memory states:**
- `value_reward_environment` (initial: 0) — tracks the global environment reward rate
- `value_reward_patch` (initial: 0) — tracks the current patch's reward value
- `value_depletion_patch` (initial: 0) — tracks within-patch reward depletion
- `value_continuation_patch` (initial: 0) — captures tendencies to continue current behavior
- `reward[t-1]` (initial: 0, auxiliary) — stores previous trial's reward for depletion computation
- `action[t-1]` (initial: 0, auxiliary) — stores previous action for continuation module

**States in logit:** `value_reward_environment`, `value_reward_patch`, `value_depletion_patch`, `value_continuation_patch`.

**Control signal preprocessing:**
- Rewards are extracted as partial feedback (only the harvested action receives a reward signal). The reward is broadcast across items for module input.
- The depletion signal $\Delta r_t = r_t - r_{t-1}$ is computed as the difference between the current and previous reward. When exiting and re-entering a patch, `reward[t-1]` is reset to 1.
- The action mask `harvested` is 1 for item 0 (harvest) when harvest was chosen; `exited` is 1 for item 0 when exit was chosen.

**Logit computation:**
$$\text{logits}_t = V^{\text{env}}_t + V^{\text{patch}}_t + V^{\text{depletion}}_t + V^{\text{continuation}}_t$$

**SINDy library degree:** 2 (default).

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
- `cumulative_reward` — total reward accumulated in current patch
- `n_harvests` — number of harvest trials in current patch
- `time_in_patch` — total time spent in current patch
- `env_reward_rate` ($\rho$) — estimated average gain rate in the environment
- `current_tree_state` ($s_i$) — expected reward from next harvest in current patch

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


### 2.4. Eckstein 2024/Castro 2025: Drifting Multi-Armed Bandit

**Task.** Participants make repeated choices among 4 arms arranged in a circle. Arm means follow independent Gaussian random walks: $\mu_{t,i} \sim \mathcal{N}(\lambda \mu_{t-1,i} + (1-\lambda) \cdot 50, \sigma_{\text{drift}})$ with $\lambda = 0.9836$, $\sigma_{\text{drift}} = 2.8$. Observations are noisy: $r_{t,i} \sim \mathcal{N}(\mu_{t,i}, \sigma_{\text{obs}})$ with $\sigma_{\text{obs}} = 4$. Rewards are normalized to [0, 1]. The action space is $A = 4$.

#### SPICE Model

The model uses 7 submodules operating on 7 memory states (5 value states + 2 auxiliary buffers).

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `value_reward_env` | $r_t$ | `value_reward_env` | all items | Global environment reward tracking |
| `value_reward_chosen` | $V^{\text{env}}_t$, $r_t$, $\bar{V}^{\text{reward}}$ | `value_reward` | chosen | Reward learning for chosen arm |
| `value_reward_not_chosen` | $V^{\text{env}}_t$, $\bar{V}^{\text{reward}}$ | `value_reward` | unchosen | Value update for unchosen arms |
| `value_choice` | $a_t$, $a_{t-1}$ | `value_choice` | all items | Choice perseveration |
| `value_exploration_chosen` | $\Delta V^+$, $\Delta V^-$ | `value_exploration` | chosen | Exploration signal for chosen arm |
| `value_exploration_not_chosen` | $\Delta V^+$, $\Delta V^-$ | `value_exploration` | unchosen | Exploration signal for unchosen arms |
| `bias_attention` | $a_{t-1}$, `is_adjacent`, `is_opposite` | `bias_attention` | all items | Spatial attention bias based on circular distance |

**Memory states:**
- `value_reward_env` (initial: 0) — global environment reward estimate
- `value_reward` (initial: 0) — per-arm reward values
- `value_choice` (initial: 0) — choice perseveration values
- `value_exploration` (initial: 0) — exploration-driven values
- `bias_attention` (initial: 0) — spatial attention biases
- `value_reward[t-1]` (initial: 0, auxiliary) — previous reward values for exploration computation
- `action[t-1]` (initial: 0, auxiliary) — previous action

**States in logit:** `value_reward_env`, `value_reward`, `value_choice`, `value_exploration`, `bias_attention` (summed at each trial).

**Control signal preprocessing:**
- $\bar{V}^{\text{reward}}$ is the mean of reward values across all arms (detached from the computation graph): $\bar{V}^{\text{reward}} = \text{mean}_i(V^{\text{reward}}_i)$.
- $\Delta V = V^{\text{reward}}_t - V^{\text{reward}}_{t-1}$ is the trial-to-trial change in reward values (detached). This is split into positive and negative components: $\Delta V^+ = \text{ReLU}(\Delta V)$ and $\Delta V^- = \text{ReLU}(-\Delta V)$.
- `is_adjacent` and `is_opposite` are binary indicators computed from the circular distance between the chosen arm and each other arm. For 4 arms arranged in a circle, adjacent means circular distance = 1 and opposite means circular distance = 2.

**Logit computation:**
$$\text{logits}_t = V^{\text{reward}}_t + V^{\text{choice}}_t + V^{\text{exploration}}_t + V^{\text{attention}}_t$$

Note: `value_reward_env` is used both as a state variable contributing to logits and as an input to the reward learning modules, providing a reference-point mechanism for computing prediction errors relative to the environment average.

**SINDy library degree:** 2 (default).

#### Benchmark Model: Discovered Program (Castro et al., 2025)

**Reference:** Castro et al. (2025), discovered via automated program synthesis.

A 13-parameter model per participant discovered by automated program search. It combines loss-averse temporal difference learning, exploration rate smoothing, Q-value decay, perseveration/switch bonuses, spatial attention biases, and cumulative choice tracking. The action space is $A = 4$.

**Parameters (per participant, 13 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\beta_r$ | $(0.01, 20)$ softplus | Value scaling in softmax |
| `lapse` | $(0.01, 0.99)$ sigmoid | Lapse rate (uniform exploration probability) |
| `prior` | $(0.01, 0.99)$ softplus | Initial Q-value for all arms |
| $\alpha_{\text{er}}$ | $(0.01, 0.99)$ sigmoid | Initial exploration rate |
| `decay_rate` | $(0.01, 0.99)$ sigmoid | Q-value multiplicative decay per trial |
| `attention_bias1` | unconstrained | Bonus toward previously chosen arm |
| `attention_bias2` | unconstrained | Bonus toward arm opposite to previous choice |
| `perseveration` | $\geq 0$ softplus | Bonus for repeating same arm |
| `switch_strength` | unconstrained | Bonus for switching to different arm |
| $\lambda$ | $(0, 1)$ softplus | Unused mixing parameter |
| $\gamma$ | $\geq 0$ softplus | Loss aversion parameter |
| `temperature` | $(10^{-6}, 100)$ softplus | Softmax temperature |
| $\beta_p$ | $\geq 0$ softplus | Cumulative-choice bonus scaling |

**Q-value update (loss-averse TD learning):**

$$\delta = r_t - \gamma (1 - r_t) - Q[a_t]$$
$$Q[a_t] \leftarrow Q[a_t] + \delta$$

The loss aversion parameter $\gamma$ asymmetrically weights negative outcomes: when $\gamma > 0$, the effective prediction error penalizes non-rewards more heavily.

**Exploration smoothing and decay:**

$$Q \leftarrow (1 - \varepsilon_t) \cdot Q + \varepsilon_t \cdot \bar{Q}$$
$$Q \leftarrow Q \cdot \text{decay\_rate}$$
$$\varepsilon_{t+1} = \varepsilon_t \cdot (1 - 10^{-3})$$

where $\varepsilon_t$ is the exploration rate (decays slowly each trial) and $\bar{Q} = \text{mean}(Q)$.

**Choice probabilities:**

$$p_{\text{base}} = \text{softmax}\left(\frac{\beta_r \cdot Q}{\text{temperature}} + \beta_p \cdot \log(1 + C)\right)$$
$$p(a) = (1 - \text{lapse}) \cdot p_{\text{base}}(a) + \frac{\text{lapse}}{4}$$

where $C_a$ is the cumulative choice count for arm $a$.

**Choice-conditioned bonuses (applied to log-probabilities):**

$$\log p(a) \mathrel{+}= \mathbb{1}[a_t = a_{t-1}] \cdot \text{perseveration} \cdot \mathbb{1}[a] + \mathbb{1}[a_t \neq a_{t-1}] \cdot \text{switch\_strength} \cdot \mathbb{1}[a]$$
$$\mathrel{+} \text{attention\_bias1} \cdot \mathbb{1}[a_{t-1}] + \text{attention\_bias2} \cdot \mathbb{1}[(a_t + 2) \bmod 4]$$
$$\mathrel{+} \mathbb{1}[a_t] \cdot \log(1 + \text{trials\_since\_last\_switch})$$

where $\mathbb{1}[a]$ denotes the one-hot encoding of arm $a$ among the 4 arms.


### 2.5. Dezfouli 2019: Two-Armed Bandit with Depression

**Task.** Participants (including individuals with depression) perform a two-armed bandit task with 12 blocks. Each block has fixed Bernoulli reward probabilities for each arm (e.g., 0.25 vs. 0.05). The task is designed to assess reinforcement learning mechanisms and how they differ in depression. The action space is $A = 2$.

#### SPICE Model

For this study, the precoded Working Memory model was used. This model augments standard reinforcement learning with explicit working memory buffers that store the last 3 trials of rewards and choices.

The model uses 4 submodules operating on 8 memory states (2 value states + 6 buffer states).

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `value_reward_chosen` | $r_t$, $r_{t-1}$, $r_{t-2}$, $r_{t-3}$ | `value_reward` | chosen | Reward learning from recent reward history |
| `value_reward_not_chosen` | $r_{t-1}$, $r_{t-2}$, $r_{t-3}$ | `value_reward` | unchosen | Unchosen value update from reward history |
| `value_choice_chosen` | $a_{t-1}$, $a_{t-2}$, $a_{t-3}$ | `value_choice` | chosen | Choice persistence from action history |
| `value_choice_not_chosen` | $a_{t-1}$, $a_{t-2}$, $a_{t-3}$ | `value_choice` | unchosen | Choice decay from action history |

**Memory states:**
- `value_reward` (initial: 0) — reward-driven action value
- `value_choice` (initial: 0) — choice-driven action value
- `buffer_reward_1`, `buffer_reward_2`, `buffer_reward_3` (initial: 0) — FIFO reward buffer storing per-action rewards from trials $t-1$, $t-2$, $t-3$
- `buffer_action_1`, `buffer_action_2`, `buffer_action_3` (initial: 0) — FIFO action buffer storing one-hot actions from trials $t-1$, $t-2$, $t-3$

**States in logit:** `value_reward`, `value_choice`.

**Buffer update mechanism.** The reward buffers shift per-action: when an action is chosen, its buffer entries shift (1 $\to$ 2 $\to$ 3) while the unchosen action's buffer remains unchanged. This implements action-specific working memory where each arm maintains its own reward history. The action buffers shift unconditionally (all actions share the same action history):

$$\text{buffer\_reward\_3} = \text{buffer\_reward\_2} \cdot a_t + \text{buffer\_reward\_3} \cdot (1 - a_t)$$
$$\text{buffer\_reward\_2} = \text{buffer\_reward\_1} \cdot a_t + \text{buffer\_reward\_2} \cdot (1 - a_t)$$
$$\text{buffer\_reward\_1}[a_t = 1] = r_t; \quad \text{buffer\_reward\_1}[a_t = 0] = \text{buffer\_reward\_1}$$
$$\text{buffer\_action\_3} = \text{buffer\_action\_2}, \quad \text{buffer\_action\_2} = \text{buffer\_action\_1}, \quad \text{buffer\_action\_1} = a_t$$

**SINDy library preprocessing.** Because rewards and choices are binary signals, polynomial terms involving powers of these binary signals (e.g., $r^2 = r$) are redundant. The model preprocesses the SINDy coefficient presence masks to remove these redundant terms, reducing the effective library size.

**Logit computation:**
$$\text{logits}_t = V^{\text{reward}}_t + V^{\text{choice}}_t$$

**SINDy library degree:** 2 (default).

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


### 2.6. Ganesh 2024a: Perceptual Contrast Bandit

**Task.** Participants choose between two Gabor patches with different visual contrasts. A latent contingency parameter $\mu$ determines the reward probability: the action matching the "correct" perceptual state (determined by which patch has higher contrast) is rewarded with probability $\mu = 0.75$, and the mismatching action with probability $1 - \mu$. The task requires jointly learning the perceptual mapping (which contrast is higher) and the reward contingency. The action space is $A = 2$ (left, right).

#### SPICE Model

The model uses 4 submodules operating on 2 memory states. Internal computations operate in a contrast-based item space (low-contrast=0, high-contrast=1) that is decoupled from the position-based action space.

| Submodule | Control signals | State updated | `include_state` | Activation | Description |
|-----------|----------------|---------------|-----------------|------------|-------------|
| `perception_certainty` | $\Delta c_t$ | — (stateless) | False | sigmoid | Maps contrast difference to perceptual certainty |
| `reward_learning_chosen` | $r_t$, $\text{cert}_t$ | `value_reward_contrast` | True | — | Reward learning for chosen item, modulated by certainty |
| `reward_learning_unchosen` | $\text{cert}_t$ | `value_reward_contrast` | True | — | Unchosen item dynamics, modulated by certainty |
| `choice_persistance` | $a^{\text{item}}_t$, $\text{cert}_t$, $\text{cert}_{t+1}$ | `value_choice_contrast` | True | — | Choice persistence in item space |

**Memory states:** `value_reward_contrast` (initial: 0), `value_choice_contrast` (initial: 0).

**Key design feature: Item-space/action-space decoupling.** The model internally represents values in item space (low-contrast, high-contrast) rather than action space (left, right). This separation is critical because the stimulus-to-position mapping changes every trial:
1. Actions are remapped from position space to item space using the sign of the contrast difference: when $\Delta c \leq 0$, left=low and right=high; when $\Delta c > 0$, left=high and right=low.
2. Learning updates operate in item space with deterministic masks.
3. At decision time, item-space logits are probabilistically mapped back to action space using the perceptual certainty for the next trial's contrast.

**Perceptual certainty computation.** The perception module maps the signed contrast difference to a certainty value via sigmoid activation: $\text{cert}_t = \sigma(f(\Delta c_t))$, where $f$ is the RNN submodule. This value represents the model's confidence that the item-space assignment is correct.

**Logit computation (soft item-to-action mapping):**
$$V^{\text{item}}_t = V^{\text{reward}}_t + V^{\text{choice}}_t$$
$$\text{cert}'_{t+1} = \text{cert}_{t+1} / 2 + 0.5 \quad \text{(rescaled to [0.5, 1.0])}$$
$$V^{\text{mixed}}_t = \text{cert}'_{t+1} \cdot V^{\text{item}}_t + (1 - \text{cert}'_{t+1}) \cdot \text{flip}(V^{\text{item}}_t)$$
$$\text{logits}_t = \begin{cases} V^{\text{mixed}}_t & \text{if } \Delta c_{t+1} < 0 \\ \text{flip}(V^{\text{mixed}}_t) & \text{if } \Delta c_{t+1} \geq 0 \end{cases}$$

This soft mapping ensures that when perceptual certainty is high, item values map cleanly to action values, and when certainty is low, the mapping is more uniform.

**SINDy library degree:** 2 (default).

#### Benchmark Model: Bayesian Belief-Update Model

**Reference:** Ganesh et al. (2024), normative agent model.

A Bayesian observer that maintains a discretized belief distribution over the contingency parameter $\mu \in [0, 1]$, which links the latent perceptual state to reward probability. The model performs exact posterior updating given its perceptual noise model. The action space is $A = 2$ (left, right).

**Parameters (per participant, 2 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $\beta$ | $(1, 25)$ softplus + 1 | Inverse softmax temperature |
| $\sigma$ | $(0.01, 0.1)$ sigmoid-scaled | Perceptual noise (std. dev. of noisy observation) |

**Generative model:**
1. $\mu \sim P(\mu)$ — prior over contingency parameter (initially uniform over $N = 100$ grid points)
2. $s \sim \text{Bernoulli}(\mu)$ — latent state determining which side is rewarded
3. $o_t | s \sim \mathcal{N}(\Delta c, \sigma^2)$ — noisy perceptual observation of contrast difference
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


### 2.7. Hwang 2026: Chimpanzee Gestural Communication

**Task.** Pairs of chimpanzees engage in dyadic interactions. At each timestep, a chimpanzee performs one of 4 behavioral actions: action (approach/movement), grooming, gesture, or waiting. The model predicts the next behavior of individual 1 (ID1) given the behavioral history of both individuals. The action space is $A = 4$.

#### SPICE Model

The model uses 3 submodules operating on 1 shared memory state. Each submodule is responsible for updating one behavioral dimension's contribution to the shared value state.

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `module_action` | $a^{\text{ID1}}_t$, $g^{\text{ID1}}_t$, $j^{\text{ID1}}_t$, $s^{\text{ID1}}_t$, $a^{\text{ID2}}_t$, $g^{\text{ID2}}_t$, $j^{\text{ID2}}_t$, $s^{\text{ID2}}_t$ | `values` | action dim only | Updates action dimension of shared values |
| `module_grooming` | (same 8 signals) | `values` | grooming dim only | Updates grooming dimension of shared values |
| `module_gesture` | (same 8 signals) | `values` | gesture dim only | Updates gesture dimension of shared values |

where $a$, $g$, $j$, $s$ denote the action, grooming, gesture, and scratch/waiting behavioral indicators for each individual.

**Memory states:** `values` (initial: 0) — a single shared state vector of size $A=4$.

**Dual identity embeddings.** This model uses both participant and experiment embeddings to represent the two individuals in the dyad:
- `participant_embedding` represents ID1 (the sender/predicted individual)
- `experiment_embedding` represents ID2 (the partner)

Both embeddings are passed to each submodule. This allows the model to capture both individual behavioral tendencies and dyad-specific interaction patterns.

**Control signal structure.** Each submodule receives all 8 behavioral indicators from both individuals. The behavioral indicators of ID1 come from the one-hot encoded action vector. The behavioral indicators of ID2 come from an additional input column that is one-hot encoded within the forward pass.

**Action masks.** Fixed masks select which dimension of the shared value state each module updates:
- `mask_action` = (1, 0, 0, 0)
- `mask_grooming` = (0, 1, 0, 0)
- `mask_gesture` = (0, 0, 1, 0)

The waiting dimension (index 3) is not explicitly modeled and serves as a baseline.

**Custom loss function.** A custom cross-entropy loss is used which filters out trials where ID1 is waiting (class 3), so the model only learns to predict non-waiting behaviors.

**Logit computation:**
$$\text{logits}_t = V_t$$

**SINDy library degree:** 2 (default).

#### Benchmark Model

No benchmark model for this study.


### 2.8. Bruckner 2025: Helicopter Task (Predictive Inference)

**Task.** Participants predict where bags of supplies will fall from a helicopter. On each trial, the participant positions a bucket at location $b_t \in [0, 300]$ on a horizontal screen. The outcome (bag drop position) is $x_t \sim \mathcal{N}(\mu_t, \sigma^2)$ with $\sigma = 15$ pixels. The helicopter's true position $\mu_t$ is stable for extended periods but undergoes occasional change points, creating an environment where participants must balance stable estimation against rapid adjustment after change points. Between trials, the bucket may be displaced to a new initial position $z_{t+1}$, inducing an anchoring effect. All positions are normalized to $[0, 1]$ by dividing by 300. The output is a continuous scalar prediction ($A = 1$), and the model is trained with MSE loss instead of cross-entropy.

#### SPICE Model

The model uses 6 submodules operating on 4 memory states. It implements a dual learning rate architecture inspired by the Reduced Bayesian Model, where prediction error magnitude determines whether a surprise-driven (changepoint) or uncertainty-driven (baseline) learning rate governs belief updating. An anchoring module captures the bias induced by bucket displacement between trials.

| Submodule | Control signals | State updated | Action mask | `include_state` | Description |
|-----------|----------------|---------------|-------------|-----------------|-------------|
| `belief_update` | $\delta_t$, `catch`, $v_t$ | `belief_value` | — | True | Belief update from prediction error, modulated by catch trial status and helicopter visibility |
| `changepoint_lr_update` | `catch`, $v_t$ | `surprise_value` | $\mathbb{1}[|\delta_t| > 3\sigma]$ | True | Surprise-driven learning rate update (fires on large PE) |
| `changepoint_lr_decay` | — | `surprise_value` | $1 - \mathbb{1}[|\delta_t| > 3\sigma]$ | True | Surprise learning rate decay (fires on small PE) |
| `uncertainty_lr_update` | `catch`, $v_t$ | `uncertainty_value` | $1 - \mathbb{1}[|\delta_t| > 3\sigma]$ | True | Uncertainty-driven learning rate update (fires on small PE) |
| `uncertainty_lr_decay` | — | `uncertainty_value` | $\mathbb{1}[|\delta_t| > 3\sigma]$ | True | Uncertainty learning rate decay (fires on large PE) |
| `anchor_update` | $y_t$ | `anchor_value` | — | False | Stateless anchoring bias from bucket displacement |

**Memory states:**
- `belief_value` (initial: 0.5) — internal estimate of helicopter position (center of screen)
- `surprise_value` (initial: 0) — changepoint probability state; $\sigma(\text{surprise\_value}) = \omega$ (surprise-driven learning rate)
- `uncertainty_value` (initial: 0) — relative uncertainty state; $\sigma(\text{uncertainty\_value}) = \tau$ (uncertainty-driven baseline learning rate)
- `anchor_value` (initial: 0) — per-trial anchoring correction (reset to zero each trial)

**States in logit:** `belief_value`, `surprise_value`, `uncertainty_value`, `anchor_value`.

**Control signal preprocessing:**
- Prediction error: $\delta_t = x_t - \hat{\mu}_t$, the difference between the observed outcome and the model's current belief.
- PE magnitude mask: $\mathbb{1}[|\delta_t| > 3\sigma]$, where $\sigma$ is the bucket width (an observable task variable, not a latent parameter). This externalized binary gate routes trials to either the surprise or uncertainty learning rate pathway, mirroring the change-point detection mechanism of the Bayesian benchmark.
- Anchor shift: $y_t = z_{t+1} - b_t$, the displacement between the next trial's initial bucket position and the participant's current bucket position.
- Catch trial indicator: `catch` $\in \{0, 1\}$, whether the participant caught the coin on this trial.
- Helicopter visibility: $v_t \in \{0, 1\}$, whether the helicopter was visible on the next trial (time-shifted by $-1$ during data loading).

**Logit computation (gated output + anchoring):**
$$\omega_t = \sigma(\text{surprise\_value}_t), \quad \tau_t = \sigma(\text{uncertainty\_value}_t)$$
$$\alpha_t = \mathbb{1}[|\delta_t| > 3\sigma] \cdot \omega_t + (1 - \mathbb{1}[|\delta_t| > 3\sigma]) \cdot \tau_t$$
$$\hat{b}_{t+1} = b_t + \alpha_t \cdot (\hat{\mu}_t - b_t) + V^{\text{anchor}}_t$$

The predicted next bucket position interpolates between the current bucket position $b_t$ and the internal belief $\hat{\mu}_t$ using a composite learning rate $\alpha_t$, plus an anchoring correction. On large-PE trials (likely change points), the surprise-driven rate $\omega_t$ governs the update; on small-PE trials (stable periods), the uncertainty-driven rate $\tau_t$ governs.

**SINDy library degree:** 2 (default).

**Note:** The `anchor_update` module uses `include_state=False` because the anchor correction is stateless — it is reset to zero each trial and computes an instantaneous mapping from the bucket displacement.

#### Benchmark Model: Reduced Bayesian Model (RBM)

**Reference:** Bruckner et al. (2025), Eqs. 4-14.

A reduced Bayesian model that maintains a belief about the helicopter's position and an uncertainty estimate that modulates learning rate. The belief is reset to the participant's actual prediction $b_t$ at each trial (subjective prediction errors), while uncertainty dynamics carry forward across trials. On catch trials (helicopter visible), the belief is additionally updated with the observed helicopter position. The output is a continuous scalar prediction ($A = 1$).

**Parameters (per participant, 4 total):**

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| $h$ | $(0, 1)$ sigmoid | Hazard rate — prior probability of a change point |
| $s$ | $(0, 1)$ sigmoid | Surprise sensitivity — modulates change-point detection |
| $u$ | unconstrained (exponentiated) | Uncertainty underestimation — $\exp(u)$ divides posterior uncertainty |
| $q$ | unconstrained | Reward bias — learning rate increase for high-value trials |

**Fixed task parameters:**
- $\sigma = 17.5$ pixels — outcome noise standard deviation
- $\sigma_H = 17.5$ pixels — helicopter cue noise standard deviation (used on catch trials)

**State variables (per session):**
- $\hat{\sigma}^2_t$ — estimation uncertainty (initialized to $100 / 300^2$)
- $\tau_t$ — relative uncertainty (initialized to 0.5)

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

The model dynamically adjusts its learning rate via change-point detection: when a change point is detected (high $\omega_t$), the learning rate increases to rapidly incorporate the new outcome; during stable periods (low $\omega_t$), the learning rate decreases for more precise estimation. The uncertainty underestimation parameter $u$ captures individual differences in confidence calibration. On catch trials, the helicopter's visible position provides an additional cue that is integrated with the model's belief according to their relative uncertainties (Eqs. 11-14).


### 2.9. Weber 2024: Laser Tracking (Shield Movement)

**Task.** Participants control a shield on a circular track (0°–360°) to intercept laser beams. On each trial, a new laser beam appears at a position drawn from a von Mises distribution whose mean undergoes discrete change points (controlled by volatility and stochasticity conditions). The shield moves at a fixed speed of 1°/frame, so the maximum movement between two laser beams is limited by the inter-beam interval (trial duration in frames). If the laser lands within the shield's angular width (±10°), it is "caught." The task requires tracking a latent laser mean under movement constraints — a circular analog of the helicopter task (Bruckner 2025). The output is a continuous 2D prediction in $(\sin, \cos)$ space representing the shield position at the next laser beam, and the model is trained with a custom clamped angular MSE loss instead of cross-entropy.

**Data representation.** All angular positions are encoded as $(\sin\theta, \cos\theta)$ pairs to handle the circular geometry. The data is event-based: each trial corresponds to one laser beam event, regardless of the inter-beam time interval. The number of actions and reward features are both $A = 2$ (the sin and cos components).

#### SPICE Model

The model uses 4 submodules operating on 2 memory states. Belief and learning rate updates are split by catch outcome (whether the shield intercepted the laser), using externalized binary gating via the `action_mask` mechanism. A hardcoded gated output equation translates the internal belief into a physically constrained shield position prediction.

| Submodule | Control signals | State updated | Action mask | Description |
|-----------|----------------|---------------|-------------|-------------|
| `belief_update_caught` | $\delta_t$ | `belief_value` | `laser_caught` | Belief update when shield catches laser (tracking well) |
| `belief_update_missed` | $\delta_t$ | `belief_value` | $1 -$ `laser_caught` | Belief update when shield misses laser (tracking poorly) |
| `lr_update_caught` | — | `lr_value` | `laser_caught` | Learning rate adaptation when catching |
| `lr_update_missed` | — | `lr_value` | $1 -$ `laser_caught` | Learning rate adaptation when missing |

**Memory states:**
- `belief_value` (initial: 0, then set to first laser observation) — internal belief about the laser mean position, represented as $(\sin, \cos)$ across items $I = 2$
- `lr_value` (initial: 0) — dynamic learning rate state; $\sigma(\text{lr\_value}) = \alpha \in [0, 1]$

**States in logit:** `belief_value`, `lr_value`.

**Control signal preprocessing:**
- Prediction error: $\delta_t = (\sin\theta^{\text{laser}}_t, \cos\theta^{\text{laser}}_t) - \hat{\mu}_t$, the component-wise difference between the observed laser position and the model's current belief, computed in $(\sin, \cos)$ space.
- Catch mask: `laser_caught` $\in \{0, 1\}$, whether the shield was within ±10° of the laser. This binary signal is externalized from the RNN inputs and used as `action_mask` to route each trial to the appropriate belief and learning rate modules.

**Belief initialization.** On the first trial (no prior state), the belief is initialized to the first observed laser position: $\hat{\mu}_0 = (\sin\theta^{\text{laser}}_0, \cos\theta^{\text{laser}}_0)$.

**Logit computation (gated output):**
$$\alpha_t = \sigma(\text{lr\_value}_t)$$
$$\hat{s}_{t+1} = s_t + \alpha_t \cdot (\hat{\mu}_t - s_t)$$

where $s_t = (\sin\theta^{\text{shield}}_t, \cos\theta^{\text{shield}}_t)$ is the current shield position and $\hat{\mu}_t$ is the internal belief. The predicted shield position interpolates between the current position and the belief, with the interpolation rate $\alpha_t$ controlled by the learned dynamic learning rate.

**Custom loss function: Clamped Angular MSE.** Because the shield has a finite movement speed (1°/frame), the raw belief prediction may be physically unreachable within the inter-beam interval. The loss function computes the feasible shield position by clamping the predicted movement:

$$\Delta = \hat{\mu}_t - s_t, \quad f = \min\left(\frac{v \cdot \Delta t \cdot (\pi / 180)}{||\Delta|| + \epsilon}, 1\right)$$
$$\hat{s}^{\text{clamped}}_{t+1} = s_t + f \cdot \Delta$$
$$\mathcal{L} = \text{MSE}(\hat{s}^{\text{clamped}}_{t+1}, s^{\text{actual}}_{t+1})$$

where $v = 1$ °/frame is the movement speed, $\Delta t$ is the inter-beam interval in frames, and the fraction $f$ ensures the predicted movement does not exceed what is physically achievable. The target $s^{\text{actual}}_{t+1}$ is the participant's actual shield position at the next laser beam. The targets are packed as $[s^{\text{actual}}_{t+1}, s_t, \Delta t]$ to provide the loss function with the current position and timing information needed for clamping.

**SINDy library degree:** 2 (default).

**Additional inputs:** `laser_caught`, `volatility`, `stochasticity`, `trial_duration_frames`, `trueMean` (ground truth laser mean, for diagnostic plotting only).

#### Benchmark Model

No study-specific benchmark model. The GRU baseline (Section 1) is used as the comparison model, adapted for continuous $(\sin, \cos)$ output with the same clamped angular MSE loss function.

---

## 3. Summary Table

| Study | Task | $A$ | SPICE Modules | SPICE States | SINDy Degree | Benchmark Model | Benchmark Params/Participant |
|-------|------|-----|--------------|-------------|-------------|----------------|------------------------------|
| Synthetic | Q-learning recovery | 2 | 4 | 2 | 2 | — | — |
| Braun 2018 | Voluntary task switching | 2 | 6 | 3 | 2 | Expected Value of Control | 5 |
| Bustamante 2023 | Patch foraging | 2 | 6 | 6 (4 value + 2 buffer) | 2 | Marginal Value Theorem | 3-5 |
| Eckstein 2024/Castro 2025 | Drifting 4-armed bandit | 4 | 7 | 7 (5 value + 2 buffer) | 2 | Discovered Program | 13 |
| Dezfouli 2019 | Two-armed bandit (depression) | 2 | 4 | 8 (2 value + 6 buffer) | 2 | Generalized Q-Learning | 12 |
| Ganesh 2024a | Perceptual contrast bandit | 2 | 4 | 2 | 2 | Bayesian Belief-Update | 2 |
| Hwang 2026 | Chimpanzee communication | 4 | 3 | 1 | 2 | — | — |
| Bruckner 2025 | Helicopter task (predictive inference) | 1 (continuous) | 6 | 4 | 2 | Reduced Bayesian Model | 4 |
| Weber 2024 | Laser tracking (shield movement) | 2 (continuous, sin/cos) | 4 | 2 | 2 | GRU baseline | — |

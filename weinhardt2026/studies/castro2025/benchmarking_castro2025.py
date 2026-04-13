import sys
import os

import torch
import numpy as np
from typing import Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_sessiondim

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir+'/../../..')

from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior


def get_dataset(path_data: str = None, test_sessions: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:

    # Load your data
    if path_data is None:
        path_data = 'data/eckstein2024.csv'

    dataset = csv_to_dataset(
        file = path_data,
        )
    dataset.normalize_rewards()

    n_participants = dataset.n_participants
    n_actions = dataset.n_actions

    if verbose:
        print(f"Shape of dataset: {dataset.xs.shape}")
        print(f"Number of participants: {n_participants}")
        print(f"Number of actions in dataset: {n_actions}")

    if test_sessions is not None:
        dataset_train, dataset_test = split_data_along_sessiondim(dataset, test_sessions)
    else:
        dataset_train, dataset_test = dataset, None
        
    info_dataset = {
        'n_participants': n_participants,
        'n_actions': n_actions,
    }
    
    return dataset_train, dataset_test, info_dataset


class Castro2025Model(torch.nn.Module):
    """
    Cognitive model from Castro et al. 2025 (discovered program) for multi-armed bandit task.
    
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------
    Model features:
    - Loss aversion learning (gamma parameter)
    - Exploration rate with decay
    - Perseveration and switch bonuses
    - Attention biases
    - Cumulative choice tracking
    - Softmax with temperature and lapse rate

    Original JAX code:
    def agent(params, choice, reward, agent_state = None):
        num_params = 13

        params = jnp.clip(params, -5, 5)

        beta_r = jnp.clip(jax.nn.softplus(params[0]), 0.01, 20)
        lapse = jnp.clip(jax.nn.sigmoid(params[1]), 0.01, 0.99)
        prior = jnp.clip(jax.nn.softplus(params[2]), 0.01, 0.99)
        alpha_exploration_rate = jnp.clip(jax.nn.sigmoid(params[3]), 0.01, 0.99)
        decay_rate = jnp.clip(jax.nn.sigmoid(params[4]), 0.01, 0.99)
        attention_bias1 = params[5]
        attention_bias2 = params[6]
        perseveration_strength = jax.nn.softplus(params[7])
        switch_strength = params[8]
        lambda_param = jnp.clip(jax.nn.softplus(params[9]), 0.0, 1.0)
        gamma = jax.nn.softplus(params[10]) # Loss aversion parameter
        temperature = jnp.clip(jax.nn.softplus(params[11]) + 1e-6, 1e-6, 100) #Softmax temperature
        beta_p = jax.nn.softplus(params[12])

        if agent_state is None:
            q_values = jnp.ones((4,)) * prior
            old_choice = -1
            trial_since_last_switch = 0
            exploration_rate = alpha_exploration_rate
            cumchoice = jnp.zeros((4,))
        else:
            q_values = agent_state[:4]
            old_choice = agent_state[4]
            trial_since_last_switch = agent_state[5]
            exploration_rate = agent_state[6]
            cumchoice = agent_state[7:11]

        if choice is not None and reward is not None:
            delta = reward - gamma*(1-reward) - q_values[choice]
            q_values = q_values.at[choice].set(q_values[choice] + delta)
            
            trial_since_last_switch = jnp.where(choice == old_choice, trial_since_last_switch + 1, 0)
            exploration_rate = exploration_rate * (1 - 1e-3) # decay exploration rate slowly
            cumchoice = cumchoice.at[choice].set(cumchoice[choice] + 1)
            
        q_values = (1 - exploration_rate) * q_values + exploration_rate * jnp.mean(q_values)
        q_values = q_values * decay_rate

        choice_probs = (1 - lapse) * jax.nn.softmax(beta_r * q_values / temperature + beta_p * jnp.log(1 + cumchoice)) + lapse / 4
        choice_logits = jnp.log(choice_probs)

        if choice is not None:
            perseveration_bonus = (choice == old_choice) * perseveration_strength * jax.nn.one_hot(choice, num_classes=4)
            switch_bonus = (choice != old_choice) * switch_strength * jax.nn.one_hot(choice, num_classes=4)
            attention_bonus1 = attention_bias1 * jax.nn.one_hot(old_choice, num_classes=4)
            attention_bonus2 = attention_bias2 * jax.nn.one_hot((choice + 2) % 4, num_classes=4)
            
            choice_logits = (
                choice_logits + perseveration_bonus + switch_bonus + attention_bonus1 + attention_bonus2 +
                jax.nn.one_hot(choice, 4) * jnp.log(trial_since_last_switch + 1))
        
        agent_state = jnp.concatenate(
            [q_values, jnp.array([choice, trial_since_last_switch, exploration_rate]),
            cumchoice])
            
        return choice_logits, agent_state
    -------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------

    Translated from the original JAX implementation. One parameter set per participant.
    
    Model parameters (per participant):
    - beta_r:                   Value scaling in softmax
    - lapse:                    Lapse rate (uniform exploration probability)
    - prior:                    Initial Q-value
    - alpha_exploration_rate:   Initial exploration rate
    - decay_rate:               Q-value decay per trial
    - attention_bias1:          Bonus toward previously chosen action
    - attention_bias2:          Bonus toward action opposite to previous choice
    - perseveration_strength:   Bonus for repeating the same action
    - switch_strength:          Bonus for switching to a different action
    - lambda_param:             Unused mixing parameter (kept for parameter recovery)
    - gamma:                    Loss aversion (asymmetric reward weighting)
    - temperature:              Softmax temperature
    - beta_p:                   Cumulative-choice bonus scaling

    State variables (per trial):
    - q_values:                 (B, n_actions) — action values
    - old_choice:               (B,)           — index of the previously chosen action (-1 initially)
    - trial_since_last_switch:  (B,)           — consecutive trials on the same action
    - exploration_rate:         (B,)           — current exploration rate (decays each trial)
    - cumchoice:                (B, n_actions) — cumulative choice counts per action
    """

    def __init__(
        self,
        n_actions: int = 4,
        n_participants: int = 1,
        batch_first: bool = False,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.n_participants = n_participants
        self.batch_first = batch_first

        # Raw (unconstrained) parameters — one value per participant.
        # Transformations applied in the properties below mirror the JAX model.
        self.beta_r_raw               = torch.nn.Parameter(torch.zeros(n_participants))
        self.lapse_raw                = torch.nn.Parameter(torch.zeros(n_participants))
        self.prior_raw                = torch.nn.Parameter(torch.zeros(n_participants))
        self.alpha_exploration_raw    = torch.nn.Parameter(torch.zeros(n_participants))
        self.decay_rate_raw           = torch.nn.Parameter(torch.zeros(n_participants))
        self.attention_bias1_raw      = torch.nn.Parameter(torch.zeros(n_participants))
        self.attention_bias2_raw      = torch.nn.Parameter(torch.zeros(n_participants))
        self.perseveration_raw        = torch.nn.Parameter(torch.zeros(n_participants))
        self.switch_strength_raw      = torch.nn.Parameter(torch.zeros(n_participants))
        self.lambda_param_raw         = torch.nn.Parameter(torch.zeros(n_participants))
        self.gamma_raw                = torch.nn.Parameter(torch.zeros(n_participants))
        self.temperature_raw          = torch.nn.Parameter(torch.zeros(n_participants))
        self.beta_p_raw               = torch.nn.Parameter(torch.zeros(n_participants))

    # ------------------------------------------------------------------
    # Parameter transforms (raw → constrained, as in the JAX code)
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_raw(x: torch.Tensor) -> torch.Tensor:
        # JAX clips all raw params to [-5, 5] before any transform.
        return torch.clamp(x, -5.0, 5.0)

    @property
    def beta_r(self):
        return torch.clamp(torch.nn.functional.softplus(self._clip_raw(self.beta_r_raw)), 0.01, 20.0)

    @property
    def lapse(self):
        return torch.clamp(torch.sigmoid(self._clip_raw(self.lapse_raw)), 0.01, 0.99)

    @property
    def prior(self):
        return torch.clamp(torch.nn.functional.softplus(self._clip_raw(self.prior_raw)), 0.01, 0.99)

    @property
    def alpha_exploration_rate(self):
        return torch.clamp(torch.sigmoid(self._clip_raw(self.alpha_exploration_raw)), 0.01, 0.99)

    @property
    def decay_rate(self):
        return torch.clamp(torch.sigmoid(self._clip_raw(self.decay_rate_raw)), 0.01, 0.99)

    @property
    def attention_bias1(self):
        return self._clip_raw(self.attention_bias1_raw)

    @property
    def attention_bias2(self):
        return self._clip_raw(self.attention_bias2_raw)

    @property
    def perseveration_strength(self):
        return torch.nn.functional.softplus(self._clip_raw(self.perseveration_raw))

    @property
    def switch_strength(self):
        return self._clip_raw(self.switch_strength_raw)

    @property
    def lambda_param(self):
        return torch.clamp(torch.nn.functional.softplus(self._clip_raw(self.lambda_param_raw)), 0.0, 1.0)

    @property
    def gamma(self):
        return torch.nn.functional.softplus(self._clip_raw(self.gamma_raw))

    @property
    def temperature(self):
        return torch.clamp(torch.nn.functional.softplus(self._clip_raw(self.temperature_raw)) + 1e-6, 1e-6, 100.0)

    @property
    def beta_p(self):
        return torch.nn.functional.softplus(self._clip_raw(self.beta_p_raw))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs, prev_state=None):
        """
        Args:
            inputs: (T, B, F) or (B, T, F) if batch_first.
                    Features: [actions (one-hot), rewards (one-hot), ..., participant_id]
            prev_state: optional dict of state tensors from a previous call.

        Returns:
            logits: (T, B, n_actions) or (B, T, n_actions) if batch_first
            state:  dict of final state tensors (detached from graph)
        """

        if self.batch_first:
            inputs = inputs.permute(1, 2, 0, 3)  # → (T, B, F)
        inputs = inputs.squeeze(1)

        T, B, _ = inputs.shape
        inputs = inputs.nan_to_num(0.0)

        # Participant IDs are stored in the last feature column (SPICE convention)
        participant_ids = inputs[0, :, -1].long()  # (B,)

        # Retrieve per-participant parameters indexed by batch participant IDs
        beta_r   = self.beta_r[participant_ids]                    # (B,)
        lapse    = self.lapse[participant_ids]                     # (B,)
        prior    = self.prior[participant_ids]                     # (B,)
        alpha_er = self.alpha_exploration_rate[participant_ids]    # (B,)
        decay    = self.decay_rate[participant_ids]                # (B,)
        ab1      = self.attention_bias1[participant_ids]           # (B,)
        ab2      = self.attention_bias2[participant_ids]           # (B,)
        perv     = self.perseveration_strength[participant_ids]    # (B,)
        sw       = self.switch_strength[participant_ids]           # (B,)
        gam      = self.gamma[participant_ids]                     # (B,)
        temp     = self.temperature[participant_ids]               # (B,)
        beta_p   = self.beta_p[participant_ids]                    # (B,)

        # Initialise or restore state
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=B, prior=prior, exploration_rate=alpha_er)

        logits = torch.zeros(T, B, self.n_actions, device=inputs.device)

        for t in range(T):
            actions_t = inputs[t, :, :self.n_actions]              # (B, n_actions) one-hot
            rewards_t = inputs[t, :, self.n_actions:2 * self.n_actions]  # (B, n_actions) one-hot

            q      = self.state['q_values']               # (B, n_actions)
            old_ch = self.state['old_choice']             # (B,)
            tsls   = self.state['trial_since_last_switch']# (B,)
            er     = self.state['exploration_rate']       # (B,)
            cc     = self.state['cumchoice']              # (B, n_actions)

            # ---- Update state based on observed choice & reward ----
            # Scalar reward for the chosen action
            reward_scalar = (rewards_t * actions_t).sum(dim=-1)   # (B,)

            # Loss-averse TD error: r - γ*(1-r) - Q[choice]
            q_chosen = (q * actions_t).sum(dim=-1)                # (B,) 
            delta = reward_scalar - gam * (1.0 - reward_scalar) - q_chosen  # (B,)

            # Update Q-value for chosen action only (vectorised via one-hot mask)
            q = q + delta.unsqueeze(-1) * actions_t               # (B, n_actions)


            # Current choice index (-1 means no previous choice)
            choice_idx = actions_t.argmax(dim=-1)         # (B,)
            had_choice = actions_t.sum(dim=-1) > 0        # (B,) whether a valid choice was made

            # Increment counter if same action was repeated, else reset to 0
            same_action = (choice_idx == old_ch).float()
            tsls = torch.where(had_choice, same_action * (tsls + 1), tsls) # if had_choice=True & same_action=True → increment; 
                                                                             # if had_choice=True & same_action=False (=0) → reset to 0; 
                                                                                # if had_choice=False → keep previous tsls (no update without a valid choice)

            # Slow exploration-rate decay
            # In JAX this only happens when a valid choice/reward update exists.
            er = torch.where(had_choice, er * (1.0 - 1e-3), er) # per batch row: if valid choice, decay; otherwise keep unchanged.

            # Cumulative choice count
            cc = cc + actions_t  # actions_t is one-hot, so this increments the count only for the chosen action (choice=True)

            # ---- Global Q-value update (exploration smoothing + decay) ----
            q_mean = q.mean(dim=-1, keepdim=True)                 # (B, 1)
            q = (1.0 - er.unsqueeze(-1)) * q + er.unsqueeze(-1) * q_mean
            q = q * decay.unsqueeze(-1)

            # ---- Compute choice probabilities ----
            # Base logit: temperature-scaled Q-values + cumulative-choice bonus
            base_logits = beta_r.unsqueeze(-1) * q / temp.unsqueeze(-1) + beta_p.unsqueeze(-1) * torch.log1p(cc)    # (B, n_actions)

            probs = torch.softmax(base_logits, dim=-1)
            choice_probs = (1.0 - lapse.unsqueeze(-1)) * probs + lapse.unsqueeze(-1) / self.n_actions     # (B, n_actions)
            #choice_logits = torch.log(choice_probs + 1e-8)              # (B, n_actions)
            choice_logits = torch.log(choice_probs)

            # ---- Add choice-conditioned bonuses (perseveration, switch, attention) ----
            # One-hot encodings for bonus computation
            choice_oh  = actions_t                                                     # (B, n_actions)
            # JAX one_hot(-1) -> all zeros; preserve that behavior explicitly.
            old_choice_oh     = torch.nn.functional.one_hot(old_ch.clamp(min=0), self.n_actions).float()
            old_choice_oh     = old_choice_oh * (old_ch >= 0).unsqueeze(-1).float()
            opposite_oh = torch.nn.functional.one_hot((choice_idx + 2) % self.n_actions, self.n_actions).float()

            persev_bonus  = same_action.unsqueeze(-1) * perv.unsqueeze(-1) * choice_oh
            switch_bonus  = (1.0 - same_action).unsqueeze(-1) * sw.unsqueeze(-1) * choice_oh
            attn_bonus1   = ab1.unsqueeze(-1) * old_choice_oh
            attn_bonus2   = ab2.unsqueeze(-1) * opposite_oh

            # Log of trials-since-last-switch bonus for chosen action
            tsls_bonus    = choice_oh * torch.log1p(tsls).unsqueeze(-1)

            # JAX applies all these bonuses only when choice is present.
            has_choice_mask = had_choice.unsqueeze(-1).float()
            choice_logits = choice_logits + has_choice_mask * (
                persev_bonus + switch_bonus + attn_bonus1 + attn_bonus2 + tsls_bonus
            )

            logits[t] = choice_logits

            # ---- Store updated state ----
            new_old_ch = torch.where(had_choice, choice_idx, old_ch)
            self.state = {
                'q_values':                q,
                'old_choice':              new_old_ch,
                'trial_since_last_switch': tsls,
                'exploration_rate':        er,
                'cumchoice':               cc,
            }
        
        logits = logits.unsqueeze(1)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)  # → (B, T, n_actions)

        return logits, self.get_state()

    # ------------------------------------------------------------------
    # State management (mirrors BaseRNN / MarginalValueTheoremModel API)
    # ------------------------------------------------------------------

    def set_initial_state(self, batch_size: int = 1, prior=None, exploration_rate=None):
        """Reset state to initial values for a new session."""
        device = self.beta_r_raw.device
        if prior is None:
            prior = self.prior[:batch_size]
        if exploration_rate is None:
            exploration_rate = self.alpha_exploration_rate[:batch_size]

        self.state = {
            # uses B even though not in JAX, becasue JAX snippet is one-session notation, Python implementation is the batched equivalent.
            # Q-values initialised to `prior` (per-participant scalar broadcast to actions)
            'q_values':                prior.unsqueeze(-1).expand(batch_size, self.n_actions).clone(),
            # -1 signals "no previous choice" — use 0 as placeholder (clamped when indexing)
            'old_choice':              torch.full((batch_size,), -1, dtype=torch.long, device=device),
            'trial_since_last_switch': torch.zeros(batch_size, dtype=torch.float32, device=device),
            'exploration_rate':        exploration_rate.clone().detach(),
            'cumchoice':               torch.zeros(batch_size, self.n_actions, dtype=torch.float32, device=device),
        }
        return self.get_state()

    def set_state(self, state_dict):
        self.state = state_dict

    def get_state(self, detach=False):
        if detach:
            return {k: v.detach() for k, v in self.state.items()}
        return self.state
    
    
class EnvironmentEckstein2024(Env):
    """Drifting multi-armed bandit (Castro et al., 2025; Eckstein et al., 2026).

    Arm means follow independent Gaussian random walks:
        mu[t,i] ~ N(lambda * mu[t-1,i] + (1 - lambda) * center, sigma_drift)
        r[t,i]  ~ N(mu[t,i], sigma_obs)

    Default parameters from Ecsktein et al. (2026):
        lambda = 0.9836, sigma_drift = 2.8, sigma_obs = 4, center = 50 (0-100 scale)

    Rewards are returned normalized to [0, 1].
    """

    def __init__(
        self,
        n_participants: int,
        n_blocks: int = 5,
        n_actions: int = 4,
        lam: float = 0.9836,
        sigma_drift: float = 2.8,
        sigma_obs: float = 4.0,
        center: float = 50.0,
        scale: float = 100.0,
    ):
        super().__init__(n_actions=n_actions, n_participants=n_participants, n_blocks=n_blocks)
        self.lam = lam
        self.sigma_drift = sigma_drift
        self.sigma_obs = sigma_obs
        self.center = center
        self.scale = scale
        self.means = None
        self._rng = np.random.default_rng()

    def reset(self, seed: int = None, **kwargs):
        """Reset arm means to center. Optionally set RNG seed for reproducibility."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.means = np.full(self.n_actions, self.center)
        return {}

    def step(self, action: int):
        """Choose an arm, receive reward, and drift all arm means.

        Returns:
            reward: float in [0, 1] (clipped)
            terminated: always False (session length managed externally)
        """
        n_sessions = action.shape[0]
        reward_raw = self._rng.normal(self.means[action], self.sigma_obs)
        reward = np.clip(reward_raw / self.scale, 0.0, 1.0).astype(float)
        reward_cols = np.full((n_sessions, self.n_actions), float('nan'))
        reward_cols[np.arange(n_sessions), action] = reward
        
        self.means = self._rng.normal(
            self.lam * self.means + (1.0 - self.lam) * self.center,
            self.sigma_drift,
        )

        return torch.Tensor(reward_cols), False
    

def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    path_data: str = None,
    dataset: SpiceDataset = None,
    save_dataset: str = None,
    ) -> SpiceDataset:
    """Generate synthetic behavior for the Dezfouli 2019 two-armed bandit task.

    Args:
        model: Fitted model (SpiceEstimator or torch.nn.Module).
        path_data: Path to the original CSV data file.
        dataset: Pre-loaded SpiceDataset (used if provided, otherwise loaded from path_data).
        save_dataset: Optional path to save the generated dataset as CSV.

    Returns:
        SpiceDataset with model-generated behavior.
    """
    if dataset is None:
        dataset, _, _ = get_dataset(path_data=path_data)

    environment = EnvironmentEckstein2024(
        n_actions=dataset.n_actions,
        n_participants=dataset.n_participants,
        n_blocks=5,
    )
    
    dataset_gen = _generate_behavior(
        dataset=dataset,
        model=model,
        environment=environment,
        save_dataset=save_dataset,
    )

    return dataset_gen

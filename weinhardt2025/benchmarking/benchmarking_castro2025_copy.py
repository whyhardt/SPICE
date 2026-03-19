import argparse
import pickle
import torch
from tqdm import tqdm
from typing import List, Optional

try:
    from adabelief_pytorch import AdaBelief
    _ADABELIEF_AVAILABLE = True
except ImportError:
    _ADABELIEF_AVAILABLE = False

from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import csv_to_dataset, split_data_along_sessiondim
from spice.utils.agent import Agent


class Castro2025Model(torch.nn.Module):
    """
    Cognitive model from Castro et al. 2025 (discovered program) for multi-armed bandit task.
    ((Fidelity assumes your dataset encoding matches the model assumptions: action one-hot and reward one-hot in the expected feature positions.))
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

    Translated from the original JAX implementation. One parameter set per participant,
    matching the SPICE convention of per-participant SINDy coefficients.

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
            inputs = inputs.permute(1, 0, 2)  # → (T, B, F)

        T, B, _ = inputs.shape
        inputs = inputs.nan_to_num(0.0)

        # Participant IDs are stored in the last feature column (SPICE convention)
        participant_ids = inputs[0, :, -1].long()  # (B,)

        # Retrieve per-participant parameters indexed by batch participant IDs
        beta_r   = self.beta_r[participant_ids]                   # (B,)
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

            # Current choice index (-1 means no previous choice)
            choice_idx = actions_t.argmax(dim=-1)         # (B,)
            had_choice = actions_t.sum(dim=-1) > 0        # (B,) whether a valid choice was made

            # ---- Update state based on observed choice & reward ----
            # Scalar reward for the chosen action
            reward_scalar = (rewards_t * actions_t).sum(dim=-1)   # (B,)

            # Loss-averse TD error: r - γ*(1-r) - Q[choice]
            q_chosen = (q * actions_t).sum(dim=-1)                # (B,)
            delta = reward_scalar - gam * (1.0 - reward_scalar) - q_chosen  # (B,)

            # Update Q-value for chosen action only (vectorised via one-hot mask)
            q = q + delta.unsqueeze(-1) * actions_t               # (B, n_actions)

            # Increment counter if same action was repeated, else reset to 0
            same_action = (choice_idx == old_ch).float()
            tsls = torch.where(had_choice, same_action * (tsls + 1), tsls)

            # Slow exploration-rate decay
            # In JAX this only happens when a valid choice/reward update exists.
            er = torch.where(had_choice, er * (1.0 - 1e-3), er)

            # Cumulative choice count
            cc = cc + actions_t

            # ---- Global Q-value update (exploration smoothing + decay) ----
            q_mean = q.mean(dim=-1, keepdim=True)                 # (B, 1)
            q = (1.0 - er.unsqueeze(-1)) * q + er.unsqueeze(-1) * q_mean
            q = q * decay.unsqueeze(-1)

            # ---- Compute choice probabilities ----
            # Base logit: temperature-scaled Q-values + cumulative-choice bonus
            base_logits = beta_r.unsqueeze(-1) * q / temp.unsqueeze(-1) \
                          + beta_p.unsqueeze(-1) * torch.log1p(cc)    # (B, n_actions)

            probs = torch.softmax(base_logits, dim=-1)
            choice_probs = (1.0 - lapse.unsqueeze(-1)) * probs \
                           + lapse.unsqueeze(-1) / self.n_actions     # (B, n_actions)
            step_logits = torch.log(choice_probs + 1e-8)              # (B, n_actions)

            # ---- Add choice-conditioned bonuses (perseveration, switch, attention) ----
            # One-hot encodings for bonus computation
            choice_oh  = actions_t                                                     # (B, n_actions)
            # JAX one_hot(-1) -> all zeros; preserve that behavior explicitly.
            old_oh     = torch.nn.functional.one_hot(old_ch.clamp(min=0), self.n_actions).float()
            old_oh     = old_oh * (old_ch >= 0).unsqueeze(-1).float()
            opposite_oh = torch.nn.functional.one_hot((choice_idx + 2) % self.n_actions, self.n_actions).float()

            persev_bonus  = same_action.unsqueeze(-1) * perv.unsqueeze(-1) * choice_oh
            switch_bonus  = (1.0 - same_action).unsqueeze(-1) * sw.unsqueeze(-1) * choice_oh
            attn_bonus1   = ab1.unsqueeze(-1) * old_oh
            attn_bonus2   = ab2.unsqueeze(-1) * opposite_oh
            # Log of trials-since-last-switch bonus for chosen action
            tsls_bonus    = choice_oh * torch.log1p(tsls).unsqueeze(-1)

            # JAX applies all these bonuses only when choice is present.
            has_choice_mask = had_choice.unsqueeze(-1).float()
            step_logits = step_logits + has_choice_mask * (
                persev_bonus + switch_bonus + attn_bonus1 + attn_bonus2 + tsls_bonus
            )

            logits[t] = step_logits

            # ---- Store updated state ----
            new_old_ch = torch.where(had_choice, choice_idx, old_ch)
            self.state = {
                'q_values':                q,
                'old_choice':              new_old_ch,
                'trial_since_last_switch': tsls,
                'exploration_rate':        er,
                'cumchoice':               cc,
            }

        if self.batch_first:
            logits = logits.permute(1, 0, 2)  # → (B, T, n_actions)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_optimizer(model: Castro2025Model, lr: float) -> torch.optim.Optimizer:
    """Return AdaBelief if available, otherwise fall back to Adam."""
    if _ADABELIEF_AVAILABLE:
        return AdaBelief(model.parameters(), lr=lr, print_change_log=False)
    return torch.optim.Adam(model.parameters(), lr=lr)


def _compute_loss(model: Castro2025Model, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Single forward pass and cross-entropy loss (NaN-masked)."""
    logits, _ = model(xs)                                          # (T, B, n_actions)
    mask = ~torch.isnan(xs[:, :, :model.n_actions].sum(dim=-1))   # (T, B)
    logits_flat = logits[mask]                                     # (N, n_actions)
    labels_flat = ys[mask].argmax(dim=-1).long()                   # (N,)
    return torch.nn.functional.cross_entropy(logits_flat, labels_flat)


def training(
    model: Castro2025Model,
    xs: torch.Tensor,
    ys: torch.Tensor,
    xs_test: Optional[torch.Tensor] = None,
    ys_test: Optional[torch.Tensor] = None,
    epochs: int = 10000,
    lr: float = 5e-2,
    convergence_threshold: float = 1e-2,
    convergence_check_interval: int = 100,
    max_restarts: int = 10,
) -> Castro2025Model:
    """
    Fit the model using the Castro et al. 2025 training strategy:
      - AdaBelief optimiser (lr = 5e-2)
      - Check convergence every `convergence_check_interval` steps:
          |(Ω_k − Ω_{k−N}) / Ω_{k−N}| < convergence_threshold
      - Restart from fresh random parameters up to `max_restarts` times
      - Stop early once 3 restarts converge to the current best loss
    """
    best_model = None
    best_loss = float('inf')
    n_converged_to_best = 0

    for restart in range(max_restarts):
        # Re-initialise parameters for each restart
        candidate = Castro2025Model(
            n_actions=model.n_actions,
            n_participants=model.n_participants,
            batch_first=model.batch_first,
        )
        optimizer = _make_optimizer(candidate, lr)

        loss_history: List[float] = []
        converged = False

        pbar = tqdm(range(epochs), desc=f"Restart {restart + 1}/{max_restarts}", leave=False)
        for epoch in pbar:
            candidate.train()
            optimizer.zero_grad()
            loss = _compute_loss(candidate, xs, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=1.0)
            optimizer.step()

            loss_history.append(loss.item())

            # Convergence check (Castro et al. 2025 criterion)
            if (epoch + 1) % convergence_check_interval == 0 and len(loss_history) >= convergence_check_interval:
                omega_k = loss_history[-1]
                omega_prev = loss_history[-convergence_check_interval]
                if abs(omega_prev) > 1e-10:
                    rel_change = abs((omega_k - omega_prev) / omega_prev)
                    if rel_change < convergence_threshold:
                        converged = True

            # Live postfix
            postfix = {'train': f'{loss_history[-1]:.4f}', 'best': f'{best_loss:.4f}'}
            if xs_test is not None:
                candidate.eval()
                with torch.no_grad():
                    postfix['test'] = f'{_compute_loss(candidate, xs_test, ys_test).item():.4f}'
                candidate.train()
            pbar.set_postfix(postfix)

            if converged:
                break

        pbar.close()

        final_loss = loss_history[-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = candidate

        # Count restarts that converged close to the current best
        if converged and abs(final_loss - best_loss) / max(abs(best_loss), 1e-10) < 0.01:
            n_converged_to_best += 1
            if n_converged_to_best >= 3:
                # Three restarts agree — stop early
                break

        status = "converged" if converged else "max epochs"
        print(f"  Restart {restart + 1}: loss={final_loss:.5f} ({status}), best={best_loss:.5f}")

    return best_model


# ---------------------------------------------------------------------------
# Agent wrapper for SPICE evaluation pipeline
# ---------------------------------------------------------------------------

class AgentCastro2025(Agent):
    """Wraps a trained Castro2025Model as a SPICE-compatible Agent."""

    def __init__(self, model: Castro2025Model, deterministic: bool = True):
        super().__init__(model_rnn=None, n_actions=model.n_actions, deterministic=deterministic)
        assert isinstance(model, Castro2025Model)
        self.model = model
        self.model.eval()

    @property
    def q(self):
        """Current Q-values (numpy array)."""
        return self.state['q_values'].squeeze(0).detach().cpu().numpy()


def setup_agent_benchmark(path_model: str, deterministic: bool = True, **kwargs):
    """Load saved model and return a list of AgentCastro2025 instances."""
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    agent = AgentCastro2025(model=model, deterministic=deterministic)
    n_parameters = 13
    return [agent], n_parameters


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    path_save_model: str,
    path_data: str,
    n_actions: int,
    n_epochs: int,
    lr: float,
    split_ratio=None,
    convergence_threshold: float = 1e-2,
    convergence_check_interval: int = 100,
    max_restarts: int = 10,
):
    dataset = csv_to_dataset(file=path_data, df_participant_id='s_id', df_choice='action')

    if split_ratio is not None:
        dataset_train, dataset_test = split_data_along_sessiondim(dataset, list_test_sessions=split_ratio)
    else:
        dataset_train, dataset_test = dataset, None

    n_participants = int(dataset_train.xs[..., -1].max().item()) + 1
    xs = dataset_train.xs.permute(1, 0, 2)  # (T, B, F)
    ys = dataset_train.ys.permute(1, 0, 2)  # (T, B, n_actions)
    xs_test = dataset_test.xs.permute(1, 0, 2) if dataset_test is not None else None
    ys_test = dataset_test.ys.permute(1, 0, 2) if dataset_test is not None else None

    model = Castro2025Model(n_actions=n_actions, n_participants=n_participants)

    print('Training Castro2025 model...')
    model = training(
        model=model,
        xs=xs, ys=ys,
        xs_test=xs_test, ys_test=ys_test,
        epochs=n_epochs,
        lr=lr,
        convergence_threshold=convergence_threshold,
        convergence_check_interval=convergence_check_interval,
        max_restarts=max_restarts,
    )

    with open(path_save_model, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {path_save_model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Castro et al. 2025 benchmark model')
    parser.add_argument('--path_save_model', type=str, default='params/castro2025/castro2025.pkl')
    parser.add_argument('--path_data', type=str, default='data/eckstein2024/eckstein2024.csv')
    parser.add_argument('--n_actions', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--split_ratio', type=str, default=None, help='Comma-separated test session indices')
    parser.add_argument('--convergence_threshold', type=float, default=1e-2)
    parser.add_argument('--convergence_check_interval', type=int, default=100)
    parser.add_argument('--max_restarts', type=int, default=10)
    args = parser.parse_args()

    main(
        path_save_model=args.path_save_model,
        path_data=args.path_data,
        n_actions=args.n_actions,
        n_epochs=args.n_epochs,
        lr=args.lr,
        split_ratio=[int(x) for x in args.split_ratio.split(',')] if args.split_ratio else None,
        convergence_threshold=args.convergence_threshold,
        convergence_check_interval=args.convergence_check_interval,
        max_restarts=args.max_restarts,
    )

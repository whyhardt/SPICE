import math
import sys

import torch
import torch.nn as nn
from typing import Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

sys.path.append('../../..')
from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior
from weinhardt2026.utils.benchmarking_gru import training
from weinhardt2026.studies.ganesh2024a.spice_ganesh2024a import CONFIG


N_GRID = 100
KAPPA_MAX = 0.1


def get_dataset(path_data: str = None, test_blocks: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:

    if path_data is None:
        path_data = 'data/ganesh2024a_choice.csv'

    dataset = csv_to_dataset(
        file=path_data,
        df_participant_id='subjID',
        df_choice='choice',
        df_feedback='reward',
        df_block='blocks',
        additional_inputs=['contrast_difference'],
    )
    dataset.normalize_rewards()

    n_participants = len(dataset.xs[..., -1].unique())
    n_actions = dataset.ys.shape[-1]

    # Add next-trial contrast difference as additional input
    contr_diff = dataset.xs[..., n_actions * 2].unsqueeze(-1)
    contr_diff_next = contr_diff[:, 1:]
    xs = torch.cat((
        dataset.xs[:, :-1, :, :n_actions * 2],
        contr_diff[:, :-1, :],
        contr_diff_next,
        dataset.xs[:, :-1, :, 2 * n_actions + 1:],
    ), dim=-1)
    dataset = SpiceDataset(xs, dataset.ys[:, :-1])

    if verbose:
        print(f"Shape of dataset: {dataset.xs.shape}")
        print(f"Number of participants: {n_participants}")
        print(f"Number of actions: {n_actions}")

    if test_blocks is None:
        test_blocks = (3, 6, 9)
    if test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train = dataset_test = dataset

    info_dataset = {
        'n_participants': dataset.n_participants,
        'n_actions': dataset.n_actions,
        'n_additional_inputs': dataset.n_additional_inputs,
    }

    return dataset_train, dataset_test, info_dataset


class BayesianModel(nn.Module):
    """Normative Bayesian belief-update model (Agent from Ganesh et al., 2024).

    Maintains a discretized belief distribution over the contingency parameter
    mu in [0, 1], which links a latent perceptual state to reward probability.
    On each trial the model:
      1. Observes a noisy contrast difference between two Gabor patches.
      2. Computes a posterior belief over which stimulus has higher contrast.
      3. Uses the current belief about mu to compute expected action values.
      4. Updates the mu belief after observing the reward.

    Generative model:
        mu ~ P(mu)                           (prior: uniform; updated each trial)
        s ~ Bernoulli(mu)                    (latent state: which side is rewarded)
        o_t | s ~ N(contrast_diff, sigma^2)  (noisy perceptual observation)
        r | a, s, mu:
            P(r=1 | a matches s) = mu
            P(r=1 | a mismatches s) = 1 - mu

    Learnable parameters (per participant):
        beta:  Inverse softmax temperature  [1, 25]
        sigma: Perceptual noise parameter   [0.01, 0.1]

    State:
        belief_product: (B, N_GRID) unnormalized discrete posterior over mu.
    """

    def __init__(self, n_participants: int, n_actions: int = 2, batch_first: bool = False):
        super().__init__()
        self.n_participants = n_participants
        self.n_actions = n_actions
        self.batch_first = batch_first

        self.beta_raw = nn.Parameter(torch.zeros(n_participants))
        self.sigma_raw = nn.Parameter(torch.zeros(n_participants))

        self.register_buffer('p_mu', torch.linspace(0, 1, N_GRID))

    # ------------------------------------------------------------------
    # Parameter transforms
    # ------------------------------------------------------------------

    @property
    def beta(self):
        return torch.clamp(torch.nn.functional.softplus(self.beta_raw) + 1.0, 1.0, 25.0)

    @property
    def sigma(self):
        return torch.clamp(torch.sigmoid(self.sigma_raw) * 0.09 + 0.01, 0.01, 0.1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs, prev_state=None):
        """
        Args:
            inputs: (T, W, B, F) or (B, T, W, F) if batch_first.
                Features: [action_0, action_1, reward_0, reward_1,
                           contrast_diff_current, contrast_diff_next,
                           timestep, trial, block, experiment, participant]
            prev_state: dict with 'belief_product' or None.

        Returns:
            logits: (T, 1, B, n_actions) or (B, T, 1, n_actions) if batch_first.
            state:  dict with updated belief_product.
        """
        if self.batch_first:
            inputs = inputs.permute(1, 2, 0, 3)  # (B,T,W,F) -> (T,W,B,F)
        inputs = inputs[:, 0]  # squeeze W -> (T, B, F)
        T, B, _ = inputs.shape
        inputs = inputs.nan_to_num(0.0)

        participant_ids = inputs[0, :, -1].long()
        beta = self.beta[participant_ids]    # (B,)
        sigma = self.sigma[participant_ids]  # (B,)

        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(B)

        belief = self.state['belief_product']  # (B, N_GRID)
        logits = torch.zeros(T, B, self.n_actions, device=inputs.device)

        for t in range(T):
            actions_t = inputs[t, :, :self.n_actions]                    # (B, 2)
            rewards_t = inputs[t, :, self.n_actions:2 * self.n_actions]  # (B, 2)
            cd_current = inputs[t, :, 2 * self.n_actions]                # (B,)
            cd_next = inputs[t, :, 2 * self.n_actions + 1]               # (B,)

            had_choice = actions_t.sum(dim=-1) > 0
            action_idx = actions_t.argmax(dim=-1)
            reward_scalar = (rewards_t * actions_t).sum(dim=-1)

            # ---- Bayesian belief update from current trial ----
            pi_0, pi_1 = self._state_beliefs(cd_current, sigma)

            # Recode reward: flip when action=1 so it reflects state-0-action-0 contingency
            r = torch.where(action_idx == 0, reward_scalar, 1.0 - reward_scalar)

            # Bayesian update factors (numerical path from Agent.m)
            q_0 = pi_1 * r + pi_0 * (1.0 - r)
            q_1 = (2.0 * r - 1.0) * (pi_0 - pi_1)

            # Multiplicative posterior update
            update = q_1.unsqueeze(-1) * self.p_mu + q_0.unsqueeze(-1)  # (B, N_GRID)
            update = torch.where(had_choice.unsqueeze(-1), update, torch.ones_like(update))
            belief = (belief * update).clamp(min=1e-30)

            # Expected mu from normalized belief
            belief_norm = belief / belief.sum(dim=-1, keepdim=True)
            E_mu = (belief_norm * self.p_mu).sum(dim=-1)  # (B,)

            # ---- Action values for next choice (based on next contrast) ----
            pi_0n, pi_1n = self._state_beliefs(cd_next, sigma)

            v_a0 = (pi_0n - pi_1n) * E_mu + pi_1n
            v_a1 = (pi_1n - pi_0n) * E_mu + pi_0n

            logits[t] = beta.unsqueeze(-1) * torch.stack([v_a0, v_a1], dim=-1)

        self.state['belief_product'] = belief

        logits = logits.unsqueeze(1)  # (T, B, 2) -> (T, 1, B, 2)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)  # -> (B, T, 1, 2)

        return logits, self.get_state()

    # ------------------------------------------------------------------
    # Perceptual belief computation
    # ------------------------------------------------------------------

    def _state_beliefs(self, contrast_diff: torch.Tensor, sigma: torch.Tensor):
        """Compute P(state=0|o_t) and P(state=1|o_t) via truncated normal model.

        state=0: left patch has lower contrast  (contrast_diff <= 0)
        state=1: left patch has higher contrast (contrast_diff > 0)
        """
        u = self._normal_cdf(0.0, contrast_diff, sigma)
        v = self._normal_cdf(-KAPPA_MAX, contrast_diff, sigma)
        w = self._normal_cdf(KAPPA_MAX, contrast_diff, sigma)

        denom = (w - v).clamp(min=1e-10)
        pi_0 = ((u - v) / denom).clamp(0.0, 1.0)
        pi_1 = ((w - u) / denom).clamp(0.0, 1.0)
        return pi_0, pi_1

    @staticmethod
    def _normal_cdf(x, mean, std):
        """CDF of N(mean, std^2) evaluated at x."""
        return 0.5 * (1.0 + torch.erf((x - mean) / (std * math.sqrt(2))))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_initial_state(self, batch_size: int = 1):
        device = self.p_mu.device
        self.state = {
            'belief_product': torch.ones(batch_size, N_GRID, device=device),
        }
        return self.get_state()

    def set_state(self, state_dict):
        self.state = state_dict

    def get_state(self, detach=False):
        if detach:
            return {k: v.detach() for k, v in self.state.items()}
        return self.state

    def count_parameters(self):
        return 2  # beta, sigma per participant


class EnvironmentGanesh2024a(Env):
    """Two-armed bandit with perceptual contrast stimuli (Ganesh et al., 2024).

    On each trial the participant sees two Gabor patches with different contrasts
    and chooses one. A latent state determined by the sign of the contrast
    difference dictates which action is correct:

        state = 1  if contrast_diff > 0  (left patch has higher contrast)
        state = 0  otherwise

    Rewards are binary Bernoulli samples:
        P(r=1 | action matches state) = mu
        P(r=1 | action mismatches)    = 1 - mu

    Contrast differences are replayed from the original dataset (they are task
    stimuli, not model-dependent).

    Sessions are processed in parallel: one session per (participant, block) pair.
    """

    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        n_blocks: int,
        contrast_current: torch.Tensor,
        contrast_next: torch.Tensor,
        mu: float = 0.75,
    ):
        """
        Args:
            n_actions: Number of actions (2).
            n_participants: Number of participants.
            n_blocks: Number of blocks per participant.
            contrast_current: (n_sessions, n_trials) contrast difference on current trial.
            contrast_next: (n_sessions, n_trials) contrast difference on next trial.
            mu: True contingency parameter (reward probability for correct action).
        """
        super().__init__(n_actions, n_participants, n_blocks)
        self.mu = mu
        self.contrast_current = contrast_current
        self.contrast_next = contrast_next
        self.trial_counter = 0

    def reset(self, block_ids: torch.Tensor = None, participant_ids: torch.Tensor = None) -> None:
        self.trial_counter = 0

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample reward for the chosen arm based on contrast-state contingency.

        Args:
            action: (n_sessions,) integer action index (0 or 1) per session.

        Returns:
            observation: (n_sessions, 4) — [reward_0, reward_1, cd_current, cd_next].
            terminated: (n_sessions,) always False.
        """
        n_sessions = action.shape[0]
        t = self.trial_counter

        cd_current = self.contrast_current[:n_sessions, t]
        cd_next = self.contrast_next[:n_sessions, t]

        # True state: which side has higher contrast
        state = (cd_current > 0).long()

        # Reward probability depends on whether action matches state
        matches = (action == state).float()
        probs = matches * self.mu + (1.0 - matches) * (1.0 - self.mu)
        reward = torch.bernoulli(probs)

        # Partial feedback: only reveal reward for chosen action
        reward_cols = torch.full((n_sessions, self.n_actions), float('nan'))
        reward_cols[torch.arange(n_sessions), action] = reward

        observation = torch.cat([
            reward_cols,
            cd_current.unsqueeze(-1),
            cd_next.unsqueeze(-1),
        ], dim=-1)

        terminated = torch.zeros(n_sessions, dtype=torch.bool)
        self.trial_counter += 1

        return observation, terminated

def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    path_data: str = None,
    dataset: SpiceDataset = None,
    save_dataset: str = None,
    mu: float = 0.75,
) -> SpiceDataset:
    """Generate synthetic behavior for the Ganesh 2024a contrast-bandit task.

    Args:
        model: Fitted model (SpiceEstimator or torch.nn.Module).
        path_data: Path to the original CSV data file.
        dataset: Pre-loaded SpiceDataset (used if provided, otherwise loaded from path_data).
        save_dataset: Optional path to save the generated dataset as CSV.
        mu: True contingency parameter for the environment.

    Returns:
        SpiceDataset with model-generated behavior.
    """
    if dataset is None:
        dataset, _, _ = get_dataset(path_data=path_data, test_blocks=())

    n_actions = dataset.n_actions
    n_blocks = len(dataset.xs[:, 0, 0, -3].unique())

    # Replay contrast sequences from original dataset
    contrast_current = dataset.xs[:, :, 0, 2 * n_actions].clone().nan_to_num(0.0)
    contrast_next = dataset.xs[:, :, 0, 2 * n_actions + 1].clone().nan_to_num(0.0)

    environment = EnvironmentGanesh2024a(
        n_actions=n_actions,
        n_participants=dataset.n_participants,
        n_blocks=n_blocks,
        contrast_current=contrast_current,
        contrast_next=contrast_next,
        mu=mu,
    )

    return _generate_behavior(
        dataset=dataset,
        model=model,
        environment=environment,
        save_dataset=save_dataset,
        kwargs_dataset=dict(
            df_participant_id='subjID',
            df_choice='choice',
            df_feedback='reward',
            df_block='blocks',
            additional_inputs=CONFIG.additional_inputs,
        ),
    )


def fit(
    path_data: str = None,
    test_blocks: tuple[int] = None,
    epochs: int = 3000,
    lr: float = 1e-2,
    device: torch.device = None,
) -> tuple[BayesianModel, SpiceDataset, SpiceDataset]:
    """Fit the BayesianModel to Ganesh 2024a data via MLE.

    Uses cross-entropy loss to optimize per-participant beta and sigma
    parameters through the shared training utility.

    Args:
        path_data: Path to CSV data file (default: data/ganesh2024a_choice.csv).
        test_blocks: Session indices for test split (default: (3, 6, 9)).
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        device: Compute device (default: auto-detect).

    Returns:
        (fitted_model, dataset_train, dataset_test)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train, dataset_test, info = get_dataset(
        path_data=path_data, test_blocks=test_blocks, verbose=True,
    )

    model = BayesianModel(
        n_participants=info['n_participants'],
        n_actions=info['n_actions'],
        batch_first=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = training(
        model=model,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=epochs,
        device=device,
    )

    return model, dataset_train, dataset_test

import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

from weinhardt2026.studies.weber2024.spice_weber2024 import CONFIG
from weinhardt2026.studies.weber2024.archive.benchmarking_weber2024 import angular_distance, move_toward


# --- Constants ---

MOVEMENT_SPEED = 1.0        # degrees per frame
CATCH_THRESHOLD = 10.0      # degrees — shield catches laser if angular distance <= this
INITIAL_SHIELD = 360.0      # degrees (shield starting position each block)


# =============================================================================
# Bayesian changepoint model (without variance inference)
# =============================================================================

class ChangePointModel(torch.nn.Module):
    """Bayesian changepoint model for circular tracking (Weber et al., 2024).

    Performs Bayesian inference on a discretized circular state space [0, 360)
    assuming changepoint dynamics and a fixed observation noise level.
    This corresponds to the "change-point model without variance inference"
    from the paper.

    The model maintains a posterior distribution over the hidden generative
    mean on a discretized grid. At each trial it:
      1. Observes the laser position.
      2. Updates the posterior via Bayes' rule (circular normal likelihood).
      3. Propagates the posterior through the changepoint transition matrix.
      4. Outputs the circular posterior mean as the belief (sin, cos).

    Learnable per-participant parameters:
        p_cp:              Probability of a changepoint at each trial.
        stochasticity_sd:  Standard deviation (degrees) of the circular normal
                           observation distribution.

    Fixed hyperparameters:
        h_step:            Discretisation step for the state space (degrees).
        allowed_changes:   Tuple of allowed absolute changepoint magnitudes
                           (degrees).  Sign-symmetric: ±20 means both +20 and -20.
    """

    def __init__(
        self,
        n_participants: int = 1,
        h_step: int = 2,
        allowed_changes: tuple[float, ...] = (20, 30, 40),
        batch_first: bool = True,
    ):
        super().__init__()
        self.n_participants = n_participants
        self.h_step = h_step
        self.allowed_changes = allowed_changes
        self.batch_first = batch_first
        self.n_actions = 2  # output: (sin, cos)

        # --- Learnable per-participant parameters (raw; transformed via properties) ---
        # Initialise at cognitively plausible values:
        #   p_cp ≈ 0.06  →  sigmoid(x) = 0.06  →  x = log(0.06/0.94) ≈ -2.75
        #   sd   ≈ 15°   →  softplus(x) = 15   →  x ≈ 15  (softplus ≈ identity for large x)
        self.p_cp_raw = torch.nn.Parameter(torch.full((n_participants,), -2.75))
        self.stochasticity_sd_raw = torch.nn.Parameter(torch.full((n_participants,), 15.0))

        # Pre-compute fixed structures as registered buffers
        self._build_fixed_matrices()

    # --- Parameter transforms ------------------------------------------------

    @property
    def p_cp(self):
        """Changepoint probability — in (0, 0.5)."""
        return torch.sigmoid(self.p_cp_raw).clamp(1e-3, 0.5)

    @property
    def stochasticity_sd(self):
        """Observation noise s.d. in degrees — positive."""
        return F.softplus(self.stochasticity_sd_raw).clamp(1.0, 60.0)

    # --------------------------------------------------------------------- #
    #  Fixed inference structures
    # --------------------------------------------------------------------- #

    def _build_fixed_matrices(self):
        """Pre-compute the changepoint redistribution matrix and state grid.

        The transition matrix is decomposed as:
            T(p) = (1 - p) * I  +  p * cp_mat
        where ``cp_mat`` is fixed (depends only on allowed_changes and h_step)
        and ``p`` (p_cp) is a learnable parameter.  This decomposition lets us
        avoid materialising the full per-participant transition matrix.
        """
        state_values = np.arange(0, 360, self.h_step, dtype=np.float64)
        n_states = len(state_values)

        # Changepoint redistribution matrix: cp_mat[i, j] = probability of
        # jumping to state i from state j, conditional on a changepoint.
        cp_mat = np.zeros((n_states, n_states))
        for j in range(n_states):
            targets = []
            for i in range(n_states):
                diff = ((state_values[i] - state_values[j]) + 180) % 360 - 180
                if any(abs(abs(diff) - c) < self.h_step / 2 for c in self.allowed_changes):
                    targets.append(i)
            if targets:
                for i in targets:
                    cp_mat[i, j] = 1.0 / len(targets)
            else:
                cp_mat[:, j] = 1.0 / n_states

        # Register as buffers so they auto-move with .to(device)
        self.register_buffer('_cp_mat', torch.tensor(cp_mat, dtype=torch.float32))
        self.register_buffer('_state_values_deg', torch.tensor(state_values, dtype=torch.float32))
        state_rad = torch.deg2rad(torch.tensor(state_values, dtype=torch.float32))
        self.register_buffer('_state_sin', torch.sin(state_rad))
        self.register_buffer('_state_cos', torch.cos(state_rad))
        self._n_states = n_states

    # --------------------------------------------------------------------- #
    #  Forward pass (differentiable)
    # --------------------------------------------------------------------- #

    def forward(
        self,
        inputs: torch.Tensor,
        prev_state: dict = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run Bayesian changepoint inference on laser observations.

        Fully differentiable w.r.t. ``p_cp`` and ``stochasticity_sd``.

        Args:
            inputs: (B, T, W, F) if batch_first else (T, W, B, F).
            prev_state: Dict with key ``'priors'`` — tensor (B, n_states).
                        If None, starts from uniform prior.

        Returns:
            predictions: (B, T, W, 2) predicted belief position (sin, cos).
            state: Dict with updated ``'priors'`` (detached).
        """
        if self.batch_first:
            B, T, W, F = inputs.shape
        else:
            inputs = inputs.permute(2, 0, 1, 3)
            B, T, W, F = inputs.shape

        device = inputs.device
        n_states = self._n_states

        # Per-session participant parameters
        participant_ids = inputs[:, 0, 0, -1].long()            # (B,)
        p_cp = self.p_cp[participant_ids]                        # (B,)
        sd = self.stochasticity_sd[participant_ids]              # (B,)

        # Initialise prior distribution
        if prev_state is not None and 'priors' in prev_state:
            prior = prev_state['priors'].to(device)              # (B, S)
        else:
            prior = torch.full((B, n_states), 1.0 / n_states,
                               device=device)

        # Validity mask (NaN-padded trials)
        valid = ~torch.isnan(inputs[:, :, 0, 0])                # (B, T)

        # Laser observations → degrees
        laser_sin = inputs[:, :, 0, 2]                           # (B, T)
        laser_cos = inputs[:, :, 0, 3]                           # (B, T)
        laser_deg = torch.atan2(laser_sin, laser_cos).rad2deg() % 360  # (B, T)

        predictions = torch.full((B, T, 2), float('nan'), device=device)

        for t in range(T):
            mask = valid[:, t]                                   # (B,)
            if not mask.any():
                continue

            # --- Circular normal log-likelihood (B, S) ---
            # angle_diff in [-180, 180)
            obs = laser_deg[:, t].unsqueeze(1)                   # (B, 1)
            angle_diff = (obs - self._state_values_deg.unsqueeze(0) + 180) % 360 - 180
            log_lik = -0.5 * (angle_diff / sd.unsqueeze(1)) ** 2  # drop constant

            # --- Posterior (log-space for stability) ---
            log_prior = torch.log(prior.clamp(min=1e-30))
            log_post = log_lik + log_prior
            log_post = log_post - log_post.max(dim=1, keepdim=True).values
            posterior = torch.exp(log_post)
            posterior = posterior / posterior.sum(dim=1, keepdim=True)

            # --- Circular mean of posterior ---
            mean_sin = (self._state_sin.unsqueeze(0) * posterior).sum(dim=1)
            mean_cos = (self._state_cos.unsqueeze(0) * posterior).sum(dim=1)
            belief_rad = torch.atan2(mean_sin, mean_cos)
            belief = torch.stack([torch.sin(belief_rad), torch.cos(belief_rad)], dim=-1)

            # Apply physical movement clamping: the shield can only move
            # speed * dt degrees per trial, so clamp the belief to the
            # reachable region around the current shield position.
            shield_t = inputs[:, t, 0, :2]                        # current shield (sin, cos)
            dt = inputs[:, t, 0, _AI_START + _AI_DT]              # trial_duration_frames
            delta = belief - shield_t
            delta_mag = delta.norm(dim=-1, keepdim=True)
            max_move = MOVEMENT_SPEED * dt.unsqueeze(1) * (math.pi / 180.0)
            fraction = torch.clamp_max(max_move / (delta_mag + 1e-8), 1.0)
            pred = shield_t + fraction * delta

            predictions[:, t] = torch.where(mask.unsqueeze(1), pred, predictions[:, t])

            # --- Dynamics propagation ---
            # T @ posterior = (1-p)*posterior + p*(cp_mat @ posterior)
            cp_posterior = torch.matmul(posterior, self._cp_mat.T)  # (B, S)
            new_prior = ((1 - p_cp.unsqueeze(1)) * posterior
                         + p_cp.unsqueeze(1) * cp_posterior)
            prior = torch.where(mask.unsqueeze(1), new_prior, prior)

        predictions = predictions.unsqueeze(2)                   # (B, T, 1, 2)

        if not self.batch_first:
            predictions = predictions.permute(1, 2, 0, 3)

        return predictions, {'priors': prior.detach()}

    def count_parameters(self) -> int:
        """Free parameters per participant: p_cp + stochasticity_sd."""
        return 2 * self.n_participants


# --- Loss function with physical speed clamping ---

def clamped_angular_mse(prediction: torch.Tensor, target: torch.Tensor, speed: float = MOVEMENT_SPEED, **kwargs) -> torch.Tensor:
    """MSE loss with physical movement speed clamping.

    The model predicts a raw belief position in (sin, cos) space.
    This loss clamps the predicted movement to be physically feasible
    given the inter-beam interval (dt) and movement speed.

    Args:
        prediction: (N, 2) model output [belief_sin, belief_cos]
        target: (N, 5) packed target [sin(shield_{t+1}), cos(shield_{t+1}),
                sin(shield_t), cos(shield_t), dt]
        speed: movement speed in degrees per frame
    """
    belief = prediction[..., :2]                    # model's belief
    actual = target[..., :2]                        # actual shield at t+1
    shield_t = target[..., 2:4]                     # shield at t (sin, cos)
    dt = target[..., 4:5]                           # inter-beam interval (frames)

    delta = belief - shield_t
    delta_mag = delta.norm(dim=-1, keepdim=True)

    # Max angular movement in sin/cos space ≈ speed * dt * (pi/180) for small angles
    # For large angles, the sin/cos delta saturates, but this is a reasonable approximation
    max_move = speed * dt * (math.pi / 180.0)
    fraction = torch.clamp_max(max_move / (delta_mag + 1e-8), 1.0)

    clamped_pred = shield_t + fraction * delta
    return F.mse_loss(clamped_pred, actual)


# --- Data Loading ---

def get_dataset(
    path_data: str = None,
    test_blocks: tuple[int] = None,
) -> tuple[SpiceDataset, SpiceDataset]:
    """Load weber2024 data for continuous shield position prediction.

    Converts the discrete stay/move dataset into a continuous prediction task
    where the model predicts the shield position (in sin/cos space) at the
    next laser beam event.

    Feature layout of xs:
        [0:2]  sin(shield_t), cos(shield_t)  — "action" (n_actions=2)
        [2:4]  sin(laser_t), cos(laser_t)    — "reward" (n_reward_features=2)
        [4]    laser_caught                   — additional input 0
        [5]    volatility                     — additional input 1
        [6]    stochasticity                  — additional input 2
        [7]    trial_duration_frames          — additional input 3
        [-5:]  time_trial, trial, block, experiment, participant  — metadata

    Target ys:
        [0:2]  sin(shield_{t+1}), cos(shield_{t+1})  — target position
        [2:4]  sin(shield_t), cos(shield_t)           — current position (for loss clamping)
        [4]    dt (trial_duration_frames)              — inter-beam interval (for loss clamping)
    """
    if path_data is None:
        path_data = 'weinhardt2026/studies/weber2024/data/weber2024.csv'

    df = pd.read_csv(path_data)

    # Convert angular positions to sin/cos (degrees → radians → trig)
    shield_rad = df['shieldRotation'] * (math.pi / 180.0)
    laser_rad = df['laserRotation'] * (math.pi / 180.0)

    df['shield_sin'] = shield_rad.apply(math.sin)
    df['shield_cos'] = shield_rad.apply(math.cos)
    df['laser_sin'] = laser_rad.apply(math.sin)
    df['laser_cos'] = laser_rad.apply(math.cos)

    # Build the continuous dataset using csv_to_dataset
    # Action = shield position (sin, cos) — what the model observes as its own position
    # Reward = laser position (sin, cos) — the outcome / prediction error source
    dataset = csv_to_dataset(
        file=df,
        df_participant_id='participant',
        df_experiment_id='experiment',
        df_choice=['shield_sin', 'shield_cos'],
        df_feedback=['laser_sin', 'laser_cos'],
        df_block='block',
        additional_inputs=CONFIG.additional_inputs,
        continuous_action=True,
    )

    # Expand ys: append shield_t and dt for loss function clamping
    # csv_to_dataset produces ys = [next_shield_sin, next_shield_cos] (2 cols)
    # We need ys = [next_shield_sin, next_shield_cos, shield_sin_t, shield_cos_t, dt] (5 cols)
    n_actions = 2
    n_rewards = 2
    dt_col = n_actions + n_rewards + 3  # column index 7: trial_duration_frames

    shield_t = dataset.xs[:, :, :, :n_actions].clone()
    dt = dataset.xs[:, :, :, dt_col:dt_col + 1].clone()
    new_ys = torch.cat([dataset.ys, shield_t, dt], dim=-1)

    dataset = SpiceDataset(dataset.xs, new_ys, n_reward_features=n_rewards, continuous_action=True)

    if test_blocks is not None:
        return split_data_along_blockdim(dataset, test_blocks)
    return dataset, None


# --- Column index helpers ---

_N_ACTIONS = 2      # shield_sin, shield_cos
_N_REWARDS = 2      # laser_sin, laser_cos
_AI_START = _N_ACTIONS + _N_REWARDS  # column 4

_AI_CAUGHT = 0      # laser_caught offset from _AI_START
_AI_DT = 3          # trial_duration_frames offset from _AI_START


# --- Behavior Generation ---

@torch.no_grad()
def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    dataset: SpiceDataset,
    save_dataset: str = None,
) -> SpiceDataset:
    """Generate synthetic behavioral data for the continuous weber2024 task.

    Replays the original laser trajectories. At each trial the model predicts
    a belief position (sin, cos) and the shield moves toward that belief,
    clamped by physical movement speed * trial duration.

    Args:
        model: Fitted SPICE model (SpiceEstimator or nn.Module).
        dataset: Original dataset (structure + laser trajectories + durations).
        save_dataset: Optional path to save generated dataset as CSV.

    Returns:
        SpiceDataset with generated shield behavior.
    """
    print("Generating behavior (continuous)...")

    n_sessions, n_trials = dataset.xs.shape[0], dataset.xs.shape[1]

    # Unwrap model
    if isinstance(model, SpiceEstimator):
        inner_model = model.model
    else:
        inner_model = model
    inner_model.eval()

    if hasattr(inner_model, 'device'):
        device = inner_model.device
    elif hasattr(inner_model, 'parameters'):
        device = next(inner_model.parameters()).device
    else:
        device = torch.device('cpu')

    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])  # (n_sessions, n_trials)
    xs_gen = dataset.xs.clone().to(device)
    ys_gen = torch.full_like(dataset.ys, float('nan'), device=device)

    # Initialize shield at 360° (≡ 0° mod 360) for every session
    shield_deg = torch.full((n_sessions,), INITIAL_SHIELD, device=device)

    model_state = None

    for t in tqdm(range(n_trials)):
        active = valid_mask[:, t].to(device)

        # Current shield → sin/cos
        shield_rad = shield_deg * (math.pi / 180.0)
        shield_sin_t = torch.sin(shield_rad)
        shield_cos_t = torch.cos(shield_rad)

        # Laser position (from original data, already in sin/cos)
        laser_sin_t = dataset.xs[:, t, 0, _N_ACTIONS].to(device)
        laser_cos_t = dataset.xs[:, t, 0, _N_ACTIONS + 1].to(device)
        laser_deg = torch.atan2(laser_sin_t, laser_cos_t) * (180.0 / math.pi) % 360

        # Compute catch
        caught = (angular_distance(shield_deg, laser_deg) <= CATCH_THRESHOLD).float()

        # Write generated shield + catch into xs (laser, vol, stoch, dt, metadata kept)
        xs_gen[:, t, 0, 0] = shield_sin_t
        xs_gen[:, t, 0, 1] = shield_cos_t
        xs_gen[:, t, 0, _AI_START + _AI_CAUGHT] = caught

        # Forward pass (single trial)
        xs_t = xs_gen[:, t:t + 1].clone()
        logits, model_state = inner_model(xs_t, model_state)

        # Normalise logits → (n_sessions, 2)
        if logits.dim() == 5:                       # BaseModel: (E, B, T, W, A)
            logits = logits.mean(dim=0)
        if logits.dim() == 4:                       # (B, T, W, A)
            belief = logits[:, -1, 0, :2]
        else:                                       # (B, T, A)
            belief = logits[:, -1, :2]

        # Convert belief to degrees
        belief_deg = torch.atan2(belief[:, 0], belief[:, 1]) * (180.0 / math.pi) % 360

        # Move shield toward belief, clamped by speed × dt
        dt = dataset.xs[:, t, 0, _AI_START + _AI_DT].to(device)
        max_move = MOVEMENT_SPEED * dt
        new_shield = move_toward(shield_deg, belief_deg, max_move)

        # NaN-out inactive sessions, advance shield for active ones
        xs_gen[~active, t] = float('nan')
        shield_deg = torch.where(active, new_shield, shield_deg)

    # Build ys: [next_shield_sin, next_shield_cos, shield_sin_t, shield_cos_t, dt]
    for t in range(n_trials - 1):
        both = (valid_mask[:, t] & valid_mask[:, t + 1]).to(device)
        ys_gen[both, t, 0, 0] = xs_gen[both, t + 1, 0, 0]                     # next sin
        ys_gen[both, t, 0, 1] = xs_gen[both, t + 1, 0, 1]                     # next cos
        ys_gen[both, t, 0, 2] = xs_gen[both, t, 0, 0]                         # sin_t
        ys_gen[both, t, 0, 3] = xs_gen[both, t, 0, 1]                         # cos_t
        ys_gen[both, t, 0, 4] = xs_gen[both, t, 0, _AI_START + _AI_DT]        # dt

    xs_gen = xs_gen.cpu()
    ys_gen = ys_gen.cpu()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=_N_REWARDS, continuous_action=True)

    if save_dataset is not None:
        from spice import dataset_to_csv
        dataset_to_csv(
            dataset_gen, save_dataset,
            df_choice=['shield_sin', 'shield_cos'],
            df_feedback='laser',
            additional_inputs=list(CONFIG.additional_inputs),
        )

    print("Done generating behavior.")
    return dataset_gen

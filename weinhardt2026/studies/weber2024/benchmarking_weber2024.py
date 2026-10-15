import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

from weinhardt2026.studies.weber2024.spice_weber2024 import (
    CONFIG, MOVEMENT_SPEED, clamped_angular_mse, prepare_dataframe, prepare_dataset,
)


# --- Constants ---

def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest circular distance in degrees between two angles.

    Handles values outside [0, 360) via modulo.
    """
    diff = (a - b) % 360
    return torch.min(diff, 360 - diff)


def move_toward(shield: torch.Tensor, laser: torch.Tensor, max_degrees: torch.Tensor) -> torch.Tensor:
    """Move shield toward laser by up to max_degrees along shortest arc.

    Returns new shield position (may be outside [0, 360)).
    """
    diff = (laser - shield % 360) % 360
    direction = torch.where(diff <= 180, torch.ones_like(diff), -torch.ones_like(diff))
    dist = angular_distance(shield, laser)
    movement = torch.min(max_degrees, dist)
    return shield + direction * movement
CATCH_THRESHOLD = 10.0      # degrees — shield catches laser if angular distance <= this
INITIAL_SHIELD = 360.0      # degrees (shield starting position each block)


# =============================================================================
# Bayesian changepoint model (Foucault et al., 2026, eLife 15:RP110137)
# =============================================================================

class ChangePointModel(torch.nn.Module):
    """Bayesian changepoint observer with expected-reward response (Foucault et al., 2026).

    Every beam runs the paper's three computations on a discretized circular grid:
      1. Inference:  posterior over the latent state given the beam (Bayesian filtering).
      2. Prediction: propagate the posterior through the changepoint transition and
                     convolve it with the observation noise -> predictive distribution
                     of the next beam.
      3. Response:   select the shield location that maximizes the expected catch
                     probability, P(|next beam - location| <= CATCH_THRESHOLD), and move
                     there with response probability HalfNormalCDF(distance; response_sd),
                     the distance between the current and the selected location.

    Adaptations to the continuous weber2024 task:
      - One update per beam: the response is the shield position at the next beam, the
        selected location clamped to the reachable arc (MOVEMENT_SPEED * trial duration).
      - The output is the expected next shield position in (sin, cos), mixing staying and
        moving by the response probability (MSE fit). With ``sample_response`` the move is
        drawn instead (generative simulation).
      - No knowledge of the task's generative process: as in the paper, a changepoint
        draws the new mean uniformly from the circle (the task's actual jump sizes are
        unknown to participants), and the hazard rate and noise levels are fitted.
      - With variance inference the latent state is (mean, observation-noise level) over
        two fitted noise levels; the level is constant within a block, so it does not
        switch. Without, a single observation-noise s.d. is fitted.
      - The paper's fixed hazard rate and response s.d. are fitted per participant.
      - The argmax over locations is a soft argmax (``choice_sharpness``) to keep the
        selected location differentiable w.r.t. the parameters.

    Learnable per-participant parameters:
        p_cp:         Assumed changepoint probability per beam.
        response_sd:  S.d. (degrees) of the response-probability CDF (paper: fixed 8°).
        noise_sd:     Observation-noise s.d. (degrees) per noise level: two levels with
                      variance inference, one without.
    """

    def __init__(
        self,
        n_participants: int = 1,
        h_step: int = 2,
        variance_inference: bool = True,
        choice_sharpness: float = 50.,
        batch_first: bool = True,
    ):
        super().__init__()
        self.n_participants = n_participants
        self.h_step = h_step
        self.variance_inference = variance_inference
        self.choice_sharpness = choice_sharpness
        self.batch_first = batch_first
        self.n_actions = 2  # output: (sin, cos)
        self.sample_response = False

        # --- Learnable per-participant parameters (raw; transformed via properties) ---
        # s.d.s in log space: the fitted response s.d. lies an order of magnitude above the paper's 8°
        #   p_cp = 1/16 (paper)          ->  logit(1/16) ≈ -2.7
        #   response_sd = 8° (paper)     ->  log(8)
        #   noise_sd: generic starting values, two distinct levels with variance inference
        noise_sd_init = (10., 30.) if variance_inference else (15.,)
        self.p_cp_raw = torch.nn.Parameter(torch.full((n_participants,), math.log(1 / 15)))
        self.response_sd_raw = torch.nn.Parameter(torch.full((n_participants,), math.log(8.)))
        self.noise_sd_raw = torch.nn.Parameter(torch.log(torch.tensor(noise_sd_init)).repeat(n_participants, 1))

        self._build_fixed_matrices()

    # --- Parameter transforms ------------------------------------------------

    @property
    def p_cp(self):
        """Changepoint probability — in (0, 1)."""
        return torch.sigmoid(self.p_cp_raw)

    @property
    def response_sd(self):
        """Response-probability s.d. in degrees — positive."""
        return torch.exp(self.response_sd_raw)

    @property
    def noise_sd(self):
        """Observation-noise s.d. per level in degrees, (n_participants, n_levels) — positive."""
        return torch.exp(self.noise_sd_raw)

    # --------------------------------------------------------------------- #
    #  Fixed inference structures
    # --------------------------------------------------------------------- #

    def _build_fixed_matrices(self):
        """State grid and the wrapped offsets of the catch kernel."""
        state_values = np.arange(0, 360, self.h_step, dtype=np.float64)
        n_states = len(state_values)

        self.register_buffer('_state_values_deg', torch.tensor(state_values, dtype=torch.float32))
        state_rad = torch.deg2rad(torch.tensor(state_values, dtype=torch.float32))
        self.register_buffer('_state_sin', torch.sin(state_rad))
        self.register_buffer('_state_cos', torch.cos(state_rad))
        # signed grid offsets wrapped to [-180, 180): the circular distances the catch kernel is defined on
        self.register_buffer('_state_offsets_deg', (self._state_values_deg + 180) % 360 - 180)
        self._n_states = n_states

    def _catch_kernel(self, sd: torch.Tensor) -> torch.Tensor:
        """P(|beam - location| <= CATCH_THRESHOLD | latent mean) for observation noise s.d. ``sd``.

        Depends only on the circular distance between location and latent mean, so the
        expected catch probability is a circular convolution of the latent distribution
        with this kernel (computed via FFT, instead of a (location, state) matrix product).

        Args:
            sd: (..., ) noise s.d. in degrees.
        Returns:
            (..., offset) catch probabilities per grid offset.
        """
        normal = torch.distributions.Normal(0., 1.)
        sd = sd.unsqueeze(-1)
        offset = self._state_offsets_deg
        return normal.cdf((offset + CATCH_THRESHOLD) / sd) - normal.cdf((offset - CATCH_THRESHOLD) / sd)

    # --------------------------------------------------------------------- #
    #  Forward pass (differentiable)
    # --------------------------------------------------------------------- #

    def forward(
        self,
        inputs: torch.Tensor,
        prev_state: dict = None,
    ) -> tuple[torch.Tensor, dict]:
        """Run inference, prediction and response generation on the beam sequence.

        Args:
            inputs: (B, T, W, F) if batch_first else (T, W, B, F).
            prev_state: Dict with key ``'priors'`` — tensor (B, n_levels, n_states).
                        If None, starts from a uniform prior.

        Returns:
            predictions: (B, T, W, 2) next shield position (sin, cos); the expectation over
                         staying and moving, or a sampled response if ``sample_response``.
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
        participant_ids = inputs[:, 0, 0, -1].long()                            # (B,)
        p_cp = self.p_cp[participant_ids][:, None, None]                         # (B, 1, 1)
        response_sd = self.response_sd[participant_ids]                          # (B,)

        sd = self.noise_sd[participant_ids]                                      # (B, L)
        n_levels = sd.shape[1]
        catch_kernel_fft = torch.fft.rfft(self._catch_kernel(sd), dim=-1)       # (B, L, S/2+1)

        if prev_state is not None and 'priors' in prev_state:
            prior = prev_state['priors'].to(device)                              # (B, L, S)
        else:
            prior = torch.full((B, n_levels, n_states), 1.0 / (n_levels * n_states), device=device)

        valid = ~torch.isnan(inputs[:, :, 0, 0])                                 # (B, T)
        # zero-fill padded trials: NaNs in the masked-out branch of torch.where would still produce NaN gradients
        trial_inputs = torch.nan_to_num(inputs[:, :, 0], nan=0.)                 # (B, T, F)
        laser_deg = torch.atan2(trial_inputs[..., 2], trial_inputs[..., 3]).rad2deg() % 360
        shield_deg = torch.atan2(trial_inputs[..., 0], trial_inputs[..., 1]).rad2deg() % 360
        max_move = MOVEMENT_SPEED * trial_inputs[..., _AI_START + _AI_DT]        # (B, T)

        predictions = torch.full((B, T, 2), float('nan'), device=device)

        for t in range(T):
            mask = valid[:, t]
            if not mask.any():
                continue

            # --- 1. Inference: circular normal likelihood per noise level (B, L, S) ---
            angle_diff = (laser_deg[:, t, None, None] - self._state_values_deg + 180) % 360 - 180
            log_lik = -0.5 * (angle_diff / sd.unsqueeze(-1)) ** 2 - torch.log(sd).unsqueeze(-1)
            log_post = log_lik + torch.log(prior.clamp(min=1e-30))
            log_post = log_post - log_post.amax(dim=(1, 2), keepdim=True)
            posterior = torch.exp(log_post)
            posterior = posterior / posterior.sum(dim=(1, 2), keepdim=True)

            # --- 2. Prediction: a changepoint redraws the mean uniformly and keeps the noise level ---
            new_prior = (1 - p_cp) * posterior + p_cp * posterior.sum(dim=2, keepdim=True) / n_states
            prior = torch.where(mask[:, None, None], new_prior, prior)

            # --- 3. Response: expected catch probability per location (B, loc) ---
            expected_catch = torch.fft.irfft(
                (torch.fft.rfft(new_prior, dim=-1) * catch_kernel_fft).sum(dim=1), n=n_states, dim=-1,
            )
            weights = torch.softmax(self.choice_sharpness * torch.log(expected_catch.clamp(min=1e-12)), dim=-1)
            selected_deg = torch.atan2(
                (weights * self._state_sin).sum(-1), (weights * self._state_cos).sum(-1),
            ).rad2deg()

            # move toward the selected location along the shortest arc, as far as reachable
            distance = (selected_deg - shield_deg[:, t] + 180) % 360 - 180
            reached_deg = shield_deg[:, t] + torch.sign(distance) * torch.minimum(distance.abs(), max_move[:, t])
            response_probability = torch.erf(distance.abs() / (response_sd * math.sqrt(2)))

            shield = torch.stack([torch.sin(torch.deg2rad(shield_deg[:, t])), torch.cos(torch.deg2rad(shield_deg[:, t]))], dim=-1)
            reached = torch.stack([torch.sin(torch.deg2rad(reached_deg)), torch.cos(torch.deg2rad(reached_deg))], dim=-1)
            if self.sample_response:
                move = torch.bernoulli(response_probability).unsqueeze(-1)
            else:
                move = response_probability.unsqueeze(-1)
            pred = (1 - move) * shield + move * reached

            predictions[:, t] = torch.where(mask.unsqueeze(1), pred, predictions[:, t])

        predictions = predictions.unsqueeze(2)                                   # (B, T, 1, 2)

        if not self.batch_first:
            predictions = predictions.permute(1, 2, 0, 3)

        return predictions, {'priors': prior.detach()}

    def count_parameters(self) -> int:
        """Free parameters per participant: p_cp + response_sd + one noise s.d. per level."""
        return 2 + self.noise_sd_raw.shape[1]


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

    df = prepare_dataframe(pd.read_csv(path_data))

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

    # Expand ys with the metadata the clamped loss needs
    dataset = prepare_dataset(dataset)

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
        inner_model = model.model   # eval mode and use_sindy are set by the caller (BaseModel.eval() would reset use_sindy)
    else:
        inner_model = model.eval()

    # stochastic-response models (ChangePointModel) draw whether the shield moves instead of its expectation
    if hasattr(inner_model, 'sample_response'):
        inner_model.sample_response = True

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

    if hasattr(inner_model, 'sample_response'):
        inner_model.sample_response = False

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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from typing import Union
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, BaseModel, split_data_along_blockdim, dataset_to_csv, csv_to_dataset

from spice_bruckner2025 import CONFIG


# --- Constants ---

POSITION_SCALE = 300.0
SIGMA = 15.0 / POSITION_SCALE          # normalized outcome noise std
N_BLOCKS = 4
N_TRIALS_PER_BLOCK = 100

# Additional input column indices (relative to start of additional inputs block)
_AI_Z_NEXT = 0      # next trial's initial bucket position z_{t+1} (normalized)
_AI_CATCH = 1        # binary: caught the coin
_AI_V_T = 2          # helicopter visible on next trial (timeshifted -1)
_AI_SIGMA = 3        # outcome noise std (normalized)
_AI_MU_T = 4         # true helicopter position (normalized)
_AI_C_T = 5          # change point indicator


# --- MSE loss compatible with SPICE training pipeline ---

def mse_loss(prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """MSE loss for continuous position prediction.

    Same interface as cross_entropy_loss: (prediction, target) -> scalar.
    Both tensors have shape (..., 1) after NaN masking in the training loop.
    Extra kwargs (e.g. label_smoothing) are accepted and ignored for compatibility.
    """
    return torch.nn.functional.mse_loss(prediction.reshape(-1), target.reshape(-1))


# --- Data Loading ---

def get_dataset(
    path_data: str = None,
    test_blocks: tuple[int] = None,
) -> tuple[SpiceDataset, SpiceDataset]:
    """Load bruckner2025 data and build a SpiceDataset for continuous prediction.

    Feature layout of xs (n_features = 1 + 1 + 6 + 5 = 13):
        [0]  b_t / 300       — "action" (bucket position, normalized)
        [1]  x_t / 300       — "reward" (outcome position, normalized)
        [2]  z_{t+1} / 300   — additional input 0: next trial's initial bucket position
        [3]  catch            — additional input 1: binary coin catch flag
        [4]  v_{t+1}          — additional input 2: helicopter visible on next trial (timeshifted -1)
        [5]  sigma / 300      — additional input 3: outcome noise std (normalized)
        [6]  mu_t / 300       — additional input 4: true helicopter position
        [7]  c_t              — additional input 5: change point indicator
        [-5] time_trial      — always 0
        [-4] trial index     — 0-based
        [-3] block           — raw block id from CSV
        [-2] experiment      — 0-indexed condition id
        [-1] participant     — 0-indexed participant id

    Target ys: [b_{t+1} / 300]  (next bucket position, normalized)
    """
    if path_data is None:
        path_data = 'weinhardt2026/studies/bruckner2025/data/bruckner2025.csv'

    df = pd.read_csv(path_data)

    # Precompute z_next = next trial's initial bucket position (within each block)
    df['z_next'] = df.groupby(['participant', 'experiment', 'block'])['z_t'].shift(-1)

    # Normalize positions to [0, 1]
    for col in ['b_t', 'x_t', 'mu_t', 'z_next', 'sigma']:
        df[col] = df[col] / POSITION_SCALE

    dataset = csv_to_dataset(
        file=df,
        df_choice='b_t',
        df_feedback='x_t',
        additional_inputs=CONFIG.additional_inputs,
        continuous_action=True,
        timeshift_additional_inputs=(0, 0, -1, 0, 0, 0),
    )

    if test_blocks is not None:
        return split_data_along_blockdim(dataset, test_blocks)
    return dataset, None


# --- Benchmark Model: Reduced Bayesian ---

class RationalResourceModel(nn.Module):
    """Reduced Bayesian model for the helicopter task (Bruckner et al. 2025, Eqs. 4–10, 15).

    Faithful reimplementation of AlAgentRbm.py from the original codebase.
    Key design: belief is reset to the participant's actual prediction (b_t) at each
    trial (subjective prediction errors), while uncertainty dynamics carry forward.

    Per-participant learnable parameters: hazard rate (h), surprise sensitivity (s),
    uncertainty underestimation (u), and anchoring bias (d).
    """

    # Fixed initial conditions (matching AgentVars defaults)
    SIGMA_0 = 100.0            # initial estimation uncertainty (original scale)
    TAU_0 = 0.5                # initial relative uncertainty
    MU_0 = 150.0               # initial belief (center of screen)

    def __init__(self, n_participants: int = 1, sigma: float = SIGMA):
        super().__init__()
        self.sigma = sigma
        self.n_participants = n_participants
        self.n_actions = 1

        # Per-participant parameters (raw → transformed in forward)
        self.h_raw = nn.Parameter(torch.zeros(n_participants))         # → sigmoid → hazard rate [0, 1]
        self.s_raw = nn.Parameter(torch.zeros(n_participants))         # → sigmoid → surprise sensitivity [0, 1]
        self.u_raw = nn.Parameter(torch.zeros(n_participants))         # raw u; used as exp(u) for uncertainty scaling
        self.d_raw = nn.Parameter(torch.zeros(n_participants))         # → tanh → anchoring bias [-1, 1]

    def forward(self, xs: torch.Tensor, state: torch.Tensor = None):
        """Forward pass: trial-by-trial Bayesian belief updating.

        Matches the original code in AlAgentRbm.py + al_task_agent_int_rbm.py:
        - Belief is reset to b_t each trial (subjective PE)
        - Uncertainty (sigma_est_sq, tau) carries forward across trials
        - u parameter: raw u is exponentiated → divides uncertainty (Eq. 9)

        Args:
            xs: (B, T, 1, F) input tensor.
            state: unused (stateless between calls for simplicity).

        Returns:
            logits: (B, T, 1, 1) predicted next bucket position (normalized).
            state: None.
        """
        B, T, _, F = xs.shape
        device = xs.device

        xs = xs.nan_to_num(0.)
        b_t = xs[:, :, 0, 0]     # bucket position (normalized)
        x_t = xs[:, :, 0, 1]     # outcome (normalized)
        z_next = xs[:, :, 0, 2]  # z_{t+1}: next trial's initial bucket position (normalized)
        participant_ids = xs[:, 0, 0, -1].long()

        # Transform per-participant parameters
        h = torch.sigmoid(self.h_raw[participant_ids])               # (B,) hazard rate [0, 1]
        s = torch.sigmoid(self.s_raw[participant_ids])               # (B,) surprise sensitivity [0, 1]
        u_exp = torch.exp(torch.clamp_max(self.u_raw[participant_ids], 10))               # (B,) exp(u) for uncertainty scaling
        d = torch.tanh(self.d_raw[participant_ids])                  # (B,) anchoring bias [-1, 1]

        # Initialize uncertainty state (fixed, matching original AgentVars)
        sigma_est_sq = torch.full((B,), self.SIGMA_0 / POSITION_SCALE ** 2, device=device)
        tau = torch.full((B,), self.TAU_0, device=device)

        logits = torch.zeros(B, T, 1, 1, device=device)

        for t in range(T):
            # Prediction error using participant's actual prediction (Eq. 6)
            # Original: self.mu_t = b_t; delta = x_t - b_t (subjective PE)
            delta = x_t[:, t] - b_t[:, t]

            # Total variance (part of Eq. 8)
            total_var = sigma_est_sq + self.sigma ** 2

            # Change-point probability (Eq. 8)
            # In normalized space [0,1]: uniform density = 1, so (1/300)^s → 1^s = 1
            # Compute likelihood^s in log space to avoid gradient singularity
            # at likelihood=0 (grad of x^s diverges when 0 < s < 1, x → 0)
            log_lik = -0.5 * delta ** 2 / total_var - 0.5 * torch.log(2 * math.pi * total_var)
            likelihood_pow_s = torch.exp(s * log_lik)
            numerator = h                                                    # 1^s * h
            denominator = likelihood_pow_s * (1 - h) + h + 1e-10
            omega = (numerator / denominator).clamp(0, 1)

            # Learning rate (Eq. 7)
            alpha = (omega + tau - tau * omega).clamp(0, 1)

            # Predicted update (Eq. 5): a_hat = alpha * delta
            a_hat = alpha * delta

            # Anchoring bias (Eq. 15): a_hat += d * y_t, where y_t = z_{t+1} - b_t
            y = z_next[:, t] - b_t[:, t]
            a_hat = a_hat + d * y

            # Predicted next position: b_t + a_hat (Eq. 4, with belief reset to b_t)
            mu_pred = (b_t[:, t] + a_hat).clamp(0, 1)
            logits[:, t, 0, 0] = mu_pred

            # Update uncertainty for next trial (Eq. 9)
            # sigma_{t+1}^2 = (omega*sigma^2 + (1-omega)*tau*sigma^2
            #                  + omega*(1-omega)*(delta*(1-tau))^2) / exp(u)
            sigma_est_sq = (
                omega * self.sigma ** 2
                + (1 - omega) * tau * self.sigma ** 2
                + omega * (1 - omega) * (delta * (1 - tau)) ** 2
            ) / u_exp
            sigma_est_sq = sigma_est_sq.clamp(min=1e-8)

            # Update relative uncertainty (Eq. 10)
            tau = sigma_est_sq / (sigma_est_sq + self.sigma ** 2 + 1e-10)

        return logits, state


# --- Environment ---

class EnvironmentBruckner2025:
    """Replays outcome positions from the original dataset.

    No dynamic simulation — simply provides per-trial outcomes (x_t)
    and metadata from the original data.
    """

    def __init__(self, dataset: SpiceDataset):
        n_sessions, n_trials = dataset.xs.shape[0], dataset.xs.shape[1]

        n_actions = 1
        n_rewards = dataset.n_reward_features
        ai = n_actions + n_rewards  # start of additional inputs block

        self.outcomes = dataset.xs[:, :, 0, 1].clone()                  # x_t / 300
        self.z_next = dataset.xs[:, :, 0, ai + _AI_Z_NEXT].clone()     # z_{t+1} / 300
        self.catch = dataset.xs[:, :, 0, ai + _AI_CATCH].clone()       # catch flag
        self.v_t = dataset.xs[:, :, 0, ai + _AI_V_T].clone()           # helicopter visible (next trial)
        self.mu_t = dataset.xs[:, :, 0, ai + _AI_MU_T].clone()         # mu_t / 300
        self.c_t = dataset.xs[:, :, 0, ai + _AI_C_T].clone()           # change point indicator

        # Detect no-anchor-shift trials: z_{t+1} ≈ b_t in original data
        # (bucket stays where participant placed it → z_next should track model's b_t during generation)
        bucket_orig = dataset.xs[:, :, 0, 0]
        pixel_tol = 1.5 / POSITION_SCALE  # slightly above 1-pixel resolution
        self.is_no_anchor = (torch.abs(self.z_next - bucket_orig) < pixel_tol).clone()

        self.n_sessions = n_sessions
        self.n_trials = n_trials
        self.trial_counter = 0

    def reset(self):
        """Reset trial counter for a new generation run."""
        self.trial_counter = 0

    def step(self) -> dict:
        """Return outcome and metadata for the current trial.

        Returns dict with keys: outcome, z_next, is_no_anchor, catch, v_t, mu_t, c_t.
        All tensors have shape (n_sessions,).
        """
        t = self.trial_counter
        self.trial_counter += 1
        return {
            'outcome': self.outcomes[:, t],
            'z_next': self.z_next[:, t],
            'is_no_anchor': self.is_no_anchor[:, t],
            'catch': self.catch[:, t],
            'v_t': self.v_t[:, t],
            'mu_t': self.mu_t[:, t],
            'c_t': self.c_t[:, t],
        }


# --- Behavior Generation ---

@torch.no_grad()
def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    dataset: SpiceDataset,
    save_dataset: str = None,
) -> SpiceDataset:
    """Generate synthetic behavioral data for the helicopter task.

    Replays the original outcome sequence (x_t) and lets the model generate
    bucket positions trial by trial. Predictions are rounded to pixel
    resolution (1 / POSITION_SCALE) to match the discrete screen positions
    available to human participants.

    Args:
        model: Fitted model (SpiceEstimator or nn.Module).
        dataset: Original dataset (for structure and environment data).
        save_dataset: Optional path to save generated dataset as CSV.

    Returns:
        SpiceDataset with model-generated behavior.
    """
    print("Generating behavior...")

    n_sessions, n_trials, _, n_features = dataset.xs.shape
    n_rewards = dataset.n_reward_features

    env = EnvironmentBruckner2025(dataset)

    # Unwrap SpiceEstimator if needed
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

    xs_gen = dataset.xs.clone().to(device)
    ys_gen = torch.zeros_like(dataset.ys, device=device)
    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])

    # Initial bucket position: center of screen (normalized)
    bucket_position = torch.full((n_sessions,), 0.5, device=device)
    state = None

    env.reset()

    n_actions = 1
    ai = n_actions + n_rewards  # start of additional inputs block

    for t in tqdm(range(n_trials)):
        trial = env.step()
        trial = {k: v.to(device) for k, v in trial.items()}

        # For no-anchor-shift trials, z_{t+1} = model's current bucket position
        # (bucket stays where the model placed it, not where the human placed it)
        z_next = trial['z_next'].clone()
        z_next[trial['is_no_anchor']] = bucket_position[trial['is_no_anchor']]

        # Build observation
        xs_gen[:, t, 0, 0] = bucket_position                       # model-generated bucket position
        xs_gen[:, t, 0, 1] = trial['outcome']                      # replayed outcome
        xs_gen[:, t, 0, ai + _AI_Z_NEXT] = z_next                 # z_{t+1}: dynamic for no-anchor, replayed for anchor
        xs_gen[:, t, 0, ai + _AI_CATCH] = trial['catch']          # replayed catch
        xs_gen[:, t, 0, ai + _AI_V_T] = trial['v_t']              # replayed visibility
        xs_gen[:, t, 0, ai + _AI_MU_T] = trial['mu_t']            # replayed true mean
        xs_gen[:, t, 0, ai + _AI_C_T] = trial['c_t']              # replayed change point

        # Zero out inactive sessions
        inactive = ~valid_mask[:, t]
        xs_gen[inactive, t] = 0

        # Forward pass (single trial)
        obs = xs_gen[:, t:t + 1]   # (B, 1, 1, F)
        logits, state = inner_model(obs, state)

        # Extract predicted position → scalar per session
        if logits.dim() == 5:                          # BaseModel: (E, B, T, W, A)
            logits = logits.mean(dim=0)
        if logits.dim() == 4:                          # (B, T, W, A)
            predicted = logits[:, -1, 0, 0]
        else:                                          # (B, T, A)
            predicted = logits[:, -1, 0]

        # Round to pixel resolution and clamp to valid range
        pixel_step = 1.0 / POSITION_SCALE
        next_position = (predicted / pixel_step).round() * pixel_step
        next_position = next_position.clamp(0, 1)

        bucket_position = next_position
        ys_gen[:, t, 0, 0] = bucket_position

    # Restore NaN padding for inactive trials
    for t in range(n_trials):
        inactive = ~valid_mask[:, t]
        xs_gen[inactive, t] = float('nan')
        ys_gen[inactive, t] = float('nan')

    xs_gen = xs_gen.cpu()
    ys_gen = ys_gen.cpu()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=n_rewards)

    if save_dataset is not None:
        _save_generated_csv(dataset_gen, save_dataset)

    print("Done generating behavior.")
    return dataset_gen


def _save_generated_csv(dataset: SpiceDataset, path: str) -> None:
    """Save generated dataset to CSV with readable column names."""
    xs = dataset.xs.numpy()
    ys = dataset.ys.numpy()
    n_sessions = xs.shape[0]

    n_actions = 1
    n_rewards = dataset.n_reward_features
    ai = n_actions + n_rewards  # start of additional inputs block

    rows = []
    for s in range(n_sessions):
        n_valid = int((~np.isnan(xs[s, :, 0, 0])).sum())
        for t in range(n_valid):
            rows.append({
                'participant': int(xs[s, t, 0, -1]),
                'experiment': int(xs[s, t, 0, -2]),
                'block': int(xs[s, t, 0, -3]),
                'trial': int(xs[s, t, 0, -4]),
                'b_t': xs[s, t, 0, 0] * POSITION_SCALE,
                'x_t': xs[s, t, 0, 1] * POSITION_SCALE,
                'z_next': xs[s, t, 0, ai + _AI_Z_NEXT] * POSITION_SCALE,
                'catch': xs[s, t, 0, ai + _AI_CATCH],
                'v_t': xs[s, t, 0, ai + _AI_V_T],
                'mu_t': xs[s, t, 0, ai + _AI_MU_T] * POSITION_SCALE,
                'c_t': xs[s, t, 0, ai + _AI_C_T],
            })

    pd.DataFrame(rows).to_csv(path, index=False)

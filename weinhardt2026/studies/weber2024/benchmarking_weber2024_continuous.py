import math
from typing import Union

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

from spice_weber2024_continuous import CONFIG
from weinhardt2026.studies.weber2024.benchmarking_weber2024 import angular_distance, move_toward


# --- Constants ---

MOVEMENT_SPEED = 1.0        # degrees per frame
CATCH_THRESHOLD = 10.0      # degrees — shield catches laser if angular distance <= this
INITIAL_SHIELD = 360.0      # degrees (shield starting position each block)


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

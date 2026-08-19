"""Collecting and reshaping the RNN's hidden-state trajectories.

Stage 2 fits the SINDy equations against these trajectories rather than against
behaviour, so everything that materializes or reshapes them lives here.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple, Union, Optional
import shutil
from torch.nn.functional import mse_loss

from ..model import BaseModel
from ..spice_utils import SpiceDataset


def _vectorize_state(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    verbose: bool = False,
) -> tuple:
    """
    Run the frozen RNN forward on the full training data to collect state buffers
    for SINDy ridge solving.

    Args:
        model: RNN model (should be in eval mode with use_sindy=False)
        xs_train: 5D tensor (E, B, T, W, F)
        ys_train: 5D tensor matching xs_train
        batch_size_fwd: Max sessions to process per forward call.
            None = auto-detect via GPU probing (or all B on CPU).
        verbose: Print auto-batch info

    Returns:
        Tuple of (input_state_buffer, target_state_buffer, xs_flat, ys_flat)
    """
    _E, B, T, W, F = xs_train.shape
    n_features_y = ys_train.shape[-1]
    E = model.ensemble_size

    if xs_train.dim() == 4:
        xs_train = xs_train.unsqueeze(0).repeat(E, 1, 1, 1, 1)
        ys_train = ys_train.unsqueeze(0).repeat(E, 1, 1, 1, 1)

    # State buffers: (within_ts, E, n_trials*B, n_items) — always full size
    state_keys = list(model.init_state(batch_size=1, within_ts=W).keys())
    state_buffer_current = {s: torch.zeros((W, E, T*B, model.n_items), dtype=torch.float32, device=model.device) for s in state_keys}
    state_buffer_next = {s: torch.zeros((W, E, T*B, model.n_items), dtype=torch.float32, device=model.device) for s in state_keys}

    # Per-session hidden state carried across timesteps: (W, E, B, n_items)
    session_states = {s: torch.full((W, E, B, model.n_items),
                                    fill_value=model.spice_config.memory_state[s] if model.spice_config.memory_state[s] is not None else 0.,
                                    dtype=torch.float32, device=model.device) for s in state_keys}

    # Apply learnable per-participant initial values
    if hasattr(model, 'learnable_initial_values') and model.learnable_initial_values:
        participant_ids = xs_train[0, :, 0, 0, -1].long().to(model.device)
        E_idx = torch.arange(E, device=model.device).unsqueeze(1)
        for key, param in model.learnable_initial_values.items():
            init_val = param[E_idx, participant_ids]  # [E, B]
            session_states[key] = init_val.unsqueeze(0).unsqueeze(-1).expand(
                W, -1, -1, model.n_items,
            ).clone().detach()

    batch_size_fwd = B
    with torch.no_grad():
        for t in range(T):
            for b_start in range(0, B, batch_size_fwd):
                b_end = min(b_start + batch_size_fwd, B)

                # Load this sub-batch's carried state into model
                sub_state = {s: session_states[s][:, :, b_start:b_end].clone() for s in state_keys}
                model.set_state(sub_state)

                # Record pre-forward state
                for s in state_keys:
                    state_buffer_current[s][0, :, t*B+b_start:t*B+b_end] = sub_state[s][-1]

                # Forward one timestep for this sub-batch
                xs_sub = xs_train[:, b_start:b_end, t:t+1].to(model.device)
                updated_state = model(xs_sub, model.get_state())[1]

                # Record post-forward state
                for s in state_keys:
                    state_buffer_current[s][1:, :, t*B+b_start:t*B+b_end] = updated_state[s][:-1]
                    state_buffer_next[s][:, :, t*B+b_start:t*B+b_end] = updated_state[s]

                # Save updated state for next timestep
                for s in state_keys:
                    session_states[s][:, :, b_start:b_end] = updated_state[s]

    del session_states

    # Flatten to (E, flat_total, 1, 1, F)
    flat_total = W * T * B
    xs_flat = xs_train.permute(0, 2, 1, 3, 4).reshape(E, flat_total, 1, 1, F)
    ys_flat = ys_train.permute(0, 2, 1, 3, 4).reshape(E, flat_total, 1, 1, n_features_y)[0]

    # Reshape state buffers: (W, E, T*B, items) -> (1, E, flat_total, items)
    for s in state_keys:
        state_buffer_current[s] = state_buffer_current[s].permute(1, 2, 0, 3).reshape(1, E, flat_total, model.n_items)
        state_buffer_next[s] = state_buffer_next[s].permute(1, 2, 0, 3).reshape(1, E, flat_total, model.n_items)

    # Remove NaN-padded samples
    nan_mask = ~torch.isnan(xs_flat[0, :, 0, 0, :model.n_actions].sum(dim=(-1)))
    xs_flat = xs_flat[:, nan_mask]
    state_buffer_current = {s: state_buffer_current[s][:, :, nan_mask] for s in state_buffer_current}
    state_buffer_next = {s: state_buffer_next[s][:, :, nan_mask] for s in state_buffer_next}

    return state_buffer_current, state_buffer_next, xs_flat, ys_flat


def _vectorize_state_sequential(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    verbose: bool = False,
) -> tuple:
    """
    Run the frozen RNN forward on the full training data to collect sequential
    state trajectories for multi-step shooting in SINDy refit.

    Unlike _vectorize_state which flattens all (h_t, h_{t+1}) pairs into i.i.d.
    samples, this function preserves the session x trial structure so that
    multi-step windows can be extracted for shooting-based optimization.

    Args:
        model: RNN model (should be in eval mode with use_sindy=False)
        xs_train: 5D tensor (E, B, T, W, F)
        ys_train: 5D tensor matching xs_train
        verbose: Print info

    Returns:
        Tuple of (state_trajectories, nan_mask) where:
        - state_trajectories: Dict[state_key -> (W, E, B, T+1, I)] containing the
          full state trajectory including the initial state at t=0.
        - nan_mask: (B, T) boolean mask where True = valid trial
    """
    _E, B, T, W, F = xs_train.shape
    E = model.ensemble_size

    if xs_train.dim() == 4:
        xs_train = xs_train.unsqueeze(0).repeat(E, 1, 1, 1, 1)
        ys_train = ys_train.unsqueeze(0).repeat(E, 1, 1, 1, 1)

    state_keys = list(model.init_state(batch_size=1, within_ts=W).keys())
    state_trajectories = {
        s: torch.zeros((W, E, B, T + 1, model.n_items), dtype=torch.float32, device=model.device)
        for s in state_keys
    }

    session_states = {
        s: torch.full(
            (W, E, B, model.n_items),
            fill_value=model.spice_config.memory_state[s] if model.spice_config.memory_state[s] is not None else 0.,
            dtype=torch.float32, device=model.device,
        )
        for s in state_keys
    }

    # Apply learnable per-participant initial values
    if hasattr(model, 'learnable_initial_values') and model.learnable_initial_values:
        participant_ids = xs_train[0, :, 0, 0, -1].long().to(model.device)
        E_idx = torch.arange(E, device=model.device).unsqueeze(1)
        for key, param in model.learnable_initial_values.items():
            init_val = param[E_idx, participant_ids]  # [E, B]
            session_states[key] = init_val.unsqueeze(0).unsqueeze(-1).expand(
                W, -1, -1, model.n_items,
            ).clone().detach()

    # Record initial state at t=0
    for s in state_keys:
        state_trajectories[s][:, :, :, 0] = session_states[s]

    with torch.no_grad():
        for t in range(T):
            model.set_state({s: session_states[s].clone() for s in state_keys})
            xs_sub = xs_train[:, :, t:t + 1].to(model.device)
            updated_state = model(xs_sub, model.get_state())[1]

            for s in state_keys:
                state_trajectories[s][:, :, :, t + 1] = updated_state[s]
                session_states[s] = updated_state[s]

    del session_states

    # Build NaN mask: (B, T) — True where trial is valid
    nan_mask = ~torch.isnan(xs_train[0, :, :, 0, :model.n_actions].sum(dim=-1))  # (B, T)

    return state_trajectories, nan_mask


def _flatten_state_trajectories_onestep(
    xs_train: torch.Tensor,
    state_trajectories: dict,
    nan_mask: torch.Tensor,
) -> tuple:
    """
    Flatten (W, T) state trajectories into independent one-step (T'=1, W'=1)
    pseudo-sessions, so every within-trial and across-trial transition becomes
    its own teacher-forced one-step regression sample -- regardless of whether
    a given model's sequential structure lives in W (e.g. a within-trial DDM,
    W>1 T=1) or T (standard multi-trial models, W=1 T>1). Feeding the result
    into _run_shooting_epoch_vectorized with K=1 then does genuine one-step
    fitting: init_forward_pass() seeds state from the given prev_state whenever
    it's not None, bypassing memory_state-based initialization entirely, so each
    pseudo-session's teacher-forced h_current is used as-is instead of being
    overwritten by the model's own default initial value.

    Args:
        xs_train: 5D tensor (E, B, T, W, F)
        state_trajectories: Dict[state_key -> (W, E, B, T+1, I)] from
            _vectorize_state_sequential
        nan_mask: (B, T) boolean validity mask

    Returns:
        Tuple of (xs_flat, state_trajectories_flat, nan_mask_flat, B_flat):
        - xs_flat: (E, B*T*W, 1, 1, F)
        - state_trajectories_flat: Dict[state_key -> (1, E, B*T*W, 2, I)]
        - nan_mask_flat: (B*T*W, 1)
        - B_flat: number of flattened pseudo-sessions
    """
    E, B, T, W, F = xs_train.shape

    xs_flat = xs_train.reshape(E, B * T * W, F).unsqueeze(2).unsqueeze(2)  # (E, B*T*W, 1, 1, F)

    state_trajectories_flat = {}
    for s, traj in state_trajectories.items():
        # h_current[w, t] = traj[w-1, t] for w>0, traj[-1, t-1] for w=0 (t-1=0 -> initial state)
        h_current = torch.cat((traj[-1:, :, :, :-1], traj[:-1, :, :, 1:]), dim=0)  # (W, E, B, T, I)
        h_next = traj[:, :, :, 1:]  # (W, E, B, T, I)
        I = traj.shape[-1]
        h_current_flat = h_current.permute(1, 2, 3, 0, 4).reshape(E, B * T * W, I)
        h_next_flat = h_next.permute(1, 2, 3, 0, 4).reshape(E, B * T * W, I)
        state_trajectories_flat[s] = torch.stack((h_current_flat, h_next_flat), dim=2).unsqueeze(0)  # (1, E, B*T*W, 2, I)

    nan_mask_flat = nan_mask.unsqueeze(-1).expand(-1, -1, W).reshape(B * T * W).unsqueeze(-1)  # (B*T*W, 1)

    return xs_flat, state_trajectories_flat, nan_mask_flat, B * T * W

"""Multi-step shooting rollouts against frozen hidden-state trajectories."""

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
from .reporting import _check_cuda_oom


def _project_after_step(model: BaseModel, optimizer: torch.optim.Optimizer,
                        sindy_lambda_loading: float = None, sindy_lambda_concept: float = None) -> None:
    """Re-establish the factorization's constraints after an optimizer step.

    Order matters, and the three steps are ordered so each penalty sees the scale it was
    tuned against. The prox on V runs *first*, while the row is still on the unit-norm
    scale the previous step left it on. The gauge is fixed *second*, which also re-masks
    the pruned coordinates. The prox on Z runs *last*, on loadings already at their final
    scale -- doing that one earlier lets the renormalization multiply Z by the row norms
    and partially undo the shrinkage just applied, which makes the effective L1 threshold
    depend on how far V happened to drift that step rather than on sindy_lambda_loading alone.

    Both run every step. Normalizing less often would be worse, not better: at
    equilibrium each step perturbs a row norm by O(lr), a slow reparametrization Adam's
    moment estimates track without trouble, whereas batching the drift into one large
    correction hands the optimizer a genuine discontinuity.
    """
    lr = max((group.get('lr', 0.0) for group in optimizer.param_groups), default=0.0)
    model.project_directions(lr=lr, sindy_lambda_concept=sindy_lambda_concept or 0.0)
    model.normalize_concept_directions()
    model.project_loadings(lr=lr, sindy_lambda_loading=sindy_lambda_loading or 0.0)


def _run_shooting_epoch_vectorized(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    xs_train: torch.Tensor,
    state_trajectories: dict,
    nan_mask: torch.Tensor,
    window_starts: list,
    K: int,
    batch_sessions: torch.Tensor,
    sindy_lambda_loading: float = None,
    sindy_lambda_concept: float = None,
) -> float:
    """Vectorized shooting epoch: fold all windows into the batch dimension.

    Instead of looping over windows one at a time (each with its own forward,
    backward, and optimizer step), this function stacks all shooting windows
    into the batch dimension and processes them in parallel.  For K=1 this
    reduces T separate forward+backward passes to a single one; for K>1 it
    reduces T/K × K passes to just K batched passes.  A single backward +
    optimizer step is performed per call.

    Args:
        model: Model in SINDy training mode (use_sindy=True, RNN frozen)
        optimizer: SINDy coefficient optimizer
        xs_train: 5D training data (E, B, T, W, F)
        state_trajectories: Dict[state_key -> (W, E, B, T+1, I)]
        nan_mask: (B, T) boolean validity mask
        window_starts: List of trial indices where shooting windows begin
        K: Shooting window size
        batch_sessions: Session indices for this batch (tensor)
        sindy_lambda_loading: L1 penalty strength on the loadings (None or 0 = disabled)
        sindy_lambda_concept: L1 penalty strength on the concept directions (None or 0 = disabled)

    Returns:
        Mean loss over all valid steps
    """
    T = nan_mask.shape[1]
    B_batch = len(batch_sessions)
    n_windows = len(window_starts)
    B_eff = B_batch * n_windows

    if B_eff == 0:
        return 0.0

    # Build index arrays for all (window, session) pairs.
    # Window-major ordering: [w0_s0, w0_s1, …, w0_sB, w1_s0, …]
    ws = torch.tensor(window_starts, dtype=torch.long)
    session_idx = batch_sessions.repeat(n_windows)                     # (B_eff,)
    time_idx = ws.unsqueeze(1).expand(-1, B_batch).reshape(-1)         # (B_eff,)

    # Gather initial states for every (window, session) pair
    current_state = {
        s: state_trajectories[s][:, :, session_idx, time_idx].to(model.device)
        for s in state_trajectories
    }  # each value: (W, E, B_eff, I)

    state_noise_std = 0.05
    # if state_noise_std > 0:
    #     current_state = {
    #         s: v + state_noise_std * torch.randn_like(v)
    #         for s, v in current_state.items()
    #     }

    model.zero_grad()
    total_loss = torch.tensor(0.0, device=model.device)
    n_valid_steps = 0

    for k in range(K):
        t_k = time_idx + k

        # Validity: within temporal bounds AND not NaN-padded
        in_bounds = t_k < T
        if not in_bounds.any():
            break
        t_k_safe = torch.clamp(t_k, max=T - 1)       # safe index for gathering

        valid = (nan_mask[session_idx, t_k_safe] & in_bounds).to(model.device)
        if not valid.any():
            continue

        # Gather xs for this step: (E, B_eff, W, F) → insert T=1 → (E, B_eff, 1, W, F)
        xs_step = xs_train[:, session_idx, t_k_safe].unsqueeze(2).to(model.device)

        if state_noise_std > 0:
            current_state = {
                s: v + state_noise_std * torch.randn_like(v)
                for s, v in current_state.items()
            }
        
        # Forward pass through full model in SINDy mode
        _, next_state = model(xs_step, current_state)

        # MSE against pre-recorded RNN states
        step_loss = torch.tensor(0.0, device=model.device)
        for s_key in model.spice_config.states_in_logit:
            target = state_trajectories[s_key][:, :, session_idx, t_k_safe + 1].to(model.device)
            pred = next_state[s_key]
            mask = valid.view(1, 1, -1, 1).expand_as(pred)
            diff = (pred - target) ** 2
            step_loss = step_loss + (diff * mask).sum() / mask.sum().clamp(min=1)

        total_loss = total_loss + step_loss
        n_valid_steps += 1
        current_state = next_state

    if n_valid_steps > 0:
        total_loss = total_loss / n_valid_steps

        if optimizer is not None:
            if sindy_lambda_loading is not None and sindy_lambda_loading > 0:
                total_loss = total_loss + model.compute_constants_penalty(strength=sindy_lambda_loading)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # The L1 on the loadings is a proximal step rather than a loss term, so it
            # has to follow the optimizer step. Re-fixing the gauge here too keeps the
            # penalty meaningful: Z @ V is invariant under (Z D, D^-1 V), so an
            # unnormalized V lets the optimizer shrink the penalty for free.
            _project_after_step(model, optimizer, sindy_lambda_loading, sindy_lambda_concept)

        return total_loss.item()

    return 0.0


def _run_shooting_eval_batched(
    model: BaseModel,
    xs_train: torch.Tensor,
    state_trajectories: dict,
    nan_mask: torch.Tensor,
    window_starts: list,
    K: int,
    B_total: int,
    sindy_lambda_loading: float,
    batch_size_sessions: int = None,
) -> tuple:
    """No-grad ridge-solution evaluation over all B_total sessions, chunked and with
    automatic OOM backoff -- mirrors the batching/backoff the SGD loops right after each
    call site already do. The plain `batch_sessions=torch.arange(B_total)` single-shot
    call this replaces has no such backoff, so on models with a large per-session state
    (e.g. SpiceDDM's evidence_pdf grid) it can OOM outright on datasets the subsequent
    (batched) SGD loop handles fine. Returns (mean_loss, safe_batch_size) so the caller
    can seed the SGD loop's batch size with a value already known to fit.
    """
    if batch_size_sessions is None or batch_size_sessions > B_total:
        batch_size_sessions = B_total

    while True:
        try:
            loss_total = 0.0
            n_batches = 0
            for b_start in range(0, B_total, batch_size_sessions):
                b_end = min(b_start + batch_size_sessions, B_total)
                batch_sessions = torch.arange(b_start, b_end)
                loss_e = _run_shooting_epoch_vectorized(
                    model=model,
                    optimizer=None,
                    xs_train=xs_train,
                    state_trajectories=state_trajectories,
                    nan_mask=nan_mask,
                    window_starts=window_starts,
                    K=K,
                    batch_sessions=batch_sessions,
                    sindy_lambda_loading=sindy_lambda_loading,
                )
                loss_total += loss_e
                n_batches += 1
            return (loss_total / n_batches if n_batches > 0 else 0.0), batch_size_sessions
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _check_cuda_oom(e):
                raise
            if batch_size_sessions <= 1:
                raise RuntimeError(f"Automatic batch size probing was unsuccessful for shooting evaluation. Current batch size is {batch_size_sessions} but could still not be started. Please try again with a smaller ensemble size (current: {model.ensemble_size}).")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch_size_sessions = max(1, batch_size_sessions // 2)

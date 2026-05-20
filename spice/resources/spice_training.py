
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
from scipy.stats import t as t_dist
from torch.nn.functional import mse_loss  # using standard mse loss for spice should be fine most of the time

from .model import BaseModel
from .spice_utils import SpiceDataset
from .sindy_differentiable import get_library_term_degrees


# === DEBUG MODE: Set to False for live-updating display, True for line-by-line output ===
DEBUG_MODE = False


def _get_terminal_width() -> int:
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 160
    return terminal_width

def _check_cuda_oom(exception) -> bool:
    return isinstance(exception, RuntimeError) and "out of memory" not in str(exception) and "CUBLAS_STATUS_ALLOC_FAILED" not in str(exception)

def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def _print_training_status(
    len_last_print: int,
    model: BaseModel,
    n_calls: int,
    epochs: int,
    loss_train: float,
    loss_test_rnn: float,
    loss_test_sindy: float,
    time_elapsed: float,
    convergence_value: float,
    sindy_weight: float,
    scheduler=None,
    warmup_steps: int = 0,
    converged: bool = False,
    finished: bool = False,
    keep_log: bool = False,
    is_notebook: bool = False,
):
    """Print live-updating training status block."""
    
    # Build comprehensive status display
    terminal_width = _get_terminal_width()
    if is_notebook:
        terminal_width = terminal_width*2
    status_lines = []
    status_lines.append("=" * terminal_width)
    
    # Training progress bar (tqdm-style)
    postfix_parts = {}
    postfix_parts['L(Train)'] = f'{loss_train:.7f}'
    if loss_test_rnn is not None:
        postfix_parts['L(Val,RNN)'] = f'{loss_test_rnn:.7f}'
    if loss_test_sindy is not None:
        postfix_parts['L(Val,SINDy)'] = f'{loss_test_sindy:.7f}'
    if convergence_value is not None:
        postfix_parts['Conv'] = f'{convergence_value:.2e}'
    if scheduler is not None:
        postfix_parts['LR'] = f'{scheduler.get_last_lr()[-1]:.2e}'

    postfix_str = ', '.join(f'{k}={v}' for k, v in postfix_parts.items())
    bar_str = tqdm.format_meter(
        n=n_calls,
        total=epochs,
        elapsed=time_elapsed if time_elapsed is not None else 0,
        ncols=terminal_width,
        postfix=postfix_str,
    )
    status_lines.append(bar_str)
    
    max_len_module = max([len(module) for module in model.get_modules()])
    
    # Add SPICE model equations if SINDy is active
    if sindy_weight > 0:
        status_lines.append("-" * terminal_width)
        status_lines.append(f"SPICE Model (Coefficients: {model.count_sindy_coefficients()[0, 0]:.0f}):")
        status_lines.append(model.get_spice_model_string(participant_id=0))
        
        # Add patience summary
        status_lines.append("-" * terminal_width)
        status_lines.append("Pruning patience:")
        for m in model.submodules_rnn:
            # Build masks for terms
            n_terms = len(model.sindy_candidate_terms[m])
            # Linear terms with fit_linear=False: show "-" (fixed at 1.0)
            fixed_linear = [False] * n_terms
            if not model.sindy_specs[m].get('fit_linear', True):
                term_degrees = get_library_term_degrees(model.sindy_candidate_terms[m])
                for idx, degree in enumerate(term_degrees):
                    if degree == 1:
                        fixed_linear[idx] = True
            # State-containing terms with include_state=False: skip entirely (not printed)
            skip_state = [False] * n_terms
            if not model.sindy_specs[m].get('include_state', True):
                for idx, term in enumerate(model.sindy_candidate_terms[m]):
                    if m in term:
                        skip_state[idx] = True

            patience_list = ""
            for ip, p in enumerate(model.sindy_pruning_patience_counters[m][0, 0, 0]):
                if skip_state[ip]:
                    continue
                elif fixed_linear[ip]:
                    patience_list += "-"
                elif model.sindy_coefficients_presence[m][0, 0, 0, ip]:
                    patience_list += str(p.item())
                else:
                    patience_list += "-"
                patience_list += ", "
            space_filler = " "+" "*(max_len_module-len(m)) if max_len_module > len(m) else " "
            status_lines.append(m + ":" + space_filler + patience_list[:-2])

        status_lines.append("-" * terminal_width)
        status_lines.append(f"Term presence across SPICE models (number of models={model.n_participants*model.n_experiments}):")
        for m in model.get_modules():
            # (E, P, X, terms) -> mean across E, sum across P and X
            presence = model.sindy_coefficients_presence[m].float().mean(dim=0).sum(dim=0).sum(dim=0).int().detach().cpu().numpy()
            presence_list = ""
            for p in presence:
                presence_list += str(p)+", "
            space_filler = " "+" "*(max_len_module-len(m)) if max_len_module > len(m) else " "
            status_lines.append(m + ":" + space_filler + presence_list[:-2])
    status_lines.append("=" * terminal_width)
    
    # Convergence messages
    if converged:
        status_lines.append('Model converged!')
    elif finished:
        status_lines.append('Maximum number of training epochs reached.')
        if not converged:
            status_lines.append('Model did not converge yet.')
    
    msg = "\n".join(status_lines)
    current_line_count = len(status_lines)
    
    # Clear and reprint
    if not keep_log and n_calls > 1:
        if is_notebook:
            from IPython.display import clear_output
            clear_output(wait=True)
        else:
            os.system('clear' if os.name == 'posix' else 'cls')
    print(msg, flush=True)
    
    return current_line_count


class SpiceLRScheduler:
    """
    Unified LR scheduler for SPICE training with warmup and post-pruning boosts.

    Manages separate schedules for RNN (param_groups[1]) and SINDy (param_groups[0]):

    - Warmup: RNN LR starts at warmup_factor * base_lr, linearly decays to base_lr
      over n_warmup_steps. SINDy LR is unchanged during warmup.
    - Post-pruning boost: After a pruning event, RNN and SINDy LRs are temporarily
      multiplied by their respective boost factors for boost_duration epochs,
      then return to their base rates.

    Args:
        optimizer: Optimizer with param_groups[0]=SINDy, param_groups[1]=RNN.
        n_warmup_steps: Number of warmup epochs for RNN LR ramp-down.
        warmup_factor: RNN LR multiplier at start (linearly decays to 1.0).
            Set to 1.0 to disable warmup.
        boost_factor_rnn: RNN LR multiplier after pruning (1.0 = no boost).
        boost_factor_sindy: SINDy LR multiplier after pruning (default 10.0:
            0.001 -> 0.01, matching previous hardcoded behavior).
        boost_duration_frac: Fraction of sindy_pruning_frequency for boost
            duration (default 0.1).
        sindy_pruning_frequency: Epochs between pruning events (used to
            compute boost duration).
    """
    def __init__(self, optimizer, n_warmup_steps=0, warmup_factor=10.,
                 boost_factor_rnn=1., boost_factor_sindy=10.,
                 boost_duration_frac=0.1, sindy_pruning_frequency=100):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.warmup_factor = warmup_factor
        self.boost_factor_rnn = boost_factor_rnn
        self.boost_factor_sindy = boost_factor_sindy
        self.boost_duration = max(1, int(boost_duration_frac * sindy_pruning_frequency))

        # Store base LRs
        self._has_sindy = len(optimizer.param_groups) > 1
        if self._has_sindy:
            self.base_lr_sindy = optimizer.param_groups[0]['lr']
            self.base_lr_rnn = optimizer.param_groups[1]['lr']
        else:
            self.base_lr_rnn = optimizer.param_groups[0]['lr']
            self.base_lr_sindy = None

        # Apply initial warmup LR
        if n_warmup_steps > 0 and warmup_factor != 1.:
            self._rnn_pg['lr'] = self.base_lr_rnn * warmup_factor

        # Boost state
        self._boost_end = 0

    @property
    def _rnn_pg(self):
        return self.optimizer.param_groups[1] if self._has_sindy else self.optimizer.param_groups[0]

    @property
    def _sindy_pg(self):
        return self.optimizer.param_groups[0] if self._has_sindy else None

    def step(self, epoch):
        """Update LRs for the current epoch (warmup interpolation + boost expiry)."""
        # Warmup: linearly interpolate RNN LR from warmup_factor*base to base
        if epoch < self.n_warmup_steps and self.warmup_factor != 1.:
            frac = epoch / max(1, self.n_warmup_steps)
            factor = self.warmup_factor + (1. - self.warmup_factor) * frac
            self._rnn_pg['lr'] = self.base_lr_rnn * factor
        elif epoch == self.n_warmup_steps and self.warmup_factor != 1.:
            self._rnn_pg['lr'] = self.base_lr_rnn

        # Boost expiry
        if self._boost_end > 0 and epoch >= self._boost_end:
            self._rnn_pg['lr'] = self.base_lr_rnn
            if self._sindy_pg is not None:
                self._sindy_pg['lr'] = self.base_lr_sindy
            self._boost_end = 0

    def notify_pruning(self, epoch):
        """Activate post-pruning LR boost for both param groups."""
        if self.boost_factor_rnn != 1.:
            self._rnn_pg['lr'] = self.base_lr_rnn * self.boost_factor_rnn
        if self._sindy_pg is not None and self.boost_factor_sindy != 1.:
            self._sindy_pg['lr'] = self.base_lr_sindy * self.boost_factor_sindy
        self._boost_end = epoch + self.boost_duration

    def get_lr(self):
        """Retrieve current learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """Retrieve current learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]


def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor, label_smoothing=0.) -> torch.Tensor:
    """Wrapper for torch's cross entropy loss which does all the reshaping when getting SpiceDataset.ys tensors as predicitons and targets."""
    n_actions = target.shape[-1]
    
    prediction = prediction.reshape(-1, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)
    
    return torch.nn.functional.cross_entropy(prediction, target, label_smoothing=label_smoothing)


def _setup_warmup_scaler(n_warmup_steps: int, exp_max: float = 1) -> torch.Tensor:
    """Create exponential warmup scaler for SINDy weight."""
    if n_warmup_steps <= 0:
        return None
    warmup_scaler = torch.exp(torch.linspace(0, exp_max, n_warmup_steps))
    warmup_scaler = (warmup_scaler - warmup_scaler.min()) / (warmup_scaler.max() - warmup_scaler.min()) + 1e-4
    return warmup_scaler



def _run_batch_training(
    model: BaseModel,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    sindy_weight: float = 0.,
    sindy_alpha: float = 0.,
    n_steps: int = None,
    loss_fn: callable = cross_entropy_loss,
    loss_fn_kwargs: dict = {},
    ):

    """
    Trains a model with the given batch.
    xs/ys are 5D (E, B, T, W, F) — already bootstrapped.
    """

    E, B = xs.shape[0], xs.shape[1]

    if n_steps is None:
        n_steps = xs.shape[2]

    model.init_state(batch_size=B, within_ts=xs.shape[3])
    state = model.get_state(detach=True)

    loss_batch = 0
    iterations = 0
    for t in range(0, xs.shape[2], n_steps):
        n_steps = min(xs.shape[2]-t, n_steps)
        xs_step = xs[:, :, t:t+n_steps]
        ys_step = ys[:, :, t:t+n_steps]

        state = model.get_state(detach=True)
        ys_pred, _ = model(xs_step, state)

        # Mask out padding (NaN values)
        # xs_step is 5D: (E, B, T_out, T_in, F)
        mask = ~torch.isnan(xs_step[..., :model.n_actions].sum(dim=(-1)))
        ys_pred = ys_pred[mask]
        ys_step = ys_step[mask]

        loss_step = loss_fn(ys_pred, ys_step, **loss_fn_kwargs)

        if torch.is_grad_enabled():
            # Add SINDy losses (decoupled: sindy_weight controls RNN regularization only)
            if sindy_weight > 0 and model.sindy_loss_reg != 0:
                loss_step = loss_step + sindy_weight * model.sindy_loss_reg + model.sindy_loss_fit

            if sindy_weight > 0 and sindy_alpha > 0:
                loss_step = loss_step + model.compute_weighted_coefficient_penalty(sindy_alpha=sindy_alpha, norm=1)
                
            # backpropagation
            optimizer.zero_grad()
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_batch += loss_step.item()
        iterations += 1

    return model, optimizer, loss_batch/iterations


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


def _ensemble_pruning(
    model: BaseModel,
    sindy_ensemble_pruning: float,
    sindy_threshold_pruning: float,
    verbose: bool,
    n_terms_pruning: int = None,
    sindy_ensemble_pruning_mode: str = 'ci',
):
    """Ensemble-based pruning with optional rate limiting.

    Args:
        n_terms_pruning: Max terms to prune per event (across all modules).
            When set, only the smallest-magnitude candidates are pruned.
            None = no limit (prune all that fail the test).
        sindy_ensemble_pruning_mode: 'ci' for minimum-effect CI test (default),
            'ratio' for ensemble ratio test.
    """
    pruned = False
    module_list = list(model.submodules_rnn.keys())

    if sindy_ensemble_pruning_mode == 'ratio':
        ensemble_test_fn = _ensemble_ratio_test
        ensemble_test_kwargs = dict(
            threshold=sindy_threshold_pruning or 0.0,
            ratio=sindy_ensemble_pruning,
        )
    else:
        ensemble_test_fn = _minimum_effect_ci_test
        ensemble_test_kwargs = dict(
            alpha=sindy_ensemble_pruning,
            delta=sindy_threshold_pruning or 0.0,
        )

    confidence_masks = _compute_pruning_masks(
        model,
        ensemble_test_fn=ensemble_test_fn,
        ensemble_test_kwargs=ensemble_test_kwargs,
        participant_threshold=None,
        verbose=verbose,
    )

    # Update patience counters for all modules
    for module in module_list:
        mask = confidence_masks[module].to(model.device)  # (P, X, terms)
        still_active = model.sindy_coefficients_presence[module].any(dim=0)  # (P, X, terms)
        failed = ~mask & still_active

        counters = model.sindy_pruning_patience_counters[module]
        failed_e = failed.unsqueeze(0).expand_as(counters)
        counters.data = torch.where(failed_e, counters + 1, torch.zeros_like(counters))

    # Collect candidates across all modules: terms with counters >= 2
    all_prune_candidates = torch.cat(
        [model.sindy_pruning_patience_counters[m][0] >= 2 for m in module_list], dim=-1
    )  # (P, X, total_terms)

    if all_prune_candidates.any():
        # Rate-limit: only prune the n_terms_pruning smallest-magnitude candidates
        if n_terms_pruning is not None:
            all_coeffs_abs = torch.cat([
                (model.sindy_coefficients[m] * model.sindy_coefficients_presence[m].float())
                .detach().abs().mean(dim=0)
                for m in module_list
            ], dim=-1)  # (P, X, total_terms)

            # Set non-candidates to inf so they won't be selected
            temp_coeffs = all_coeffs_abs.clone()
            temp_coeffs[~all_prune_candidates] = torch.inf

            k = min(n_terms_pruning, all_prune_candidates.sum(dim=-1).max().item())
            if k > 0:
                _, indices = torch.topk(temp_coeffs, k, dim=-1, largest=False)
                limited_mask = torch.zeros_like(all_prune_candidates)
                limited_mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
                all_prune_candidates = all_prune_candidates & limited_mask

        # Split back to modules and apply pruning
        start_idx = 0
        for module in module_list:
            n_terms = model.sindy_coefficients[module].shape[-1]
            prune = all_prune_candidates[..., start_idx:start_idx + n_terms]  # (P, X, terms)
            if prune.any():
                prune_e = prune.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1)
                model.sindy_coefficients_presence[module].data &= ~prune_e
                model.sindy_coefficients[module].data *= model.sindy_coefficients_presence[module].float()
                model.sindy_pruning_patience_counters[module].data *= (~prune_e).int()
                pruned = True
            start_idx += n_terms

    return model, pruned


def _ridge_solve_sindy(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
) -> bool:
    """
    Lightweight ridge solve: snap SINDy coefficients to their MSE-optimal
    values without touching the optimizer state.

    Runs a frozen RNN forward pass over flattened (trial, session) pairs.
    Inside each call_module(), sindy_ridge_solve() accumulates per-participant
    normal equations and solves in closed form.

    Returns:
        True if the ridge solve succeeded for all modules, False otherwise.
    """
    was_training = model.training
    prev_use_sindy = model.use_sindy
    prev_ridge_mode = model.ridge_mode

    model.eval(use_sindy=False)
    input_state_buffer, _, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    model._ridge_solve_success = True

    with torch.no_grad():
        model.ridge_mode = True
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()
        initial_state = {s: t.clone() for s, t in input_state_buffer.items()}
        model(xs_flat.to(model.device), initial_state)

    success = model._ridge_solve_success

    model.ridge_mode = prev_ridge_mode
    if was_training:
        model.train(use_sindy=prev_use_sindy)
    else:
        model.eval(use_sindy=prev_use_sindy)

    return success


def _ridge_recalibrate_sindy(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_reconditioning_epochs: int = 3,
):
    """
    Ridge-recalibrate SINDy coefficients after a pruning event, then
    run a short SGD reconditioning phase to warm-start the optimizer.

    Steps:
        1. Reinitialize SINDy coefficients (respecting current presence mask)
        2. Clear SINDy optimizer state (Adam momentum/variance)
        3. Freeze RNN, collect state buffers, ridge-solve for optimal coefficients
        4. One-shot gradient seeding + N epochs of pure SINDy SGD to
           warm-start the optimizer for the new coefficient landscape

    Args:
        model: RNN model with updated sindy_coefficients_presence masks
        xs_train: 5D training data (E, B, T, W, F)
        ys_train: 5D training targets
        optimizer: optimizer (SINDy param group state will be cleared)
        n_reconditioning_epochs: number of pure SINDy SGD epochs after ridge
            solve to warm-start the optimizer (default: 3)
    """
    # 1. Reinitialize SINDy coefficient values (respecting current presence mask)
    for module in model.submodules_rnn:
        presence = model.sindy_coefficients_presence[module]
        model.sindy_coefficients[module].data = (
            torch.randn_like(model.sindy_coefficients[module].data) * 0.001 * presence.float()
        )

    # 2. Clear SINDy optimizer state (Adam momentum/variance)
    for param in optimizer.param_groups[0]['params']:
        if param in optimizer.state:
            optimizer.state[param] = {}

    # 3. Freeze RNN, run forward to collect state buffers, then ridge solve
    was_training = model.training
    prev_use_sindy = model.use_sindy
    prev_ridge_mode = model.ridge_mode
    prev_fit_sindy = model.fit_sindy

    model.eval(use_sindy=False)
    input_state_buffer, target_state_buffer, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    # Ridge solve (temporarily enable ridge_mode to trigger lstsq path in call_module)
    with torch.no_grad():
        model.ridge_mode = True
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()
        initial_state = {s: t.clone() for s, t in input_state_buffer.items()}
        model(xs_flat.to(model.device), initial_state)

    # 4. SGD reconditioning: warm-start optimizer at the new coefficient landscape
    if n_reconditioning_epochs > 0:
        model.ridge_mode = False
        model.fit_sindy = False  # disable internal sindy_loss accumulation
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        criterion = nn.MSELoss()
        for epoch in range(n_reconditioning_epochs):
            optimizer.zero_grad()
            iter_prev_state = {s: t.clone() for s, t in input_state_buffer.items()}
            _, pred_state = model(xs_flat.to(model.device), prev_state=iter_prev_state)
            loss = sum(
                criterion(pred_state[s], target_state_buffer[s])
                for s in model.spice_config.states_in_logit
            )
            loss.backward()
            # Null RNN param grads so the optimizer only updates SINDy coefficients
            # (avoids toggling requires_grad which causes torch.compile recompilation)
            for param in optimizer.param_groups[1]['params']:
                param.grad = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # One-shot gradient seeding on first epoch: initialize Adam state
            # from the current gradient so momentum/variance start reasonable
            if epoch == 0:
                for param in optimizer.param_groups[0]['params']:
                    if param.grad is not None:
                        optimizer.state[param] = {
                            'step': torch.tensor(10.0),
                            'exp_avg': param.grad.clone(),
                            'exp_avg_sq': (param.grad ** 2).clone(),
                        }

            optimizer.step()

    # 5. Restore model state
    model.ridge_mode = prev_ridge_mode
    model.fit_sindy = prev_fit_sindy

    if was_training:
        model.train(use_sindy=prev_use_sindy)
    else:
        model.eval(use_sindy=prev_use_sindy)


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


def _run_shooting_epoch(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    xs_train: torch.Tensor,
    state_trajectories: dict,
    nan_mask: torch.Tensor,
    window_starts: list,
    K: int,
    batch_sessions: torch.Tensor,
    sindy_alpha: float = None,
) -> float:
    """Run one epoch of multi-step shooting training.

    Args:
        model: Model in SINDy training mode (use_sindy=True, RNN frozen)
        optimizer: SINDy coefficient optimizer
        xs_train: 5D training data (E, B, T, W, F)
        state_trajectories: Dict[state_key -> (W, E, B, T+1, I)]
        nan_mask: (B, T) boolean validity mask
        window_starts: List of trial indices where shooting windows begin
        K: Shooting window size
        batch_sessions: Session indices for this epoch (tensor)
        sindy_alpha: L1 penalty strength (None or 0 = disabled)

    Returns:
        Mean loss over all windows
    """
    T = nan_mask.shape[1]
    loss_epoch = 0.0
    n_batches = 0
    window_perm = torch.randperm(len(window_starts))

    for w_idx in window_perm:
        t_start = window_starts[w_idx.item()]
        t_end = min(t_start + K, T)
        K_actual = t_end - t_start

        model.zero_grad()

        current_state = {
            s: state_trajectories[s][:, :, batch_sessions, t_start].to(model.device)
            for s in state_trajectories
        }

        loss_window = torch.tensor(0.0, device=model.device)
        n_valid_steps = 0

        for k in range(K_actual):
            t_idx = t_start + k
            valid = nan_mask[batch_sessions, t_idx].to(model.device)
            if not valid.any():
                continue

            xs_step = xs_train[:, batch_sessions, t_idx:t_idx + 1].to(model.device)
            _, next_state = model(xs_step, current_state)

            step_loss = torch.tensor(0.0, device=model.device)
            for s_key in model.spice_config.states_in_logit:
                target = state_trajectories[s_key][:, :, batch_sessions, t_idx + 1].to(model.device)
                pred = next_state[s_key]
                mask = valid.view(1, 1, -1, 1).expand_as(pred)
                diff = (pred - target) ** 2
                step_loss = step_loss + (diff * mask).sum() / mask.sum().clamp(min=1)

            loss_window = loss_window + step_loss
            n_valid_steps += 1
            current_state = next_state

        if n_valid_steps > 0:
            loss_window = loss_window / n_valid_steps

            if sindy_alpha is not None and sindy_alpha > 0:
                loss_window = loss_window + model.compute_weighted_coefficient_penalty(sindy_alpha=sindy_alpha, norm=1)

            loss_window.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for module in model.get_modules():
                    model.sindy_coefficients[module].data *= model.sindy_coefficients_presence[module]

            loss_epoch += loss_window.item()
            n_batches += 1

    return loss_epoch / max(n_batches, 1)


def _run_shooting_epoch_vectorized(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    xs_train: torch.Tensor,
    state_trajectories: dict,
    nan_mask: torch.Tensor,
    window_starts: list,
    K: int,
    batch_sessions: torch.Tensor,
    sindy_alpha: float = None,
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
        sindy_alpha: L1 penalty strength (None or 0 = disabled)

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
            if sindy_alpha is not None and sindy_alpha > 0:
                total_loss = total_loss + model.compute_weighted_coefficient_penalty(sindy_alpha=sindy_alpha, norm=1)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for module in model.get_modules():
                    model.sindy_coefficients[module].data *= model.sindy_coefficients_presence[module]

        return total_loss.item()

    return 0.0


def _run_sindy_training(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    xs_train_original: torch.Tensor = None,
    ys_train_original: torch.Tensor = None,
    epochs: int = 1000,
    n_warmup_steps: int = 100,
    sindy_alpha: float = None,
    sindy_pruning_frequency: int = None,
    sindy_ensemble_pruning: float = None,
    sindy_ensemble_pruning_mode: str = 'ci',
    sindy_threshold_pruning: float = None,
    sindy_pruning_terms: int = None,
    shooting_steps: int = 20,
    sindy_ridge: bool = True,
    verbose: bool = True,
    ):

    """
    Two-phase SINDy refit on frozen RNN hidden states.

    Stage 2.1 — Sparsity discovery:
        Reset all presence masks to fully active, re-initialize coefficients,
        and train with one-step-ahead shooting (K=1) + pruning + L1 penalty.
        Uses the same LR adaptation as Stage 1: warmup at 0.01, then drop to
        0.001, boost back to 0.01 after pruning events.

    Stage 2.2 — Coefficient estimation:
        Freeze the discovered sparsity pattern, re-initialize coefficients
        within the support, and fit using multi-step shooting (K=shooting_steps)
        without pruning or L1 penalty. Same LR schedule (warmup 0.01 -> 0.001).

    Stage 2.1 uses bootstrapped data. Stage 2.2 optionally re-computes
    trajectories on the full (non-bootstrapped) dataset and averages them
    across ensemble members to create consensus targets.

    Args:
        model: Trained RNN model with SINDy coefficients
        xs_train: 5D training data (E, B, T, W, F) — bootstrapped
        ys_train: 5D training targets — bootstrapped
        xs_train_original: 4D original training data (B, T, W, F) before
            bootstrapping. If provided and E > 1, Stage 2.2 re-computes
            trajectories on full data with ensemble-averaged targets.
        ys_train_original: 4D original training targets
        epochs: Training epochs per stage (default: 1000)
        n_warmup_steps: Warmup epochs per stage (default: 100)
        sindy_alpha: Degree-weighted L1 penalty strength for Stage 2.1
        sindy_pruning_frequency: Epochs between pruning events in Stage 2.1
        sindy_ensemble_pruning: CI test alpha level for Stage 2.1
        sindy_threshold_pruning: Minimum effect size delta for Stage 2.1
        shooting_steps: Rollout horizon K for Stage 2.2. (default: 20)
        verbose: Print progress

    Returns:
        BaseModel: Model with refitted SINDy coefficients
    """
    _E, B, T, W, F = xs_train.shape
    E = model.ensemble_size

    # Collect RNN state trajectories (shared across both stages)
    model.eval(use_sindy=False)
    state_trajectories, nan_mask = _vectorize_state_sequential(model, xs_train, ys_train, verbose=verbose)

    # LR constants (same schedule as Stage 1)
    lr_base = 0.001
    lr_warmup = 0.01
    lr_post_pruning = 0.01
    pruning_boost_duration = max(1, int(0.1 * sindy_pruning_frequency)) if sindy_pruning_frequency else 0

    has_sparsity = (sindy_ensemble_pruning is not None
                    or sindy_threshold_pruning is not None)

    # ── Stage 2.1: Sparsity discovery (K=1, with pruning) ────────────────
    if has_sparsity:
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "=" * terminal_width)
            print("Stage 2.1: SINDy sparsity discovery")
            print("=" * terminal_width)

        # Reset presence masks and re-initialize coefficients for full exploration
        for module in model.get_modules():
            model.sindy_coefficients_presence[module].fill_(True)
            model.sindy_pruning_patience_counters[module].zero_()
            model.sindy_coefficients[module].data = (
                torch.randn_like(model.sindy_coefficients[module].data) * 0.001
            )

        sindy_parameters = [p for name, p in model.named_parameters() if 'sindy' in name]
        optimizer_21 = torch.optim.AdamW(sindy_parameters, lr=lr_warmup, weight_decay=0)

        # Build K=1 shooting windows (= every trial is its own window)
        window_starts_k1 = list(range(T))
        K_21 = 1

        model.fit_sindy = False
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        lr_boost_end = 0  # epoch at which post-pruning LR boost expires

        batch_size_sessions = B
        while True:
            try:
                pbar = tqdm(range(epochs))
                for epoch in pbar:
                    session_perm = torch.randperm(B)

                    # LR schedule: warmup -> base, boost after pruning
                    if epoch == n_warmup_steps:
                        for pg in optimizer_21.param_groups:
                            pg['lr'] = lr_base
                    if lr_boost_end > 0 and epoch >= lr_boost_end:
                        for pg in optimizer_21.param_groups:
                            pg['lr'] = lr_base
                        lr_boost_end = 0

                    loss_epoch = 0.0
                    n_batches = 0

                    for b_start in range(0, B, batch_size_sessions):
                        b_end = min(b_start + batch_size_sessions, B)
                        batch_sessions = session_perm[b_start:b_end]
                        loss_e = _run_shooting_epoch_vectorized(
                            model=model,
                            optimizer=optimizer_21,
                            xs_train=xs_train,
                            state_trajectories=state_trajectories,
                            nan_mask=nan_mask,
                            window_starts=window_starts_k1,
                            K=K_21,
                            batch_sessions=batch_sessions,
                            sindy_alpha=sindy_alpha,
                        )
                        loss_epoch += loss_e
                        n_batches += 1

                    if n_batches > 0:
                        loss_epoch /= n_batches

                    pbar.set_postfix(
                        loss=f"{loss_epoch:.7f}",
                        n_params=f"{model.count_sindy_coefficients().mean():.2f}+/-{model.count_sindy_coefficients().std():.2f}",
                    )

                    # Pruning
                    if sindy_pruning_frequency is not None:
                        if ((sindy_ensemble_pruning is None or model.ensemble_size==1)
                            and sindy_threshold_pruning is not None
                            and epoch >= n_warmup_steps
                            ):
                            model.sindy_coefficient_patience(threshold=sindy_threshold_pruning)

                        if (epoch % sindy_pruning_frequency == 0 or epoch == 1) and epoch >= n_warmup_steps:
                            pruned = False
                            if sindy_ensemble_pruning is not None and model.ensemble_size > 1:
                                model, pruned = _ensemble_pruning(
                                    model=model,
                                    sindy_ensemble_pruning=sindy_ensemble_pruning,
                                    sindy_threshold_pruning=sindy_threshold_pruning,
                                    n_terms_pruning=sindy_pruning_terms,
                                    verbose=verbose,
                                    sindy_ensemble_pruning_mode=sindy_ensemble_pruning_mode,
                                )
                            elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                                model.sindy_coefficient_pruning(patience=sindy_pruning_frequency, n_terms_pruning=sindy_pruning_terms)
                                pruned = True

                            if pruned and pruning_boost_duration > 0:
                                for pg in optimizer_21.param_groups:
                                    pg['lr'] = lr_post_pruning
                                lr_boost_end = epoch + pruning_boost_duration

                break
            except KeyboardInterrupt:
                if verbose:
                    print('\nStage 2.1 interrupted. Continuing...')
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if _check_cuda_oom(e):
                    raise
                if batch_size_sessions <= 1:
                    raise RuntimeError(f"Automatic batch size probing was unsuccessful. Current batch size is {batch_size_sessions} but could still not be started. Please try again with a smaller ensemble size (current: {model.ensemble_size}).")
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                batch_size_sessions = max(1, batch_size_sessions // 2)

    # ── Stage 2.2: Coefficient estimation (K=shooting_steps, no pruning) ──
    if verbose:
        terminal_width = _get_terminal_width()
        print("\n" + "=" * terminal_width)
        if shooting_steps > 1:
            print(f"Stage 2.2: SINDy coefficient estimation (multi-step shooting, K={shooting_steps})")
        else:
            print("Stage 2.2: SINDy coefficient estimation (one-step-ahead)")
        print("=" * terminal_width)

    # Switch to full (non-bootstrapped) data with ensemble-averaged targets
    if xs_train_original is not None and E > 1:
        if verbose:
            print("Re-computing state trajectories on full data (ensemble-averaged)...")
        xs_full_5d = xs_train_original.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
        ys_full_5d = ys_train_original.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
        model.eval(use_sindy=False)
        state_trajectories, nan_mask = _vectorize_state_sequential(
            model, xs_full_5d, ys_full_5d, verbose=verbose
        )
        # Average over ensemble → consensus targets
        for s in state_trajectories:
            state_trajectories[s] = state_trajectories[s].mean(
                dim=1, keepdim=True
            ).expand(-1, E, -1, -1, -1).contiguous()
        xs_train = xs_full_5d
        ys_train = ys_full_5d
        B = xs_train.shape[1]

    K = shooting_steps

    # Build shooting windows (shared by ridge evaluation and SGD)
    if K > 1:
        n_windows = T // K
        window_starts = [i * K for i in range(n_windows)]
        if T % K > 0 and T > K:
            window_starts.append(T - K)
        elif T % K > 0 and T <= K:
            window_starts = [0]
            K = T
    else:
        window_starts = list(range(T))

    # Re-initialize coefficients within discovered support
    for module in model.get_modules():
        model.sindy_coefficients[module].data = (
            torch.randn_like(model.sindy_coefficients[module].data) * 0.001
            * model.sindy_coefficients_presence[module].float()
        )

    # ── Ridge regression (closed-form one-step solve) ──
    ridge_success = False
    if sindy_ridge:
        ridge_success = _ridge_solve_sindy(model, xs_train, ys_train)
        if ridge_success:
            sgd_epochs = min(200, epochs)
            # Evaluate ridge solution with K-step shooting loss
            model.fit_sindy = False
            model.train(use_sindy=True)
            for rnn_module in model.submodules_rnn.values():
                rnn_module.eval()
            with torch.no_grad():
                ridge_loss = _run_shooting_epoch_vectorized(
                    model=model,
                    optimizer=None,
                    xs_train=xs_train,
                    state_trajectories=state_trajectories,
                    nan_mask=nan_mask,
                    window_starts=window_starts,
                    K=K,
                    batch_sessions=torch.arange(B),
                    sindy_alpha=None,
                )
            if verbose:
                print(f"Ridge regression succeeded (K={K} loss: {ridge_loss:.7f}). Running SGD refinement...")
        else:
            # Re-initialize coefficients since ridge may have partially written
            for module in model.get_modules():
                model.sindy_coefficients[module].data = (
                    torch.randn_like(model.sindy_coefficients[module].data) * 0.001
                    * model.sindy_coefficients_presence[module].float()
                )
            if verbose:
                print("Ridge regression failed. Falling back to full SGD...")

    if not ridge_success:
        sgd_epochs = epochs

    # ── SGD shooting (refinement after ridge, or full fallback) ──
    sgd_lr_init = lr_base if ridge_success else lr_warmup
    sindy_parameters = [p for name, p in model.named_parameters() if 'sindy' in name]
    optimizer_22 = torch.optim.AdamW(sindy_parameters, lr=sgd_lr_init, weight_decay=0)

    model.fit_sindy = False
    model.train(use_sindy=True)
    for rnn_module in model.submodules_rnn.values():
        rnn_module.eval()

    batch_size_sessions = B
    while True:
        try:
            pbar = tqdm(range(sgd_epochs))
            for epoch in pbar:
                session_perm = torch.randperm(B)

                # LR schedule: warmup -> base (skip if ridge provided init)
                if not ridge_success and epoch == n_warmup_steps:
                    for pg in optimizer_22.param_groups:
                        pg['lr'] = lr_base

                loss_epoch = 0.0
                n_batches = 0

                for b_start in range(0, B, batch_size_sessions):
                    b_end = min(b_start + batch_size_sessions, B)
                    batch_sessions = session_perm[b_start:b_end]
                    loss_e = _run_shooting_epoch_vectorized(
                        model=model,
                        optimizer=optimizer_22,
                        xs_train=xs_train,
                        state_trajectories=state_trajectories,
                        nan_mask=nan_mask,
                        window_starts=window_starts,
                        K=K,
                        batch_sessions=batch_sessions,
                        sindy_alpha=None,  # unpenalized
                    )
                    loss_epoch += loss_e
                    n_batches += 1

                if n_batches > 0:
                    loss_epoch /= n_batches

                pbar.set_postfix(
                    loss=f"{loss_epoch:.7f}",
                    n_params=f"{model.count_sindy_coefficients().mean():.2f}+/-{model.count_sindy_coefficients().std():.2f}",
                    K=K,
                )
            break
        except KeyboardInterrupt:
            if verbose:
                print('\nStage 2.2 interrupted. Continuing...')
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _check_cuda_oom(e):
                raise
            if batch_size_sessions <= 1:
                raise RuntimeError(f"Automatic batch size probing was unsuccessful. Current batch size is {batch_size_sessions} but could still not be started. Please try again with a smaller ensemble size (current: {model.ensemble_size}).")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch_size_sessions = max(1, batch_size_sessions // 2)

    model.fit_sindy = True
    return model
    
    # return model.to(original_device)


def _run_joint_training(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    dataset_test: SpiceDataset,
    optimizer: torch.optim.Optimizer,

    epochs: int = 1,
    batch_size: int = None,
    n_warmup_steps: int = 0,
    n_steps: int = None,
    loss_fn: callable = cross_entropy_loss,
    loss_fn_kwargs: dict = {},

    lr_warmup_factor: float = 10.,
    lr_boost_factor_rnn: float = 1.,
    lr_boost_factor_sindy: float = 10.,
    lr_boost_duration_frac: float = 0.1,

    sindy_weight: float = 0,
    sindy_alpha: float = 0,
    sindy_pruning_frequency: int = None,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_ensemble_pruning_mode: str = 'ci',
    sindy_population_pruning: float = None,
    sindy_pruning_terms: int = None,
    sindy_reconditioning_epochs: int = 3,

    convergence_threshold: float = 0,
    verbose: bool = False,
    keep_log: bool = False,
    path_save_checkpoints: str = None,
) -> Tuple[BaseModel, torch.optim.Optimizer, float, float, float]:
    """
    Joint RNN-SINDy optimization with ensemble pruning.

    Trains RNN to predict behavior while SINDy regularization pushes
    dynamics toward sparse equations. Periodic pruning via minimum-effect
    CI test: a term survives iff |mean| - t_crit * SE > delta, combining
    statistical significance and practical significance in one test.
    Optional participant filtering for cross-participant consistency.
    After each pruning event, SINDy coefficients are ridge-recalibrated.

    Objective: L_total = L_CE(y, ŷ) + λ_sindy * L_SINDy + α * L_penalty

    Returns:
        Tuple of (model, optimizer, loss_train, loss_test_rnn, loss_test_sindy)
    """
    
    B_total = xs_train.shape[1]
    iterations_per_epoch = max(B_total, 64) // batch_size if batch_size < max(B_total, 64) else 1
    
    dataloader_test = None
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))

    warmup_scaler_sindy_weight = _setup_warmup_scaler(n_warmup_steps=n_warmup_steps, exp_max=5)
    lr_scheduler = SpiceLRScheduler(
        optimizer=optimizer,
        n_warmup_steps=n_warmup_steps,
        warmup_factor=lr_warmup_factor,
        boost_factor_rnn=lr_boost_factor_rnn,
        boost_factor_sindy=lr_boost_factor_sindy,
        boost_duration_frac=lr_boost_duration_frac,
        sindy_pruning_frequency=sindy_pruning_frequency if sindy_pruning_frequency else 100,
    )

    # Handle zero epochs case
    # if epochs == 0:
    #     if verbose:
    #         print('No training epochs specified. Model will not be trained.')
    #     return model, optimizer, 0., 0.

    # Training state
    continue_training = True
    converged = False
    n_calls_to_train_model = 0
    convergence_value = 1
    last_loss = 1
    recency_factor = 0.5
    len_last_print = 0
    t_start_total = time.time()
    loss_train = 0
    loss_test_rnn = None
    loss_test_sindy = None
    is_notebook = _is_notebook()
    
    # Main training loop
    while continue_training:
        try:  # try because of possibility for manual early stopping via keyboard interrupt
            # --- Learning rate adaptation ---
            lr_scheduler.step(n_calls_to_train_model)

            if epochs > 0:
                loss_train = 0
                t_start = time.time()

                # Compute warmup-scaled SINDy weight
                if n_calls_to_train_model >= n_warmup_steps:
                    sindy_weight_epoch = sindy_weight
                else:
                    sindy_weight_epoch = sindy_weight * warmup_scaler_sindy_weight[n_calls_to_train_model]

                # Training iterations for this epoch
                for _ in range(iterations_per_epoch):
                    # Manual batching along session dim (dim 1) of 5D data
                    if batch_size < B_total:
                        batch_idx = torch.randperm(B_total, device=xs_train.device)[:batch_size]
                        xs = xs_train[:, batch_idx]
                        ys = ys_train[:, batch_idx]
                    else:
                        xs = xs_train
                        ys = ys_train

                    if xs.device != model.device:
                        xs = xs.to(model.device)
                        ys = ys.to(model.device)

                    model, optimizer, loss_i = _run_batch_training(
                        model=model,
                        xs=xs,
                        ys=ys,
                        optimizer=optimizer,
                        n_steps=n_steps,
                        sindy_weight=sindy_weight_epoch,
                        sindy_alpha=sindy_alpha,
                        loss_fn=loss_fn,
                        loss_fn_kwargs=loss_fn_kwargs,
                    )
                    loss_train += loss_i

                n_calls_to_train_model += 1
                loss_train /= iterations_per_epoch

            # Validation (test data is 4D, unsqueeze to 5D for _run_batch_training)
            if dataloader_test is not None:
                model = model.eval(use_sindy=False)
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    if xs.device != model.device:
                        xs = xs.to(model.device)
                        ys = ys.to(model.device)
                    _, _, loss_test_rnn = _run_batch_training(model=model, xs=xs.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), ys=ys.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), loss_fn=loss_fn, loss_fn_kwargs=loss_fn_kwargs)
                    
                if sindy_weight > 0:
                    model = model.eval(use_sindy=True)
                    with torch.no_grad():
                        xs, ys = next(iter(dataloader_test))
                        if xs.device != model.device:
                            xs = xs.to(model.device)
                            ys = ys.to(model.device)
                        _, _, loss_test_sindy = _run_batch_training(model=model, xs=xs.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), ys=ys.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), loss_fn=loss_fn, loss_fn_kwargs=loss_fn_kwargs)

                model = model.train()
            
            # Unified pruning event with ridge recalibration
            if (sindy_weight > 0
                and sindy_pruning_frequency is not None
                # and n_calls_to_train_model >= n_warmup_steps
                ):

                if ((sindy_ensemble_pruning is None or model.ensemble_size==1)
                    and sindy_threshold_pruning is not None
                    and n_calls_to_train_model >= n_warmup_steps
                    ):
                    # Fallback: per-epoch patience tracking for per-member threshold pruning
                    model.sindy_coefficient_patience(threshold=sindy_threshold_pruning)

                
                if (n_calls_to_train_model % sindy_pruning_frequency == 0
                    or n_calls_to_train_model == 1
                    # and n_calls_to_train_model >= n_warmup_steps
                    ):
                    
                    # pruning
                    if n_calls_to_train_model >= n_warmup_steps:       
                        
                        # Ridge solve before pruning: snap coefficients to MSE-optimal
                        # so pruning decisions are based on clean estimates
                        # _ridge_solve_sindy(model, xs_train, ys_train)
                        
                        pruned = False
                        if (sindy_ensemble_pruning is not None 
                            and model.ensemble_size > 1
                            ):
                            model, pruned = _ensemble_pruning(
                                model=model,
                                sindy_ensemble_pruning=sindy_ensemble_pruning,
                                sindy_threshold_pruning=sindy_threshold_pruning,
                                n_terms_pruning=sindy_pruning_terms,
                                verbose=verbose,
                                sindy_ensemble_pruning_mode=sindy_ensemble_pruning_mode,
                                )

                        elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                            # Fallback: per-member threshold pruning only (no ensemble test)
                            model.sindy_coefficient_pruning(patience=sindy_pruning_frequency, n_terms_pruning=sindy_pruning_terms)
                            pruned = True

                        # Reset coefficients + optimizer state, ridge solve + reconditioning
                        # if pruned:
                        #     _ridge_recalibrate_sindy(model, xs_train, ys_train, optimizer, n_reconditioning_epochs=sindy_reconditioning_epochs)

                        # LR boost after pruning event
                        if pruned:
                            lr_scheduler.notify_pruning(n_calls_to_train_model)

            # Check convergence
            dloss = last_loss - (loss_test_rnn if dataloader_test is not None else loss_train)
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = loss_test_rnn if dataloader_test is not None else loss_train

            # Save checkpoint
            # if path_save_checkpoints and n_calls_to_train_model == save_at_epoch:
            #     torch.save(model.state_dict(), path_save_checkpoints.replace('.', f'_ep{n_calls_to_train_model}.'))
            #     save_at_epoch *= 2

            # Display training status
            if verbose:
                len_last_print = _print_training_status(
                    len_last_print=len_last_print,
                    model=model,
                    n_calls=n_calls_to_train_model,
                    epochs=epochs,
                    loss_train=loss_train,
                    loss_test_rnn=loss_test_rnn if dataloader_test is not None else None,
                    loss_test_sindy=loss_test_sindy if dataloader_test is not None else None,
                    time_elapsed=time.time() - t_start_total,
                    convergence_value=convergence_value,
                    sindy_weight=sindy_weight,
                    scheduler=lr_scheduler,
                    warmup_steps=n_warmup_steps,
                    converged=converged,
                    finished=not continue_training,
                    keep_log=keep_log,
                    is_notebook=is_notebook,
                )

        except KeyboardInterrupt:
            continue_training = False
            if verbose:
                print('\nTraining interrupted. Continuing with further operations...')

    return model, optimizer, loss_train, loss_test_rnn, loss_test_sindy


def _minimum_effect_ci_test(
    coefficients: torch.Tensor,
    presence: torch.Tensor,
    alpha: float = 0.05,
    delta: float = 0.0,
) -> torch.Tensor:
    """
    Minimum-effect confidence interval test on absolute coefficient values.

    Tests whether the ensemble mean of |coefficient| is confidently above
    delta: mean(|coeff|) - t_crit * SE(|coeff|) > delta.

    Operating on absolute values makes the test sign-agnostic: a term where
    half the ensemble learns +0.3 and half learns -0.3 will survive (all
    members agree the magnitude is non-zero), whereas testing raw values
    would cancel them out and incorrectly prune the term.

    Pruned coefficients are treated as zero estimates, so terms only found by
    a few ensemble members are naturally penalized.

    Args:
        coefficients: [E, P, X, terms] raw coefficient values
        presence: [E, P, X, terms] boolean presence mask
        alpha: confidence level (default: 0.05)
        delta: minimum effect size threshold (default: 0.0)

    Returns:
        [P, X, terms] boolean mask — True where term passes the CI test
    """

    effective_coeffs = (coefficients * presence.float()).detach().abs()
    E = effective_coeffs.shape[0]

    mean = effective_coeffs.mean(dim=0)                # [P, X, terms]
    std = effective_coeffs.std(dim=0, correction=1)    # [P, X, terms]
    se = std / (E ** 0.5)

    t_critical = t_dist.ppf(1 - alpha / 2, df=E - 1)

    # CI lower bound for mean(|coeff|) must exceed delta
    ci_lower = mean - t_critical * se
    significant = ci_lower > delta

    # Require at least 2 active ensemble members for a valid test
    n_active = presence.float().sum(dim=0)
    significant = significant & (n_active >= 2)

    return significant


def _ensemble_ratio_test(
    coefficients: torch.Tensor,
    presence: torch.Tensor,
    threshold: float = 0.0,
    ratio: float = 0.6,
) -> torch.Tensor:
    """
    Ensemble ratio test: a term survives iff a sufficient fraction of
    ensemble members have |coefficient| > threshold.

    Args:
        coefficients: [E, P, X, terms] raw coefficient values
        presence: [E, P, X, terms] boolean presence mask
        threshold: minimum absolute coefficient value for a member to
                   count as supporting the term (default: 0.0)
        ratio: minimum fraction of ensemble members that must exceed
               the threshold for the term to survive (default: 0.6)

    Returns:
        [P, X, terms] boolean mask — True where term passes the ratio test
    """
    effective_coeffs = (coefficients * presence.float()).detach()
    E = effective_coeffs.shape[0]

    # Count how many ensemble members exceed threshold per (P, X, term)
    n_above = (effective_coeffs.abs() > threshold).float().sum(dim=0)  # [P, X, terms]

    # Term survives if ratio of members above threshold >= required ratio
    significant = (n_above / E) >= ratio

    return significant


def _compute_ensemble_masks(
    model: BaseModel,
    test_fn: callable = _minimum_effect_ci_test,
    verbose: bool = True,
    **test_fn_kwargs,
) -> dict:
    """
    Per-participant ensemble confidence filtering.

    For each (participant, experiment, term), applies a statistical test
    across ensemble members to determine if the term is robustly identified.

    Args:
        model: trained model with SINDy coefficients
        test_fn: statistical test function with signature
                 (coefficients [E,P,X,T], presence [E,P,X,T], **kwargs) -> [P,X,T] bool
        verbose: print filtering results
        **test_fn_kwargs: keyword arguments passed to test_fn

    Returns:
        Dict mapping module names to [P, X, terms] boolean masks
    """
    ensemble_masks = {}

    if verbose:
        print("Ensemble confidence filtering:")

    for module in model.submodules_rnn:
        coeffs = model.sindy_coefficients[module].detach()
        presence = model.sindy_coefficients_presence[module]

        mask = test_fn(coeffs, presence, **test_fn_kwargs)
        ensemble_masks[module] = mask

        if verbose:
            n_before = presence.any(dim=0).sum().item()
            n_after = mask.sum().item()
            total = mask.numel()
            print(f"\t{module}: {n_before} -> {n_after} / {total} (participant, experiment, term) slots")

    return ensemble_masks


def _compute_participant_masks(
    ensemble_masks: dict,
    confidence_threshold: float,
    n_participants: int,
    n_experiments: int,
    verbose: bool = True,
) -> dict:
    """
    Cross-participant confidence filtering on (optionally ensemble-filtered) masks.
    
    A term passes if it is present in at least
    (confidence_threshold * n_participants * n_experiments) participant-experiment slots.

    Args:
        ensemble_masks: {module: [P, X, terms] bool} per-participant masks
        confidence_threshold: required fraction of (P * X) slots (0-1)
        n_participants: number of participants
        n_experiments: number of experiments
        verbose: print filtering results

    Returns:
        Dict mapping module names to [terms] boolean masks
    """
    confidence_masks = {}
    min_occurrences = int(confidence_threshold * n_participants * n_experiments)

    if verbose:
        print(f"Participant confidence filtering (threshold={confidence_threshold}, min_occurrences={min_occurrences}):")

    for module, mask in ensemble_masks.items():
        # mask: [P, X, terms] -> sum across P and X -> [terms]
        participant_presence = mask.float().sum(dim=0).sum(dim=0)  # (terms,)
        global_mask = participant_presence >= min_occurrences  # (terms,)
        # Intersect: term must be significant for this participant AND common enough globally
        confidence_masks[module] = mask & global_mask.unsqueeze(0).unsqueeze(0)  # (P, X, terms)

        if verbose:
            n_before = mask.sum().item()
            n_after = confidence_masks[module].sum().item()
            n_global = global_mask.sum().item()
            print(f"\t{module}: {n_before} -> {n_after} (participant, experiment, term) slots ({n_global} global terms)")

    return confidence_masks


def _compute_pruning_masks(
    model: BaseModel,
    ensemble_alpha: float = None,
    ensemble_delta: float = 0.0,
    participant_threshold: float = None,
    ensemble_test_fn: callable = _minimum_effect_ci_test,
    ensemble_test_kwargs: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Two-level confidence filtering: ensemble filtering -> participant filtering.

    Args:
        model: trained model with SINDy coefficients
        ensemble_alpha: confidence level for ensemble CI test (None to skip)
        ensemble_delta: minimum effect size for ensemble CI test (default: 0.0)
        participant_threshold: required fraction of (P * X) slots (None to skip)
        ensemble_test_fn: statistical test function for ensemble filtering
        ensemble_test_kwargs: keyword arguments passed to the test function.
            If None, defaults to {'alpha': ensemble_alpha, 'delta': ensemble_delta}
            for backward compatibility with the CI test.
        verbose: print filtering results

    Returns:
        Dict mapping module names to [P, X, terms] boolean masks
    """
    # Step 1: Ensemble filtering (per participant)
    if ensemble_alpha is not None or ensemble_test_kwargs is not None:
        if ensemble_test_kwargs is None:
            ensemble_test_kwargs = dict(alpha=ensemble_alpha, delta=ensemble_delta)
        ensemble_masks = _compute_ensemble_masks(
            model, test_fn=ensemble_test_fn, verbose=verbose,
            **ensemble_test_kwargs,
        )
    else:
        # No ensemble filtering — term is present for (P, X) if any ensemble member has it
        ensemble_masks = {
            module: model.sindy_coefficients_presence[module].any(dim=0)
            for module in model.submodules_rnn
        }

    # Step 2: Participant filtering (global)
    if participant_threshold is not None and participant_threshold > 0:
        confidence_masks = _compute_participant_masks(
            ensemble_masks, participant_threshold, model.n_participants, model.n_experiments, verbose=verbose
        )
    else:
        # No participant filtering — keep terms that pass ensemble filter for any (P, X)
        # confidence_masks = {module: mask.any(dim=1).any(dim=1).unsqueeze(1).unsqueeze(1) for module, mask in ensemble_masks.items()}
        confidence_masks = ensemble_masks
        
    return confidence_masks


def fit_spice(
    model: BaseModel,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset = None,
    optimizer: torch.optim.Optimizer = None,

    epochs: int = 1,
    batch_size: int = None,
    n_steps: int = None,
    convergence_threshold: float = 1e-7,
    loss_fn: callable = cross_entropy_loss,
    loss_fn_kwargs: dict = {},

    lr_warmup_factor: float = 10.,
    lr_boost_factor_rnn: float = 1.,
    lr_boost_factor_sindy: float = 10.,
    lr_boost_duration_frac: float = 0.1,
    
    sindy_weight: float = 0.,
    sindy_alpha: float = 0.,
    sindy_pruning_frequency: int = 1,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_ensemble_pruning_mode: str = 'ci',
    sindy_population_pruning: float = None,
    sindy_pruning_terms: int = None,
    sindy_reconditioning_epochs: int = 3,
    sindy_refit: bool = True,
    sindy_ridge: bool = True,
    sindy_shooting_steps: int = 20,

    verbose: bool = True,
    keep_log: bool = False,
    n_warmup_steps: int = 0,
    path_save_checkpoints: str = None,
) -> Tuple[BaseModel, torch.optim.Optimizer, float]:
    """
    Two-stage SPICE training pipeline with ensemble pruning.

    Stage 1 (Joint Training with Minimum-Effect CI Pruning):
        Train RNN + SINDy jointly with L_CE + λ_sindy * L_SINDy + α * L_penalty.
        Periodic pruning via minimum-effect CI test: a term survives iff
        |mean| - t_crit * SE > delta, where delta = sindy_threshold_pruning.
        This unifies statistical significance and practical significance in
        a single test, avoiding the cascade where per-member threshold pruning
        erodes ensemble consensus. Terms must fail 2 consecutive pruning events
        before permanent removal. After each pruning event, SINDy coefficients
        are ridge-recalibrated and the optimizer is reconditioned.

    Stage 2 (Final SINDy Refit):
        Freeze RNN weights and refit SINDy coefficients via ridge solve on
        stable hidden states. Single-pass: solve → prune → refit.

    Args:
        model: RNN model with SINDy integration
        dataset_train: Training dataset
        dataset_test: Validation dataset (optional)
        optimizer: PyTorch optimizer
        epochs: Total training epochs
        batch_size: Training batch size (None = auto-detect max via GPU probing, int = fixed)
        n_steps: BPTT truncation length
        convergence_threshold: Early stopping threshold
        loss_fn: Loss function for behavioral prediction
        lr_warmup_factor: RNN LR multiplier at start of training (default 10).
            LR linearly decays from warmup_factor * base_lr to base_lr over
            n_warmup_steps. Set to 1.0 to disable warmup.
        lr_boost_factor_rnn: RNN LR multiplier after pruning (default 1.0 = no boost)
        lr_boost_factor_sindy: SINDy LR multiplier after pruning (default 10.0)
        lr_boost_duration_frac: Fraction of sindy_pruning_frequency for boost
            duration (default 0.1)
        sindy_weight: λ_sindy regularization strength
        sindy_alpha: Degree-weighted L1 penalty strength
        sindy_threshold_pruning: Minimum effect size (delta) for the CI test.
            When sindy_ensemble_pruning is set, this serves as the delta threshold.
            When sindy_ensemble_pruning is None, falls back to per-member hard
            thresholding. (None or 0 to disable; default: None)
        sindy_pruning_frequency: Epochs between pruning events
        sindy_ensemble_pruning: Confidence level for ensemble CI test (e.g. 0.05),
            or minimum ensemble ratio for ratio test (e.g. 0.6).
            Primary pruning mechanism. None to disable.
        sindy_ensemble_pruning_mode: Ensemble pruning strategy. 'ci' for
            minimum-effect CI test (default), 'ratio' for ensemble ratio test
            (prune if fewer than sindy_ensemble_pruning fraction of members
            have |coeff| > sindy_threshold_pruning).
        sindy_population_pruning: Optional participant presence threshold (0-1).
            None to disable.
        sindy_reconditioning_epochs: Number of pure SINDy SGD epochs after each
            ridge recalibration to warm-start the optimizer. Uses one-shot
            gradient seeding on the first epoch. (default: 3, 0 to disable)
        sindy_shooting_steps: Number of consecutive trials to roll out in SINDy
            mode during Stage 2 coefficient fitting before computing loss.
            1 = one-step-ahead. Values > 1 enable multi-step shooting which
            penalizes error accumulation and produces more stable autoregressive
            rollouts. (default: 20)
        verbose: Print progress
        keep_log: Keep full training log (vs. live update)
        n_warmup_steps: Warmup epochs for SINDy weight (no pruning during warmup)
        path_save_checkpoints: Path for saving checkpoints

    Returns:
        Tuple of (trained_model, optimizer)
    """

    if n_warmup_steps is None:
        if epochs is not None:
            n_warmup_steps = epochs // 4
        else:
            n_warmup_steps = 0

    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n" + status_lines)
        print("SPICE Training Configuration:")
        if epochs > 0:
            print("\tSPICE-RNN training: [x]")
        else:
            print("\tSPICE-RNN training: [ ]")
        if epochs > 0 and sindy_weight > 0:
            print("\tSINDy regularization: [x]")
        else:
            print("\tSINDy regularization: [ ]")
        # Pruning details
        pruning_details = []
        if sindy_ensemble_pruning is not None:
            if sindy_ensemble_pruning_mode == 'ratio':
                pruning_details.append(f"ratio test ratio={sindy_ensemble_pruning}")
            else:
                pruning_details.append(f"CI test alpha={sindy_ensemble_pruning}")
            if sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                pruning_details.append(f"delta={sindy_threshold_pruning}")
        elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
            pruning_details.append(f"threshold={sindy_threshold_pruning} (per-member)")
        if sindy_population_pruning is not None and sindy_population_pruning > 0:
            pruning_details.append(f"participant threshold={sindy_population_pruning}")
        if pruning_details:
            print(f"\tPruning (every {sindy_pruning_frequency} epochs): {', '.join(pruning_details)}")
        else:
            print("\tPruning: [ ]")
        if sindy_refit:
            print("\tSINDy refit: [x]")
        else:
            print("\tSINDy refit: [ ]")
        print(status_lines)

    # Bootstrap training data once: 4D (B, T, W, F) -> 5D (E, B, T, W, F)
    E = model.ensemble_size
    B = dataset_train.xs.shape[0]
    if E > 1:
        bootstrap_indices = torch.randint(0, B, (E, B))
        xs_train_5d = dataset_train.xs[bootstrap_indices]
        ys_train_5d = dataset_train.ys[bootstrap_indices]
    else:
        xs_train_5d = dataset_train.xs.unsqueeze(0)
        ys_train_5d = dataset_train.ys.unsqueeze(0)

    # Auto-compute sindy_pruning_terms: distribute total terms evenly across
    # available pruning events so the model can reach 0 coefficients within
    # (epochs - warmup) epochs.
    if sindy_pruning_terms is None and sindy_weight > 0 and sindy_pruning_frequency is not None:
        total_terms = sum(model.sindy_coefficients[m].shape[-1] for m in model.submodules_rnn)
        n_pruning_events = max(1, (epochs - n_warmup_steps) // max(1, sindy_pruning_frequency))
        sindy_pruning_terms = math.ceil(total_terms / n_pruning_events)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Joint RNN-SINDy Training with Fused Pruning
    # ══════════════════════════════════════════════════════════════════════════
    if epochs > 0:
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "=" * terminal_width)
            if sindy_weight > 0:
                print("Stage 1: SPICE joint training (RNN+SINDy)")
            else:
                print("Stage 1: SPICE-RNN training (without SINDy-regularization)")
            print("=" * terminal_width)
            
        if batch_size is None:
            batch_size = xs_train_5d.shape[1]
        
        while True:    
            try:
                results = _run_joint_training(
                    model=model,
                    optimizer=optimizer,
                    xs_train=xs_train_5d,
                    ys_train=ys_train_5d,
                    dataset_test=dataset_test,

                    epochs=epochs,
                    n_warmup_steps=n_warmup_steps,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    convergence_threshold=convergence_threshold,
                    loss_fn=loss_fn,
                    loss_fn_kwargs=loss_fn_kwargs,

                    lr_warmup_factor=lr_warmup_factor,
                    lr_boost_factor_rnn=lr_boost_factor_rnn,
                    lr_boost_factor_sindy=lr_boost_factor_sindy,
                    lr_boost_duration_frac=lr_boost_duration_frac,
                    
                    sindy_weight=sindy_weight,
                    sindy_alpha=sindy_alpha,
                    sindy_threshold_pruning=sindy_threshold_pruning,
                    sindy_pruning_frequency=sindy_pruning_frequency,
                    sindy_ensemble_pruning=sindy_ensemble_pruning,
                    sindy_ensemble_pruning_mode=sindy_ensemble_pruning_mode,
                    sindy_pruning_terms=sindy_pruning_terms,
                    sindy_population_pruning=sindy_population_pruning,
                    sindy_reconditioning_epochs=sindy_reconditioning_epochs,

                    verbose=verbose,
                    keep_log=keep_log,
                    path_save_checkpoints=path_save_checkpoints,
                )
                model, optimizer, loss_train, loss_test_rnn, loss_test_sindy = results
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if _check_cuda_oom(e):
                    raise
                if batch_size <= 1:
                    raise RuntimeError(f"Automatic batch size probing was unsuccessful. Current batch size is {batch_size} but could still not be started. Please try again with a smaller ensemble size (current: {model.ensemble_size}).")
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Final SINDy Refit
    # ══════════════════════════════════════════════════════════════════════════
    if sindy_refit:
        # Save Stage 1 model checkpoint before Stage 2 overwrites coefficients
        if path_save_checkpoints is not None:
            stage1_path = path_save_checkpoints.replace('.pkl', '_stage1.pkl')
        elif hasattr(model, '_save_path') and model._save_path is not None:
            stage1_path = model._save_path.replace('.pkl', '_stage1.pkl')
        else:
            stage1_path = None
        if stage1_path is not None:
            os.makedirs(os.path.dirname(stage1_path) or '.', exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sindy_coefficients_presence': model.sindy_coefficients_presence,
            }, stage1_path)
            if verbose:
                print(f"\nStage 1 model saved to: {stage1_path}")

        _run_sindy_training(
            model=model,
            xs_train=xs_train_5d.to(torch.device('cpu')),
            ys_train=ys_train_5d.to(torch.device('cpu')),
            xs_train_original=dataset_train.xs,
            ys_train_original=dataset_train.ys,
            epochs=1000,
            n_warmup_steps=100,
            sindy_alpha=sindy_alpha,
            sindy_pruning_frequency=sindy_pruning_frequency,
            sindy_ensemble_pruning=sindy_ensemble_pruning,
            sindy_ensemble_pruning_mode=sindy_ensemble_pruning_mode,
            sindy_threshold_pruning=sindy_threshold_pruning,
            sindy_pruning_terms=sindy_pruning_terms,
            shooting_steps=sindy_shooting_steps,
            sindy_ridge=sindy_ridge,
            verbose=verbose,
        )
        
        
    # ══════════════════════════════════════════════════════════════════════════
    # Final evaluation summary
    # ══════════════════════════════════════════════════════════════════════════
    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n" + status_lines)
        print("Losses:")
        batch_size = xs_train_5d.shape[1]
        
        if dataset_test is not None:
            with torch.no_grad():
                _, _, _, loss_train_rnn, loss_train_sindy = _run_joint_training(
                    model=model,
                    optimizer=optimizer,
                    xs_train=xs_train_5d,
                    ys_train=ys_train_5d,
                    dataset_test=dataset_train,

                    epochs=0,
                    n_warmup_steps=999,
                    batch_size=batch_size,
                    convergence_threshold=0,
                    n_steps=n_steps,
                    loss_fn=loss_fn,
                    loss_fn_kwargs=loss_fn_kwargs,

                    sindy_weight=1,
                    sindy_alpha=0,
                    sindy_threshold_pruning=None,
                    sindy_pruning_frequency=None,
                    sindy_ensemble_pruning=None,
                    sindy_population_pruning=None,

                    verbose=False,
                    keep_log=False,
                    path_save_checkpoints=None,
                )
                
                _, _, _, loss_test_rnn, loss_test_sindy = _run_joint_training(
                    model=model,
                    optimizer=optimizer,
                    xs_train=xs_train_5d,
                    ys_train=ys_train_5d,
                    dataset_test=dataset_test,

                    epochs=0,
                    n_warmup_steps=999,
                    batch_size=batch_size,
                    convergence_threshold=0,
                    n_steps=n_steps,
                    loss_fn=loss_fn,
                    loss_fn_kwargs=loss_fn_kwargs,

                    sindy_weight=1,
                    sindy_alpha=0,
                    sindy_threshold_pruning=None,
                    sindy_pruning_frequency=None,
                    sindy_ensemble_pruning=None,
                    sindy_population_pruning=None,

                    verbose=False,
                    keep_log=False,
                    path_save_checkpoints=None,
                )

            msg_result = "\t         Training    Validation"
            msg_result += f"\n\tRNN      {loss_train_rnn:.5f}     {loss_test_rnn:.5f}"
            msg_result += f"\n\tSINDy    {loss_train_sindy:.5f}     {loss_test_sindy:.5f}"
            
        print(msg_result)
        print(status_lines)

    return model.eval(use_sindy=True), optimizer

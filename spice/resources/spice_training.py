
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        terminal_width = 80
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
        if isinstance(scheduler, (ReduceLROnPlateau, ReduceOnPlateauWithRestarts)):
            postfix_parts['Metric'] = f'{scheduler.best:.7f}'
            postfix_parts['BadEp'] = f'{scheduler.num_bad_epochs}/{scheduler.patience}'

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


class ReduceOnPlateauWithRestarts:
    """
    Plateau-based LR scheduler with restarts for RNN parameters only.

    When the learning rate hits min_lr and stays there for patience
    epochs without improvement, it resets to the base learning rate.
    SINDy coefficients (param_groups[0]) are not affected.
    """
    def __init__(self, optimizer, min_lr, factor, patience, restart: bool):
        """
        Parameters:
        - optimizer: Optimizer instance.
        - min_lr: The minimum learning rate after reductions.
        - factor: Multiplicative factor to reduce the LR on plateau.
        - patience: Number of epochs with no improvement before reducing/restarting LR.
        """
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.base_lr_rnn = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
        self.factor = factor
        self.patience = patience
        self.restart = restart
        
        self.best = float('inf')
        self.num_bad_epochs = 0
        self.num_cycles_completed = 0
        self.increase_lr = False

    def step(self, metrics):
        """
        Update the learning rate based on the validation loss.
        Only affects RNN parameters (param_groups[1]).
        """
        if metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
            self.increase_lr = False
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            current_lr = self._get_rnn_lr()
            if current_lr <= self.min_lr and self.restart:
                self.increase_lr = True
            elif current_lr >= self.base_lr_rnn and self.increase_lr:
                self.increase_lr = False
            self._adjust_lr()
            self.num_bad_epochs = 0

    def _get_rnn_lr(self):
        """Get current RNN learning rate."""
        if len(self.optimizer.param_groups) > 1:
            return self.optimizer.param_groups[1]['lr']
        return self.optimizer.param_groups[0]['lr']

    def _adjust_lr(self):
        """Adjust the learning rate for RNN parameters only (param_groups[1])."""
        if len(self.optimizer.param_groups) > 1:
            param_group = self.optimizer.param_groups[1]
        else:
            param_group = self.optimizer.param_groups[0]

        old_lr = param_group['lr']
        # determine whether to increase or decrease lr
        factor = 1/self.factor if self.increase_lr else self.factor
        new_lr = max(old_lr * factor, self.min_lr)
        param_group['lr'] = new_lr

    def _restart_lr(self):
        """Restart the learning rate for RNN parameters back to base LR."""
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = self.base_lr_rnn
        else:
            self.optimizer.param_groups[0]['lr'] = self.base_lr_rnn
        self.num_cycles_completed += 1

    def get_lr(self):
        """Retrieve the current learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """Retrieve the last computed learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]


def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Wrapper for torch's cross entropy loss which does all the reshaping when getting SpiceDataset.ys tensors as predicitons and targets."""
    n_actions = target.shape[-1]
    
    prediction = prediction.reshape(-1, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)
    
    return torch.nn.functional.cross_entropy(prediction, target)


def _setup_warmup_scaler(n_warmup_steps: int, exp_max: float = 1) -> torch.Tensor:
    """Create exponential warmup scaler for SINDy weight."""
    if n_warmup_steps <= 0:
        return None
    warmup_scaler = torch.exp(torch.linspace(0, exp_max, n_warmup_steps))
    warmup_scaler = (warmup_scaler - warmup_scaler.min()) / (warmup_scaler.max() - warmup_scaler.min()) + 1e-4
    return warmup_scaler


def _setup_lr_scheduler(optimizer: torch.optim.Optimizer):
    # Initialize learning rate scheduler
    return ReduceOnPlateauWithRestarts(
        optimizer=optimizer,
        min_lr=1e-5,
        factor=0.1,
        patience=100,
        restart=True,
    )
    

def _run_batch_training(
    model: BaseModel,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    sindy_weight: float = 0.,
    sindy_alpha: float = 0.,
    n_steps: int = None,
    loss_fn: callable = cross_entropy_loss,
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
        ys_pred, _ = model(xs_step, state, batch_first=True)

        # Mask out padding (NaN values)
        # xs_step is 5D: (E, B, T_out, T_in, F)
        mask = ~torch.isnan(xs_step[..., :model.n_actions].sum(dim=(-1)))
        ys_pred = ys_pred[mask]
        ys_step = ys_step[mask]

        loss_step = loss_fn(ys_pred, ys_step)

        if torch.is_grad_enabled():
            # Add SINDy regularization loss
            if sindy_weight > 0 and model.sindy_loss != 0:
                loss_step = loss_step + sindy_weight * model.sindy_loss

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
                                    fill_value=model.spice_config.memory_state[s],
                                    dtype=torch.float32, device=model.device) for s in state_keys}
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
                updated_state = model(xs_sub, model.get_state(), batch_first=True)[1]

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
):
    
    pruned = False
    
    # Unified minimum-effect CI test: threshold serves as delta
    confidence_masks = _compute_pruning_masks(
        model,
        ensemble_alpha=sindy_ensemble_pruning,
        ensemble_delta=sindy_threshold_pruning or 0.0,
        participant_threshold=None,
        verbose=verbose,
    )
    for module in model.submodules_rnn:
        mask = confidence_masks[module].to(model.device)  # (P, X, terms)
        still_active = model.sindy_coefficients_presence[module].any(dim=0)  # (P, X, terms)
        failed = ~mask & still_active

        # Update patience: increment on failure, reset on success
        counters = model.sindy_pruning_patience_counters[module]
        failed_e = failed.unsqueeze(0).expand_as(counters)
        counters.data = torch.where(failed_e, counters + 1, torch.zeros_like(counters))

        # Prune terms that failed 2+ consecutive pruning events
        prune = (counters[0] >= 2)  # (P, X, terms)
        if prune.any():
            prune_e = prune.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1)
            model.sindy_coefficients_presence[module].data &= ~prune_e
            model.sindy_coefficients[module].data *= model.sindy_coefficients_presence[module].float()
            counters.data *= (~prune_e).int()
            pruned = True

    return model, pruned


def _ridge_solve_sindy(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
):
    """
    Lightweight ridge solve: snap SINDy coefficients to their MSE-optimal
    values without touching the optimizer state.

    Used before pruning decisions so the CI test / threshold check evaluates
    coefficients at their dynamics-optimal values rather than SGD-noisy ones.
    """
    was_training = model.training
    prev_use_sindy = model.use_sindy
    prev_rnn_training_finished = model.rnn_training_finished

    model.eval(use_sindy=False, aggregate=False)
    input_state_buffer, _, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    with torch.no_grad():
        model.rnn_training_finished = True
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()
        prev_state_args = dict(prev_state={s: t.clone() for s, t in input_state_buffer.items()})
        model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)

    model.rnn_training_finished = prev_rnn_training_finished
    if was_training:
        model.train(use_sindy=prev_use_sindy)
    else:
        model.eval(use_sindy=prev_use_sindy)


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
    prev_rnn_training_finished = model.rnn_training_finished
    prev_fit_sindy = model.fit_sindy

    model.eval(use_sindy=False, aggregate=False)
    input_state_buffer, target_state_buffer, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    # Ridge solve (temporarily set rnn_training_finished=True to trigger lstsq path)
    with torch.no_grad():
        model.rnn_training_finished = True
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()
        prev_state_args = dict(prev_state={s: t.clone() for s, t in input_state_buffer.items()})
        model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)

    # 4. SGD reconditioning: warm-start optimizer at the new coefficient landscape
    if n_reconditioning_epochs > 0:
        model.rnn_training_finished = False
        model.fit_sindy = False  # disable internal sindy_loss accumulation
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        criterion = nn.MSELoss()
        for epoch in range(n_reconditioning_epochs):
            optimizer.zero_grad()
            iter_prev_state = {s: t.clone() for s, t in input_state_buffer.items()}
            _, pred_state = model(xs_flat.to(model.device), prev_state=iter_prev_state, batch_first=True)
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
    model.rnn_training_finished = prev_rnn_training_finished
    model.fit_sindy = prev_fit_sindy

    if was_training:
        model.train(use_sindy=prev_use_sindy)
    else:
        model.eval(use_sindy=prev_use_sindy)


def _run_sindy_training(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    epochs: int = 1000,
    n_warmup_steps: int = 0,
    sindy_alpha: float = None,
    sindy_pruning_frequency: int = None,
    sindy_ensemble_pruning: float = None,
    sindy_threshold_pruning: float = None,
    verbose: bool = True,
    ):
    
    """
    Final SINDy refit: freeze RNN weights and refit SINDy coefficients
    on the trained RNN hidden states via batched MSE 
    old: via ridge solve → prune → refit.

    Args:
        model (BaseModel): Trained RNN model with SINDy coefficients
        dataset_train: Training dataset (4D)
        epochs (int): Number of epochs
        batch_size (int): Batch size for training
        verbose (bool): Print progress

    Returns:
        BaseModel: Model with refitted SINDy coefficients
    """
    
    criterion = nn.MSELoss()

    # re-initialize sindy coefficients with fitted presence mask
    for module in model.get_modules():
        model.sindy_coefficients[module].data = torch.randn_like(model.sindy_coefficients[module].data) * 0.001 * model.sindy_coefficients_presence[module]

    # setup optimizer
    sindy_parameters = []
    for name, p in model.named_parameters():
        if 'sindy' in name:
            sindy_parameters.append(p)
    optimizer = torch.optim.AdamW(sindy_parameters, lr=0.01, weight_decay=0)

    # Vectorize training data using shared helper
    model.eval(use_sindy=False, aggregate=False)
    input_state_buffer_train, target_state_buffer_train, xs_flat, _ = _vectorize_state(model, xs_train, ys_train, verbose=verbose)
    model.fit_sindy = False  # disable internal sindy_loss accumulation
    model.train(use_sindy=True)
    for rnn_module in model.submodules_rnn.values():
        rnn_module.eval()

    # State buffers: (W=1, E, N, I); xs_flat: (E, N, 1, 1, F) — sample dim is 2 / 1
    n_samples_vectorized = input_state_buffer_train[list(input_state_buffer_train.keys())[0]].shape[2]
    batch_size = n_samples_vectorized

    while True:
        try:  # batch size probing
            pbar = tqdm(range(epochs))
            for epoch in pbar:
                # shuffle along sample dimension (deferred to batch indexing to avoid full-dataset gather)
                shuffle_index = torch.randperm(n_samples_vectorized)

                loss_epoch = 0
                for index_batch in range(0, n_samples_vectorized, batch_size):
                    
                    model.zero_grad()

                    batch_idx = shuffle_index[index_batch:index_batch+batch_size]
                    batch_prev_state = {s: t[:, :, batch_idx] for s, t in input_state_buffer_train.items()}
                    batch_target_state = {s: t[:, :, batch_idx] for s, t in target_state_buffer_train.items()}
                    batch_xs_flat = xs_flat[:, batch_idx].to(model.device)

                    _, pred_state = model(batch_xs_flat, prev_state=batch_prev_state, batch_first=True)

                    loss_batch = 0
                    
                    # prediction loss
                    for s in model.spice_config.states_in_logit:
                        loss_batch += criterion(pred_state[s], batch_target_state[s])
                        
                    # sparsity loss
                    if sindy_alpha is not None:
                        loss_batch += model.compute_weighted_coefficient_penalty(sindy_alpha=sindy_alpha, norm=1)
                    
                    loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # safety check: keep pruned coefficients zero
                    with torch.no_grad():
                        for module in model.get_modules():
                            model.sindy_coefficients[module].data *= model.sindy_coefficients_presence[module]

                    loss_epoch += loss_batch.item()
                    
                pbar.set_postfix(loss=f"{loss_epoch:.7f}", n_params=f"{model.count_sindy_coefficients().mean():.2f}+/-{model.count_sindy_coefficients().std():.2f}")
                
                # Unified pruning event with ridge recalibration
                if (
                    sindy_pruning_frequency is not None
                    # and n_calls_to_train_model >= n_warmup_steps
                    ):

                    if (sindy_ensemble_pruning is None 
                        and sindy_threshold_pruning is not None
                        and epoch >= n_warmup_steps
                        ):
                        # Fallback: per-epoch patience tracking for per-member threshold pruning
                        model.sindy_coefficient_patience(threshold=sindy_threshold_pruning)

                    
                    if (epoch % sindy_pruning_frequency == 0
                        or epoch == 1
                        # and n_calls_to_train_model >= n_warmup_steps
                        ):
                        
                        # pruning
                        if epoch >= n_warmup_steps:       
                            
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
                                    verbose=verbose,
                                    )

                            elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                                # Fallback: per-member threshold pruning only (no ensemble test)
                                model.sindy_coefficient_pruning(patience=sindy_pruning_frequency)
                                pruned = True
            break
        except KeyboardInterrupt:
            if verbose:
                print('\nTraining interrupted. Continuing with further operations...')
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if _check_cuda_oom(e):
                raise
            if batch_size <= 1:
                raise RuntimeError(f"Automatic batch size probing was unsuccessful. Current batch size is {batch_size} but could still not be started. Please try again with a smaller ensemble size (current: {model.ensemble_size}).")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            
    model.fit_sindy = True
    return model

    # old version: RIDGE SOLVE -> was very unstable; unpredictable singular matrix + worse accuracy + small residual terms 
    
    # Disable dropout in RNN submodules so the solve target is deterministic
    # for rnn_module in model.submodules_rnn.values():
    #     rnn_module.eval()
    
    # disable compilation to move to cpu
    # original_device = model.device
    # # Disable compilation before device transfer to avoid
    # # CUDA→CPU dispatch key recompilation storm
    # for rnn_module in model.submodules_rnn.values():
    #     rnn_module._compile = False
    # model.to('cpu')
    
    # # Step 1: Solve (respects current presence mask)
    # prev_state_args = dict(prev_state={s: t.clone() for s, t in input_state_buffer_train.items()})
    # with torch.no_grad():
    #     model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)

    # # Step 2: Prune small terms
    # model.sindy_coefficient_patience(threshold=pruning_threshold)
    # model.sindy_coefficient_pruning(patience=1)
    
    # # Step 3: Refit with surviving terms only - pure lstsq
    # sindy_alpha = 0
    # sindy_alpha += model.sindy_alpha
    # model.sindy_alpha = 0
    # with torch.no_grad():
    #     model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)
    # model.sindy_alpha += sindy_alpha
    
    # # Evaluate
    # model.eval(use_sindy=True)
    # with torch.no_grad():
    #     _, state_pred = model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)
    # # model.train(use_sindy=True)
    # # for rnn_module in model.submodules_rnn.values():
    # #     rnn_module.eval()
    
    # loss = 0
    # for s in model.spice_config.states_in_logit:
    #     loss += criterion(state_pred[s], target_state_buffer_train[s]).item()

    # len_last_print = _print_training_status(
    #     len_last_print=len_last_print,
    #     model=model,
    #     n_calls=1,
    #     epochs=1,
    #     loss_train=loss,
    #     loss_test_rnn=None,
    #     loss_test_sindy=None,
    #     time_elapsed=time.time()-t_start,
    #     convergence_value=None,
    #     sindy_weight=1,
    #     scheduler=None,
    #     keep_log=DEBUG_MODE,
    # )
    
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
    use_scheduler: bool = False,
    loss_fn: callable = cross_entropy_loss,

    sindy_weight: float = 0,
    sindy_alpha: float = 0,
    sindy_pruning_frequency: int = None,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_population_pruning: float = None,
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
    lr_scheduler = _setup_lr_scheduler(optimizer=optimizer) if use_scheduler else None
    
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
                    _, _, loss_test_rnn = _run_batch_training(model=model, xs=xs.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), ys=ys.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), loss_fn=loss_fn)
                    
                if sindy_weight > 0:
                    model = model.eval(use_sindy=True)
                    with torch.no_grad():
                        xs, ys = next(iter(dataloader_test))
                        if xs.device != model.device:
                            xs = xs.to(model.device)
                            ys = ys.to(model.device)
                        _, _, loss_test_sindy = _run_batch_training(model=model, xs=xs.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), ys=ys.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), loss_fn=loss_fn)

                model = model.train()
            
            # Unified pruning event with ridge recalibration
            if (sindy_weight > 0
                and sindy_pruning_frequency is not None
                # and n_calls_to_train_model >= n_warmup_steps
                ):

                if (sindy_ensemble_pruning is None 
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
                                verbose=verbose,
                                )

                        elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                            # Fallback: per-member threshold pruning only (no ensemble test)
                            model.sindy_coefficient_pruning(patience=sindy_pruning_frequency)
                            pruned = True

                        # Reset coefficients + optimizer state, ridge solve + reconditioning
                        # if pruned:
                        #     _ridge_recalibrate_sindy(model, xs_train, ys_train, optimizer, n_reconditioning_epochs=sindy_reconditioning_epochs)

            # Check convergence
            dloss = last_loss - (loss_test_rnn if dataloader_test is not None else loss_train)
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = loss_test_rnn if dataloader_test is not None else loss_train

            # Update learning rate scheduler
            if lr_scheduler is not None and n_calls_to_train_model >= n_warmup_steps:
                metric = loss_test_rnn if dataloader_test is not None else loss_train
                lr_scheduler.step(metric)

            # Save checkpoint
            if path_save_checkpoints and n_calls_to_train_model == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace('.', f'_ep{n_calls_to_train_model}.'))
                save_at_epoch *= 2

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
    Minimum-effect confidence interval test across ensemble members.

    Tests whether the confidence interval for the ensemble mean coefficient
    lies entirely outside the negligible zone [-delta, delta]. A term
    survives iff: |mean| - t_crit * SE > delta.

    When delta=0, equivalent to a standard two-tailed t-test for non-zero mean.

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

    effective_coeffs = (coefficients * presence.float()).detach()
    E = effective_coeffs.shape[0]

    mean = effective_coeffs.mean(dim=0)                # [P, X, terms]
    std = effective_coeffs.std(dim=0, correction=1)    # [P, X, terms]
    se = std / (E ** 0.5)

    t_critical = t_dist.ppf(1 - alpha / 2, df=E - 1)

    # CI lower bound for |mean| must exceed delta
    ci_lower = mean.abs() - t_critical * se
    significant = ci_lower > delta

    # Require at least 2 active ensemble members for a valid test
    n_active = presence.float().sum(dim=0)
    significant = significant & (n_active >= 2)

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
        verbose: print filtering results

    Returns:
        Dict mapping module names to [P, X, terms] boolean masks
    """
    # Step 1: Ensemble filtering (per participant)
    if ensemble_alpha is not None:
        ensemble_masks = _compute_ensemble_masks(
            model, test_fn=ensemble_test_fn, verbose=verbose,
            alpha=ensemble_alpha, delta=ensemble_delta,
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
    scheduler: bool = False,
    n_steps: int = None,
    convergence_threshold: float = 1e-7,
    loss_fn: callable = cross_entropy_loss,

    sindy_weight: float = 0.,
    sindy_alpha: float = 0.,
    sindy_pruning_frequency: int = 1,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_population_pruning: float = None,
    sindy_reconditioning_epochs: int = 3,
    sindy_refit: bool = True,
    
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
        scheduler: Enable learning rate scheduler
        n_steps: BPTT truncation length
        convergence_threshold: Early stopping threshold
        loss_fn: Loss function for behavioral prediction
        sindy_weight: λ_sindy regularization strength
        sindy_alpha: Degree-weighted L1 penalty strength
        sindy_threshold_pruning: Minimum effect size (delta) for the CI test.
            When sindy_ensemble_pruning is set, this serves as the delta threshold.
            When sindy_ensemble_pruning is None, falls back to per-member hard
            thresholding. (None or 0 to disable; default: None)
        sindy_pruning_frequency: Epochs between pruning events
        sindy_ensemble_pruning: Confidence level for ensemble CI test (e.g. 0.05).
            Primary pruning mechanism. None to disable.
        sindy_population_pruning: Optional participant presence threshold (0-1).
            None to disable.
        sindy_reconditioning_epochs: Number of pure SINDy SGD epochs after each
            ridge recalibration to warm-start the optimizer. Uses one-shot
            gradient seeding on the first epoch. (default: 3, 0 to disable)
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
                    use_scheduler=scheduler,
                    batch_size=batch_size,
                    convergence_threshold=convergence_threshold,
                    loss_fn=loss_fn,
                    
                    sindy_weight=sindy_weight,
                    sindy_alpha=sindy_alpha,
                    sindy_threshold_pruning=sindy_threshold_pruning,
                    sindy_pruning_frequency=sindy_pruning_frequency,
                    sindy_ensemble_pruning=sindy_ensemble_pruning,
                    sindy_population_pruning=sindy_population_pruning,
                    sindy_reconditioning_epochs=sindy_reconditioning_epochs,

                    verbose=verbose,
                    keep_log=keep_log,
                    path_save_checkpoints=path_save_checkpoints,
                )
                model, optimizer, loss_train, loss_test_rnn, loss_test_sindy = results
                model.rnn_training_finished = True
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
        # try:
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "=" * terminal_width)
            print("Stage 2: SINDy refit")
            print("=" * terminal_width)

        _run_sindy_training(
            model=model,
            xs_train=xs_train_5d.to(torch.device('cpu')),
            ys_train=ys_train_5d.to(torch.device('cpu')),
            epochs=1000,
            n_warmup_steps=n_warmup_steps,
            sindy_pruning_frequency=sindy_pruning_frequency,
            sindy_threshold_pruning=sindy_threshold_pruning,
            sindy_ensemble_pruning=sindy_ensemble_pruning,
            sindy_alpha=sindy_alpha,
            verbose=verbose,
        )
        # for rnn_module in model.submodules_rnn.values():
        #     rnn_module._compile = True
            
        # except torch._C._LinAlgError as e:
        #     model.to(original_device)
        #     for rnn_module in model.submodules_rnn.values():
        #         rnn_module._compile = True
        #     print('Stage 2: Ridge solve is not possible because a singular candidate term matrix was found. SPICE will omit a ridge solve and return Stage 1 SINDy coefficients instead. No worries---the SPICE model is still great!')
        
        
    # ══════════════════════════════════════════════════════════════════════════
    # Final evaluation summary
    # ══════════════════════════════════════════════════════════════════════════
    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n" + status_lines)
        print("Training results:")
        msg = "\t"

        if epochs > 0:
            msg += f"L(Train, RNN): {loss_train:.7f}"
            msg += "\n\t"

        batch_size = xs_train_5d.shape[1]
        
        if dataset_test is not None:
            with torch.no_grad():
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
                    use_scheduler=False,
                    loss_fn=loss_fn,

                    sindy_weight=sindy_weight,
                    sindy_alpha=0,
                    sindy_threshold_pruning=None,
                    sindy_pruning_frequency=None,
                    sindy_ensemble_pruning=None,
                    sindy_population_pruning=None,

                    verbose=False,
                    keep_log=False,
                    path_save_checkpoints=None,
                )

            msg += f"L(Val, RNN):   {loss_test_rnn:.7f}"
            if sindy_weight > 0:
                msg += "\n\t"
                msg += f"L(Val, SINDy): {loss_test_sindy:.7f}"

        print(msg)
        print(status_lines)

    return model.eval(use_sindy=True), optimizer

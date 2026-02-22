
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple
import shutil
from scipy.stats import t as t_dist
from torch.nn.functional import mse_loss  # using standard mse loss for spice should be fine most of the time

from .rnn import BaseRNN
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


def _print_training_status(
    len_last_print: int,
    model: BaseRNN,
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
):
    """Print live-updating training status block."""
    
    # Build comprehensive status display
    terminal_width = _get_terminal_width()
    status_lines = []
    status_lines.append("=" * terminal_width)
    
    # Training progress line
    progress_msg = f'Epoch {n_calls}/{epochs} --- L(Train): {loss_train:.7f}'
    progress_msg += f' --- L(Val, RNN): {loss_test_rnn:.7f}' if loss_test_rnn is not None else ''
    progress_msg += f' --- L(Val, SINDy): {loss_test_sindy:.7f}' if loss_test_sindy is not None else ''
    progress_msg += f' --- Time: {time_elapsed:.2f}s;' if time_elapsed is not None else ''
    progress_msg += f' --- Convergence: {convergence_value:.2e}' if convergence_value is not None else ''
    
    if scheduler is not None:
        progress_msg += f'; LR: {scheduler.get_last_lr()[-1]:.2e}'
        if isinstance(scheduler, (ReduceLROnPlateau, ReduceOnPlateauWithRestarts)):
            progress_msg += f"; Metric: {scheduler.best:.7f}; Bad epochs: {scheduler.num_bad_epochs}/{scheduler.patience}"
    
    status_lines.append(progress_msg)
    
    max_len_module = max([len(module) for module in model.get_modules()])
    
    # Add SPICE model equations if SINDy is active
    if sindy_weight > 0:
        status_lines.append("-" * terminal_width)
        status_lines.append(f"SPICE Model (Coefficients: {model.count_sindy_coefficients().mean():.0f}):")
        status_lines.append(model.get_spice_model_string(participant_id=0, ensemble_idx=0))
        
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
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        print(msg, flush=True)
    else:
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
    model: BaseRNN,
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
                coefficient_penalty = model.compute_weighted_coefficient_penalty(sindy_alpha=sindy_alpha, norm=1)
                loss_step = loss_step + sindy_alpha * coefficient_penalty

            # backpropagation
            optimizer.zero_grad()
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_batch += loss_step.item()
        iterations += 1

    return model, optimizer, loss_batch/iterations


def _vectorize_state(
    model: BaseRNN,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
) -> tuple:
    """
    Run the frozen RNN forward on the full training data to collect state buffers
    for SINDy ridge solving.

    Args:
        model: RNN model (should be in eval mode with use_sindy=False)
        xs_train: 5D tensor (E, B, T, W, F)
        ys_train: 5D tensor matching xs_train

    Returns:
        Tuple of (input_state_buffer, target_state_buffer, xs_flat, ys_flat)
    """
    _E, B, T, W, F = xs_train.shape
    n_features_y = ys_train.shape[-1]
    E = model.ensemble_size

    if xs_train.dim() == 4:
        xs_train = xs_train.unsqueeze(0).repeat(E, 1, 1, 1, 1)
        ys_train = ys_train.unsqueeze(0).repeat(E, 1, 1, 1, 1)

    model.init_state(batch_size=B, within_ts=W)

    # State buffers: (within_ts, E, n_trials*B, n_items)
    state_buffer_current = {s: torch.zeros((W, E, T*B, model.n_items), dtype=torch.float32, device=model.device) for s in model.get_state()}
    state_buffer_next = {s: torch.zeros((W, E, T*B, model.n_items), dtype=torch.float32, device=model.device) for s in model.get_state()}

    with torch.no_grad():
        for t in range(T):
            for s in model.get_state():
                state_buffer_current[s][0, :, t*B:(t+1)*B] = model.get_state()[s][-1]

            # 5D input: (E, B, 1, W, F)
            updated_state = model(xs_train[:, :, t:t+1].to(model.device), model.get_state(), batch_first=True)[1]

            for s in model.get_state():
                state_buffer_current[s][1:, :, t*B:(t+1)*B] = updated_state[s][:-1]
                state_buffer_next[s][:, :, t*B:(t+1)*B] = updated_state[s]

    # Flatten to (E, flat_total, 1, 1, F)
    flat_total = W * T * B
    xs_flat = xs_train.permute(0, 2, 1, 3, 4).reshape(E, flat_total, 1, 1, F)
    ys_flat = ys_train.permute(0, 2, 1, 3, 4).reshape(E, flat_total, 1, 1, n_features_y)[0]

    # Reshape state buffers: (W, E, T*B, items) -> (1, E, flat_total, items)
    for s in model.get_state():
        state_buffer_current[s] = state_buffer_current[s].permute(1, 2, 0, 3).reshape(1, E, flat_total, model.n_items)
        state_buffer_next[s] = state_buffer_next[s].permute(1, 2, 0, 3).reshape(1, E, flat_total, model.n_items)

    # Remove NaN-padded samples
    nan_mask = ~torch.isnan(xs_flat[0, :, 0, 0, :model.n_actions].sum(dim=(-1)))
    xs_flat = xs_flat[:, nan_mask]
    state_buffer_current = {s: state_buffer_current[s][:, :, nan_mask] for s in state_buffer_current}
    state_buffer_next = {s: state_buffer_next[s][:, :, nan_mask] for s in state_buffer_next}

    return state_buffer_current, state_buffer_next, xs_flat, ys_flat


def _ridge_recalibrate_sindy(
    model: BaseRNN,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
):
    """
    Reset SINDy coefficients and optimizer state, then ridge-solve to recalibrate
    surviving terms after a pruning event.

    Called during joint training after ensemble t-test / threshold pruning to
    instantly set coefficients to optimal values for the current library structure.

    Args:
        model: RNN model with updated sindy_coefficients_presence masks
        xs_train: 5D training data (E, B, T, W, F)
        ys_train: 5D training targets
        optimizer: optimizer (SINDy param group state will be cleared)
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

    # Freeze RNN parameters
    rnn_grad_state = {}
    for name, param in model.named_parameters():
        rnn_grad_state[name] = param.requires_grad
        param.requires_grad = 'sindy' in name

    model.eval(use_sindy=False)
    input_state_buffer, _, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    with torch.no_grad():
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        prev_state_args = dict(prev_state={s: t.clone() for s, t in input_state_buffer.items()})
        model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)

    # 4. Restore model state
    for name, param in model.named_parameters():
        param.requires_grad = rnn_grad_state[name]

    if was_training:
        model.train(use_sindy=prev_use_sindy)
    else:
        model.eval(use_sindy=prev_use_sindy)


def _run_sindy_training(
    model: BaseRNN,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    pruning_threshold: float = 0.05,
    verbose: bool = True,
    ):
    """
    Final SINDy refit: freeze RNN weights and refit SINDy coefficients
    on the trained RNN hidden states via ridge solve → prune → refit.

    Args:
        model (BaseRNN): Trained RNN model with SINDy coefficients
        dataset_train: Training dataset (4D)
        epochs (int): Number of epochs
        batch_size (int): Batch size for training
        verbose (bool): Print progress

    Returns:
        BaseRNN: Model with refitted SINDy coefficients
    """

    t_start = time.time()
    
    if verbose:
        terminal_width = _get_terminal_width()
        print("\n" + "="*terminal_width)
        print(f"Starting SINDy finetuning...")
        print("="*terminal_width)

    criterion = nn.MSELoss()

    # Freeze all RNN parameters, only train SINDy coefficients
    for name, param in model.named_parameters():
        param.requires_grad = 'sindy' in name

    model.eval(use_sindy=False)

    # Vectorize training data using shared helper
    input_state_buffer_train, target_state_buffer_train, xs_flat, _ = _vectorize_state(model, xs_train, ys_train)

    len_last_print = 0
    
    with torch.no_grad():
        model.train(use_sindy=True)
        # Disable dropout in RNN submodules so the solve target is deterministic
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        prev_state_args = dict(prev_state={s: t.clone() for s, t in input_state_buffer_train.items()})

        # Step 1: Solve (respects current presence mask)
        model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)

        # Step 2: Prune small terms
        model.sindy_coefficient_pruning(threshold=pruning_threshold, patience=1)

        # Step 3: Refit with surviving terms only - pure lstsq
        sindy_alpha = 0
        sindy_alpha += model.sindy_alpha
        model.sindy_alpha = 0
        model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)
        model.sindy_alpha += sindy_alpha
        
        # Evaluate
        model.eval(use_sindy=True)
        _, state_pred = model(inputs=xs_flat.to(model.device), **prev_state_args, batch_first=True)
        # model.train(use_sindy=True)
        # for rnn_module in model.submodules_rnn.values():
        #     rnn_module.eval()
        
        loss = 0
        for s in model.spice_config.states_in_logit:
            loss += criterion(state_pred[s], target_state_buffer_train[s]).item()

        len_last_print = _print_training_status(
            len_last_print=len_last_print,
            model=model,
            n_calls=1,
            epochs=1,
            loss_train=loss,
            loss_test_rnn=None,
            loss_test_sindy=None,
            time_elapsed=time.time()-t_start,
            convergence_value=None,
            sindy_weight=1,
            scheduler=None,
            keep_log=DEBUG_MODE,
        )

    # Restore requires_grad for all parameters
    for param in model.parameters():
        param.requires_grad = True

    return model


def _run_joint_training(
    model: BaseRNN,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    dataset_test: SpiceDataset,
    optimizer: torch.optim.Optimizer,

    epochs: int,
    batch_size: int,
    n_warmup_steps: int,
    n_steps: int,
    use_scheduler: bool,
    loss_fn: callable,

    sindy_weight: float,
    sindy_alpha: float,
    sindy_pruning_frequency: int,
    sindy_threshold_pruning: float,
    sindy_ensemble_pruning: float,
    sindy_population_pruning: float,

    convergence_threshold: float,
    verbose: bool,
    keep_log: bool,
    path_save_checkpoints: str,
) -> Tuple[BaseRNN, torch.optim.Optimizer, float, float, float]:
    """
    Joint RNN-SINDy optimization with fused ensemble pruning.

    Trains RNN to predict behavior while SINDy regularization pushes
    dynamics toward sparse equations. Periodic pruning events combine:
    - Ensemble t-test filtering (primary): cross-member consistency check
    - Optional threshold pruning: per-member hard thresholding
    - Optional participant filtering: cross-participant consistency check
    After each pruning event, SINDy coefficients are reset and ridge-solved
    for instant recalibration.

    Objective: L_total = L_CE(y, ŷ) + λ_sindy * L_SINDy + α * L_penalty

    Returns:
        Tuple of (model, optimizer, loss_train, loss_test_rnn, loss_test_sindy)
    """
    
    # Setup of training components
    # xs_train/ys_train are 5D: (E, B, T, W, F)
    B_total = xs_train.shape[1]
    if batch_size is None:
        batch_size = B_total
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
    loss_train = 0
    loss_test_rnn = None
    loss_test_sindy = None

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
                and n_calls_to_train_model >= n_warmup_steps
                and n_calls_to_train_model % sindy_pruning_frequency == 0
                ):
                # Optional: per-member threshold pruning
                if sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                    model.sindy_coefficient_pruning(threshold=sindy_threshold_pruning, patience=1)

                # Primary: ensemble t-test (+ optional participant filter)
                if sindy_ensemble_pruning is not None:
                    confidence_masks = _compute_pruning_masks(
                        model,
                        ensemble_alpha=sindy_ensemble_pruning,
                        participant_threshold=sindy_population_pruning,
                        verbose=verbose,
                    )
                    for module in model.submodules_rnn:
                        mask = confidence_masks[module].to(model.device)
                        mask = mask.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1)
                        model.sindy_coefficients_presence[module].data &= mask
                        model.sindy_coefficients[module].data *= model.sindy_coefficients_presence[module].float()

                # Reset coefficients + optimizer state, ridge solve for recalibration
                _ridge_recalibrate_sindy(model, xs_train, ys_train, optimizer)

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
                    time_elapsed=time.time() - t_start,
                    convergence_value=convergence_value,
                    sindy_weight=sindy_weight,
                    scheduler=lr_scheduler,
                    warmup_steps=n_warmup_steps,
                    converged=converged,
                    finished=not continue_training,
                    keep_log=keep_log,
                )

        except KeyboardInterrupt:
            continue_training = False
            if verbose:
                print('\nTraining interrupted. Continuing with further operations...')

    return model, optimizer, loss_train, loss_test_rnn, loss_test_sindy


def _ttest_zero_exclusion(
    coefficients: torch.Tensor,
    presence: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Per-(participant, experiment, term) t-test for whether the mean coefficient
    across ensemble members is significantly different from zero.

    Pruned coefficients are treated as zero estimates, so terms only found by
    a few ensemble members are naturally penalized.

    Args:
        coefficients: [E, P, X, terms] raw coefficient values
        presence: [E, P, X, terms] boolean presence mask
        alpha: significance level for two-tailed test (default: 0.05)

    Returns:
        [P, X, terms] boolean mask — True where coefficient is significantly != 0
    """

    effective_coeffs = (coefficients * presence.float()).detach()
    E = effective_coeffs.shape[0]

    mean = effective_coeffs.mean(dim=0)                # [P, X, terms]
    std = effective_coeffs.std(dim=0, correction=1)    # [P, X, terms]
    se = std / (E ** 0.5)

    t_stat = mean.abs() / se.clamp(min=1e-10)
    t_critical = t_dist.ppf(1 - alpha / 2, df=E - 1)

    significant = t_stat > t_critical

    # Require at least 2 active ensemble members for a valid test
    n_active = presence.float().sum(dim=0)
    significant = significant & (n_active >= 2)

    return significant


def _compute_ensemble_masks(
    model: BaseRNN,
    test_fn: callable = _ttest_zero_exclusion,
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
    model: BaseRNN,
    ensemble_alpha: float = None,
    participant_threshold: float = None,
    ensemble_test_fn: callable = _ttest_zero_exclusion,
    verbose: bool = True,
) -> dict:
    """
    Two-level confidence filtering: ensemble filtering -> participant filtering.

    Args:
        model: trained model with SINDy coefficients
        ensemble_alpha: significance level for ensemble t-test (None to skip)
        participant_threshold: required fraction of (P * X) slots (None to skip)
        ensemble_test_fn: statistical test function for ensemble filtering
        verbose: print filtering results

    Returns:
        Dict mapping module names to [terms] boolean masks
    """
    # Step 1: Ensemble filtering (per participant)
    if ensemble_alpha is not None:
        ensemble_masks = _compute_ensemble_masks(
            model, test_fn=ensemble_test_fn, verbose=verbose, alpha=ensemble_alpha
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
    model: BaseRNN,
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

    verbose: bool = True,
    keep_log: bool = False,
    n_warmup_steps: int = 0,
    path_save_checkpoints: str = None,
) -> Tuple[BaseRNN, torch.optim.Optimizer, float]:
    """
    Two-stage SPICE training pipeline with fused ensemble pruning.

    Stage 1 (Joint Training with Hierarchical Pruning):
        Train RNN + SINDy jointly with L_CE + λ_sindy * L_SINDy + α * L_penalty.
        Periodic pruning events combine ensemble t-test filtering (primary) with
        optional threshold pruning and participant filtering. After each pruning
        event, SINDy coefficients are reset and ridge-solved for instant
        recalibration. The RNN continuously adapts to the pruned library.

    Stage 2 (Final SINDy Refit):
        Freeze RNN weights and refit SINDy coefficients via ridge solve on
        stable hidden states. Single-pass: solve → prune → refit.

    Args:
        model: RNN model with SINDy integration
        dataset_train: Training dataset
        dataset_test: Validation dataset (optional)
        optimizer: PyTorch optimizer
        epochs: Total training epochs
        batch_size: Training batch size
        scheduler: Enable learning rate scheduler
        n_steps: BPTT truncation length
        convergence_threshold: Early stopping threshold
        loss_fn: Loss function for behavioral prediction
        sindy_weight: λ_sindy regularization strength
        sindy_alpha: Degree-weighted L1 penalty strength
        sindy_pruning_threshold: Optional threshold for per-member pruning
            (None or 0 to disable; default: None)
        sindy_pruning_frequency: Epochs between pruning events
        sindy_ensemble_alpha: Ensemble t-test significance level (e.g. 0.05).
            Primary pruning mechanism. None to disable.
        sindy_confidence_threshold: Optional participant presence threshold (0-1).
            None to disable.
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
            print("\tJoint training: [x]")
        else:
            print("\tJoint training: [ ]")
        # Pruning details
        pruning_details = []
        if sindy_ensemble_pruning is not None:
            pruning_details.append(f"ensemble t-test alpha={sindy_ensemble_pruning}")
        if sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
            pruning_details.append(f"threshold={sindy_threshold_pruning}")
        if sindy_population_pruning is not None and sindy_population_pruning > 0:
            pruning_details.append(f"participant threshold={sindy_population_pruning}")
        if pruning_details:
            print(f"\tPruning (every {sindy_pruning_frequency} epochs): {', '.join(pruning_details)}")
        else:
            print("\tPruning: [ ] (no pruning configured)")
        if sindy_weight > 0:
            print("\tFinal SINDy refit: [x]")
        else:
            print("\tFinal SINDy refit: [ ]")
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
            print("Stage 1: Joint training with ensemble pruning")
            print("=" * terminal_width)

        model, optimizer, loss_train, loss_test_rnn, loss_test_sindy = _run_joint_training(
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

            verbose=verbose,
            keep_log=keep_log,
            path_save_checkpoints=path_save_checkpoints,
        )
        model.rnn_training_finished = True

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Final SINDy Refit (single-pass lstsq)
    # ══════════════════════════════════════════════════════════════════════════
    if sindy_weight > 0:
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "=" * terminal_width)
            print("Stage 2: Final SINDy refit")
            print("=" * terminal_width)
        model = _run_sindy_training(
            model=model,
            xs_train=xs_train_5d,
            ys_train=ys_train_5d,
            pruning_threshold=0.05,#sindy_threshold_pruning if sindy_threshold_pruning else 0.05,
            verbose=verbose,
        )

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
                    batch_size=None,
                    convergence_threshold=0,
                    n_steps=n_steps,
                    use_scheduler=False,
                    loss_fn=loss_fn,

                    sindy_weight=sindy_weight,
                    sindy_alpha=0,
                    sindy_threshold_pruning=0,
                    sindy_pruning_frequency=999,
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

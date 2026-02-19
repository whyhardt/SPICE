
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


def _setup_dataloaders(
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset,
    batch_size: int,
    bagging: bool,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Set up training and test dataloaders.

    Returns:
        Tuple of (train_dataloader, test_dataloader, iterations_per_epoch)
    """
    if batch_size is None:
        batch_size = len(dataset_train)

    if bagging:
        batch_size = max(batch_size, 64)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size)
    else:
        sampler = None

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None)
    )

    dataloader_test = None
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))

    iterations_per_epoch = max(len(dataset_train), 64) // batch_size if batch_size < max(len(dataset_train), 64) else 1

    return dataloader_train, dataloader_test, iterations_per_epoch


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
    

def _run_sindy_training(
    model: BaseRNN,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    # dataset_test: SpiceDataset = None,
    epochs: int = 1,
    pruning_threshold: float = 0.05,
    pruning_frequency: int = 1,
    pruning_n_terms: int = None,
    pruning_patience: int = 1,
    pruning_warmup: int = 0,
    l2_lambda: float = 0.,
    batch_size: int = None,
    verbose: bool = True,
    ):
    """
    Second stage SINDy fitting: freeze RNN weights and refit SINDy coefficients
    on the trained RNN hidden states.

    Args:
        model (BaseRNN): Trained RNN model with SINDy coefficients
        dataset_train: Training dataset (4D)
        epochs (int): Number of epochs
        batch_size (int): Batch size for training
        verbose (bool): Print progress

    Returns:
        BaseRNN: Model with refitted SINDy coefficients
    """

    if verbose:
        terminal_width = _get_terminal_width()
        print("\n" + "="*terminal_width)
        print(f"Starting SINDy finetuning...")
        print("="*terminal_width)

    criterion = nn.MSELoss()
    _E, B, T, W, F = xs_train.shape
    n_features_y = ys_train.shape[-1]
    E = model.ensemble_size
    
    # Freeze all RNN parameters, only train SINDy coefficients
    for name, param in model.named_parameters():
        param.requires_grad = 'sindy' in name

    model.eval(use_sindy=False)

    # --------------------------------------------------------
    # VECTORIZATION OF SINDY TRAINING
    # --------------------------------------------------------

    def vectorize_training_data(xs, ys):
        """Vectorize training data for SINDy fitting.

        Args:
            xs: 5D tensor (E, B, T, W, F) for train, or 4D (B, T, W, F) for test
            ys: matching targets
        """
        if xs.dim() == 4:
            xs = xs.unsqueeze(0).repeat(E, 1, 1, 1, 1)
            ys = ys.unsqueeze(0).repeat(E, 1, 1, 1, 1)

        model.init_state(batch_size=B, within_ts=W)

        # State buffers: (within_ts, E, n_trials*B, n_items)
        state_buffer_current = {s: torch.zeros((W, E, T*B, model.n_items), dtype=torch.float32, device=model.device) for s in model.get_state()}
        state_buffer_next = {s: torch.zeros((W, E, T*B, model.n_items), dtype=torch.float32, device=model.device) for s in model.get_state()}

        with torch.no_grad():
            for t in range(T):
                for s in model.get_state():
                    state_buffer_current[s][0, :, t*B:(t+1)*B] = model.get_state()[s][-1]

                # 5D input: (E, B, 1, W, F)
                updated_state = model(xs[:, :, t:t+1].to(model.device), model.get_state(), batch_first=True)[1]

                for s in model.get_state():
                    state_buffer_current[s][1:, :, t*B:(t+1)*B] = updated_state[s][:-1]
                    state_buffer_next[s][:, :, t*B:(t+1)*B] = updated_state[s]

        # Flatten to (flat_total, 1, 1, F) for iteration — use first ensemble member for xs/ys metadata
        # NOTE: Must permute (E,B,T,W,F) -> (E,T,B,W,F) before reshape so flat index
        # order is (t,b,w), matching the state buffers which are filled trial-major.
        flat_total = W * T * B
        xs_flat = xs.permute(0, 2, 1, 3, 4).reshape(E, flat_total, 1, 1, F)[0]
        ys_flat = ys.permute(0, 2, 1, 3, 4).reshape(E, flat_total, 1, 1, n_features_y)[0]
        
        # Reshape state buffers: (W, E, T*B, items) -> (1, E, flat_total, items)
        for s in model.get_state():
            state_buffer_current[s] = state_buffer_current[s].permute(1, 2, 0, 3).reshape(1, E, flat_total, model.n_items)
            state_buffer_next[s] = state_buffer_next[s].permute(1, 2, 0, 3).reshape(1, E, flat_total, model.n_items)

        return state_buffer_current, state_buffer_next, xs_flat, ys_flat

    input_state_buffer_train, target_state_buffer_train, xs_flat, ys_flat = vectorize_training_data(xs_train, ys_train)
    # if dataset_test:
    #     input_state_buffer_test, target_state_buffer_test, dataset_test_flat = vectorize_training_data(dataset_test.xs, dataset_test.ys)

    # --------------------------------------------------------
    # SINDY TRAINING LOOP
    # --------------------------------------------------------

    model.use_sindy = True
    nan_mask = ~torch.isnan(xs_flat[:, 0, 0, :model.n_actions].sum(dim=(-1)))
    batch_size = xs_flat.shape[0] if batch_size is None else batch_size
    len_dataset = xs_flat.shape[0]
    len_last_print = 0
    try:
        for epoch in range(epochs+1):
            t_start = time.time()
            loss_train = 0
            iterations = 0

            for idx in range(0, len_dataset, batch_size):

                optimizer.zero_grad()

                batched_nan_mask = nan_mask[idx:idx+batch_size].to(model.device)
                # Vectorized slicing across all ensemble members at once (no Python loop over E)
                batched_input_state_buffer = {s: input_state_buffer_train[s][:, :, idx:idx+batch_size][:, :, batched_nan_mask] for s in input_state_buffer_train}
                batched_target_state_buffer = {s: target_state_buffer_train[s][:, :, idx:idx+batch_size][:, :, batched_nan_mask] for s in target_state_buffer_train}
                    
                # 4D input → model promotes to 5D; state is already (1, E, batch, items)
                _, state_pred = model(xs_flat[idx:idx+batch_size, :1].to(model.device)[batched_nan_mask], batched_input_state_buffer, batch_first=True)

                loss_batch = 0
                for s in model.spice_config.states_in_logit:
                    loss_batch += criterion(batched_target_state_buffer[s], state_pred[s])

                if l2_lambda > 0:
                    coefficient_penalty = model.compute_weighted_coefficient_penalty(l2_lambda)
                    loss_batch += coefficient_penalty

                loss_batch.backward()
                optimizer.step()

                loss_train += loss_batch.item()
                iterations += 1

            loss_train /= iterations

            loss_test = 0

            # THRESHOLDING STEP
            if epoch >= pruning_warmup and epoch % pruning_frequency == 0 and epoch != 0:
                model.sindy_coefficient_pruning(threshold=pruning_threshold, n_terms_pruning=pruning_n_terms, patience=pruning_patience)

            len_last_print = _print_training_status(
                len_last_print=len_last_print,
                model=model,
                n_calls=epoch+1,
                epochs=epochs,
                loss_train=loss_train,
                loss_test_rnn=None,
                loss_test_sindy=loss_test,
                time_elapsed=time.time()-t_start,
                convergence_value=None,
                sindy_weight=1,
                scheduler=None,
                warmup_steps=pruning_warmup,
                keep_log=DEBUG_MODE,
            )
    except KeyboardInterrupt:
        if verbose:
            print('\nTraining interrupted. Continuing with further operations...')

    # Restore requires_grad for all parameters
    for param in model.parameters():
        param.requires_grad = True

    return model.train()


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
    sindy_l2_lambda: float,
    sindy_pruning_threshold: float,
    sindy_pruning_frequency: int,
    sindy_pruning_terms: int,
    sindy_pruning_patience: int,
    sindy_optimizer_reset: int,

    convergence_threshold: float,
    verbose: bool,
    keep_log: bool,
    path_save_checkpoints: str,
) -> Tuple[BaseRNN, torch.optim.Optimizer, float, float, float]:
    """
    Stage 1: Joint RNN-SINDy optimization.

    Trains RNN to predict behavior while SINDy regularization pushes
    dynamics toward sparse equations. Uses patience-based sequential
    thresholding (STLSQ variant) to prune coefficients.

    Objective: L_total = L_CE(y, ŷ) + λ_sindy * L_SINDy

    Returns:
        Tuple of (model, optimizer, loss_train, loss_test_rnn)
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
                        sindy_alpha=sindy_l2_lambda,
                        loss_fn=loss_fn,
                    )
                    loss_train += loss_i

                n_calls_to_train_model += 1
                loss_train /= iterations_per_epoch

                # Reset SINDy optimizer state periodically after warmup
                if (sindy_optimizer_reset is not None
                    and n_calls_to_train_model % sindy_optimizer_reset == 0
                    and n_calls_to_train_model >= n_warmup_steps
                    and sindy_weight > 0):
                    num_reset = 0
                    for param in optimizer.param_groups[0]['params']:
                        if param in optimizer.state:
                            optimizer.state[param] = {}
                            num_reset += 1
                    if verbose and num_reset > 0:
                        print(f"\n>>> Epoch {n_calls_to_train_model}: Reset optimizer state for {num_reset} SINDy parameters.\n")

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

            # Periodic coefficient pruning
            if (sindy_weight > 0
                and n_calls_to_train_model >= n_warmup_steps
                and n_calls_to_train_model % sindy_pruning_frequency == 0):
                model.sindy_coefficient_pruning(
                    threshold=sindy_pruning_threshold,
                    n_terms_pruning=sindy_pruning_terms,
                    patience=sindy_pruning_patience
                )

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

    return significant.unsqueeze(0)


def _compute_ensemble_confidence_masks(
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


def _compute_participant_confidence_masks(
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
        participant_presence = mask.float().sum(dim=1).sum(dim=1)
        confidence_mask = participant_presence >= min_occurrences
        confidence_masks[module] = confidence_mask.unsqueeze(1).unsqueeze(1)

        if verbose:
            n_before = mask.any(dim=1).any(dim=1).sum().item()
            n_after = confidence_mask.sum().item()
            print(f"\t{module}: {n_before} terms -> {n_after} terms")

    return confidence_masks


def _compute_confidence_masks(
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
        ensemble_masks = _compute_ensemble_confidence_masks(
            model, test_fn=ensemble_test_fn, verbose=verbose, alpha=ensemble_alpha
        )
    else:
        # No ensemble filtering — term is present for (P, X) if any ensemble member has it
        ensemble_masks = {
            module: model.sindy_coefficients_presence[module].any(dim=0).unsqueeze(0)
            for module in model.submodules_rnn
        }

    # Step 2: Participant filtering (global)
    if participant_threshold is not None and participant_threshold > 0:
        confidence_masks = _compute_participant_confidence_masks(
            ensemble_masks, participant_threshold, model.n_participants, model.n_experiments, verbose=verbose
        )
    else:
        # No participant filtering — keep terms that pass ensemble filter for any (P, X)
        # confidence_masks = {module: mask.any(dim=1).any(dim=1).unsqueeze(1).unsqueeze(1) for module, mask in ensemble_masks.items()}
        confidence_masks = ensemble_masks
        
    return confidence_masks


def _reset_model_with_masks(
    model: BaseRNN,
    confidence_masks: dict,
    optimizer: torch.optim.Optimizer,
    verbose: bool = True,
) -> Tuple[BaseRNN, torch.optim.Optimizer]:
    """
    Reset RNN weights and SINDy coefficients, applying confidence masks.

    This prepares the model for retraining with a filtered candidate library.
    """
    if verbose:
        print("Model reset with confidence-filtered mask.")

    # Reset RNN submodule weights
    for module_name, module in model.submodules_rnn.items():
        for submodule in module.modules():
            if hasattr(submodule, 'reset_parameters'):
                submodule.reset_parameters()

    # Reset participant embedding if exists
    if hasattr(model, 'participant_embedding'):
        for submodule in model.participant_embedding.modules():
            if hasattr(submodule, 'reset_parameters'):
                submodule.reset_parameters()

    # Reinitialize SINDy coefficients with confidence masks applied
    for module in model.submodules_rnn:
        model.setup_sindy_coefficients(key_module=module)
        confidence_mask = confidence_masks[module].to(model.device)
        # Broadcast mask (terms,) -> (E, P, X, terms)
        confidence_mask = confidence_mask.reshape(1, 1, 1, -1).expand(model.ensemble_size, model.n_participants, model.n_experiments, -1)
        model.sindy_coefficients_presence[module] = confidence_mask.clone().to(model.device)
        model.sindy_coefficients[module].data *= confidence_mask.float().to(torch.device('cpu'))

    model.rnn_training_finished = False
    model.to(model.device)

    # Recreate optimizer with new parameters
    sindy_params = []
    rnn_params = []
    for name, param in model.named_parameters():
        if 'sindy' in name:
            sindy_params.append(param)
        else:
            rnn_params.append(param)

    lr_sindy = optimizer.param_groups[0]['lr']
    lr_rnn = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
    weight_decay_rnn = optimizer.param_groups[1].get('weight_decay', 0) if len(optimizer.param_groups) > 1 else 0

    optimizer = torch.optim.AdamW([
        {'params': sindy_params, 'weight_decay': 0, 'lr': lr_sindy},
        {'params': rnn_params, 'weight_decay': weight_decay_rnn, 'lr': lr_rnn},
    ])

    return model, optimizer


def _reset_sindy_with_masks(
    model: BaseRNN,
    confidence_masks: dict,
    lr: float,
    verbose: bool = True,
) -> Tuple[BaseRNN, torch.optim.Optimizer]:
    """
    Reset RNN weights and SINDy coefficients, applying confidence masks.

    This prepares the model for retraining with a filtered candidate library.
    """
    if verbose:
        print("SINDy reset with confidence-filtered mask.")

    # Reinitialize SINDy coefficients with confidence masks applied
    for module in model.submodules_rnn:
        model.setup_sindy_coefficients(key_module=module, polynomial_degree=model.sindy_specs[module]['polynomial_degree'])
        if confidence_masks is not None:
            confidence_mask = confidence_masks[module].to(model.device)
            # Broadcast mask (terms,) -> (E, P, X, terms)
            confidence_mask = confidence_mask.reshape(1, 1, 1, -1).expand(model.ensemble_size, model.n_participants, model.n_experiments, -1)
            model.sindy_coefficients_presence[module] = confidence_mask.clone().to(model.device)
            model.sindy_coefficients[module].data *= confidence_mask.float().to(torch.device('cpu'))
            
    model.rnn_training_finished = True
    model.to(model.device)

    # Recreate optimizer with new parameters
    sindy_params = []
    for name, param in model.named_parameters():
        if 'sindy' in name:
            sindy_params.append(param)

    optimizer = torch.optim.AdamW(sindy_params, lr=lr)
    
    return model, optimizer


def fit_spice(
    model: BaseRNN,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset = None,
    optimizer: torch.optim.Optimizer = None,
    
    epochs: int = 1,
    epochs_confidence: int = None,
    batch_size: int = None,
    scheduler: bool = False,
    n_steps: int = None,
    convergence_threshold: float = 1e-7,
    loss_fn: callable = cross_entropy_loss,
    
    sindy_weight: float = 0.,
    sindy_l2_lambda: float = 0.,
    sindy_epochs: int = 1000,
    sindy_pruning_threshold: float = 0.01,
    sindy_pruning_frequency: int = 1,
    sindy_pruning_terms: int = None,
    sindy_pruning_patience: int = 0,
    sindy_optimizer_reset: int = None,
    sindy_confidence_threshold: float = None,
    sindy_ensemble_alpha: float = None,

    verbose: bool = True,
    keep_log: bool = False,
    n_warmup_steps: int = 0,
    path_save_checkpoints: str = None,
) -> Tuple[BaseRNN, torch.optim.Optimizer, float]:
    """
    Three-stage SPICE training pipeline.

    Stage 1 (Joint Training):
        Train RNN + SINDy jointly with L_CE + λ_sindy * L_SINDy.
        Patience-based coefficient thresholding allows diverse per-participant
        equations (high recall).

    Stage 2 (Confidence Filtering, optional):
        Two-level filtering to remove unreliable terms:
        1. Ensemble filtering: per (participant, experiment, term), apply a
           statistical test (default: t-test) across ensemble members to keep
           only robustly identified terms.
        2. Participant filtering: keep terms present in >= sindy_confidence_threshold
           fraction of participant-experiment slots.
        Reset all parameters and retrain with filtered library.

    Stage 3 (SINDy Refinement):
        Freeze RNN weights and refit SINDy coefficients on stable hidden states.
        Improves coefficient precision.

    Args:
        model: RNN model with SINDy integration
        dataset_train: Training dataset
        dataset_test: Validation dataset (optional)
        optimizer: PyTorch optimizer
        convergence_threshold: Early stopping threshold
        sindy_weight: λ_sindy regularization strength
        sindy_l2_lambda: Degree-weighted penalty strength
        sindy_epochs: Number of epochs for Stage 3
        sindy_pruning_threshold: Coefficient pruning threshold (η)
        sindy_pruning_frequency: Epochs between pruning
        sindy_pruning_terms: Max terms to prune per step
        sindy_pruning_patience: Patience before coefficient pruning
        sindy_optimizer_reset: Epochs between optimizer state resets
        sindy_confidence_threshold: Stage 2 participant presence threshold
        sindy_ensemble_alpha: Stage 2 ensemble t-test significance level (e.g. 0.05)
        epochs: Total training epochs for Stage 1
        batch_size: Training batch size
        bagging: Enable bootstrap aggregation
        scheduler: Enable learning rate scheduler
        n_steps: BPTT truncation length
        verbose: Print progress
        keep_log: Keep full training log (vs. live update)
        n_warmup_steps: Warmup epochs for SINDy weight
        path_save_checkpoints: Path for saving checkpoints

    Returns:
        Tuple of (trained_model, optimizer, final_loss)
    """
    
    if epochs_confidence is None:
        epochs_confidence = epochs
    if sindy_epochs is None:
        sindy_epochs = epochs
    if n_warmup_steps is None:
        if epochs is not None:
            n_warmup_steps = epochs//4
        elif epochs_confidence is not None:
            n_warmup_steps = epochs_confidence//4
        else:
            n_warmup_steps = 0
    
    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n"+status_lines)
        print("SPICE Training Configuration:")
        if epochs > 0:
            print("\tSPICE joint training: [x]")
        else:
            print("\tSPICE joint training: [ ]")
        has_ensemble_filter = sindy_ensemble_alpha is not None
        has_participant_filter = sindy_confidence_threshold is not None and sindy_confidence_threshold > 0
        if (has_ensemble_filter or has_participant_filter) and epochs_confidence > 0:
            detail = []
            if has_ensemble_filter:
                detail.append(f"ensemble alpha={sindy_ensemble_alpha}")
            if has_participant_filter:
                detail.append(f"participant threshold={sindy_confidence_threshold}")
            print(f"\tConfidence-based SINDy coefficient filtering: [x] ({', '.join(detail)})")
        else:
            print("\tConfidence-based SINDy coefficient filtering: [ ]")
        if sindy_epochs > 0:
            print("\tSINDy-only finetuning: [x]")
        else:
            print("\tSINDy-only finetuning: [ ]")
        print(status_lines)
    
    # Store original learning rate for Stage 3
    original_lr = optimizer.param_groups[0]['lr']
    sindy_finetuned = False

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
    # STAGE 1: Joint RNN-SINDy Training
    # ══════════════════════════════════════════════════════════════════════════
    if epochs > 0:
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "="*terminal_width)
            print(f"Stage 1: SPICE joint training")
            print("="*terminal_width)
            
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
            sindy_l2_lambda=sindy_l2_lambda,
            sindy_pruning_threshold=sindy_pruning_threshold,
            sindy_pruning_frequency=sindy_pruning_frequency,
            sindy_pruning_terms=sindy_pruning_terms,
            sindy_pruning_patience=sindy_pruning_patience,
            sindy_optimizer_reset=sindy_optimizer_reset,

            verbose=verbose,
            keep_log=keep_log,
            path_save_checkpoints=path_save_checkpoints,
        )
        model.rnn_training_finished = True

        # SINDy finetuning (freeze RNN weights -> steady optimization target)
        if sindy_weight > 0 and sindy_epochs > 0:
            model, optimizer_sindy = _reset_sindy_with_masks(model=model, confidence_masks=None, lr=original_lr, verbose=verbose)
            model = _run_sindy_training(
                model=model,
                optimizer=optimizer_sindy,
                xs_train=xs_train_5d,
                ys_train=ys_train_5d,
                # dataset_test=dataset_test,
                
                epochs=sindy_epochs,
                batch_size=None,
                
                l2_lambda=sindy_l2_lambda,
                pruning_threshold=sindy_pruning_threshold,
                pruning_n_terms=sindy_pruning_terms,
                pruning_patience=sindy_pruning_patience,
                pruning_warmup=sindy_epochs // 4,

                verbose=verbose,
            )
            sindy_finetuned = True

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Confidence Filtering (optional)
    # ══════════════════════════════════════════════════════════════════════════
    confidence_masks = None
    has_ensemble_filter = sindy_ensemble_alpha is not None
    has_participant_filter = sindy_confidence_threshold is not None and sindy_confidence_threshold > 0
    if epochs_confidence > 0 and sindy_weight > 0 and (has_ensemble_filter or has_participant_filter):
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "="*terminal_width)
            print(f"Stage 2: Confidence filtering")
            if has_ensemble_filter:
                print(f"  Ensemble filtering: alpha={sindy_ensemble_alpha}")
            if has_participant_filter:
                print(f"  Participant filtering: threshold={sindy_confidence_threshold}")
            print("="*terminal_width)

        # Compute confidence masks (ensemble -> participant pipeline)
        confidence_masks = _compute_confidence_masks(
            model,
            ensemble_alpha=sindy_ensemble_alpha,
            participant_threshold=sindy_confidence_threshold,
            verbose=verbose,
        )
        
        for module in model.get_modules():
            model.sindy_coefficients_presence[module].data = model.sindy_coefficients_presence[module].data * confidence_masks[module]
            model.sindy_coefficients[module].data = model.sindy_coefficients[module] * model.sindy_coefficients_presence[module]
        
        # Reset model with filtered masks
        # sindy_finetuned = False
        # model, optimizer = _reset_model_with_masks(model, confidence_masks, optimizer, verbose)
        
        # if verbose:
        #     print("Starting second training pass with confidence-filtered terms...")

        # model, optimizer, loss_train, loss_test_rnn, loss_test_sindy = _run_joint_training(
        #     model=model,
        #     optimizer=optimizer,
        #     xs_train=xs_train_5d,
        #     ys_train=ys_train_5d,
        #     dataset_test=dataset_test,

        #     epochs=epochs_confidence,
        #     n_warmup_steps=n_warmup_steps,
        #     n_steps=n_steps,
        #     use_scheduler=scheduler,
        #     batch_size=batch_size,
        #     convergence_threshold=convergence_threshold,
        #     loss_fn=loss_fn,

        #     sindy_weight=sindy_weight,
        #     sindy_l2_lambda=sindy_l2_lambda,
        #     sindy_pruning_threshold=sindy_pruning_threshold,
        #     sindy_pruning_frequency=sindy_pruning_frequency,
        #     sindy_pruning_terms=sindy_pruning_terms,
        #     sindy_pruning_patience=sindy_pruning_patience,
        #     sindy_optimizer_reset=sindy_optimizer_reset,

        #     verbose=verbose,
        #     keep_log=keep_log,
        #     path_save_checkpoints=path_save_checkpoints,
        # )
        # model.rnn_training_finished = True

        # # SINDy finetuning
        # if sindy_weight > 0 and sindy_epochs > 0:
        #     model, optimizer_sindy = _reset_sindy_with_masks(model=model, confidence_masks=confidence_masks, lr=original_lr)
        #     model = _run_sindy_training(
        #         model=model,
        #         optimizer=optimizer_sindy,
        #         xs_train=xs_train_5d,
        #         ys_train=ys_train_5d,
        #         # dataset_test=dataset_test,
                
        #         epochs=sindy_epochs,
        #         batch_size=batch_size*10,
                
        #         l2_lambda=sindy_l2_lambda,
        #         pruning_threshold=sindy_pruning_threshold,
        #         pruning_n_terms=sindy_pruning_terms,
        #         pruning_patience=sindy_pruning_patience,
        #         pruning_warmup=sindy_epochs//4,
                
        #         verbose=verbose,
        #     )
        #     sindy_finetuned=True

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: SINDy finetuning (if not already happened in previous stages)
    # ══════════════════════════════════════════════════════════════════════════
    if not sindy_finetuned and sindy_weight > 0 and sindy_epochs > 0:
        # confidence_masks = _compute_confidence_masks(model, sindy_confidence_threshold, verbose)
        model, optimizer_sindy = _reset_sindy_with_masks(model=model, confidence_masks=confidence_masks, lr=original_lr)
        model = _run_sindy_training(
            model=model,
            optimizer=optimizer_sindy,
            xs_train=xs_train_5d,
            ys_train=ys_train_5d,
            # dataset_test=dataset_test,
            
            epochs=sindy_epochs,
            batch_size=batch_size*10,
            
            l2_lambda=sindy_l2_lambda,
            pruning_threshold=sindy_pruning_threshold,
            pruning_n_terms=sindy_pruning_terms,
            pruning_patience=sindy_pruning_patience,
            pruning_warmup=sindy_epochs//4,
            
            verbose=verbose,
        )
        sindy_finetuned=True
    
    # ══════════════════════════════════════════════════════════════════════════
    # Final evaluation summary
    # ══════════════════════════════════════════════════════════════════════════
    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n"+status_lines)
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
                    sindy_l2_lambda=0,
                    sindy_pruning_threshold=0,
                    sindy_pruning_frequency=999,
                    sindy_pruning_terms=0,
                    sindy_pruning_patience=999,
                    sindy_optimizer_reset=None,

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

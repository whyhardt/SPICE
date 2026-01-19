import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple
from collections import defaultdict
import shutil

from .rnn import BaseRNN
from .spice_utils import SpiceDataset
from .sindy_differentiable import threshold_coefficients, get_library_term_degrees
import sys


# === DEBUG MODE: Set to False for live-updating display, True for line-by-line output ===
DEBUG_MODE = False


import os

def print_training_status(
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
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 80
    
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
        if isinstance(scheduler, (ReduceLROnPlateau, ReduceLROnPlateauRNNOnly, ReduceOnPlateauWithRestarts)):
            progress_msg += f"; Metric: {scheduler.best:.7f}; Bad epochs: {scheduler.num_bad_epochs}/{scheduler.patience}"
    
    status_lines.append(progress_msg)
    
    # Add SPICE model equations if SINDy is active
    if sindy_weight > 0:
        status_lines.append("-" * terminal_width)
        status_lines.append(f"SPICE Model (Coefficients: {model.count_sindy_coefficients().mean():.0f}):")
        status_lines.append(model.get_spice_model_string(participant_id=0, ensemble_idx=0))
        
        # Add patience summary
        status_lines.append("-" * terminal_width)
        status_lines.append("Cutoff patience:")
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
            for ip, p in enumerate(model.sindy_cutoff_patience_counters[m][0,0,0]):
                if skip_state[ip]:
                    continue  # Skip state-containing terms entirely (not printed in equations)
                elif fixed_linear[ip]:
                    patience_list += "-"  # Show "-" for fixed linear terms (fit_linear=False)
                elif model.sindy_coefficients_presence[m][0,0,0, ip]:
                    patience_list += str(p.item())
                else:
                    patience_list += "-"  # Thresholded-out terms
                patience_list += ", "
            status_lines.append(m + ": " + patience_list[:-2])
            
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


class ReduceLROnPlateauRNNOnly:
    """
    ReduceLROnPlateau scheduler that only applies to RNN parameters (param_groups[1]),
    leaving SINDy coefficient learning rates unchanged.
    """
    def __init__(self, optimizer, mode='min', factor=0.9, patience=100, min_lr=0, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics):
        """Update learning rate based on the validation loss (only for RNN params)."""
        current = metrics

        if self.mode == 'min':
            is_better = current < self.best
        else:
            is_better = current > self.best

        if is_better:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        """Reduce learning rate only for RNN parameters (param_groups[1])."""
        # Only update param_groups[1] (RNN parameters), skip param_groups[0] (SINDy coefficients)
        if len(self.optimizer.param_groups) > 1:
            old_lr = self.optimizer.param_groups[1]['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.optimizer.param_groups[1]['lr'] = new_lr
            self._last_lr[1] = new_lr

            if self.verbose and new_lr != old_lr:
                print(f'Reducing RNN learning rate to {new_lr:.4e}')

    def get_last_lr(self):
        """Return the last computed learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]


class ReduceOnPlateauWithRestarts:
    def __init__(self, optimizer, min_lr, factor, patience):
        """
        Plateau-based LR scheduler with restarts to the base learning rate when min_lr is hit.

        Parameters:
        - optimizer: Optimizer instance.
        - min_lr: The minimum learning rate after reductions.
        - factor: Multiplicative factor to reduce the LR on plateau.
        - patience: Number of epochs with no improvement before reducing the LR.
        """
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]  # Extract base LRs
        self.factor = factor
        self.patience = patience
        self.base_patience = patience
        
        self.best = float('inf')  # Initialize the best validation loss as infinity
        self.num_bad_epochs = 0  # Initialize the count of bad epochs
        self.num_cycles_completed = 0
        
        # Store the current learning rate for each parameter group
        self.current_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics):
        """
        Update the learning rate based on the validation loss.
        """
        if metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0  # Reset bad epochs counter
        else:
            self.num_bad_epochs += 1
        
        # Check if patience is exceeded
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()  # Reduce learning rates
            self._adjust_patience()  # Adjust the patience according to the learning rate
            self.num_bad_epochs = 0  # Reset bad epochs counter

    def _reduce_lr(self):
        """
        Reduce the learning rate for all parameter groups by the given factor.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):            
            
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            # Check if the new learning rate has hit min_lr and reset if so
            if new_lr <= self.min_lr:
                param_group['lr'] = self.base_lrs[i]
                self.num_cycles_completed += 1
            else:
                param_group['lr'] = new_lr
    
    def _adjust_patience(self):
        """
        Adjust the patience according to the learning rate.
        """
        # self.patience = max([self.patience * (1+self.num_cycles_completed) if self.get_lr()[-1] < self.base_lrs[-1] else self.base_patience, 200])
        self.patience = self.patience * 2 if self.get_lr()[-1] < self.base_lrs[-1] else self.base_patience

    def get_lr(self):
        """
        Retrieve the current learning rates for all parameter groups.
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self):
        """
        Retrieve the last computed learning rates for all parameter groups.
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    

def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    sindy_weight: float = 0.,
    sindy_alpha: float = 0.,
    n_steps: int = None,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(label_smoothing=0.),
    ):

    """
    Trains a model with the given batch.
    """

    if n_steps is None:
        n_steps = xs.shape[1]

    model.init_state(batch_size=len(xs))
    state = model.get_state(detach=True)

    loss_batch = 0
    iterations = 0
    for t in range(0, xs.shape[1], n_steps):
        n_steps = min(xs.shape[1]-t, n_steps)
        xs_step = xs[:, t:t+n_steps]
        ys_step = ys[:, t:t+n_steps]
        
        state = model.get_state(detach=True)
        ys_pred, _ = model(xs_step, state, batch_first=True)
        
        # Mask out padding (NaN values) - valid trials have non-NaN actions
        mask = ~torch.isnan(xs_step[..., 0])
        mask = mask.reshape(-1)
        
        loss_step = loss_fn(
            ys_pred.reshape(-1, model.n_actions)[mask],
            torch.argmax(ys_step.reshape(-1, model.n_actions), dim=1)[mask],
            )
        
        if torch.is_grad_enabled():
            # Add SINDy regularization loss
            if sindy_weight > 0 and model.sindy_loss != 0:
                loss_step = loss_step + sindy_weight * model.sindy_loss
                
            # # Polynomial degree weighted coefficient penalty for SINDy coefficients
            if sindy_weight > 0 and sindy_alpha > 0:
                coefficient_penalty = model.compute_weighted_coefficient_penalty(sindy_alpha=sindy_alpha, norm=1)
                # coefficient_penalty = 0
                # for module in model.submodules_rnn:
                #     coefficient_penalty += (model.sindy_coefficients[module] * model.sindy_coefficients_presence[module]).abs().sum(dim=-1).mean()
                loss_step = loss_step + sindy_alpha * coefficient_penalty

            # backpropagation
            optimizer.zero_grad()
            loss_step.backward()

            # Apply gradient masks to prevent updating zeroed coefficients
            # model.apply_gradient_masks()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        loss_batch += loss_step.item()
        iterations += 1
            
    return model, optimizer, loss_batch/iterations
    

def fit_sindy_second_stage(
    model: BaseRNN,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset = None,
    learning_rate: float = 1e-3,
    epochs: int = 1,
    cutoff_threshold: float = 0.05,
    cutoff_frequency: int = 1,
    cutoff_n_terms: int = None,
    cutoff_patience: int = 1,
    cutoff_warmup: int = 0,
    sindy_alpha: float = 0.,
    batch_size: int = None,
    verbose: bool = True,
    ):
    """
    Second stage SINDy fitting: freeze RNN weights and refit SINDy coefficients
    on the trained RNN hidden states with no thresholding (threshold=0).

    This follows the approach from sindy-shred where SINDy coefficients are
    discarded after initial training and refitted on the learned hidden states.

    IMPORTANT: Collapses ensemble to a single SINDy model (ensemble_size=1).

    Args:
        model (BaseRNN): Trained RNN model with SINDy coefficients
        dataset_train (DatasetRNN): Training dataset
        dataset_test (DatasetRNN, optional): Validation dataset. Defaults to None.
        learning_rate (float): Learning rate for SINDy coefficient optimization
        epochs (int): Number of epochs for second stage training
        batch_size (int): Batch size for training
        verbose (bool): Print progress

    Returns:
        BaseRNN: Model with refitted SINDy coefficients (single model, no ensemble)
    """

    # Always print header for second stage (important step)
    print("\n" + "="*80)
    print(f"Starting second stage SINDy fitting (threshold={cutoff_threshold}, single model)")
    print("="*80)

    criterion = nn.MSELoss()

    # Freeze all RNN parameters, only train SINDy coefficients
    for param in model.parameters():
        param.requires_grad = False
    
    # COLLAPSE ENSEMBLE TO SINGLE MODEL -> Set ensemble size to 1 for stage 2 (no ensemble in second stage)
    model.sindy_ensemble_size = 1

    # Re-initialize SINDy coefficients with single model (ensemble_size=1)
    for key_module in model.submodules_rnn:
        model.setup_sindy_coefficients(key_module=key_module)
    model.to(model.device)
    
    # Create optimizer for only SINDy coefficients
    optimizer_sindy = torch.optim.AdamW(list(model.sindy_coefficients.values()), lr=learning_rate)
    
    model.eval(use_sindy=False)
    
    # --------------------------------------------------------
    # VECTORIZATION OF SINDY TRAINING
    # --------------------------------------------------------
    # rnn-weights are frozen -> we can vectorize the whole SINDy training by computing all rnn states for each timestep ad-hoc
    
    def vectorize_training_data(dataset):
        # Initialize model state for correct batch size based on actual dataset
        # dataset.xs.shape[0] gives the number of sessions/participants in first dimension
        len_dataset = dataset.xs.shape[0]
        n_timesteps = dataset.xs.shape[1]
        model.init_state(batch_size=len_dataset)

        # initialize state buffer with time-flattened batch-dim
        state_buffer_current = {state: torch.zeros((n_timesteps*len_dataset, model.n_items), dtype=torch.float32, device=model.device) for state in model.get_state()}
        state_buffer_next = {state: torch.zeros((n_timesteps*len_dataset, model.n_items), dtype=torch.float32, device=model.device) for state in model.get_state()}

        with torch.no_grad():
            for t in range(n_timesteps):
                # save current state (input to forward-pass) in state buffer
                for state in state_buffer_next:
                    state_buffer_current[state][t*len_dataset:(t+1)*len_dataset] = model.get_state()[state]

                # compute updated state
                updated_state = model(dataset.xs[:, t][:, None].to(model.device), model.get_state(), batch_first=True)[1]
                
                # save updated state (training target) in state buffer
                for state in state_buffer_next:
                    state_buffer_next[state][t*len_dataset:(t+1)*len_dataset] = updated_state[state]
        
        # reshape the dataset to be aligned with state buffer
        xs = dataset.xs.transpose(0, 1).reshape(n_timesteps*len_dataset, 1, -1)
        ys = dataset.ys.transpose(0, 1).reshape(n_timesteps*len_dataset, 1, -1)
        dataset = SpiceDataset(xs, ys)
        
        return state_buffer_current, state_buffer_next, dataset
    
    input_state_buffer_train, target_state_buffer_train, dataset_train = vectorize_training_data(dataset_train)
    if dataset_test:
        input_state_buffer_test, target_state_buffer_test, dataset_test = vectorize_training_data(dataset_test)
    
    
    # --------------------------------------------------------
    # SINDY TRAINING LOOP
    # --------------------------------------------------------
    
    model.use_sindy = True
    xs = dataset_train.xs
    nan_mask = ~torch.isnan(xs[:, 0, :model.n_actions].sum(dim=-1))
    batch_size = dataset_train.xs.shape[0] if batch_size is None else batch_size
    len_dataset = dataset_train.xs.shape[0]
    len_last_print = 0
    
    for epoch in range(epochs+1):
        t_start = time.time()
        loss_train = 0
        iterations = 0
        
        if epoch+1 == epochs:
            final_run = True
        
        for idx in range(0, len_dataset, batch_size):
            
            optimizer_sindy.zero_grad()
            
            batched_nan_mask = nan_mask[idx:idx+batch_size].to(model.device)
            batched_input_state_buffer, batched_target_state_buffer = {}, {}
            for state in input_state_buffer_train:
                batched_input_state_buffer[state] = input_state_buffer_train[state][idx:idx+batch_size][batched_nan_mask]
                batched_target_state_buffer[state] = target_state_buffer_train[state][idx:idx+batch_size][batched_nan_mask]
            
            # get sindy-based state updates from original rnn states
            _, state_pred = model(xs[idx:idx+batch_size, :1].to(model.device)[batched_nan_mask], batched_input_state_buffer, batch_first=True)
            
            loss_batch = 0
            for state in model.spice_config.states_in_logit:
                loss_batch += criterion(batched_target_state_buffer[state], state_pred[state])

            # l1 regularization (unweighted)
            # for module in model.submodules_rnn:
            #     loss_batch += model.sindy_coefficients[module].abs().sum() * sindy_alpha

            # Polynomial degree weighted coefficient penalty for SINDy coefficients
            if sindy_alpha > 0:
                coefficient_penalty = model.compute_weighted_coefficient_penalty(sindy_alpha)
                loss_batch += coefficient_penalty

            # Backward pass - only update SINDy coefficients
            loss_batch.backward()
            # Apply gradient masks to prevent updating zeroed coefficients
            # model.apply_gradient_masks()
            optimizer_sindy.step()
                
            loss_train += loss_batch.item()
            iterations += 1
            
        loss_train /= iterations

        loss_test = 0
        
        # THRESHOLDING STEP
        if epoch >= cutoff_warmup and epoch % cutoff_frequency == 0 and epoch != 0:
            model.sindy_coefficient_cutoff(threshold=0, base_threshold=cutoff_threshold, n_terms_cutoff=cutoff_n_terms, patience=cutoff_patience)
        
        # Print training progress
        len_last_print = print_training_status(
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
            warmup_steps=cutoff_warmup,
            keep_log=DEBUG_MODE,
        )    
        
    # Restore requires_grad for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    return model.train()


def fit_model(
    model: BaseRNN,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset = None,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-7,
    sindy_weight: float = 0.,
    sindy_alpha: float = 0.,
    sindy_epochs: int = 1000,
    sindy_threshold: float = 0.01,
    sindy_threshold_frequency: int = 1,
    sindy_threshold_terms: int = None,
    sindy_threshold_patience: int = 0,
    epochs: int = 1,
    batch_size: int = None,
    bagging: bool = False,
    scheduler: bool = False,
    n_steps: int = None,
    verbose: bool = True,
    keep_log: bool = False,
    n_warmup_steps: int = 0,
    path_save_checkpoints: str = None,
    ) -> Tuple[BaseRNN, torch.optim.Optimizer, float]:
    """_summary_
    
    Args:
        model (BaseRNN): A child class of the BaseRNN, which implements the forward method
        dataset_train (DatasetRNN): training data for the RNN of shape (Batch, Timesteps, Features) with Features being (Actions, Rewards, Participant ID) -> (n_actions, n_actions, 1)
        dataset_test (DatasetRNN, optional): Validation dataset during training. Defaults to None.
        optimizer (torch.optim.Optimizer, optional): Torch-optimizer. Defaults to None.
        convergence_threshold (float, optional): Threshold of convergence value, which determines early stopping of training. Defaults to 1e-5.
        epochs (int, optional): Total number of training epochs. Defaults to 1.
        batch_size (int, optional): Batch size. Defaults to -1.
        bagging (bool, optional): Enables bootstrap aggregation. Defaults to False.
        n_steps (int, optional): Number of steps passed at once through the RNN to compute a gradient over steps. Defaults to -1.
        verbose (bool, optional): Verbosity. Defaults to True.

    Returns:
        (BaseRNN, Optimizer, float): (Trained RNN, Optimizer with last state, Training loss)
    """

    # initialize dataloader
    if batch_size is None:
        batch_size = len(dataset_train)
    
    # use random sampling with replacement
    if bagging:
        batch_size = max(batch_size, 64)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size) if bagging else None
    else:
        sampler = None
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False)
    # if dataset_test is None:
    #     dataset_test = dataset_train
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    
    # set up warmup phase characteristics and learning rate scheduler
    def setup_warmup_scaler(exp_max: float = 1):
        warmup_scaler = torch.exp(torch.linspace(0, exp_max, n_warmup_steps))
        warmup_scaler = (warmup_scaler - warmup_scaler.min()) / (warmup_scaler.max() - warmup_scaler.min()) + 1e-4 if n_warmup_steps > 0 else None
        return warmup_scaler
    
    warmup_scaler_sindy_weight = setup_warmup_scaler(5)
    
    # original learning rates
    original_lr = optimizer.param_groups[1]['lr']

    # Initialize learning rate scheduler (only for RNN parameters, not SINDy coefficients)
    lr_scheduler = None
    if scheduler:
        lr_scheduler = ReduceLROnPlateauRNNOnly(
            optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            min_lr=1e-4,
        )

    if epochs == 0:
        continue_training = False
        msg = 'No training epochs specified. Model will not be trained.'
        if verbose:
            print(msg)
    else:
        continue_training = True
        converged = False
        n_calls_to_train_model = 0
        convergence_value = 1
        last_loss = 1
        recency_factor = 0.5
    
    len_last_print = 0
    loss_train = 0
    loss_test = 0
    iterations_per_epoch = max(len(dataset_train), 64) // batch_size if batch_size < max(len(dataset_train), 64) else 1
    
    # save_at_epoch = warmup_steps
    
    # start training
    while continue_training:
        try:
            loss_train = 0
            loss_test = 0
            t_start = time.time()
            
            # warmup updates
            sindy_weight_epoch = sindy_weight * (1 if n_calls_to_train_model >= n_warmup_steps else warmup_scaler_sindy_weight[n_calls_to_train_model])
            # optimizer.param_groups[1]['lr'] = original_lr if n_calls_to_train_model >= n_warmup_steps else original_lr * 10
            
            for _ in range(iterations_per_epoch):
                # get next batch
                xs, ys = next(iter(dataloader_train))
                if xs.device != model.device:
                    xs = xs.to(model.device)
                    ys = ys.to(model.device)
                # train model
                model, optimizer, loss_i = batch_train(
                    model=model,
                    xs=xs,
                    ys=ys,
                    optimizer=optimizer,
                    n_steps=n_steps,
                    sindy_weight=sindy_weight_epoch,
                    sindy_alpha=sindy_alpha,
                )
                loss_train += loss_i
            
            n_calls_to_train_model += 1
            loss_train /= iterations_per_epoch

            # Reset SINDy optimizer state after warmup (when SINDy weight reaches full strength)
            # This is important because optimizer momentum/adaptive LRs were calibrated on
            # scaled-down gradients during warmup and may not be optimal for full-strength gradients
            if n_calls_to_train_model == n_warmup_steps and n_warmup_steps > 0 and sindy_weight > 0:
                # Only reset state for SINDy parameters (param_groups[0])
                num_reset = 0
                for param in optimizer.param_groups[0]['params']:
                    if param in optimizer.state:
                        optimizer.state[param] = {}
                        num_reset += 1
                if verbose and num_reset > 0:
                    print(f"\n>>> Warmup complete (epoch {n_calls_to_train_model}). Reset optimizer state for {num_reset} SINDy parameters (fresh start at full regularization strength).\n")

            if dataset_test is not None:
                model = model.eval(use_sindy=False)
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    if xs.device != model.device:
                        xs = xs.to(model.device)
                        ys = ys.to(model.device)
                    # evaluate model
                    _, _, loss_test_rnn = batch_train(model=model, xs=xs, ys=ys)
                
                if sindy_weight > 0:
                    model = model.eval(use_sindy=True)
                    with torch.no_grad():
                        xs, ys = next(iter(dataloader_test))
                        if xs.device != model.device:
                            xs = xs.to(model.device)
                            ys = ys.to(model.device)
                        # evaluate model
                        _, _, loss_test_sindy = batch_train(model=model, xs=xs, ys=ys)
                else:
                    loss_test_sindy = None   
                model = model.train()
            
            # periodic pruning of sindy coefficients with L0 norm
            if sindy_weight > 0 and n_calls_to_train_model >= n_warmup_steps and n_calls_to_train_model % sindy_threshold_frequency == 0:
                model.sindy_coefficient_cutoff(threshold=sindy_threshold, base_threshold=0., n_terms_cutoff=sindy_threshold_terms, patience=sindy_threshold_patience)
            
            # check for convergence
            dloss = last_loss - loss_test_rnn if dataset_test is not None else last_loss - loss_train
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = 0
            last_loss += loss_test_rnn if dataset_test is not None else loss_train

            # Update learning rate scheduler
            if lr_scheduler is not None and n_calls_to_train_model >= n_warmup_steps:
                metric = loss_test_rnn if dataset_test is not None else loss_train
                lr_scheduler.step(metric)

            # save checkpoint
            if path_save_checkpoints and n_calls_to_train_model == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace('.', f'_ep{n_calls_to_train_model}.'))
                save_at_epoch *= 2

            # Display training status
            if verbose:
                len_last_print = print_training_status(
                    len_last_print=len_last_print,
                    model=model,
                    n_calls=n_calls_to_train_model,
                    epochs=epochs,
                    loss_train=loss_train,
                    loss_test_rnn=loss_test_rnn if dataset_test is not None else None,
                    loss_test_sindy=loss_test_sindy if dataset_test is not None else None,
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
    
    model.rnn_training_finished = True
    
    # Second stage: Refit SINDy coefficients on trained RNN hidden states (always run when sindy_weight > 0)
    if sindy_weight > 0 and sindy_epochs > 0:
        model = fit_sindy_second_stage(
            model=model,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            learning_rate=original_lr,
            epochs=sindy_epochs,
            cutoff_threshold=sindy_threshold,
            cutoff_n_terms=sindy_threshold_terms,
            cutoff_patience=sindy_threshold_patience,
            cutoff_warmup=sindy_epochs//4,
            sindy_alpha=0.001,
            batch_size=None,
            verbose=verbose,
        )
        
    if verbose:
        msg = f"L(Train): {loss_train:.7f}"
        
        if epochs > 0 and dataset_test is not None:
            msg += f" --- L(Val, RNN): {loss_test_rnn:.7f}"
        
        if sindy_weight > 0 and dataset_test is not None:
            model = model.eval(use_sindy=True)
            with torch.no_grad():
                xs, ys = next(iter(dataloader_test))
                if xs.device != model.device:
                    xs = xs.to(model.device)
                    ys = ys.to(model.device)
                # evaluate model
                _, _, loss_test_sindy = batch_train(model=model, xs=xs, ys=ys)
            
            msg += f" --- L(Val, SINDy): {loss_test_sindy:.7f}"
        
        if lr_scheduler is not None:
            msg += f" --- LR: {lr_scheduler.get_last_lr()[-1]:.7e}"
        
        print("\nTraining result:")
        print(msg)
        
    return model.eval(), optimizer, loss_train

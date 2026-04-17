
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
from .model import BaseModel
from .spice_utils import SpiceDataset
from .sindy_differentiable import get_library_term_degrees


# -----------------------------------------------------------------------------------------
# AUXILIARY/UTILITY FUNCTIONS
# -----------------------------------------------------------------------------------------

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
    loss_test: float,
    time_elapsed: float,
    convergence_value: float,
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
    postfix_parts['L(Train)'] = f'{loss_train:.5f}'
    if loss_test is not None:
        postfix_parts['L(Val)'] = f'{loss_test:.5f}'
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
    # if sindy_weight > 0:
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


# -----------------------------------------------------------------------------------------
# TRAINING ENHANCEMENT (LEARNING RATE SCHEDULER, WARMUP, ETC.)
# -----------------------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# -----------------------------------------------------------------------------------------

def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Wrapper for torch's cross entropy loss which does all the reshaping when getting SpiceDataset.ys tensors as predicitons and targets."""
    n_actions = target.shape[-1]
    
    prediction = prediction.reshape(-1, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)
    
    return torch.nn.functional.cross_entropy(prediction, target)


# -----------------------------------------------------------------------------------------
# PRUNING FUNCTIONS
# -----------------------------------------------------------------------------------------

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
            model.sindy_coefficients_presence[module] &= ~prune_e
            counters.data *= (~prune_e).int()
            pruned = True

    return model, pruned


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
    unfolded = model._get_unfolded_coefficients_for_pruning()

    if verbose:
        print("Ensemble confidence filtering:")

    for module in model.submodules_rnn:
        coeffs = unfolded[module]  # (E, P, X, n_terms) — unfolded from RNN weights
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


# -----------------------------------------------------------------------------------------
# CORE TRAINING FUNCTIONS
# -----------------------------------------------------------------------------------------

def _run_batch_training(
    model: BaseModel,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
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

    model.aggregate = False
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

        loss_step = loss_fn(ys_pred, ys_step)

        if torch.is_grad_enabled():
            optimizer.zero_grad()
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_batch += loss_step.item()
        iterations += 1

    return model, optimizer, loss_batch/iterations


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

    sindy_pruning_frequency: int = None,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_population_pruning: float = None,

    convergence_threshold: float = 0,
    verbose: bool = False,
    keep_log: bool = False,
    path_save_checkpoints: str = None,
) -> Tuple[BaseModel, torch.optim.Optimizer, float, float]:
    """
    RNN training with polynomial coefficient pruning.

    Trains RNN to predict behavior (behavioral loss only + L2 weight decay
    via AdamW). Periodic pruning via minimum-effect CI test on unfolded
    polynomial coefficients: a term survives iff |mean| - t_crit * SE > delta.

    Returns:
        Tuple of (model, optimizer, loss_train, loss_test)
    """

    B_total = xs_train.shape[1]
    iterations_per_epoch = max(B_total, 64) // batch_size if batch_size < max(B_total, 64) else 1

    dataloader_test = None
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))

    lr_scheduler = _setup_lr_scheduler(optimizer=optimizer) if use_scheduler else None

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
    loss_test = None
    use_pruning = sindy_pruning_frequency is not None and (sindy_ensemble_pruning is not None or sindy_threshold_pruning is not None)
    is_notebook = _is_notebook()

    # Main training loop
    while continue_training:
        try:
            if epochs > 0:
                loss_train = 0
                t_start = time.time()

                # Training iterations for this epoch
                for _ in range(iterations_per_epoch):
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
                        loss_fn=loss_fn,
                    )
                    loss_train += loss_i

                n_calls_to_train_model += 1
                loss_train /= iterations_per_epoch

            # Validation
            if dataloader_test is not None:
                model = model.eval()
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    if xs.device != model.device:
                        xs = xs.to(model.device)
                        ys = ys.to(model.device)
                    _, _, loss_test = _run_batch_training(model=model, xs=xs.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), ys=ys.unsqueeze(0).repeat(model.ensemble_size, 1, 1, 1, 1), loss_fn=loss_fn)

                model = model.train()

            # Pruning on unfolded polynomial coefficients
            if use_pruning and n_calls_to_train_model >= n_warmup_steps:

                if (sindy_ensemble_pruning is None or model.ensemble_size == 1) and sindy_threshold_pruning is not None:
                    model.sindy_coefficient_patience(threshold=sindy_threshold_pruning)

                if n_calls_to_train_model % sindy_pruning_frequency == 0:
                    if sindy_ensemble_pruning is not None and model.ensemble_size > 1:
                        model, pruned = _ensemble_pruning(
                            model=model,
                            sindy_ensemble_pruning=sindy_ensemble_pruning,
                            sindy_threshold_pruning=sindy_threshold_pruning,
                            verbose=verbose,
                        )
                    elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                        model.sindy_coefficient_pruning(patience=sindy_pruning_frequency)

            # Check convergence
            dloss = last_loss - (loss_test if dataloader_test is not None else loss_train)
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = loss_test if dataloader_test is not None else loss_train

            # Update learning rate scheduler
            if lr_scheduler is not None and n_calls_to_train_model >= n_warmup_steps:
                metric = loss_test if dataloader_test is not None else loss_train
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
                    loss_test=loss_test if dataloader_test is not None else None,
                    time_elapsed=time.time() - t_start_total,
                    convergence_value=convergence_value,
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

    return model, optimizer, loss_train, loss_test


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

    sindy_pruning_frequency: int = 1,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_population_pruning: float = None,

    verbose: bool = True,
    keep_log: bool = False,
    n_warmup_steps: int = 0,
    path_save_checkpoints: str = None,
) -> Tuple[BaseModel, torch.optim.Optimizer, float]:
    """
    SPICE training pipeline: behavioral loss + L2 weight decay + polynomial pruning.

    Trains the polynomial RNN on behavioral prediction loss (cross-entropy by
    default). Regularization comes from AdamW's L2 weight decay which implicitly
    penalizes higher-degree polynomial terms more. Periodic pruning applies
    minimum-effect CI tests on unfolded polynomial coefficients.

    Args:
        model: RNN model
        dataset_train: Training dataset
        dataset_test: Validation dataset (optional)
        optimizer: PyTorch optimizer
        epochs: Total training epochs
        batch_size: Training batch size (None = auto-detect)
        scheduler: Enable learning rate scheduler
        n_steps: BPTT truncation length
        convergence_threshold: Early stopping threshold
        loss_fn: Loss function for behavioral prediction
        sindy_pruning_frequency: Epochs between pruning events
        sindy_threshold_pruning: Minimum effect size (delta) for CI test
        sindy_ensemble_pruning: Confidence level for ensemble CI test
        sindy_population_pruning: Cross-participant presence threshold (0-1)
        verbose: Print progress
        keep_log: Keep full training log
        n_warmup_steps: Warmup epochs (no pruning during warmup)
        path_save_checkpoints: Path for saving checkpoints

    Returns:
        Tuple of (trained_model, optimizer)
    """

    if n_warmup_steps is None:
        if epochs is not None:
            n_warmup_steps = epochs // 4
        else:
            n_warmup_steps = 0

    use_pruning = sindy_pruning_frequency is not None and (sindy_ensemble_pruning is not None or sindy_threshold_pruning is not None)

    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n" + status_lines)
        print("SPICE Training Configuration:")
        print(f"\tRNN training: [{'x' if epochs > 0 else ' '}]")
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
    # Training with polynomial pruning
    # ══════════════════════════════════════════════════════════════════════════
    if epochs > 0:
        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "=" * terminal_width)
            print("SPICE training" + (" with polynomial pruning" if use_pruning else ""))
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

                    sindy_threshold_pruning=sindy_threshold_pruning,
                    sindy_pruning_frequency=sindy_pruning_frequency,
                    sindy_ensemble_pruning=sindy_ensemble_pruning,
                    sindy_population_pruning=sindy_population_pruning,

                    verbose=verbose,
                    keep_log=keep_log,
                    path_save_checkpoints=path_save_checkpoints,
                )
                model, optimizer, loss_train, loss_test = results
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
    # Final evaluation summary
    # ══════════════════════════════════════════════════════════════════════════
    if verbose:
        status_lines = "=" * _get_terminal_width()
        print("\n" + status_lines)
        print("Training results:")
        msg = "\t"

        if epochs > 0:
            msg += f"L(Train): {loss_train:.7f}"

        if dataset_test is not None:
            with torch.no_grad():
                _, _, _, loss_test = _run_joint_training(
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

                    sindy_threshold_pruning=None,
                    sindy_pruning_frequency=None,
                    sindy_ensemble_pruning=None,
                    sindy_population_pruning=None,

                    verbose=False,
                    keep_log=False,
                    path_save_checkpoints=None,
                )

            msg += f"\n\tL(Val): {loss_test:.7f}"

        print(msg)
        print(status_lines)

    return model.eval(), optimizer
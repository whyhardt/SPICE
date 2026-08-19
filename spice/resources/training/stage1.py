"""Stage 1: joint RNN + SINDy training against behaviour."""

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
from .reporting import _get_terminal_width, _check_cuda_oom, _print_training_status, _is_notebook, DEBUG_MODE
from .losses import cross_entropy_loss, _setup_warmup_scaler
from .pruning import _ensemble_pruning
from .shooting import _project_after_step


def _rnn_lr(optimizer: torch.optim.Optimizer) -> float:
    """Current RNN learning rate, looked up by group role rather than by index."""
    for group in optimizer.param_groups:
        if group.get('role') == 'rnn':
            return group['lr']
    return optimizer.param_groups[-1]['lr']


def _run_batch_training(
    model: BaseModel,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    sindy_weight: float = 0.,
    sindy_weight_fit: float = 0.1,
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

    # state=None on the first chunk so init_forward_pass applies learnable
    # per-participant initial values (model.py's else-branch there is the only
    # place that reads them) -- pre-populating via init_state()/get_state()
    # here would always hand forward() a non-None prev_state, permanently
    # bypassing that branch and leaving learnable_initial_values disconnected
    # from the loss. Later BPTT-truncation chunks (t>0) still carry the real
    # previous state forward as before.
    state = None

    loss_batch = 0
    iterations = 0
    for t in range(0, xs.shape[2], n_steps):
        n_steps = min(xs.shape[2]-t, n_steps)
        xs_step = xs[:, :, t:t+n_steps]
        ys_step = ys[:, :, t:t+n_steps]

        ys_pred, _ = model(xs_step, state)
        state = model.get_state(detach=True)

        # Mask out padding (NaN values)
        # xs_step is 5D: (E, B, T_out, T_in, F)
        mask = ~torch.isnan(xs_step[..., :model.n_actions].sum(dim=(-1)))
        ys_pred = ys_pred[mask]
        ys_step = ys_step[mask]

        loss_step = loss_fn(ys_pred, ys_step, **loss_fn_kwargs)

        if torch.is_grad_enabled():
            # Add SINDy losses (decoupled gradients: sindy_loss_reg -> RNN, sindy_loss_fit -> concept factorization)
            if sindy_weight > 0 and model.sindy_loss_reg != 0:
                loss_step = loss_step + sindy_weight * model.sindy_loss_reg #+ sindy_weight_fit * model.sindy_loss_fit

            if sindy_weight > 0 and sindy_alpha > 0:
                loss_step = loss_step + model.compute_constants_penalty(sindy_alpha=sindy_alpha)
                
            # backpropagation
            optimizer.zero_grad()
            loss_step.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Clip RNN and SINDy gradients independently so unweighted
            # sindy_loss_fit doesn't starve RNN gradients via shared norm
            # if len(optimizer.param_groups) > 1:
            #     torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
            #     torch.nn.utils.clip_grad_norm_(optimizer.param_groups[1]['params'], max_norm=1.0)
            # else:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # L1 on the loadings is applied proximally rather than as a loss term, and
            # the unit-norm gauge on the directions must be re-fixed every step or the
            # penalty can be defeated by inflating V.
            _project_after_step(model, optimizer, sindy_alpha if sindy_weight > 0 else 0.0)

        loss_batch += loss_step.item()
        # if sindy_weight > 0 and model.sindy_loss_reg != 0:
        #     loss_batch -= model.sindy_loss_fit.item() * sindy_weight_fit
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
    loss_fn: callable = cross_entropy_loss,
    loss_fn_kwargs: dict = {},

    sindy_weight: float = 0,
    sindy_alpha: float = 0,
    sindy_pruning_frequency: int = None,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_pruning_terms: int = None,

    convergence_threshold: float = 0,
    verbose: bool = False,
    keep_log: bool = False,
    path_save_checkpoints: str = None,
) -> Tuple[BaseModel, torch.optim.Optimizer, float, float, float]:
    """
    Joint RNN-SINDy optimization with concept pruning.

    Trains the RNN to predict behaviour while the SINDy branch regularizes its
    submodule dynamics toward the concept dictionary. Pruning fires periodically
    past warmup and acts on two levels: concept gates per unit, and concept
    support across the population. When the ensemble has more than one member and
    `sindy_ensemble_pruning` is set, gate decisions go through the ensemble ratio
    test (a concept survives iff at least that fraction of members load on it
    above `sindy_threshold_pruning`); otherwise per-member thresholding is used.

    Objective: L_total = L_CE(y, y_hat) + sindy_weight * L_SINDy, with the L1 on
    the loadings applied as a proximal step after the optimizer rather than as a
    loss term, so it produces exact zeros.

    Returns:
        Tuple of (model, optimizer, loss_train, loss_test_rnn, loss_test_sindy)
    """
    
    B_total = xs_train.shape[1]
    iterations_per_epoch = max(B_total, 64) // batch_size if batch_size < max(B_total, 64) else 1
    
    dataloader_test = None
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))

    warmup_scaler_sindy_weight = _setup_warmup_scaler(n_warmup_steps=n_warmup_steps, exp_max=5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=50, 
        # Floors are assigned by group role, not by position: the estimator builds three
        # groups (directions, loadings, rnn) and a positional list silently mis-assigns
        # them the moment that count or order changes.
        min_lr=[
            group['lr'] if group.get('role', 'rnn') != 'rnn' else 1e-5
            for group in optimizer.param_groups
        ],
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
            # lr_scheduler.step(n_calls_to_train_model)

            if epochs > 0:
                loss_train = 0
                t_start = time.time()

                # Compute warmup-scaled SINDy weights
                if n_calls_to_train_model >= n_warmup_steps:
                    sindy_weight_epoch = sindy_weight
                    sindy_alpha_epoch = sindy_alpha
                    sindy_weight_fit_epoch = 1.0
                else:
                    warmup_scale = warmup_scaler_sindy_weight[n_calls_to_train_model]
                    sindy_weight_epoch = sindy_weight * warmup_scale
                    # Ramp the L1 prox in alongside the SINDy weight. Loadings start
                    # small and strictly positive; a full-strength prox from step 0
                    # would shrink them to exactly zero, where they get no gradient and
                    # their concepts are dead for the rest of the run.
                    sindy_alpha_epoch = sindy_alpha * warmup_scale
                    sindy_weight_fit_epoch = warmup_scale

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
                        sindy_weight_fit=sindy_weight_fit_epoch,
                        sindy_alpha=sindy_alpha_epoch,
                        loss_fn=loss_fn,
                        loss_fn_kwargs=loss_fn_kwargs,
                    )
                    loss_train += loss_i

                n_calls_to_train_model += 1
                loss_train /= iterations_per_epoch
                
                lr_scheduler.step(loss_train)

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
            
            # Pruning event: concept gates per unit, concept support per population
            if (sindy_weight > 0
                and sindy_pruning_frequency is not None
                # and n_calls_to_train_model >= n_warmup_steps
                ):

                if ((sindy_ensemble_pruning is None or model.ensemble_size==1)
                    and sindy_threshold_pruning is not None
                    and n_calls_to_train_model >= n_warmup_steps
                    ):
                    # Fallback: per-epoch patience tracking for per-member threshold pruning
                    model.concept_gate_patience(threshold=sindy_threshold_pruning)
                    model.concept_support_patience(threshold=sindy_threshold_pruning)

                
                if (n_calls_to_train_model % sindy_pruning_frequency == 0
                    or n_calls_to_train_model == 1
                    # and n_calls_to_train_model >= n_warmup_steps
                    ):
                    
                    # pruning
                    if n_calls_to_train_model >= n_warmup_steps:
                        if (sindy_ensemble_pruning is not None 
                            and model.ensemble_size > 1
                            ):
                            model, _ = _ensemble_pruning(
                                model=model,
                                sindy_ensemble_pruning=sindy_ensemble_pruning,
                                sindy_threshold_pruning=sindy_threshold_pruning,
                                n_terms_pruning=sindy_pruning_terms,
                                verbose=verbose,
                                )

                        elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                            # Fallback: per-member threshold pruning only (no ensemble test)
                            model.prune_concept_gates(patience=sindy_pruning_frequency, n_concepts_pruning=sindy_pruning_terms)

                        # Term-level structure is a population decision and runs on every
                        # pruning event, independently of the per-unit gate pruning above.
                        if sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                            model.prune_concept_support(patience=sindy_pruning_frequency, n_terms_pruning=sindy_pruning_terms)

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
                    lr=_rnn_lr(optimizer),
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

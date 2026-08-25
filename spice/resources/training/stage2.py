"""Stage 2: refit the SINDy equations on the frozen RNN's trajectories.

Stage 2.1 discovers structure (sparsity / concepts); stage 2.2 estimates the
coefficients within the discovered structure.
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
from .reporting import _get_terminal_width, _check_cuda_oom, _print_training_status, _spice_parameter_postfix
from .losses import cross_entropy_loss
from .trajectories import _vectorize_state_sequential, _flatten_state_trajectories_onestep
from .ridge import _ridge_solve_sindy
from .shooting import _run_shooting_epoch_vectorized, _run_shooting_eval_batched
from .pruning import _ensemble_pruning


def _run_sindy_training(
    model: BaseModel,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    xs_train_original: torch.Tensor = None,
    ys_train_original: torch.Tensor = None,
    epochs: int = 1000,
    n_warmup_steps: int = 100,
    sindy_lambda_loading: float = None,
    sindy_lambda_concept: float = None,
    sindy_pruning_frequency: int = None,
    sindy_ensemble_pruning: float = None,
    sindy_threshold_pruning: float = None,
    sindy_pruning_terms: int = None,
    shooting_steps: int = 20,
    sindy_ridge: bool = True,
    verbose: bool = True,
    ):

    """
    Two-phase SINDy refit on the frozen RNN's hidden-state trajectories.

    Stage 2.1 — Structure discovery:
        Re-draw the concept factorization from scratch (full support, all gates
        open, random unit-norm directions, small positive loadings) rather than
        inheriting whatever Stage 1 converged to, then train with one-step
        teacher forcing plus pruning and the L1 prox on both halves of the
        factorization. Pruning acts on both levels: concept gates per unit,
        concept support across the population. LR warms at 0.01 and is boosted
        back after pruning events.

    Stage 2.2 — Loading estimation:
        Freeze the discovered structure (directions, support and gates),
        re-draw only the loadings, and fit by shooting (K=shooting_steps)
        without pruning or L1.

    Stage 2.1 uses bootstrapped data. Stage 2.2 optionally re-computes
    trajectories on the full (non-bootstrapped) dataset and averages them
    across ensemble members to create consensus targets.

    Args:
        model: Trained RNN model carrying the concept factorization
        xs_train: 5D training data (E, B, T, W, F) — bootstrapped
        ys_train: 5D training targets — bootstrapped
        xs_train_original: 4D original training data (B, T, W, F) before
            bootstrapping. If provided and E > 1, Stage 2.2 re-computes
            trajectories on full data with ensemble-averaged targets.
        ys_train_original: 4D original training targets
        epochs: Training epochs per stage (default: 1000)
        n_warmup_steps: Warmup epochs per stage (default: 100)
        sindy_lambda_loading: L1 strength for the proximal step on the loadings. Applied
            as shrinkage of lr * sindy_lambda_loading per step, so it is not comparable
            across different LR schedules.
        sindy_lambda_concept: L1 strength for the proximal step on the concept directions,
            applied the same way. Sets how dense each concept's support is
            allowed to be; without it the penalty on the loadings alone drives
            the directions toward maximum density.
        sindy_pruning_frequency: Epochs between pruning events in Stage 2.1
        sindy_ensemble_pruning: Minimum fraction of ensemble members that must
            load on a concept for it to survive the ensemble ratio test
        sindy_threshold_pruning: Loading magnitude below which a concept gate
            becomes a pruning candidate. Measured against unit-norm directions,
            so it lives on the loading scale, not the raw coefficient scale.
        shooting_steps: Rollout horizon K for Stage 2.2. (default: 20)
        verbose: Print progress

    Returns:
        BaseModel: Model with a refitted concept factorization
    """
    _E, B, T, W, F = xs_train.shape
    E = model.ensemble_size

    # Collect RNN state trajectories (shared across both stages)
    model.eval(use_sindy=False)
    state_trajectories, nan_mask = _vectorize_state_sequential(model, xs_train, ys_train, verbose=verbose)

    # LR constants (same schedule as Stage 1)
    lr_base = 0.01
    lr_warmup = 0.01
    lr_post_pruning = 0.01
    pruning_boost_duration = max(1, int(0.1 * sindy_pruning_frequency)) if sindy_pruning_frequency else 0

    has_sparsity = (sindy_ensemble_pruning is not None
                    or sindy_threshold_pruning is not None)

    # ── Stage 2.1: Structure discovery (K=1, with pruning) ───────────────
    if has_sparsity:
        # Auto-compute pruning rate for Stage 2.1 budget
        if sindy_pruning_terms is None and sindy_pruning_frequency is not None:
            total_terms = sum(model.sindy_concept_gates[m].shape[-1] for m in model.submodules_rnn)
            n_pruning_events = max(1, (epochs - n_warmup_steps) // max(1, sindy_pruning_frequency))
            sindy_pruning_terms = math.ceil(total_terms / n_pruning_events)

        if verbose:
            terminal_width = _get_terminal_width()
            print("\n" + "=" * terminal_width)
            print(f"Stage 2.1: SINDy structure discovery (K={1})")
            print("=" * terminal_width)

        # Re-draw the factorization for full exploration; reset_concepts respects
        # the theory-driven term prior mask (e.g. binary^2 terms)
        model.reset_concepts()

        # Genuine one-step (dw=1) fitting: flatten every (w, t) transition into
        # its own independent T'=1, W'=1 pseudo-session, so K=1 here means one
        # teacher-forced step regardless of whether this model's sequential
        # structure lives in W (within-trial dynamics) or T (across-trial).
        xs_21, state_trajectories_21, nan_mask_21, B_21 = _flatten_state_trajectories_onestep(
            xs_train, state_trajectories, nan_mask,
        )
        K_21 = 1
        window_starts_21 = [0]

        # Ridge solve with L2 penalty to initialize the loadings. Uses the
        # original (non-flattened) xs_train -- sindy_ridge_accumulate's h_current is
        # already the per-step preceding value regardless of how many within-trial
        # steps a single forward call spans, so this is already one-step-correct.
        ridge_success_21 = _ridge_solve_sindy(model, xs_train, ys_train)

        model.fit_sindy = False
        model.train(use_sindy=True)
        for rnn_module in model.submodules_rnn.values():
            rnn_module.eval()

        batch_size_sessions = B_21
        if ridge_success_21:
            # Evaluate ridge solution with the flattened one-step loss
            with torch.no_grad():
                ridge_loss_21, batch_size_sessions = _run_shooting_eval_batched(
                    model=model,
                    xs_train=xs_21,
                    state_trajectories=state_trajectories_21,
                    nan_mask=nan_mask_21,
                    window_starts=window_starts_21,
                    K=K_21,
                    B_total=B_21,
                    sindy_lambda_loading=sindy_lambda_loading,
                    batch_size_sessions=batch_size_sessions,
                )
            # A closed-form solve can "succeed" (no LinAlgError) while still
            # producing coefficients that are non-finite once evaluated --
            # don't hand those to SGD as a starting point.
            if not math.isfinite(ridge_loss_21):
                ridge_success_21 = False
            elif verbose:
                print(f"Ridge initialization succeeded (K={K_21} loss: {ridge_loss_21:.7f}). Running SGD refinement...")

        if not ridge_success_21:
            model.reinit_loadings()
            if verbose:
                print("Ridge initialization failed. Starting from random init.")

        sindy_parameters = [p for name, p in model.named_parameters() if 'sindy' in name]
        optimizer_21 = torch.optim.AdamW(sindy_parameters, lr=lr_warmup, weight_decay=0)
        scheduler_21 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_21, mode='min', factor=0.5, patience=10, min_lr=1e-5,
        )

        lr_boost_end = 0  # epoch at which post-pruning LR boost expires

        while True:
            try:
                pbar = tqdm(range(epochs))
                for epoch in pbar:
                    session_perm = torch.randperm(B_21)

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

                    for b_start in range(0, B_21, batch_size_sessions):
                        b_end = min(b_start + batch_size_sessions, B_21)
                        batch_sessions = session_perm[b_start:b_end]
                        loss_e = _run_shooting_epoch_vectorized(
                            model=model,
                            optimizer=optimizer_21,
                            xs_train=xs_21,
                            state_trajectories=state_trajectories_21,
                            nan_mask=nan_mask_21,
                            window_starts=window_starts_21,
                            K=K_21,
                            batch_sessions=batch_sessions,
                            sindy_lambda_loading=sindy_lambda_loading,
                            sindy_lambda_concept=sindy_lambda_concept,
                        )
                        loss_epoch += loss_e
                        n_batches += 1

                    if n_batches > 0:
                        loss_epoch /= n_batches

                    scheduler_21.step(loss_epoch)

                    pbar.set_postfix(
                        loss=f"{loss_epoch:.7f}",
                        lr=f"{optimizer_21.param_groups[0]['lr']:.1e}",
                        **_spice_parameter_postfix(model),
                    )

                    # Pruning
                    if sindy_pruning_frequency is not None:
                        if sindy_threshold_pruning is not None and epoch >= n_warmup_steps:
                            # Support patience is ensemble-independent (V is shared), so it
                            # advances in both modes; gate patience is the non-ensemble path.
                            model.concept_support_patience(threshold=sindy_threshold_pruning)

                            if sindy_ensemble_pruning is None or model.ensemble_size == 1:
                                model.concept_gate_patience(threshold=sindy_threshold_pruning)

                        if (epoch % sindy_pruning_frequency == 0 or epoch == 1) and epoch >= n_warmup_steps:
                            pruned = False
                            if sindy_ensemble_pruning is not None and model.ensemble_size > 1:
                                model, pruned = _ensemble_pruning(
                                    model=model,
                                    sindy_ensemble_pruning=sindy_ensemble_pruning,
                                    sindy_threshold_pruning=sindy_threshold_pruning,
                                    verbose=verbose,
                                )
                            elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                                model.prune_concept_gates(patience=sindy_pruning_frequency, n_concepts_pruning=sindy_pruning_terms)
                                pruned = True

                            # Term-level structure is decided for the population, not per
                            # unit, so support pruning runs on every pruning event
                            # alongside the per-unit gate pruning above.
                            if sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                                model.prune_concept_support(
                                    patience=sindy_pruning_frequency,
                                    n_terms_pruning=sindy_pruning_terms,
                                )
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

    # Switch to full (non-bootstrapped) data with per-member targets
    if xs_train_original is not None and E > 1:
        if verbose:
            print("Re-computing state trajectories on full data (per-member targets)...")
        xs_full_5d = xs_train_original.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
        ys_full_5d = ys_train_original.unsqueeze(0).expand(E, -1, -1, -1, -1).contiguous()
        model.eval(use_sindy=False)
        state_trajectories, nan_mask = _vectorize_state_sequential(
            model, xs_full_5d, ys_full_5d, verbose=verbose
        )
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

    if verbose:
        terminal_width = _get_terminal_width()
        print("\n" + "=" * terminal_width)
        # K here is the across-trial rollout horizon (clamped to T, the number of
        # outer trials) -- it says nothing about within-trial resolution. Any
        # within-trial (W) dynamics are already rolled out autoregressively
        # inside each of these K forward calls (see call_module's SINDy branch),
        # so a model with T=1 and W>1 (e.g. this study) still gets full W-length
        # supervision per step even though K clamps to 1.
        if K > 1:
            print(f"Stage 2.2: SINDy loading estimation (multi-step shooting across trials, K={K}, W={W})")
        else:
            print(f"Stage 2.2: SINDy loading estimation (one-step-ahead across trials, W={W})")
        print("=" * terminal_width)

    # Re-draw only the loadings; directions, support and gates stay frozen
    for module in model.get_modules():
        model.reinit_loadings(module)

    # ── Ridge regression (closed-form one-step solve) ──
    ridge_success = False
    batch_size_sessions = B
    if sindy_ridge:
        ridge_success = _ridge_solve_sindy(model, xs_train, ys_train)
        if ridge_success:
            sgd_epochs = min(1000, epochs)
            # Evaluate ridge solution with K-step shooting loss
            model.fit_sindy = False
            model.train(use_sindy=True)
            for rnn_module in model.submodules_rnn.values():
                rnn_module.eval()
            with torch.no_grad():
                ridge_loss, batch_size_sessions = _run_shooting_eval_batched(
                    model=model,
                    xs_train=xs_train,
                    state_trajectories=state_trajectories,
                    nan_mask=nan_mask,
                    window_starts=window_starts,
                    K=K,
                    B_total=B,
                    sindy_lambda_loading=sindy_lambda_loading,
                    batch_size_sessions=batch_size_sessions,
                )
            # A closed-form solve can "succeed" (no LinAlgError) while still
            # producing a coefficient set that's unstable under the K-step
            # rollout used to evaluate it -- e.g. a self-coefficient a>0 blows
            # up to inf/NaN over many steps even though the linear solve
            # itself was well-posed. Treat that the same as an outright ridge
            # failure rather than handing NaN-producing coefficients to SGD.
            if not math.isfinite(ridge_loss):
                ridge_success = False
                ridge_fail_reason = f"solved but produced a non-finite loss (K={K}: {ridge_loss})"
            elif verbose:
                print(f"Ridge regression succeeded (K={K} loss: {ridge_loss:.7f}). Running SGD refinement...")
        else:
            ridge_fail_reason = "failed"

        if not ridge_success:
            # Re-draw the loadings since ridge may have partially written
            for module in model.get_modules():
                model.reinit_loadings(module)
            if verbose:
                print(f"Ridge regression {ridge_fail_reason}. Falling back to full SGD...")

    if not ridge_success:
        sgd_epochs = epochs

    # ── SGD shooting (refinement after ridge, or full fallback) ──
    sgd_lr_init = lr_base if ridge_success else lr_warmup
    sindy_parameters = [p for name, p in model.named_parameters() if 'sindy' in name]
    optimizer_22 = torch.optim.AdamW(sindy_parameters, lr=sgd_lr_init, weight_decay=0)
    scheduler_22 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_22, mode='min', factor=0.5, patience=10, min_lr=1e-5,
    )

    model.fit_sindy = False
    model.train(use_sindy=True)
    for rnn_module in model.submodules_rnn.values():
        rnn_module.eval()

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
                        sindy_lambda_loading=None,  # unpenalized
                    )
                    loss_epoch += loss_e
                    n_batches += 1

                if n_batches > 0:
                    loss_epoch /= n_batches

                scheduler_22.step(loss_epoch)

                pbar.set_postfix(
                    loss=f"{loss_epoch:.7f}",
                    lr=f"{optimizer_22.param_groups[0]['lr']:.1e}",
                    **_spice_parameter_postfix(model),
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

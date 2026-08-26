"""fit_spice: the two-stage SPICE training pipeline orchestrator."""

import os
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
from .reporting import _get_terminal_width, _print_training_status, _check_cuda_oom
from .losses import cross_entropy_loss
from .pruning import compute_pruning_budgets
from .stage1 import _run_joint_training
from .stage2 import _run_sindy_training


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

    sindy_weight: float = 0.,
    sindy_lambda_loading: float = 0.,
    sindy_lambda_concept: float = 0.,
    sindy_pruning_frequency: int = 1,
    sindy_threshold_pruning: float = None,
    sindy_ensemble_pruning: float = None,
    sindy_pruning_terms: int = None,
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

    Stage 1 (Joint Training with Minimum-Effect Agreement Pruning):
        Train RNN + SINDy jointly with L_CE + λ_sindy * L_SINDy + α * L_penalty.
        Periodic pruning via minimum-effect agreement test: a term survives iff
        sum_e (|coef| > sindy_threshold_pruning) >= E * sindy_ensemble_pruning.
        This unifies statistical significance and practical significance in
        a single test, avoiding the cascade where per-member threshold pruning
        erodes ensemble consensus. Terms must fail 2 consecutive pruning events
        before permanent removal.

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
        sindy_weight: λ_sindy regularization strength
        sindy_lambda_loading: L1 strength on the concept loadings (Z), as a term in the objective
        sindy_lambda_concept: L1 strength on the concept directions (V), as a term in the objective.
            Controls how dense each concept's support is; with sindy_lambda_loading alone the
            objective is minimised by maximally dense concepts
        sindy_threshold_pruning: Minimum |coefficient| for a member to count as
            supporting a term in the ensemble ratio test. When
            sindy_ensemble_pruning is None, falls back to per-member hard
            thresholding. (None or 0 to disable; default: None)
        sindy_pruning_frequency: Epochs between pruning events
        sindy_ensemble_pruning: Minimum fraction of ensemble members that must
            load on a concept above sindy_threshold_pruning for it to survive
            (ensemble ratio test). Primary pruning mechanism. None to disable.
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
            pruning_details.append(f"ratio test ratio={sindy_ensemble_pruning}")
            if sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
                pruning_details.append(f"delta={sindy_threshold_pruning}")
        elif sindy_threshold_pruning is not None and sindy_threshold_pruning > 0:
            pruning_details.append(f"threshold={sindy_threshold_pruning} (per-member)")
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

    # Per-event pruning budgets, computed once from the initial factorization: the
    # gates (Z) and the concept support (V) each get a rate that reaches 0 over the
    # expected number of pruning events, independently of one another.
    # Stage 2 sizes its own budgets from its own epoch/warmup schedule.
    n_prune_z, n_prune_v = compute_pruning_budgets(
        model=model,
        epochs=epochs,
        n_warmup_steps=n_warmup_steps,
        sindy_pruning_frequency=sindy_pruning_frequency,
        override=sindy_pruning_terms,
    )

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

                    sindy_weight=sindy_weight,
                    sindy_lambda_loading=sindy_lambda_loading,
                    sindy_lambda_concept=sindy_lambda_concept,
                    sindy_threshold_pruning=sindy_threshold_pruning,
                    sindy_pruning_frequency=sindy_pruning_frequency,
                    sindy_ensemble_pruning=sindy_ensemble_pruning,
                    n_prune_z=n_prune_z,
                    n_prune_v=n_prune_v,

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
        
        # Save Stage 1 model checkpoint before Stage 2 re-draws the factorization
        # if path_save_checkpoints is not None:
        #     stage1_path = path_save_checkpoints.replace('.pkl', '_stage1.pkl')
        # elif hasattr(model, '_save_path') and model._save_path is not None:
        #     stage1_path = model._save_path.replace('.pkl', '_stage1.pkl')
        # else:
        #     stage1_path = None
        # if stage1_path is not None:
        #     os.makedirs(os.path.dirname(stage1_path) or '.', exist_ok=True)
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'sindy_concept_support': model.sindy_concept_support,
        #         'sindy_concept_gates': model.sindy_concept_gates,
        #     }, stage1_path)
        #     if verbose:
        #         print(f"\nStage 1 model saved to: {stage1_path}")        
                

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Final SINDy Refit
    # ══════════════════════════════════════════════════════════════════════════
    if sindy_refit:
        _run_sindy_training(
            model=model,
            xs_train=xs_train_5d.to(torch.device('cpu')),
            ys_train=ys_train_5d.to(torch.device('cpu')),
            xs_train_original=dataset_train.xs,
            ys_train_original=dataset_train.ys,
            epochs=1000,
            n_warmup_steps=100,
            sindy_lambda_loading=sindy_lambda_loading,
            sindy_lambda_concept=sindy_lambda_concept,
            sindy_pruning_frequency=sindy_pruning_frequency,
            sindy_ensemble_pruning=sindy_ensemble_pruning,
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
                sindy_lambda_loading=0,
                sindy_threshold_pruning=None,
                sindy_pruning_frequency=None,
                sindy_ensemble_pruning=None,

                verbose=False,
                keep_log=False,
                path_save_checkpoints=None,
            )
           
        if dataset_test is not None:
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
                sindy_lambda_loading=0,
                sindy_threshold_pruning=None,
                sindy_pruning_frequency=None,
                sindy_ensemble_pruning=None,

                verbose=False,
                keep_log=False,
                path_save_checkpoints=None,
            )

            msg_result = "\t         Training    Validation"
            msg_result += f"\n\tRNN      {loss_train_rnn:.5f}     {loss_test_rnn:.5f}"
            msg_result += f"\n\tSINDy    {loss_train_sindy:.5f}     {loss_test_sindy:.5f}"
        else:
            msg_result = "\t         Training"
            msg_result += f"\n\tRNN      {loss_train_rnn:.5f}"
            msg_result += f"\n\tSINDy    {loss_train_sindy:.5f}"
        print(msg_result)
        print(status_lines)

    return model.eval(use_sindy=True), optimizer

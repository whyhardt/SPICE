import os
import math
import sys
from typing import Callable, Optional

import argparse
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# standard methods and classes used for every model evaluation
from spice import SpiceEstimator, csv_to_dataset, SpiceDataset


def get_choice_probs(logits: torch.Tensor) -> torch.Tensor:
    # softmax normalization
    return torch.softmax(logits, dim=-1)


def log_likelihood(data: torch.tensor, probs: torch.tensor, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1 
    
    # Sum over all data points
    # return torch.sum(torch.sum(data * torch.log(probs), axis=-1), axis=axis) / normalization
    # Ensure probabilities are within a valid range to prevent log(0)
    epsilon = 1e-9
    probs = torch.clip(probs, epsilon, 1 - epsilon)
    
    # Calculate log-likelihood for each observation
    log_likelihoods = data * torch.log(probs)
    
    # Sum log-likelihoods over all observations
    return log_likelihoods


def bayesian_information_criterion(data: torch.Tensor, probs: torch.Tensor, n_parameters: int, nll: torch.Tensor = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if nll is None:
        nll = log_likelihood(data=data, probs=probs)
    
    # n_samples = (data[:, 0] != -1).sum()
    n_samples = (~torch.isnan(data[..., 0])).sum()
    return 2 * nll + n_parameters * torch.log(n_samples)


def akaike_information_criterion(data: torch.Tensor, probs: torch.Tensor, n_parameters: int, nll: torch.Tensor = None, **kwargs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if nll is None:
        nll = log_likelihood(data=data, probs=probs)
    
    return 2 * nll + 2 * n_parameters


def get_scores(probs: torch.Tensor, targets: torch.Tensor, n_parameters: int) -> tuple[tuple[float, float, float], torch.Tensor]:
    nll = -log_likelihood(data=targets, probs=probs)
    nll_sum = torch.nansum(nll)

    bic = bayesian_information_criterion(data=targets, probs=probs, n_parameters=n_parameters, nll=nll_sum)
    aic = akaike_information_criterion(data=targets, probs=probs, n_parameters=n_parameters, nll=nll_sum)


    return (nll_sum, aic, bic), nll


def get_participant_experiment_groups(dataset: SpiceDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each session to a (participant, experiment) group.

    Sessions belonging to the same participant/experiment share the same
    fitted parameters (per-subject benchmark parameters, or SPICE's
    per-participant SINDy coefficients), so BIC/AIC must be computed per
    group and then averaged across groups (see ``grouped_information_criteria``).

    Returns:
        unique_pairs: (G, 2) tensor of unique (participant_id, experiment_id) pairs.
        group_index: (B,) tensor mapping each session to its row in unique_pairs.
    """
    participant_ids = dataset.xs[:, 0, 0, -1].long().cpu()
    experiment_ids = dataset.xs[:, 0, 0, -2].long().cpu()
    pair_ids = torch.stack([participant_ids, experiment_ids], dim=1)
    unique_pairs, group_index = torch.unique(pair_ids, dim=0, return_inverse=True)
    return unique_pairs, group_index


def grouped_information_criteria(
    nll_per_session: torch.Tensor,
    n_trials_per_session: torch.Tensor,
    group_index: torch.Tensor,
    n_groups: int,
    n_parameters_per_group: torch.Tensor,
    n_actions_baseline: int,
) -> dict:
    """Compute BIC/AIC/ΔBIC-per-trial per (participant, experiment) group, then
    summarize as mean ± std across groups.

    Pooling NLL across the whole dataset while scaling the parameter count
    with the number of subjects (as is correct for per-subject-fit models,
    e.g. individually-fit benchmark parameters or SPICE's per-participant
    SINDy coefficients) makes such models lose against a fixed-size
    shared-weight model purely as dataset size grows, independent of actual
    fit quality. Computing BIC per group with that group's own trial count
    and own parameter count -- then averaging across groups -- removes that
    scaling artifact.
    """
    # All grouping/reduction happens on CPU: inputs may come from a mix of
    # devices (e.g. coefficients read off a CUDA model vs. CPU-side NLLs).
    nll_per_session = nll_per_session.cpu()
    n_trials_per_session = n_trials_per_session.float().cpu()
    group_index = group_index.cpu()
    n_parameters_per_group = n_parameters_per_group.float().cpu()

    nll_group = torch.zeros(n_groups, dtype=nll_per_session.dtype).scatter_add_(0, group_index, nll_per_session)
    n_trials_group = torch.zeros(n_groups).scatter_add_(0, group_index, n_trials_per_session)

    valid = n_trials_group > 0
    nll_group = nll_group[valid]
    n_trials_group = n_trials_group[valid]
    k_group = n_parameters_per_group[valid]

    bic_group = 2 * nll_group + k_group * torch.log(n_trials_group)
    aic_group = 2 * nll_group + 2 * k_group

    nll_random_group = n_trials_group * math.log(n_actions_baseline)
    bic_random_group = 2 * nll_random_group
    delta_bic_per_trial_group = (bic_random_group - bic_group) / n_trials_group

    def _mean_std(x: torch.Tensor) -> tuple[float, float]:
        return x.mean().item(), (x.std().item() if x.numel() > 1 else 0.0)

    bic_mean, bic_std = _mean_std(bic_group)
    aic_mean, aic_std = _mean_std(aic_group)
    dbic_mean, dbic_std = _mean_std(delta_bic_per_trial_group)

    return {
        'nll_total': nll_group.sum().item(),
        'bic_mean': bic_mean, 'bic_std': bic_std,
        'aic_mean': aic_mean, 'aic_std': aic_std,
        'delta_bic_per_trial_mean': dbic_mean, 'delta_bic_per_trial_std': dbic_std,
    }


@torch.no_grad()
def analysis_model_evaluation(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    trial_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    n_actions_random_baseline: Optional[int] = None,
    ):
    """
    Args:
        trial_filter: Optional function ``(ys: Tensor) -> BoolTensor (B, T)``
            that returns True for trials to **include** in the evaluation.
            When None (default), all non-NaN trials are included.
        n_actions_random_baseline: Number of actions for the random-choice
            baseline in ΔBIC computation. Defaults to ``dataset.n_actions``.
            Set to a smaller value when some actions are excluded via
            ``trial_filter`` (e.g. n_actions-1 when filtering out waiting).

    BIC/AIC/ΔBIC-per-trial are computed per (participant, experiment) group and
    reported as mean ± std across groups -- see ``grouped_information_criteria``
    for why a single dataset-pooled BIC is not a fair comparison across models
    with different parameter-sharing structure.
    """

    unique_pairs, group_index = get_participant_experiment_groups(dataset)
    n_groups = unique_pairs.shape[0]

    # Build valid-trial mask: non-NaN AND passes optional filter
    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])
    if trial_filter is not None:
        valid_mask = valid_mask & trial_filter(dataset.ys)

    # Masked targets: excluded trials set to NaN so nansum skips them
    targets_eval = dataset.ys.clone()
    targets_eval[~valid_mask] = float('nan')

    considered_trials_participant = valid_mask.sum(dim=1).float()
    considered_trials = considered_trials_participant.sum()

    n_actions_baseline = n_actions_random_baseline if n_actions_random_baseline is not None else dataset.n_actions

    # ------------------------------------------------------------
    # Compute choice probs + per-group parameter counts
    # ------------------------------------------------------------
    models = {}

    if benchmark_model is not None:
        print("Computing choice probabilities with benchmark model...")
        benchmark_parameters = benchmark_model.count_parameters() if hasattr(benchmark_model, 'count_parameters') else len([p for p in benchmark_model.parameters()])
        benchmark_predictions, _ = benchmark_model(dataset.xs)
        benchmark_choice_probs = get_choice_probs(benchmark_predictions).detach().cpu()
        models['Benchmark'] = (benchmark_choice_probs, torch.full((n_groups,), float(benchmark_parameters)))

    # setup GRU model
    if gru_model is not None:
        print("Computing choice probabilities with GRU model...")
        gru_model.eval()
        gru_parameters = sum(p.numel() for p in gru_model.parameters())
        gru_predictions, _ = gru_model(dataset.xs)
        gru_choice_probs = get_choice_probs(gru_predictions).detach().cpu()
        models['GRU'] = (gru_choice_probs, torch.full((n_groups,), float(gru_parameters)))

    # setup SPICE model
    if spice_model is not None:
        spice_parameters = spice_model.count_sindy_coefficients()  # (P, X)

        spice_rnn_parameters = 0
        for module in spice_model.get_modules():
            spice_rnn_parameters += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
        spice_rnn_parameters += spice_model.model.embedding_size

        # use spice
        print("Computing choice probabilities with SPICE model...")
        spice_model.eval(use_sindy=True)

        spice_predictions, _ = spice_model(dataset.xs.to(spice_model.device))
        spice_choice_probs = get_choice_probs(spice_predictions.mean(dim=0)).detach().cpu()

        spice_model.use_sindy(False)
        spice_model.model.init_state(batch_size=dataset.xs.shape[0])
        spice_rnn_predictions, _ = spice_model(dataset.xs.to(spice_model.device))
        spice_rnn_choice_probs = get_choice_probs(spice_rnn_predictions.mean(dim=0)).detach().cpu()
        spice_model.use_sindy(True)

        models['SPICE-RNN'] = (spice_rnn_choice_probs, torch.full((n_groups,), float(spice_rnn_parameters)))
        # Per-participant/experiment active coefficient counts -- these are
        # genuinely independent per-group parameters, unlike SPICE-RNN's shared weights.
        models['SPICE-EQ'] = (spice_choice_probs, spice_parameters[unique_pairs[:, 0], unique_pairs[:, 1]].float())

    # ------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------
    rows = {}
    for name, (probs, n_parameters_per_group) in models.items():
        nll = -log_likelihood(data=targets_eval, probs=probs)
        nll_per_session = nll.sum(dim=-1).nansum(dim=1)[..., 0]

        trial_lik = torch.exp(-nll_per_session.sum() / considered_trials).item()
        trial_lik_participant = np.exp(-nll_per_session.numpy() / considered_trials_participant.numpy())
        trial_lik_std = trial_lik_participant.std()

        info = grouped_information_criteria(
            nll_per_session=nll_per_session,
            n_trials_per_session=considered_trials_participant,
            group_index=group_index,
            n_groups=n_groups,
            n_parameters_per_group=n_parameters_per_group,
            n_actions_baseline=n_actions_baseline,
        )

        rows[name] = {
            'Trial Lik.': trial_lik,
            '(std)': trial_lik_std,
            'n_parameters': n_parameters_per_group.mean().item(),
            'n_parameters (std)': n_parameters_per_group.std().item() if n_parameters_per_group.numel() > 1 else 0.0,
            'NLL': info['nll_total'],
            'AIC': info['aic_mean'],
            'AIC (std)': info['aic_std'],
            'BIC': info['bic_mean'],
            'BIC (std)': info['bic_std'],
            'ΔBIC/trial': info['delta_bic_per_trial_mean'],
            'ΔBIC/trial (std)': info['delta_bic_per_trial_std'],
        }

    # ------------------------------------------------------------
    # Printing model performance table
    # ------------------------------------------------------------

    df = pd.DataFrame(rows).T.reindex(['Benchmark', 'GRU', 'SPICE-RNN', 'SPICE-EQ'])

    if verbose:
        print(df)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'model_evaluation.csv'))

    return df


# ============================================================================
# MSE-based model evaluation (for continuous action spaces)
# ============================================================================

def _get_predictions(
    model,
    dataset: SpiceDataset,
) -> torch.Tensor:
    """Run forward pass and return predictions with shape (B, T, W, A).

    Handles SpiceEstimator (with ensemble averaging), BaseModel, and vanilla nn.Module.
    """
    if isinstance(model, SpiceEstimator):
        preds, _ = model(dataset.xs.to(model.device))
        if isinstance(preds, tuple):
            preds = preds[0]
        # SpiceEstimator returns (E, B, T, W, A) — average over ensemble
        if preds.dim() == 5:
            preds = preds.mean(dim=0)
        return preds.detach().cpu()
    else:
        preds, _ = model(dataset.xs)
        # BaseModel forward returns (T, W, E, B, A) via post_forward_pass → (B, T, W, A)
        # GRU / benchmark return (B, T, W, A) directly
        if preds.dim() == 5:
            preds = preds.mean(dim=0)  # ensemble average if present
        return preds.detach().cpu()


def _compute_mse_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_fn: Callable = None,
) -> dict:
    """Compute MSE, RMSE, MAE, R² from masked predictions and targets.

    Args:
        predictions: (B, T, W, A) raw model outputs.
        targets: (B, T, W, F) ground truth. F may exceed A when a custom
            loss_fn requires extra columns (e.g. clamping metadata).
        valid_mask: (B, T) bool mask for non-padded trials.
        loss_fn: Optional loss function ``(prediction, target) → scalar``.
            When provided, MSE is computed via this function and targets
            may have more features than predictions. When ``None``
            (default), plain MSE is used and targets must match
            predictions in the last dimension.

    Returns:
        Dict with aggregate and per-session metrics.
    """
    n_sessions = predictions.shape[0]
    n_actions = predictions.shape[-1]
    n_target_features = targets.shape[-1]

    # Flatten valid predictions
    mask_pred = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
    pred_valid = predictions[mask_pred].reshape(-1, n_actions)

    # Flatten valid targets (may have more features than predictions)
    mask_tgt = valid_mask.unsqueeze(-1).unsqueeze(-1).expand(
        *valid_mask.shape, predictions.shape[2], n_target_features)
    tgt_valid = targets[mask_tgt].reshape(-1, n_target_features)

    if loss_fn is not None:
        mse = loss_fn(pred_valid, tgt_valid).item()
        tgt_actual = tgt_valid[:, :n_actions]
    else:
        tgt_actual = tgt_valid
        mse = ((pred_valid - tgt_actual) ** 2).mean().item()

    # Derived metrics (always from raw prediction errors against actual targets)
    errors = pred_valid - tgt_actual
    rmse = mse ** 0.5
    mae = errors.abs().mean().item()

    ss_res = mse * pred_valid.numel()
    ss_tot = ((tgt_actual - tgt_actual.mean(dim=0)) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Per-session metrics
    mse_per_session = torch.full((n_sessions,), float('nan'))
    for s in range(n_sessions):
        s_mask = valid_mask[s]
        if s_mask.sum() == 0:
            continue
        s_pred = predictions[s, s_mask, 0]  # (n_valid, A)
        s_tgt = targets[s, s_mask, 0]       # (n_valid, F)
        if loss_fn is not None:
            mse_per_session[s] = loss_fn(s_pred, s_tgt)
        else:
            mse_per_session[s] = ((s_pred - s_tgt) ** 2).mean()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse_per_session': mse_per_session,
    }


@torch.no_grad()
def analysis_model_evaluation_mse(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    loss_fn: Callable = None,
    n_actions: int = None,
) -> pd.DataFrame:
    """Evaluate models on continuous prediction tasks using MSE-based metrics.

    Counterpart to ``analysis_model_evaluation`` for continuous action spaces.
    Computes MSE, RMSE, MAE, and R² for each model.

    Args:
        dataset: Test dataset (SpiceDataset with continuous targets).
        spice_model: Fitted SpiceEstimator (optional).
        benchmark_model: Fitted benchmark nn.Module (optional).
        gru_model: Fitted GRU nn.Module (optional).
        output_dir: If provided, save summary CSV and bar plot to this directory.
        verbose: Print results to console.
        loss_fn: Optional loss function ``(prediction, target) → scalar``.
            When provided, MSE is computed via this function (e.g.
            ``clamped_angular_mse``). Defaults to plain MSE.
        n_actions: Number of prediction target columns in ``dataset.ys``.
            Required when ``loss_fn`` is provided and targets contain extra
            metadata columns beyond the actual prediction targets. When
            ``None`` (default), all target columns are used.

    Returns:
        DataFrame with MSE, RMSE, MAE, R², n_parameters per model.
    """
    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])
    targets = dataset.ys.cpu()

    models = {}

    # --- Benchmark model ---
    if benchmark_model is not None:
        print("Evaluating benchmark model...")
        benchmark_model.eval()
        preds = _get_predictions(benchmark_model, dataset)
        if hasattr(benchmark_model, 'count_parameters'):
            n_params = benchmark_model.count_parameters()
        else:
            n_params = sum(p.numel() for p in benchmark_model.parameters())
            if getattr(benchmark_model, 'n_participants', 1) > 1:
                # Fall back assumes per-participant nn.Parameters (shape (n_participants,));
                # report the per-participant count, not the total across all participants.
                n_params /= benchmark_model.n_participants
        models['Benchmark'] = (_compute_mse_metrics(preds, targets, valid_mask, loss_fn=loss_fn), n_params)

    # --- GRU model ---
    if gru_model is not None:
        print("Evaluating GRU model...")
        gru_model.eval()
        preds = _get_predictions(gru_model, dataset)
        n_params = sum(p.numel() for p in gru_model.parameters())
        models['GRU'] = (_compute_mse_metrics(preds, targets, valid_mask, loss_fn=loss_fn), n_params)

    # --- SPICE model (RNN mode + SINDy mode) ---
    if spice_model is not None:
        # SPICE-RNN
        print("Evaluating SPICE-RNN model...")
        spice_model.eval(use_sindy=False)
        spice_model.model.init_state(batch_size=dataset.xs.shape[0])
        preds_rnn, _ = spice_model(dataset.xs.to(spice_model.device))
        if isinstance(preds_rnn, tuple):
            preds_rnn = preds_rnn[0]
        if preds_rnn.dim() == 5:
            preds_rnn = preds_rnn.mean(dim=0)
        preds_rnn = preds_rnn.detach().cpu()

        spice_rnn_params = 0
        for module in spice_model.get_modules():
            spice_rnn_params += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
        spice_rnn_params += spice_model.model.embedding_size
        models['SPICE-RNN'] = (_compute_mse_metrics(preds_rnn, targets, valid_mask, loss_fn=loss_fn), spice_rnn_params)

        # SPICE (SINDy)
        print("Evaluating SPICE model...")
        spice_model.eval(use_sindy=True)
        preds_sindy, _ = spice_model(dataset.xs.to(spice_model.device))
        if isinstance(preds_sindy, tuple):
            preds_sindy = preds_sindy[0]
        if preds_sindy.dim() == 5:
            preds_sindy = preds_sindy.mean(dim=0)
        preds_sindy = preds_sindy.detach().cpu()

        spice_params_tensor = spice_model.count_sindy_coefficients()
        participant_ids = dataset.xs[:, 0, 0, -1].long().cpu()
        experiment_ids = dataset.xs[:, 0, 0, -2].long().cpu()
        unique_pairs = torch.unique(torch.stack([participant_ids, experiment_ids], dim=1), dim=0)
        unique_param_counts = spice_params_tensor[unique_pairs[:, 0], unique_pairs[:, 1]]
        spice_n_params = unique_param_counts.float().mean().item()
        models['SPICE-EQ'] = (_compute_mse_metrics(preds_sindy, targets, valid_mask, loss_fn=loss_fn), spice_n_params)

        spice_model.use_sindy(True)

    # --- Build results table ---
    n_valid = valid_mask.sum().item()

    # Random baseline BIC (predict target mean → MSE = var(targets))
    # When targets contain extra metadata columns, restrict to prediction targets
    n_tgt = n_actions if n_actions is not None else targets.shape[-1]
    tgt_for_bic = targets[..., :n_tgt]
    mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(tgt_for_bic)
    tgt_valid_all = tgt_for_bic[mask_expanded].reshape(-1, n_tgt)
    mse_random = tgt_valid_all.var(dim=0).mean().item()
    bic_random = n_valid * (1 + math.log(2 * math.pi) + math.log(max(mse_random, 1e-30)))

    rows = []
    for name, (metrics, n_params) in models.items():
        mse_std = metrics['mse_per_session'][~metrics['mse_per_session'].isnan()].std().item()
        mse_val = max(metrics['mse'], 1e-30)
        bic = n_valid * (1 + math.log(2 * math.pi) + math.log(mse_val)) + n_params * math.log(n_valid)
        delta_bic_per_trial = (bic_random - bic) / n_valid
        rows.append({
            'Model': name,
            'MSE': metrics['mse'],
            'MSE (std)': mse_std,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2'],
            'n_parameters': n_params,
            'BIC': bic,
            'ΔBIC/trial': delta_bic_per_trial,
        })

    df = pd.DataFrame(rows).set_index('Model')

    if verbose:
        print("\nModel Evaluation (MSE):")
        print(df.to_string(float_format='{:.6f}'.format))

    # --- Plot ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'model_evaluation_mse.csv'))
        _plot_mse_comparison(models, output_dir)

    return df


def _plot_mse_comparison(models: dict, output_dir: str) -> None:
    """Bar chart of MSE per model + per-session violin overlay."""
    model_names = list(models.keys())
    n_models = len(model_names)
    colors = plt.cm.tab10.colors[:n_models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: Aggregate metrics bar chart ---
    ax = axes[0]
    metric_names = ['MSE', 'RMSE', 'MAE']
    x = np.arange(len(metric_names))
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        metrics = models[name][0]
        vals = [metrics['mse'], metrics['rmse'], metrics['mae']]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Error')
    ax.set_title('Aggregate Metrics')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: Per-session MSE violin ---
    ax = axes[1]
    data = []
    for name in model_names:
        mse_ps = models[name][0]['mse_per_session']
        data.append(mse_ps[~mse_ps.isnan()].numpy())

    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    for j, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[j])
        body.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('gray')
    parts['cmedians'].set_linestyle('--')

    ax.set_xticks(range(1, n_models + 1))
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel('MSE per session')
    ax.set_title('Per-Session MSE Distribution')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_evaluation_mse.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, 'model_evaluation_mse.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
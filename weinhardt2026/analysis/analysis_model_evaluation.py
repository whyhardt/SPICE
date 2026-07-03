import os
import math
import sys
from typing import Optional

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


@torch.no_grad()
def analysis_model_evaluation(
    dataset: SpiceDataset,
    spice_model: SpiceEstimator = None,
    benchmark_model: torch.nn.Module = None,
    gru_model: torch.nn.Module = None,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    ):
    
    # ------------------------------------------------------------
    # Compute choice probs
    # ------------------------------------------------------------
    
    if benchmark_model is not None:
        print("Computing choice probabilities with benchmark model...")
        benchmark_parameters = benchmark_model.count_parameters() if hasattr(benchmark_model, 'count_parameters') else len([p for p in benchmark_model.parameters()])
        benchmark_predictions, _ = benchmark_model(dataset.xs)
        benchmark_choice_probs = get_choice_probs(benchmark_predictions).detach().cpu()
    else:
        benchmark_parameters = torch.nan
        
    # setup GRU model
    if gru_model is not None:
        print("Computing choice probabilities with GRU model...")
        gru_model.eval()
        gru_parameters = sum(p.numel() for p in gru_model.parameters())
        gru_predictions, _ = gru_model(dataset.xs)
        gru_choice_probs = get_choice_probs(gru_predictions).detach().cpu()
    else:
        gru_parameters = torch.nan
        
    # setup SPICE model
    if spice_model is not None:
        spice_parameters = spice_model.count_sindy_coefficients()
        
        spice_rnn_parameters = 0
        for module in spice_model.get_modules():
            spice_rnn_parameters += sum(p.numel() for p in spice_model.model.submodules_rnn[module].parameters())
        spice_rnn_parameters += spice_model.model.embedding_size
        
        # use spice
        print("Computing choice probabilities with SPICE model...")
        spice_model.eval(use_sindy=True)
        
        spice_predictions, _ = spice_model(dataset.xs.to(spice_model.device))           
        spice_choice_probs = get_choice_probs(spice_predictions[0]).detach().cpu()
        
        spice_model.use_sindy(False)
        spice_model.model.init_state(batch_size=dataset.xs.shape[0])
        spice_rnn_predictions, _ = spice_model(dataset.xs.to(spice_model.device))           
        spice_rnn_choice_probs = get_choice_probs(spice_rnn_predictions.mean(dim=0)).detach().cpu()
        spice_model.use_sindy(True)
    else:
        spice_parameters = torch.nan
        spice_rnn_parameters = torch.nan
        
    # ------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------
    
    scores = torch.zeros((4, 3))
    metric_participant = torch.zeros((len(scores), len(dataset)))
    
    considered_trials_participant = (~torch.isnan(dataset.xs[:, :, 0, 0])).sum(dim=1)
    considered_trials = considered_trials_participant.sum()
    
    # SPICE model
    if spice_model is not None:
        participant_ids = dataset.xs[:, 0, 0, -1].long().cpu()
        experiment_ids = dataset.xs[:, 0, 0, -2].long().cpu()

        # Compute parameter stats over unique (participant, experiment) pairs, not sessions
        unique_pairs = torch.unique(torch.stack([participant_ids, experiment_ids], dim=1), dim=0)
        unique_param_counts = spice_parameters[unique_pairs[:, 0], unique_pairs[:, 1]]
        spice_n_params_mean = unique_param_counts.mean().item()
        spice_n_params_std = unique_param_counts.std().item() if len(unique_param_counts) > 1 else 0.0

        scores_spice, nll_per_sample = get_scores(targets=dataset.ys, probs=spice_choice_probs, n_parameters=spice_n_params_mean)
        scores[3] += torch.tensor(scores_spice)
        metric_participant[3] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]

        scores_spice_rnn, nll_per_sample = get_scores(targets=dataset.ys, probs=spice_rnn_choice_probs, n_parameters=spice_rnn_parameters)
        scores[2] += torch.tensor(scores_spice_rnn)
        metric_participant[2] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]
    else:
        spice_n_params_mean = torch.nan
        spice_n_params_std = 0.0
        
    # Benchmark model
    if benchmark_model is not None:
        scores_benchmark, nll_per_sample = get_scores(targets=dataset.ys, probs=benchmark_choice_probs, n_parameters=benchmark_parameters)
        scores[0] += torch.tensor(scores_benchmark)
        metric_participant[0] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]
        
    # GRU model
    if gru_model is not None:
        scores_gru, nll_per_sample = get_scores(targets=dataset.ys, probs=gru_choice_probs, n_parameters=gru_parameters)
        scores[1] += torch.tensor(scores_gru)
        metric_participant[1] = nll_per_sample.sum(dim=-1).nansum(dim=1)[..., 0]
        
    # ------------------------------------------------------------
    # Post processing
    # ------------------------------------------------------------

    # compute trial-level metrics (and NLL -> Likelihood)
    # scores = scores / considered_trials
    avg_trial_likelihood = torch.exp(-scores[:, 0] / considered_trials)

    avg_trial_likelihood_participant = np.exp(- metric_participant / considered_trials_participant)
    avg_trial_likelihood_participant_std = avg_trial_likelihood_participant.std(dim=1)

    # compute average number of parameters
    n_parameters = torch.tensor([
        benchmark_parameters,
        gru_parameters,
        spice_rnn_parameters,
        spice_n_params_mean,
        ])
    n_parameters_std = torch.tensor([
        0,
        0,
        0,
        spice_n_params_std,
    ])

    scores = torch.concatenate((
        avg_trial_likelihood.reshape(-1, 1), 
        avg_trial_likelihood_participant_std.reshape(-1, 1), 
        n_parameters.reshape(-1, 1), 
        n_parameters_std.reshape(-1, 1),
        scores[:, :1], 
        scores[:, 1:],
        ), dim=1)
    
    # ------------------------------------------------------------
    # Printing model performance table
    # ------------------------------------------------------------

    df = pd.DataFrame(
        data=scores,
        index=['Benchmark', 'GRU', 'SPICE-RNN', 'SPICE-EQ'],
        columns = ('Trial Lik.', '(std)', 'n_parameters', '(std)', 'NLL', 'AIC', 'BIC'),
        )

    # ΔBIC/trial: anchored to random-choice baseline (0 parameters)
    n_actions = dataset.n_actions
    n_trials = considered_trials.item()
    nll_random = n_trials * math.log(n_actions)
    bic_random = 2 * nll_random  # 0 parameters → penalty = 0
    df['ΔBIC/trial'] = (bic_random - df['BIC'].astype(float)) / n_trials

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
) -> dict:
    """Compute MSE, RMSE, MAE, R² from masked predictions and targets.

    Args:
        predictions: (B, T, W, A) raw model outputs.
        targets: (B, T, W, A) ground truth.
        valid_mask: (B, T) bool mask for non-padded trials.

    Returns:
        Dict with aggregate and per-session metrics.
    """
    n_sessions = predictions.shape[0]
    n_actions = predictions.shape[-1]

    # Flatten valid predictions/targets
    mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(predictions)
    pred_valid = predictions[mask_expanded].reshape(-1, n_actions)
    tgt_valid = targets[mask_expanded].reshape(-1, n_actions)

    # Aggregate metrics
    errors = pred_valid - tgt_valid
    mse = (errors ** 2).mean().item()
    rmse = mse ** 0.5
    mae = errors.abs().mean().item()

    ss_res = (errors ** 2).sum().item()
    ss_tot = ((tgt_valid - tgt_valid.mean(dim=0)) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Per-session metrics
    mse_per_session = torch.full((n_sessions,), float('nan'))
    for s in range(n_sessions):
        s_mask = valid_mask[s]
        if s_mask.sum() == 0:
            continue
        s_pred = predictions[s, s_mask, 0]  # (n_valid, A)
        s_tgt = targets[s, s_mask, 0]
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
        n_params = (
            benchmark_model.count_parameters()
            if hasattr(benchmark_model, 'count_parameters')
            else sum(p.numel() for p in benchmark_model.parameters())
        )
        models['Benchmark'] = (_compute_mse_metrics(preds, targets, valid_mask), n_params)

    # --- GRU model ---
    if gru_model is not None:
        print("Evaluating GRU model...")
        gru_model.eval()
        preds = _get_predictions(gru_model, dataset)
        n_params = sum(p.numel() for p in gru_model.parameters())
        models['GRU'] = (_compute_mse_metrics(preds, targets, valid_mask), n_params)

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
        models['SPICE-RNN'] = (_compute_mse_metrics(preds_rnn, targets, valid_mask), spice_rnn_params)

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
        models['SPICE-EQ'] = (_compute_mse_metrics(preds_sindy, targets, valid_mask), spice_n_params)

        spice_model.use_sindy(True)

    # --- Build results table ---
    n_valid = valid_mask.sum().item()

    # Random baseline BIC (predict target mean → MSE = var(targets))
    mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(targets)
    tgt_valid_all = targets[mask_expanded].reshape(-1, targets.shape[-1])
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
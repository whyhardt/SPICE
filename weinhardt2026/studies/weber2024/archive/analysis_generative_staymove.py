import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from spice import SpiceDataset
from weinhardt2026.studies.weber2024.archive.benchmarking_weber2024 import get_dataset, angular_distance, MISS_PENALTY


METRIC_LABELS = {
    'p_move': 'P(Move)',
    'p_catch': 'P(Catch)',
    'mean_prediction_error': 'Mean Prediction Error',
    'avg_learning_rate': 'Avg Learning Rate',
    'integration_kernel_steepness': 'Integration Kernel Steepness',
    'total_reward': 'Total Reward',
}

# Additional input column offsets (relative to start of additional inputs block)
_AI_SHIELD_ROT = 1      # shieldRotation
_AI_LASER_ROT = 2       # laserRotation
_AI_CAUGHT = 8          # laser_caught


def _extract_data(dataset: SpiceDataset):
    """Extract behavioural signals from a weber2024 SpiceDataset.

    Returns:
        actions: (n_sessions, n_trials) float — 0=stay, 1=move; NaN for padded.
        prediction_errors: (n_sessions, n_trials) float — angular distance
            shield→laser; NaN for padded.
        learning_rates: (n_sessions, n_trials) float — fractional movement
            toward laser; NaN where undefined.
        signed_pes: (n_sessions, n_trials) float — signed prediction error
            (positive = laser clockwise of shield); NaN for padded.
        rewards: (n_sessions, n_trials) float — per-trial reward; NaN for padded.
        valid: (n_sessions, n_trials) bool — True for non-padded trials.
    """
    n_actions = dataset.n_actions
    ai_start = n_actions
    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)

    # Valid-trial mask
    valid = ~torch.isnan(xs[:, :, 0])

    # Actions
    actions_oh = xs[:, :, :n_actions]
    actions = actions_oh.argmax(dim=-1).float()
    actions[~valid] = float('nan')

    # Positions
    laser = xs[:, :, ai_start + _AI_LASER_ROT].clone()
    shield = xs[:, :, ai_start + _AI_SHIELD_ROT].clone()
    laser[~valid] = float('nan')
    shield[~valid] = float('nan')

    # Prediction error = angular distance shield→laser
    prediction_errors = torch.full_like(laser, float('nan'))
    prediction_errors[valid] = angular_distance(shield[valid], laser[valid])

    # Signed PE: positive = laser clockwise of shield (shortest arc direction)
    signed_pes = torch.full_like(laser, float('nan'))
    diff = (laser - shield % 360) % 360
    signed = torch.where(diff <= 180, diff, diff - 360)
    signed_pes[valid] = signed[valid]

    # Learning rate = |shield_movement| / prediction_error
    # shield_movement[t] = angular_distance(shield[t], shield[t+1])
    shield_movement = torch.full_like(shield, float('nan'))
    both_valid = valid[:, :-1] & valid[:, 1:]
    shield_movement[:, :-1][both_valid] = angular_distance(
        shield[:, :-1][both_valid], shield[:, 1:][both_valid]
    )

    learning_rates = torch.full_like(shield, float('nan'))
    computable = both_valid & (prediction_errors[:, :-1] > 1.0)  # avoid div-by-zero for PE ≈ 0
    lr_vals = shield_movement[:, :-1][computable] / prediction_errors[:, :-1][computable]
    learning_rates[:, :-1][computable] = lr_vals.clamp(0, 1)

    # Rewards (computed from laser_caught — no reward columns in dataset)
    laser_caught = xs[:, :, ai_start + _AI_CAUGHT]
    rewards = torch.where(
        laser_caught == 1,
        torch.zeros_like(laser_caught),
        torch.full_like(laser_caught, MISS_PENALTY),
    )
    rewards[~valid] = float('nan')

    return (
        actions.numpy(),
        prediction_errors.numpy(),
        learning_rates.numpy(),
        signed_pes.numpy(),
        rewards.numpy(),
        valid.numpy(),
    )


def _compute_metrics(actions, prediction_errors, learning_rates, signed_pes, rewards, valid):
    """Compute per-session behavioural metrics.

    All inputs are (n_sessions, n_trials) numpy arrays (NaN for invalid).

    Returns dict mapping metric name → (n_sessions,) numpy array.
    """
    n_sessions = actions.shape[0]

    p_move = np.nanmean(actions, axis=1)
    p_catch = np.nanmean(prediction_errors <= CATCH_THRESHOLD_NP, axis=1)
    mean_pe = np.nanmean(prediction_errors, axis=1)
    avg_lr = np.nanmean(learning_rates, axis=1)
    total_reward = np.nansum(rewards, axis=1)

    # Integration kernel steepness: regression of last 5 signed PEs on movement
    kernel_steepness = np.full(n_sessions, np.nan)
    for s in range(n_sessions):
        v = valid[s]
        spe = signed_pes[s]
        act = actions[s]
        steepness = _compute_integration_kernel_steepness(spe, act, v)
        kernel_steepness[s] = steepness

    return {
        'p_move': p_move,
        'p_catch': p_catch,
        'mean_prediction_error': mean_pe,
        'avg_learning_rate': avg_lr,
        'integration_kernel_steepness': kernel_steepness,
        'total_reward': total_reward,
    }


CATCH_THRESHOLD_NP = 10.0


def _compute_integration_kernel_steepness(
    signed_pes: np.ndarray,
    actions: np.ndarray,
    valid: np.ndarray,
    n_lags: int = 5,
) -> float:
    """Compute integration kernel steepness for a single session.

    Fits a regression of lagged signed prediction errors (last n_lags events)
    onto the binary movement decision, then returns weight[lag=1] - weight[lag=2]
    as the steepness measure.

    Args:
        signed_pes: (n_trials,) — signed PE per trial (NaN for invalid).
        actions: (n_trials,) — 0=stay, 1=move (NaN for invalid).
        valid: (n_trials,) bool.
        n_lags: number of lags to include.

    Returns:
        Steepness (float) or NaN if insufficient data.
    """
    n = len(signed_pes)
    if n < n_lags + 1:
        return np.nan

    # Build lagged PE matrix
    X_rows = []
    y_rows = []
    for t in range(n_lags, n):
        if not valid[t] or np.isnan(actions[t]):
            continue
        lags = signed_pes[t - n_lags:t]
        if np.any(np.isnan(lags)):
            continue
        X_rows.append(lags)
        y_rows.append(actions[t])

    if len(X_rows) < n_lags + 1:
        return np.nan

    X = np.array(X_rows)  # (n_valid, n_lags)
    y = np.array(y_rows)  # (n_valid,)

    reg = LinearRegression().fit(X, y)
    weights = reg.coef_  # (n_lags,) — weights[0] = lag n_lags, weights[-1] = lag 1

    # weights[-1] is lag 1 (most recent), weights[-2] is lag 2
    if len(weights) >= 2:
        return float(weights[-1] - weights[-2])
    return np.nan


def _plot_violins(all_metrics, output_dir):
    """Create violin plots comparing behavioural metrics across datasets."""
    model_names = list(all_metrics.keys())
    metric_names = list(METRIC_LABELS.keys())
    colors = plt.cm.tab10.colors[:len(model_names)]

    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = []
        for model_name in model_names:
            values = all_metrics[model_name][metric]
            data.append(values[~np.isnan(values)])

        parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)

        for j, body in enumerate(parts['bodies']):
            body.set_facecolor(colors[j])
            body.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('gray')
        parts['cmedians'].set_linestyle('--')

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'generative_behavior.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, 'generative_behavior.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def analysis_generative_behavior(
    path_data_real: str = None,
    path_data_benchmark: str = None,
    path_data_gru: str = None,
    path_data_spice_rnn: str = None,
    path_data_spice: str = None,
    output_dir: str = 'results',
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Run generative behaviour analysis for weber2024.

    Loads each dataset via get_dataset(), extracts task-specific behavioural
    metrics, and produces violin plots comparing real and model-generated data.

    Returns:
        df_summary: DataFrame with mean +/- std per metric per model.
        df_comparison: DataFrame with NMAE per metric (if real data provided).
    """
    os.makedirs(output_dir, exist_ok=True)

    datasets = {}
    if path_data_real is not None:
        datasets['real'] = path_data_real
    if path_data_benchmark is not None:
        datasets['benchmark'] = path_data_benchmark
    if path_data_gru is not None:
        datasets['gru'] = path_data_gru
    if path_data_spice_rnn is not None:
        datasets['spice_rnn'] = path_data_spice_rnn
    if path_data_spice is not None:
        datasets['spice'] = path_data_spice

    all_metrics = {}
    for name, path in datasets.items():
        print(f"Loading {name} from {path}...")
        dataset, _ = get_dataset(path_data=path)
        data = _extract_data(dataset)
        all_metrics[name] = _compute_metrics(*data)

    # Summary table
    rows = []
    for name in all_metrics:
        row = {'Model': name}
        for metric in METRIC_LABELS:
            values = all_metrics[name][metric]
            values_clean = values[~np.isnan(values)]
            row[METRIC_LABELS[metric]] = f"{values_clean.mean():.3f} +/- {values_clean.std():.3f}"
        rows.append(row)

    df_summary = pd.DataFrame(rows).set_index('Model')
    print(df_summary)

    _plot_violins(all_metrics, output_dir)

    # Quantitative comparison: normalised MAE vs real data
    df_comparison = None
    if 'real' in all_metrics:
        real_means = {}
        real_stds = {}
        for metric in METRIC_LABELS:
            vals = all_metrics['real'][metric]
            vals = vals[~np.isnan(vals)]
            real_means[metric] = vals.mean()
            real_stds[metric] = vals.std()

        comp_rows = []
        for name in all_metrics:
            if name == 'real':
                continue
            row = {'Model': name}
            norm_errors = []
            for metric in METRIC_LABELS:
                vals = all_metrics[name][metric]
                vals = vals[~np.isnan(vals)]
                mae = abs(vals.mean() - real_means[metric])
                nmae = mae / real_stds[metric] if real_stds[metric] > 0 else 0.0
                row[METRIC_LABELS[metric]] = f"{nmae:.4f}"
                norm_errors.append(nmae)
            row['Aggregate NMAE'] = f"{np.mean(norm_errors):.4f} +/- {np.std(norm_errors):.4f}"
            comp_rows.append(row)

        df_comparison = pd.DataFrame(comp_rows).set_index('Model')
        print("\nNormalized MAE (|model_mean - real_mean| / real_std):")
        print(df_comparison)

    return df_summary, df_comparison

import os
import math
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.weber2024.benchmarking_weber2024 import angular_distance
from weinhardt2026.studies.weber2024.benchmarking_weber2024_continuous import (
    get_dataset, CATCH_THRESHOLD, _N_ACTIONS, _N_REWARDS, _AI_START, _AI_CAUGHT, _AI_DT,
)


METRIC_LABELS = {
    'mean_prediction_error': 'Mean Prediction Error (°)',
    'p_catch': 'P(Catch)',
    'mean_shield_movement': 'Mean Shield Movement (°)',
    'avg_learning_rate': 'Avg Learning Rate',
    'p_perseveration': 'P(Perseveration)',
    'total_reward': 'Total Reward',
}


def _sincos_to_degrees(sin_vals: torch.Tensor, cos_vals: torch.Tensor) -> torch.Tensor:
    """Convert sin/cos to degrees in [0, 360)."""
    return torch.atan2(sin_vals, cos_vals) * (180.0 / math.pi) % 360


def _extract_data(dataset: SpiceDataset):
    """Extract behavioural signals from a continuous weber2024 SpiceDataset.

    Returns:
        shield_deg: (n_sessions, n_trials) shield position in degrees.
        laser_deg: (n_sessions, n_trials) laser position in degrees.
        caught: (n_sessions, n_trials) binary catch indicator.
        valid: (n_sessions, n_trials) bool mask for non-padded trials.
    """
    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)
    valid = ~torch.isnan(xs[:, :, 0])

    shield_deg = _sincos_to_degrees(xs[:, :, 0], xs[:, :, 1])
    laser_deg = _sincos_to_degrees(xs[:, :, _N_ACTIONS], xs[:, :, _N_ACTIONS + 1])
    caught = xs[:, :, _AI_START + _AI_CAUGHT].clone()

    shield_deg[~valid] = float('nan')
    laser_deg[~valid] = float('nan')
    caught[~valid] = float('nan')

    return shield_deg.numpy(), laser_deg.numpy(), caught.numpy(), valid.numpy()


def _compute_metrics(shield_deg, laser_deg, caught, valid):
    """Compute per-session behavioural metrics.

    All inputs are (n_sessions, n_trials) numpy arrays (NaN for invalid).

    Returns dict mapping metric name -> (n_sessions,) numpy array.
    """
    n_sessions, n_trials = shield_deg.shape

    # Prediction error: angular distance shield→laser
    pe = np.full_like(shield_deg, np.nan)
    for s in range(n_sessions):
        for t in range(n_trials):
            if valid[s, t]:
                diff = (shield_deg[s, t] - laser_deg[s, t]) % 360
                pe[s, t] = min(diff, 360 - diff)

    mean_pe = np.nanmean(pe, axis=1)

    # P(Catch)
    p_catch = np.nanmean(pe <= CATCH_THRESHOLD, axis=1)

    # Shield movement: angular distance between consecutive shield positions
    movement = np.full_like(shield_deg, np.nan)
    both_valid = valid[:, :-1] & valid[:, 1:]
    for s in range(n_sessions):
        for t in range(n_trials - 1):
            if both_valid[s, t]:
                diff = (shield_deg[s, t] - shield_deg[s, t + 1]) % 360
                movement[s, t] = min(diff, 360 - diff)

    mean_movement = np.nanmean(movement, axis=1)

    # Learning rate: |movement| / PE, where PE > threshold
    pe_threshold = 1.0  # degrees
    lr = np.full_like(movement, np.nan)
    computable = both_valid & (pe[:, :-1] > pe_threshold)
    lr[:, :-1][computable] = movement[:, :-1][computable] / pe[:, :-1][computable]
    lr = np.clip(lr, 0, 2)
    avg_lr = np.nanmean(lr, axis=1)

    # Perseveration: P(|movement| < threshold)
    perseveration_threshold = 5.0  # degrees
    p_perseveration = np.nanmean(movement < perseveration_threshold, axis=1)

    # Total reward (based on catches)
    total_reward = np.nansum(caught, axis=1)

    return {
        'mean_prediction_error': mean_pe,
        'p_catch': p_catch,
        'mean_shield_movement': mean_movement,
        'avg_learning_rate': avg_lr,
        'p_perseveration': p_perseveration,
        'total_reward': total_reward,
    }


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

        if any(len(d) == 0 for d in data):
            ax.set_xticks(range(1, len(model_names) + 1))
            ax.set_xticklabels(model_names, fontsize=9)
            ax.set_title(f"{METRIC_LABELS[metric]} (no data)", fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            continue

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
    fig.savefig(os.path.join(output_dir, 'generative_behavior_continuous.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, 'generative_behavior_continuous.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def analysis_generative_behavior(
    dataset_real: SpiceDataset = None,
    dataset_gru: SpiceDataset = None,
    dataset_spice_rnn: SpiceDataset = None,
    dataset_spice: SpiceDataset = None,
    output_dir: str = 'results',
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Run generative behaviour analysis for continuous weber2024.

    Accepts SpiceDataset objects directly. Extracts angular position metrics
    and produces violin plots comparing real and model-generated data.

    Returns:
        df_summary: DataFrame with mean +/- std per metric per model.
        df_comparison: DataFrame with NMAE per metric (if real data provided).
    """
    os.makedirs(output_dir, exist_ok=True)

    datasets = {}
    entries = [
        ('real', dataset_real),
        ('gru', dataset_gru),
        ('spice_rnn', dataset_spice_rnn),
        ('spice', dataset_spice),
    ]
    for name, ds in entries:
        if ds is not None:
            datasets[name] = ds

    all_metrics = {}
    for name, ds in datasets.items():
        print(f"Extracting metrics for {name}...")
        data = _extract_data(ds)
        all_metrics[name] = _compute_metrics(*data)

    # Summary table
    rows = []
    for name in all_metrics:
        row = {'Model': name}
        for metric in METRIC_LABELS:
            values = all_metrics[name][metric]
            values_clean = values[~np.isnan(values)]
            if len(values_clean) > 0:
                row[METRIC_LABELS[metric]] = f"{values_clean.mean():.3f} +/- {values_clean.std():.3f}"
            else:
                row[METRIC_LABELS[metric]] = "N/A"
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
                if len(vals) == 0:
                    row[METRIC_LABELS[metric]] = "N/A"
                    continue
                mae = abs(vals.mean() - real_means[metric])
                nmae = mae / real_stds[metric] if real_stds[metric] > 0 else 0.0
                row[METRIC_LABELS[metric]] = f"{nmae:.4f}"
                norm_errors.append(nmae)
            if norm_errors:
                row['Aggregate NMAE'] = f"{np.mean(norm_errors):.4f} +/- {np.std(norm_errors):.4f}"
            comp_rows.append(row)

        df_comparison = pd.DataFrame(comp_rows).set_index('Model')
        print("\nNormalized MAE (|model_mean - real_mean| / real_std):")
        print(df_comparison)

    return df_summary, df_comparison

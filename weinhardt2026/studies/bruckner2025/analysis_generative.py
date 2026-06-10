import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import get_dataset, POSITION_SCALE, _AI_MU_T, _AI_C_T


METRIC_LABELS = {
    'mean_prediction_error': 'Mean Prediction Error',
    'mean_estimation_error': 'Mean Estimation Error',
    'avg_learning_rate': 'Avg Learning Rate',
    'p_perseveration': 'P(Perseveration)',
    'mean_update_magnitude': 'Mean |Update|',
    'post_cp_learning_rate': 'Post-CP Learning Rate',
}


def _extract_data(dataset: SpiceDataset):
    """Extract behavioural signals from a bruckner2025 SpiceDataset.

    Returns:
        bucket: (n_sessions, n_trials) bucket positions (normalized).
        outcome: (n_sessions, n_trials) outcome positions (normalized).
        mu_t: (n_sessions, n_trials) true helicopter positions (normalized).
        c_t: (n_sessions, n_trials) change point indicators.
        valid: (n_sessions, n_trials) bool mask for non-padded trials.
    """
    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)

    valid = ~torch.isnan(xs[:, :, 0])

    n_actions = 1
    n_rewards = dataset.n_reward_features
    ai_start = n_actions + n_rewards

    bucket = xs[:, :, 0].clone()                          # b_t / 300
    outcome = xs[:, :, 1].clone()                         # x_t / 300
    mu_t = xs[:, :, ai_start + _AI_MU_T].clone()         # mu_t / 300
    c_t = xs[:, :, ai_start + _AI_C_T].clone()           # change point

    bucket[~valid] = float('nan')
    outcome[~valid] = float('nan')
    mu_t[~valid] = float('nan')
    c_t[~valid] = float('nan')

    return bucket.numpy(), outcome.numpy(), mu_t.numpy(), c_t.numpy(), valid.numpy()


def _compute_metrics(bucket, outcome, mu_t, c_t, valid):
    """Compute per-session behavioural metrics.

    All inputs are (n_sessions, n_trials) numpy arrays (NaN for invalid).

    Returns dict mapping metric name → (n_sessions,) numpy array.
    """
    n_sessions, n_trials = bucket.shape

    # Prediction error: |x_t - b_t|
    prediction_error = np.abs(outcome - bucket)
    mean_pe = np.nanmean(prediction_error, axis=1)

    # Estimation error: |mu_t - b_t|
    estimation_error = np.abs(mu_t - bucket)
    mean_ee = np.nanmean(estimation_error, axis=1)

    # Update: b_{t+1} - b_t
    update = np.full_like(bucket, np.nan)
    update[:, :-1] = bucket[:, 1:] - bucket[:, :-1]
    both_valid = valid[:, :-1] & valid[:, 1:]
    update[:, :-1][~both_valid] = np.nan

    mean_update_mag = np.nanmean(np.abs(update), axis=1)

    # Learning rate: |b_{t+1} - b_t| / |x_t - b_t|, where PE > threshold
    pe_threshold = 0.01  # avoid division by tiny values (normalized scale)
    lr = np.full_like(update, np.nan)
    computable = both_valid & (prediction_error[:, :-1] > pe_threshold)
    lr[:, :-1][computable] = (
        np.abs(update[:, :-1][computable]) / prediction_error[:, :-1][computable]
    )
    lr = np.clip(lr, 0, 2)
    avg_lr = np.nanmean(lr, axis=1)

    # Perseveration: P(update == 0)
    p_perseveration = np.nanmean(np.abs(update) < 1e-6, axis=1)

    # Post change-point learning rate (trials 1-3 after a CP)
    post_cp_lr = np.full(n_sessions, np.nan)
    for s in range(n_sessions):
        cp_lrs = []
        for t in range(n_trials - 1):
            if not valid[s, t] or np.isnan(c_t[s, t]):
                continue
            if c_t[s, t] == 1.0:
                # Collect learning rates for the next 3 trials after CP
                for dt in range(1, 4):
                    idx = t + dt
                    if idx < n_trials - 1 and not np.isnan(lr[s, idx]):
                        cp_lrs.append(lr[s, idx])
        if cp_lrs:
            post_cp_lr[s] = np.mean(cp_lrs)

    return {
        'mean_prediction_error': mean_pe,
        'mean_estimation_error': mean_ee,
        'avg_learning_rate': avg_lr,
        'p_perseveration': p_perseveration,
        'mean_update_magnitude': mean_update_mag,
        'post_cp_learning_rate': post_cp_lr,
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
    dataset_real: SpiceDataset = None,
    dataset_benchmark: SpiceDataset = None,
    dataset_gru: SpiceDataset = None,
    dataset_spice_rnn: SpiceDataset = None,
    dataset_spice: SpiceDataset = None,
    output_dir: str = 'results',
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Run generative behaviour analysis for bruckner2025.

    Accepts either CSV paths or SpiceDataset objects directly.
    Extracts task-specific behavioural metrics and produces violin plots.

    Returns:
        df_summary: DataFrame with mean +/- std per metric per model.
        df_comparison: DataFrame with NMAE per metric (if real data provided).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect datasets (prefer direct SpiceDataset over CSV paths)
    datasets = {}
    entries = [
        ('real', path_data_real, dataset_real),
        ('benchmark', path_data_benchmark, dataset_benchmark),
        ('gru', path_data_gru, dataset_gru),
        ('spice_rnn', path_data_spice_rnn, dataset_spice_rnn),
        ('spice', path_data_spice, dataset_spice),
    ]
    for name, path, ds in entries:
        if ds is not None:
            datasets[name] = ds
        elif path is not None:
            datasets[name] = get_dataset(path_data=path)[0]

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
                row[METRIC_LABELS[metric]] = f"{values_clean.mean():.4f} +/- {values_clean.std():.4f}"
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

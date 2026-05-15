import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a import get_dataset


METRIC_LABELS = {
    'accuracy': 'Accuracy',
    'accuracy_low_contrast': 'Accuracy (Low |Contrast|)',
    'accuracy_mid_contrast': 'Accuracy (Mid |Contrast|)',
    'accuracy_high_contrast': 'Accuracy (High |Contrast|)',
    'avg_reward': 'Average Reward',
    'win_stay': 'P(Stay | Win)',
    'lose_shift': 'P(Shift | Loss)',
    'p_stay': 'P(Stay)',
}


def _extract_data(dataset: SpiceDataset):
    """Extract choices, rewards, and contrast from a Ganesh 2024a dataset.

    Feature layout (n_actions=2, n_reward_features=2):
        [action_0, action_1, reward_0, reward_1, contrast_current,
         contrast_next, time_trial, trials, block, experiment_id, participant_id]

    Returns:
        choices:  (n_sessions, n_trials) float — chosen action index, NaN = padded
        rewards:  (n_sessions, n_trials) float — scalar reward, NaN = padded
        contrast: (n_sessions, n_trials) float — current-trial contrast difference
    """
    n_actions = dataset.n_actions
    xs = dataset.xs[:, :, 0, :]

    valid = ~torch.isnan(xs[:, :, 0])

    choices = xs[:, :, :n_actions].argmax(dim=-1).float().numpy()
    rewards = xs[:, :, n_actions:2 * n_actions].nan_to_num(0.0).sum(dim=-1).numpy()
    contrast = xs[:, :, 2 * n_actions].numpy()

    choices[~valid.numpy()] = np.nan
    rewards[~valid.numpy()] = np.nan
    contrast[~valid.numpy()] = np.nan

    return choices, rewards, contrast


def _compute_metrics(choices, rewards, contrast):
    """Compute perceptual-bandit metrics per session."""
    n_sessions = choices.shape[0]

    # True state: side with higher contrast (1 if contrast > 0, else 0)
    true_state = (contrast > 0).astype(float)
    true_state[np.isnan(contrast)] = np.nan
    correct = (choices == true_state).astype(float)
    correct[np.isnan(choices)] = np.nan

    # Contrast magnitude tercile thresholds (computed across all valid values)
    abs_contrast = np.abs(contrast)
    all_valid = abs_contrast[~np.isnan(abs_contrast)]
    if len(all_valid) > 0:
        t_low, t_high = np.percentile(all_valid, [33.3, 66.7])
    else:
        t_low, t_high = 0.0, 0.0

    # Stay/shift: compare consecutive choices
    stays = (choices[:, 1:] == choices[:, :-1]).astype(float)
    prev_rewards = rewards[:, :-1]
    invalid = np.isnan(choices[:, 1:]) | np.isnan(choices[:, :-1])
    stays[invalid] = np.nan

    metrics = {k: np.full(n_sessions, np.nan) for k in METRIC_LABELS}
    metrics['accuracy'] = np.nanmean(correct, axis=1)
    metrics['avg_reward'] = np.nanmean(rewards, axis=1)
    metrics['p_stay'] = np.nanmean(stays, axis=1)

    for s in range(n_sessions):
        # Accuracy by contrast tercile
        valid_c = ~np.isnan(correct[s]) & ~np.isnan(abs_contrast[s])
        low = valid_c & (abs_contrast[s] <= t_low)
        mid = valid_c & (abs_contrast[s] > t_low) & (abs_contrast[s] <= t_high)
        high = valid_c & (abs_contrast[s] > t_high)
        if low.sum() > 0:
            metrics['accuracy_low_contrast'][s] = correct[s][low].mean()
        if mid.sum() > 0:
            metrics['accuracy_mid_contrast'][s] = correct[s][mid].mean()
        if high.sum() > 0:
            metrics['accuracy_high_contrast'][s] = correct[s][high].mean()

        # Win-stay / Lose-shift
        valid_t = ~np.isnan(prev_rewards[s]) & ~np.isnan(stays[s])
        wins = valid_t & (prev_rewards[s] > 0.5)
        losses = valid_t & (prev_rewards[s] < 0.5)
        if wins.sum() > 0:
            metrics['win_stay'][s] = stays[s][wins].mean()
        if losses.sum() > 0:
            metrics['lose_shift'][s] = 1.0 - stays[s][losses].mean()

    return metrics


def _plot_violins(all_metrics, output_dir):
    """Create violin plots comparing behavioral metrics across datasets."""
    model_names = list(all_metrics.keys())
    metric_names = list(METRIC_LABELS.keys())
    colors = plt.cm.tab10.colors[: len(model_names)]

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
    fig.savefig(
        os.path.join(output_dir, 'generative_behavior.pdf'),
        bbox_inches='tight', dpi=150,
    )
    fig.savefig(
        os.path.join(output_dir, 'generative_behavior.png'),
        bbox_inches='tight', dpi=150,
    )
    plt.show()
    plt.close(fig)


def analysis_generative_behavior(
    path_data_real: str = None,
    path_data_benchmark: str = None,
    path_data_gru: str = None,
    path_data_spice_rnn: str = None,
    path_data_spice: str = None,
    output_dir: str = 'results',
) -> pd.DataFrame:
    """Run generative behavior analysis for the Ganesh 2024a contrast bandit.

    Computes accuracy (overall and by contrast tercile), reward, win-stay,
    lose-shift, and stay metrics, then produces violin plots.

    Args:
        path_data_real: CSV path for real data.
        path_data_benchmark: CSV path for benchmark model data.
        path_data_gru: CSV path for GRU model data.
        path_data_spice_rnn: CSV path for SPICE-RNN data.
        path_data_spice: CSV path for SPICE equation data.
        output_dir: Directory for output plots.

    Returns:
        DataFrame with summary statistics per dataset and metric.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = {}

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

    for name, path in datasets.items():
        print(f"Loading {name} from {path}...")
        dataset, _, _ = get_dataset(path_data=path, test_sessions=())
        choices, rewards, contrast = _extract_data(dataset)
        all_metrics[name] = _compute_metrics(choices, rewards, contrast)

    # Summary table
    rows = []
    for name in all_metrics:
        row = {'Model': name}
        for metric in METRIC_LABELS:
            values = all_metrics[name][metric]
            values_clean = values[~np.isnan(values)]
            row[METRIC_LABELS[metric]] = (
                f"{values_clean.mean():.3f} +/- {values_clean.std():.3f}"
            )
        rows.append(row)

    df_summary = pd.DataFrame(rows).set_index('Model')
    print(df_summary)

    _plot_violins(all_metrics, output_dir)

    # Quantitative comparison: per-metric MAE and aggregate
    df_comparison = None
    if 'real' in all_metrics:
        real_means = {}
        for metric in METRIC_LABELS:
            vals = all_metrics['real'][metric]
            vals = vals[~np.isnan(vals)]
            real_means[metric] = vals.mean()

        comp_rows = []
        for name in all_metrics:
            if name == 'real':
                continue
            row = {'Model': name}
            abs_errors = []
            for metric in METRIC_LABELS:
                vals = all_metrics[name][metric]
                vals = vals[~np.isnan(vals)]
                mae = abs(vals.mean() - real_means[metric])
                row[METRIC_LABELS[metric]] = f"{mae:.4f}"
                abs_errors.append(mae)
            row['Aggregate MAE'] = f"{np.mean(abs_errors):.4f} +/- {np.std(abs_errors):.4f}"
            comp_rows.append(row)

        df_comparison = pd.DataFrame(comp_rows).set_index('Model')
        print("\nPer-metric MAE (|model_mean - real_mean|):")
        print(df_comparison)

    return df_summary, df_comparison

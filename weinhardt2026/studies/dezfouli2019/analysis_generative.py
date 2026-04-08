import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.castro2025.benchmarking_castro2025 import get_dataset


METRIC_LABELS = {
    'avg_reward': 'Average Reward',
    'total_reward': 'Total Reward',
    'win_stay': 'P(Stay | Win)',
    'lose_stay': 'P(Stay | Loss)',
    'p_stay': 'P(Stay)',
}


def _extract_choices_and_rewards(dataset: SpiceDataset):
    """Extract per-session choice indices and scalar rewards from a SpiceDataset.

    Returns:
        choices: (n_sessions, n_trials) float array (NaN for padded trials)
        rewards: (n_sessions, n_trials) float array (NaN for padded trials)
    """
    n_actions = dataset.n_actions
    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)

    actions_oh = xs[:, :, :n_actions]
    rewards_oh = xs[:, :, n_actions:2 * n_actions]

    valid = ~torch.isnan(xs[:, :, 0])

    choices = actions_oh.argmax(dim=-1).float().numpy()
    rewards = rewards_oh.nan_to_num(0.0).sum(dim=-1).numpy()

    choices[~valid.numpy()] = np.nan
    rewards[~valid.numpy()] = np.nan

    return choices, rewards


def _compute_metrics(choices, rewards):
    """Compute behavioral metrics from choices and rewards arrays."""
    n_sessions = choices.shape[0]

    stays = (choices[:, 1:] == choices[:, :-1]).astype(float)
    prev_rewards = rewards[:, :-1]

    # Where either trial is NaN-padded, stay is undefined
    invalid = np.isnan(choices[:, 1:]) | np.isnan(choices[:, :-1])
    stays[invalid] = np.nan

    avg_reward = np.nanmean(rewards, axis=1)
    total_reward = np.nansum(rewards, axis=1)

    win_stay = np.full(n_sessions, np.nan)
    lose_stay = np.full(n_sessions, np.nan)
    for s in range(n_sessions):
        valid_t = ~np.isnan(prev_rewards[s]) & ~np.isnan(stays[s])
        wins = valid_t & (prev_rewards[s] > 0.5)
        losses = valid_t & (prev_rewards[s] < 0.5)
        if wins.sum() > 0:
            win_stay[s] = stays[s][wins].mean()
        if losses.sum() > 0:
            lose_stay[s] = stays[s][losses].mean()

    p_stay = np.nanmean(stays, axis=1)

    return {
        'avg_reward': avg_reward,
        'total_reward': total_reward,
        'win_stay': win_stay,
        'lose_stay': lose_stay,
        'p_stay': p_stay,
    }


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

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, 'generative_behavior.pdf'),
        bbox_inches='tight',
        dpi=150,
    )
    fig.savefig(
        os.path.join(output_dir, 'generative_behavior.png'),
        bbox_inches='tight',
        dpi=150,
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
    """Run generative behavior analysis by comparing datasets loaded from CSV files.

    Loads each dataset via get_dataset(), extracts behavioral metrics, and
    produces violin plots comparing across conditions (real data and model-generated).

    Args:
        datasets: Dict mapping display names to CSV file paths.
            Example: {'Real Data': 'data/eckstein2024.csv', 'SPICE': 'results/gen_spice.csv'}
        output_dir: Directory for output plots.
        test_sessions: Session indices for train/test split (forwarded to get_dataset).

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
        datasets['spice_rnn'] = path_data_spice
    if path_data_spice is not None:
        datasets['spice'] = path_data_spice_rnn
        
    for name, path in datasets.items():
        print(f"Loading {name} from {path}...")
        dataset_train, _, _ = get_dataset(path_data=path)
        choices, rewards = _extract_choices_and_rewards(dataset_train)
        all_metrics[name] = _compute_metrics(choices, rewards)

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

    df = pd.DataFrame(rows).set_index('Model')
    print(df)

    _plot_violins(all_metrics, output_dir)

    return df

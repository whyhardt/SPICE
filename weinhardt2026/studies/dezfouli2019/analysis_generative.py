import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import get_dataset
from weinhardt2026.analysis.analysis_generative_comparison import compute_generative_comparison


METRIC_LABELS = {
    'avg_reward': 'Average Reward',
    'total_reward': 'Total Reward',
    'win_stay': 'P(Stay | Win)',
    'lose_stay': 'P(Stay | Loss)',
    'lose_shift': 'P(Shift | Loss)',
    'p_stay': 'P(Stay)',
    'choice_entropy': 'Choice Entropy',
    'early_reward': 'Early Reward (Q1)',
    'late_reward': 'Late Reward (Q4)',
    'learning_slope': 'Learning Slope',
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
    """Compute behavioral metrics from choices and rewards arrays.

    Returns per-session arrays for each metric.
    """
    n_sessions = choices.shape[0]
    n_trials = choices.shape[1]

    stays = (choices[:, 1:] == choices[:, :-1]).astype(float)
    prev_rewards = rewards[:, :-1]

    # Where either trial is NaN-padded, stay is undefined
    invalid = np.isnan(choices[:, 1:]) | np.isnan(choices[:, :-1])
    stays[invalid] = np.nan

    avg_reward = np.nanmean(rewards, axis=1)
    total_reward = np.nansum(rewards, axis=1)

    win_stay = np.full(n_sessions, np.nan)
    lose_stay = np.full(n_sessions, np.nan)
    lose_shift = np.full(n_sessions, np.nan)
    choice_entropy = np.full(n_sessions, np.nan)
    early_reward = np.full(n_sessions, np.nan)
    late_reward = np.full(n_sessions, np.nan)
    learning_slope = np.full(n_sessions, np.nan)

    for s in range(n_sessions):
        valid_t = ~np.isnan(prev_rewards[s]) & ~np.isnan(stays[s])
        wins = valid_t & (prev_rewards[s] > 0.5)
        losses = valid_t & (prev_rewards[s] < 0.5)
        if wins.sum() > 0:
            win_stay[s] = stays[s][wins].mean()
        if losses.sum() > 0:
            lose_stay[s] = stays[s][losses].mean()
            lose_shift[s] = 1.0 - lose_stay[s]

        # Choice entropy: -sum(p * log2(p)) over action frequencies
        valid_choices = choices[s][~np.isnan(choices[s])]
        if len(valid_choices) > 1:
            _, counts = np.unique(valid_choices, return_counts=True)
            probs = counts / counts.sum()
            choice_entropy[s] = -np.sum(probs * np.log2(np.clip(probs, 1e-10, 1.0)))

        # Early vs late reward (first and last quarter of valid trials)
        valid_rewards = rewards[s][~np.isnan(rewards[s])]
        n_valid = len(valid_rewards)
        if n_valid >= 4:
            q = n_valid // 4
            early_reward[s] = valid_rewards[:q].mean()
            late_reward[s] = valid_rewards[-q:].mean()
            learning_slope[s] = late_reward[s] - early_reward[s]

    p_stay = np.nanmean(stays, axis=1)

    return {
        'avg_reward': avg_reward,
        'total_reward': total_reward,
        'win_stay': win_stay,
        'lose_stay': lose_stay,
        'lose_shift': lose_shift,
        'p_stay': p_stay,
        'choice_entropy': choice_entropy,
        'early_reward': early_reward,
        'late_reward': late_reward,
        'learning_slope': learning_slope,
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


def _aggregate_metrics_per_participant(all_metrics, participant_ids, output_dir):
    """Aggregate session-level metrics to per-participant means and save as CSV.

    For each model (e.g. 'real', 'spice', 'benchmark'), produces a DataFrame
    with one row per participant and one column per metric.

    Args:
        all_metrics: {model_name: {metric_name: (n_sessions,) array}}
        participant_ids: {model_name: (n_sessions,) array}
        output_dir: Directory for output CSVs.

    Returns:
        Dict mapping model names to per-participant DataFrames.
    """
    result = {}
    for name in all_metrics:
        if name not in participant_ids:
            continue
        pids = participant_ids[name]
        unique_pids = np.unique(pids[~np.isnan(pids)])

        rows = []
        for pid in unique_pids:
            mask = pids == pid
            row = {'participant_id': int(pid)}
            for metric, values in all_metrics[name].items():
                vals = values[mask]
                clean = vals[~np.isnan(vals)]
                row[metric] = clean.mean() if len(clean) > 0 else np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        result[name] = df

        if output_dir is not None:
            path = os.path.join(output_dir, f'behavioral_metrics_{name}.csv')
            df.to_csv(path, index=False)
            print(f"Saved per-participant metrics: {path}")

    return result


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
            Example: {'Real Data': 'data/eckstein2024.csv', 'SPICE-EQ': 'results/gen_spice.csv'}
        output_dir: Directory for output plots.
        test_blocks: Session indices for train/test split (forwarded to get_dataset).

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

    participant_ids = {}
    for name, data in datasets.items():
        if isinstance(data, str):
            print(f"Loading {name} from {data}...")
            dataset, _, _ = get_dataset(path_data=data)
        else:
            dataset = data
        participant_ids[name] = dataset.xs[:, 0, 0, -1].long().numpy()
        choices, rewards = _extract_choices_and_rewards(dataset)
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

    # Distributional similarity + Spearman comparison
    df_similarity, df_spearman = None, None
    if 'real' in all_metrics and participant_ids:
        df_similarity, df_spearman = compute_generative_comparison(
            all_metrics, participant_ids, output_dir,
        )

    # Per-participant behavioral metrics (aggregated from sessions)
    df_participants = _aggregate_metrics_per_participant(
        all_metrics, participant_ids, output_dir,
    )

    return df, df_similarity, df_spearman

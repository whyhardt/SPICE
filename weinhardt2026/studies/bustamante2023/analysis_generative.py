import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import get_dataset
from weinhardt2026.analysis.analysis_generative_comparison import compute_generative_comparison


METRIC_LABELS = {
    'p_harvest': 'P(Harvest)',
    'patch_residence': 'Mean Patch Residence',
    'exit_reward': 'Mean Exit Reward',
    'avg_reward': 'Average Reward',
    'total_reward': 'Total Reward',
}


def _extract_data(dataset: SpiceDataset):
    """Extract choices and rewards from a Bustamante 2023 dataset.

    Feature layout (n_actions=2, n_reward_features=2):
        [action_harvest, action_exit, reward_harvest, reward_exit,
         harvest_duration, travel_duration, time_trial, trials, block,
         experiment_id, participant_id]

    Returns:
        choices: (n_sessions, n_trials) float — 0 = harvest, 1 = exit, NaN = padded
        rewards: (n_sessions, n_trials) float — scalar reward, NaN = padded
    """
    n_actions = dataset.n_actions
    xs = dataset.xs[:, :, 0, :]

    valid = ~torch.isnan(xs[:, :, 0])

    choices = xs[:, :, :n_actions].argmax(dim=-1).float().numpy()
    rewards = xs[:, :, n_actions:2 * n_actions].nan_to_num(0.0).sum(dim=-1).numpy()

    choices[~valid.numpy()] = np.nan
    rewards[~valid.numpy()] = np.nan

    return choices, rewards


def _compute_metrics(choices, rewards):
    """Compute foraging-specific metrics per session."""
    n_sessions = choices.shape[0]

    metrics = {k: np.full(n_sessions, np.nan) for k in METRIC_LABELS}

    metrics['avg_reward'] = np.nanmean(rewards, axis=1)
    metrics['total_reward'] = np.nansum(rewards, axis=1)

    for s in range(n_sessions):
        c = choices[s]
        r = rewards[s]
        valid = ~np.isnan(c)
        c_valid = c[valid]
        r_valid = r[valid]

        if len(c_valid) == 0:
            continue

        # P(harvest)
        metrics['p_harvest'][s] = (c_valid == 0).mean()

        # Segment into patches: a patch is a run of harvest (0) actions
        # ending with an exit (1) or end of session.
        patch_lengths = []
        exit_rewards = []
        run_length = 0
        last_harvest_reward = np.nan
        for t in range(len(c_valid)):
            if c_valid[t] == 0:  # harvest
                run_length += 1
                last_harvest_reward = r_valid[t]
            else:  # exit
                if run_length > 0:
                    patch_lengths.append(run_length)
                    exit_rewards.append(last_harvest_reward)
                run_length = 0
                last_harvest_reward = np.nan
        # Don't count a trailing harvest run with no exit as a complete patch

        if len(patch_lengths) > 0:
            metrics['patch_residence'][s] = np.mean(patch_lengths)
            exit_rews = [r for r in exit_rewards if not np.isnan(r)]
            if len(exit_rews) > 0:
                metrics['exit_reward'][s] = np.mean(exit_rews)

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
    """Run generative behavior analysis for the Bustamante 2023 foraging task.

    Computes foraging metrics (harvest probability, patch residence,
    exit reward, average and total reward) and produces violin plots.

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

    participant_ids = {}
    for name, path in datasets.items():
        print(f"Loading {name} from {path}...")
        dataset, _, _ = get_dataset(path_data=path, test_blocks=())
        participant_ids[name] = dataset.xs[:, 0, 0, -1].long().numpy()
        choices, rewards = _extract_data(dataset)
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

    df_summary = pd.DataFrame(rows).set_index('Model')
    print(df_summary)

    _plot_violins(all_metrics, output_dir)

    # Quantitative comparison: per-metric MAE and aggregate
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

    # Distributional similarity + Spearman comparison
    df_similarity, df_spearman = None, None
    if 'real' in all_metrics and participant_ids:
        df_similarity, df_spearman = compute_generative_comparison(
            all_metrics, participant_ids, output_dir,
        )

    return df_summary, df_comparison, df_similarity, df_spearman

import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.braun2018.benchmarking_braun2018 import get_dataset
from weinhardt2026.analysis.analysis_generative_comparison import compute_generative_comparison


METRIC_LABELS = {
    'p_switch': 'P(Switch)',
    'p_switch_both_change': 'P(Switch | Both Change)',
    'p_switch_neither_change': 'P(Switch | Neither Change)',
    'p_switch_current_decrease': 'P(Switch | Current Decrease)',
    'p_switch_other_increase': 'P(Switch | Other Increase)',
    'p_switch_diff_neg1': 'P(Switch | Diff = -1)',
    'p_switch_diff_0': 'P(Switch | Diff = 0)',
    'p_switch_diff_pos1': 'P(Switch | Diff = +1)',
}


def _extract_data(dataset: SpiceDataset):
    """Extract choices and reward-condition indicators from a Braun 2018 dataset.

    Feature layout (n_actions=2, no reward columns):
        [action_repeat, action_switch, difference, current_decreased,
         other_increased, time_trial, trials, block, experiment_id, participant_id]

    Returns:
        switches:     (n_sessions, n_trials) float — 1 = switch, 0 = repeat, NaN = padded
        current_dec:  (n_sessions, n_trials) float — 1 = current task value decreased
        other_inc:    (n_sessions, n_trials) float — 1 = other task value increased
        difference:   (n_sessions, n_trials) float — point difference (other - current)
    """
    n_actions = dataset.n_actions
    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)

    valid = ~torch.isnan(xs[:, :, 0])

    choices = xs[:, :, :n_actions].argmax(dim=-1).float().numpy()  # 0=repeat, 1=switch
    current_dec = xs[:, :, n_actions + 1].numpy()  # column index 3
    other_inc = xs[:, :, n_actions + 2].numpy()    # column index 4
    difference = xs[:, :, n_actions].numpy()       # column index 2

    choices[~valid.numpy()] = np.nan
    current_dec[~valid.numpy()] = np.nan
    other_inc[~valid.numpy()] = np.nan
    difference[~valid.numpy()] = np.nan

    return choices, current_dec, other_inc, difference


def _conditional_mean(switches, mask):
    """Mean of switches where mask is True, or NaN if no valid trials."""
    if mask.sum() > 0:
        return switches[mask].mean()
    return np.nan


def _compute_metrics(choices, current_dec, other_inc, difference):
    """Compute task-switching metrics per session."""
    n_sessions = choices.shape[0]

    switches = choices  # 1 = switch, 0 = repeat

    metrics = {k: np.full(n_sessions, np.nan) for k in METRIC_LABELS}
    metrics['p_switch'] = np.nanmean(switches, axis=1)

    for s in range(n_sessions):
        valid = ~np.isnan(switches[s]) & ~np.isnan(current_dec[s]) & ~np.isnan(other_inc[s])
        sw = switches[s]
        cd = current_dec[s]
        oi = other_inc[s]
        diff = difference[s]

        # 2x2 factorial conditions
        metrics['p_switch_both_change'][s] = _conditional_mean(sw, valid & (cd == 1) & (oi == 1))
        metrics['p_switch_neither_change'][s] = _conditional_mean(sw, valid & (cd == 0) & (oi == 0))
        metrics['p_switch_current_decrease'][s] = _conditional_mean(sw, valid & (cd == 1))
        metrics['p_switch_other_increase'][s] = _conditional_mean(sw, valid & (oi == 1))

        # Point difference conditions
        valid_diff = valid & ~np.isnan(diff)
        metrics['p_switch_diff_neg1'][s] = _conditional_mean(sw, valid_diff & (diff == -1))
        metrics['p_switch_diff_0'][s] = _conditional_mean(sw, valid_diff & (diff == 0))
        metrics['p_switch_diff_pos1'][s] = _conditional_mean(sw, valid_diff & (diff == 1))

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
    """Run generative behavior analysis for the Braun 2018 rVTS task.

    Computes task-switching metrics (overall switch rate, switch rate
    conditioned on value-change conditions and point difference) and
    produces violin plots.

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
    for name, data in datasets.items():
        if isinstance(data, str):
            print(f"Loading {name} from {data}...")
            dataset, _, _ = get_dataset(path_data=data, test_blocks=())
        else:
            dataset = data
        participant_ids[name] = dataset.xs[:, 0, 0, -1].long().numpy()
        choices, current_dec, other_inc, difference = _extract_data(dataset)
        all_metrics[name] = _compute_metrics(choices, current_dec, other_inc, difference)

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

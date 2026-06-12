import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.hwang2026.benchmarking_hwang2026 import get_dataset, ACTION_NAMES


METRIC_LABELS = {
    'p_action': 'P(Action)',
    'p_grooming': 'P(Grooming)',
    'p_gesture': 'P(Gesture)',
    'p_scratch': 'P(Scratch)',
    'p_switch': 'P(Switch)',
    'p_groom_given_partner_groom': 'P(Groom | Partner Groom)',
    'p_groom_given_partner_action': 'P(Groom | Partner Action)',
    'p_groom_given_partner_gesture': 'P(Groom | Partner Gesture)',
    'p_groom_sender_lower': 'P(Groom | Sender Lower Rank)',
    'p_groom_sender_higher': 'P(Groom | Sender Higher Rank)',
}


def _get_rank_mapping() -> dict[int, int]:
    """Load ape ID → dominance rank mapping from the original CSV."""
    path_data = 'weinhardt2026/studies/hwang2026/data/hwang2025_processed.csv'
    df = pd.read_csv(path_data)
    # Build mapping from both ID1 and ID2 columns
    rank_map = {}
    for _, row in df[['ID1', 'Dominance rank_ID1']].drop_duplicates().iterrows():
        rank_map[int(row['ID1'])] = int(row['Dominance rank_ID1'])
    for _, row in df[['ID2', 'Dominance rank_ID2']].drop_duplicates().iterrows():
        rank_map[int(row['ID2'])] = int(row['Dominance rank_ID2'])
    return rank_map


def _extract_data(dataset: SpiceDataset):
    """Extract choices and partner actions from a hwang2026 dataset.

    Feature layout (n_actions=4, no reward columns):
        [action_oh(4), SigAct_ID2, ID1, ID2, time_trial, trials, block, experiment_id, participant_id]

    Returns:
        choices: (n_sessions, n_trials) int — action category index (0-3), NaN for padded.
        partner_actions: (n_sessions, n_trials) int — partner's action category, NaN for padded.
        id1s: (n_sessions,) int — sender ID per session.
        id2s: (n_sessions,) int — receiver ID per session.
    """
    n_actions = dataset.n_actions
    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)

    valid = ~torch.isnan(xs[:, :, 0])

    choices = xs[:, :, :n_actions].nan_to_num(0).argmax(dim=-1).float().numpy()
    partner_actions = xs[:, :, n_actions].numpy()  # SigAct_ID2

    choices[~valid.numpy()] = np.nan
    partner_actions[~valid.numpy()] = np.nan

    id1s = xs[:, 0, -1].nan_to_num(0).long().numpy()
    id2s = xs[:, 0, -2].nan_to_num(0).long().numpy()

    return choices, partner_actions, id1s, id2s


def _compute_metrics(choices, partner_actions, id1s, id2s, rank_map=None):
    """Compute per-session behavioral metrics for chimpanzee communication.

    Args:
        choices: (n_sessions, n_trials) float — sender's action (0-3), NaN for padded.
        partner_actions: (n_sessions, n_trials) float — partner's action, NaN for padded.
        id1s: (n_sessions,) int — sender IDs.
        id2s: (n_sessions,) int — receiver IDs.
        rank_map: dict mapping ape ID → dominance rank (optional).

    Returns:
        dict mapping metric name → (n_sessions,) numpy array.
    """
    n_sessions = choices.shape[0]

    metrics = {k: np.full(n_sessions, np.nan) for k in METRIC_LABELS}

    for s in range(n_sessions):
        valid = ~np.isnan(choices[s])
        ch = choices[s][valid]
        pa = partner_actions[s][valid]
        pa_valid = ~np.isnan(pa)

        if len(ch) == 0:
            continue

        # Action frequencies
        metrics['p_action'][s] = (ch == 0).mean()
        metrics['p_grooming'][s] = (ch == 1).mean()
        metrics['p_gesture'][s] = (ch == 2).mean()
        metrics['p_scratch'][s] = (ch == 3).mean()

        # Switching rate
        if len(ch) > 1:
            switches = (ch[1:] != ch[:-1])
            metrics['p_switch'][s] = switches.mean()

        # Conditional grooming given partner's action
        pa_ch = pa[pa_valid]
        ch_pa = ch[pa_valid]
        if len(pa_ch) > 0:
            for act_cat, metric_key in [(1, 'p_groom_given_partner_groom'),
                                         (0, 'p_groom_given_partner_action'),
                                         (2, 'p_groom_given_partner_gesture')]:
                mask = pa_ch == act_cat
                if mask.sum() > 0:
                    metrics[metric_key][s] = (ch_pa[mask] == 1).mean()

        # Rank-conditional grooming
        if rank_map is not None:
            id1, id2 = int(id1s[s]), int(id2s[s])
            rank1 = rank_map.get(id1)
            rank2 = rank_map.get(id2)
            if rank1 is not None and rank2 is not None:
                groom_rate = (ch == 1).mean()
                if rank1 < rank2:
                    metrics['p_groom_sender_lower'][s] = groom_rate
                elif rank1 > rank2:
                    metrics['p_groom_sender_higher'][s] = groom_rate

    return metrics


def _plot_violins(all_metrics, output_dir):
    """Create violin plots comparing behavioral metrics across datasets."""
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

        if all(len(d) == 0 for d in data):
            ax.set_visible(False)
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
    fig.savefig(os.path.join(output_dir, 'generative_behavior.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, 'generative_behavior.png'), bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def analysis_generative_behavior(
    dataset_real: SpiceDataset = None,
    dataset_gru: SpiceDataset = None,
    dataset_spice_rnn: SpiceDataset = None,
    dataset_spice: SpiceDataset = None,
    output_dir: str = 'results',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run generative behavior analysis for the hwang2026 chimpanzee communication task.

    Computes behavioral metrics (action frequencies, switching rates,
    grooming contingencies, rank-conditional grooming) and produces violin plots.

    Args:
        dataset_real: Real behavioral SpiceDataset.
        dataset_gru: GRU-generated SpiceDataset.
        dataset_spice_rnn: SPICE-RNN generated SpiceDataset.
        dataset_spice: SPICE equation-generated SpiceDataset.
        output_dir: Directory for output plots.

    Returns:
        df_summary: DataFrame with mean +/- std per metric per model.
        df_comparison: DataFrame with NMAE per metric (if real data provided).
    """
    os.makedirs(output_dir, exist_ok=True)

    rank_map = _get_rank_mapping()

    datasets = {}
    if dataset_real is not None:
        datasets['real'] = dataset_real
    if dataset_gru is not None:
        datasets['gru'] = dataset_gru
    if dataset_spice_rnn is not None:
        datasets['spice_rnn'] = dataset_spice_rnn
    if dataset_spice is not None:
        datasets['spice'] = dataset_spice

    all_metrics = {}
    for name, ds in datasets.items():
        print(f"Computing metrics for {name}...")
        choices, partner_actions, id1s, id2s = _extract_data(ds)
        all_metrics[name] = _compute_metrics(choices, partner_actions, id1s, id2s, rank_map)

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
            real_means[metric] = vals.mean() if len(vals) > 0 else np.nan
            real_stds[metric] = vals.std() if len(vals) > 0 else np.nan

        comp_rows = []
        for name in all_metrics:
            if name == 'real':
                continue
            row = {'Model': name}
            norm_errors = []
            for metric in METRIC_LABELS:
                vals = all_metrics[name][metric]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0 and not np.isnan(real_means[metric]):
                    mae = abs(vals.mean() - real_means[metric])
                    nmae = mae / real_stds[metric] if real_stds[metric] > 0 else 0.0
                    row[METRIC_LABELS[metric]] = f"{nmae:.4f}"
                    norm_errors.append(nmae)
                else:
                    row[METRIC_LABELS[metric]] = "N/A"
            if norm_errors:
                row['Aggregate NMAE'] = f"{np.mean(norm_errors):.4f} +/- {np.std(norm_errors):.4f}"
            else:
                row['Aggregate NMAE'] = "N/A"
            comp_rows.append(row)

        df_comparison = pd.DataFrame(comp_rows).set_index('Model')
        print("\nNormalized MAE (|model_mean - real_mean| / real_std):")
        print(df_comparison)

    return df_summary, df_comparison

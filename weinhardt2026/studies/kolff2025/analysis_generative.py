import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spice import SpiceDataset
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import (
    ACTION_NAMES,
    DEFAULT_DATA_PATH,
    GROOMING_ACTION,
    LOWER_STATUS_HAS_LARGER_RANK_NUMBER,
)
from weinhardt2026.analysis.analysis_generative_comparison import compute_generative_comparison


LOWER_RANK = 'sender_lower_rank'
HIGHER_RANK = 'sender_higher_rank'


def _action_slug(action_name: str) -> str:
    return action_name.lower().replace(' ', '_')


ACTION_FREQUENCY_METRIC_LABELS = {
    f'p_{_action_slug(action_name)}': f'P({action_name.title()})'
    for action_name in ACTION_NAMES.values()
}

OVERVIEW_METRIC_LABELS = {
    **ACTION_FREQUENCY_METRIC_LABELS,
    'p_switch': 'P(Switch)',
    'p_groom_sender_lower_rank': 'P(Next Groom | Sender Lower Rank)',
    'p_groom_sender_higher_rank': 'P(Next Groom | Sender Higher Rank)',
}

CONDITIONAL_METRIC_LABELS = {}
for action_name in ACTION_NAMES.values():
    slug = _action_slug(action_name)
    label = action_name.title()
    CONDITIONAL_METRIC_LABELS[f'p_groom_given_partner_{slug}'] = (
        f'P(Next Groom | Partner {label})'
    )
    CONDITIONAL_METRIC_LABELS[f'p_groom_given_partner_{slug}_sender_lower_rank'] = (
        f'P(Next Groom | Partner {label}, Sender Lower Rank)'
    )
    CONDITIONAL_METRIC_LABELS[f'p_groom_given_partner_{slug}_sender_higher_rank'] = (
        f'P(Next Groom | Partner {label}, Sender Higher Rank)'
    )

METRIC_LABELS = {
    **OVERVIEW_METRIC_LABELS,
    **CONDITIONAL_METRIC_LABELS,
}
OVERVIEW_METRICS = list(OVERVIEW_METRIC_LABELS.keys())
CONDITIONAL_METRICS = list(CONDITIONAL_METRIC_LABELS.keys())


def _get_rank_mapping(path_data: str = DEFAULT_DATA_PATH) -> dict[int, int]:
    """Load ape ID -> dominance-rank mapping from the original CSV."""
    if path_data is None or not os.path.exists(path_data):
        return {}

    df = pd.read_csv(path_data)
    rank_map = {}
    if {'ID1', 'Dominance rank_ID1'}.issubset(df.columns):
        for _, row in df[['ID1', 'Dominance rank_ID1']].dropna().drop_duplicates().iterrows():
            rank_map[int(row['ID1'])] = int(row['Dominance rank_ID1'])
    if {'ID2', 'Dominance rank_ID2'}.issubset(df.columns):
        for _, row in df[['ID2', 'Dominance rank_ID2']].dropna().drop_duplicates().iterrows():
            rank_map[int(row['ID2'])] = int(row['Dominance rank_ID2'])
    return rank_map


def _extract_data(dataset: SpiceDataset):
    """Extract choices, partner actions, and per-trial sender/receiver IDs.

    Feature layout for the current Hwang setup is:
        [action_oh(4), SigAct_ID2, ID1, ID2, time_trial, trial, block, experiment_id, participant_id]

    SPICE datasets store model inputs in xs and response targets in ys. For this
    analysis the focal behavior is the response action, so choices must be read
    from ys while partner-action conditions and IDs come from xs.

    Returns:
        choices: (n_sessions, n_trials) float, action index or NaN for padding.
        partner_actions: (n_sessions, n_trials) float, partner action or NaN.
        sender_ids: (n_sessions, n_trials) float, current ID1/sender or NaN.
        receiver_ids: (n_sessions, n_trials) float, current ID2/receiver or NaN.
    """
    n_actions = dataset.n_actions
    xs = dataset.xs.detach().cpu()[:, :, 0, :]
    ys = dataset.ys.detach().cpu()[:, :, 0, :]

    valid = (~torch.isnan(xs[:, :, 0])) & (~torch.isnan(ys[:, :, 0]))
    valid_np = valid.numpy()

    choices = ys[:, :, :n_actions].nan_to_num(0).argmax(dim=-1).float().numpy()
    partner_actions = xs[:, :, n_actions].float().numpy()
    sender_ids = xs[:, :, -1].float().numpy()
    receiver_ids = xs[:, :, -2].float().numpy()

    choices[~valid_np] = np.nan
    partner_actions[~valid_np] = np.nan
    sender_ids[~valid_np] = np.nan
    receiver_ids[~valid_np] = np.nan

    return choices, partner_actions, sender_ids, receiver_ids


def _rank_relation(sender_id: float, receiver_id: float, rank_map: dict[int, int]) -> str | None:
    """Return rank relation for a generated/real directed row.

    The rank-number convention is controlled by LOWER_STATUS_HAS_LARGER_RANK_NUMBER.
    """
    if np.isnan(sender_id) or np.isnan(receiver_id):
        return None

    sender_rank = rank_map.get(int(sender_id))
    receiver_rank = rank_map.get(int(receiver_id))
    if sender_rank is None or receiver_rank is None or sender_rank == receiver_rank:
        return None
    if LOWER_STATUS_HAS_LARGER_RANK_NUMBER:
        return LOWER_RANK if sender_rank > receiver_rank else HIGHER_RANK
    return LOWER_RANK if sender_rank < receiver_rank else HIGHER_RANK


def _relation_mask(
    sender_ids: np.ndarray,
    receiver_ids: np.ndarray,
    rank_map: dict[int, int],
    relation: str,
) -> np.ndarray:
    mask = np.zeros(sender_ids.shape, dtype=bool)
    for idx, (sender_id, receiver_id) in enumerate(zip(sender_ids, receiver_ids)):
        mask[idx] = _rank_relation(sender_id, receiver_id, rank_map) == relation
    return mask


def _compute_metrics(choices, partner_actions, sender_ids, receiver_ids, rank_map=None):
    """Compute per-session behavioral metrics for chimpanzee communication."""
    n_sessions = choices.shape[0]
    rank_map = rank_map or {}

    metrics = {k: np.full(n_sessions, np.nan) for k in METRIC_LABELS}

    for s in range(n_sessions):
        valid = ~np.isnan(choices[s])
        ch = choices[s][valid]
        pa = partner_actions[s][valid]
        sid = sender_ids[s][valid]
        rid = receiver_ids[s][valid]

        if len(ch) == 0:
            continue

        for action_id, action_name in ACTION_NAMES.items():
            metrics[f'p_{_action_slug(action_name)}'][s] = (ch == action_id).mean()

        if len(ch) > 1:
            metrics['p_switch'][s] = (ch[1:] != ch[:-1]).mean()

        lower_mask = _relation_mask(sid, rid, rank_map, LOWER_RANK)
        higher_mask = _relation_mask(sid, rid, rank_map, HIGHER_RANK)

        if lower_mask.sum() > 0:
            metrics['p_groom_sender_lower_rank'][s] = (ch[lower_mask] == GROOMING_ACTION).mean()
        if higher_mask.sum() > 0:
            metrics['p_groom_sender_higher_rank'][s] = (ch[higher_mask] == GROOMING_ACTION).mean()

        pa_valid = ~np.isnan(pa)
        for action_id, action_name in ACTION_NAMES.items():
            slug = _action_slug(action_name)
            action_mask = pa_valid & (pa == action_id)
            if action_mask.sum() > 0:
                metrics[f'p_groom_given_partner_{slug}'][s] = (
                    ch[action_mask] == GROOMING_ACTION
                ).mean()

            lower_action_mask = action_mask & lower_mask
            if lower_action_mask.sum() > 0:
                metrics[f'p_groom_given_partner_{slug}_sender_lower_rank'][s] = (
                    ch[lower_action_mask] == GROOMING_ACTION
                ).mean()

            higher_action_mask = action_mask & higher_mask
            if higher_action_mask.sum() > 0:
                metrics[f'p_groom_given_partner_{slug}_sender_higher_rank'][s] = (
                    ch[higher_action_mask] == GROOMING_ACTION
                ).mean()

    return metrics


def _plot_violins(all_metrics, output_dir, metric_names, filename):
    """Create violin plots comparing behavioral metrics across datasets."""
    model_names = list(all_metrics.keys())
    if not model_names or not metric_names:
        return

    colors = plt.cm.tab10.colors
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        plot_data = []
        plot_positions = []
        plot_model_indices = []
        for model_idx, model_name in enumerate(model_names):
            values = all_metrics[model_name][metric]
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            plot_data.append(values)
            plot_positions.append(model_idx + 1)
            plot_model_indices.append(model_idx)

        if not plot_data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(
            plot_data,
            positions=plot_positions,
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )

        for body, model_idx in zip(parts['bodies'], plot_model_indices):
            body.set_facecolor(colors[model_idx % len(colors)])
            body.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('gray')
        parts['cmedians'].set_linestyle('--')

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, fontsize=9, rotation=20, ha='right')
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def analysis_generative_behavior(
    dataset_real: SpiceDataset = None,
    dataset_gru: SpiceDataset = None,
    dataset_spice_rnn: SpiceDataset = None,
    dataset_spice: SpiceDataset = None,
    output_dir: str = 'results',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run generative behavior analysis for the Hwang chimpanzee task.

    Computes action frequencies plus task-specific conditional grooming metrics:
    P(Grooming | partner action category) and the same quantities split by
    sender/receiver dominance-rank relation.
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
        datasets['spice_eq'] = dataset_spice

    all_metrics = {}
    for name, ds in datasets.items():
        print(f"Computing metrics for {name}...")
        choices, partner_actions, sender_ids, receiver_ids = _extract_data(ds)
        all_metrics[name] = _compute_metrics(
            choices,
            partner_actions,
            sender_ids,
            receiver_ids,
            rank_map,
        )

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
    df_summary.to_csv(os.path.join(output_dir, 'generative_behavior_summary.csv'))

    _plot_violins(all_metrics, output_dir, OVERVIEW_METRICS, 'generative_behavior')
    _plot_violins(
        all_metrics,
        output_dir,
        CONDITIONAL_METRICS,
        'generative_behavior_conditional_grooming',
    )

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
        df_comparison.to_csv(os.path.join(output_dir, 'generative_behavior_nmae.csv'))

    # Distributional similarity + Spearman comparison
    df_similarity, df_spearman = None, None
    if 'real' in all_metrics:
        pid_dict = {
            name: ds.xs[:, 0, 0, -1].long().numpy()
            for name, ds in datasets.items()
        }
        df_similarity, df_spearman = compute_generative_comparison(
            all_metrics, pid_dict, output_dir,
        )

    return df_summary, df_comparison, df_similarity, df_spearman

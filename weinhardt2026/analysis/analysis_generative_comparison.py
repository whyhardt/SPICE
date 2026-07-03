import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, spearmanr


def compute_generative_comparison(
    all_metrics: dict[str, dict[str, np.ndarray]],
    participant_ids: dict[str, np.ndarray],
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute distributional similarity and individual-differences recovery.

    Args:
        all_metrics: ``{model_name: {metric_name: (n_sessions,) array}}``.
            Must include ``'real'`` as one of the model names.
        participant_ids: ``{model_name: (n_sessions,) array}`` of participant
            IDs per session for each model (including ``'real'``).  Each
            model's array length must match the corresponding arrays in
            ``all_metrics``.
        output_dir: If provided, save CSV files to this directory.

    Returns:
        df_similarity: 1 - Wasserstein distance per metric per model.
        df_spearman: Spearman rho per metric per model.
    """
    if 'real' not in all_metrics:
        raise ValueError("all_metrics must contain a 'real' key")

    metric_names = list(all_metrics['real'].keys())
    model_names = [m for m in all_metrics if m != 'real']

    if len(model_names) == 0:
        empty = pd.DataFrame()
        return empty, empty

    # ------------------------------------------------------------------
    # Normalize each metric to [0, 1]
    # lower = 0, upper = max observed across all models (incl. real)
    # ------------------------------------------------------------------
    upper_bounds = {}
    for metric in metric_names:
        global_max = 0.0
        for name in all_metrics:
            vals = all_metrics[name][metric]
            clean = vals[~np.isnan(vals)]
            if len(clean) > 0:
                global_max = max(global_max, clean.max())
        upper_bounds[metric] = global_max if global_max > 0 else 1.0

    def _normalize(values, metric):
        return values / upper_bounds[metric]

    # ------------------------------------------------------------------
    # Aggregate sessions to participants (mean per participant)
    # ------------------------------------------------------------------
    # Build the set of unique participant IDs across all models
    all_pids = []
    for pids in participant_ids.values():
        all_pids.append(pids[~np.isnan(pids)])
    unique_pids = np.unique(np.concatenate(all_pids)) if all_pids else np.array([])

    def _aggregate_to_participants(values, pids):
        """Return (n_participants,) array of per-participant means."""
        result = np.full(len(unique_pids), np.nan)
        for i, pid in enumerate(unique_pids):
            mask = pids == pid
            vals = values[mask]
            clean = vals[~np.isnan(vals)]
            if len(clean) > 0:
                result[i] = clean.mean()
        return result

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    real_pids = participant_ids.get('real', np.array([]))

    similarity_rows = []
    spearman_rows = []

    for model in model_names:
        sim_row = {'Model': model}
        spr_row = {'Model': model}
        sim_values = []
        spr_values = []

        model_pids = participant_ids.get(model, np.array([]))

        for metric in metric_names:
            real_vals = all_metrics['real'][metric]
            model_vals = all_metrics[model][metric]

            # Clean NaNs (pairwise for consistency)
            real_clean = real_vals[~np.isnan(real_vals)]
            model_clean = model_vals[~np.isnan(model_vals)]

            # --- Wasserstein (on normalized values) ---
            if len(real_clean) > 0 and len(model_clean) > 0:
                real_norm = _normalize(real_clean, metric)
                model_norm = _normalize(model_clean, metric)
                w = wasserstein_distance(real_norm, model_norm)
                sim = 1.0 - w
            else:
                sim = np.nan
            sim_row[metric] = sim
            if not np.isnan(sim):
                sim_values.append(sim)

            # --- Spearman (on per-participant means) ---
            if len(real_pids) > 0 and len(model_pids) > 0:
                real_part = _aggregate_to_participants(real_vals, real_pids)
                model_part = _aggregate_to_participants(model_vals, model_pids)
                valid = ~np.isnan(real_part) & ~np.isnan(model_part)
                if valid.sum() >= 3:
                    rho, _ = spearmanr(real_part[valid], model_part[valid])
                else:
                    rho = np.nan
            else:
                rho = np.nan
            spr_row[metric] = rho
            if not np.isnan(rho):
                spr_values.append(rho)

        sim_row['Mean'] = np.mean(sim_values) if sim_values else np.nan
        spr_row['Mean'] = np.mean(spr_values) if spr_values else np.nan
        similarity_rows.append(sim_row)
        spearman_rows.append(spr_row)

    df_similarity = pd.DataFrame(similarity_rows).set_index('Model')
    df_spearman = pd.DataFrame(spearman_rows).set_index('Model')

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df_similarity.to_csv(os.path.join(output_dir, 'generative_similarity.csv'))
        df_spearman.to_csv(os.path.join(output_dir, 'generative_spearman.csv'))

    return df_similarity, df_spearman

"""Deeper analyses on one fitted SPICE checkpoint.

Four analyses that read a fitted model and describe how its discovered
equations vary across participants:

  1. ``analysis_embedding_dynamics`` — the participants furthest apart in
     participant-embedding space, with their choice / value-update dynamics and
     their equations rendered as LaTeX.
  2. ``analysis_metric_fingerprint`` — structural (presence) and parametric
     (magnitude) coefficient differences ordered by a behavioral metric.
  3. ``analysis_group_betas`` — presence regression of every coefficient on a
     discrete grouping variable (e.g. a diagnosis).
  4. ``analysis_metric_betas`` — the same against a continuous participant-level
     metric.

Everything task-specific (which checkpoint, which metric, which grouping
column, how modules are named in LaTeX) is a parameter, so a study calls these
from its own ``<study>.py``.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform

from weinhardt2026.analysis.analysis_coefficients_individuals import (
    analysis_coefficients_individuals,
)
from weinhardt2026.figures.panel_utils import save_panel
from weinhardt2026.figures.figure2 import (
    select_distinctive_participants,
    _plot_panel_d_dynamics,
)
from weinhardt2026.figures.figure4 import (
    _extract_equation_features,
    _cluster_participants,
    _plot_panel_b,
    _plot_panel_d,
    low_variance_terms,
    term_metric_effects,
)
from weinhardt2026.utils.checkpoints import log_trials_per_participant


# ---------------------------------------------------------------------------
# LaTeX rendering of the discovered equations
# ---------------------------------------------------------------------------

def _latex_factor(factor, module_symbols):
    """Render one library factor (a signal, a state, or a power of either)."""
    factor = factor.strip()

    power = ''
    if '^' in factor:
        factor, _, exponent = factor.partition('^')
        power = f'^{{{exponent}}}'

    # Delayed signals: reward[t-1] -> r_{t-1}, choice[t] -> c_{t}
    match = re.fullmatch(r'(reward|choice)\[(t(?:-\d+)?)\]', factor)
    if match:
        symbol = 'r' if match.group(1) == 'reward' else 'c'
        return f'{symbol}_{{{match.group(2)}}}{power}'

    # Module states are evaluated at the current trial
    if factor in module_symbols:
        base = module_symbols[factor]
        return f'{base}(t){power}' if power else f'{base}(t)'

    return f'{factor}{power}'


def _latex_term(term, module_symbols):
    """Render a library term (a product of factors); '1' is the constant."""
    if term == '1':
        return ''
    return r' \cdot '.join(_latex_factor(f, module_symbols) for f in term.split('*'))


def format_equations_latex(estimator, participant_ids, participant_labels=None,
                           module_symbols=None, min_coef=0.0, precision=3):
    """LaTeX for each participant's discovered equation set.

    The SINDy library predicts the per-trial increment -- the model integrates
    `h_next = h + dt * (library @ coefficients)` with dt=1 -- so the left-hand
    side is the change in the state, and coefficients are printed exactly as
    stored (no identity term is folded in).

    *module_symbols* maps a module name to its LaTeX symbol; modules it omits
    are printed by name.
    """
    model = estimator.model
    module_symbols = module_symbols or {}
    coefs_dict = model.get_sindy_coefficients(aggregate=True)
    terms_dict = model.sindy_candidate_terms

    lines = []
    for col, pid in enumerate(participant_ids):
        header = f'Participant {pid}'
        if participant_labels is not None:
            header += f' ({participant_labels[col]})'
        lines.append(f'% ===== {header} =====')
        lines.append(r'\begin{align}')

        body = []
        for module in model.get_modules():
            coefs = coefs_dict[module][pid, 0].detach().cpu().numpy()
            terms = terms_dict[module]

            rendered = []
            for coef, term in zip(coefs, terms):
                coef = float(coef)
                if abs(coef) <= min_coef:
                    continue
                sign = '-' if coef < 0 else '+'
                body_term = _latex_term(term, module_symbols)
                magnitude = f'{abs(coef):.{precision}f}'
                rendered.append(f'{sign} {magnitude}' + (f' \\, {body_term}' if body_term else ''))

            rhs = ' '.join(rendered).lstrip('+ ').strip() or '0'
            lhs = rf'\Delta {module_symbols.get(module, module)}'
            body.append(f'  {lhs} &= {rhs}')

        lines.append(' \\\\\n'.join(body))
        lines.append(r'\end{align}')
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Analysis 1: embedding extremes + dynamics
# ---------------------------------------------------------------------------

def _embedding_matrix(model):
    """Ensemble-averaged participant embeddings, shape (P, D)."""
    n_participants = next(iter(model.sindy_coefficients.values())).shape[1]
    with torch.no_grad():
        embedding = model.participant_embedding(torch.arange(n_participants))
    embedding = embedding.detach().cpu().numpy()
    return embedding.mean(axis=0) if embedding.ndim == 3 else embedding


def analysis_embedding_dynamics(estimator, dataset, output_dir, results_dir,
                                label_map=None, module_symbols=None, n_select=3,
                                n_trials_show=100):
    """Pick the n_select participants furthest apart in embedding space and plot
    their choice / value-update dynamics."""
    print('=' * 70)
    print('ANALYSIS 1: embedding extremes -> choice & value-update dynamics')
    print('=' * 70)

    model = estimator.model
    participant_ids = select_distinctive_participants(
        model, n_select=n_select, metric='embedding')

    embedding = _embedding_matrix(model)
    distances = squareform(pdist(embedding, metric='euclidean'))
    block = distances[np.ix_(participant_ids, participant_ids)]

    print(f"Selected participants (max-min Euclidean embedding distance): {participant_ids}")
    if label_map is not None:
        for pid in participant_ids:
            print(f"  P{pid}: label={label_map.get(pid, 'n/a')}")
    median = np.median(distances[np.triu_indices_from(distances, 1)])
    print(f"Pairwise embedding distances (median over all pairs = {median:.3f}):")
    for i, pid_i in enumerate(participant_ids):
        for j, pid_j in enumerate(participant_ids):
            if j > i:
                print(f"  P{pid_i} <-> P{pid_j}: {block[i, j]:.3f}")

    labels = None
    if label_map is not None:
        labels = [f"{label_map.get(pid, '?')}" for pid in participant_ids]

    latex = format_equations_latex(estimator, participant_ids,
                                   participant_labels=labels,
                                   module_symbols=module_symbols)
    equations_path = os.path.join(results_dir, 'embedding_extremes_equations.txt')
    with open(equations_path, 'w') as handle:
        handle.write(latex)
    print(f"\nEquations (LaTeX) written to {equations_path}\n")
    print(latex)

    figure = _plot_panel_d_dynamics(
        dataset, model, participant_ids,
        session_idx=None, n_trials_show=n_trials_show, participant_labels=labels)
    save_panel(figure, output_dir, 'embedding_extremes_dynamics')

    pd.DataFrame({
        'rank': np.arange(len(participant_ids)),
        'participant_id': participant_ids,
        'label': [label_map.get(p) if label_map else None for p in participant_ids],
        'min_distance_to_others': [
            min(distances[p, q] for q in participant_ids if q != p) for p in participant_ids],
    }).to_csv(os.path.join(results_dir, 'embedding_extremes.csv'), index=False)

    return participant_ids


# ---------------------------------------------------------------------------
# Analysis 2: fingerprint vs. a behavioral metric
# ---------------------------------------------------------------------------

def analysis_metric_fingerprint(estimator, output_dir, results_dir, metrics_csv,
                                data_path, metric='avg_reward', label_map=None,
                                n_clusters=3, participant_col='participant'):
    """Structural + parametric coefficient differences ordered by *metric*."""
    print('\n' + '=' * 70)
    print(f'ANALYSIS 2: structural & parametric differences vs. {metric}')
    print('=' * 70)

    df_metrics = pd.read_csv(metrics_csv)
    coeff_df, presence_df, _ = _extract_equation_features(estimator)
    labels, _, _, _ = _cluster_participants(df_metrics, n_clusters)
    covariates = log_trials_per_participant(data_path, participant_col)

    sort_values = df_metrics[metric].values
    figure = _plot_panel_b(
        coeff_df, presence_df, labels,
        sort_values, sort_label=metric,
        diagnosis_per_participant=label_map,
        participant_ids=df_metrics['participant_id'].values,
        show_term_labels=True,
        selection='fdr',
        covariates=covariates,
    )
    save_panel(figure, output_dir, f'{metric}_fingerprint')

    stats = term_metric_effects(
        coeff_df, presence_df, sort_values, covariates=covariates).reset_index()
    out_csv = os.path.join(results_dir, f'{metric}_term_effects.csv')
    stats.to_csv(out_csv, index=False)

    print(f"Terms tested: {len(stats)}")
    print(f"  structural (presence ~ {metric}, FDR<0.05): "
          f"{int((stats['presence_p_fdr'] < 0.05).sum())}")
    print(f"  parametric (coefficient ~ {metric}, FDR<0.05): "
          f"{int((stats['magnitude_p_fdr'] < 0.05).sum())}")
    print("\nTop structural effects:")
    top = stats.dropna(subset=['presence_beta']).reindex(
        stats['presence_beta'].abs().sort_values(ascending=False).index).head(8)
    for _, row in top.iterrows():
        print(f"  {row['term']:<48s} beta={row['presence_beta']:+.3f}  "
              f"p_fdr={row['presence_p_fdr']:.4g}")
    print("\nTop parametric effects:")
    top = stats.dropna(subset=['magnitude_rho']).reindex(
        stats['magnitude_rho'].abs().sort_values(ascending=False).index).head(8)
    for _, row in top.iterrows():
        print(f"  {row['term']:<48s} rho={row['magnitude_rho']:+.3f}  "
              f"p_fdr={row['magnitude_p_fdr']:.4g}")
    print(f"\nSaved {out_csv}")

    return stats


# ---------------------------------------------------------------------------
# Analysis 3: beta effects by group
# ---------------------------------------------------------------------------

def analysis_group_betas(estimator, output_dir, results_dir, data_path,
                         criterion='diag', reference='Control'):
    """Logistic presence regression of every coefficient on a discrete group."""
    print('\n' + '=' * 70)
    print(f"ANALYSIS 3: beta effects by {criterion} (reference='{reference}')")
    print('=' * 70)

    analysis_coefficients_individuals(
        path_data=data_path,
        criterion=criterion,
        analysis='disc',
        reference=reference,
        spice_model=estimator,
        output_dir=results_dir,
    )

    # Drop near-constant terms, as in figure 4 panel d
    coeff_df, presence_df, _ = _extract_equation_features(estimator)
    excluded_terms = low_variance_terms(coeff_df, presence_df, verbose=True)

    beta_csv = os.path.join(results_dir, 'discrete_odds_ratio_results.csv')
    figure = _plot_panel_d(beta_csv, excluded_terms=excluded_terms)
    save_panel(figure, output_dir, f'{criterion}_beta_effects')

    return beta_csv


# ---------------------------------------------------------------------------
# Analysis 4: continuous beta effects vs. a behavioral metric
# ---------------------------------------------------------------------------

def _augmented_data_with_metric(data_path, metrics_csv, metric, out_path,
                                participant_col='participant'):
    """Copy the behavioural CSV with a per-participant metric column attached.

    The coefficient pipeline builds its criterion with
    `groupby(participant).first()`, so the criterion has to live in the data CSV
    as a participant-constant column. The metrics CSV keys participants by their
    0-based index, in the same order as `participant.unique()` -- the mapping
    csv_to_dataset uses. The extra column is inert for the dataset:
    csv_to_dataset only reads columns it is told about.
    """
    raw = pd.read_csv(data_path)
    metrics = pd.read_csv(metrics_csv)

    index_to_participant = {i: p for i, p in enumerate(raw[participant_col].unique())}
    metrics = metrics.assign(
        **{participant_col: metrics['participant_id'].map(index_to_participant)})
    missing = metrics[participant_col].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows in {metrics_csv} have no matching participant.")

    merged = raw.merge(metrics[[participant_col, metric]], on=participant_col, how='left')
    if merged[metric].isna().any():
        raise ValueError(f"'{metric}' is missing for some participants after the merge.")
    merged.to_csv(out_path, index=False)
    return out_path


def analysis_metric_betas(estimator, output_dir, results_dir, data_path,
                          metrics_csv, metric='avg_reward',
                          participant_col='participant'):
    """Continuous presence regression of every coefficient on *metric*.

    Unlike the group analysis this criterion is continuous, so it uses the
    continuous pipeline: logistic regression of term presence on the
    standardized metric, adjusted for log data volume, with profile-likelihood
    intervals and FDR across all tested coefficients.
    """
    print('\n' + '=' * 70)
    print(f"ANALYSIS 4: continuous beta effects vs. {metric}")
    print('=' * 70)

    metric_dir = os.path.join(results_dir, metric)
    os.makedirs(metric_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        augmented = _augmented_data_with_metric(
            data_path, metrics_csv, metric,
            os.path.join(tmp, 'data_with_metric.csv'), participant_col)
        analysis_coefficients_individuals(
            path_data=augmented,
            criterion=metric,
            analysis='cont',
            spice_model=estimator,
            output_dir=metric_dir,
        )

    print(f"\nResults written to {metric_dir}/")
    return metric_dir

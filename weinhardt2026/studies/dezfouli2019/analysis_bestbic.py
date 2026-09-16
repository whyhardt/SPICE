"""Deeper analyses on the lowest-BIC dezfouli2019 checkpoint from params_array/.

The params_array/ checkpoints are workingmemory.SpiceModel instances from the
(sindy_threshold_pruning x sindy_ensemble_pruning) training scan. This script
picks the checkpoint with the lowest BIC in results/hpscan_results.csv and runs
three analyses on it:

  1. Embedding extremes  — the three participants furthest apart in participant
     embedding space, with their choice / value-update dynamics
     (as in figures/figure2/panel_d_dynamics.png).
  2. Behavioural fingerprint — structural (presence) and parametric (magnitude)
     coefficient differences ordered by avg. reward across blocks
     (as in figures/figure4/panel_b_fingerprint.png).
  3. Beta effects — logistic presence regression of every coefficient on
     diagnosis group (as in figures/figure4/panel_d_beta_effects.pdf).

Usage:
    python -m weinhardt2026.studies.dezfouli2019.analysis_bestbic
    python -m weinhardt2026.studies.dezfouli2019.analysis_bestbic --params <path.pkl>
"""

import argparse
import os
import re
import tempfile

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform

from spice import SpiceEstimator, csv_to_dataset
from spice.precoded.workingmemory import SpiceModel, CONFIG

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


STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_ARRAY_DIR = os.path.join(STUDY_DIR, 'params_array')
RESULTS_DIR = os.path.join(STUDY_DIR, 'results')
DATA_PATH = os.path.join(STUDY_DIR, 'data', 'dezfouli2019.csv')
HPSCAN_CSV = os.path.join(RESULTS_DIR, 'hpscan_results.csv')
METRICS_CSV = os.path.join(RESULTS_DIR, 'behavioral_metrics_real.csv')

N_ACTIONS = 2
POLYNOMIAL_DEGREE = 2
N_TRIALS_SHOW = 100
N_STRUCTURAL_TERMS = 10
N_PARAMETRIC_TERMS = 10
DIAGNOSIS_REFERENCE = 'Control'


# ---------------------------------------------------------------------------
# Model selection / loading
# ---------------------------------------------------------------------------

def select_lowest_bic_checkpoint(hpscan_csv=HPSCAN_CSV, params_dir=PARAMS_ARRAY_DIR):
    """Return (path, row) of the params_array checkpoint with the lowest BIC."""
    df = pd.read_csv(hpscan_csv)
    row = df.loc[df['BIC'].idxmin()]
    path = os.path.join(params_dir, row['path'])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint referenced by {hpscan_csv} not found: {path}")
    return path, row


def load_estimator(path, n_actions=N_ACTIONS):
    """Load a params_array checkpoint into a SpiceEstimator."""
    ckpt = torch.load(path, map_location='cpu')
    first_module = next(iter(CONFIG.library_setup))
    ensemble_size, n_participants = ckpt['model'][f'sindy_coefficients.{first_module}'].shape[:2]
    del ckpt

    estimator = SpiceEstimator(
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        kwargs_spice_class={'reward_binary': True},
        sindy_library_polynomial_degree=POLYNOMIAL_DEGREE,
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(path)
    estimator.model.eval()
    return estimator


def load_diagnosis_map(data_path=DATA_PATH):
    """Map the 0-indexed participant index used by SPICE to its diagnosis label."""
    df_raw = pd.read_csv(data_path)
    id_map = {val: idx for idx, val in enumerate(df_raw['participant'].unique())}
    return {
        id_map[p]: df_raw[df_raw['participant'] == p]['diag'].iloc[0]
        for p in id_map
    }


# ---------------------------------------------------------------------------
# LaTeX rendering of the discovered equations
# ---------------------------------------------------------------------------

# workingmemory's four submodules, as LaTeX symbols
LATEX_MODULE_SYMBOLS = {
    'value_reward_chosen': r'V^{r}_{\mathrm{ch}}',
    'value_reward_not_chosen': r'V^{r}_{\mathrm{un}}',
    'value_choice_chosen': r'V^{c}_{\mathrm{ch}}',
    'value_choice_not_chosen': r'V^{c}_{\mathrm{un}}',
}


def _latex_factor(factor):
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
    if factor in LATEX_MODULE_SYMBOLS:
        base = LATEX_MODULE_SYMBOLS[factor]
        return f'{base}(t){power}' if power else f'{base}(t)'

    return f'{factor}{power}'


def _latex_term(term):
    """Render a library term (a product of factors); '1' is the constant."""
    if term == '1':
        return ''
    return r' \cdot '.join(_latex_factor(f) for f in term.split('*'))


def format_equations_latex(estimator, participant_ids, participant_labels=None,
                           min_coef=0.0, precision=3):
    """LaTeX for each participant's discovered equation set.

    The SINDy library predicts the per-trial increment -- the model integrates
    `h_next = h + dt * (library @ coefficients)` with dt=1 -- so the left-hand
    side is the change in the state, and coefficients are printed exactly as
    stored (no identity term is folded in).
    """
    model = estimator.model
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
                body_term = _latex_term(term)
                magnitude = f'{abs(coef):.{precision}f}'
                rendered.append(f'{sign} {magnitude}' + (f' \\, {body_term}' if body_term else ''))

            rhs = ' '.join(rendered).lstrip('+ ').strip() or '0'
            lhs = rf'\Delta {LATEX_MODULE_SYMBOLS.get(module, module)}'
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
        emb = model.participant_embedding(torch.arange(n_participants))
    emb = emb.detach().cpu().numpy()
    return emb.mean(axis=0) if emb.ndim == 3 else emb


def analysis_embedding_dynamics(estimator, dataset, output_dir, results_dir,
                                diagnosis_map=None, n_select=3,
                                n_trials_show=N_TRIALS_SHOW):
    """Pick the n_select participants furthest apart in embedding space and plot
    their choice / value-update dynamics."""
    print('=' * 70)
    print('ANALYSIS 1: embedding extremes -> choice & value-update dynamics')
    print('=' * 70)

    model = estimator.model
    participant_ids = select_distinctive_participants(
        model, n_select=n_select, metric='embedding')

    emb = _embedding_matrix(model)
    dist = squareform(pdist(emb, metric='euclidean'))
    sub = dist[np.ix_(participant_ids, participant_ids)]

    print(f"Selected participants (max-min Euclidean embedding distance): {participant_ids}")
    if diagnosis_map is not None:
        for pid in participant_ids:
            print(f"  P{pid}: diagnosis={diagnosis_map.get(pid, 'n/a')}")
    print(f"Pairwise embedding distances (median over all pairs = {np.median(dist[np.triu_indices_from(dist, 1)]):.3f}):")
    for i, pid_i in enumerate(participant_ids):
        for j, pid_j in enumerate(participant_ids):
            if j > i:
                print(f"  P{pid_i} <-> P{pid_j}: {sub[i, j]:.3f}")

    labels = None  # per-participant title prefix
    if diagnosis_map is not None:
        labels = [f"{diagnosis_map.get(pid, '?')}" for pid in participant_ids]

    latex = format_equations_latex(estimator, participant_ids, participant_labels=labels)
    eq_path = os.path.join(results_dir, 'embedding_extremes_equations.txt')
    with open(eq_path, 'w') as f:
        f.write(latex)
    print(f"\nEquations (LaTeX) written to {eq_path}\n")
    print(latex)

    fig_dyn = _plot_panel_d_dynamics(
        dataset, model, participant_ids,
        session_idx=None, n_trials_show=n_trials_show, participant_labels=labels)
    save_panel(fig_dyn, output_dir, 'embedding_extremes_dynamics')

    pd.DataFrame({
        'rank': np.arange(len(participant_ids)),
        'participant_id': participant_ids,
        'diagnosis': [diagnosis_map.get(p) if diagnosis_map else None
                      for p in participant_ids],
        'min_distance_to_others': [
            min(dist[p, q] for q in participant_ids if q != p) for p in participant_ids],
    }).to_csv(os.path.join(results_dir, 'embedding_extremes.csv'), index=False)

    return participant_ids


# ---------------------------------------------------------------------------
# Analysis 2: fingerprint vs. average reward
# ---------------------------------------------------------------------------

def _log_trials_per_participant(data_path=DATA_PATH):
    """log trial count per participant, indexed 0..P-1 like the coefficient frame.

    The coefficient pipeline adjusts presence regressions for this: sparsity is
    decided by data-driven pruning, so a participant with more trials supports
    more surviving terms. The fingerprint has to adjust for it identically or
    its structural set will not match the forest plot's.
    """
    raw = pd.read_csv(data_path)
    counts = raw.groupby('participant').size()
    order = list(raw['participant'].unique())
    return np.log(np.array([counts[pid] for pid in order], dtype=float))

def analysis_reward_fingerprint(estimator, output_dir, results_dir,
                                metrics_csv=METRICS_CSV, diagnosis_map=None,
                                n_clusters=3):
    """Structural + parametric coefficient differences ordered by avg reward."""
    print('\n' + '=' * 70)
    print('ANALYSIS 2: structural & parametric differences vs. avg reward')
    print('=' * 70)

    df_metrics = pd.read_csv(metrics_csv)
    coeff_df, presence_df, term_names = _extract_equation_features(estimator)
    labels, _, _, _ = _cluster_participants(df_metrics, n_clusters)

    sort_values = df_metrics['avg_reward'].values
    fig_b = _plot_panel_b(
        coeff_df, presence_df, labels,
        sort_values, sort_label='Avg Reward',
        diagnosis_per_participant=diagnosis_map,
        participant_ids=df_metrics['participant_id'].values,
        show_term_labels=True,
        selection='fdr',
        covariates=_log_trials_per_participant(),
    )
    save_panel(fig_b, output_dir, 'reward_fingerprint')

    stats_df = term_metric_effects(
        coeff_df, presence_df, sort_values,
        covariates=_log_trials_per_participant()).reset_index()
    out_csv = os.path.join(results_dir, 'reward_term_effects.csv')
    stats_df.to_csv(out_csv, index=False)

    n_struct = int((stats_df['presence_p_fdr'] < 0.05).sum())
    n_param = int((stats_df['magnitude_p_fdr'] < 0.05).sum())
    print(f"Terms tested: {len(stats_df)}")
    print(f"  structural (presence ~ avg_reward, FDR<0.05): {n_struct}")
    print(f"  parametric (coefficient ~ avg_reward, FDR<0.05): {n_param}")
    print("\nTop structural effects:")
    top = stats_df.dropna(subset=['presence_beta']).reindex(
        stats_df['presence_beta'].abs().sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        print(f"  {r['term']:<48s} beta={r['presence_beta']:+.3f}  "
              f"p_fdr={r['presence_p_fdr']:.4g}")
    print("\nTop parametric effects:")
    top = stats_df.dropna(subset=['magnitude_rho']).reindex(
        stats_df['magnitude_rho'].abs().sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        print(f"  {r['term']:<48s} rho={r['magnitude_rho']:+.3f}  "
              f"p_fdr={r['magnitude_p_fdr']:.4g}")
    print(f"\nSaved {out_csv}")

    return stats_df


# ---------------------------------------------------------------------------
# Analysis 3: beta effects by diagnosis
# ---------------------------------------------------------------------------

def analysis_diagnosis_betas(estimator, output_dir, results_dir,
                             data_path=DATA_PATH, reference=DIAGNOSIS_REFERENCE):
    """Logistic presence regression of every coefficient on diagnosis group."""
    print('\n' + '=' * 70)
    print(f"ANALYSIS 3: beta effects by diagnosis (reference='{reference}')")
    print('=' * 70)

    analysis_coefficients_individuals(
        path_data=data_path,
        criterion='diag',
        analysis='disc',
        reference=reference,
        spice_model=estimator,
        output_dir=results_dir,
    )

    # Drop near-constant terms, as in figure 4 panel d
    coeff_df, presence_df, _ = _extract_equation_features(estimator)
    excluded_terms = low_variance_terms(coeff_df, presence_df, verbose=True)

    beta_csv = os.path.join(results_dir, 'discrete_odds_ratio_results.csv')
    fig_d = _plot_panel_d(beta_csv, excluded_terms=excluded_terms)
    save_panel(fig_d, output_dir, 'diagnosis_beta_effects')

    return beta_csv


# ---------------------------------------------------------------------------
# Analysis 4: continuous beta effects vs. average reward
# ---------------------------------------------------------------------------

def _augmented_data_with_metric(data_path, metrics_csv, metric, out_path):
    """Copy the behavioural CSV with a per-participant metric column attached.

    `prepare` builds its criterion with `groupby(participant).first()`, so the
    criterion has to live in the data CSV as a participant-constant column.
    `behavioral_metrics_real.csv` keys participants by their 0-based index, in
    the same order as `participant.unique()` -- the mapping csv_to_dataset uses.
    The extra column is inert for the dataset: csv_to_dataset only reads columns
    it is told about, and `additional_inputs` defaults to None.
    """
    raw = pd.read_csv(data_path)
    metrics = pd.read_csv(metrics_csv)

    index_to_pid = {i: pid for i, pid in enumerate(raw['participant'].unique())}
    metrics = metrics.assign(participant=metrics['participant_id'].map(index_to_pid))
    missing = metrics['participant'].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows in {metrics_csv} have no matching participant.")

    merged = raw.merge(metrics[['participant', metric]], on='participant', how='left')
    if merged[metric].isna().any():
        raise ValueError(f"'{metric}' is missing for some participants after the merge.")
    merged.to_csv(out_path, index=False)
    return out_path


def analysis_reward_betas(estimator, output_dir, results_dir,
                          data_path=DATA_PATH, metrics_csv=METRICS_CSV,
                          metric='avg_reward'):
    """Continuous presence regression of every coefficient on average reward.

    Unlike the diagnosis analysis this criterion is continuous, so it uses the
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
            data_path, metrics_csv, metric, os.path.join(tmp, 'data_with_metric.csv'))
        analysis_coefficients_individuals(
            path_data=augmented,
            criterion=metric,
            analysis='cont',
            spice_model=estimator,
            output_dir=metric_dir,
        )

    print(f"\nResults written to {metric_dir}/")
    return metric_dir

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params', default=None,
                        help='Checkpoint path (default: lowest BIC in hpscan_results.csv)')
    parser.add_argument('--output-dir', default=os.path.join(STUDY_DIR, 'figures', 'bestbic'))
    parser.add_argument('--results-dir', default=os.path.join(RESULTS_DIR, 'bestbic'))
    parser.add_argument('--n-select', type=int, default=3)
    parser.add_argument('--analyses', nargs='+', default=['1', '2', '3', '4'],
                        help='Which analyses to run (1=dynamics, 2=fingerprint, '
                             '3=diagnosis betas, 4=avg-reward betas)')
    args = parser.parse_args()

    if args.params is None:
        params_path, row = select_lowest_bic_checkpoint()
        print(f"Lowest-BIC checkpoint: {os.path.basename(params_path)}")
        print(f"  sindy_threshold_pruning={row['threshold']}, "
              f"sindy_ensemble_pruning={row['test']}")
        summary = [f"BIC={row['BIC']:.1f}",
                   f"n_params={row['n_params_mean']:.1f}",
                   f"trial likelihood={row['trial_likelihood']:.4f}"]
        # Hold-out reporting changed shape across versions of the scan
        for col, label in (('trial_likelihood_test', 'test trial likelihood'),
                           ('generalization_gap', 'gen. gap')):
            if col in row.index and pd.notna(row[col]):
                summary.append(f"{label}={row[col]:.4f}")
        print('  ' + ', '.join(summary) + '\n')
    else:
        params_path = args.params
        print(f"Checkpoint (explicit): {params_path}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    estimator = load_estimator(params_path)
    diagnosis_map = load_diagnosis_map()

    if '1' in args.analyses:
        dataset = csv_to_dataset(file=DATA_PATH)
        dataset.normalize_rewards()
        analysis_embedding_dynamics(
            estimator, dataset, args.output_dir, args.results_dir,
            diagnosis_map=diagnosis_map, n_select=args.n_select)

    if '2' in args.analyses:
        analysis_reward_fingerprint(
            estimator, args.output_dir, args.results_dir, diagnosis_map=diagnosis_map)

    if '3' in args.analyses:
        analysis_diagnosis_betas(estimator, args.output_dir, args.results_dir)

    if '4' in args.analyses:
        analysis_reward_betas(estimator, args.output_dir, args.results_dir)

    print(f"\nFigures -> {args.output_dir}/")
    print(f"Results -> {args.results_dir}/")


if __name__ == '__main__':
    main()

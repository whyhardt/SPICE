"""
Model evaluation figure: one column per study, rows = held-out prediction, posterior predictive check, AIC, BIC.

Within a row, the columns share the y-axis span (so equal differences look equal), wherever their
values share a unit: all discrete studies, and the continuous studies' AIC/BIC (see SHARED_SPAN).

Models per study, every seed of the final models in ``studies/<study>/params/``
(see ``weinhardt2026/utils/final_models.py``; bars = mean over seeds, dots = single seeds):
  - Benchmark: the study's hand-crafted cognitive model
  - GRU:       a GRU without participant embedding (hidden size read from the checkpoint)
  - SPICE-RNN / SPICE-EQ: the study's SPICE grid-search checkpoints

Data and SPICE estimators are rebuilt exactly as ``slurm_jobs/grid_search.sh`` →
``weinhardt2026/run.py`` built them, so all models are scored on the split the
SPICE checkpoints were trained on.

Metrics:
  - Held-out prediction (test blocks):
      discrete   → average trial likelihood exp(-NLL / n_trials)
      continuous → one-step MSE per trial in the response's units (bruckner2025: px², weber2024: deg²)
  - Posterior predictive check (all blocks; computed and saved per study by
    ``weinhardt2026/analysis/evaluation/run_posterior_predictive.py``, read here per seed):
      discrete   → nRMSE of the summary statistics across participants (mean over panels)
      continuous → per-trial MSE of an autonomous rollout (bruckner2025: px², weber2024: deg²)
  - AIC / BIC per trial (training data: per (participant, experiment) group divided by the group's trials, then averaged over groups, see
    ``model_evaluation.grouped_information_criteria``) — only for the
    interpretable models (Benchmark, SPICE-EQ). Continuous studies use a Gaussian
    likelihood with per-group MLE variance.

Written to weinhardt2026/results/model_evaluation/ (figure, model_evaluation_summary.csv: mean and SD over seeds);
per study, the per-seed evaluation is cached in studies/<study>/results/model_evaluation/model_evaluation.csv.

Usage:
    python weinhardt2026/analysis/evaluation/run_model_evaluation.py            # evaluate (cached) + plot
    python weinhardt2026/analysis/evaluation/run_model_evaluation.py --recompute
"""

import argparse
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from weinhardt2026.analysis.evaluation.model_evaluation import (
    get_participant_experiment_groups,
    grouped_information_criteria,
)
from weinhardt2026.analysis.evaluation.run_posterior_predictive import (
    ROLLOUT_UNITS as RESPONSE_UNITS,
    load_scores as load_ppc_scores,
)
from weinhardt2026.utils.final_models import checkpoint_seed, final_checkpoints
from weinhardt2026.utils.studies import (
    ALL_STUDIES, GROUP_STUDIES, LABELS, STUDIES, continuous_session_sse, discrete_session_nll, load_spice, load_study,
    n_benchmark_parameters, predict, response_session_sse, study_path,
)

OUTPUT_DIR = os.path.join(ROOT, 'weinhardt2026', 'results', 'model_evaluation')


def study_evaluation_path(study):
    """Per-study cache of evaluate_study (one row per model and seed)."""
    return study_path(study, 'results', 'model_evaluation', 'model_evaluation.csv')


# ── Metrics ─────────────────────────────────────────────────────────

def evaluate_discrete(outputs_train, outputs_test, dataset_train, dataset_test, k_per_group):
    nll_test, n_test = discrete_session_nll(outputs_test, dataset_test)
    metrics = {'heldout': math.exp(-nll_test.sum().item() / n_test.sum().item())}
    if k_per_group is not None:
        nll_train, n_train = discrete_session_nll(outputs_train, dataset_train)
        pairs, group_index = get_participant_experiment_groups(dataset_train)
        info = grouped_information_criteria(
            nll_per_session=nll_train, n_trials_per_session=n_train, group_index=group_index,
            n_groups=pairs.shape[0], n_parameters_per_group=k_per_group(pairs),
            n_actions_baseline=dataset_train.n_actions,
        )
        metrics.update(aic=info['aic_per_trial_mean'], bic=info['bic_per_trial_mean'])
    return metrics


def evaluate_continuous(outputs_train, outputs_test, dataset_train, dataset_test, k_per_group, study):
    sse_train, n_train, dim = continuous_session_sse(outputs_train, dataset_train, study)

    # held-out one-step MSE per trial in the response's units
    sse_test, n_test = response_session_sse(outputs_test, dataset_test, study)
    metrics = {'heldout': sse_test.sum().item() / n_test.sum().item()}

    if k_per_group is not None:
        pairs, group_index = get_participant_experiment_groups(dataset_train)
        n_groups = pairs.shape[0]
        sse_group = torch.zeros(n_groups, dtype=sse_train.dtype).scatter_add_(0, group_index, sse_train)
        n_group = torch.zeros(n_groups).scatter_add_(0, group_index, n_train)
        keep = n_group > 0
        sse_group, n_group, k = sse_group[keep], n_group[keep], k_per_group(pairs).float().cpu()[keep]
        # per-group Gaussian with MLE variance: 2*NLL = n*d*(1 + log(2*pi*mse))
        two_nll = n_group * dim * (1 + torch.log(2 * math.pi * sse_group / (n_group * dim)))
        metrics.update(
            aic=((two_nll + 2 * k) / n_group).mean().item(),
            bic=((two_nll + k * torch.log(n_group)) / n_group).mean().item(),
        )
    return metrics


def evaluate_study(study):
    config = STUDIES[study]
    spice_module, dataset_train, dataset_test = load_study(study)
    continuous = bool(dataset_train.continuous_action)
    n_participants = len(dataset_train.xs[..., -1].unique())
    n_actions = dataset_train.n_actions

    def evaluate(outputs_train, outputs_test, k_per_group):
        if continuous:
            return evaluate_continuous(outputs_train, outputs_test, dataset_train, dataset_test, k_per_group, study)
        return evaluate_discrete(outputs_train, outputs_test, dataset_train, dataset_test, k_per_group)

    rows = []

    for path in final_checkpoints(study, 'benchmark'):
        benchmark = config['benchmark'](path, n_participants, n_actions)
        k_benchmark = n_benchmark_parameters(benchmark, n_participants)
        metrics = evaluate(predict(benchmark, dataset_train), predict(benchmark, dataset_test),
                           lambda pairs: torch.full((pairs.shape[0],), k_benchmark))
        rows.append(dict(model='Benchmark', seed=checkpoint_seed(path), n_parameters=k_benchmark, **metrics))

    for path in final_checkpoints(study, 'gru'):
        gru = config['gru'](path, n_participants, n_actions)
        assert gru.participant_embedding_size == 0, f'{study}: GRU must not use a participant embedding'
        metrics = evaluate(predict(gru, dataset_train), predict(gru, dataset_test), None)
        rows.append(dict(model='GRU', seed=checkpoint_seed(path), **metrics))

    for path in final_checkpoints(study, 'spice'):
        seed = checkpoint_seed(path)
        print(f'  {os.path.basename(path)}', flush=True)
        estimator = load_spice(path, spice_module, dataset_train)
        coefficient_counts = estimator.count_sindy_coefficients().cpu()

        metrics = evaluate(predict(estimator, dataset_train, use_sindy=False),
                           predict(estimator, dataset_test, use_sindy=False), None)
        rows.append(dict(model='SPICE-RNN', seed=seed, **metrics))

        # a model with one experiment slot (weber2024: conditions are module inputs) shares it across experiments
        shared_experiment = coefficient_counts.shape[1] == 1
        k_spice = lambda pairs: coefficient_counts[pairs[:, 0], 0 if shared_experiment else pairs[:, 1]].float()
        metrics = evaluate(predict(estimator, dataset_train, use_sindy=True),
                           predict(estimator, dataset_test, use_sindy=True), k_spice)
        pairs, _ = get_participant_experiment_groups(dataset_train)
        rows.append(dict(model='SPICE-EQ', seed=seed, n_parameters=k_spice(pairs).mean().item(), **metrics))

        del estimator
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.insert(0, 'study', study)
    df['continuous'] = continuous
    return df


def evaluate_group_study(study):
    """kolff2025: group models (one parameter set for all apes) scored only where the ape acts next, on the
    held-out dyads of fold 0; AIC/BIC per trial on the pooled training data, since the parameters are shared.
    The benchmark (logistic regression) is fit in closed form, so it has a single fit (seed 0)."""
    from weinhardt2026.studies.kolff2025.simulation_kolff2025 import load_spice as load_group_spice
    from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import (
        LogisticRegressionModel, ModelInputGRU, evaluation_dataset, get_dataset,
    )
    train, test, _ = get_dataset(test_fold=0)
    dataset_train, dataset_test = evaluation_dataset(train), evaluation_dataset(test)

    def evaluate(model, k=None, use_sindy=None):
        nll_test, n_test = discrete_session_nll(predict(model, dataset_test, use_sindy), dataset_test)
        metrics = {'heldout': math.exp(-nll_test.sum().item() / n_test.sum().item())}
        if k is not None:
            nll_train, n_train = discrete_session_nll(predict(model, dataset_train, use_sindy), dataset_train)
            two_nll, n = 2 * nll_train.sum().item(), n_train.sum().item()
            metrics.update(aic=(two_nll + 2 * k) / n, bic=(two_nll + k * math.log(n)) / n, n_parameters=k)
        return metrics

    benchmark = LogisticRegressionModel().fit(train)
    rows = [dict(model='Benchmark', seed=0, **evaluate(benchmark, k=float(benchmark.count_parameters())))]
    for path in final_checkpoints(study, 'gru'):
        rows.append(dict(model='GRU', seed=checkpoint_seed(path), **evaluate(load_gru(path, ModelInputGRU))))
    for path in final_checkpoints(study, 'spice'):
        print(f'  {os.path.basename(path)}', flush=True)
        estimator = load_group_spice(path)
        k_spice = estimator.count_sindy_coefficients().cpu()[0, 0].float().item()
        rows.append(dict(model='SPICE-RNN', seed=checkpoint_seed(path), **evaluate(estimator, use_sindy=False)))
        rows.append(dict(model='SPICE-EQ', seed=checkpoint_seed(path), **evaluate(estimator, k=k_spice, use_sindy=True)))
        del estimator
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.insert(0, 'study', study)
    df['continuous'] = False
    return df


# ── Plot ────────────────────────────────────────────────────────────

COLORS = {
    'Benchmark': '#1a3a5c',
    'GRU':       '#5b8db8',
    'SPICE-RNN': '#c44e52',
    'SPICE-EQ':  '#8b2529',
}
MODEL_ORDER = ['GRU', 'Benchmark', 'SPICE-EQ', 'SPICE-RNN']
ROWS = [
    ('heldout', None),  # ylabel depends on discrete/continuous
    ('ppc', None),      # ylabel depends on discrete/continuous
    ('aic', 'AIC / trial (train) ↓'),
    ('bic', 'BIC / trial (train) ↓'),
]
HELDOUT_LABEL = {False: 'Held-out trial likelihood ↑', True: 'Held-out one-step MSE ↓'}
PPC_LABEL = {False: 'PPC nRMSE ↓', True: 'Rollout MSE ↓'}
# (metric, continuous) -> whether the columns share the y-axis span; continuous MSEs differ in unit (px², deg²)
SHARED_SPAN = {(metric, continuous): not continuous or metric in ('aic', 'bic')
               for metric, _ in ROWS for continuous in (False, True)}
KIND_LABELS = {'heldout': HELDOUT_LABEL, 'ppc': PPC_LABEL}


def add_posterior_predictive(results: pd.DataFrame) -> pd.DataFrame:
    """Column 'ppc': the study's posterior predictive score of every (model, seed), NaN if not computed."""
    frames = []
    for study, study_results in results.groupby('study', sort=False):
        ppc = load_ppc_scores(study)
        frames.append(study_results.drop(columns='ppc', errors='ignore').merge(ppc, on=['model', 'seed'], how='left'))
    return pd.concat(frames, ignore_index=True)


def plot_model_evaluation(results: pd.DataFrame, output_path: str, clean: bool = False):
    """Bars = mean over seeds (± SD if several seeds); white dots = single seeds.

    The y-axes are zoomed onto the model differences, so bars rise from the bottom
    of each axis rather than from 0 (the continuous studies' AIC/BIC are negative). Where
    SHARED_SPAN holds, the columns of a row (of one kind) span the same y-range: the largest
    range among them. With *clean*, also saved without titles, axis labels, units and legend
    (<output_path>_clean).
    """
    present = set(results['study'])
    continuous = results.groupby('study')['continuous'].first().astype(bool)
    studies = [s for s in ALL_STUDIES if s in present and not continuous[s]] \
        + [s for s in ALL_STUDIES if s in present and continuous[s]]
    n_rows, n_cols = len(ROWS), len(studies)
    positions = {model: index for index, model in enumerate(MODEL_ORDER)}

    # values per (row, col) and the shared range per (metric, kind)
    panel_values, shared_range = {}, {}
    for col, study in enumerate(studies):
        study_results = results[results['study'] == study]
        for row, (metric, _) in enumerate(ROWS):
            model_values = {}
            for model in MODEL_ORDER:
                values = study_results.loc[study_results['model'] == model, metric].dropna().to_numpy()
                if values.size > 0:
                    model_values[model] = values
            panel_values[row, col] = model_values
            group = (metric, bool(continuous[study]))
            if model_values and SHARED_SPAN[group]:
                all_values = np.concatenate(list(model_values.values()))
                shared_range[group] = max(shared_range.get(group, 0.), np.ptp(all_values))

    for annotated in ((True, False) if clean else (True,)):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols + 0.8, 2.2 * n_rows), squeeze=False)
        rng = np.random.default_rng(0)   # the same seed jitter in both versions
        for col, study in enumerate(studies):
            is_continuous = bool(continuous[study])
            first_of_kind = col == 0 or is_continuous != bool(continuous[studies[col - 1]])
            for row, (metric, ylabel) in enumerate(ROWS):
                ax = axes[row, col]
                model_values = panel_values[row, col]
                if not model_values:
                    ax.axis('off')
                    continue

                all_values = np.concatenate(list(model_values.values()))
                low = all_values.min()
                value_range = shared_range.get((metric, is_continuous), np.ptp(all_values))
                margin = 0.15 * value_range if value_range > 0 else 0.05 * abs(low) + 1e-3
                floor = low - 2 * margin  # keeps the lowest bar visible
                ax.set_ylim(floor, low + value_range + margin)

                for model, values in model_values.items():
                    x, mean = positions[model], values.mean()
                    ax.bar(x, mean - floor, bottom=floor, width=0.75, color=COLORS[model],
                           edgecolor='white', linewidth=0.5, label=model, zorder=2)
                    if values.size > 1:
                        ax.errorbar(x, mean, yerr=values.std(), color='#222222', capsize=2.5, linewidth=1, zorder=4)
                        jitter = rng.uniform(-0.18, 0.18, values.size)
                        ax.scatter(x + jitter, values, s=8, color='white', edgecolor='#222222', linewidth=0.5, zorder=3)

                ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
                ax.set_xticks([])
                ax.grid(axis='y', color='#dddddd', linewidth=0.5, zorder=0)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
                ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
                ax.tick_params(axis='y', labelsize=8)
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(False)

                if not annotated:
                    continue
                if row == 0:
                    ax.set_title(LABELS[study], fontsize=9)
                if metric in KIND_LABELS:
                    if first_of_kind:
                        ax.set_ylabel(KIND_LABELS[metric][is_continuous], fontsize=8.5)
                    if is_continuous:
                        ax.text(0.02, 0.98, RESPONSE_UNITS[study], transform=ax.transAxes, ha='left', va='top',
                                fontsize=7, color='#555555')
                elif col == 0:
                    ax.set_ylabel(ylabel, fontsize=8.5)

        if annotated:
            handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODEL_ORDER]
            fig.legend(handles, MODEL_ORDER, loc='upper center', ncol=len(MODEL_ORDER), fontsize=9,
                       frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = output_path if annotated else f'{output_path}_clean'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(f'{path}.pdf', bbox_inches='tight')
        fig.savefig(f'{path}.png', bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f'Saved {path}.pdf/.png')


# ── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--studies', nargs='+', default=ALL_STUDIES)
    parser.add_argument('--recompute', action='store_true', help='Ignore cached per-study CSVs')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []
    for study in args.studies:
        cache = study_evaluation_path(study)
        if os.path.isfile(cache) and not args.recompute:
            frames.append(pd.read_csv(cache))
            continue
        print(f'Evaluating {study}...', flush=True)
        df = evaluate_group_study(study) if study in GROUP_STUDIES else evaluate_study(study)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)
        frames.append(df)

    results = add_posterior_predictive(pd.concat(frames, ignore_index=True))
    summary = results.groupby(['study', 'model'], sort=False)[['heldout', 'ppc', 'aic', 'bic', 'n_parameters']].agg(['mean', 'std'])
    print(summary.to_string(float_format='{:.4f}'.format))
    summary.to_csv(os.path.join(OUTPUT_DIR, 'model_evaluation_summary.csv'))
    plot_model_evaluation(results, os.path.join(OUTPUT_DIR, 'figure_model_evaluation'), clean=True)

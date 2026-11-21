"""
SPICE -> SPICE parameter recovery: which terms of a known SPICE model are found again by refitting SPICE to its data.

Uses the fits of the model identification (analysis/validation/model_identification/):
    ground truth   the SPICE generator that simulated the dataset: the ensemble member of data/generators.csv
    recovered      the refits to that dataset (runs/<study>/spice_<study>_simspice_rs*.pkl, 10 seeds). Within a seed a
                   participant carries a term if the majority of the ensemble members do; across seeds if >= 6 seeds do.
Presence comes from the models' presence masks, without a coefficient cutoff (as in run_parameter_recovery.py).

Terms are signed: `term (+)` and `term (-)` are different terms, so a match needs the same sign.
    precision  share of the recovered participant x term cells that the generator has
    recall     share of the generator's participant x term cells that are recovered
Effect size of a term = |coefficient| x RMS of the term's library value along the model's own SPICE-EQ trajectories on the
simulated training data (valid trials, updated items only). Prevalence = share of participants carrying the term.
Precision is binned by the recovered model's effect size and prevalence (what a term of the fitted model is worth, the
only side a user sees), recall by the generator's. Effect-size bins are percentiles within a study; prevalence bins are fixed. Studies with one group model have no prevalence x effect analysis.

Per study, in weinhardt2026/studies/<study>/results/parameter_recovery/, and across studies, in
weinhardt2026/results/parameter_recovery/spice_recovery/ (figures also as _clean):
    recovery_overall.csv                  precision, recall per study
    recovery_by_effect_size.csv           figure_recovery_by_effect_size             10 percentile bins (y), studies (x)
    recovery_by_prevalence.csv            figure_recovery_by_prevalence              10% bins (x), studies (y)
    recovery_by_prevalence_effect.csv     figure_recovery_by_prevalence_effect       20% bins (x) x effect quintiles (y)
    prevalence_effect_regression.csv      odds ratios per SD of effect percentile and prevalence, SEs clustered by term
    prevalence_effect_correlation.csv     Spearman correlation of prevalence and effect size over terms
Across studies only (each study weighs equally; a study counts in a cell with >= 3 terms, a cell needs >= 3 studies):
    recovery_by_prevalence_effect_across_studies.csv
    figure_prevalence_effect_across_studies       colour: mean; text: mean +- SD; dots: individual studies
    figure_prevalence_effect_mean_sd              mean (first row) and SD (second row)
The models' coefficients and effect sizes are cached in the study folder (models.pkl).

Usage:
    python weinhardt2026/analysis/validation/run_spice_recovery.py
    python weinhardt2026/analysis/validation/run_spice_recovery.py --studies dezfouli2019 --recompute
"""

import argparse
import os
import pickle
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules import 'studies.<study>...'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from spice.resources.sindy_differentiable import compute_polynomial_library
from spice.utils.analysis.coefficient_dendrogram import get_module_matrices
from spice.utils.plotting import EMPTY_COLOR, rate_cmap, text_color
from weinhardt2026.analysis.validation.model_identification.fit_confusion_matrix import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR as DEFAULT_RUNS_DIR
from weinhardt2026.analysis.validation.model_identification.run_model_identification import fit_paths, simulated_splits
from weinhardt2026.utils.final_models import model_setup
from weinhardt2026.utils.studies import ALL_STUDIES, GROUP_STUDIES, LABELS, STUDIES, study_path

VOTES = 6
N_EFFECT_BINS = 10
N_PREVALENCE_BINS = 10
N_GRID_BINS = 5
MIN_TERMS_PER_CELL = 3
MIN_STUDIES_PER_CELL = 3
OUTPUT_ACROSS = os.path.join(ROOT, 'weinhardt2026', 'results', 'parameter_recovery', 'spice_recovery')
METRICS = ('precision', 'recall')
TITLES = {'precision': 'Precision', 'recall': 'Recall'}
GRID_EFFECT_LABELS = ['Q1\nsmallest', 'Q2', 'Q3', 'Q4', 'Q5\nlargest']


def ordered_studies():
    """As in the model evaluation figure: discrete studies first, then continuous ones."""
    continuous = {s for s in STUDIES if STUDIES[s]['data_kwargs'].get('continuous_action')}
    return [s for s in ALL_STUDIES if s not in continuous] + [s for s in ALL_STUDIES if s in continuous]


def percent_labels(n_bins):
    return [f'{round(100 * i / n_bins)}–{round(100 * (i + 1) / n_bins)}%' for i in range(n_bins)]


# ── Models ──────────────────────────────────────────────────────────

class TrackedTrials:
    """The trial indices of a forward pass; records the current trial while the model iterates over them."""

    def __init__(self, trials, context):
        self.trials, self.context = trials, context

    def __iter__(self):
        for trial in self.trials:
            self.context['trial'] = int(trial)
            yield trial

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        return self.trials[index]


@torch.no_grad()
def term_library_rms(model, inputs):
    """Per module: RMS of each term's library value along the model's own SINDy trajectories on `inputs` (valid trials,
    updated items only), (ensemble, participants, terms). Works on any BaseModel, e.g. a ground-truth QLearning."""
    model.eval(use_sindy=True)
    inputs = inputs.to(next(iter(model.sindy_coefficients.values())).device)
    valid_trials = ~inputs[:, :, :, 0].isnan().all(dim=2)  # (B, T)
    n_members, n_participants = model.ensemble_size, model.n_participants
    squares, counts, context = {}, {}, {'trial': 0, 'mask': None}
    init_forward_pass, call_module, forward_sindy = model.init_forward_pass, model.call_module, model.forward_sindy

    def tracked_init_forward_pass(inputs, state):
        signals = init_forward_pass(inputs, state)
        signals.trials = TrackedTrials(signals.trials, context)
        return signals

    def tracked_call_module(*args, **kwargs):
        context['mask'] = kwargs.get('action_mask', args[2] if len(args) > 2 else None)
        return call_module(*args, **kwargs)

    def tracked_forward_sindy(h_current, key_module, participant_ids, experiment_ids, controls, polynomial_degree):
        n_within, _, n_batch, n_items = h_current.shape
        library = compute_polynomial_library(
            h_current.reshape(n_within, n_members * n_batch, n_items),
            controls.reshape(n_within, n_members * n_batch, n_items, -1),
            degree=polynomial_degree,
            feature_names=model.sindy_specs[key_module]['input_names'],
            library=model.sindy_candidate_terms[key_module],
        ).reshape(n_members, n_batch, n_items, -1)
        mask = context['mask']
        if mask is None:
            mask = torch.ones(n_members, n_batch, n_items, device=library.device)
        elif mask.dim() >= 4:
            mask = mask[-1]
        weight = (mask == 1).float() * valid_trials[:, context['trial']].view(1, n_batch, 1).float()
        if key_module not in squares:
            squares[key_module] = torch.zeros(n_members, n_participants, library.shape[-1], device=library.device)
            counts[key_module] = torch.zeros(n_members, n_participants, device=library.device)
        square = (library ** 2 * weight.unsqueeze(-1)).sum(dim=2)  # (E, B, terms)
        for member in range(n_members):
            squares[key_module][member].index_add_(0, participant_ids[member], square[member])
            counts[key_module][member].index_add_(0, participant_ids[member], weight[member].sum(dim=-1))
        return forward_sindy(h_current, key_module, participant_ids, experiment_ids, controls, polynomial_degree)

    model.init_forward_pass, model.call_module, model.forward_sindy = (
        tracked_init_forward_pass, tracked_call_module, tracked_forward_sindy)
    try:
        model(inputs)
    finally:
        del model.init_forward_pass, model.call_module, model.forward_sindy  # back to the class methods
    return {module: (squares[module] / counts[module].clamp_min(1).unsqueeze(-1)).sqrt().cpu().numpy() for module in squares}


def term_effect_sizes(estimator, dataset):
    """Per module: |coefficient| x RMS of the term's library value, (ensemble, participants, terms); 0 for absent terms."""
    rms = term_library_rms(estimator.model, dataset.xs)
    effects = {}
    for module, coefficients in estimator.get_sindy_coefficients(aggregate=False).items():  # (E, P, X, terms), masked
        coefficients = coefficients[:, :, 0].cpu().numpy()
        effects[module] = np.abs(coefficients) * rms[module] if module in rms else np.zeros(coefficients.shape)
    return effects


def member_matrices(estimator, member):
    """Coefficients, presence and term names of one ensemble member, as get_module_matrices gives them for the ensemble."""
    coefficients = estimator.get_sindy_coefficients(aggregate=False)
    terms = estimator.get_candidate_terms()
    return {module: dict(values=coefficients[module][member, :, 0].detach().cpu().numpy(),
                         presence=estimator.model.sindy_coefficients_presence[module][member, :, 0].float().cpu().numpy(),
                         terms=list(terms[module]))
            for module in estimator.get_modules()}


def ensemble_effects(effects, matrices_presence):
    """Mean effect size over the ensemble members carrying the term, (participants, terms)."""
    carrying = matrices_presence.astype(float)
    return (effects * carrying).sum(axis=0) / np.maximum(carrying.sum(axis=0), 1)


def load_models(study, recompute=False):
    """The generator (truth, effect size per participant x term) and per refit seed its ensemble matrices and effects."""
    output_dir = study_path(study, 'results', 'parameter_recovery')
    os.makedirs(output_dir, exist_ok=True)
    cache = os.path.join(output_dir, 'models.pkl')
    if os.path.exists(cache) and not recompute:
        with open(cache, 'rb') as file:
            return pickle.load(file)

    generators = pd.read_csv(os.path.join(ROOT, DEFAULT_DATA_DIR, 'generators.csv'))
    generator = generators[(generators['study'] == study) & (generators['generator'] == 'spice')].iloc[0]
    load, _ = model_setup(study, 'spice')
    train, _ = simulated_splits(study, 'spice', os.path.join(ROOT, DEFAULT_DATA_DIR))

    estimator = load(os.path.join(ROOT, generator['checkpoint']))
    member = int(generator['member'])
    truth = member_matrices(estimator, member)
    truth_effects = {module: values[member] for module, values in term_effect_sizes(estimator, train).items()}

    refits, refit_effects = [], []
    for path in fit_paths(os.path.join(ROOT, DEFAULT_RUNS_DIR), study, 'spice', 'spice'):
        estimator = load(str(path))
        effects = term_effect_sizes(estimator, train)
        presence = {module: estimator.model.sindy_coefficients_presence[module][:, :, 0].cpu().numpy() > 0 for module in effects}
        refits.append(get_module_matrices(estimator))
        refit_effects.append({module: ensemble_effects(effects[module], presence[module]) for module in effects})
        del estimator
        torch.cuda.empty_cache()
    models = dict(truth=truth, truth_effects=truth_effects, refits=refits, refit_effects=refit_effects)
    with open(cache, 'wb') as file:
        pickle.dump(models, file)
    return models


# ── Term cells ──────────────────────────────────────────────────────

def signed_presence(matrices, sign):
    return (matrices['presence'] > 0) & (np.sign(matrices['values']) == sign)


def term_cells(study, models):
    """All participant x signed term cells that the generator has or the seeds' vote recovers."""
    rows = []
    for module, truth in models['truth'].items():
        for sign, label in ((1, '+'), (-1, '-')):
            true = signed_presence(truth, sign)
            present = np.stack([signed_presence(refit[module], sign) for refit in models['refits']])  # (S, P, T)
            n_seeds = present.sum(axis=0)
            found = n_seeds >= VOTES
            effects = np.stack([refit[module] for refit in models['refit_effects']])
            effect_found = (effects * present).sum(axis=0) / np.maximum(n_seeds, 1)
            coefficients = np.stack([refit[module]['values'] for refit in models['refits']])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN slices: no seed has the term with this sign
                coefficient_found = np.nan_to_num(np.nanmedian(np.where(present, coefficients, np.nan), axis=0))
            participants, terms = np.nonzero(true | found)
            rows.append(pd.DataFrame(dict(
                study=study, module=module, term=[f"{truth['terms'][t]} ({label})" for t in terms], participant=participants,
                true=true[participants, terms], found=found[participants, terms],
                coefficient_true=truth['values'][participants, terms] * true[participants, terms],
                coefficient_found=coefficient_found[participants, terms] * found[participants, terms],
                effect_true=models['truth_effects'][module][participants, terms], effect_found=effect_found[participants, terms],
                prevalence_true=true.mean(axis=0)[terms], prevalence_found=found.mean(axis=0)[terms],
            )))
    cells = pd.concat(rows, ignore_index=True)
    cells['term_id'] = cells['module'] + '|' + cells['term']
    return cells


def metric_cells(cells, metric):
    """The cells a metric is computed on, with the bin variables and the outcome of that metric."""
    select, outcome, source = ('found', 'true', 'found') if metric == 'precision' else ('true', 'found', 'true')
    selected = cells[cells[select]].copy()
    selected['effect'] = selected[f'effect_{source}']
    selected['prevalence'] = selected[f'prevalence_{source}']
    selected['hit'] = selected[outcome].astype(float)
    return selected


def percentile_bin(values, n_bins):
    return np.floor(n_bins * (values.rank(method='first') - 1) / len(values)).astype(int)


def prevalence_bin(values, n_bins):
    return np.minimum(np.floor(values * n_bins + 1e-9).astype(int), n_bins - 1)


# ── Tables ──────────────────────────────────────────────────────────

def overall_table(cells):
    rows = []
    for study, group in cells.groupby('study', sort=False):
        hits = (group['true'] & group['found']).sum()
        rows.append(dict(study=study, n_true=int(group['true'].sum()), n_found=int(group['found'].sum()),
                         precision=hits / max(group['found'].sum(), 1), recall=hits / max(group['true'].sum(), 1)))
    return pd.DataFrame(rows)


def binned_table(cells, by):
    """Precision and recall per study and bin of `by` ('effect' or 'prevalence')."""
    tables = []
    for metric in METRICS:
        selected = metric_cells(cells, metric)
        if by == 'effect':
            selected['bin'] = selected.groupby('study')['effect'].transform(lambda values: percentile_bin(values, N_EFFECT_BINS))
        else:
            selected['bin'] = prevalence_bin(selected['prevalence'], N_PREVALENCE_BINS)
        table = selected.groupby(['study', 'bin'], sort=False).agg(
            value=('hit', 'mean'), n_cells=('hit', 'size'), n_terms=('term_id', 'nunique'),
            effect_min=('effect', 'min'), effect_median=('effect', 'median'), effect_max=('effect', 'max')).reset_index()
        tables.append(table.assign(metric=metric))
    return pd.concat(tables, ignore_index=True)


def grid_cells(cells, metric):
    selected = metric_cells(cells, metric)
    selected['effect_percentile'] = selected.groupby('study')['effect'].rank(method='first', pct=True)
    selected['effect_bin'] = selected.groupby('study')['effect'].transform(lambda values: percentile_bin(values, N_GRID_BINS))
    selected['prevalence_bin'] = prevalence_bin(selected['prevalence'], N_GRID_BINS)
    return selected


def grid_table(cells):
    """Precision and recall per study, prevalence bin and effect-size quintile."""
    tables = [grid_cells(cells, metric).groupby(['study', 'prevalence_bin', 'effect_bin'], sort=False).agg(
        value=('hit', 'mean'), n_cells=('hit', 'size'), n_terms=('term_id', 'nunique')).reset_index().assign(metric=metric)
        for metric in METRICS]
    return pd.concat(tables, ignore_index=True)


def across_studies_table(grid):
    """Mean, SD and range over the studies with enough terms in a cell; only cells with enough studies."""
    grid = grid[grid['n_terms'] >= MIN_TERMS_PER_CELL]
    table = grid.groupby(['metric', 'prevalence_bin', 'effect_bin'])['value'].agg(
        mean='mean', sd='std', minimum='min', maximum='max', n_studies='count').reset_index()
    return table[table['n_studies'] >= MIN_STUDIES_PER_CELL]


def standardized(values):
    return (values - values.mean()) / values.std() if values.std() > 0 else values * 0


def regression_table(cells):
    """Per study and metric: hit ~ effect percentile + prevalence (logistic, z-scored, SEs clustered by term)."""
    rows = []
    for metric in METRICS:
        selected = grid_cells(cells, metric)
        for study, group in selected.groupby('study', sort=False):
            row = dict(study=study, metric=metric, n_cells=len(group), n_terms=group['term_id'].nunique(),
                       odds_ratio_effect=np.nan, p_effect=np.nan, odds_ratio_prevalence=np.nan, p_prevalence=np.nan)
            if group['hit'].nunique() == 2 and group['prevalence'].std() > 0:
                predictors = sm.add_constant(pd.DataFrame(dict(effect=standardized(group['effect_percentile']),
                                                               prevalence=standardized(group['prevalence']))))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit = sm.GLM(group['hit'], predictors, family=sm.families.Binomial()).fit(
                        cov_type='cluster', cov_kwds=dict(groups=pd.factorize(group['term_id'])[0]))
                row.update(odds_ratio_effect=np.exp(fit.params['effect']), p_effect=fit.pvalues['effect'],
                           odds_ratio_prevalence=np.exp(fit.params['prevalence']), p_prevalence=fit.pvalues['prevalence'])
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(cells):
    """Per study: Spearman correlation over the generator's terms of prevalence and median effect size of the carriers."""
    rows = []
    for study, group in cells[cells['true']].groupby('study', sort=False):
        terms = group.groupby('term_id').agg(prevalence=('prevalence_true', 'first'), effect=('effect_true', 'median'))
        rho, p_value = spearmanr(terms['prevalence'], terms['effect'])
        rows.append(dict(study=study, n_terms=len(terms), rho=rho, p=p_value))
    return pd.DataFrame(rows)


# ── Figures ─────────────────────────────────────────────────────────


def style_heatmap(ax, n_rows, n_columns):
    ax.set_xticks(np.arange(-0.5, n_columns), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='both', length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def save_figure(fig, path):
    for extension in ('pdf', 'png'):
        fig.savefig(f'{path}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def study_ticks(ax, studies):
    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels([LABELS[study] for study in studies], rotation=45, ha='right', rotation_mode='anchor', fontsize=8)


def plot_binned(table, studies, bin_labels, bin_titles, title, path, show_terms=False, bins_on_x=False):
    """Precision and recall per study and bin; bins on the y-axis and studies on the x-axis, or the reverse with
    `bins_on_x`. Annotated and _clean."""
    cmap = rate_cmap()
    n_bins = len(bin_labels)
    if bins_on_x:
        figsize, index, columns, origin = ((0.62 * n_bins + 2.4) * 2, 0.55 * len(studies) + 2.4), ('study', studies), ('bin', range(n_bins)), 'upper'
    else:
        figsize, index, columns, origin = (max(3.6, 0.78 * len(studies) + 1.5) * 2, 5.9), ('bin', range(n_bins)), ('study', studies), 'lower'
    for annotated in (True, False):
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        for ax, metric in zip(axes, METRICS):
            subset = table[table['metric'] == metric]
            values, n_terms = (subset.pivot(index=index[0], columns=columns[0], values=column).reindex(index=index[1], columns=columns[1]).to_numpy(float)
                               for column in ('value', 'n_terms'))
            image = ax.imshow(np.ma.masked_invalid(values), origin=origin, aspect='auto', cmap=cmap, vmin=0, vmax=1)
            for (row, column), value in np.ndenumerate(values):
                if annotated and not np.isnan(value):
                    text = f'{value:.2f}' + (f'\n({int(n_terms[row, column])})' if show_terms else '')
                    ax.text(column, row, text, ha='center', va='center', fontsize=7.5, linespacing=0.95, color=text_color(cmap, value))
            if bins_on_x:
                ax.set_xticks(range(n_bins))
                ax.set_xticklabels(bin_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=8.5)
                ax.set_yticks(range(len(studies)))
                ax.set_yticklabels([LABELS[study].replace('\n', ' ') for study in studies], fontsize=8.5)
                style_heatmap(ax, len(studies), n_bins)
            else:
                study_ticks(ax, studies)
                ax.set_yticks(range(n_bins))
                ax.set_yticklabels(bin_labels, fontsize=9)
                style_heatmap(ax, n_bins, len(studies))
            if annotated:
                (ax.set_xlabel if bins_on_x else ax.set_ylabel)(bin_titles[metric], fontsize=9.5)
                ax.set_title(TITLES[metric], fontsize=12, loc='left', fontweight='bold')
        colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01)
        if annotated:
            colorbar.set_label('score')
            fig.suptitle(title, fontsize=11.5)
        save_figure(fig, path if annotated else f'{path}_clean')


def grid_matrix(table, column):
    """(effect quintile, prevalence bin): effect size on the y-axis, prevalence on the x-axis."""
    return table.pivot(index='effect_bin', columns='prevalence_bin', values=column).reindex(
        index=range(N_GRID_BINS), columns=range(N_GRID_BINS)).to_numpy(float)


def style_grid_axis(ax, annotated, x_title, rotate=False):
    ax.set_xticks(range(N_GRID_BINS))
    if rotate:
        ax.set_xticklabels(percent_labels(N_GRID_BINS), fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
    else:
        ax.set_xticklabels(percent_labels(N_GRID_BINS), fontsize=8.5)
    ax.set_yticks(range(N_GRID_BINS))
    ax.set_yticklabels([label.replace('\n', ' ') for label in GRID_EFFECT_LABELS], fontsize=8)
    if annotated:
        ax.set_xlabel(x_title, fontsize=8.5)
        ax.set_ylabel('effect size within study (quintile)', fontsize=8.5)


GRID_X_TITLES = {'precision': "participants carrying the term\n(recovered model's prevalence and effect size)",
                 'recall': "participants carrying the term\n(generator's prevalence and effect size)"}


def plot_grid_per_study(grid, studies, path):
    """Prevalence x effect size per study: precision (first row) and recall (second row)."""
    cmap = rate_cmap()
    for annotated in (True, False):
        fig, axes = plt.subplots(2, len(studies), figsize=(2.9 * len(studies) + 1.2, 7), constrained_layout=True, squeeze=False)
        for row, metric in enumerate(METRICS):
            for column, study in enumerate(studies):
                ax = axes[row, column]
                subset = grid[(grid['metric'] == metric) & (grid['study'] == study)]
                values, n_terms = grid_matrix(subset, 'value'), grid_matrix(subset, 'n_terms')
                image = ax.imshow(np.ma.masked_invalid(values), origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
                for (y, x), value in np.ndenumerate(values):
                    if annotated and not np.isnan(value):
                        ax.text(x, y, f'{value:.2f}\n({int(n_terms[y, x])})', ha='center', va='center', fontsize=6.5,
                                linespacing=0.95, color=text_color(cmap, value))
                style_grid_axis(ax, annotated and column == 0, f'{TITLES[metric]}: {GRID_X_TITLES[metric]}', rotate=True)
                style_heatmap(ax, N_GRID_BINS, N_GRID_BINS)
                if annotated and row == 0:
                    ax.set_title(LABELS[study], fontsize=8.5)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.01)
        if annotated:
            colorbar.set_label('score')
            fig.suptitle('Term recovery by prevalence and effect size (brackets: number of terms)', fontsize=11.5)
        save_figure(fig, path if annotated else f'{path}_clean')


def plot_across_studies(grid, path):
    """One panel per metric: colour = mean over studies, text = mean ± SD, dots = the individual studies on a 0-1 scale."""
    cmap = rate_cmap()
    grid = grid[grid['n_terms'] >= MIN_TERMS_PER_CELL]
    for annotated in (True, False):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 7), constrained_layout=True)
        for ax, metric in zip(axes, METRICS):
            subset = grid[grid['metric'] == metric]
            for row in range(N_GRID_BINS):
                for column in range(N_GRID_BINS):
                    values = subset[(subset['effect_bin'] == row) & (subset['prevalence_bin'] == column)]['value'].to_numpy()
                    enough = len(values) >= MIN_STUDIES_PER_CELL
                    color = cmap(values.mean()) if enough else EMPTY_COLOR
                    ax.add_patch(Rectangle((column - 0.47, row - 0.47), 0.94, 0.94, facecolor=color, edgecolor='none'))
                    if not enough:
                        continue
                    ink = text_color(cmap, values.mean())
                    if annotated:
                        ax.text(column, row + 0.17, f'{values.mean():.2f} ± {values.std(ddof=1):.2f}', ha='center', va='center',
                                fontsize=10, fontweight='bold', color=ink)
                    left, right, y = column - 0.39, column + 0.39, row - 0.2
                    ax.plot([left, right], [y, y], color=ink, linewidth=0.8, alpha=0.75)
                    for edge in (left, right):
                        ax.plot([edge, edge], [y - 0.035, y + 0.035], color=ink, linewidth=0.8, alpha=0.75)
                    ax.scatter(left + values * (right - left), np.full(len(values), y), s=25, facecolor='white',
                               edgecolor='#1f1f1f', linewidth=0.7, alpha=0.92, zorder=3)
            ax.set_xlim(-0.5, N_GRID_BINS - 0.5)
            ax.set_ylim(-0.5, N_GRID_BINS - 0.5)
            ax.tick_params(length=0)
            style_grid_axis(ax, annotated, GRID_X_TITLES[metric])
            for side in ax.spines.values():
                side.set_visible(False)
            if annotated:
                ax.set_title(TITLES[metric], fontsize=12.5, loc='left', fontweight='bold')
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap), ax=axes, shrink=0.7, pad=0.012)
        if annotated:
            colorbar.set_label('mean score across studies')
            fig.suptitle('Term recovery by prevalence and effect size. Dots: individual studies on a 0–1 scale', fontsize=12.5)
        save_figure(fig, path if annotated else f'{path}_clean')


def plot_mean_sd(across, path):
    """Mean over studies (first row) and SD between studies (second row)."""
    rows = (('mean', 'mean across studies', rate_cmap(), 1), ('sd', 'SD across studies', rate_cmap(), 0.5))
    for annotated in (True, False):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.4), constrained_layout=True)
        for row, (column_name, title, cmap, vmax) in enumerate(rows):
            for column, metric in enumerate(METRICS):
                ax = axes[row, column]
                subset = across[across['metric'] == metric]
                values, n_studies = grid_matrix(subset, column_name), grid_matrix(subset, 'n_studies')
                image = ax.imshow(np.ma.masked_invalid(values), origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
                for (y, x), value in np.ndenumerate(values):
                    if annotated and not np.isnan(value):
                        ax.text(x, y, f'{value:.2f}\n({int(n_studies[y, x])})', ha='center', va='center', fontsize=9,
                                linespacing=1, color=text_color(cmap, value, vmax=vmax))
                style_grid_axis(ax, annotated, GRID_X_TITLES[metric])
                style_heatmap(ax, N_GRID_BINS, N_GRID_BINS)
                if annotated:
                    ax.set_title(f'{TITLES[metric]}: {title}', fontsize=11.5, loc='left', fontweight='bold')
            colorbar = fig.colorbar(image, ax=axes[row, :], shrink=0.82, pad=0.012)
            if annotated:
                colorbar.set_label(title)
        if annotated:
            fig.suptitle('Term recovery by prevalence and effect size (brackets: studies contributing)', fontsize=12)
        save_figure(fig, path if annotated else f'{path}_clean')


# ── Pipeline ────────────────────────────────────────────────────────

EFFECT_Y_TITLES = {'precision': "recovered model's effect size (percentile bin; top = largest)",
                   'recall': "generator's effect size (percentile bin; top = largest)"}
PREVALENCE_TITLES = {'precision': 'participants given the term by the recovered model',
                       'recall': 'participants carrying the term in the generator'}


def write_results(cells, studies, output_dir):
    """Every table and figure for the given studies; the analyses of individual differences skip group models."""
    os.makedirs(output_dir, exist_ok=True)
    overall_table(cells).to_csv(os.path.join(output_dir, 'recovery_overall.csv'), index=False)

    by_effect = binned_table(cells, 'effect')
    by_effect.to_csv(os.path.join(output_dir, 'recovery_by_effect_size.csv'), index=False)
    plot_binned(by_effect, studies, percent_labels(N_EFFECT_BINS), EFFECT_Y_TITLES, 'Term recovery by effect size',
                os.path.join(output_dir, 'figure_recovery_by_effect_size'))

    by_prevalence = binned_table(cells, 'prevalence')
    by_prevalence.to_csv(os.path.join(output_dir, 'recovery_by_prevalence.csv'), index=False)
    plot_binned(by_prevalence, studies, percent_labels(N_PREVALENCE_BINS), PREVALENCE_TITLES,
                'Term recovery by share of participants carrying the term (brackets: number of terms)',
                os.path.join(output_dir, 'figure_recovery_by_prevalence'), show_terms=True, bins_on_x=True)

    individual = [study for study in studies if study not in GROUP_STUDIES]
    if not individual:
        return None
    cells = cells[cells['study'].isin(individual)]
    grid = grid_table(cells)
    grid.to_csv(os.path.join(output_dir, 'recovery_by_prevalence_effect.csv'), index=False)
    plot_grid_per_study(grid, individual, os.path.join(output_dir, 'figure_recovery_by_prevalence_effect'))
    regression_table(cells).to_csv(os.path.join(output_dir, 'prevalence_effect_regression.csv'), index=False)
    correlation_table(cells).to_csv(os.path.join(output_dir, 'prevalence_effect_correlation.csv'), index=False)
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--studies', nargs='+', default=None, help='default: all; the across-study results need all studies')
    parser.add_argument('--recompute', action='store_true', help='ignore the cached models')
    args = parser.parse_args()
    all_studies = ordered_studies()
    studies = [study for study in all_studies if args.studies is None or study in args.studies]

    all_cells = []
    for study in studies:
        cells = term_cells(study, load_models(study, args.recompute))
        write_results(cells, [study], study_path(study, 'results', 'parameter_recovery'))
        all_cells.append(cells)
        print(f'{study}: done', flush=True)

    if studies == all_studies:
        grid = write_results(pd.concat(all_cells, ignore_index=True), studies, OUTPUT_ACROSS)
        across = across_studies_table(grid)
        across.to_csv(os.path.join(OUTPUT_ACROSS, 'recovery_by_prevalence_effect_across_studies.csv'), index=False)
        plot_across_studies(grid, os.path.join(OUTPUT_ACROSS, 'figure_prevalence_effect_across_studies'))
        plot_mean_sd(across, os.path.join(OUTPUT_ACROSS, 'figure_prevalence_effect_mean_sd'))
        print(f'Across studies: {OUTPUT_ACROSS}')


if __name__ == '__main__':
    main()

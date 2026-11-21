"""
Parameter recovery by target: recall, precision, coefficient slope and coefficient SD of the Q-learning recovery
(run_parameter_recovery.py) and the SPICE -> SPICE recovery (run_spice_recovery.py), binned by four targets:
    trials per dataset       Q-learning: one bin per dataset size (8 datasets each); SPICE -> SPICE: one bin per study
    trials per participant   Q-learning: one bin (all datasets have 400 trials per participant); SPICE -> SPICE: per study.
                             A target with a single bin over the shown recoveries is left out (Q-learning alone)
    carriers per term        participants of a dataset carrying the term: 1-2, 3-4, 5-8, ..., 257-512, > 512
    share carrying the term  carriers / participants of the dataset: 20% bins (0-20%, ..., 80-100%)
    effect size per term     quintiles within a dataset (effect size as in run_spice_recovery.py)

Cells are participant x signed term cells (`term (+)` and `term (-)` are different terms):
    precision  share of the recovered cells the ground truth has; binned by the recovered model's carriers / share / effect
               size: what a term of the fitted model is worth, the only side a user sees (terms the ground truth gives no
               participant included)
    recall     share of the ground truth's cells that are recovered; binned by the ground truth's carriers / share / effect size
    slope      true positives, per term: OLS slope of the recovered on the true coefficient over participants
               (>= 3 cells, true coefficients not constant); median over the terms of a bin           (target: 1)
    sd         as slope, residual SD / SD of the true coefficients                                         (target: 0)
Each dataset (Q-learning: 40 datasets; SPICE -> SPICE: one per study) weighs equally: a metric is computed per dataset and
bin (precision / recall need >= 3 cells) and averaged over the datasets. SPICE -> SPICE bins of the first two targets are
the studies, sorted from the smallest to the biggest value.

Output in weinhardt2026/results/parameter_recovery/:
    recovery_by_target.csv
    recovery_spurious_share.csv  per recovery: share of the recovered cells on spurious terms (mean over datasets)
With --split (one recovery) the metrics per dataset size / study and the share x effect-size matrices of precision and
recall: one panel per dataset size / study and the mean over all datasets (each dataset weighs equally; a dataset counts
in a cell with >= 3 cells):
    recovery_by_target_<recovery>_by_<size / study>.csv
    recovery_grid_<recovery>_by_<size / study>.csv
    figure_recovery_grid_<recovery>_by_<size / study>(.pdf/.png, _clean)

Usage:
    python weinhardt2026/analysis/validation/run_recovery_by_target.py                          # CSVs of both recoveries
    python weinhardt2026/analysis/validation/run_recovery_by_target.py --recoveries qlearning --split   # per dataset size
    python weinhardt2026/analysis/validation/run_recovery_by_target.py --recoveries spice --split       # per study
"""

import argparse
import glob
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules import 'studies.<study>...'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spice.utils.plotting import rate_cmap
from weinhardt2026.analysis.validation.model_identification.fit_confusion_matrix import DEFAULT_DATA_DIR
from weinhardt2026.analysis.validation.model_identification.run_model_identification import simulated_splits
from weinhardt2026.analysis.validation.run_spice_recovery import (GRID_EFFECT_LABELS, load_models, ordered_studies,
                                                                   percentile_bin, term_cells, text_color)
from weinhardt2026.utils.studies import DOMAINS, STUDY_DOMAINS

OUTPUT_DIR = os.path.join(ROOT, 'weinhardt2026', 'results', 'parameter_recovery')
QLEARNING_CELLS = os.path.join(OUTPUT_DIR, 'spice_synthetic_*_cells.csv')

RECOVERIES = {'qlearning': 'Q-learning', 'spice': 'SPICE → SPICE'}


def recovery_label(recovery):
    """'qlearning', 'spice' or, split (SPLITS), '<recovery>_<dataset size or study>'."""
    return RECOVERIES.get(recovery, recovery.split('_', 1)[-1])


# --split: per recovery the column that defines a unit (dataset size / study)
SPLITS = {'qlearning': 'n_participants', 'spice': 'dataset'}


def split_recoveries(cells, recovery):
    """Relabel the cells of one recovery by dataset size (Q-learning) or study (SPICE -> SPICE); the lines from the
    smallest to the biggest dataset."""
    column = SPLITS[recovery]
    suffix = cells[column].astype(str) + ('p' if column == 'n_participants' else '')
    cells = cells.assign(recovery=recovery + '_' + suffix)
    order = list(cells.groupby('recovery')['n_trials'].mean().sort_values().index)
    if recovery == 'spice':  # grouped by task domain, within a domain by trials per dataset
        domains = list(DOMAINS)
        order.sort(key=lambda name: domains.index(STUDY_DOMAINS[recovery_label(name)]))
    return cells, order
METRICS = {'recall': ('Recall', 1.), 'precision': ('Precision', 1.), 'slope': ('Coefficient slope', 1.), 'sd': ('Coefficient SD', 0.)}
TARGETS = {
    'trials_dataset': 'Trials per dataset',
    'trials_participant': 'Trials per participant',
    'carriers': 'Carriers per term',
    'share': 'Share carrying the term',
    'effect': 'Effect size per term',
}
N_CARRIER_BINS = 10  # bins close at powers of two: 1-2, 3-4, 5-8, ..., 257-512, > 512
N_SHARE_BINS = 5  # 0-20%, ..., 80-100%
BINNED_TARGETS = ('carriers', 'share', 'effect')  # fixed bins in both recoveries (the others: one bin per study)
N_EFFECT_BINS = 5
MIN_CELLS = 3  # precision / recall of a dataset in a bin
MIN_TERM_CELLS = 3  # true positives of a term for its slope / SD


# ── Cells ───────────────────────────────────────────────────────────

def dataset_size(study):
    """Valid trials and participants with valid trials of the simulated training data of a study."""
    train, _ = simulated_splits(study, 'spice', os.path.join(ROOT, DEFAULT_DATA_DIR))
    valid = ~train.xs[:, :, :, 0].isnan().all(dim=2)  # (B, T)
    participant_ids = train.xs[:, 0, 0, -1].long()  # metadata: participant id is the last column
    return int(valid.sum()), int(participant_ids[valid.any(dim=1)].unique().numel())


def qlearning_cells(path_pattern):
    paths = glob.glob(path_pattern)
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one Q-learning cells file for {path_pattern}, found {paths}; run run_parameter_recovery.py.')
    cells = pd.read_csv(paths[0])
    return cells.assign(recovery='qlearning', dataset_label=cells['n_participants'].astype(str) + 'p')


def spice_cells():
    tables = []
    for study in ordered_studies():
        n_trials, n_participants = dataset_size(study)
        tables.append(term_cells(study, load_models(study)).assign(
            recovery='spice', dataset=study, dataset_label=study, n_trials=n_trials, n_participants=n_participants))
    return pd.concat(tables, ignore_index=True)


# ── Metrics ─────────────────────────────────────────────────────────

def metric_side(metric):
    """The cells ('true' / 'found') a metric is computed on: precision the recovered model's, the others the ground truth's."""
    return 'found' if metric == 'precision' else 'true'


def metric_cells(binned, metric):
    """The binned cells of a metric: for slope and SD the true positives."""
    return binned[binned['found']] if metric in ('slope', 'sd') else binned


def term_fit(group):
    """OLS of the recovered on the true coefficients of one term: (slope, residual SD / SD of the true coefficients)."""
    x, y = group['coefficient_true'].to_numpy(), group['coefficient_found'].to_numpy()
    if len(x) < MIN_TERM_CELLS or np.std(x) == 0:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return slope, np.std(y - (slope * x + intercept)) / np.std(x)


def metric_value(selected, metric):
    """Metric of one dataset in one bin; NaN without enough cells / terms."""
    if metric in ('precision', 'recall'):
        if len(selected) < MIN_CELLS:
            return np.nan
        return selected[{'precision': 'true', 'recall': 'found'}[metric]].mean()
    fits = [term_fit(group) for _, group in selected.groupby('term_id')]
    values = np.array([fit[metric == 'sd'] for fit in fits], dtype=float)
    return np.nanmedian(values) if np.isfinite(values).any() else np.nan


def bin_cells(cells, metric, target):
    """The cells of one recovery a metric is computed on (its side: precision the recovered model's, the others the
    ground truth's), with the bin and the bin value of one target, from the same side. Carriers, share and effect
    quintiles over all cells of the side, so the true positives of slope and SD keep the ground truth's bins."""
    side = metric_side(metric)
    selected = cells[cells[side]].copy()
    if target in ('carriers', 'share'):
        carriers = selected.groupby(['dataset', 'term_id'])[side].transform('sum')
    if target == 'carriers':
        selected['bin'] = np.clip(np.ceil(np.log2(carriers) - 1e-9).astype(int) - 1, 0, N_CARRIER_BINS - 1)
        selected['bin_value'] = 2. ** (selected['bin'] + 1)  # the bin's upper edge
    elif target == 'share':
        share = carriers / selected['n_participants']
        selected['bin'] = np.minimum(np.floor(share * N_SHARE_BINS + 1e-9).astype(int), N_SHARE_BINS - 1)
        selected['bin_value'] = (selected['bin'] + 0.5) / N_SHARE_BINS  # bin centre: the log shading needs values > 0
    elif target == 'effect':
        selected['bin'] = selected.groupby('dataset')[f'effect_{side}'].transform(lambda values: percentile_bin(values, N_EFFECT_BINS))
        selected['bin_value'] = selected.groupby(['dataset', 'bin'])[f'effect_{side}'].transform('median')
    elif target == 'trials_dataset':
        selected['bin_value'] = selected['n_trials']
        selected['bin'] = selected['n_participants'] if selected['recovery'].iloc[0].startswith('qlearning') else selected['dataset']
    else:
        selected['bin_value'] = selected['n_trials'] / selected['n_participants']
        selected['bin'] = 0 if selected['recovery'].iloc[0].startswith('qlearning') else selected['dataset']
    return selected


def bin_label(recovery, target, bin_id, bin_value):
    if target == 'carriers':
        low, high = 2 ** bin_id + 1 if bin_id else 1, 2 ** (bin_id + 1)
        return f'>{2 ** bin_id}' if bin_id == N_CARRIER_BINS - 1 else f'{low}–{high}'
    if target == 'share':
        return f'{100 * bin_id // N_SHARE_BINS}–{100 * (bin_id + 1) // N_SHARE_BINS}%'
    if target == 'effect':
        return f'Q{bin_id + 1}'
    value = f'{bin_value / 1000:.1f}k' if target == 'trials_dataset' else f'{bin_value:.0f}'
    if recovery.startswith('qlearning'):
        return f'{bin_id}p: {value}' if target == 'trials_dataset' else value
    return f'{bin_id}: {value}'


def spurious_share(cells):
    """Per recovery: share of the recovered cells on terms no participant of the dataset carries (mean over datasets)."""
    found = cells[cells['found']].copy()
    found['spurious'] = cells.groupby(['dataset', 'term_id'])['true'].transform('sum').loc[found.index] == 0
    per_dataset = found.groupby(['recovery', 'dataset'], sort=False)['spurious'].mean()
    return per_dataset.groupby('recovery', sort=False).agg(spurious_share='mean', sd_datasets='std', n_datasets='size').reset_index()


def target_table(cells):
    """One row per recovery, target, bin and metric: the mean over datasets of the per-dataset metric."""
    rows = []
    for recovery, recovery_cells in cells.groupby('recovery', sort=False):
        for metric in METRICS:
            for target in TARGETS:
                binned = metric_cells(bin_cells(recovery_cells, metric, target), metric)
                for bin_id, group in binned.groupby('bin'):
                    per_dataset = np.array([metric_value(dataset_cells, metric) for _, dataset_cells in group.groupby('dataset')])
                    valid = per_dataset[np.isfinite(per_dataset)]
                    bin_value = group.groupby('dataset')['bin_value'].first().mean()
                    rows.append(dict(
                        recovery=recovery, target=target, metric=metric, bin=bin_id,
                        bin_label=bin_label(recovery, target, bin_id, bin_value), bin_value=bin_value,
                        value=valid.mean() if len(valid) else np.nan, sd_datasets=valid.std() if len(valid) > 1 else np.nan,
                        n_datasets=len(valid), n_cells=len(group), n_terms=group.groupby(['dataset', 'term_id']).ngroups))
    table = pd.DataFrame(rows)
    # bins in plotting order: numeric bins ascending, studies by their bin value
    table['order'] = table.groupby(['recovery', 'target', 'metric'])['bin_value'].rank(method='first')
    numeric = table['target'].isin(BINNED_TARGETS) | table['recovery'].str.startswith('qlearning')
    table.loc[numeric, 'order'] = table[numeric].groupby(['recovery', 'target', 'metric'])['bin'].rank(method='first')
    return table.sort_values(['target', 'recovery', 'metric', 'order']).reset_index(drop=True)


GRID_METRICS = ('precision', 'recall')
MEAN_UNIT = 'mean'  # the panel over all datasets


def grid_table(cells, units):
    """Precision and recall per dataset, share bin and effect quintile (within the dataset; both from the metric's side:
    precision the recovered model's, recall the ground truth's), averaged per unit (dataset size / study) and over all datasets (MEAN_UNIT): every dataset weighs
    equally."""
    tables = []
    for metric in GRID_METRICS:
        selected = bin_cells(cells, metric, 'share').rename(columns={'bin': 'share_bin'})
        effect = f'effect_{metric_side(metric)}'
        selected['effect_bin'] = selected.groupby('dataset')[effect].transform(lambda values: percentile_bin(values, N_EFFECT_BINS))
        outcome = {'precision': 'true', 'recall': 'found'}[metric]
        per_dataset = selected.groupby(['recovery', 'dataset', 'share_bin', 'effect_bin'])[outcome].agg(value='mean', n_cells='size').reset_index()
        n_terms = selected.groupby(['recovery', 'dataset', 'share_bin', 'effect_bin'])['term_id'].nunique().rename('n_terms').reset_index()
        per_dataset = per_dataset.merge(n_terms, on=['recovery', 'dataset', 'share_bin', 'effect_bin'])
        per_dataset = per_dataset[per_dataset['n_cells'] >= MIN_CELLS]
        for unit, group in [(unit, per_dataset[per_dataset['recovery'] == unit]) for unit in units] + [(MEAN_UNIT, per_dataset)]:
            table = group.groupby(['share_bin', 'effect_bin']).agg(value=('value', 'mean'), sd_datasets=('value', 'std'),
                                                                   n_datasets=('value', 'size'), n_terms=('n_terms', 'sum')).reset_index()
            tables.append(table.assign(metric=metric, unit=unit))
    return pd.concat(tables, ignore_index=True)


def plot_grid(grid, path, units, split_recovery):
    """Share (x) x effect quintile (y): precision (first row) and recall (second row), one column per unit and the mean."""
    cmap = rate_cmap()
    columns = list(units) + [MEAN_UNIT]
    mean_title = f'Mean over {"datasets" if split_recovery == "qlearning" else "studies"}'
    share_labels = [bin_label('', 'share', bin_id, None) for bin_id in range(N_SHARE_BINS)]
    for annotated in (True, False):
        fig, axes = plt.subplots(len(GRID_METRICS), len(columns), figsize=(2.6 * len(columns) + 1.2, 6), constrained_layout=True, squeeze=False)
        for row, metric in enumerate(GRID_METRICS):
            for column, unit in enumerate(columns):
                ax = axes[row, column]
                subset = grid[(grid['metric'] == metric) & (grid['unit'] == unit)]
                values = subset.pivot(index='effect_bin', columns='share_bin', values='value').reindex(
                    index=range(N_EFFECT_BINS), columns=range(N_SHARE_BINS)).to_numpy(float)
                image = ax.imshow(np.ma.masked_invalid(values), origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
                if annotated:
                    for (y, x), value in np.ndenumerate(values):
                        if np.isfinite(value):
                            ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=6.5, color=text_color(cmap, value))
                    ax.set_title(mean_title if unit == MEAN_UNIT else recovery_label(unit), fontsize=9,
                                 fontweight='bold' if unit == MEAN_UNIT else 'normal')
                ax.set_xticks(range(N_SHARE_BINS))
                ax.set_xticklabels(share_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=7)
                ax.set_yticks(range(N_EFFECT_BINS))
                ax.set_yticklabels([label.replace('\n', ' ') for label in GRID_EFFECT_LABELS] if column == 0 else [], fontsize=7)
                ax.spines[:].set_visible(False)
                ax.tick_params(length=0)
                if annotated and column == 0:
                    side = "recovered model's" if metric == 'precision' else "ground truth's"
                    ax.set_ylabel(f'{METRICS[metric][0]} ({side} share and effect)\neffect size (quintile)', fontsize=8.5)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01)
        if annotated:
            colorbar.set_label('score')
            fig.supxlabel('share of participants carrying the term', fontsize=9)
            fig.suptitle(f'{RECOVERIES[split_recovery]} recovery by share carrying the term and effect size', fontsize=11.5)
        suffix = '' if annotated else '_clean'
        for extension in ('pdf', 'png'):
            fig.savefig(f'{path}{suffix}.{extension}', dpi=300, bbox_inches='tight')
        plt.close(fig)


# ── Figure ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qlearning-cells', default=QLEARNING_CELLS, help='cells file (glob) written by run_parameter_recovery.py')
    parser.add_argument('--recoveries', nargs='+', choices=list(RECOVERIES), default=list(RECOVERIES),
                        help='recovery studies to show; a subset is written with its names as suffix')
    parser.add_argument('--split', action='store_true',
                        help='one recovery only: metrics and share x effect-size matrices per dataset size (Q-learning) or study (SPICE -> SPICE), '
                             'no trials targets; suffix _by_size / _by_study')
    args = parser.parse_args()
    if args.split and len(args.recoveries) != 1:
        parser.error('--split needs a single recovery (--recoveries qlearning or --recoveries spice)')
    loaders = {'qlearning': lambda: qlearning_cells(args.qlearning_cells), 'spice': spice_cells}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        cells = pd.concat([loaders[recovery]() for recovery in args.recoveries], ignore_index=True)
        recoveries = list(args.recoveries)
        if args.split:
            cells, recoveries = split_recoveries(cells, args.recoveries[0])
        table = target_table(cells)
        spurious = spurious_share(cells)
        grid = grid_table(cells, recoveries) if args.split else None
        if args.split:
            table = table[~table['target'].isin(['trials_dataset', 'trials_participant'])]  # constant within a line
    # a target with one bin over the shown recoveries has nothing to compare (Q-learning alone: 400 trials per participant)
    table = table[table.groupby('target')['bin'].transform('nunique') > 1]
    suffix = '' if set(args.recoveries) == set(RECOVERIES) else '_' + '_'.join(args.recoveries)
    suffix += {'qlearning': '_by_size', 'spice': '_by_study'}[args.recoveries[0]] if args.split else ''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table.drop(columns='order').to_csv(os.path.join(OUTPUT_DIR, f'recovery_by_target{suffix}.csv'), index=False)
    spurious.to_csv(os.path.join(OUTPUT_DIR, f'recovery_spurious_share{suffix}.csv'), index=False)
    print(spurious.to_string(index=False))
    if args.split:
        grid.to_csv(os.path.join(OUTPUT_DIR, f'recovery_grid{suffix}.csv'), index=False)
        plot_grid(grid, os.path.join(OUTPUT_DIR, f'figure_recovery_grid{suffix}'), recoveries, args.recoveries[0])
    print(f'Results: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

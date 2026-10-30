"""Model confusion matrix of SPICE and the benchmark, from the fits of fit_confusion_matrix.py.

Every fitted model is scored on the simulated dataset it was fit to, by three criteria:
    loglik  held-out log-likelihood per trial on the test blocks, per participant (discrete: of the chosen
            actions; continuous: Gaussian with the participant's MLE variance on the fit's training residuals)
    aic     AIC on the training blocks, per participant with the participant's own parameter count
    bic     (continuous: Gaussian with the participant's MLE variance); only for the interpretable models,
            the benchmark and SPICE-EQ
kolff2025 is scored on its targets only, and its AIC/BIC on the pooled training data: both models are group
models, their parameters are shared by all apes. For SPICE, each seed is represented by its ensemble member
with the highest SPICE-EQ training likelihood on that dataset (AIC/BIC: that member's active coefficients);
SPICE-EQ is compared with the benchmark, and SPICE-RNN (held-out likelihood only) in the same way alongside.

Per study, generator, criterion and unit (participant; kolff2025's AIC/BIC: the pooled data), the model with the better score (mean over
the fitting seeds) wins. The confusion matrix is P(fitted model wins | generator), the inversion matrix
P(generator | fitted model wins) (equal priors: both generators simulate the same participants). The seed
spread pairs the k-th SPICE seed with the k-th benchmark seed (kolff2025's single closed-form benchmark with
every SPICE seed).

Written to <out_dir>/ (default weinhardt2026/results/model_identification/):
    confusion_scores.csv   per (study, generator, model, seed, criterion, unit): n_trials, score
    confusion_matrix.csv   per (study, criterion, spice variant, generator): P(SPICE wins) (mean over units and
                           over seed pairs, with SD), mean score difference SPICE - benchmark, P(generator | winner)
    confusion_matrix_<criterion>.csv   the confusion matrices: rows (study, simulating model), columns (comparison,
                           fitted model, p / sd): P(fitted model wins | simulating model) and its SD over seed pairs
    figure_confusion_matrix_<criterion>.png/.pdf   one 2x2 matrix per study (rows: simulating model, columns: fitted
                           model, diagonal = correct identification); one row of panels per comparison

Usage (from the repository root):
    python weinhardt2026/analysis/validation/model_identification/run_model_identification.py
    python weinhardt2026/analysis/validation/model_identification/run_model_identification.py --cached   # re-aggregate confusion_scores.csv
"""

import sys
import argparse
import re
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'weinhardt2026'))  # study modules import 'studies.<study>...' (as in run.py)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spice import SpiceDataset, split_data_along_blockdim
from weinhardt2026.analysis.validation.model_identification.fit_confusion_matrix import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR as DEFAULT_RUNS_DIR
from weinhardt2026.analysis.validation.model_identification.generate_datasets import GENERATORS, STUDIES, dataset_path, load_csv, predict, study_data


DEFAULT_OUT_DIR = 'weinhardt2026/results/model_identification'
SPICE_VARIANTS = ('SPICE-EQ', 'SPICE-RNN')
CRITERIA = ('loglik', 'aic', 'bic')
HIGHER_IS_BETTER = {'loglik': True, 'aic': False, 'bic': False}
INFORMATION_CRITERIA_MODELS = ('Benchmark', 'SPICE-EQ')
GROUP_MODEL_STUDIES = ('kolff2025',)   # parameters shared by all participants: AIC/BIC on the pooled data


# -------------------------------------------------------------------------------------------
# DATA AND MODELS
# -------------------------------------------------------------------------------------------

def simulated_splits(study: str, generator: str, data_dir: str) -> tuple[SpiceDataset, SpiceDataset]:
    """Train and test blocks of a simulated dataset, split as for the fits (kolff2025: scored targets only)."""
    dataset = load_csv(study, dataset_path(data_dir, study, generator))
    train, test = split_data_along_blockdim(dataset, study_data(study)[1])
    if study == 'kolff2025':
        from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import evaluation_dataset
        train, test = evaluation_dataset(train), evaluation_dataset(test)
    return train, test


def fit_paths(runs_dir: str, study: str, fitter: str, generator: str) -> list[Path]:
    """Finished fits (with json) of `fitter` on the dataset of `generator`, by seed."""
    paths = [Path(path).with_suffix('.pkl') for path in glob(str(Path(runs_dir) / study / f'{fitter}_{study}_sim{generator}_rs*.json'))]
    return sorted(paths, key=lambda path: int(re.search(r'_rs(\d+)\.pkl$', path.name).group(1)))


def seed_of(path: Path) -> int:
    return int(re.search(r'_rs(\d+)\.pkl$', path.name).group(1))


def load_benchmark(study: str, path: Path, dataset: SpiceDataset):
    if study == 'kolff2025':
        from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import LogisticRegressionModel
        model = LogisticRegressionModel()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
    from weinhardt2026.utils.studies import STUDIES as REGISTRY
    return REGISTRY[study]['benchmark'](str(path), dataset.n_participants, dataset.n_actions)


def load_spice_member(study: str, path: Path, train: SpiceDataset):
    """The fit's member with the highest SPICE-EQ training likelihood, and its active coefficients per participant:
    summed over the experiments the participant has training data in. SPICE allocates a coefficient set per
    experiment; a model that passes no experiment index (all studies here) only uses the first, and the unused
    ones stay zero (e.g. bruckner2025's second set), so this is the count of the set the model uses."""
    from weinhardt2026.utils.study_models import EnsembleMember, member_train_loglik
    from weinhardt2026.utils.final_models import model_setup
    load, _ = model_setup(study, 'spice')
    estimator = load(str(path))
    member = int(np.argmax(member_train_loglik(study, estimator, train)))
    n_coefficients = estimator.count_sindy_coefficients(aggregate=False)[member].cpu()   # (n_participants, n_experiments)
    observed = torch.zeros_like(n_coefficients, dtype=torch.bool)
    participant, experiment = train.xs[:, 0, 0, -1].long().cpu(), train.xs[:, 0, 0, -2].long().cpu()
    if n_coefficients.shape[0] == 1:   # group model
        participant = torch.zeros_like(participant)
    observed[participant, experiment.clamp_max(n_coefficients.shape[1] - 1)] = True
    return EnsembleMember(estimator, member), (n_coefficients * observed).sum(dim=1)


def n_benchmark_parameters(study: str, model, dataset: SpiceDataset) -> float:
    """Parameters per participant (kolff2025: of the group model), as analysis/evaluation/run_model_evaluation.py counts them."""
    if hasattr(model, 'count_parameters'):
        return float(model.count_parameters())
    return sum(p.numel() for p in model.parameters()) / dataset.n_participants


# -------------------------------------------------------------------------------------------
# PER-PARTICIPANT HELD-OUT LIKELIHOOD
# -------------------------------------------------------------------------------------------

def per_participant(values: torch.Tensor, dataset: SpiceDataset) -> pd.Series:
    participant = dataset.xs[:, 0, 0, -1].long().numpy()
    return pd.Series(values.numpy(), index=participant).groupby(level=0).sum()


def participant_loglik(study: str, outputs_train: torch.Tensor, outputs_test: torch.Tensor,
                       train: SpiceDataset, test: SpiceDataset) -> pd.DataFrame:
    """Per participant (index): n_trials (test) and the held-out log-likelihood per trial (score)."""
    from weinhardt2026.utils.studies import continuous_session_sse, discrete_session_nll
    if test.continuous_action:
        sse_train, n_train, dim = continuous_session_sse(outputs_train, train, study)
        sse_test, n_test, _ = continuous_session_sse(outputs_test, test, study)
        variance = per_participant(sse_train, train) / (per_participant(n_train, train) * dim)
        sse, n = per_participant(sse_test, test), per_participant(n_test, test)
        variance = variance.reindex(sse.index)
        loglik = -0.5 * dim * np.log(2 * np.pi * variance) - sse / (2 * variance * n)
    else:
        nll, n_test = discrete_session_nll(outputs_test, test)
        nll, n = per_participant(nll, test), per_participant(n_test, test)
        loglik = -nll / n
    table = pd.DataFrame({'n_trials': n, 'score': loglik})
    table = table[table['n_trials'] > 0]
    table.index = table.index.map(lambda participant: f'participant {participant}')
    return table.assign(criterion='loglik')


def participant_information_criteria(study: str, outputs_train: torch.Tensor, train: SpiceDataset, k_per_participant) -> pd.DataFrame:
    """AIC and BIC on the training data per participant (group models: the pooled data).

    k_per_participant: participant ids -> parameter counts of those participants.
    """
    from weinhardt2026.utils.studies import continuous_session_sse
    pooled = study in GROUP_MODEL_STUDIES
    participant = train.xs[:, 0, 0, -1].long()
    if pooled:
        participant = torch.zeros_like(participant)

    def by_participant(values):
        return pd.Series(values.numpy(), index=participant.numpy()).groupby(level=0).sum()

    if train.continuous_action:
        sse, n, dim = continuous_session_sse(outputs_train, train, study)
        sse, n = by_participant(sse), by_participant(n)
        two_nll = n * dim * (1 + np.log(2 * np.pi * sse / (n * dim)))   # Gaussian with the participant's MLE variance
    else:
        from weinhardt2026.utils.studies import discrete_session_nll
        nll, n = discrete_session_nll(outputs_train, train)
        two_nll, n = 2 * by_participant(nll), by_participant(n)
    keep = n > 0
    two_nll, n = two_nll[keep], n[keep]
    k = pd.Series(np.asarray(k_per_participant(n.index.to_numpy()), dtype=float), index=n.index)

    units = ['all' if pooled else f'participant {p}' for p in n.index]
    frames = [pd.DataFrame({'n_trials': n.to_numpy(), 'score': (two_nll + 2 * k).to_numpy(), 'criterion': 'aic'}, index=units),
              pd.DataFrame({'n_trials': n.to_numpy(), 'score': (two_nll + k * np.log(n)).to_numpy(), 'criterion': 'bic'}, index=units)]
    return pd.concat(frames)


def score_study(study: str, data_dir: str, runs_dir: str) -> pd.DataFrame:
    """Per (generator, model, seed, criterion, unit) score of every fit of the study."""
    frames = []

    def add(table, generator, model, path):
        frames.append(table.assign(study=study, generator=generator, model=model, seed=seed_of(path)))

    for generator in GENERATORS:
        train, test = simulated_splits(study, generator, data_dir)

        for path in fit_paths(runs_dir, study, 'benchmark', generator):
            model = load_benchmark(study, path, train)
            outputs_train = predict(model, train)
            k = n_benchmark_parameters(study, model, train)
            add(participant_loglik(study, outputs_train, predict(model, test), train, test), generator, 'Benchmark', path)
            add(participant_information_criteria(study, outputs_train, train, lambda p: np.full(len(p), k)), generator, 'Benchmark', path)

        for path in fit_paths(runs_dir, study, 'spice', generator):
            member, n_coefficients = load_spice_member(study, path, train)
            for variant in SPICE_VARIANTS:
                member.eval(use_sindy=variant == 'SPICE-EQ')
                outputs_train = predict(member, train)
                add(participant_loglik(study, outputs_train, predict(member, test), train, test), generator, variant, path)
                if variant in INFORMATION_CRITERIA_MODELS:
                    add(participant_information_criteria(study, outputs_train, train, lambda p: n_coefficients[p].numpy()),
                        generator, variant, path)
            del member
            torch.cuda.empty_cache()
        print(f'{study}: scored the fits on the data of {generator}', flush=True)

    scores = pd.concat(frames).rename_axis('unit').reset_index()
    return scores[['study', 'generator', 'model', 'seed', 'criterion', 'unit', 'n_trials', 'score']]


# -------------------------------------------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------------------------------------------

def spice_wins(spice: pd.Series, benchmark: pd.Series, criterion: str) -> pd.Series:
    """Per unit: SPICE scores better than the benchmark (inputs: unit -> score)."""
    paired = pd.concat([spice.rename('spice'), benchmark.rename('benchmark')], axis=1, join='inner')
    return paired['spice'] > paired['benchmark'] if HIGHER_IS_BETTER[criterion] else paired['spice'] < paired['benchmark']


def confusion_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, criterion, generator), group in scores.groupby(['study', 'criterion', 'generator'], sort=False):
        benchmark = group[group['model'] == 'Benchmark']
        benchmark_mean = benchmark.groupby('unit')['score'].mean()
        benchmark_seeds = sorted(benchmark['seed'].unique())
        for variant in SPICE_VARIANTS:
            spice = group[group['model'] == variant]
            if spice.empty:
                continue
            spice_mean = spice.groupby('unit')['score'].mean()
            wins = spice_wins(spice_mean, benchmark_mean, criterion)
            difference = (spice_mean - benchmark_mean).dropna()

            # seed pairs: k-th SPICE seed vs. k-th benchmark seed (a single benchmark fit: with every SPICE seed)
            spice_seeds = sorted(spice['seed'].unique())
            pairs = zip(spice_seeds, benchmark_seeds if len(benchmark_seeds) > 1 else benchmark_seeds * len(spice_seeds))
            wins_pairs = [spice_wins(spice[spice['seed'] == s].set_index('unit')['score'],
                                     benchmark[benchmark['seed'] == b].set_index('unit')['score'], criterion).mean()
                          for s, b in pairs]
            rows.append(dict(
                study=study, criterion=criterion, spice=variant, generator=generator, n_units=len(wins),
                n_seeds_spice=len(spice_seeds), n_seeds_benchmark=len(benchmark_seeds),
                p_spice_wins=wins.mean(), p_spice_wins_seeds=np.mean(wins_pairs), p_spice_wins_seeds_sd=np.std(wins_pairs),
                score_difference=difference.mean(), score_difference_sd=difference.std(),
            ))
    matrix = pd.DataFrame(rows)

    # inversion: P(generator | winner), from the counts of both generators
    wins_spice = matrix['p_spice_wins'] * matrix['n_units']
    wins_benchmark = (1 - matrix['p_spice_wins']) * matrix['n_units']
    keys = [matrix[key] for key in ('study', 'criterion', 'spice')]
    matrix['p_generator_given_spice_wins'] = wins_spice / wins_spice.groupby(keys).transform('sum')
    matrix['p_generator_given_benchmark_wins'] = wins_benchmark / wins_benchmark.groupby(keys).transform('sum')
    return matrix


GENERATOR_LABELS = {'spice': 'SPICE-EQ', 'benchmark': 'Benchmark'}
CRITERION_LABELS = {'loglik': 'held-out log-likelihood per trial', 'aic': 'AIC', 'bic': 'BIC'}


def confusion_tables(matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per criterion: rows (study, simulating model), columns (comparison, fitted model), P(fitted model wins)
    with its SD over seed pairs ('p' / 'sd')."""
    tables = {}
    for criterion in CRITERIA:
        rows = matrix[matrix['criterion'] == criterion]
        if rows.empty:
            continue
        columns = {}
        for variant in SPICE_VARIANTS:
            pair = rows[rows['spice'] == variant]
            if pair.empty:
                continue
            index = pd.MultiIndex.from_arrays([pair['study'], pair['generator'].map(GENERATOR_LABELS)], names=['study', 'simulated'])
            comparison = f'{variant} vs. Benchmark'
            columns[(comparison, 'Benchmark', 'p')] = (1 - pair['p_spice_wins']).to_numpy()
            columns[(comparison, 'Benchmark', 'sd')] = pair['p_spice_wins_seeds_sd'].to_numpy()
            columns[(comparison, variant, 'p')] = pair['p_spice_wins'].to_numpy()
            columns[(comparison, variant, 'sd')] = pair['p_spice_wins_seeds_sd'].to_numpy()
        table = pd.DataFrame(columns, index=index)
        table.columns.names = ['comparison', 'fitted', 'value']
        tables[criterion] = table
    return tables


def print_tables(tables: dict[str, pd.DataFrame]) -> None:
    for criterion, table in tables.items():
        cells = table.xs('p', axis=1, level='value').map('{:.2f}'.format) + ' ± ' \
            + table.xs('sd', axis=1, level='value').map('{:.2f}'.format)
        print(f'\n{CRITERION_LABELS[criterion]}: P(fitted model wins | simulating model), per participant '
              f'(kolff2025 AIC/BIC: pooled data); ± SD over seed pairs')
        print(cells.to_string())


CMAP = 'viridis'   # as the confusion matrices of run_parameter_recovery.py (dark: P = 0, light: P = 1)
INK, INK_ON_DARK, INK_MUTED = '#222222', '#ffffff', '#6b6b6b'


def study_label(study: str) -> str:
    from weinhardt2026.utils.studies import STUDIES as REGISTRY
    return REGISTRY[study]['label'] if study in REGISTRY else {'kolff2025': 'Chimpanzee Grooming\n(Kolff 2025)'}.get(study, study)


def plot_confusion(table: pd.DataFrame, criterion: str, path: Path, clean: bool = False) -> None:
    """One 2x2 matrix per study and comparison: rows simulating model, columns fitted model (same order). With
    `clean`, also without values, titles and labels (<path>_clean)."""
    studies = list(dict.fromkeys(table.index.get_level_values('study')))
    comparisons = list(dict.fromkeys(table.columns.get_level_values('comparison')))
    for annotated in ((True, False) if clean else (True,)):
        fig, axes = plt.subplots(len(comparisons), len(studies), figsize=(1.95 * len(studies) + 1.0, 2.0 * len(comparisons) + 0.9),
                                 squeeze=False)

        for row, comparison in enumerate(comparisons):
            spice = comparison.split(' vs. ')[0]
            for col, study in enumerate(studies):
                ax = axes[row, col]
                cells = table.loc[study, comparison]
                simulated = ['SPICE-EQ', 'Benchmark']
                fitted = [spice, 'Benchmark']
                p = np.array([[cells.loc[s, (f, 'p')] for f in fitted] for s in simulated])
                sd = np.array([[cells.loc[s, (f, 'sd')] for f in fitted] for s in simulated])
                image = ax.imshow(p, cmap=CMAP, vmin=0, vmax=1)
                for i in range(2 if annotated else 0):
                    for j in range(2):
                        ax.text(j, i, f'{p[i, j]:.2f}\n±{sd[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                                color=INK if p[i, j] > 0.6 else INK_ON_DARK, linespacing=1.15)
                ax.set_xticks([0, 1], [label.replace('Benchmark', 'Bench.') for label in fitted], fontsize=7.5)
                ax.set_yticks([0, 1], [label.replace('Benchmark', 'Bench.') for label in simulated], fontsize=7.5)
                ax.tick_params(which='both', length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([0.5], minor=True)
                ax.set_yticks([0.5], minor=True)
                ax.grid(which='minor', color='white', linewidth=2)
                if col > 0:
                    ax.set_yticklabels([])
                if not annotated:
                    continue
                if row == 0:
                    ax.set_title(study_label(study), fontsize=8.5)
                if col == 0:
                    ax.set_ylabel(f'{comparison}\nsimulated by', fontsize=8)
                if row == len(comparisons) - 1:
                    ax.set_xlabel('fitted model', fontsize=8, color=INK_MUTED)

        colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.015)
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.outline.set_visible(False)
        if annotated:
            colorbar.set_label('P(fitted model wins)', fontsize=8)
            fig.suptitle(f'Model recovery by {CRITERION_LABELS[criterion]}: share of participants'
                         + (' (Chimpanzee Grooming: pooled group model, one decision per seed pair)' if criterion != 'loglik' else '')
                         + '; ± SD over seed pairs', fontsize=9, y=0.995)
        output = path if annotated else path.with_name(f'{path.name}_clean')
        for extension in ('png', 'pdf'):
            fig.savefig(f'{output}.{extension}', bbox_inches='tight', dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Confusion matrix of SPICE and the benchmark from the refits on simulated data.')
    parser.add_argument('--studies', nargs='+', default=list(STUDIES), choices=STUDIES)
    parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--runs_dir', default=DEFAULT_RUNS_DIR)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--cached', action='store_true', help='re-aggregate confusion_scores.csv instead of scoring')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_scores = out_dir / 'confusion_scores.csv'

    if args.cached:
        scores = pd.read_csv(path_scores)
        scores = scores[scores['study'].isin(args.studies)]
    else:
        scores = pd.concat([score_study(study, args.data_dir, args.runs_dir) for study in args.studies], ignore_index=True)
        scores.to_csv(path_scores, index=False)

    matrix = confusion_matrix(scores)
    matrix.to_csv(out_dir / 'confusion_matrix.csv', index=False)
    tables = confusion_tables(matrix)
    for criterion, table in tables.items():
        table.to_csv(out_dir / f'confusion_matrix_{criterion}.csv')
        plot_confusion(table, criterion, out_dir / f'figure_confusion_matrix_{criterion}', clean=True)
    print_tables(tables)
    print(f'\nSaved to {out_dir}')

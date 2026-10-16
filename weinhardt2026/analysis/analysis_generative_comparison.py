"""
Generative comparison: simulate behavior from every fitted model in the study's task and compare
it with the real participants' behavior.

The study-specific parts live in ``weinhardt2026/studies/<study>/analysis_generative.py``, which
defines:
    METRIC_LABELS              {metric: axis label}
    N_REPEATS                  simulated runs of the experiment per model
    load_dataset()             full dataset (all blocks), built as the models were trained on it
    load_models(dataset)       {'GRU': [...], 'Benchmark': [...], 'SPICE': [EnsembleMember, ...]}: one model per
                               final seed (weinhardt2026/utils/final_models.py); for SPICE the ensemble member
                               with the highest training likelihood (select_spice_members; table in
                               results/generative_spice_selection.csv)
    generate(model, dataset)   one simulated run of `dataset` -> SpiceDataset
    compute_metrics(dataset)   {metric: (n_sessions,) array}

Every model (instance, i.e. seed) is simulated N_REPEATS times. A single fit (e.g. kolff2025's closed-form
benchmark) is simulated as often as the model with the most instances, N_REPEATS runs each. Scores are
computed per instance and averaged over instances.

Per study, written to ``weinhardt2026/studies/<study>/results/``:
    generative_behavior.png/.pdf     one panel per metric; violins in MODEL_ORDER, participants as
                                     dots joined across Benchmark - Real - SPICE-EQ
    generative_metrics.csv           per (model, instance, participant, run) metric values
    generative_scores.csv            per (model, metric) and across metrics ('all'), mean and SD over instances:
                                       distance: Wasserstein distance between the simulated and the
                                                 real participant distribution / SD(real), lower is better
                                       rho:      Spearman correlation across participants between the
                                                 real value and the instance's mean over runs,
                                                 i.e. how well individual differences are recovered
    generative_scores_instances.csv  the same per instance
    generative_spice_selection.csv   training likelihood of every (seed, member); selected = best per seed
    behavioral_metrics_real.csv      per-participant real metrics (read by behavioral clustering)

The cross-study summary (several studies) is weinhardt2026/results/generative_scores_all_studies.csv,
including one number per model across studies (study = 'all': mean of the per-study scores).

Usage:
    python weinhardt2026/analysis/analysis_generative_comparison.py dezfouli2019
    python weinhardt2026/analysis/analysis_generative_comparison.py dezfouli2019 kolff2025 --n_repeats 10
    python weinhardt2026/analysis/analysis_generative_comparison.py all --cached
"""

import argparse
import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules import 'studies.<study>...' (as in run.py)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import wasserstein_distance, spearmanr

from spice import SpiceDataset


STUDIES = (
    'dezfouli2019', 'ganesh2024a', 'eckstein2026', 'bustamante2023',
    'braun2018', 'kolff2025', 'bruckner2025', 'weber2024',
)

MODEL_ORDER = ['GRU', 'Benchmark', 'Real', 'SPICE-EQ', 'SPICE-RNN']
SIMULATED_MODELS = [model for model in MODEL_ORDER if model != 'Real']
PAIRED_MODELS = ['Benchmark', 'Real', 'SPICE-EQ']   # participants drawn as dots joined by lines

# model colors as in figures/figure_model_evaluation.py; Real is neutral ink
COLORS = {
    'GRU':       '#5b8db8',
    'Benchmark': '#1a3a5c',
    'Real':      '#6b6b6b',
    'SPICE-EQ':  '#8b2529',
    'SPICE-RNN': '#c44e52',
}

# sessions simulated in one batch (replications are tiled along the session dimension)
MAX_SESSIONS_PER_BATCH = 20000


def study_dir(study: str, *parts) -> str:
    return os.path.join(ROOT, 'weinhardt2026', 'studies', study, *parts)


def load_study_module(study: str):
    return importlib.import_module(f'weinhardt2026.studies.{study}.analysis_generative')


def load_registered_dataset(study: str) -> SpiceDataset:
    """Full dataset as figures/figure_model_evaluation.py (and run.py) builds it."""
    from weinhardt2026.figures.figure_model_evaluation import load_dataset
    return load_dataset(study)[1]


def load_registered_models(study: str, dataset: SpiceDataset) -> dict:
    """Every final seed of GRU, benchmark and SPICE (best member per seed), built as figures/figure_model_evaluation.py registers them."""
    from weinhardt2026.figures.figure_model_evaluation import STUDIES as REGISTRY, load_study, load_spice
    from weinhardt2026.utils.final_models import final_checkpoints
    config = REGISTRY[study]
    n_participants, n_actions = dataset.n_participants, dataset.n_actions
    spice_module, dataset_train, _ = load_study(study)
    return {
        'GRU': [config['gru'](path, n_participants, n_actions) for path in final_checkpoints(study, 'gru')],
        'Benchmark': [config['benchmark'](path, n_participants, n_actions) for path in final_checkpoints(study, 'benchmark')],
        'SPICE': select_spice_members(
            study, final_checkpoints(study, 'spice'), lambda path: load_spice(path, spice_module, dataset_train), dataset_train,
        ),
    }


# -------------------------------------------------------------------------------------------
# SPICE MODEL SELECTION: PER SEED, THE ENSEMBLE MEMBER WITH THE HIGHEST TRAINING LIKELIHOOD
# -------------------------------------------------------------------------------------------

class EnsembleMember:
    """One ensemble member of a SpiceEstimator, called like the estimator (outputs keep a member axis of 1)."""

    def __init__(self, estimator, member: int):
        self.estimator = estimator
        self.member = member

    @property
    def device(self):
        return self.estimator.device

    def eval(self, use_sindy: bool = None) -> 'EnsembleMember':
        """Without `use_sindy` the current mode is kept (generation wrappers call eval() on their model)."""
        if use_sindy is not None:
            self.estimator.eval(use_sindy=use_sindy)
        return self

    def __call__(self, xs: torch.Tensor, state=None):
        outputs, state = self.estimator(xs, state)
        return outputs[self.member:self.member + 1], state


def trial_loglik(study: str, outputs: torch.Tensor, dataset: SpiceDataset) -> float:
    """Log-likelihood per trial of one model's outputs (B, T, W, A) on `dataset`.

    Discrete: mean log-probability of the chosen actions. Continuous: Gaussian log-likelihood with the
    model's MLE noise variance (likelihood definitions as in figures/figure_model_evaluation.py).
    """
    from weinhardt2026.figures.figure_model_evaluation import discrete_session_nll, continuous_session_sse
    if dataset.continuous_action:
        sse, n_trials, dim = continuous_session_sse(outputs, dataset, study)
        variance = sse.sum().item() / (n_trials.sum().item() * dim)
        return -0.5 * dim * (np.log(2 * np.pi * variance) + 1)
    nll, n_trials = discrete_session_nll(outputs, dataset)
    return -nll.sum().item() / n_trials.sum().item()


@torch.no_grad()
def member_train_loglik(study: str, estimator, dataset_train: SpiceDataset) -> np.ndarray:
    """Training log-likelihood per trial of every ensemble member's SPICE-EQ."""
    estimator.eval(use_sindy=True)
    estimator.model.init_state(batch_size=dataset_train.xs.shape[0])
    outputs, _ = estimator(dataset_train.xs.to(estimator.device))
    outputs = outputs.detach().cpu()   # (E, B, T, W, A)
    return np.array([trial_loglik(study, outputs_member, dataset_train) for outputs_member in outputs])


def model_train_loglik(study: str, model: torch.nn.Module, dataset_train: SpiceDataset) -> float:
    """Training log-likelihood per trial of a benchmark or GRU."""
    from weinhardt2026.figures.figure_model_evaluation import predict
    return trial_loglik(study, predict(model, dataset_train), dataset_train)


def select_spice_members(study: str, checkpoints: list[str], load, dataset_train: SpiceDataset) -> list[EnsembleMember]:
    """Per checkpoint (seed), the ensemble member with the highest training likelihood.

    Args:
        checkpoints: Paths of the SPICE checkpoints (one per seed).
        load: path -> SpiceEstimator.
        dataset_train: The data the checkpoints were trained on (likelihood is computed on it).

    Writes every (checkpoint, member) likelihood to results/generative_spice_selection.csv.
    """
    if not checkpoints:
        raise FileNotFoundError(f'{study}: no SPICE checkpoints found')
    members, rows = [], []
    for path in checkpoints:
        estimator = load(path)
        loglik = member_train_loglik(study, estimator, dataset_train)
        best = int(np.argmax(loglik))
        members.append(EnsembleMember(estimator, best))
        for member, value in enumerate(loglik):
            rows.append(dict(checkpoint=os.path.relpath(path, ROOT), member=member, loglik_train=value, selected=member == best))
        print(f'SPICE: {os.path.basename(path)}, member {best} (training log-lik. / trial {loglik[best]:.4f})')
    pd.DataFrame(rows).to_csv(study_dir(study, 'results', 'generative_spice_selection.csv'), index=False)
    return members


# -------------------------------------------------------------------------------------------
# METRIC HELPERS (vectorized over sessions; arrays are (n_sessions, n_trials), NaN = padded)
# -------------------------------------------------------------------------------------------

def valid_trials(dataset: SpiceDataset) -> np.ndarray:
    return ~torch.isnan(dataset.xs[:, :, 0, 0]).numpy()


def choices_and_rewards(dataset: SpiceDataset) -> tuple[np.ndarray, np.ndarray]:
    """Chosen action index and the scalar reward (sum over the reward columns), NaN where padded."""
    xs = dataset.xs[:, :, 0]
    n_actions = dataset.n_actions
    valid = valid_trials(dataset)
    choices = xs[..., :n_actions].argmax(dim=-1).float().numpy()
    rewards = xs[..., n_actions:n_actions + dataset.n_reward_features].nan_to_num(0.).sum(dim=-1).numpy()
    choices[~valid] = np.nan
    rewards[~valid] = np.nan
    return choices, rewards


def session_mean(values: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Per-session mean of `values` over the trials in `mask` (and non-NaN); NaN if there are none."""
    keep = ~np.isnan(values) if mask is None else mask & ~np.isnan(values)
    n = keep.sum(axis=1)
    total = np.where(keep, values, 0.).sum(axis=1)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def stays(choices: np.ndarray) -> np.ndarray:
    """(n_sessions, n_trials - 1): 1 if trial t+1 repeats the choice of trial t, NaN if either is padded."""
    stay = (choices[:, 1:] == choices[:, :-1]).astype(float)
    stay[np.isnan(choices[:, 1:]) | np.isnan(choices[:, :-1])] = np.nan
    return stay


def choice_entropy(choices: np.ndarray, n_actions: int) -> np.ndarray:
    """Entropy (bits) of the per-session choice frequencies."""
    entropy = np.zeros(choices.shape[0])
    for action in range(n_actions):
        p = session_mean((choices == action).astype(float), ~np.isnan(choices))
        entropy -= np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1.)), 0.)
    entropy[np.isnan(choices).all(axis=1)] = np.nan
    return entropy


def quarter_masks(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of the first and the last quarter of each session's valid trials (sessions with < 4 trials: empty)."""
    n_valid = valid.sum(axis=1, keepdims=True)
    quarter = n_valid // 4
    t = np.arange(valid.shape[1])[None, :]
    return valid & (t < quarter), valid & (t >= n_valid - quarter) & (quarter > 0)


# -------------------------------------------------------------------------------------------
# SIMULATION
# -------------------------------------------------------------------------------------------

def tile_dataset(dataset: SpiceDataset, n_repeats: int) -> SpiceDataset:
    """`n_repeats` copies of every session, replication-major; metadata (block, participant) unchanged."""
    return SpiceDataset(
        dataset.xs.repeat(n_repeats, 1, 1, 1),
        dataset.ys.repeat(n_repeats, 1, 1, 1),
        n_reward_features=dataset.n_reward_features,
        continuous_action=dataset.continuous_action,
    )


def simulate(generate, model, dataset: SpiceDataset, n_repeats: int):
    """Yield (simulated dataset, replication index per session), several replications per batch."""
    n_sessions = dataset.xs.shape[0]
    repeats_per_batch = max(1, MAX_SESSIONS_PER_BATCH // n_sessions)
    for first in range(0, n_repeats, repeats_per_batch):
        n_batch = min(repeats_per_batch, n_repeats - first)
        dataset_sim = generate(model, tile_dataset(dataset, n_batch))
        yield dataset_sim, np.repeat(np.arange(first, first + n_batch), n_sessions)


def participant_table(metrics: dict, dataset: SpiceDataset, model: str, instance: int, repeats: np.ndarray) -> pd.DataFrame:
    """Session metrics -> one row per (participant, run): mean over the participant's sessions."""
    df = pd.DataFrame({name: np.asarray(values, dtype=float) for name, values in metrics.items()})
    df.insert(0, 'repeat', repeats)
    df.insert(0, 'participant_id', dataset.xs[:, 0, 0, -1].long().numpy())
    df = df.groupby(['participant_id', 'repeat'], as_index=False).mean()
    df.insert(0, 'instance', instance)
    df.insert(0, 'model', model)
    return df


def simulate_metrics(study_module, dataset: SpiceDataset, models: dict, n_repeats: int) -> pd.DataFrame:
    """Per (model, instance, participant, run) metrics for the real data and every simulated model.

    Every model may be a list (one model per seed); every list entry is an instance. A single fit is
    simulated as many times as the model with the most instances has instances (with other random runs).
    SPICE-EQ and SPICE-RNN are both simulated from models['SPICE'].
    """
    n_sessions = dataset.xs.shape[0]
    fits = {name: models[name] if isinstance(models[name], (list, tuple)) else None for name in ('GRU', 'Benchmark', 'SPICE')}
    n_instances = max([len(instances) for instances in fits.values() if instances is not None], default=1)
    frames = [participant_table(study_module.compute_metrics(dataset), dataset, 'Real', 0, np.zeros(n_sessions, dtype=int))]
    for name in SIMULATED_MODELS:
        key = 'SPICE' if name.startswith('SPICE') else name
        instances = fits[key] if fits[key] is not None else [models[key]] * n_instances
        print(f'Simulating {name} ({len(instances)} models x {n_repeats} runs of {n_sessions} sessions)...')
        for instance, model in enumerate(instances):
            model.eval(use_sindy=name == 'SPICE-EQ') if name.startswith('SPICE') else model.eval()
            torch.manual_seed(instance)
            for dataset_sim, repeats in simulate(study_module.generate, model, dataset, n_repeats):
                frames.append(participant_table(study_module.compute_metrics(dataset_sim), dataset_sim, name, instance, repeats))
    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------------------------------------------
# SCORES
# -------------------------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y)[0])


def _score_instance(real: pd.DataFrame, simulated: pd.DataFrame, metric_names: list[str]) -> list[dict]:
    """distance = W1(simulated participant values over all runs, real participant values) / SD(real);
    rho = Spearman(real participant value, participant mean over runs); 'all' = mean over the metrics."""
    simulated_mean = simulated.groupby('participant_id')[metric_names].mean()
    rows = []
    for metric in metric_names:
        values_real = real[metric].dropna().to_numpy()
        values_sim = simulated[metric].dropna().to_numpy()
        sd_real = values_real.std(ddof=1) if len(values_real) > 1 else 0.
        distance = wasserstein_distance(values_real, values_sim) / sd_real \
            if sd_real > 0 and len(values_sim) > 0 else np.nan
        paired = pd.concat([real[metric], simulated_mean[metric]], axis=1, join='inner').dropna().to_numpy()
        rows.append(dict(metric=metric, distance=distance, rho=_spearman(paired[:, 0], paired[:, 1])))
    rows.append(dict(metric='all', distance=np.nanmean([r['distance'] for r in rows]), rho=np.nanmean([r['rho'] for r in rows])))
    return rows


def score_models(df: pd.DataFrame, metric_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores per (model, instance, metric), and their mean and SD over a model's instances.

    Returns:
        scores: model, metric, distance, distance_sd, rho, rho_sd ('all' = across metrics).
        scores_instances: model, instance, metric, distance, rho.
    """
    real = df[df['model'] == 'Real'].set_index('participant_id')[metric_names]
    rows = []
    for model in SIMULATED_MODELS:
        for instance, simulated in df[df['model'] == model].groupby('instance'):
            rows += [dict(model=model, instance=instance, **row) for row in _score_instance(real, simulated, metric_names)]
    scores_instances = pd.DataFrame(rows)
    scores = scores_instances.groupby(['model', 'metric'], sort=False).agg(
        distance=('distance', 'mean'), distance_sd=('distance', 'std'),
        rho=('rho', 'mean'), rho_sd=('rho', 'std'),
    ).reset_index()
    return scores, scores_instances


# -------------------------------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------------------------------

def _format(value: float) -> str:
    return '–' if np.isnan(value) else f'{value:.2f}'


def plot_generative_comparison(df: pd.DataFrame, scores: pd.DataFrame, metric_labels: dict, title: str, path: str):
    """One panel per metric. Violins over participants x models x runs in MODEL_ORDER; for Benchmark,
    Real and SPICE-EQ every participant is a dot (models: mean over models and runs) and the dots of
    one participant are joined. Above each violin: distance (d) and, for paired models, rho, both
    averaged over the models (instances)."""
    metric_names = list(metric_labels)
    models = [model for model in MODEL_ORDER if (df['model'] == model).any()]
    positions = {model: index for index, model in enumerate(models)}
    score = scores.set_index(['model', 'metric'])

    participant_means = {
        model: df[df['model'] == model].groupby('participant_id')[metric_names].mean()
        for model in PAIRED_MODELS if model in positions
    }
    participants = participant_means['Real'].index
    jitter = pd.Series(np.random.default_rng(0).uniform(-0.12, 0.12, len(participants)), index=participants)

    n_cols = min(4, len(metric_names))
    n_rows = int(np.ceil(len(metric_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.4 * n_rows + 0.9), squeeze=False)

    for ax, metric in zip(axes.flat, metric_names):
        for model in models:
            values = df.loc[df['model'] == model, metric].dropna().to_numpy()
            if len(values) == 0:
                continue
            x = positions[model]
            if np.ptp(values) > 0:
                body = ax.violinplot(values, positions=[x], widths=0.8, showextrema=False)['bodies'][0]
                body.set_facecolor(COLORS[model])
                body.set_edgecolor('none')
                body.set_alpha(0.3)
            ax.hlines(np.median(values), x - 0.3, x + 0.3, color=COLORS[model], linewidth=2, zorder=4)

        # the same participant across Benchmark - Real - SPICE-EQ
        paired = pd.DataFrame({model: means[metric] for model, means in participant_means.items()}).dropna()
        x_paired = np.array([positions[model] for model in paired.columns])
        offsets = jitter.loc[paired.index].to_numpy()[:, None]
        ax.plot((x_paired[None, :] + offsets).T, paired.to_numpy().T, color='#9a9a9a', linewidth=0.4, alpha=0.35, zorder=2)
        for column, model in enumerate(paired.columns):
            ax.scatter(x_paired[column] + offsets[:, 0], paired[model], s=7, color=COLORS[model],
                       alpha=0.75, linewidths=0, zorder=3)

        # scores above each simulated model
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for model in models:
            if model == 'Real':
                continue
            text = f"d {_format(score.loc[(model, metric), 'distance'])}"
            if model in PAIRED_MODELS:
                text += f"\nρ {_format(score.loc[(model, metric), 'rho'])}"
            ax.text(positions[model], 0.985, text, transform=transform, ha='center', va='top', fontsize=7, color='#333333')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.22 * (top - bottom))

        ax.set_title(metric_labels[metric], fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', color='#e6e6e6', linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in axes.flat[len(metric_names):]:
        ax.set_visible(False)

    overall = score.xs('all', level='metric')
    summary = '   '.join(
        f"{model}: d {_format(overall.loc[model, 'distance'])}"
        + (f", ρ {_format(overall.loc[model, 'rho'])}" if model in PAIRED_MODELS else '')
        for model in models if model in overall.index
    )
    fig.suptitle(f'{title}\nacross metrics — {summary}', fontsize=11)
    fig.text(
        0.5, 0.005,
        'Violins: participants × models × runs. Dots: participants (models: mean over models and runs), '
        'lines join the same participant.\nd: Wasserstein distance to Real / SD(Real), lower is better. '
        'ρ: Spearman correlation with Real across participants (individual differences). d and ρ: mean over models.',
        ha='center', va='bottom', fontsize=7.5, color='#555555',
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(f'{path}.{extension}', bbox_inches='tight', dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------------------------------

def run_generative_analysis(study: str, models: dict = None, n_repeats: int = None, cached: bool = False) -> pd.DataFrame:
    """Simulate, score and plot one study; see the module docstring for the outputs.

    Args:
        study: Study directory name, e.g. 'dezfouli2019'.
        models: {'GRU', 'Benchmark', 'SPICE'} fitted models (each one model or a list, one per seed);
            loaded by the study module if None.
        n_repeats: Simulated runs per model; the study's N_REPEATS if None.
        cached: Re-score and re-plot the existing generative_metrics.csv instead of simulating.

    Returns:
        Scores per (model, metric), mean and SD over the model instances, including the across-metric row 'all'.
    """
    study_module = load_study_module(study)
    metric_names = list(study_module.METRIC_LABELS)
    output_dir = study_dir(study, 'results')
    os.makedirs(output_dir, exist_ok=True)
    path_metrics = os.path.join(output_dir, 'generative_metrics.csv')

    if cached:
        df = pd.read_csv(path_metrics)
    else:
        dataset = study_module.load_dataset()
        if models is None:
            models = study_module.load_models(dataset)
        df = simulate_metrics(study_module, dataset, models, n_repeats or study_module.N_REPEATS)
        df.to_csv(path_metrics, index=False)
        df[df['model'] == 'Real'].drop(columns=['model', 'instance', 'repeat']).to_csv(
            os.path.join(output_dir, 'behavioral_metrics_real.csv'), index=False)

    scores, scores_instances = score_models(df, metric_names)
    scores.insert(0, 'study', study)
    scores.to_csv(os.path.join(output_dir, 'generative_scores.csv'), index=False)
    scores_instances.to_csv(os.path.join(output_dir, 'generative_scores_instances.csv'), index=False)

    plot_generative_comparison(df, scores, study_module.METRIC_LABELS, study, os.path.join(output_dir, 'generative_behavior'))

    print(f'\n{study}: distance (lower is better)')
    print(scores.pivot(index='model', columns='metric', values='distance').round(3).to_string())
    print(f'\n{study}: rho (individual differences)')
    print(scores.pivot(index='model', columns='metric', values='rho').round(3).to_string())
    print(f'Saved to {output_dir}')
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative comparison of real and model-simulated behavior.')
    parser.add_argument('studies', nargs='+', help=f"study names or 'all' ({', '.join(STUDIES)})")
    parser.add_argument('--n_repeats', type=int, default=None, help="simulated runs per model (default: the study's N_REPEATS)")
    parser.add_argument('--cached', action='store_true', help='re-score and re-plot generative_metrics.csv without simulating')
    args = parser.parse_args()

    studies = STUDIES if args.studies == ['all'] else args.studies
    all_scores = [run_generative_analysis(study, n_repeats=args.n_repeats, cached=args.cached) for study in studies]

    if len(all_scores) > 1:
        summary = pd.concat(all_scores)
        summary = summary[summary['metric'] == 'all'].drop(columns='metric')
        # one number per model across studies: mean of the per-study across-metric scores
        summary = summary.drop(columns=['distance_sd', 'rho_sd'])
        overall = summary.groupby('model', as_index=False)[['distance', 'rho']].mean()
        overall.insert(0, 'study', 'all')
        summary = pd.concat([summary, overall], ignore_index=True)
        path_summary = os.path.join(ROOT, 'weinhardt2026', 'results', 'generative_scores_all_studies.csv')
        os.makedirs(os.path.dirname(path_summary), exist_ok=True)
        summary.to_csv(path_summary, index=False)
        print('\nAcross metrics, all studies:')
        print(summary.pivot(index='study', columns='model', values=['distance', 'rho']).round(3).to_string())
        print(f'Saved to {path_summary}')

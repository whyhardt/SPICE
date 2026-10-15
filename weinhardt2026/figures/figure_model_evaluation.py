"""
Model evaluation figure: one column per study, rows = held-out likelihood, AIC, BIC.

Models per study:
  - Benchmark: the study's hand-crafted cognitive model
  - GRU:       a GRU without participant embedding
  - SPICE-RNN / SPICE-EQ: every seed of the study's grid-search checkpoints
    (bars = mean over seeds, dots = single seeds)

Data and SPICE estimators are rebuilt exactly as ``slurm_jobs/grid_search.sh`` →
``weinhardt2026/run.py`` built them, so all models are scored on the split the
SPICE checkpoints were trained on.

Metrics:
  - Held-out likelihood (test blocks):
      discrete   → average trial likelihood exp(-NLL / n_trials)
      continuous → Gaussian log-likelihood per trial; the noise variance is the
                   model's MLE on the training residuals
  - AIC / BIC (training data, mean over (participant, experiment) groups, see
    ``analysis_model_evaluation.grouped_information_criteria``) — only for the
    interpretable models (Benchmark, SPICE-EQ). Continuous studies use a Gaussian
    likelihood with per-group MLE variance.

Usage:
    python weinhardt2026/figures/figure_model_evaluation.py            # evaluate (cached) + plot
    python weinhardt2026/figures/figure_model_evaluation.py --recompute
"""

import argparse
import importlib
import json
import math
import os
import sys
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules are imported as 'studies.<study>...' (as in run.py)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice.resources.spice_training import cross_entropy_loss
from weinhardt2026.analysis.analysis_model_evaluation import (
    get_participant_experiment_groups,
    grouped_information_criteria,
)
from weinhardt2026.utils.benchmarking_gru import GRUModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-9
OUTPUT_DIR = os.path.join(ROOT, 'weinhardt2026', 'figures', 'model_evaluation')

GRID_DEFAULT = 'th0.05_sw0_al0.0005_er0.5_gp0.01_fp0.0001'
GRID_CONTINUOUS = 'th0.05_sw0_al0.0005_er0.5_gp0_fp0'


# ── Study configuration ─────────────────────────────────────────────
# data_kwargs / test_blocks mirror slurm_jobs/grid_search.sh

def _benchmark(module_name, class_name, params_file, **kwargs):
    def build(n_participants, n_actions):
        module = importlib.import_module(module_name)
        model = getattr(module, class_name)(n_participants=n_participants, **kwargs)
        model.load_state_dict(torch.load(params_file, map_location='cpu'))
        return model
    return build


def _gru(params_file, **kwargs):
    def build(n_participants, n_actions):
        model = GRUModel(**kwargs)
        model.load_state_dict(torch.load(params_file, map_location='cpu'))
        return model
    return build


def _study_path(study, *parts):
    return os.path.join(ROOT, 'weinhardt2026', 'studies', study, *parts)


def _grid_checkpoints(study, grid):
    return sorted(glob(_study_path(study, 'params', 'params', f'spice_{study}_{grid}_rs[0-9]*.pkl')))


STUDIES = {
    'dezfouli2019': dict(
        label='Two-Armed Bandit\n(Dezfouli 2019)',
        test_blocks=(3, 6, 9),
        data_kwargs={},
        checkpoints=lambda: _grid_checkpoints('dezfouli2019', GRID_DEFAULT),
        benchmark=_benchmark('weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019', 'GQLModel',
                             _study_path('dezfouli2019', 'params', 'benchmark_dezfouli2019.pkl'), batch_first=True),
        gru=_gru(_study_path('dezfouli2019', 'params', 'gru_dezfouli2019.pkl'),
                 n_actions=2, additional_inputs=2, dropout=0.25, hidden_size=8),
    ),
    'ganesh2024a': dict(
        label='Perceptual Bandit\n(Ganesh 2024)',
        test_blocks=(3, 6, 9),
        data_kwargs={"df_participant_id": "subjID", "df_choice": "choice", "df_feedback": "reward",
                     "df_block": "blocks", "additional_inputs": ["contrast_difference"]},
        # params/params holds the old architecture -> single checkpoint of the current one
        checkpoints=lambda: [_study_path('ganesh2024a', 'params', f'spice_ganesh2024a_{GRID_DEFAULT}_emb8_rs0.pkl')],
        benchmark=_benchmark('weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a', 'BayesianModel',
                             _study_path('ganesh2024a', 'params', 'bay_ganesh2024a.pkl'), batch_first=True),
        gru=_gru(_study_path('ganesh2024a', 'params', 'gru_ganesh2024a.pkl'),
                 n_actions=2, additional_inputs=2, dropout=0.25, hidden_size=8),
    ),
    'eckstein2026': dict(
        label='Four-Armed Bandit\n(Eckstein 2026)',
        test_blocks=(2,),
        data_kwargs={},
        checkpoints=lambda: _grid_checkpoints('eckstein2026', GRID_DEFAULT),
        benchmark=_benchmark('weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026', 'BestRLModel',
                             _study_path('eckstein2026', 'params', 'bestrl_eckstein2026_cont.pkl'),
                             n_actions=4, batch_first=True),
        gru=_gru(_study_path('eckstein2026', 'params', 'gru_eckstein2026.pkl'),
                 n_actions=4, additional_inputs=2, dropout=0.1, hidden_size=32),
    ),
    'bustamante2023': dict(
        label='Patch Foraging\n(Bustamante 2023)',
        test_blocks=(3, 6),
        data_kwargs={"df_participant_id": "subject_id", "df_choice": "decision", "df_feedback": "reward",
                     "df_block": "overall_round", "additional_inputs": ["harvest_duration", "travel_duration", "is_msit", "is_high"]},
        checkpoints=lambda: _grid_checkpoints('bustamante2023', GRID_DEFAULT),
        benchmark=_benchmark('weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023',
                             'EffortCostMVTModel', _study_path('bustamante2023', 'params', 'effort_mvt_bustamante2023.pkl'),
                             batch_first=True),
        gru=_gru(_study_path('bustamante2023', 'params', 'gru_bustamante2023.pkl'), n_actions=2),
    ),
    'braun2018': dict(
        label='Task Switching\n(Braun 2018)',
        test_blocks=(3, 6, 9),
        data_kwargs={"df_participant_id": "subject", "df_choice": "transcode", "df_feedback": None,
                     "df_block": "block", "additional_inputs": ["difference", "current", "other"],
                     "timeshift_additional_inputs": [-1, -1, -1]},
        checkpoints=lambda: _grid_checkpoints('braun2018', GRID_DEFAULT),
        benchmark=_benchmark('weinhardt2026.studies.braun2018.benchmarking_braun2018', 'ExpectedValueControl',
                             _study_path('braun2018', 'params', 'benchmark_braun2018.pkl')),
        gru=_gru(_study_path('braun2018', 'params', 'gru_braun2018.pkl'),
                 n_actions=2, additional_inputs=4, n_reward_features=0),
    ),
    'bruckner2025': dict(
        label='Predictive Inference\n(Bruckner 2025)',
        test_blocks=(2, 4),
        data_kwargs={"df_choice": "b_t", "df_feedback": "x_t",
                     "additional_inputs": ["z_next", "catch", "v_t", "sigma", "r_t", "mu_t", "c_t"],
                     "continuous_action": True, "timeshift_additional_inputs": [0, 0, 0, 0, 0, 0, 0]},
        checkpoints=lambda: _grid_checkpoints('bruckner2025', GRID_CONTINUOUS),
        benchmark=_benchmark('weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025', 'RationalResourceModel',
                             _study_path('bruckner2025', 'params', 'benchmark_bruckner2025.pkl')),
        gru=_gru(_study_path('bruckner2025', 'params', 'gru_bruckner2025.pkl'),
                 n_actions=1, additional_inputs=6, n_reward_features=1),
    ),
    'weber2024': dict(
        label='Circular Predictive\nInference (Weber 2024)',
        test_blocks=(3, 5, 9, 12),
        data_kwargs={"df_participant_id": "participant", "df_experiment_id": "experiment",
                     "df_choice": ["shield_sin", "shield_cos"], "df_feedback": ["laser_sin", "laser_cos"],
                     "df_block": "block",
                     "additional_inputs": ["laser_caught", "volatility", "stochasticity", "trial_duration_frames", "trueMean"],
                     "continuous_action": True},
        checkpoints=lambda: _grid_checkpoints('weber2024', GRID_CONTINUOUS),
        benchmark=_benchmark('weinhardt2026.studies.weber2024.benchmarking_weber2024', 'ChangePointModel',
                             _study_path('weber2024', 'params', 'benchmark_weber2024.pkl'),
                             h_step=2),
        gru=_gru(_study_path('weber2024', 'params', 'gru_weber2024_continuous.pkl'),
                 n_actions=2, additional_inputs=4, n_reward_features=2,
                 additional_inputs_scale=(1., 1., 1., 90.)),
    ),
}


# ── Data + model loading ────────────────────────────────────────────

def load_dataset(study):
    """Rebuild the full (unsplit) data as run.py does, and the study's SPICE module."""
    spice_module = importlib.import_module(f'studies.{study}.spice_{study}')

    data = _study_path(study, 'data', f'{study}.csv')
    if hasattr(spice_module, 'prepare_dataframe'):
        data = spice_module.prepare_dataframe(pd.read_csv(data))
    dataset = csv_to_dataset(file=data, **STUDIES[study]['data_kwargs'])
    if getattr(spice_module, 'NORMALIZE_REWARDS', True):
        dataset.normalize_rewards()
    if hasattr(spice_module, 'prepare_dataset'):
        dataset = spice_module.prepare_dataset(dataset)
    return spice_module, dataset


def load_study(study):
    """Rebuild train/test data as run.py does, and the study's SPICE module."""
    spice_module, dataset = load_dataset(study)
    dataset_train, dataset_test = split_data_along_blockdim(dataset, list(STUDIES[study]['test_blocks']))
    return spice_module, dataset_train, dataset_test


def load_spice(path, spice_module, dataset_train):
    n_actions = dataset_train.ys.shape[-1]
    estimator_kwargs = dict(
        n_actions=n_actions,
        n_items=n_actions,
        n_experiments=len(dataset_train.xs[..., -2].unique()),
        n_reward_features=dataset_train.n_reward_features,
        loss_fn=getattr(spice_module, 'LOSS_FN', cross_entropy_loss),
    )
    estimator_kwargs.update(getattr(spice_module, 'ESTIMATOR_KWARGS', {}))
    estimator = SpiceEstimator(
        spice_class=spice_module.SpiceModel,
        spice_config=spice_module.CONFIG,
        n_participants=len(dataset_train.xs[..., -1].unique()),
        embedding_size=8,
        compiled_forward=False,
        device=DEVICE,
        **estimator_kwargs,
    )
    estimator.load_spice(path)
    return estimator


def n_benchmark_parameters(model, n_participants):
    if hasattr(model, 'count_parameters'):
        return float(model.count_parameters())
    # per-participant nn.Parameters of shape (n_participants, ...)
    return sum(p.numel() for p in model.parameters()) / n_participants


# ── Predictions ─────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, dataset, use_sindy=None):
    """Raw model outputs (B, T, W, A) on CPU. ``use_sindy`` only for SPICE estimators."""
    if use_sindy is None:
        model.eval()
        outputs, _ = model.to('cpu')(dataset.xs)
    else:
        model.eval(use_sindy=use_sindy)
        model.model.init_state(batch_size=dataset.xs.shape[0])
        outputs, _ = model(dataset.xs.to(model.device))
        outputs = outputs.mean(dim=0)  # ensemble average
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs.detach().cpu()


def valid_trials(dataset):
    return ~torch.isnan(dataset.xs[:, :, 0, 0]) & ~torch.isnan(dataset.ys[:, :, 0, 0])


# ── Per-session statistics ──────────────────────────────────────────

def discrete_session_nll(logits, dataset):
    """Per-session NLL and trial count."""
    valid = valid_trials(dataset)
    targets = dataset.ys.clone()
    targets[~valid] = float('nan')
    probs = torch.softmax(logits, dim=-1).clamp(EPS, 1 - EPS)
    nll = -(targets * torch.log(probs))
    return nll.sum(dim=-1).nansum(dim=1)[..., 0], valid.sum(dim=1).float()


def continuous_session_sse(outputs, dataset, study):
    """Per-session sum of squared errors, trial count, and target dimensionality."""
    valid = valid_trials(dataset)
    targets = dataset.ys
    if study == 'weber2024':
        # same clamping as spice_weber2024.clamped_angular_mse: prediction moves at most speed*dt from shield_t
        from weinhardt2026.studies.weber2024.spice_weber2024 import MOVEMENT_SPEED
        shield_t, dt = targets[..., 2:4], targets[..., 4:5]
        delta = outputs[..., :2] - shield_t
        max_move = MOVEMENT_SPEED * dt * (math.pi / 180.0)
        fraction = torch.clamp_max(max_move / (delta.norm(dim=-1, keepdim=True) + 1e-8), 1.0)
        prediction, actual = shield_t + fraction * delta, targets[..., :2]
    else:
        prediction, actual = outputs, targets[..., :outputs.shape[-1]]
    squared_error = ((prediction - actual) ** 2).sum(dim=-1)[..., 0]  # (B, T)
    squared_error[~valid] = 0.
    return squared_error.sum(dim=1), valid.sum(dim=1).float(), actual.shape[-1]


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
        metrics.update(aic=info['aic_mean'], bic=info['bic_mean'])
    return metrics


def evaluate_continuous(outputs_train, outputs_test, dataset_train, dataset_test, k_per_group, study):
    sse_train, n_train, dim = continuous_session_sse(outputs_train, dataset_train, study)
    sse_test, n_test, _ = continuous_session_sse(outputs_test, dataset_test, study)

    # held-out Gaussian log-likelihood per trial, noise variance fit on training residuals
    variance = sse_train.sum().item() / (n_train.sum().item() * dim)
    loglik_test = -0.5 * n_test.sum().item() * dim * math.log(2 * math.pi * variance) - sse_test.sum().item() / (2 * variance)
    metrics = {'heldout': loglik_test / n_test.sum().item()}

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
            aic=(two_nll + 2 * k).mean().item(),
            bic=(two_nll + k * torch.log(n_group)).mean().item(),
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

    benchmark = config['benchmark'](n_participants, n_actions)
    k_benchmark = n_benchmark_parameters(benchmark, n_participants)
    metrics = evaluate(predict(benchmark, dataset_train), predict(benchmark, dataset_test),
                       lambda pairs: torch.full((pairs.shape[0],), k_benchmark))
    rows.append(dict(model='Benchmark', seed=-1, n_parameters=k_benchmark, **metrics))

    gru = config['gru'](n_participants, n_actions)
    assert gru.participant_embedding_size == 0, f'{study}: GRU must not use a participant embedding'
    metrics = evaluate(predict(gru, dataset_train), predict(gru, dataset_test), None)
    rows.append(dict(model='GRU', seed=-1, **metrics))

    checkpoints = config['checkpoints']()
    if not checkpoints:
        raise FileNotFoundError(f'{study}: no SPICE checkpoints found')
    for path in checkpoints:
        seed = int(path.rsplit('_rs', 1)[1].split('.')[0])
        print(f'  {os.path.basename(path)}', flush=True)
        estimator = load_spice(path, spice_module, dataset_train)
        coefficient_counts = estimator.count_sindy_coefficients().cpu()

        metrics = evaluate(predict(estimator, dataset_train, use_sindy=False),
                           predict(estimator, dataset_test, use_sindy=False), None)
        rows.append(dict(model='SPICE-RNN', seed=seed, **metrics))

        k_spice = lambda pairs: coefficient_counts[pairs[:, 0], pairs[:, 1]].float()
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
    ('aic', 'AIC (train) ↓'),
    ('bic', 'BIC (train) ↓'),
]
HELDOUT_LABEL = {False: 'Held-out trial likelihood ↑', True: 'Held-out log-lik. / trial ↑'}


def plot_model_evaluation(results: pd.DataFrame, output_path: str):
    """Bars = mean over seeds (± SD for multi-seed SPICE); white dots = single seeds.

    The y-axes are zoomed onto the model differences, so bars rise from the bottom
    of each axis rather than from 0 (the continuous studies' AIC/BIC are negative).
    """
    present = set(results['study'])
    continuous = results.groupby('study')['continuous'].first().astype(bool)
    studies = [s for s in STUDIES if s in present and not continuous[s]] \
        + [s for s in STUDIES if s in present and continuous[s]]
    n_rows, n_cols = len(ROWS), len(studies)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols + 0.8, 2.2 * n_rows), squeeze=False)
    positions = {model: index for index, model in enumerate(MODEL_ORDER)}
    rng = np.random.default_rng(0)

    for col, study in enumerate(studies):
        study_results = results[results['study'] == study]
        is_continuous = bool(continuous[study])
        first_of_kind = col == 0 or is_continuous != bool(continuous[studies[col - 1]])
        for row, (metric, ylabel) in enumerate(ROWS):
            ax = axes[row, col]
            model_values = {}
            for model in MODEL_ORDER:
                values = study_results.loc[study_results['model'] == model, metric].dropna().to_numpy()
                if values.size > 0:
                    model_values[model] = values

            all_values = np.concatenate(list(model_values.values()))
            low, high = all_values.min(), all_values.max()
            margin = 0.15 * (high - low) if high > low else 0.05 * abs(high) + 1e-3
            floor = low - 2 * margin  # keeps the lowest bar visible
            ax.set_ylim(floor, high + margin)

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

            if row == 0:
                ax.set_title(STUDIES[study]['label'], fontsize=9)
                if first_of_kind:
                    ax.set_ylabel(HELDOUT_LABEL[is_continuous], fontsize=8.5)
            elif col == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODEL_ORDER]
    fig.legend(handles, MODEL_ORDER, loc='upper center', ncol=len(MODEL_ORDER), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {output_path}.pdf/.png')


# ── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--studies', nargs='+', default=list(STUDIES))
    parser.add_argument('--recompute', action='store_true', help='Ignore cached per-study CSVs')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = []
    for study in args.studies:
        cache = os.path.join(OUTPUT_DIR, f'model_evaluation_{study}.csv')
        if os.path.isfile(cache) and not args.recompute:
            frames.append(pd.read_csv(cache))
            continue
        print(f'Evaluating {study}...', flush=True)
        df = evaluate_study(study)
        df.to_csv(cache, index=False)
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    summary = results.groupby(['study', 'model'], sort=False)[['heldout', 'aic', 'bic', 'n_parameters']].agg(['mean', 'std'])
    print(summary.to_string(float_format='{:.4f}'.format))
    summary.to_csv(os.path.join(OUTPUT_DIR, 'model_evaluation_summary.csv'))
    plot_model_evaluation(results, os.path.join(OUTPUT_DIR, 'figure_model_evaluation'))

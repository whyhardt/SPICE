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
    ``weinhardt2026/analysis/analysis_posterior_predictive.py``, read here per seed):
      discrete   → nRMSE of the summary statistics across participants (mean over panels)
      continuous → per-trial MSE of an autonomous rollout (bruckner2025: px², weber2024: deg²)
  - AIC / BIC per trial (training data: per (participant, experiment) group divided by the group's trials, then averaged over groups, see
    ``analysis_model_evaluation.grouped_information_criteria``) — only for the
    interpretable models (Benchmark, SPICE-EQ). Continuous studies use a Gaussian
    likelihood with per-group MLE variance.

Written to weinhardt2026/results/model_evaluation/ (figure, model_evaluation_summary.csv: mean and SD over seeds);
per study, the per-seed evaluation is cached in studies/<study>/results/model_evaluation/model_evaluation.csv.

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
from weinhardt2026.analysis.analysis_posterior_predictive import ROLLOUT_UNITS as RESPONSE_UNITS, load_scores as load_ppc_scores
from weinhardt2026.utils.final_models import checkpoint_seed, final_checkpoints, load_gru


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-9
OUTPUT_DIR = os.path.join(ROOT, 'weinhardt2026', 'results', 'model_evaluation')


# ── Study configuration ─────────────────────────────────────────────
# data_kwargs / test_blocks mirror slurm_jobs/grid_search.sh; the benchmark / GRU kwargs mirror
# benchmark_grid_search/benchmark_grid_search.py. benchmark / gru: (path, n_participants, n_actions) -> model

def _benchmark(module_name, class_name, **kwargs):
    def build(path, n_participants, n_actions):
        module = importlib.import_module(module_name)
        model = getattr(module, class_name)(n_participants=n_participants, **kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
    return build


def _gru(**kwargs):
    def build(path, n_participants, n_actions):
        return load_gru(path, **kwargs)
    return build


def _study_path(study, *parts):
    return os.path.join(ROOT, 'weinhardt2026', 'studies', study, *parts)


def study_evaluation_path(study):
    """Per-study cache of evaluate_study (one row per model and seed)."""
    return _study_path(study, 'results', 'model_evaluation', 'model_evaluation.csv')


STUDIES = {
    'dezfouli2019': dict(
        label='Two-Armed Bandit\n(Dezfouli 2019)',
        test_blocks=(3, 6, 9),
        data_kwargs={},
        benchmark=_benchmark('weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019', 'GQLModel', batch_first=True),
        gru=_gru(n_actions=2, additional_inputs=2),
    ),
    'ganesh2024a': dict(
        label='Perceptual Bandit\n(Ganesh 2024)',
        test_blocks=(3, 6, 9),
        data_kwargs={"df_participant_id": "subjID", "df_choice": "choice", "df_feedback": "reward",
                     "df_block": "blocks", "additional_inputs": ["contrast_difference"]},
        benchmark=_benchmark('weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a', 'BayesianModel', batch_first=True),
        gru=_gru(n_actions=2, additional_inputs=2),
    ),
    'eckstein2026': dict(
        label='Four-Armed Bandit\n(Eckstein 2026)',
        test_blocks=(2,),
        data_kwargs={},
        benchmark=_benchmark('weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026', 'BestRLModel',
                             n_actions=4, batch_first=True),
        gru=_gru(n_actions=4, additional_inputs=2),
    ),
    'bustamante2023': dict(
        label='Patch Foraging\n(Bustamante 2023)',
        test_blocks=(3, 6),
        data_kwargs={"df_participant_id": "subject_id", "df_choice": "decision", "df_feedback": "reward",
                     "df_block": "overall_round", "additional_inputs": ["harvest_duration", "travel_duration", "is_msit", "is_high"]},
        benchmark=_benchmark('weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023',
                             'EffortCostMVTModel', batch_first=True),
        gru=_gru(n_actions=2, additional_inputs=4, additional_inputs_scale=(10., 10., 1., 1.)),   # harvest_duration, travel_duration (constant, scaled to [0, 1]), is_msit, is_high
    ),
    'braun2018': dict(
        label='Task Switching\n(Braun 2018)',
        test_blocks=(3, 6, 9),
        data_kwargs={"df_participant_id": "subject", "df_choice": "transcode", "df_feedback": None,
                     "df_block": "block", "additional_inputs": ["difference", "current", "other"],
                     "timeshift_additional_inputs": [-1, -1, -1]},
        benchmark=_benchmark('weinhardt2026.studies.braun2018.benchmarking_braun2018', 'ExpectedValueControl'),
        gru=_gru(n_actions=2, additional_inputs=4, n_reward_features=0),
    ),
    'bruckner2025': dict(
        label='Predictive Inference\n(Bruckner 2025)',
        test_blocks=(2, 4),
        data_kwargs={"df_choice": "b_t", "df_feedback": "x_t",
                     "additional_inputs": ["z_next", "catch", "v_t", "sigma", "r_t", "mu_t", "c_t"],
                     "continuous_action": True, "timeshift_additional_inputs": [0, 0, 0, 0, 0, 0, 0]},
        benchmark=_benchmark('weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025', 'RationalResourceModel'),
        gru=_gru(n_actions=1, additional_inputs=6, n_reward_features=1),
    ),
    'weber2024': dict(
        label='Circular Predictive\nInference (Weber 2024)',
        test_blocks=(3, 5, 9, 12),
        data_kwargs={"df_participant_id": "participant", "df_experiment_id": "experiment",
                     "df_choice": ["shield_sin", "shield_cos"], "df_feedback": ["laser_sin", "laser_cos"],
                     "df_block": "block",
                     "additional_inputs": ["laser_caught", "volatility", "stochasticity", "trial_duration_frames", "trueMean"],
                     "continuous_action": True},
        benchmark=_benchmark('weinhardt2026.studies.weber2024.benchmarking_weber2024', 'ChangePointModel', h_step=2),
        gru=_gru(n_actions=2, additional_inputs=4, n_reward_features=2,
                 additional_inputs_scale=(1., 1., 1., 90.)),
    ),
}


# kolff2025: one group model for all apes, evaluated by evaluate_group_study (not by the registry above)
GROUP_STUDIES = {'kolff2025': dict(label='Chimpanzee Grooming\n(Kolff 2025)')}
ALL_STUDIES = list(STUDIES) + list(GROUP_STUDIES)
LABELS = {study: config['label'] for study, config in {**STUDIES, **GROUP_STUDIES}.items()}


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


def spice_estimator(spice_module, dataset_train, **overrides):
    """The study's SpiceEstimator as run.py builds it; overrides replace single arguments (e.g. n_participants)."""
    n_actions = dataset_train.ys.shape[-1]
    kwargs = dict(
        spice_class=spice_module.SpiceModel,
        spice_config=spice_module.CONFIG,
        n_actions=n_actions,
        n_items=n_actions,
        n_participants=len(dataset_train.xs[..., -1].unique()),
        n_experiments=len(dataset_train.xs[..., -2].unique()),
        n_reward_features=dataset_train.n_reward_features,
        loss_fn=getattr(spice_module, 'LOSS_FN', cross_entropy_loss),
        embedding_size=8,
        compiled_forward=False,
        device=DEVICE,
    )
    kwargs.update(getattr(spice_module, 'ESTIMATOR_KWARGS', {}))
    kwargs.update(overrides)
    return SpiceEstimator(**kwargs)


def load_spice(path, spice_module, dataset_train):
    estimator = spice_estimator(spice_module, dataset_train)
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


def continuous_prediction_and_target(outputs, dataset, study):
    """The models' predicted and the actual next response, (B, T, W, D) each."""
    targets = dataset.ys
    if study == 'weber2024':
        # same clamping as spice_weber2024.clamped_angular_mse: prediction moves at most speed*dt from shield_t
        from weinhardt2026.studies.weber2024.spice_weber2024 import MOVEMENT_SPEED
        shield_t, dt = targets[..., 2:4], targets[..., 4:5]
        delta = outputs[..., :2] - shield_t
        max_move = MOVEMENT_SPEED * dt * (math.pi / 180.0)
        fraction = torch.clamp_max(max_move / (delta.norm(dim=-1, keepdim=True) + 1e-8), 1.0)
        return shield_t + fraction * delta, targets[..., :2]
    return outputs, targets[..., :outputs.shape[-1]]


def continuous_session_sse(outputs, dataset, study):
    """Per-session sum of squared errors, trial count, and target dimensionality."""
    valid = valid_trials(dataset)
    prediction, actual = continuous_prediction_and_target(outputs, dataset, study)
    squared_error = ((prediction - actual) ** 2).sum(dim=-1)[..., 0]  # (B, T)
    squared_error[~valid] = 0.
    return squared_error.sum(dim=1), valid.sum(dim=1).float(), actual.shape[-1]


def response_session_sse(outputs, dataset, study):
    """Per-session sum of squared errors in the response's units (RESPONSE_UNITS: bruckner2025 bucket
    position in px, weber2024 angular error of the shield in degrees) and trial count."""
    valid = valid_trials(dataset)
    prediction, actual = continuous_prediction_and_target(outputs, dataset, study)
    if study == 'weber2024':
        angle = lambda vector: torch.rad2deg(torch.atan2(vector[..., 0], vector[..., 1]))
        difference = (angle(prediction) - angle(actual)) % 360
        squared_error = torch.minimum(difference, 360 - difference)[..., 0] ** 2
    else:
        from weinhardt2026.studies.bruckner2025.spice_bruckner2025 import POSITION_SCALE
        squared_error = (((prediction - actual) * POSITION_SCALE) ** 2).sum(dim=-1)[..., 0]
    squared_error[~valid] = 0.
    return squared_error.sum(dim=1), valid.sum(dim=1).float()


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


def plot_model_evaluation(results: pd.DataFrame, output_path: str):
    """Bars = mean over seeds (± SD if several seeds); white dots = single seeds.

    The y-axes are zoomed onto the model differences, so bars rise from the bottom
    of each axis rather than from 0 (the continuous studies' AIC/BIC are negative). Where
    SHARED_SPAN holds, the columns of a row (of one kind) span the same y-range: the largest
    range among them.
    """
    present = set(results['study'])
    continuous = results.groupby('study')['continuous'].first().astype(bool)
    studies = [s for s in ALL_STUDIES if s in present and not continuous[s]] \
        + [s for s in ALL_STUDIES if s in present and continuous[s]]
    n_rows, n_cols = len(ROWS), len(studies)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols + 0.8, 2.2 * n_rows), squeeze=False)
    positions = {model: index for index, model in enumerate(MODEL_ORDER)}
    rng = np.random.default_rng(0)

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
    plot_model_evaluation(results, os.path.join(OUTPUT_DIR, 'figure_model_evaluation'))

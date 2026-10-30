"""The paper's studies: registry, data and model loading, and per-session scores.

``STUDIES`` holds per study its label, test blocks, ``csv_to_dataset`` arguments and the
loaders of its benchmark and GRU checkpoints (path, n_participants, n_actions -> model);
``GROUP_STUDIES`` the studies fitted with one group model (kolff2025). Data and SPICE
estimators are rebuilt exactly as ``slurm_jobs/grid_search.sh`` -> ``weinhardt2026/run.py``
built them, so every analysis scores the models on the split they were trained on.
"""

import importlib
import math
import os
import sys

import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules are imported as 'studies.<study>...' (as in run.py)

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice.resources.spice_training import cross_entropy_loss
from weinhardt2026.utils.final_models import load_gru

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-9


# ── Study configuration ─────────────────────────────────────────────
# data_kwargs / test_blocks mirror slurm_jobs/grid_search.sh; the benchmark / GRU kwargs mirror
# analysis/grid_search/benchmark_grid_search.py. benchmark / gru: (path, n_participants, n_actions) -> model

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


def study_path(study, *parts):
    """weinhardt2026/studies/<study>/<parts>."""
    return os.path.join(ROOT, 'weinhardt2026', 'studies', study, *parts)


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

    data = study_path(study, 'data', f'{study}.csv')
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

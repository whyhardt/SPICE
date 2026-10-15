"""
Generative analysis for weber2024 (circular predictive inference / shield task).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py weber2024
or
    python weinhardt2026/studies/weber2024/analysis_generative.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from spice import SpiceDataset
from weinhardt2026.analysis.analysis_generative_comparison import (
    load_registered_dataset, load_registered_models, run_generative_analysis,
    valid_trials, session_mean,
)
from weinhardt2026.studies.weber2024.benchmarking_weber2024 import generate_behavior, CATCH_THRESHOLD, _N_ACTIONS, _AI_START, _AI_CAUGHT


STUDY = 'weber2024'
N_REPEATS = 1

# the models predict the next shield position deterministically: one replication suffices
METRIC_LABELS = {
    'mean_prediction_error': 'Mean Prediction Error (°)',
    'p_catch': 'P(Catch)',
    'total_reward': 'Total Catches',
    'mean_shield_movement': 'Mean Shield Movement (°)',
    'avg_learning_rate': 'Avg Learning Rate',
    'p_perseveration': 'P(Perseveration)',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset)


def _degrees(sin_values: np.ndarray, cos_values: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(sin_values, cos_values)) % 360


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = (a - b) % 360
    return np.minimum(difference, 360 - difference)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    """Prediction error = angular distance shield - laser; movement = angular distance between
    consecutive shield positions; learning rate = movement / prediction error (prediction error > 1°,
    clipped to [0, 2]); perseveration = movement < 5°."""
    xs = dataset.xs[:, :, 0].numpy()
    valid = valid_trials(dataset)
    shield = _degrees(xs[..., 0], xs[..., 1])
    laser = _degrees(xs[..., _N_ACTIONS], xs[..., _N_ACTIONS + 1])
    caught = xs[..., _AI_START + _AI_CAUGHT].copy()
    shield[~valid], laser[~valid], caught[~valid] = np.nan, np.nan, np.nan

    prediction_error = _angular_distance(shield, laser)
    movement = np.full_like(shield, np.nan)
    movement[:, :-1] = _angular_distance(shield[:, 1:], shield[:, :-1])

    learning_rate = np.full_like(shield, np.nan)
    computable = ~np.isnan(movement) & (prediction_error > 1.)
    learning_rate[computable] = np.clip(movement[computable] / prediction_error[computable], 0, 2)

    catch = (prediction_error <= CATCH_THRESHOLD).astype(float)
    catch[np.isnan(prediction_error)] = np.nan
    perseveration = (movement < 5.).astype(float)
    perseveration[np.isnan(movement)] = np.nan
    return {
        'mean_prediction_error': session_mean(prediction_error),
        'p_catch': session_mean(catch),
        'total_reward': np.nansum(caught, axis=1),
        'mean_shield_movement': session_mean(movement),
        'avg_learning_rate': session_mean(learning_rate),
        'p_perseveration': session_mean(perseveration),
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

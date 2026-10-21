"""
Generative analysis for bruckner2025 (predictive inference / helicopter task).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py bruckner2025
or
    python weinhardt2026/studies/bruckner2025/analysis_generative.py
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
from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import generate_behavior, _AI_MU_T, _AI_C_T
from weinhardt2026.studies.bruckner2025.spice_bruckner2025 import POSITION_SCALE


STUDY = 'bruckner2025'
N_REPEATS = 1

# the models predict the next bucket position deterministically: one replication suffices
METRIC_LABELS = {
    'mean_prediction_error': 'Mean |Prediction Error|',
    'mean_estimation_error': 'Mean |Estimation Error|',
    'avg_learning_rate': 'Avg Learning Rate',
    'post_cp_learning_rate': 'Post-CP Learning Rate (t+1..t+3)',
    'p_perseveration': 'P(Perseveration)',
    'mean_update_magnitude': 'Mean |Update|',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset, noise_std=None) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset, noise_std=noise_std)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    """Positions are normalized by POSITION_SCALE. Prediction error |x_t - b_t|, estimation error
    |mu_t - b_t|, update b_{t+1} - b_t, learning rate |update| / prediction error (prediction error
    > 0.01, clipped to [0, 2]), perseveration |update| < 10 px."""
    xs = dataset.xs[:, :, 0].numpy()
    valid = valid_trials(dataset)
    first_input = 1 + dataset.n_reward_features
    bucket, outcome = xs[..., 0].copy(), xs[..., 1].copy()
    mu, change_point = xs[..., first_input + _AI_MU_T].copy(), xs[..., first_input + _AI_C_T]
    for values in (bucket, outcome, mu):
        values[~valid] = np.nan

    prediction_error = np.abs(outcome - bucket)
    update = np.full_like(bucket, np.nan)
    update[:, :-1] = bucket[:, 1:] - bucket[:, :-1]

    learning_rate = np.full_like(bucket, np.nan)
    computable = ~np.isnan(update) & (prediction_error > 0.01)
    learning_rate[computable] = np.clip(np.abs(update[computable]) / prediction_error[computable], 0, 2)

    after_change_point = np.zeros_like(valid)
    for lag in (1, 2, 3):
        after_change_point[:, lag:] |= change_point[:, :-lag] == 1

    perseveration = (np.abs(update) < 10. / POSITION_SCALE).astype(float)
    perseveration[np.isnan(update)] = np.nan
    return {
        'mean_prediction_error': session_mean(prediction_error),
        'mean_estimation_error': session_mean(np.abs(mu - bucket)),
        'avg_learning_rate': session_mean(learning_rate),
        'post_cp_learning_rate': session_mean(learning_rate, after_change_point),
        'p_perseveration': session_mean(perseveration),
        'mean_update_magnitude': session_mean(np.abs(update)),
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

"""
Generative analysis for ganesh2024a (two-armed bandit with perceptual contrast stimuli).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py ganesh2024a
or
    python weinhardt2026/studies/ganesh2024a/analysis_generative.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from spice import SpiceDataset
from weinhardt2026.analysis.analysis_generative_comparison import (
    load_registered_dataset, load_registered_models, run_generative_analysis,
    choices_and_rewards, session_mean, stays,
)
from weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a import generate_behavior, get_reward_schedule


STUDY = 'ganesh2024a'
N_REPEATS = 10   # runs per model (SPICE: one model per seed)
PATH_DATA = f'weinhardt2026/studies/{STUDY}/data/{STUDY}.csv'

METRIC_LABELS = {
    'accuracy': 'Accuracy (block contingency)',
    'accuracy_low_contrast': 'Accuracy (Low |Contrast|)',
    'accuracy_mid_contrast': 'Accuracy (Mid |Contrast|)',
    'accuracy_high_contrast': 'Accuracy (High |Contrast|)',
    'avg_reward': 'Average Reward',
    'p_stay': 'P(Stay)',
    'win_stay': 'P(Stay | Win)',
    'lose_shift': 'P(Shift | Loss)',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def _reward_schedule(dataset: SpiceDataset) -> torch.Tensor:
    """The real outcomes of both sides, repeated to the replications held in `dataset`."""
    rewards = get_reward_schedule(PATH_DATA)
    return rewards.repeat(dataset.xs.shape[0] // rewards.shape[0], 1, 1)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset, rewards=_reward_schedule(dataset))


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    """Action 0 = left; contrast difference > 0 = left has the higher contrast.

    Each block rewards either the higher- or the lower-contrast side more often (the contingency is
    read from the replayed outcomes, so it is the same for real and simulated behavior). Accuracy =
    choosing the side the block's contingency favors, also split into terciles of |contrast
    difference| over all trials of the dataset.
    """
    choices, rewards = choices_and_rewards(dataset)
    valid = ~np.isnan(choices)
    contrast = dataset.xs[:, :, 0, 2 * dataset.n_actions].numpy()
    left_is_high = contrast > 0

    schedule = _reward_schedule(dataset).numpy()
    reward_high = np.where(left_is_high, schedule[..., 0], schedule[..., 1])
    high_favored = session_mean(reward_high, valid) > 0.5

    chose_high = (choices == 0) == left_is_high
    correct = (chose_high == high_favored[:, None]).astype(float)
    correct[~valid] = np.nan
    abs_contrast = np.where(valid, np.abs(contrast), np.nan)
    low, high = np.nanpercentile(abs_contrast, [100 / 3, 200 / 3])

    stay = stays(choices)
    previous_reward = rewards[:, :-1]
    return {
        'accuracy': session_mean(correct),
        'accuracy_low_contrast': session_mean(correct, abs_contrast <= low),
        'accuracy_mid_contrast': session_mean(correct, (abs_contrast > low) & (abs_contrast <= high)),
        'accuracy_high_contrast': session_mean(correct, abs_contrast > high),
        'avg_reward': session_mean(rewards),
        'p_stay': session_mean(stay),
        'win_stay': session_mean(stay, previous_reward > 0.5),
        'lose_shift': 1. - session_mean(stay, previous_reward < 0.5),
    }

if __name__ == '__main__':
    run_generative_analysis(STUDY)

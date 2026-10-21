"""
Generative analysis for eckstein2026 (four-armed drifting bandit).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py eckstein2026
or
    python weinhardt2026/studies/eckstein2026/analysis_generative.py
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
from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import generate_behavior, get_reward_schedule


STUDY = 'eckstein2026'
N_REPEATS = 10   # runs per model (SPICE: one model per seed)
PATH_DATA = f'weinhardt2026/studies/{STUDY}/data/{STUDY}.csv'

METRIC_LABELS = {
    'avg_reward': 'Average Reward',
    'total_reward': 'Total Reward',
    'p_stay': 'P(Stay)',
    'win_stay': 'P(Stay | Win)',
    'lose_stay': 'P(Stay | Loss)',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    """The real payouts of all arms are replayed; `dataset` holds whole replications of the sessions."""
    rewards = get_reward_schedule(PATH_DATA)
    rewards = rewards.repeat(dataset.xs.shape[0] // rewards.shape[0], 1, 1)
    return generate_behavior(model=model, dataset=dataset, rewards=rewards)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    choices, rewards = choices_and_rewards(dataset)
    stay = stays(choices)
    previous_reward = rewards[:, :-1]
    return {
        'avg_reward': session_mean(rewards),
        'total_reward': np.nansum(rewards, axis=1),
        'p_stay': session_mean(stay),
        'win_stay': session_mean(stay, previous_reward > 0.5),
        'lose_stay': session_mean(stay, previous_reward < 0.5),
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

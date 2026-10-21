"""
Generative analysis for dezfouli2019 (two-armed bandit with block-wise reward probabilities).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py dezfouli2019
or
    python weinhardt2026/studies/dezfouli2019/analysis_generative.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from spice import SpiceDataset
from weinhardt2026.analysis.analysis_generative_comparison import (
    load_registered_dataset, load_registered_models, run_generative_analysis,
    choices_and_rewards, valid_trials, session_mean, stays, choice_entropy, quarter_masks,
)
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import generate_behavior


STUDY = 'dezfouli2019'
N_REPEATS = 10   # runs per model (SPICE: one model per seed)

METRIC_LABELS = {
    'avg_reward': 'Average Reward',
    'total_reward': 'Total Reward',
    'p_stay': 'P(Stay)',
    'win_stay': 'P(Stay | Win)',
    'lose_stay': 'P(Stay | Loss)',
    'choice_entropy': 'Choice Entropy',
    'early_reward': 'Early Reward (Q1)',
    'late_reward': 'Late Reward (Q4)',
    'learning_slope': 'Learning Slope (Q4 - Q1)',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    choices, rewards = choices_and_rewards(dataset)
    valid = valid_trials(dataset)
    stay = stays(choices)
    previous_reward = rewards[:, :-1]
    early, late = quarter_masks(valid)
    early_reward, late_reward = session_mean(rewards, early), session_mean(rewards, late)
    return {
        'avg_reward': session_mean(rewards),
        'total_reward': np.nansum(rewards, axis=1),
        'p_stay': session_mean(stay),
        'win_stay': session_mean(stay, previous_reward > 0.5),
        'lose_stay': session_mean(stay, previous_reward < 0.5),
        'choice_entropy': choice_entropy(choices, dataset.n_actions),
        'early_reward': early_reward,
        'late_reward': late_reward,
        'learning_slope': late_reward - early_reward,
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

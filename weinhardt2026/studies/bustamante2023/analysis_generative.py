"""
Generative analysis for bustamante2023 (patch foraging with depleting rewards).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py bustamante2023
or
    python weinhardt2026/studies/bustamante2023/analysis_generative.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from spice import SpiceDataset
from weinhardt2026.analysis.analysis_generative_comparison import (
    load_registered_dataset, load_registered_models, run_generative_analysis,
    choices_and_rewards, session_mean,
)
from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import generate_behavior


STUDY = 'bustamante2023'
N_REPEATS = 10   # runs per model (SPICE: one model per seed)

METRIC_LABELS = {
    'p_harvest': 'P(Harvest)',
    'patch_residence': 'Mean Patch Residence (harvests)',
    'exit_reward': 'Mean Exit Reward',
    'avg_reward': 'Average Reward',
    'total_reward': 'Total Reward',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    """Action 0 = harvest, 1 = exit. A patch is a run of harvests ended by an exit (a trailing run
    without exit is not counted); its exit reward is the reward of the last harvest before leaving."""
    choices, rewards = choices_and_rewards(dataset)
    n_sessions, n_trials = choices.shape

    run_length = np.zeros(n_sessions)
    last_reward = np.zeros(n_sessions)
    n_patches = np.zeros(n_sessions)
    sum_residence = np.zeros(n_sessions)
    sum_exit_reward = np.zeros(n_sessions)
    for t in range(n_trials):
        harvest, leave = choices[:, t] == 0, choices[:, t] == 1
        patch_end = leave & (run_length > 0)
        n_patches += patch_end
        sum_residence += np.where(patch_end, run_length, 0.)
        sum_exit_reward += np.where(patch_end, last_reward, 0.)
        run_length = np.where(harvest, run_length + 1, 0.)
        last_reward = np.where(harvest, np.nan_to_num(rewards[:, t]), last_reward)

    with np.errstate(invalid='ignore', divide='ignore'):
        patch_residence = np.where(n_patches > 0, sum_residence / n_patches, np.nan)
        exit_reward = np.where(n_patches > 0, sum_exit_reward / n_patches, np.nan)
    return {
        'p_harvest': session_mean((choices == 0).astype(float), ~np.isnan(choices)),
        'patch_residence': patch_residence,
        'exit_reward': exit_reward,
        'avg_reward': session_mean(rewards),
        'total_reward': np.nansum(rewards, axis=1),
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

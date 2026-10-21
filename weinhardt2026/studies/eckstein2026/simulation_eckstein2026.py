"""
Simulation of eckstein2026 (four-armed drifting bandit) for the posterior predictive checks (analysis/analysis_posterior_predictive.py) and
the confusion matrix (confusion_matrix/generate_datasets.py); interface: weinhardt2026/utils/study_models.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spice import SpiceDataset
from weinhardt2026.utils.study_models import load_registered_dataset, load_registered_models
from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import generate_behavior, get_reward_schedule


STUDY = 'eckstein2026'
PATH_DATA = f'weinhardt2026/studies/{STUDY}/data/{STUDY}.csv'


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    """The real outcomes are replayed; `dataset` holds whole replications of the sessions."""
    rewards = get_reward_schedule(PATH_DATA)
    rewards = rewards.repeat(dataset.xs.shape[0] // rewards.shape[0], 1, 1)
    return generate_behavior(model=model, dataset=dataset, rewards=rewards)

"""
Simulation of weber2024 (circular predictive inference / shield task) for the posterior predictive checks (analysis/evaluation/run_posterior_predictive.py) and
the confusion matrix (analysis/validation/model_identification/generate_datasets.py); interface: weinhardt2026/utils/study_models.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spice import SpiceDataset
from weinhardt2026.utils.study_models import load_registered_dataset, load_registered_models
from weinhardt2026.studies.weber2024.benchmarking_weber2024 import generate_behavior


STUDY = 'weber2024'


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset)

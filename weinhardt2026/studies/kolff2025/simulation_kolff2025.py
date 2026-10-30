"""
Simulation of kolff2025 (chimpanzee grooming interactions, both perspectives of every interaction) for the posterior predictive checks (analysis/evaluation/run_posterior_predictive.py) and
the confusion matrix (analysis/validation/model_identification/generate_datasets.py); interface: weinhardt2026/utils/study_models.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from spice import SpiceDataset, SpiceEstimator, csv_to_dataset
from weinhardt2026.utils.study_models import select_spice_members
from weinhardt2026.utils.final_models import final_checkpoints, load_gru
from weinhardt2026.studies.kolff2025.spice_kolff2025 import (
    SpiceModel, CONFIG, N_ACTIONS, LOSS_FN, ESTIMATOR_KWARGS, prepare_dataset,
)
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import (
    DATA_PATH, DATA_KWARGS, PARTNER_COLUMNS, SELF_COLUMNS,
    get_dataset, evaluation_dataset, generate_behavior, LogisticRegressionModel, ModelInputGRU,
)


STUDY = 'kolff2025'

# input columns of the partner's and the ape's own acts (in xs)
COLUMNS_PARTNER = [N_ACTIONS + CONFIG.additional_inputs.index(c) for c in PARTNER_COLUMNS]
COLUMNS_SELF = [N_ACTIONS + CONFIG.additional_inputs.index(c) for c in SELF_COLUMNS]


def load_dataset() -> SpiceDataset:
    """All interactions (both perspectives), prepare_dataset layout as in get_dataset."""
    get_dataset()  # writes the modelling csv if it does not exist yet
    return prepare_dataset(csv_to_dataset(file=DATA_PATH, **DATA_KWARGS))


def load_spice(path: str) -> SpiceEstimator:
    estimator = SpiceEstimator(
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_participants=1,
        n_reward_features=0,
        loss_fn=LOSS_FN,
        **ESTIMATOR_KWARGS,
        embedding_size=8,
        sindy_library_polynomial_degree=1,
        compiled_forward=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    estimator.load_spice(path)
    return estimator


def load_models(dataset: SpiceDataset) -> dict:
    """The final group models (GRU, SPICE: every seed); the benchmark is refit on the same training fold.
    SPICE: per seed the member with the highest training likelihood, scored where the ape acts next."""
    dataset_train, _, _ = get_dataset(test_fold=0)
    gru = [load_gru(path, ModelInputGRU) for path in final_checkpoints(STUDY, 'gru')]
    spice = select_spice_members(STUDY, final_checkpoints(STUDY, 'spice'), load_spice, evaluation_dataset(dataset_train))
    return {'GRU': gru, 'Benchmark': LogisticRegressionModel().fit(dataset_train), 'SPICE': spice}


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    """Closed-loop on whether the ape's next act is grooming; the partner's acts and the timing of the ape's
    own acts are replayed (benchmarking_kolff2025.generate_behavior)."""
    return generate_behavior(model=model, dataset=dataset)

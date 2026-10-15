"""
Generative analysis for kolff2025 (bonobo grooming interactions, both perspectives of every interaction).

The models predict only whether the focal ape's next act is grooming, so simulation is closed-loop on
that decision while the partner's acts and the timing of the ape's own acts are replayed
(benchmarking_kolff2025.generate_behavior). The metrics describe when the ape grooms:
overall, after the partner's negotiation / preparation / grooming acts, after its own grooming, and
how many own acts precede its first grooming in an interaction. They are computed on events from the
second one on (the first event is replayed), identically for real and simulated interactions.

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py kolff2025
or
    python weinhardt2026/studies/kolff2025/analysis_generative.py
"""

import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from spice import SpiceDataset, SpiceEstimator, csv_to_dataset
from weinhardt2026.analysis.analysis_generative_comparison import run_generative_analysis, select_spice_members
from weinhardt2026.studies.kolff2025.spice_kolff2025 import (
    SpiceModel, CONFIG, N_ACTIONS, LOSS_FN, ESTIMATOR_KWARGS, prepare_dataset,
)
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import (
    DATA_PATH, DATA_KWARGS, PARTNER_COLUMNS, SELF_COLUMNS,
    get_dataset, evaluation_dataset, generate_behavior, LogisticRegressionModel, ModelInputGRU,
)


STUDY = 'kolff2025'
N_REPEATS = 10   # runs per model (SPICE: one model per seed)
# grid-search seeds (slurm_jobs/grid_search.sh: group model, linear library)
PATTERN_SPICE = 'weinhardt2026/studies/kolff2025/params/params/spice_kolff2025_th0.05_sw0.01_al0.0005_er0.5_gp0_fp0_rs[0-9]*.pkl'
PATH_GRU = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025.pkl'

METRIC_LABELS = {
    'p_groom': 'P(Groom)',
    'p_groom_after_partner_solicitation': 'P(Groom | Partner Solicited)',
    'p_groom_after_partner_preparation': 'P(Groom | Partner Prepared)',
    'p_groom_after_partner_groom': 'P(Groom | Partner Groomed)',
    'p_groom_after_own_groom': 'P(Groom | Own Groom)',
    'first_groom_latency': 'Own Acts Before First Groom',
}

COLUMNS_PARTNER = [N_ACTIONS + CONFIG.additional_inputs.index(c) for c in PARTNER_COLUMNS]
COLUMNS_SELF = [N_ACTIONS + CONFIG.additional_inputs.index(c) for c in SELF_COLUMNS]


def load_dataset() -> SpiceDataset:
    """All interactions (both perspectives), prepare_dataset layout as in get_dataset."""
    get_dataset()  # writes the modelling csv if it does not exist yet
    return prepare_dataset(csv_to_dataset(file=DATA_PATH, **DATA_KWARGS))


def _load_spice(path: str) -> SpiceEstimator:
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
    """The group models of kolff2025.py; the benchmark is refit on the same training fold. SPICE: per
    grid-search seed the member with the highest training likelihood, scored where the ape acts next."""
    dataset_train, _, _ = get_dataset(test_fold=0)

    gru = ModelInputGRU()
    gru.load_state_dict(torch.load(PATH_GRU, map_location='cpu'))

    spice = select_spice_members(STUDY, sorted(glob(PATTERN_SPICE)), _load_spice, evaluation_dataset(dataset_train))
    return {'GRU': gru, 'Benchmark': LogisticRegressionModel().fit(dataset_train), 'SPICE': spice}


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset)


def _conditional(groom: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """Per-session P(groom) over decision events (groom not NaN) where `condition` holds."""
    keep = condition & ~np.isnan(groom)
    n = keep.sum(axis=1)
    return np.where(n > 0, np.where(keep, groom, 0.).sum(axis=1) / np.maximum(n, 1), np.nan)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    xs = dataset.xs[:, :, 0].numpy()
    valid = ~np.isnan(xs[..., 0])
    own = np.nan_to_num(xs[..., COLUMNS_SELF])
    partner = np.nan_to_num(xs[..., COLUMNS_PARTNER])
    category = {name: index for index, name in enumerate(c.split('_', 1)[1] for c in SELF_COLUMNS)}

    # decision events: the ape acts, from the second event on; groom = its act is grooming
    decision = np.zeros_like(valid)
    decision[:, 1:] = valid[:, 1:] & (own[:, 1:].sum(axis=-1) > 0)
    groom = np.where(decision, own[..., category['groom']], np.nan)

    previous_partner = np.zeros_like(partner)
    previous_partner[:, 1:] = partner[:, :-1]
    previous_own_groom = np.zeros_like(valid)
    previous_own_groom[:, 1:] = own[:, :-1, category['groom']] == 1

    # own acts (decision events) before the first grooming; interactions without grooming: NaN
    groomed = groom == 1
    first = groomed.argmax(axis=1)
    acts_before = np.cumsum(decision, axis=1) - decision
    latency = np.where(groomed.any(axis=1), acts_before[np.arange(len(first)), first], np.nan)

    return {
        'p_groom': _conditional(groom, decision),
        'p_groom_after_partner_solicitation': _conditional(groom, previous_partner[..., category['grooming_solicitation']] == 1),
        'p_groom_after_partner_preparation': _conditional(
            groom, (previous_partner[..., category['reposition_body']] == 1) | (previous_partner[..., category['grooming_process']] == 1)),
        'p_groom_after_partner_groom': _conditional(groom, previous_partner[..., category['groom']] == 1),
        'p_groom_after_own_groom': _conditional(groom, previous_own_groom),
        'first_groom_latency': latency,
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

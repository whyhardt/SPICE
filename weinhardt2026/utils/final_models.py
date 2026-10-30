"""The final fitted models of every study.

``weinhardt2026/studies/<study>/params/`` holds, per model kind, all seeds of the study's final configuration:

    spice_<study>_<grid tags>_rs<seed>.pkl       SPICE (slurm_jobs/grid_search.sh)
    benchmark_<study>_<grid point>_rs<seed>.pkl  benchmark model (state dict)  } winning grid point of
    gru_<study>_<grid point>_rs<seed>.pkl        GRU baseline (state dict)     } analysis/grid_search/

Performance analyses use every seed. Analyses that need a single model (mechanisms, the study scripts) use
the seed with the highest training likelihood (`best_checkpoint`; for SPICE: of its best ensemble member).
"""

import os
import pickle
import re
from functools import lru_cache
from glob import glob

import torch

from weinhardt2026.utils.benchmarking_gru import GRUModel


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
KINDS = ('spice', 'benchmark', 'gru')


def checkpoint_seed(path: str) -> int:
    """Seed from a `..._rs<seed>.pkl` name; 0 for a single fit without a seed tag."""
    match = re.search(r'_rs(\d+)\.pkl$', os.path.basename(path))
    return int(match.group(1)) if match else 0


def final_checkpoints(study: str, kind: str) -> list[str]:
    """All seeds of a study's final `kind` model ('spice', 'benchmark' or 'gru'), ordered by seed."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got '{kind}'")
    paths = glob(os.path.join(ROOT, 'weinhardt2026', 'studies', study, 'params', f'{kind}_{study}_*.pkl'))
    if not paths:
        raise FileNotFoundError(f'{study}: no final {kind} checkpoints in params/')
    return sorted(paths, key=checkpoint_seed)


def gru_hidden_size(state_dict: dict) -> int:
    """Hidden size of a GRUModel state dict (weight_hh_l0 is (3 * hidden_size, hidden_size))."""
    return state_dict['gru.weight_hh_l0'].shape[1]


def load_gru(path: str, gru_class: type = GRUModel, **kwargs) -> GRUModel:
    """GRU with the hidden size of its checkpoint; kwargs set the inputs (dropout does not matter for a fit model)."""
    state_dict = torch.load(path, map_location='cpu')
    model = gru_class(hidden_size=gru_hidden_size(state_dict), **kwargs)
    model.load_state_dict(state_dict)
    return model


@lru_cache
def model_setup(study: str, kind: str):
    """(path -> model, training data) for a study's `kind` models."""
    return _kolff2025_setup(kind) if study == 'kolff2025' else _registry_setup(study, kind)


def _kolff2025_setup(kind: str):
    """kolff2025 is not in the studies registry (utils/studies.py): its own loaders and evaluation data."""
    from weinhardt2026.studies.kolff2025.simulation_kolff2025 import load_spice
    from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import ModelInputGRU, get_dataset, evaluation_dataset
    if kind == 'benchmark':
        raise ValueError('kolff2025: the benchmark is fit in closed form, there are no checkpoints')
    dataset_train = evaluation_dataset(get_dataset(test_fold=0)[0])
    load = load_spice if kind == 'spice' else lambda path: load_gru(path, ModelInputGRU)
    return load, dataset_train


def _registry_setup(study: str, kind: str):
    from weinhardt2026.utils.studies import STUDIES, load_study, load_spice
    spice_module, dataset_train, _ = load_study(study)
    n_participants, n_actions = dataset_train.n_participants, dataset_train.n_actions
    if kind == 'spice':
        load = lambda path: load_spice(path, spice_module, dataset_train)
    else:
        load = lambda path: STUDIES[study][kind](path, n_participants, n_actions)
    return load, dataset_train


@lru_cache
def best_checkpoint(study: str, kind: str) -> str:
    """The final seed with the highest training log-likelihood per trial (SPICE: of its best member's SPICE-EQ)."""
    from weinhardt2026.utils.study_models import member_train_loglik, model_train_loglik
    load, dataset_train = model_setup(study, kind)

    scores = {}
    for path in final_checkpoints(study, kind):
        model = load(path)
        if kind == 'spice':
            scores[path] = float(member_train_loglik(study, model, dataset_train).max())
        else:
            scores[path] = model_train_loglik(study, model, dataset_train)
        del model
        torch.cuda.empty_cache()

    best = max(scores, key=scores.get)
    print(f'{study} {kind}: {os.path.basename(best)} (training log-lik. / trial {scores[best]:.4f})')
    return best


def load_best(study: str, kind: str):
    """The model of `best_checkpoint`, built as analysis/evaluation/run_model_evaluation.py registers it."""
    load, _ = model_setup(study, kind)
    return load(best_checkpoint(study, kind))


def seed_matrices(study: str) -> dict:
    """Per final SPICE seed (checkpoint name): each module's coefficient values, presence (ensemble majority) and
    term names (spice.utils.analysis.coefficient_dendrogram.get_module_matrices). Cached in
    studies/<study>/results/equation_stability/matrices.pkl."""
    from spice.utils.analysis.coefficient_dendrogram import get_module_matrices
    cache = os.path.join(ROOT, 'weinhardt2026', 'studies', study, 'results', 'equation_stability', 'matrices.pkl')
    checkpoints = final_checkpoints(study, 'spice')
    if os.path.exists(cache):
        with open(cache, 'rb') as file:
            matrices = pickle.load(file)
        if sorted(matrices) == sorted(map(os.path.basename, checkpoints)):
            return matrices
    load, _ = model_setup(study, 'spice')
    matrices = {os.path.basename(path): get_module_matrices(load(path)) for path in checkpoints}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'wb') as file:
        pickle.dump(matrices, file)
    return matrices

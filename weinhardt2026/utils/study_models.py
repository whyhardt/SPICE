"""Loading a study's data and final models for simulation, and per-member likelihoods of SPICE ensembles.

Every study has ``weinhardt2026/studies/<study>/simulation_<study>.py`` with
    load_dataset()             full dataset (all blocks), built as the models were trained on it
    load_models(dataset)       {'GRU': [...], 'Benchmark': [...], 'SPICE': [EnsembleMember, ...]}: one model per
                               final seed (utils/final_models.py); for SPICE the ensemble member with the
                               highest training likelihood (select_spice_members)
    generate(model, dataset)   one simulated run of `dataset` -> SpiceDataset
used by analysis/analysis_posterior_predictive.py and confusion_matrix/generate_datasets.py.
"""

import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules import 'studies.<study>...' (as in run.py)

import numpy as np
import torch

from spice import SpiceDataset


def study_dir(study: str, *parts) -> str:
    return os.path.join(ROOT, 'weinhardt2026', 'studies', study, *parts)


def load_study_module(study: str):
    """The study's simulation module (studies/<study>/simulation_<study>.py)."""
    return importlib.import_module(f'weinhardt2026.studies.{study}.simulation_{study}')


def load_registered_dataset(study: str) -> SpiceDataset:
    """Full dataset as figures/figure_model_evaluation.py (and run.py) builds it."""
    from weinhardt2026.figures.figure_model_evaluation import load_dataset
    return load_dataset(study)[1]


def load_registered_models(study: str, dataset: SpiceDataset) -> dict:
    """Every final seed of GRU, benchmark and SPICE (best member per seed), built as figures/figure_model_evaluation.py registers them."""
    from weinhardt2026.figures.figure_model_evaluation import STUDIES as REGISTRY, load_study, load_spice
    from weinhardt2026.utils.final_models import final_checkpoints
    config = REGISTRY[study]
    n_participants, n_actions = dataset.n_participants, dataset.n_actions
    spice_module, dataset_train, _ = load_study(study)
    return {
        'GRU': [config['gru'](path, n_participants, n_actions) for path in final_checkpoints(study, 'gru')],
        'Benchmark': [config['benchmark'](path, n_participants, n_actions) for path in final_checkpoints(study, 'benchmark')],
        'SPICE': select_spice_members(
            study, final_checkpoints(study, 'spice'), lambda path: load_spice(path, spice_module, dataset_train), dataset_train,
        ),
    }


# -------------------------------------------------------------------------------------------
# SPICE MODEL SELECTION: PER SEED, THE ENSEMBLE MEMBER WITH THE HIGHEST TRAINING LIKELIHOOD
# -------------------------------------------------------------------------------------------

class EnsembleMember:
    """One ensemble member of a SpiceEstimator, called like the estimator (outputs keep a member axis of 1)."""

    def __init__(self, estimator, member: int):
        self.estimator = estimator
        self.member = member

    @property
    def device(self):
        return self.estimator.device

    def eval(self, use_sindy: bool = None) -> 'EnsembleMember':
        """Without `use_sindy` the current mode is kept (generation wrappers call eval() on their model)."""
        if use_sindy is not None:
            self.estimator.eval(use_sindy=use_sindy)
        return self

    def __call__(self, xs: torch.Tensor, state=None):
        outputs, state = self.estimator(xs, state)
        return outputs[self.member:self.member + 1], state


def trial_loglik(study: str, outputs: torch.Tensor, dataset: SpiceDataset) -> float:
    """Log-likelihood per trial of one model's outputs (B, T, W, A) on `dataset`.

    Discrete: mean log-probability of the chosen actions. Continuous: Gaussian log-likelihood with the
    model's MLE noise variance (likelihood definitions as in figures/figure_model_evaluation.py).
    """
    from weinhardt2026.figures.figure_model_evaluation import discrete_session_nll, continuous_session_sse
    if dataset.continuous_action:
        sse, n_trials, dim = continuous_session_sse(outputs, dataset, study)
        variance = sse.sum().item() / (n_trials.sum().item() * dim)
        return -0.5 * dim * (np.log(2 * np.pi * variance) + 1)
    nll, n_trials = discrete_session_nll(outputs, dataset)
    return -nll.sum().item() / n_trials.sum().item()


@torch.no_grad()
def member_train_loglik(study: str, estimator, dataset_train: SpiceDataset) -> np.ndarray:
    """Training log-likelihood per trial of every ensemble member's SPICE-EQ."""
    estimator.eval(use_sindy=True)
    estimator.model.init_state(batch_size=dataset_train.xs.shape[0])
    outputs, _ = estimator(dataset_train.xs.to(estimator.device))
    outputs = outputs.detach().cpu()   # (E, B, T, W, A)
    return np.array([trial_loglik(study, outputs_member, dataset_train) for outputs_member in outputs])


def model_train_loglik(study: str, model: torch.nn.Module, dataset_train: SpiceDataset) -> float:
    """Training log-likelihood per trial of a benchmark or GRU."""
    from weinhardt2026.figures.figure_model_evaluation import predict
    return trial_loglik(study, predict(model, dataset_train), dataset_train)


def select_spice_members(study: str, checkpoints: list[str], load, dataset_train: SpiceDataset) -> list[EnsembleMember]:
    """Per checkpoint (seed), the ensemble member with the highest training likelihood (load: path -> SpiceEstimator)."""
    if not checkpoints:
        raise FileNotFoundError(f'{study}: no SPICE checkpoints found')
    members = []
    for path in checkpoints:
        estimator = load(path)
        loglik = member_train_loglik(study, estimator, dataset_train)
        best = int(np.argmax(loglik))
        members.append(EnsembleMember(estimator, best))
        print(f'SPICE: {os.path.basename(path)}, member {best} (training log-lik. / trial {loglik[best]:.4f})')
    return members


# -------------------------------------------------------------------------------------------
# DATA HELPERS (arrays are (n_sessions, n_trials), NaN = padded)
# -------------------------------------------------------------------------------------------

def valid_trials(dataset: SpiceDataset) -> np.ndarray:
    return ~torch.isnan(dataset.xs[:, :, 0, 0]).numpy()


def choices_and_rewards(dataset: SpiceDataset) -> tuple[np.ndarray, np.ndarray]:
    """Chosen action index and the scalar reward (sum over the reward columns), NaN where padded."""
    xs = dataset.xs[:, :, 0]
    n_actions = dataset.n_actions
    valid = valid_trials(dataset)
    choices = xs[..., :n_actions].argmax(dim=-1).float().numpy()
    rewards = xs[..., n_actions:n_actions + dataset.n_reward_features].nan_to_num(0.).sum(dim=-1).numpy()
    choices[~valid] = np.nan
    rewards[~valid] = np.nan
    return choices, rewards

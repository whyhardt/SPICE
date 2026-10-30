"""
Posterior predictive checks: simulate every fitted model in the study's task and compare the simulated
behavior with the real participants' behavior on summary statistics.

Discrete studies (dezfouli2019, ganesh2024a, eckstein2026, bustamante2023, braun2018, kolff2025):
    Every model instance (final seed of GRU, Benchmark, SPICE-EQ, SPICE-RNN) runs the experiment N_RUNS
    times, run r with torch seed r. Every statistic is a ratio of sums over a participant's sessions
    (e.g. P(Switch | cell) = switches in the cell / decisions in the cell); a conditional probability
    needs at least MIN_COUNT decisions of the participant (per run for the violins, over runs for the scores).

    nRMSE per statistic and instance: RMSE across participants between the real value and the instance's
    value pooled over runs, divided by SD(real) across participants; lower is better. A grid of cells
    (e.g. P(Switch | point difference x repeats)) is one panel: its score is the mean over its cells.
    The instance's overall score is the mean over the panels; reported as mean and SD over instances.
    kolff2025's models are group models (one fit for all apes; the benchmark a single closed-form fit), so
    their participant values differ only through the replayed partner acts.

Continuous studies (bruckner2025, weber2024):
    The models predict deterministically, so every instance makes one autonomous rollout of the
    experiment (its own predictions are its responses). Score: per-trial MSE between the rollout and the
    real responses (bruckner2025: bucket position in px², weber2024: shield angle in deg²).

All blocks of the data are simulated (models: final seeds, weinhardt2026/utils/final_models.py; SPICE:
the ensemble member with the highest training likelihood per seed). Study loading and simulation use the
study's simulation_<study>.py (load_dataset, load_models, generate; utils/study_models.py).

Per study, written to weinhardt2026/studies/<study>/results/posterior_predictive/:
    ppc_<panel>.png/.pdf         discrete: one figure for the single statistics ('scalars') and one per
                                 grid; violins over participants x runs (x instances) per model,
                                 participants as dots (models: pooled over instances and runs) joined
                                 Benchmark - Real - SPICE-EQ - SPICE-RNN.
                                 continuous: 'rollout', per-participant rollout MSE per model.
    ppc_bins.json                the bins of the grids, from the real data (95% of the decisions captured)
    ppc_participants.csv         per (model, participant, statistic) value (models: pooled over instances and runs)
    ppc_scores_instances.csv     discrete: per (model, instance, seed, panel, statistic) nRMSE; statistic 'all' = panel
                                 score, panel 'all' = overall score. continuous: per (model, instance, seed) MSE
                                 (panel = statistic = 'all').
    ppc_scores.csv               the same, mean and SD over instances
    cache_runs<N>_bins<hash>/    simulated counts per (model, instance) for these bins; re-used unless --recompute

figures/figure_model_evaluation.py reads the overall scores (ppc_scores_instances.csv, panel 'all').

Usage:
    python weinhardt2026/analysis/analysis_posterior_predictive.py all
    python weinhardt2026/analysis/analysis_posterior_predictive.py braun2018 dezfouli2019 --n_runs 10
    python weinhardt2026/analysis/analysis_posterior_predictive.py all --recompute
"""

import os

os.environ.setdefault('TQDM_DISABLE', '1')   # generate_behavior shows a progress bar per run

import argparse
import contextlib
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026'))  # study modules import 'studies.<study>...' (as in run.py)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from spice import SpiceDataset
from weinhardt2026.utils.study_models import load_study_module, study_dir, choices_and_rewards, valid_trials
from weinhardt2026.utils.final_models import checkpoint_seed, final_checkpoints


N_RUNS = 100            # runs per model instance, run r with torch seed r

MODEL_ORDER = ['GRU', 'Benchmark', 'Real', 'SPICE-EQ', 'SPICE-RNN']
SIMULATED_MODELS = [model for model in MODEL_ORDER if model != 'Real']
# model colors as in figures/figure_model_evaluation.py; Real is neutral ink
COLORS = {
    'GRU':       '#5b8db8',
    'Benchmark': '#1a3a5c',
    'Real':      '#6b6b6b',
    'SPICE-EQ':  '#8b2529',
    'SPICE-RNN': '#c44e52',
}
MIN_COUNT = 5           # decisions of a participant for a conditional probability
MIN_PARTICIPANTS = 10   # participants for an nRMSE
COVERAGE = 0.95         # share of the real decisions captured by the grids
DOT_MODELS = ['Benchmark', 'Real', 'SPICE-EQ', 'SPICE-RNN']   # participants as dots, joined in this order
KIND_TO_MODELS = {'GRU': ['GRU'], 'Benchmark': ['Benchmark'], 'SPICE': ['SPICE-EQ', 'SPICE-RNN']}


# -------------------------------------------------------------------------------------------
# PANELS AND STUDY SPECIFICATIONS
# -------------------------------------------------------------------------------------------

@dataclass
class Panel:
    """One statistic (grid of 1 x 1) or a grid of statistics (one per cell); scored as one unit."""
    name: str
    title: str
    row_labels: list = field(default_factory=lambda: [''])
    col_labels: list = field(default_factory=lambda: [''])
    xlabel: str = ''
    ylabel: str = ''
    min_count: int = 1

    @property
    def is_grid(self) -> bool:
        return len(self.row_labels) * len(self.col_labels) > 1

    def stat(self, row: int, col: int) -> str:
        return f'{self.name}[{row},{col}]' if self.is_grid else self.name

    @property
    def stats(self) -> list[str]:
        return [self.stat(row, col) for row in range(len(self.row_labels)) for col in range(len(self.col_labels))]


@dataclass
class StudySpec:
    """counts(dataset, bins) -> {statistic: (numerator, denominator)}, both (n_sessions,);
    bins(real dataset) -> dict (JSON-serializable); panels(bins) -> [Panel]."""
    counts: Callable
    panels: Callable
    bins: Callable = lambda dataset: {}


def _ratio(values: np.ndarray, mask: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Per session: sum of `values` and number of trials in `mask` (NaN values excluded)."""
    keep = ~np.isnan(values) if mask is None else mask & ~np.isnan(values)
    return np.where(keep, values, 0.).sum(axis=1), keep.sum(axis=1).astype(float)


def _bin_index(values: np.ndarray, edges: list, closed_last: bool = False) -> np.ndarray:
    """Bin i = [edges[i], edges[i + 1]) (the last bin closed if `closed_last`); -1 outside (or NaN)."""
    edges = np.asarray(edges, dtype=float)
    index = np.searchsorted(edges, values, side='right') - 1
    if closed_last:
        index[values == edges[-1]] = len(edges) - 2
    index[np.isnan(values) | (index < 0) | (index > len(edges) - 2)] = -1
    return index


def _grid_counts(panel: str, values: np.ndarray, mask: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 n_rows: int, n_cols: int) -> dict:
    return {
        f'{panel}[{row},{col}]': _ratio(values, mask & (rows == row) & (cols == col))
        for row in range(n_rows) for col in range(n_cols)
    }


def _integer_edges(values: np.ndarray, n_bins: int, n_single: int = 4) -> list[int]:
    """Up to `n_bins` bins of integer `values` from their minimum to the COVERAGE quantile: the first
    `n_single` values one bin each, the rest log-spaced (counts like repeats are right-skewed)."""
    lower, upper = int(np.min(values)), int(np.floor(np.quantile(values, COVERAGE)))
    singles = np.arange(lower, min(lower + n_single, upper + 1))
    start = lower + len(singles)
    spaced = np.round(np.geomspace(start, upper + 1, n_bins - len(singles) + 1)) if start <= upper else [upper + 1]
    return [int(edge) for edge in np.unique(np.concatenate([singles, spaced]))]


def _integer_labels(edges: list[int]) -> list[str]:
    return [f'{low}' if high - low == 1 else f'{low}–{high - 1}' for low, high in zip(edges[:-1], edges[1:])]


def _quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    """`n_bins` quantile bins over the central COVERAGE of `values`."""
    tail = (1 - COVERAGE) / 2
    return [float(edge) for edge in np.quantile(values, np.linspace(tail, 1 - tail, n_bins + 1))]


# ── Braun 2018: reward-based voluntary task switching ──────────────────────────────────────
# Action 1 = switch. The inputs of trial t ([difference = other - current, current_decreased,
# other_increased]) are the state shown before the choice of trial t + 1 (timeshift -1), so they
# condition the decision choices[:, t + 1]. Repeats = trials in the current task up to t.

def _braun_state(dataset: SpiceDataset):
    switches, _ = choices_and_rewards(dataset)
    xs = dataset.xs[:, :, 0].numpy()
    first_input = dataset.n_actions + dataset.n_reward_features
    difference, current_decreased, other_increased = (xs[:, :-1, first_input + i] for i in range(3))
    repeats = np.full_like(switches, np.nan)
    run = np.zeros(switches.shape[0])
    for t in range(switches.shape[1]):
        run = np.where((switches[:, t] == 1) | (t == 0), 1., run + 1.)
        repeats[:, t] = np.where(np.isnan(switches[:, t]), np.nan, run)
    decision = switches[:, 1:]
    decided = ~np.isnan(decision) & ~np.isnan(switches[:, :-1])
    return decision, decided, difference, current_decreased, other_increased, repeats[:, :-1]


def _braun_bins(dataset: SpiceDataset) -> dict:
    _, decided, difference, _, _, repeats = _braun_state(dataset)
    tail = (1 - COVERAGE) / 2
    low, high = np.quantile(difference[decided], [tail, 1 - tail])
    return dict(difference=[int(d) for d in np.arange(np.floor(low), np.ceil(high) + 1)],
                repeats=_integer_edges(repeats[decided], n_bins=7))


def _braun_counts(dataset: SpiceDataset, bins: dict) -> dict:
    decision, decided, difference, current_decreased, other_increased, repeats = _braun_state(dataset)
    differences = bins['difference']
    cols = np.full(difference.shape, -1)
    for index, value in enumerate(differences):
        cols[difference == value] = index
    rows = _bin_index(repeats, bins['repeats'])
    counts = _grid_counts('switch_difference_repeats', decision, decided, rows, cols,
                          len(bins['repeats']) - 1, len(differences))
    counts.update(_grid_counts('switch_change', decision, decided, np.nan_to_num(current_decreased, nan=-1).astype(int),
                               np.nan_to_num(other_increased, nan=-1).astype(int), 2, 2))
    return counts


def _braun_panels(bins: dict) -> list[Panel]:
    return [
        Panel('switch_difference_repeats', 'P(Switch | Point Difference, Repeats)',
              row_labels=_integer_labels(bins['repeats']), col_labels=[f'{d:+d}' for d in bins['difference']],
              xlabel='Point difference (other − current)', ylabel='Repeats of the current task', min_count=MIN_COUNT),
        Panel('switch_change', 'P(Switch | Value Change)',
              row_labels=['Current constant', 'Current decrease'], col_labels=['Other constant', 'Other increase'],
              min_count=MIN_COUNT),
    ]


# ── Bustamante 2023: patch foraging ────────────────────────────────────────────────────────
# Action 0 = harvest, 1 = exit. After a harvest at trial t, the decision choices[:, t + 1] is taken with
# the patch's harvests so far and the last reward (the exit threshold when the participant exits).
# Inputs: [harvest_duration, travel_duration, is_msit, is_high].

@lru_cache
def _bustamante_reward_scale() -> float:
    from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import DATA_PATH
    return float(pd.read_csv(DATA_PATH, usecols=['reward']).reward.max())


def _bustamante_state(dataset: SpiceDataset):
    choices, rewards = choices_and_rewards(dataset)
    harvests = np.zeros_like(choices)
    run = np.zeros(choices.shape[0])
    for t in range(choices.shape[1]):
        run = np.where(choices[:, t] == 0, run + 1., 0.)
        harvests[:, t] = run
    in_patch = (choices[:, :-1] == 0) & ~np.isnan(choices[:, 1:])
    exit_next = (choices[:, 1:] == 1).astype(float)
    return choices, rewards, in_patch, exit_next, harvests[:, :-1], rewards[:, :-1]


def _bustamante_bins(dataset: SpiceDataset) -> dict:
    _, _, in_patch, exit_next, harvests, last_reward = _bustamante_state(dataset)
    exits = in_patch & (exit_next == 1)
    return dict(harvests=_integer_edges(harvests[exits], n_bins=6),
                exit_reward=_quantile_edges(last_reward[exits], n_bins=5))


def _bustamante_counts(dataset: SpiceDataset, bins: dict) -> dict:
    choices, rewards, in_patch, exit_next, harvests, last_reward = _bustamante_state(dataset)
    rows = _bin_index(harvests, bins['harvests'])
    cols = _bin_index(last_reward, bins['exit_reward'], closed_last=True)
    counts = _grid_counts('exit_harvests_reward', exit_next, in_patch, rows, cols,
                          len(bins['harvests']) - 1, len(bins['exit_reward']) - 1)

    counts['exit_reward'] = _ratio(last_reward, in_patch & (exit_next == 1))
    first_input = dataset.n_actions + dataset.n_reward_features
    xs = dataset.xs[:, :, 0].numpy()
    duration = np.where(choices == 0, xs[..., first_input], xs[..., first_input + 1])
    duration[np.isnan(choices)] = np.nan
    counts['reward_rate'] = (np.nansum(rewards, axis=1), np.nansum(duration, axis=1))
    return counts


def _bustamante_panels(bins: dict) -> list[Panel]:
    scale = _bustamante_reward_scale()
    edges = bins['exit_reward']
    return [
        Panel('exit_harvests_reward', 'P(Exit | Harvests, Reward)',
              row_labels=_integer_labels(bins['harvests']),
              col_labels=[f'{low * scale:.1f}–{high * scale:.1f}' for low, high in zip(edges[:-1], edges[1:])],
              xlabel='Last reward (apples; exit threshold bins)', ylabel='Harvests in the patch', min_count=MIN_COUNT),
        Panel('exit_reward', 'Mean Exit Reward'),
        Panel('reward_rate', 'Reward Rate (reward / s)'),
    ]


# ── Bandits: Dezfouli 2019, Eckstein 2026, Ganesh 2024a ────────────────────────────────────
# best(dataset) -> (n_sessions, n_trials) 1 if the chosen arm is the best arm, NaN where padded.
# Win = reward > 0.5 (eckstein2026: payouts min-max scaled to [0, 1]).

N_LEARNING_BINS = 5


def _bandit_counts(dataset: SpiceDataset, best: np.ndarray) -> dict:
    choices, rewards = choices_and_rewards(dataset)
    valid = ~np.isnan(choices)
    stay = (choices[:, 1:] == choices[:, :-1]).astype(float)
    decided = valid[:, 1:] & valid[:, :-1]
    previous_reward = rewards[:, :-1]

    position = np.cumsum(valid, axis=1) - 1
    progress = np.floor(N_LEARNING_BINS * position / np.maximum(valid.sum(axis=1, keepdims=True), 1)).astype(int)
    counts = _grid_counts('learning_curve', best, valid, np.zeros_like(progress), np.where(valid, progress, -1),
                          1, N_LEARNING_BINS)
    counts.update({
        'p_best': _ratio(best, valid),
        'switch_lose': _ratio(1. - stay, decided & (previous_reward <= 0.5)),
        'stay_win': _ratio(stay, decided & (previous_reward > 0.5)),
        'reward_rate': _ratio(rewards, valid),
    })
    return counts


def _bandit_panels(bins: dict) -> list[Panel]:
    return [
        Panel('p_best', 'P(Best Arm)'),
        Panel('switch_lose', 'P(Switch | Lose)', min_count=MIN_COUNT),
        Panel('stay_win', 'P(Stay | Win)', min_count=MIN_COUNT),
        Panel('reward_rate', 'Reward Rate (reward / trial)'),
        Panel('learning_curve', 'Learning Curve: P(Best Arm)',
              col_labels=[f'{100 * i // N_LEARNING_BINS}–{100 * (i + 1) // N_LEARNING_BINS}%' for i in range(N_LEARNING_BINS)],
              xlabel='Block progress (share of trials)', min_count=MIN_COUNT),
    ]


def _chose(choices: np.ndarray, arm: np.ndarray) -> np.ndarray:
    best = (choices == arm).astype(float)
    best[np.isnan(choices)] = np.nan
    return best


@lru_cache
def _dezfouli_probabilities() -> torch.Tensor:
    from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import session_reward_probabilities
    return session_reward_probabilities()


def _dezfouli_counts(dataset: SpiceDataset, bins: dict) -> dict:
    """Best arm: the higher reward probability of the session (from the raw data's condition)."""
    choices, _ = choices_and_rewards(dataset)
    metadata = dataset.xs[:, :, 0, -3:].nan_to_num(-1).max(dim=1).values.long()   # trial 0 may be padded
    probabilities = _dezfouli_probabilities()[metadata[:, -1], metadata[:, 0]]
    return _bandit_counts(dataset, _chose(choices, probabilities.argmax(dim=-1).numpy()[:, None]))


@lru_cache
def _payout_schedule(study: str) -> np.ndarray:
    """(n_sessions, n_trials, n_arms) outcome of every arm on every trial, aligned with the study's dataset."""
    module = load_study_module(study)
    return module.get_reward_schedule(module.PATH_DATA).numpy()


def _eckstein_counts(dataset: SpiceDataset, bins: dict) -> dict:
    """Best arm: the highest payout on the trial. Cyclic sequence: 4 consecutive choices that visit all arms
    in one rotational direction (each step +1 or each step -1 mod 4); counted per block."""
    choices, _ = choices_and_rewards(dataset)
    payouts = _payout_schedule('eckstein2026')[:, :choices.shape[1]]
    counts = _bandit_counts(dataset, _chose(choices, payouts.argmax(axis=-1)))

    n_arms = dataset.n_actions
    step = (choices[:, 1:] - choices[:, :-1]) % n_arms
    rotation = (step == 1) | (step == n_arms - 1)
    window = n_arms - 1
    cyclic = np.ones((choices.shape[0], step.shape[1] - window + 1), dtype=bool)
    for k in range(window):
        cyclic &= rotation[:, k:k + cyclic.shape[1]] & (step[:, k:k + cyclic.shape[1]] == step[:, :cyclic.shape[1]])
    has_trials = (~np.isnan(choices)).sum(axis=1) > n_arms
    counts['cyclic'] = (cyclic.sum(axis=1).astype(float), has_trials.astype(float))
    return counts


def _ganesh_contrast(dataset: SpiceDataset) -> tuple[np.ndarray, np.ndarray]:
    """Absolute contrast difference (NaN where padded) and whether the left side has the higher contrast."""
    choices, _ = choices_and_rewards(dataset)
    contrast = dataset.xs[:, :, 0, 2 * dataset.n_actions].numpy()
    return np.where(np.isnan(choices), np.nan, np.abs(contrast)), contrast > 0


def _ganesh_bins(dataset: SpiceDataset) -> dict:
    abs_contrast, _ = _ganesh_contrast(dataset)
    return dict(contrast=_quantile_edges(abs_contrast[~np.isnan(abs_contrast)], n_bins=5))


def _ganesh_counts(dataset: SpiceDataset, bins: dict) -> dict:
    """Best arm: the side the block's contingency favors (higher- or lower-contrast side, read from the
    replayed outcomes, so the same for real and simulated behavior)."""
    choices, _ = choices_and_rewards(dataset)
    valid = ~np.isnan(choices)
    abs_contrast, left_is_high = _ganesh_contrast(dataset)
    schedule = _payout_schedule('ganesh2024a')[:, :choices.shape[1]]
    reward_high = np.where(left_is_high, schedule[..., 0], schedule[..., 1])
    high_favored = _ratio(reward_high, valid)[0] > 0.5 * valid.sum(axis=1)
    best = ((choices == 0) == left_is_high) == high_favored[:, None]
    best = np.where(valid, best.astype(float), np.nan)

    counts = _bandit_counts(dataset, best)
    cols = _bin_index(abs_contrast, bins['contrast'], closed_last=True)
    counts.update(_grid_counts('best_contrast', best, valid, np.zeros_like(cols), cols, 1, len(bins['contrast']) - 1))
    return counts


def _ganesh_panels(bins: dict) -> list[Panel]:
    edges = bins['contrast']
    return _bandit_panels(bins) + [
        Panel('best_contrast', 'P(Best Arm | Contrast Difference)',
              col_labels=[f'{low:.2f}–{high:.2f}' for low, high in zip(edges[:-1], edges[1:])],
              xlabel='|Contrast difference|', min_count=MIN_COUNT),
    ]


# ── Kolff 2025: grooming interactions (group models) ───────────────────────────────────────
# Decision events: the focal ape acts, from the second event on; groom = its act is grooming
# (as in studies/kolff2025/simulation_kolff2025.py; the partner's acts are replayed).

def _kolff_counts(dataset: SpiceDataset, bins: dict) -> dict:
    from weinhardt2026.studies.kolff2025.simulation_kolff2025 import COLUMNS_PARTNER, COLUMNS_SELF, SELF_COLUMNS
    xs = dataset.xs[:, :, 0].numpy()
    valid = ~np.isnan(xs[..., 0])
    own = np.nan_to_num(xs[..., COLUMNS_SELF])
    partner = np.nan_to_num(xs[..., COLUMNS_PARTNER])
    category = {name: index for index, name in enumerate(c.split('_', 1)[1] for c in SELF_COLUMNS)}

    decision = np.zeros_like(valid)
    decision[:, 1:] = valid[:, 1:] & (own[:, 1:].sum(axis=-1) > 0)
    groom = np.where(decision, own[..., category['groom']], np.nan)
    previous_partner = np.zeros_like(partner)
    previous_partner[:, 1:] = partner[:, :-1]
    previous_own_groom = np.zeros_like(valid)
    previous_own_groom[:, 1:] = own[:, :-1, category['groom']] == 1

    groomed = groom == 1
    first = groomed.argmax(axis=1)
    acts_before = np.cumsum(decision, axis=1) - decision
    any_groom = groomed.any(axis=1)
    latency = np.where(any_groom, acts_before[np.arange(len(first)), first], 0.)
    return {
        'p_groom': _ratio(groom, decision),
        'groom_partner_solicitation': _ratio(groom, previous_partner[..., category['grooming_solicitation']] == 1),
        'groom_partner_preparation': _ratio(groom, (previous_partner[..., category['reposition_body']] == 1)
                                            | (previous_partner[..., category['grooming_process']] == 1)),
        'groom_partner_groom': _ratio(groom, previous_partner[..., category['groom']] == 1),
        'groom_own_groom': _ratio(groom, previous_own_groom),
        'first_groom_latency': (latency, any_groom.astype(float)),
    }


def _kolff_panels(bins: dict) -> list[Panel]:
    return [
        Panel('p_groom', 'P(Groom)'),
        Panel('groom_partner_solicitation', 'P(Groom | Partner Solicited)', min_count=MIN_COUNT),
        Panel('groom_partner_preparation', 'P(Groom | Partner Prepared)', min_count=MIN_COUNT),
        Panel('groom_partner_groom', 'P(Groom | Partner Groomed)', min_count=MIN_COUNT),
        Panel('groom_own_groom', 'P(Groom | Own Groom)', min_count=MIN_COUNT),
        Panel('first_groom_latency', 'Own Acts Before First Groom'),
    ]


STUDY_SPECS = {
    'dezfouli2019': StudySpec(counts=_dezfouli_counts, panels=_bandit_panels),
    'ganesh2024a': StudySpec(counts=_ganesh_counts, panels=_ganesh_panels, bins=_ganesh_bins),
    'eckstein2026': StudySpec(counts=_eckstein_counts,
                              panels=lambda bins: _bandit_panels(bins) + [Panel('cyclic', 'Cyclic Sequences / Block')]),
    'bustamante2023': StudySpec(counts=_bustamante_counts, panels=_bustamante_panels, bins=_bustamante_bins),
    'braun2018': StudySpec(counts=_braun_counts, panels=_braun_panels, bins=_braun_bins),
    'kolff2025': StudySpec(counts=_kolff_counts, panels=_kolff_panels),
}

# continuous studies: rollout error in the response's units
ROLLOUT_UNITS = {'bruckner2025': 'px²', 'weber2024': 'deg²'}
STUDIES = tuple(STUDY_SPECS) + tuple(ROLLOUT_UNITS)


# -------------------------------------------------------------------------------------------
# SIMULATION
# -------------------------------------------------------------------------------------------

def _seeds(study: str, kind: str) -> list[int]:
    """Seeds of the final checkpoints of `kind`; [0] for a single fit without checkpoints (kolff2025's benchmark)."""
    try:
        return [checkpoint_seed(path) for path in final_checkpoints(study, kind.lower())]
    except (FileNotFoundError, ValueError):
        return [0]


def _model_instances(study: str, models: dict) -> list[tuple[str, int, int, object]]:
    """(model name, instance, seed of the checkpoint, model) for every simulated model."""
    instances = []
    for kind, names in KIND_TO_MODELS.items():
        fits = models[kind] if isinstance(models[kind], (list, tuple)) else [models[kind]]
        seeds = _seeds(study, kind)
        for name in names:
            instances += [(name, instance, seeds[instance], model) for instance, model in enumerate(fits)]
    return sorted(instances, key=lambda item: SIMULATED_MODELS.index(item[0]))


def _set_mode(name: str, model):
    if name.startswith('SPICE'):
        model.eval(use_sindy=name == 'SPICE-EQ')
    else:
        model.eval()


def _generate(study_module, model, dataset: SpiceDataset, seed: int) -> SpiceDataset:
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return study_module.generate(model, dataset)


def _participant_counts(counts: dict, participants: np.ndarray) -> pd.DataFrame:
    """Session counts -> one row per participant: numerators and denominators summed over sessions."""
    columns = {}
    for stat, (numerator, denominator) in counts.items():
        columns[f'{stat}|num'] = numerator
        columns[f'{stat}|den'] = denominator
    df = pd.DataFrame(columns)
    df.insert(0, 'participant_id', participants)
    return df.groupby('participant_id', as_index=False).sum()


def _cache_path(cache_dir: str, name: str, instance: int) -> str:
    return os.path.join(cache_dir, f'{name}_{instance}.csv.gz')


def simulate_counts(study: str, spec: StudySpec, bins: dict, n_runs: int, recompute: bool) -> pd.DataFrame:
    """Per (model, instance, seed, run, participant) counts of every statistic, Real included (instance 0, run 0).

    Every (model, instance) is cached; the models are only loaded if one of them is missing (or `recompute`).
    """
    study_module = load_study_module(study)
    dataset = study_module.load_dataset()
    participants = dataset.xs[:, :, 0, -1].nan_to_num(-1).max(dim=1).values.long().numpy()
    bins_hash = hashlib.sha1(json.dumps(bins, sort_keys=True).encode()).hexdigest()[:8]   # cell counts depend on the bins
    cache_dir = study_dir(study, 'results', 'posterior_predictive', f'cache_runs{n_runs}_bins{bins_hash}')
    os.makedirs(cache_dir, exist_ok=True)

    frames = [_participant_counts(spec.counts(dataset, bins), participants).assign(model='Real', instance=0, seed=0, run=0)]
    n_instances = {name: len(_seeds(study, kind)) for kind, names in KIND_TO_MODELS.items() for name in names}
    cached = all(os.path.isfile(_cache_path(cache_dir, name, instance)) for name, n in n_instances.items() for instance in range(n))
    if cached and not recompute:
        return pd.concat(frames + [pd.read_csv(_cache_path(cache_dir, name, instance))
                                   for name, n in n_instances.items() for instance in range(n)], ignore_index=True)

    for name, instance, seed, model in _model_instances(study, study_module.load_models(dataset)):
        path = _cache_path(cache_dir, name, instance)
        if os.path.isfile(path) and not recompute:
            frames.append(pd.read_csv(path))
            continue
        print(f'{study}: simulating {name} instance {instance} (seed {seed}), {n_runs} runs', flush=True)
        _set_mode(name, model)
        runs = []
        for run in range(n_runs):
            dataset_sim = _generate(study_module, model, dataset, seed=run)
            runs.append(_participant_counts(spec.counts(dataset_sim, bins), participants).assign(run=run))
        df = pd.concat(runs, ignore_index=True).assign(model=name, instance=instance, seed=seed)
        df.to_csv(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------------------------------------------
# SCORES (discrete)
# -------------------------------------------------------------------------------------------

def _values(df: pd.DataFrame, stat: str, min_count: int) -> pd.Series:
    """Ratio numerator / denominator, NaN below `min_count`."""
    denominator = df[f'{stat}|den']
    return (df[f'{stat}|num'] / denominator.where(denominator > 0)).where(denominator >= min_count)


def _pooled(df: pd.DataFrame, panels: list[Panel]) -> pd.DataFrame:
    """Participant values pooled over the rows of `df` (runs, instances): one row per participant."""
    sums = df.groupby('participant_id').sum(numeric_only=True)
    return pd.DataFrame({stat: _values(sums, stat, panel.min_count) for panel in panels for stat in panel.stats})


def _nrmse(real: pd.Series, simulated: pd.Series) -> float:
    paired = pd.concat([real, simulated], axis=1, join='inner').dropna().to_numpy()
    if len(paired) < MIN_PARTICIPANTS or paired[:, 0].std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(np.mean((paired[:, 1] - paired[:, 0]) ** 2)) / paired[:, 0].std(ddof=1))


def score_discrete(df: pd.DataFrame, panels: list[Panel]) -> pd.DataFrame:
    """nRMSE per (model, instance, panel, statistic); statistic 'all' = mean over the panel's statistics,
    panel 'all' = mean over the panels."""
    real = _pooled(df[df['model'] == 'Real'], panels)
    rows = []
    for (model, instance, seed), simulated in df[df['model'] != 'Real'].groupby(['model', 'instance', 'seed']):
        simulated = _pooled(simulated, panels)
        panel_scores = []
        for panel in panels:
            stat_scores = [_nrmse(real[stat], simulated[stat]) for stat in panel.stats]
            rows += [dict(model=model, instance=instance, seed=seed, panel=panel.name, statistic=stat, nrmse=score)
                     for stat, score in zip(panel.stats, stat_scores)]
            panel_score = np.nanmean(stat_scores) if np.isfinite(stat_scores).any() else np.nan
            if panel.is_grid:
                rows.append(dict(model=model, instance=instance, seed=seed, panel=panel.name, statistic='all', nrmse=panel_score))
            panel_scores.append(panel_score)
        rows.append(dict(model=model, instance=instance, seed=seed, panel='all', statistic='all', nrmse=np.nanmean(panel_scores)))
    return pd.DataFrame(rows)


def summarize(scores_instances: pd.DataFrame, value: str) -> pd.DataFrame:
    return scores_instances.groupby(['model', 'panel', 'statistic'], sort=False)[value].agg(['mean', 'std']).reset_index()


# -------------------------------------------------------------------------------------------
# ROLLOUTS (continuous)
# -------------------------------------------------------------------------------------------

def _responses(study: str, dataset: SpiceDataset) -> np.ndarray:
    """Responses in their units: bruckner2025 bucket position (px), weber2024 shield angle (deg); NaN where padded."""
    xs = dataset.xs[:, :, 0].numpy()
    if study == 'bruckner2025':
        from weinhardt2026.studies.bruckner2025.spice_bruckner2025 import POSITION_SCALE
        response = xs[..., 0] * POSITION_SCALE
    else:
        response = np.degrees(np.arctan2(xs[..., 0], xs[..., 1])) % 360
    return np.where(valid_trials(dataset), response, np.nan)


def _squared_error(study: str, real: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    if study == 'weber2024':
        difference = (simulated - real) % 360
        return np.minimum(difference, 360 - difference) ** 2
    return (simulated - real) ** 2


def simulate_rollouts(study: str, recompute: bool) -> pd.DataFrame:
    """Per (model, instance, seed, participant): summed squared rollout error and number of trials."""
    path = study_dir(study, 'results', 'posterior_predictive', 'cache_rollout.csv')
    if os.path.isfile(path) and not recompute:
        return pd.read_csv(path)
    study_module = load_study_module(study)
    dataset = study_module.load_dataset()
    participants = dataset.xs[:, :, 0, -1].nan_to_num(-1).max(dim=1).values.long().numpy()
    real = _responses(study, dataset)
    frames = []
    for name, instance, seed, model in _model_instances(study, study_module.load_models(dataset)):
        print(f'{study}: rollout of {name} instance {instance} (seed {seed})', flush=True)
        _set_mode(name, model)
        error = _squared_error(study, real, _responses(study, _generate(study_module, model, dataset, seed=0)))
        sse, n_trials = _ratio(error)
        df = pd.DataFrame(dict(participant_id=participants, sse=sse, n_trials=n_trials))
        frames.append(df.groupby('participant_id', as_index=False).sum().assign(model=name, instance=instance, seed=seed))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def score_rollouts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trial MSE per (model, instance), pooled over all trials."""
    scores = df.groupby(['model', 'instance', 'seed'], as_index=False)[['sse', 'n_trials']].sum()
    scores['mse'] = scores['sse'] / scores['n_trials']
    return scores.drop(columns=['sse', 'n_trials']).assign(panel='all', statistic='all')


# -------------------------------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------------------------------

def _style(ax, models: list[str], show_xticks: bool):
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models if show_xticks else [], rotation=35, ha='right', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', color='#e6e6e6', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _draw(ax, violins: dict, dots: pd.DataFrame, models: list[str], jitter: pd.Series, dot_size: float = 6):
    """Violins (values per model), participant dots (columns of `dots`) joined in DOT_MODELS order."""
    positions = {model: index for index, model in enumerate(models)}
    for model, values in violins.items():
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        x = positions[model]
        if np.ptp(values) > 0:
            body = ax.violinplot(values, positions=[x], widths=0.8, showextrema=False)['bodies'][0]
            body.set_facecolor(COLORS[model])
            body.set_edgecolor('none')
            body.set_alpha(0.3)
        ax.hlines(np.median(values), x - 0.3, x + 0.3, color=COLORS[model], linewidth=1.6, zorder=4)

    columns = [model for model in DOT_MODELS if model in dots.columns and model in positions]
    paired = dots[columns].dropna()
    if paired.empty:
        return
    x_paired = np.array([positions[model] for model in columns])
    offsets = jitter.reindex(paired.index).fillna(0.).to_numpy()[:, None]
    ax.plot((x_paired[None, :] + offsets).T, paired.to_numpy().T, color='#9a9a9a', linewidth=0.35, alpha=0.3, zorder=2)
    for column, model in enumerate(columns):
        ax.scatter(x_paired[column] + offsets[:, 0], paired[model], s=dot_size, color=COLORS[model],
                   alpha=0.75, linewidths=0, zorder=3)


def _score_text(scores: pd.DataFrame, panel: str, statistic: str, models: list[str]) -> dict:
    table = scores[(scores['panel'] == panel) & (scores['statistic'] == statistic)].set_index('model')
    return {model: table.loc[model, 'mean'] for model in models if model in table.index}


def _format(value: float) -> str:
    return '–' if value is None or np.isnan(value) else f'{value:.2f}'


def _add_headroom(ax, fraction: float = 0.15):
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + fraction * (top - bottom))


def _annotate_scores(ax, scores: pd.DataFrame, panel: str, statistic: str, models: list[str], fontsize: float = 7):
    """Each model's nRMSE vs Real for this statistic, at the top of the axes above its violin."""
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for model, value in _score_text(scores, panel, statistic, models).items():
        ax.text(models.index(model), 0.985, _format(value), transform=transform, ha='center', va='top',
                fontsize=fontsize, color='#333333')


def _save(fig, path: str):
    for extension in ('png', 'pdf'):
        fig.savefig(f'{path}.{extension}', bbox_inches='tight', dpi=200)
    plt.close(fig)


CAPTION = ('Violins: participants × runs × instances (Real: participants). Dots: participants (models: pooled over '
           'instances and runs), joined Benchmark – Real – SPICE-EQ – SPICE-RNN. nRMSE: RMSE to Real across participants '
           '/ SD(Real), mean over instances, lower is better.')


def plot_discrete(df: pd.DataFrame, panels: list[Panel], scores: pd.DataFrame, study: str, output_dir: str):
    """One figure with the single statistics, one per grid."""
    models = [model for model in MODEL_ORDER if (df['model'] == model).any()]
    run_values = {}   # model -> per (participant, run, instance) values of every statistic
    for model in models:
        rows = df[df['model'] == model]
        run_values[model] = pd.DataFrame({stat: _values(rows, stat, panel.min_count).to_numpy()
                                          for panel in panels for stat in panel.stats})
    dots = {model: _pooled(df[df['model'] == model], panels) for model in DOT_MODELS if model in models}
    participants = dots['Real'].index
    jitter = pd.Series(np.random.default_rng(0).uniform(-0.12, 0.12, len(participants)), index=participants)

    def draw_stat(ax, stat):
        _draw(ax, {model: values[stat].to_numpy() for model, values in run_values.items()},
              pd.DataFrame({model: values[stat] for model, values in dots.items()}), models, jitter)

    scalars = [panel for panel in panels if not panel.is_grid]
    if scalars:
        n_cols = min(4, len(scalars))
        n_rows = int(np.ceil(len(scalars) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.1 * n_rows + 0.6), squeeze=False)
        for ax, panel in zip(axes.flat, scalars):
            draw_stat(ax, panel.name)
            _add_headroom(ax)
            _annotate_scores(ax, scores, panel.name, panel.name, models)
            ax.set_title(panel.title, fontsize=9)
            _style(ax, models, show_xticks=True)
        for ax in axes.flat[len(scalars):]:
            ax.set_visible(False)
        fig.suptitle(f'{study}: posterior predictive checks', fontsize=10)
        fig.text(0.5, 0.0, CAPTION, ha='center', va='top', fontsize=6.5, color='#555555', wrap=True)
        fig.tight_layout()
        _save(fig, os.path.join(output_dir, 'ppc_scalars'))

    for panel in panels:
        if not panel.is_grid:
            continue
        n_rows, n_cols = len(panel.row_labels), len(panel.col_labels)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7 * n_cols + 1.0, 1.5 * n_rows + 1.3),
                                 sharey=True, squeeze=False)
        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row, col]
                draw_stat(ax, panel.stat(row, col))
                _style(ax, models, show_xticks=row == n_rows - 1)
                if row == 0:
                    ax.set_title(panel.col_labels[col], fontsize=8)
                if col == 0 and panel.row_labels[row]:
                    ax.set_ylabel(panel.row_labels[row], fontsize=8)
        _add_headroom(axes[0, 0], fraction=0.2)   # shared y-axis: once for all cells
        for row in range(n_rows):
            for col in range(n_cols):
                _annotate_scores(axes[row, col], scores, panel.name, panel.stat(row, col), models, fontsize=5.5)
        if panel.xlabel:
            fig.supxlabel(panel.xlabel, fontsize=9)
        if panel.ylabel:
            fig.supylabel(panel.ylabel, fontsize=9)
        overall = _score_text(scores, panel.name, 'all', models)
        fig.suptitle(f'{study}: {panel.title}\nnRMSE per cell above the violins; mean over cells — '
                     + '   '.join(f'{model} {_format(value)}' for model, value in overall.items()), fontsize=9)
        fig.text(0.5, 0.0, CAPTION, ha='center', va='top', fontsize=6.5, color='#555555', wrap=True)
        fig.tight_layout()
        _save(fig, os.path.join(output_dir, f'ppc_{panel.name}'))


def plot_rollouts(df: pd.DataFrame, scores: pd.DataFrame, study: str, output_dir: str):
    """Per-participant rollout MSE (pooled over instances) per model; violins over (participant, instance)."""
    unit = ROLLOUT_UNITS[study]
    models = [model for model in SIMULATED_MODELS if (df['model'] == model).any()]
    df = df.assign(mse=df['sse'] / df['n_trials'].where(df['n_trials'] > 0))
    sums = df.groupby(['model', 'participant_id'])[['sse', 'n_trials']].sum()
    dots = (sums['sse'] / sums['n_trials']).unstack('model')
    jitter = pd.Series(np.random.default_rng(0).uniform(-0.12, 0.12, len(dots)), index=dots.index)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    _draw(ax, {model: df.loc[df['model'] == model, 'mse'].to_numpy() for model in models}, dots, models, jitter)
    summary = _score_text(scores, 'all', 'all', models)
    ax.set_title(f'{study}: autonomous rollout\nMSE — ' + '   '.join(f'{model} {value:.1f}' for model, value in summary.items()),
                 fontsize=8)
    ax.set_ylabel(f'Per-trial MSE to the real responses ({unit})', fontsize=8)
    _style(ax, models, show_xticks=True)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, 'ppc_rollout'))


# -------------------------------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------------------------------

def run_posterior_predictive(study: str, n_runs: int = N_RUNS, recompute: bool = False) -> pd.DataFrame:
    """Simulate (or load the cache), score and plot one study; returns the scores per (model, panel, statistic)."""
    output_dir = study_dir(study, 'results', 'posterior_predictive')
    os.makedirs(output_dir, exist_ok=True)

    if study in ROLLOUT_UNITS:
        df = simulate_rollouts(study, recompute)
        scores_instances = score_rollouts(df)
        value = 'mse'
        sums = df.groupby(['model', 'participant_id'])[['sse', 'n_trials']].sum()
        participants = (sums['sse'] / sums['n_trials']).rename('value').reset_index().assign(statistic='mse')
    else:
        spec = STUDY_SPECS[study]
        path_bins = os.path.join(output_dir, 'ppc_bins.json')
        if os.path.isfile(path_bins) and not recompute:
            with open(path_bins) as file:
                bins = json.load(file)
        else:
            bins = spec.bins(load_study_module(study).load_dataset())
            with open(path_bins, 'w') as file:
                json.dump(bins, file, indent=2)
        panels = spec.panels(bins)
        df = simulate_counts(study, spec, bins, n_runs, recompute)
        scores_instances = score_discrete(df, panels)
        value = 'nrmse'
        participants = pd.concat([
            _pooled(df[df['model'] == model], panels).rename_axis('participant_id').reset_index()
            .melt(id_vars='participant_id', var_name='statistic', value_name='value').assign(model=model)
            for model in MODEL_ORDER if (df['model'] == model).any()
        ], ignore_index=True)

    scores = summarize(scores_instances, value)
    scores_instances.to_csv(os.path.join(output_dir, 'ppc_scores_instances.csv'), index=False)
    scores.to_csv(os.path.join(output_dir, 'ppc_scores.csv'), index=False)
    participants[['model', 'participant_id', 'statistic', 'value']].to_csv(os.path.join(output_dir, 'ppc_participants.csv'), index=False)

    if study in ROLLOUT_UNITS:
        plot_rollouts(df, scores, study, output_dir)
    else:
        plot_discrete(df, panels, scores, study, output_dir)

    overview = scores[scores['statistic'] == 'all'].pivot(index='panel', columns='model', values='mean')
    print(f'\n{study}: {value} (mean over instances, lower is better)')
    print(overview.reindex(columns=[model for model in SIMULATED_MODELS if model in overview.columns]).round(3).to_string())
    print(f'Saved to {output_dir}')
    return scores.assign(study=study)


def load_scores(study: str) -> pd.DataFrame:
    """Overall posterior predictive score per (model, seed): columns model, seed, ppc; empty if not computed."""
    path = study_dir(study, 'results', 'posterior_predictive', 'ppc_scores_instances.csv')
    if not os.path.isfile(path):
        return pd.DataFrame(columns=['model', 'seed', 'ppc'])
    df = pd.read_csv(path)
    df = df[(df['panel'] == 'all') & (df['statistic'] == 'all')]
    return df.rename(columns={'nrmse' if 'nrmse' in df.columns else 'mse': 'ppc'})[['model', 'seed', 'ppc']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Posterior predictive checks of every fitted model.')
    parser.add_argument('studies', nargs='+', help=f"study names or 'all' ({', '.join(STUDIES)})")
    parser.add_argument('--n_runs', type=int, default=N_RUNS, help='runs per model instance, seeds 0..n_runs-1 (discrete studies)')
    parser.add_argument('--recompute', action='store_true', help='ignore the cached simulations and bins')
    args = parser.parse_args()

    for study in (STUDIES if args.studies == ['all'] else args.studies):
        run_posterior_predictive(study, n_runs=args.n_runs, recompute=args.recompute)

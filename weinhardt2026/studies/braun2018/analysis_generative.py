"""
Generative analysis for braun2018 (reward-based voluntary task switching).

Study-specific part of weinhardt2026/analysis/analysis_generative_comparison.py. Run either
    python weinhardt2026/analysis/analysis_generative_comparison.py braun2018
or
    python weinhardt2026/studies/braun2018/analysis_generative.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from spice import SpiceDataset
from weinhardt2026.analysis.analysis_generative_comparison import (
    load_registered_dataset, load_registered_models, run_generative_analysis,
    choices_and_rewards, valid_trials, session_mean,
)
from weinhardt2026.studies.braun2018.benchmarking_braun2018 import generate_behavior


STUDY = 'braun2018'
N_REPEATS = 10   # runs per model (SPICE: one model per seed)

METRIC_LABELS = {
    'p_switch': 'P(Switch)',
    'p_switch_both_change': 'P(Switch | Both Change)',
    'p_switch_neither_change': 'P(Switch | Neither Change)',
    'p_switch_current_decrease': 'P(Switch | Current Decrease)',
    'p_switch_other_increase': 'P(Switch | Other Increase)',
    'p_switch_diff_neg1': 'P(Switch | Diff = -1)',
    'p_switch_diff_0': 'P(Switch | Diff = 0)',
    'p_switch_diff_pos1': 'P(Switch | Diff = +1)',
}


def load_dataset() -> SpiceDataset:
    return load_registered_dataset(STUDY)


def load_models(dataset: SpiceDataset) -> dict:
    return load_registered_models(STUDY, dataset)


def generate(model, dataset: SpiceDataset) -> SpiceDataset:
    return generate_behavior(model=model, dataset=dataset)


def compute_metrics(dataset: SpiceDataset) -> dict[str, np.ndarray]:
    """Action 1 = switch, 0 = repeat. Inputs: [difference, current_decreased, other_increased]
    (point difference other - current, and whether the current task lost / the other gained a point)."""
    switches, _ = choices_and_rewards(dataset)
    valid = valid_trials(dataset)
    xs = dataset.xs[:, :, 0].numpy()
    n_actions = dataset.n_actions
    difference, current_decreased, other_increased = (xs[..., n_actions + i] for i in range(3))
    return {
        'p_switch': session_mean(switches),
        'p_switch_both_change': session_mean(switches, valid & (current_decreased == 1) & (other_increased == 1)),
        'p_switch_neither_change': session_mean(switches, valid & (current_decreased == 0) & (other_increased == 0)),
        'p_switch_current_decrease': session_mean(switches, valid & (current_decreased == 1)),
        'p_switch_other_increase': session_mean(switches, valid & (other_increased == 1)),
        'p_switch_diff_neg1': session_mean(switches, valid & (difference == -1)),
        'p_switch_diff_0': session_mean(switches, valid & (difference == 0)),
        'p_switch_diff_pos1': session_mean(switches, valid & (difference == 1)),
    }


if __name__ == '__main__':
    run_generative_analysis(STUDY)

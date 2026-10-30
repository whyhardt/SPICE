"""Paper runner: equation discovery for every study (spice.utils.analysis.equation_discovery).

Per study, on its final SPICE seeds (studies/<study>/params/; the benchmark and GRU
checkpoints there do not load as SPICE models and are skipped): the mechanisms
recurring in >= VOTES of the seeds, the reduced equation matrix, the correlation
between the mechanisms and their structural beta effects on the study's criteria
(CRITERIA; data with derived criterion columns: utils/participant_metrics.study_data).
Every figure is saved with and without annotations.

Output: weinhardt2026/studies/<study>/results/equation_discovery/

Usage:
    python weinhardt2026/analysis/discovery/run_equation_discovery.py                # every study
    python weinhardt2026/analysis/discovery/run_equation_discovery.py dezfouli2019   # some studies
"""

import argparse
import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

from spice.utils.analysis.equation_discovery import equation_discovery, load_dataset, model_loader, seed_matrices
from weinhardt2026.utils.studies import STUDIES
from weinhardt2026.utils.participant_metrics import study_data

VOTES = 6

# bruckner2025 age_group codes (data_raw/participants.tsv): 1 children (7-11 y), 3 younger adults (20-28 y),
# 4 older adults (61-76 y)
AGE_GROUPS = {1: 'children', 3: 'younger_adults', 4: 'older_adults'}

# Criteria of the beta effects per study, as keyword arguments of analysis_mechanism_betas (reward rates: see
# utils/participant_metrics.REWARD_COLUMNS)
CRITERIA = {
    'dezfouli2019': [
        dict(criterion_col='reward'),
        dict(criterion_col='diag', criterion_type='discrete',
             comparisons=[('Depression', 'Control'), ('Bipolar', 'Control'), ('Bipolar', 'Depression')]),
        dict(criterion_col='diag', criterion_type='discrete',
             criterion_map={'Control': 'Control', 'Depression': 'Patients', 'Bipolar': 'Patients'},
             comparisons=[('Patients', 'Control')]),
    ],
    'ganesh2024a': [dict(criterion_col='reward')],
    'eckstein2026': [dict(criterion_col='reward')],
    'bustamante2023': [dict(criterion_col='reward_per_second', block_col='overall_round')],
    'braun2018': [dict(criterion_col='rt_switch_cost')],
    'bruckner2025': [
        dict(criterion_col='trial_reward'),
        dict(criterion_col='age_group', criterion_type='discrete', criterion_map=AGE_GROUPS,
             comparisons=[('younger_adults', 'children'), ('older_adults', 'children'),
                          ('older_adults', 'younger_adults')]),
        dict(criterion_col='age_group', criterion_type='discrete',
             criterion_map={1: 'children', 3: 'adults', 4: 'adults'}, comparisons=[('adults', 'children')]),
    ],
    'weber2024': [dict(criterion_col='laser_caught')],
}


def run_study(study):
    study_dir = os.path.join(ROOT, 'weinhardt2026', 'studies', study)
    spice_module = importlib.import_module(f'weinhardt2026.studies.{study}.spice_{study}')
    data_kwargs = STUDIES[study]['data_kwargs']
    frame, dataset = load_dataset(study_data(study), spice_module, **data_kwargs)
    output_dir = os.path.join(study_dir, 'results', 'equation_discovery')
    os.makedirs(output_dir, exist_ok=True)

    matrices_per_seed = seed_matrices(os.path.join(study_dir, 'params'), model_loader(spice_module, dataset),
                                      os.path.join(output_dir, 'equation_matrices.pkl'))
    equation_discovery(matrices_per_seed, output_dir, votes=VOTES, criteria=CRITERIA[study], data=frame,
                       participant_col=data_kwargs.get('df_participant_id', 'participant'), clean=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # kolff2025 is not in CRITERIA: one group model, no individual differences to discover
    parser.add_argument('studies', nargs='*', help=f'any of {", ".join(CRITERIA)} (default: all)')
    studies = parser.parse_args().studies or list(CRITERIA)
    for study in studies:
        if study not in CRITERIA:
            parser.error(f'unknown study {study}')
    for study in studies:
        print(f'\n=== {study}')
        run_study(study)

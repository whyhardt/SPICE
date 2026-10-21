"""
Tables of the discovered equations per study, from the final SPICE models (weinhardt2026/utils/final_models.py).

  term_presence.csv        per (participant, module, term): presence = share of the final seeds in which the
                           participant carries the term (within a seed: majority of the ensemble members, see
                           utils/final_models.seed_matrices); present = presence >= 0.5
  example_participants.csv per (role, participant, module, term): coefficient of three participants' equations
  example_participants.txt the same equations, written out

The three participants are chosen in the participants' RNN embedding: the two furthest apart and the most
average one (closest to the centroid). Distances are averaged over every seed and ensemble member, each distance
matrix normalized by its mean (the members' embedding spaces have their own orientation and scale, so distances
can be averaged, the embeddings themselves cannot). Their equations are those of the best final seed (highest
training likelihood, utils/final_models.best_checkpoint) and, within it, the best ensemble member.
kolff2025 is one group model: its single equation set is written instead.

Written to weinhardt2026/studies/<study>/results/equations/.

Usage:
    python weinhardt2026/analysis/analysis_equation_tables.py all
    python weinhardt2026/analysis/analysis_equation_tables.py dezfouli2019 bruckner2025
"""

import argparse
import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform

from weinhardt2026.utils.study_models import member_train_loglik
from weinhardt2026.utils.final_models import best_checkpoint, final_checkpoints, model_setup, seed_matrices

GROUP_MODEL_STUDIES = ('kolff2025',)
STUDIES = ('dezfouli2019', 'ganesh2024a', 'eckstein2026', 'bustamante2023', 'braun2018', 'bruckner2025', 'weber2024',
           'kolff2025')


def study_path(study: str, *parts) -> str:
    return os.path.join(ROOT, 'weinhardt2026', 'studies', study, *parts)


def participant_ids(study: str) -> list:
    """Participant labels of the data in SPICE index order (csv_to_dataset numbers them in order of appearance)."""
    if study in GROUP_MODEL_STUDIES:
        return ['group']
    from weinhardt2026.figures.figure_model_evaluation import STUDIES as REGISTRY
    column = REGISTRY[study]['data_kwargs'].get('df_participant_id', 'participant')
    ids = pd.read_csv(study_path(study, 'data', f'{study}.csv'), usecols=[column])[column].unique()
    return [int(i) if isinstance(i, float) and i.is_integer() else i for i in ids]


# ── Selection in embedding space ────────────────────────────────────

def embedding_distances(checkpoints: list[str]) -> np.ndarray:
    """(P, P) distance between participant embeddings, averaged over seeds and members (each normalized by its mean)."""
    distances = []
    for path in checkpoints:
        embedding = torch.load(path, map_location='cpu')['model']['participant_embedding.weight'].numpy()   # (E, P, D)
        for member in embedding:
            distance = squareform(pdist(member))
            distances.append(distance / distance[np.triu_indices_from(distance, 1)].mean())
    return np.mean(distances, axis=0)


def select_furthest_pair(distance: np.ndarray) -> list[int]:
    return list(max(itertools.combinations(range(distance.shape[0]), 2), key=lambda pair: distance[pair]))


def select_most_average(checkpoints: list[str]) -> int:
    """Participant closest to the embedding centroid, averaged over every seed and ensemble member."""
    distances = []
    for path in checkpoints:
        embedding = torch.load(path, map_location='cpu')['model']['participant_embedding.weight'].numpy()   # (E, P, D)
        for member in embedding:
            distance = np.linalg.norm(member - member.mean(axis=0), axis=1)
            distances.append(distance / distance.mean())
    return int(np.argmin(np.mean(distances, axis=0)))


# ── Equations ───────────────────────────────────────────────────────

def equation_coefficients(estimator, participant: int, member: int) -> list[tuple[str, str, float]]:
    """(module, term, coefficient) of the active terms of one participant's equations (experiment 0). The module's
    own state is stored as a deviation from the identity, so its coefficient is reported + 1."""
    model = estimator.model
    coefficients = model.get_sindy_coefficients()
    rows = []
    for module in model.get_modules():
        values = coefficients[module][member, participant, 0]
        values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
        for term, value in zip(model.get_candidate_terms(key_module=module), values):
            value = value + 1 if term == module else value
            if value != 0:
                rows.append((module, term, float(value)))
    return rows


def format_equations(estimator, participant: int, member: int) -> list[str]:
    """One line per module: module[t+1] = coefficient·term + ..."""
    terms_by_module = {module: [] for module in estimator.model.get_modules()}
    for module, term, value in equation_coefficients(estimator, participant, member):
        name = f'{module}[t]' if term == module else term
        terms_by_module[module].append(f'{value:+.3f}' if term == '1' else f'{value:+.3f}·{name}')
    return [f'{module}[t+1] = ' + (' '.join(parts) if parts else '0') for module, parts in terms_by_module.items()]


def best_member(study: str, estimator) -> int:
    _, dataset_train = model_setup(study, 'spice')
    return int(np.argmax(member_train_loglik(study, estimator, dataset_train)))


# ── Tables ──────────────────────────────────────────────────────────

def term_presence(study: str) -> pd.DataFrame:
    matrices = seed_matrices(study)
    seeds = sorted(matrices)
    ids = participant_ids(study)
    frames = []
    for module, data in matrices[seeds[0]].items():
        presence = np.mean([matrices[seed][module]['presence'] for seed in seeds], axis=0)   # (P, T)
        frame = pd.DataFrame(presence, columns=data['terms'])
        frame.insert(0, 'participant', range(len(frame)))
        frame = frame.melt(id_vars='participant', var_name='term', value_name='presence')
        frames.append(frame.assign(module=module))
    table = pd.concat(frames, ignore_index=True)
    table.insert(1, 'participant_id', table['participant'].map(dict(enumerate(ids))))
    table['present'] = table['presence'] >= 0.5
    table['n_seeds'] = len(seeds)
    return table[['participant', 'participant_id', 'module', 'term', 'presence', 'present', 'n_seeds']]


def example_participants(study: str) -> tuple[pd.DataFrame, list[str]]:
    checkpoints = final_checkpoints(study, 'spice')
    if study in GROUP_MODEL_STUDIES:
        selected = [('group model', 0)]
    else:
        pair = select_furthest_pair(embedding_distances(checkpoints))
        selected = [('most distinct 1', pair[0]), ('most average', select_most_average(checkpoints)), ('most distinct 2', pair[1])]

    path = best_checkpoint(study, 'spice')
    load, _ = model_setup(study, 'spice')
    estimator = load(path)
    member = best_member(study, estimator)
    ids = participant_ids(study)

    rows, lines = [], [f'checkpoint: {os.path.basename(path)}', f'ensemble member: {member}', '']
    for role, participant in selected:
        lines.append(f'{role}: participant {participant} (ID {ids[participant]})')
        lines.extend('  ' + line for line in format_equations(estimator, participant, member))
        lines.append('')
        rows += [dict(role=role, participant=participant, participant_id=ids[participant], module=module, term=term,
                      coefficient=value) for module, term, value in equation_coefficients(estimator, participant, member)]
    return pd.DataFrame(rows), lines


def run_equation_tables(study: str):
    output_dir = study_path(study, 'results', 'equations')
    os.makedirs(output_dir, exist_ok=True)
    term_presence(study).to_csv(os.path.join(output_dir, 'term_presence.csv'), index=False)
    table, lines = example_participants(study)
    table.to_csv(os.path.join(output_dir, 'example_participants.csv'), index=False)
    with open(os.path.join(output_dir, 'example_participants.txt'), 'w') as file:
        file.write('\n'.join(lines))
    print(f'\n{study}\n' + '\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('studies', nargs='+', help=f"study names or 'all' ({', '.join(STUDIES)})")
    args = parser.parse_args()
    for study in (STUDIES if args.studies == ['all'] else args.studies):
        run_equation_tables(study)

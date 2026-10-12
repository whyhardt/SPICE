"""Mechanism recovery across seeds for every cell of a hyperparameter grid.

Mechanisms are found in two stacked levels (`stacked_clustering` in
`analysis_coefficient_dendrogram`):

  - **presence**: a significant co-occurrence group of signed terms (presence
    dendrogram, per-module permutation test) -- a module plus the exact set of
    signed terms that switch on together across participants;
  - **ties**: inside each presence group, a significant group of terms whose
    coefficient values move together across the participants carrying them.

Only multi-term groups count. Recovery is reported for both levels.

Two views per grid cell (one cell = all seeds of one hyperparameter setting):

  - **per seed** (no consensus): the dendrogram is run on each seed's model.
    `recovery_pairwise` is the share of one seed's mechanisms that another seed
    finds exactly, averaged over all ordered seed pairs.
  - **majority vote**: a consensus model keeps a signed term for a participant
    when at least `votes` of the cell's seeds carry it (e.g. 2 of 3), with the
    mean coefficient of the agreeing seeds. Its dendrograms give the consensus mechanisms; `recovery_consensus` is the
    share of them that at least `votes` individual seeds also find.

Groups from different seeds that describe the same mechanism often differ by a
term or two. **Core mechanisms** strip those: within a module, groups of
different seeds sharing >= 2 terms form a family, and the family's core is the
largest term set that at least k seeds all contain (k = all seeds, or the
majority vote). Terms of the family outside the core are reported as spurious,
with how many seeds carried them.

Usage:
    python weinhardt2026/analysis/analysis_grid_mechanisms.py
"""

import importlib
import os
import pickle
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'weinhardt2026'))

from weinhardt2026.analysis.analysis_coefficient_dendrogram import (
    get_module_matrices,
    stacked_clustering,
)
from weinhardt2026.analysis.analysis_grid_search import find_grid_checkpoints, swept_tags
from weinhardt2026.utils.checkpoints import load_estimator


# ── Inputs ───────────────────────────────────────────────────────────

def extract_matrices(pkl_pattern, module, cache_path, exclude=(), n_actions=2, polynomial_degree=2):
    """Per checkpoint: tags + each module's (values, presence, terms). Cached, so reruns skip loading."""
    cached = pickle.load(open(cache_path, 'rb')) if os.path.exists(cache_path) else {}
    spice_module = importlib.import_module(module)
    for path, tags in find_grid_checkpoints(pkl_pattern).items():
        name = os.path.basename(path)
        if name in cached or name in exclude:
            continue
        print(f"  loading {name}")
        estimator = load_estimator(path, spice_module.SpiceModel, spice_module.CONFIG,
                                   n_actions=n_actions, polynomial_degree=polynomial_degree)
        cached[name] = {'tags': tags, 'matrices': get_module_matrices(estimator)}
        pickle.dump(cached, open(cache_path, 'wb'))
    return {name: run for name, run in cached.items() if name not in exclude}


def consensus_matrices(runs, votes):
    """Signed-term majority vote across runs: a participant keeps `term (±)` when >= `votes` runs carry it."""
    consensus = {}
    for module, first in runs[0]['matrices'].items():
        stacked = [run['matrices'][module] for run in runs]
        values = np.stack([m['values'] for m in stacked])            # (S, P, T)
        present = np.stack([m['presence'] for m in stacked]) > 0
        carries_positive = present & (values > 0)
        carries_negative = present & (values < 0)
        positive = carries_positive.sum(0) >= votes
        negative = carries_negative.sum(0) >= votes
        # with votes > S/2 a participant can't win both signs; the value is the agreeing seeds' mean
        with np.errstate(invalid='ignore'):
            mean_positive = np.where(carries_positive, values, 0).sum(0) / carries_positive.sum(0)
            mean_negative = np.where(carries_negative, values, 0).sum(0) / carries_negative.sum(0)
        consensus[module] = {
            'values': np.where(positive, mean_positive, np.where(negative, mean_negative, 0.0)),
            'presence': (positive | negative).astype(float),
            'terms': first['terms'],
        }
    return consensus


# ── Mechanisms ───────────────────────────────────────────────────────

LEVELS = ('presence', 'ties')


def find_mechanisms(matrices, n_permutations=10000, grouping_alpha=0.05, min_active_fraction=0.05, seed=0):
    """Per level, the set of (module, frozenset of signed terms) of every significant multi-term group."""
    mechanisms = {level: set() for level in LEVELS}
    for module, data in matrices.items():
        result = stacked_clustering(data, n_permutations=n_permutations, grouping_alpha=grouping_alpha,
                                    min_active_fraction=min_active_fraction, seed=seed)
        for group in result['groups']:
            mechanisms['presence'].add((module, frozenset(group['members'])))
            mechanisms['ties'].update((module, frozenset(tie)) for tie in group['ties'])
    return mechanisms


def recovery_pairwise(mechanism_sets):
    """Mean over ordered seed pairs of the share of seed a's mechanisms that seed b also finds."""
    shares = [len(a & b) / len(a) for a, b in permutations(mechanism_sets, 2) if a]
    return float(np.mean(shares)) if shares else np.nan


def summarize_cell(seed_sets, consensus_set, votes):
    """Recovery numbers for one grid cell from its per-seed and consensus mechanism sets."""
    union = set().union(*seed_sets)
    counts = {m: sum(m in s for s in seed_sets) for m in union}
    consensus_recovered = {m for m in consensus_set if counts.get(m, 0) >= votes}
    return {
        'n_mechanisms_per_seed': float(np.mean([len(s) for s in seed_sets])),
        'n_mechanisms_union': len(union),
        'recovery_pairwise': recovery_pairwise(seed_sets),
        'n_in_all_seeds': sum(c == len(seed_sets) for c in counts.values()),
        'n_consensus': len(consensus_set),
        'n_consensus_recovered': len(consensus_recovered),
        'recovery_consensus': len(consensus_recovered) / len(consensus_set) if consensus_set else np.nan,
    }


def mechanism_families(seed_sets, min_shared=2):
    """Link groups of different seeds within a module that share >= `min_shared` terms.

    Returns (module, {seed index: union of that seed's terms in the family}) per family.
    """
    groups = [(index, module, terms) for index, mechanisms in enumerate(seed_sets) for module, terms in mechanisms]
    parent = list(range(len(groups)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in combinations(range(len(groups)), 2):
        if groups[a][1] == groups[b][1] and len(groups[a][2] & groups[b][2]) >= min_shared:
            parent[root(a)] = root(b)

    families = {}
    for i, (index, module, terms) in enumerate(groups):
        module_of, per_seed = families.setdefault(root(i), (module, {}))
        per_seed[index] = per_seed.get(index, frozenset()) | terms
    return list(families.values())


def family_core(per_seed, votes):
    """Largest term set contained in at least `votes` seeds' groups, and how many seeds contain it."""
    best = frozenset()
    for subset in combinations(per_seed, votes):
        shared = frozenset.intersection(*(per_seed[s] for s in subset))
        if len(shared) > len(best):
            best = shared
    return best, sum(best <= terms for terms in per_seed.values())


def core_mechanisms(seed_sets, votes):
    """Per family: core (>= 2 terms, else empty), its seed support, and spurious terms with seed counts."""
    cores = []
    for module, per_seed in mechanism_families(seed_sets):
        core, support = family_core(per_seed, votes) if len(per_seed) >= votes else (frozenset(), 0)
        if len(core) < 2:
            core, support = frozenset(), 0
        union = frozenset().union(*per_seed.values())
        cores.append({
            'module': module, 'core': core, 'support': support, 'n_seeds_in_family': len(per_seed),
            'spurious': {t: sum(t in terms for terms in per_seed.values()) for t in union - core},
            # share of each supporting seed's terms that lie outside the core
            'spurious_share': (np.mean([len(per_seed[s] - core) / len(per_seed[s])
                                        for s in per_seed if core and core <= per_seed[s]])
                               if core else np.nan),
        })
    return cores


def summarize_cores(seed_sets, votes):
    cores = [c for c in core_mechanisms(seed_sets, votes) if c['core']]
    return {
        'n_cores': len(cores),
        'core_size': float(np.mean([len(c['core']) for c in cores])) if cores else np.nan,
        'spurious_share': float(np.nanmean([c['spurious_share'] for c in cores])) if cores else np.nan,
    }


def format_mechanism(mechanism):
    module, terms = mechanism
    return f"{module}: {{{', '.join(sorted(terms))}}}"


def cluster_grid(runs, votes=2, n_permutations=10000, n_jobs=-1, cache_path=None):
    """Mechanism sets of every seed model and every cell's consensus model (cached: clustering is the slow part)."""
    names = sorted(runs)
    if cache_path and os.path.exists(cache_path):
        cached = pickle.load(open(cache_path, 'rb'))
        if cached['names'] == names and cached['votes'] == votes:
            return cached['seed'], cached['consensus']

    print(f"Clustering {len(names)} seed models ...")
    seed_mechanisms = dict(zip(names, Parallel(n_jobs=n_jobs)(
        delayed(find_mechanisms)(runs[n]['matrices'], n_permutations) for n in names)))

    eligible = [(key, members) for key, members in grid_cells(runs) if len(members) >= votes]
    print(f"Clustering {len(eligible)} consensus models ({votes}-vote) ...")
    consensus_sets = Parallel(n_jobs=n_jobs)(
        delayed(find_mechanisms)(consensus_matrices([runs[n] for n in members], votes), n_permutations)
        for _, members in eligible)
    consensus_of = {key: s for (key, _), s in zip(eligible, consensus_sets)}

    if cache_path:
        pickle.dump({'names': names, 'votes': votes, 'seed': seed_mechanisms, 'consensus': consensus_of},
                    open(cache_path, 'wb'))
    return seed_mechanisms, consensus_of


def grid_cells(runs):
    """[(cell key, member run names)] grouped by the swept hyperparameters."""
    frame = pd.DataFrame([{**runs[n]['tags'], 'name': n} for n in sorted(runs)])
    return [(key, group['name'].tolist()) for key, group in frame.groupby(swept_tags(frame))]


def analysis_grid_mechanisms(runs, votes=2, n_permutations=10000, n_jobs=-1, cache_path=None):
    """Per-cell recovery + core table (columns prefixed by level), the mechanisms and the cores behind it."""
    seed_mechanisms, consensus_of = cluster_grid(runs, votes, n_permutations, n_jobs, cache_path)
    tags = swept_tags(pd.DataFrame([runs[n]['tags'] for n in runs]))
    cells = grid_cells(runs)

    rows, details, core_rows = [], [], []
    for key, members in cells:
        row = {**dict(zip(tags, key)), 'n_seeds': len(members)}
        for level in LEVELS:
            seed_sets = [seed_mechanisms[n][level] for n in members]
            consensus_set = consensus_of.get(key, {}).get(level, set())
            row.update({f'{level}_{k}': v for k, v in summarize_cell(seed_sets, consensus_set, votes).items()})
            for label, k in (('all', len(members)), ('majority', votes)):
                if len(members) < k:
                    continue
                row.update({f'{level}_{key_}_{label}': v for key_, v in summarize_cores(seed_sets, k).items()})
                for core in core_mechanisms(seed_sets, k):
                    core_rows.append({
                        **dict(zip(tags, key)), 'level': level, 'votes': f'{k}/{len(members)}',
                        'module': core['module'],
                        'core': ', '.join(sorted(core['core'])) if core['core'] else '(no core)',
                        'support': core['support'], 'n_seeds_in_family': core['n_seeds_in_family'],
                        'spurious': ', '.join(f'{t} [{n}]' for t, n in sorted(core['spurious'].items())),
                    })
            for mechanism in set().union(*seed_sets) | consensus_set:
                details.append({**dict(zip(tags, key)), 'level': level, 'mechanism': format_mechanism(mechanism),
                                'n_seeds_found': sum(mechanism in s for s in seed_sets),
                                'in_consensus': mechanism in consensus_set})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(details), pd.DataFrame(core_rows)


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == '__main__':

    study = 'dezfouli2019'
    module = 'studies.dezfouli2019.spice_dezfouli2019'
    votes = 2

    study_dir = f'weinhardt2026/studies/{study}'
    results_dir = os.path.join(study_dir, 'results', 'grid_search')
    os.makedirs(results_dir, exist_ok=True)

    # diverged refits carry NaN coefficients and no mechanisms; they would count as failed recoveries
    evaluation = pd.read_csv(os.path.join(results_dir, 'grid_search_runs.csv'))
    failed = set(evaluation.loc[evaluation['sindy_failed'], 'path'])

    runs = extract_matrices(
        pkl_pattern=os.path.join(study_dir, 'params', 'params_grid_search', f'spice_{study}_*.pkl'),
        module=module,
        cache_path=os.path.join(results_dir, 'grid_search_matrices.pkl'),
        exclude=failed,
    )
    table, details, cores = analysis_grid_mechanisms(
        runs, votes=votes, n_jobs=8, cache_path=os.path.join(results_dir, 'grid_search_mechanism_sets.pkl'))

    # scores from the likelihood evaluation, for reading recovery against fit
    scores = evaluation[~evaluation['sindy_failed']].groupby(swept_tags(evaluation))[['BIC', 'AIC']].mean()
    table = table.merge(scores.reset_index(), on=swept_tags(evaluation), how='left')

    pd.set_option('display.width', 250)
    hps = swept_tags(evaluation)
    for level in LEVELS:
        columns = hps + ['n_seeds'] + [f'{level}_{c}' for c in (
            'n_mechanisms_per_seed', 'recovery_pairwise', 'n_in_all_seeds',
            'n_consensus', 'n_consensus_recovered', 'recovery_consensus')] + ['BIC', 'AIC']
        print(f'\n[{level}] without consensus — ranked by pairwise recovery across seeds:')
        print(table.sort_values([f'{level}_recovery_pairwise', f'{level}_n_in_all_seeds'], ascending=False)[columns]
              .head(12).to_string(index=False, float_format='{:.3f}'.format))
        print(f'\n[{level}] {votes}-vote consensus — ranked by consensus mechanisms recovered in >= {votes} seeds:')
        print(table[table['n_seeds'] >= 3].sort_values([f'{level}_recovery_consensus', f'{level}_n_consensus_recovered'],
                                                       ascending=False)[columns]
              .head(12).to_string(index=False, float_format='{:.3f}'.format))

    table.to_csv(os.path.join(results_dir, 'grid_search_mechanism_recovery.csv'), index=False)
    details.to_csv(os.path.join(results_dir, 'grid_search_mechanisms.csv'), index=False)
    cores.to_csv(os.path.join(results_dir, 'grid_search_core_mechanisms.csv'), index=False)

    for level in LEVELS:
        columns = hps + ['n_seeds'] + [f'{level}_{c}_{v}' for v in ('all', 'majority')
                                       for c in ('n_cores', 'core_size', 'spurious_share')] + ['BIC', 'AIC']
        print(f'\n[{level}] core mechanisms — ranked by cores shared by a {votes}-seed majority:')
        print(table[table['n_seeds'] >= 3].sort_values([f'{level}_n_cores_majority', f'{level}_n_cores_all'],
                                                       ascending=False)[columns]
              .head(12).to_string(index=False, float_format='{:.3f}'.format))
    print(f'\nSaved recovery tables to {results_dir}')

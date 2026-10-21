"""
Recurring term groups across seeds with their dendrogram distances (any study of figure_model_evaluation.STUDIES).

Per seed (final SPICE models, see weinhardt2026/utils/final_models.py): the
presence dendrogram of ``analysis_coefficient_dendrogram`` — terms split by sign,
distance 1 - phi, per-module permutation test (FWER) — gives that seed's
significant groups. ``analysis_grid_mechanisms.core_mechanisms`` then keeps, per
family of overlapping groups, the largest term set that at least VOTES seeds
share (agreement 0.5); terms only some seeds add are dropped as spurious.

Figure, one panel per module:
  - consensus tree: average linkage on the 1 - phi distance averaged over seeds
    (per pair, over the seeds in which both signed terms pass the activity filter);
    leaves are the signed terms kept in >= VOTES seeds
  - recurring groups coloured; the marker + bar is the group's merge distance in
    the individual seeds (mean ± SD over the seeds carrying the full group as a
    significant group)
  - grey band: per-seed critical distance at p = .05 (mean ± SD); merges left of
    it are significant in a typical seed

Then the structural beta effects of these mechanisms plus every other signed term
(analysis_coefficient_betas.analysis_mechanism_betas), on reward rate and on
diagnosis; presence = carried in >= VOTES seeds.

Written to weinhardt2026/studies/<study>/results/:
    mechanisms/     figure_<study>_mechanisms.pdf/.png, core_mechanisms_distances_votes<VOTES>.csv,
                    caches (matrices.pkl: utils/final_models.seed_matrices; seed_clusters.pkl)
    beta_effects/   betas_structural_<study>_mechanisms_votes<VOTES>_<criterion>.csv/.png

Usage:
    python weinhardt2026/figures/figure_mechanisms.py --study dezfouli2019
    python weinhardt2026/figures/figure_mechanisms.py --study dezfouli2019 --merged   # + merged-group contrasts
"""

import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import squareform

from weinhardt2026.figures import figure_model_evaluation as evaluation
from weinhardt2026.analysis.analysis_coefficient_dendrogram import (
    cluster_module,
    split_by_sign,
)
from weinhardt2026.analysis.analysis_grid_mechanisms import core_mechanisms
from weinhardt2026.analysis.analysis_coefficient_betas import analysis_mechanism_betas
from weinhardt2026.utils.final_models import seed_matrices
from weinhardt2026.utils.participant_metrics import study_data

VOTES = 5
N_PERMUTATIONS = 10000
GROUPING_ALPHA = 0.05

GROUP_COLORS = ['#c44e52', '#2f6f9f', '#d99a1e', '#5a9e6f', '#7b5ea7', '#8c6d31']
NEUTRAL = '#9a9a9a'


# ── Per-seed clustering ─────────────────────────────────────────────

def cluster_seed(matrices):
    """Per module: presence-mode clustering of the sign-split terms (with permutation test)."""
    results = {}
    for module, data in matrices.items():
        split = split_by_sign(data)
        result = cluster_module(values=split['values'], presence=split['presence'], terms=split['terms'],
                                mode='presence', n_permutations=N_PERMUTATIONS, grouping_alpha=GROUPING_ALPHA)
        groups = [frozenset(t for t, c in zip(result['terms'], result['clusters']) if c == cluster_id)
                  for cluster_id in np.unique(result['clusters'])] if result['clusters'] is not None else []
        results[module] = {
            'terms': result['terms'],
            'distance': result.get('distance'),
            'cophenetic': squareform(cophenet(result['linkage'])) if result['linkage'] is not None else None,
            'critical_distance': result['significance']['critical_distance'] if result['significance'] else np.nan,
            'groups': [g for g in groups if len(g) > 1],
        }
    return results


def cluster_seeds(matrices, output_dir):
    cache = os.path.join(output_dir, 'seed_clusters.pkl')
    names = sorted(matrices)
    if os.path.exists(cache):
        cached = pickle.load(open(cache, 'rb'))
        if sorted(cached) == names:
            return cached
    clusters = dict(zip(names, Parallel(n_jobs=8)(delayed(cluster_seed)(matrices[n]) for n in names)))
    pickle.dump(clusters, open(cache, 'wb'))
    return clusters


# ── Consensus ───────────────────────────────────────────────────────

def consensus_tree(module, seed_results):
    """Average-linkage tree on the seed-averaged 1 - phi distance of terms kept in >= VOTES seeds."""
    counts = pd.Series([t for r in seed_results for t in r[module]['terms']]).value_counts()
    terms = sorted(counts[counts >= VOTES].index)
    if len(terms) < 2:   # nothing to cluster
        return terms, None
    index = {t: i for i, t in enumerate(terms)}
    total, n = np.zeros((len(terms),) * 2), np.zeros((len(terms),) * 2)
    for result in seed_results:
        kept = [t for t in result[module]['terms'] if t in index]
        if result[module]['distance'] is None or not kept:
            continue
        position = {t: i for i, t in enumerate(result[module]['terms'])}
        rows = [index[t] for t in kept]
        block = result[module]['distance'][np.ix_([position[t] for t in kept], [position[t] for t in kept])]
        total[np.ix_(rows, rows)] += block
        n[np.ix_(rows, rows)] += 1
    distance = np.where(n > 0, total / np.maximum(n, 1), 1.0)  # never co-kept: phi = 0
    np.fill_diagonal(distance, 0.0)
    return terms, linkage(squareform(distance, checks=False), method='average')


def group_merge_distances(module, group, seed_results):
    """Per seed carrying `group` as a significant group: the height at which all its terms have joined."""
    heights = []
    for result in seed_results:
        r = result[module]
        if any(group <= found for found in r['groups']):
            position = {t: i for i, t in enumerate(r['terms'])}
            members = [position[t] for t in group]
            heights.append(r['cophenetic'][np.ix_(members, members)].max())
    return np.array(heights)


# ── Plot ────────────────────────────────────────────────────────────

def node_positions(linkage_matrix, n_leaves, leaf_order):
    """y position (scipy's left-orientation leaf coordinates) and members of every tree node."""
    y = {leaf: 5.0 + 10.0 * rank for rank, leaf in enumerate(leaf_order)}
    members = {leaf: {leaf} for leaf in range(n_leaves)}
    for row, (left, right, _, _) in enumerate(linkage_matrix):
        node = n_leaves + row
        y[node] = (y[int(left)] + y[int(right)]) / 2
        members[node] = members[int(left)] | members[int(right)]
    return y, members


def plot_mechanisms(modules, seed_results, cores, output_path):
    trees = {m: consensus_tree(m, seed_results) for m in modules}
    modules = [m for m in modules if trees[m][1] is not None]   # modules with < 2 recurring terms have no tree
    height = [max(1.6, 0.36 * len(trees[m][0]) + 0.6) for m in modules]
    fig, axes = plt.subplots(len(modules), 1, figsize=(7.5, sum(height) + 0.8), squeeze=False,
                             gridspec_kw={'height_ratios': height})
    rows_out = []

    for ax, module in zip(axes.ravel(), modules):
        terms, linkage_matrix = trees[module]
        n_leaves = len(terms)
        module_cores = [c for c in cores if c['module'] == module and c['core']]
        module_cores.sort(key=lambda c: -c['support'])
        group_of = {}
        for k, core in enumerate(module_cores):
            for term in core['core']:
                group_of[term] = k

        order = dendrogram(linkage_matrix, no_plot=True)['leaves']
        y, members = node_positions(linkage_matrix, n_leaves, order)

        def color(link):
            groups = {group_of.get(terms[i]) for i in members[link]}
            return GROUP_COLORS[groups.pop()] if len(groups) == 1 and None not in groups else NEUTRAL

        dendrogram(linkage_matrix, labels=terms, orientation='left', link_color_func=color, ax=ax)

        critical = np.array([r[module]['critical_distance'] for r in seed_results])
        critical = critical[np.isfinite(critical)]
        if critical.size:
            ax.axvspan(critical.mean() - critical.std(), critical.mean() + critical.std(),
                       color='#dddddd', alpha=0.6, zorder=0, linewidth=0)
            ax.axvline(critical.mean(), color='#777777', linestyle='--', linewidth=1, zorder=1)

        for k, core in enumerate(module_cores):
            heights = group_merge_distances(module, core['core'], seed_results)
            leaves = {terms.index(t) for t in core['core']}
            node = min((n for n, m in members.items() if leaves <= m), key=lambda n: len(members[n]))
            if heights.size:
                ax.errorbar(heights.mean(), y[node], xerr=heights.std(), fmt='o', color=GROUP_COLORS[k],
                            markersize=6, markeredgecolor='white', capsize=3, linewidth=1.4, zorder=5)
                ax.annotate(f'{core["support"]}/{len(seed_results)}', (heights.mean(), y[node]), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=7.5, color=GROUP_COLORS[k],
                            fontweight='bold')
            rows_out.append({'module': module, 'group': ', '.join(sorted(core['core'])), 'support': core['support'],
                             'merge_distance_mean': heights.mean() if heights.size else np.nan,
                             'merge_distance_sd': heights.std() if heights.size else np.nan,
                             'consensus_merge_distance': linkage_matrix[node - n_leaves, 2] if node >= n_leaves else 0.0,
                             'critical_distance_mean': critical.mean() if critical.size else np.nan})

        for label in ax.get_yticklabels():
            k = group_of.get(label.get_text())
            if k is not None:
                label.set_color(GROUP_COLORS[k])
                label.set_fontweight('bold')
        left, right = ax.get_xlim()
        ax.set_xlim(max(left, 1.05), 0)
        ax.set_title(module, fontsize=10, fontweight='bold')
        if ax is axes.ravel()[len(modules) - 1]:
            ax.set_xlabel('distance (1 − φ)', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # one distance axis for the whole column, so merge heights compare across modules
    left = max(ax.get_xlim()[0] for ax in axes.ravel())
    for ax in axes.ravel():
        ax.set_xlim(left, 0)

    handles = [plt.Line2D([], [], color=GROUP_COLORS[0], linewidth=2.5, label=f'recurring group (≥ {VOTES}/{len(seed_results)} seeds; one colour per group)'),
               plt.Line2D([], [], color=NEUTRAL, linewidth=2.5, label='other merges'),
               plt.Line2D([], [], marker='o', color='#444444', linestyle='-', label='group merge distance per seed (mean ± SD)'),
               plt.Rectangle((0, 0), 1, 1, color='#dddddd', label='critical distance p = .05 (mean ± SD over seeds)')]
    fig.legend(handles=handles, loc='upper center', ncol=1, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 1 - 1.0 / fig.get_figheight()])
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {output_path}.pdf/.png')
    return pd.DataFrame(rows_out)


# ── Main ────────────────────────────────────────────────────────────

# Structural beta effects per study: (criterion kwargs, ...) for analysis_mechanism_betas, on the study's data with
# its per-trial reward column (utils/participant_metrics.study_data). Continuous criteria are per-trial columns
# averaged per participant (reward rate; bustamante2023: reward per second, averaged over the fixed-length rounds;
# braun2018: RT switch cost; definitions in utils/participant_metrics.py).
# bruckner2025 age_group codes (data_raw/participants.tsv): 1 children (7-11 y), 3 younger adults (20-28 y), 4 older adults (61-76 y)
BRUCKNER_AGE_GROUPS = {1: 'children', 3: 'younger_adults', 4: 'older_adults'}
BETA_CRITERIA = {
    'dezfouli2019': (
        dict(criterion_col='reward'),
        dict(criterion_col='diag', criterion_type='discrete',
             comparisons=[('Depression', 'Control'), ('Bipolar', 'Control'), ('Bipolar', 'Depression')]),
    ),
    'ganesh2024a': (dict(criterion_col='reward', participant_col='subjID'),),
    'eckstein2026': (dict(criterion_col='reward'),),
    'bustamante2023': (dict(criterion_col='reward_per_second', participant_col='subject_id', block_col='overall_round'),),
    'braun2018': (dict(criterion_col='rt_switch_cost', participant_col='subject'),),
    'bruckner2025': (
        dict(criterion_col='trial_reward'),
        dict(criterion_col='age_group', criterion_type='discrete', criterion_map=BRUCKNER_AGE_GROUPS,
             comparisons=[('younger_adults', 'children'), ('older_adults', 'children'),
                          ('older_adults', 'younger_adults')]),
    ),
    'weber2024': (dict(criterion_col='laser_caught'),),
}
# Opt-in (--merged): contrasts of merged groups
MERGED_CRITERIA = {
    'dezfouli2019': (
        dict(criterion_col='diag', criterion_type='discrete',
             criterion_map={'Control': 'Control', 'Depression': 'Patients', 'Bipolar': 'Patients'},
             comparisons=[('Patients', 'Control')]),
    ),
    'bruckner2025': (
        dict(criterion_col='age_group', criterion_type='discrete',
             criterion_map={1: 'children', 3: 'adults', 4: 'adults'}, comparisons=[('adults', 'children')]),
    ),
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study', default='dezfouli2019', choices=list(evaluation.STUDIES))
    parser.add_argument('--merged', action='store_true', help='also run the merged-group contrasts (MERGED_CRITERIA)')
    args = parser.parse_args()
    study = args.study
    output_dir = evaluation._study_path(study, 'results', 'mechanisms')
    betas_dir = evaluation._study_path(study, 'results', 'beta_effects')
    os.makedirs(output_dir, exist_ok=True)

    matrices = seed_matrices(study)
    clusters = cluster_seeds(matrices, output_dir)
    names = sorted(clusters)
    seed_results = [clusters[n] for n in names]
    modules = list(seed_results[0])

    seed_sets = [{(module, group) for module in modules for group in result[module]['groups']} for result in seed_results]
    cores = core_mechanisms(seed_sets, VOTES)

    table = plot_mechanisms(modules, seed_results, cores, os.path.join(output_dir, f'figure_{study}_mechanisms'))
    table.to_csv(os.path.join(output_dir, f'core_mechanisms_distances_votes{VOTES}.csv'), index=False)
    pd.set_option('display.width', 250)
    print(table.to_string(index=False, float_format='{:.3f}'.format))

    # structural beta effects of the recurring mechanisms + all other signed terms
    mechanisms = [(c['module'], c['core']) for c in cores if c['core']]
    matrices_per_seed = [matrices[n] for n in names]
    data = study_data(study)

    def run_betas(kwargs):
        return analysis_mechanism_betas(matrices_per_seed, mechanisms, data, betas_dir, votes=VOTES,
                                        prefix=f'{study}_mechanisms_votes{VOTES}', plot_significant_only=True, **kwargs)

    for kwargs in BETA_CRITERIA.get(study, ()) + (MERGED_CRITERIA.get(study, ()) if args.merged else ()):
        run_betas(kwargs)

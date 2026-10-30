"""Equation discovery across the seeds of a SPICE fit: the mechanisms that recur, how they co-occur, and how
they relate to behaviour.

Input: a directory of fits (seeds) of one SPICE model to one dataset. Every file in it
is loaded with ``SpiceEstimator.load_spice`` for the architecture of ``--module``; a
file that does not load (another model, another architecture) is skipped and logged.
Each fit gives every participant their own equation; a mechanism is a set of signed
terms that switch on together across participants, reproducibly across seeds.

  1. Per seed, the significant groups: presence dendrogram of the sign-split terms
     (``coefficient_dendrogram.cluster_module``: distance 1 - phi, per-module
     permutation test with family-wise error control; only terms carried by at
     least ``min_carriers`` participants are clustered).
  2. Mechanisms (``core_mechanisms``): per module, an agglomerative tree merges the
     terms by joint support (the number of seeds with one significant group that
     contains all of them). Its clusters cut at ``votes`` seeds are the maximal term
     sets grouped all together in >= ``votes`` seeds.
  3. The reduced equation matrix of the mechanisms and the correlation between them
     (phi of their presence across participants); the joint-support dendrogram.
  4. Optionally (``--criterion``), the structural beta effects of the mechanisms on a
     per-participant criterion (``coefficient_betas.analysis_mechanism_betas``).

A participant carries a term when at least ``votes`` seeds give it to them. Every
figure is saved with annotations (``<name>.pdf/.png``) and, with ``--clean``, also
without them (``<name>_clean.pdf/.png``).

Written to --output-dir:
    figure_equation_matrix                     reduced equation matrix (M1, M2, ...)
    figure_mechanism_correlations, .csv        phi between the mechanisms; per pair: carriers, co-occurrence, phi, q
    figure_mechanism_dendrogram, mechanisms.csv  joint-support tree; per mechanism: joint support, merge distances
    figure_mechanism_betas_<criterion>[_<group>_vs_<reference>], betas_structural_mechanisms_<criterion>[...].csv
    caches: equation_matrices.pkl (coefficients per seed), seed_clusters.pkl (groups per seed)

Usage:
    python -m spice.utils.analysis.equation_discovery \
        --models-dir fits/ --module my_study.spice_model --data my_study.csv \
        --criterion reward --output-dir results/equation_discovery

``--module`` is a dotted module path exposing ``SpiceModel`` and ``CONFIG`` (and
optionally ``ESTIMATOR_KWARGS``, ``prepare_dataframe``, ``prepare_dataset``); the data
columns are passed on to ``csv_to_dataset`` (``--df-choice``, ``--additional-inputs``, ...).
"""

import argparse
import importlib
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster
from scipy.spatial.distance import squareform

from spice.utils import csv_to_dataset
from spice.utils.analysis.coefficient_betas import analysis_mechanism_betas, consensus_presence
from spice.utils.analysis.coefficient_dendrogram import cluster_module, get_module_matrices, split_by_sign
from spice.utils.analysis.loading import load_estimator

GROUP_COLORS = ['#c44e52', '#2f6f9f', '#d99a1e', '#5a9e6f', '#7b5ea7', '#8c6d31']
NEUTRAL = '#9a9a9a'
# phi, shared by the equation matrix (within a mechanism) and the mechanism correlations (between mechanisms)
PHI_CMAP, PHI_NORM = plt.cm.viridis, Normalize(-1, 1)


def text_color(color):
    """Dark text on light cells, white text on dark ones."""
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return '#1f1f1f' if luminance > 0.55 else 'white'


# ── Mechanisms recurring across seeds ───────────────────────────────

def joint_support(groups, terms):
    """Number of seeds with one group (of `groups`, per seed a list of term sets) that contains all of `terms`."""
    return sum(any(terms <= g for g in seed) for seed in groups)


def joint_tree(seed_sets, module, min_support=1):
    """Agglomerative tree of `module`'s grouped terms on joint support.

    Repeatedly merges the two clusters whose union lies inside one single group in the most seeds, at height
    1 - joint support / R. Adding terms can only lower joint support, so heights never decrease up the tree, and a
    cut at 1 - k/R yields exactly clusters whose terms are grouped all together in >= k seeds. Leaves: terms that
    some other term is grouped with in >= `min_support` seeds (the others can only merge above that level).
    Returns (terms, linkage) in scipy format; (terms, None) for < 2 terms.
    """
    groups = [[g for m, g in mechanisms if m == module] for mechanisms in seed_sets]
    n_seeds = len(seed_sets)
    grouped = sorted({t for seed in groups for g in seed for t in g})
    terms = [t for t in grouped
             if any(joint_support(groups, frozenset((t, u))) >= min_support for u in grouped if u != t)]
    if len(terms) < 2:
        return terms, None
    clusters = {i: frozenset([t]) for i, t in enumerate(terms)}
    rows = []
    while len(clusters) > 1:
        # ties in support: the smaller merged cluster first (conservative), then term order
        support, _, a, b = max((joint_support(groups, clusters[a] | clusters[b]),
                                -len(clusters[a] | clusters[b]), -a, -b)
                               for a in clusters for b in clusters if a < b)
        a, b = -a, -b
        merged = clusters.pop(a) | clusters.pop(b)
        rows.append([a, b, 1 - support / n_seeds, len(merged)])
        clusters[len(terms) + len(rows) - 1] = merged
    return terms, np.array(rows, dtype=float)


def carrying_groups(seed_sets, module, terms):
    """Per seed: its group of `module` that contains all of `terms` (a seed's groups are disjoint: at most one)."""
    return {s: g for s, mechanisms in enumerate(seed_sets) for m, g in mechanisms if m == module and terms <= g}


def core_mechanisms(seed_sets, votes):
    """Mechanisms recurring across seeds: clusters of the joint-support tree (``joint_tree``) cut at `votes` seeds,
    i.e. maximal term sets (>= 2 terms) that lie inside one single group in >= `votes` seeds.
    Spurious terms: extra terms in the groups that carry a mechanism.
    """
    n_seeds = len(seed_sets)
    cores = []
    for module in sorted({m for mechanisms in seed_sets for m, _ in mechanisms}):
        terms, tree = joint_tree(seed_sets, module, min_support=votes)
        if tree is None:
            continue
        labels = fcluster(tree, t=1 - votes / n_seeds + 1e-9, criterion='distance')
        for label in np.unique(labels):
            core = frozenset(t for t, x in zip(terms, labels) if x == label)
            if len(core) < 2:
                continue
            carriers = carrying_groups(seed_sets, module, core)
            cores.append({
                'module': module, 'core': core, 'support': len(carriers),
                'spurious': {t: sum(t in g for g in carriers.values())
                             for t in frozenset().union(*carriers.values()) - core},
                # share of each carrying group's terms that lie outside the core
                'spurious_share': np.mean([len(g - core) / len(g) for g in carriers.values()]),
            })
    return cores


# ── Per-seed groups ─────────────────────────────────────────────────

def cluster_seed(matrices, n_permutations=10000, grouping_alpha=0.05, min_carriers=5):
    """One seed's significant groups, per module: presence-mode clustering of the sign-split terms
    (``cluster_module``, per-module permutation test)."""
    results = {}
    for module, data in matrices.items():
        split = split_by_sign(data)
        result = cluster_module(values=split['values'], presence=split['presence'], terms=split['terms'],
                                mode='presence', n_permutations=n_permutations, grouping_alpha=grouping_alpha,
                                min_carriers=min_carriers)
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


def cluster_seeds(matrices_per_seed, cache_path=None, n_jobs=8, **settings):
    """`cluster_seed` for every seed, in parallel. With `cache_path`, the result is cached there together with its
    `settings` and reused only when the seeds and settings are unchanged."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as file:
            cached = pickle.load(file)
        if isinstance(cached, dict) and cached.get('settings') == settings \
                and cached.get('n_seeds') == len(matrices_per_seed):
            return cached['clusters']
    clusters = Parallel(n_jobs=n_jobs)(delayed(cluster_seed)(matrices, **settings) for matrices in matrices_per_seed)
    if cache_path:
        with open(cache_path, 'wb') as file:
            pickle.dump({'settings': settings, 'n_seeds': len(matrices_per_seed), 'clusters': clusters}, file)
    return clusters

# ── Loading the seeds ───────────────────────────────────────────────

def seed_matrices(models_dir, load, cache_path=None):
    """Per seed (file of `models_dir` that `load` can load, in name order): each module's coefficient values,
    presence and term names (``get_module_matrices``). Files that do not load are skipped and logged. With
    `cache_path`, cached there and reused while the directory holds the same files."""
    names = sorted(name for name in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, name)))
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as file:
            cached = pickle.load(file)
        if sorted(list(cached['matrices']) + cached['skipped']) == names:
            return [cached['matrices'][name] for name in sorted(cached['matrices'])]

    matrices, skipped = {}, []
    for name in names:
        try:
            estimator = load(os.path.join(models_dir, name))
        except Exception as error:
            print(f'Skipped {name}: {type(error).__name__}: {error}')
            skipped.append(name)
            continue
        matrices[name] = get_module_matrices(estimator)
    if not matrices:
        raise ValueError(f'No file in {models_dir} loads as this SPICE model.')
    print(f'Loaded {len(matrices)} seeds from {models_dir} ({len(skipped)} files skipped)')
    if cache_path:
        with open(cache_path, 'wb') as file:
            pickle.dump({'matrices': matrices, 'skipped': skipped}, file)
    return [matrices[name] for name in sorted(matrices)]


def load_dataset(data, spice_module, **csv_kwargs):
    """The data (CSV path or DataFrame) as the model saw it: the module's optional ``prepare_dataframe``, then
    ``csv_to_dataset(**csv_kwargs)``, then the module's optional ``prepare_dataset``."""
    frame = pd.read_csv(data, low_memory=False) if isinstance(data, str) else data
    if hasattr(spice_module, 'prepare_dataframe'):
        frame = spice_module.prepare_dataframe(frame)
    dataset = csv_to_dataset(file=frame, **csv_kwargs)
    if hasattr(spice_module, 'prepare_dataset'):
        dataset = spice_module.prepare_dataset(dataset)
    return frame, dataset


def model_loader(spice_module, dataset, polynomial_degree=2):
    """path -> SpiceEstimator of `spice_module` (SpiceModel, CONFIG, optional ESTIMATOR_KWARGS), sized by `dataset`."""
    estimator_kwargs = dict(
        n_actions=dataset.n_actions,
        n_items=dataset.n_actions,
        n_reward_features=dataset.n_reward_features,
        n_experiments=len(dataset.xs[..., -2].unique()),
    )
    estimator_kwargs.update(getattr(spice_module, 'ESTIMATOR_KWARGS', {}))
    return lambda path: load_estimator(path, spice_module.SpiceModel, spice_module.CONFIG,
                                       polynomial_degree=polynomial_degree, **estimator_kwargs)


# ── Figures ─────────────────────────────────────────────────────────

def mechanism_coefficients(module, group, seed_results, matrices_per_seed):
    """Per signed term of `group`: its coefficients in the participants carrying all of `group`, pooled over the
    seeds in which `group` lies inside a significant group (one value per participant and seed)."""
    values = {term: [] for term in group}
    for result, matrices in zip(seed_results, matrices_per_seed):
        if not any(group <= found for found in result[module]['groups']):
            continue
        split = split_by_sign(matrices[module])
        columns = {term: split['terms'].index(term) for term in group}
        carriers = (split['presence'][:, list(columns.values())] > 0).all(axis=1)
        for term, column in columns.items():
            values[term].extend(split['values'][carriers, column])
    return {term: np.asarray(v) for term, v in values.items()}


def join_distances(module, group, seed_results):
    """Per signed term of `group`: the distance (1 - phi) at which it first joins another member of `group`,
    mean over the seeds in which `group` lies inside a significant group."""
    per_term = {term: [] for term in group}
    for result in seed_results:
        r = result[module]
        if not any(group <= found for found in r['groups']):
            continue
        position = {t: i for i, t in enumerate(r['terms'])}
        for term in group:
            others = [position[t] for t in group if t != term]
            per_term[term].append(r['cophenetic'][position[term], others].min())
    return {term: np.mean(d) for term, d in per_term.items()}

def format_coefficient(value):
    """Two decimals, with a typographic minus."""
    return f'{value:.2f}'.replace('-', '−')


def sort_cores(cores, module_order):
    """Mechanisms in figure order (M1, M2, ...): by module in `module_order` (the model's), then by joint support."""
    return sorted(cores, key=lambda c: (module_order.index(c['module']), -c['support']))


def mechanism_codes(cores):
    """'M1', 'M2', ... for mechanisms already in figure order."""
    return [f'M{i + 1}' for i in range(len(cores))]


def save_figure(fig, output_path):
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {output_path}.pdf/.png')


def add_horizontal_colorbar(fig, ax, x_start, x_end, cmap, norm, label, gap=0.07):
    """Horizontal colourbar `gap` (figure fraction) below `ax`, spanning its data x-range [x_start, x_end]; `label`
    None leaves it unlabelled."""
    ax.apply_aspect()
    to_figure = ax.transData + fig.transFigure.inverted()
    bottom_y = max(ax.get_ylim())
    (left, bottom), (right, _) = to_figure.transform([(x_start, bottom_y), (x_end, bottom_y)])
    cax = fig.add_axes([left, bottom - gap, right - left, 0.018])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    bar.ax.tick_params(labelsize=7)
    if label:
        bar.set_label(label, fontsize=8)


# ── Equation matrix ─────────────────────────────────────────────────
#
# One row per mechanism, one column per candidate term. Modules that differ only in the option they update
# (chosen / not chosen, repeat / switch) share one term axis; every other module has its own. In the column labels
# a module's own state is written as its state symbol (v_wm, v_r, v_c, v_b; x for any other module).

SHARED_AXIS_SUFFIXES = ('_not_chosen', '_chosen', '_repeat', '_switch')
STATE_SYMBOLS = {
    'value_wm_reward': r'v_{\mathrm{wm}}',
    'value_reward': r'v_{\mathrm{r}}',
    'value_choice': r'v_{\mathrm{c}}',
    'bias_attention': r'v_{\mathrm{b}}',
}
INPUT_SYMBOLS = {'reward[t]': 'r', 'choice[t]': 'c', 'is_adjacent': r'\mathrm{adj}', 'is_opposite': r'\mathrm{opp}'}
MODULE_LABELS = {
    'value_wm_reward_chosen': 'WM reward (chosen)',
    'value_wm_reward_not_chosen': 'WM reward (not chosen)',
    'value_reward_chosen': 'reward value (chosen)',
    'value_reward_not_chosen': 'reward value (not chosen)',
    'value_choice': 'choice value',
    'bias_attention': 'attention bias',
}
CELL = 0.5        # inches per matrix cell
HEADER = 0.8      # height of a term axis above its rows (cells)
BLOCK_GAP = 0.4   # gap between two term axes (cells)


def module_family(module):
    """The module's name without the option it updates: modules of one family share a term axis."""
    for suffix in SHARED_AXIS_SUFFIXES:
        if module.endswith(suffix):
            return module[:-len(suffix)]
    return module


def generic_term(term, module):
    """A term as factors (name, power), with the module's own state renamed 'state', so that the same term of two
    modules of one family matches: 'value_reward_chosen*reward[t]' -> (('state', 1), ('reward[t]', 1))."""
    factors = []
    for factor in term.split('*'):
        name, _, power = factor.partition('^')
        factors.append(('state' if name == module else name, int(power or 1)))
    return tuple(factors)


def term_label(generic, family):
    """Mathtext label of a generic term, e.g. '$v_{r}\\,r$'."""
    if generic == (('1', 1),):
        return '$1$'
    symbols = []
    for name, power in generic:
        if name == 'state':
            symbol = STATE_SYMBOLS.get(family, 'x')
        else:
            symbol = INPUT_SYMBOLS.get(name, r'\mathrm{' + name.replace('_', r'\_') + '}')
        symbols.append(symbol + (f'^{power}' if power > 1 else ''))
    return '$' + r'\,'.join(symbols) + '$'


def term_axes(modules, library, cores):
    """Per module family that has a mechanism: (family, its modules that have one, columns). The columns are the
    generic terms used by these mechanisms, in library order."""
    axes = []
    for family in dict.fromkeys(module_family(m) for m in modules):
        members = [m for m in modules if module_family(m) == family and any(c['module'] == m for c in cores)]
        if not members:
            continue
        used = {generic_term(signed.rsplit(' (', 1)[0], c['module'])
                for c in cores if c['module'] in members for signed in c['core']}
        in_library_order = dict.fromkeys(generic_term(t, m) for m in members for t in library[m])
        axes.append((family, members, [term for term in in_library_order if term in used]))
    return axes


def plot_equation_matrix(modules, library, seed_results, matrices_per_seed, cores, output_path, clean=False):
    """One study's mechanisms as a reduced equation matrix.

    A filled cell is a signed term of the mechanism; its value is the term's median coefficient in the participants
    carrying the whole mechanism (pooled over the seeds carrying it; the median, as coefficients are bounded at zero
    per sign and skewed), its colour the phi at which the term joins the mechanism (1 - join
    distance; same scale as the mechanism correlations). Hatched cells: the term is not in that module's library. †: stateless module (its value is the sum of its terms).
    Saved annotated (<output_path>) and, with `clean`, without values and colourbar label (<output_path>_clean)."""
    if not cores:
        print(f'No recurring mechanisms; skipped {output_path}')
        return
    cores = sort_cores(cores, modules)
    codes = mechanism_codes(cores)
    axes = term_axes(modules, library, cores)
    n_columns = max(len(columns) for _, _, columns in axes)
    labels = {m: MODULE_LABELS.get(m, m.replace('_', ' ')) + (' †' if m not in library[m] else '') for m in modules}
    label_x = -(0.15 * max(len(labels[m]) for _, members, _ in axes for m in members) + 1.4)
    n_rows = [sum(c['module'] in members for c in cores) for _, members, _ in axes]
    height = sum(HEADER + rows for rows in n_rows) + BLOCK_GAP * (len(axes) - 1)

    for annotated in ((True, False) if clean else (True,)):
        fig, ax = plt.subplots(figsize=(CELL * (n_columns - label_x), CELL * height + 1.0))
        y = 0
        for family, members, columns in axes:
            # the term axis
            for j, term in enumerate(columns):
                ax.text(j + 0.5, y + HEADER - 0.15, term_label(term, family), ha='center', va='bottom', fontsize=9.5)
            y += HEADER

            for k, module in enumerate(members):
                module_cores = [(code, c) for code, c in zip(codes, cores) if c['module'] == module]
                if k:   # thin line between two modules on one axis
                    ax.plot([label_x, n_columns], [y, y], color='#9a9a9a', linewidth=0.8)
                ax.text(label_x, y + len(module_cores) / 2, labels[module], va='center', fontsize=8,
                        color='#555555', fontweight='bold')
                applicable = {generic_term(t, module) for t in library[module]}

                for code, core in module_cores:
                    ax.text(-0.2, y + 0.5, code, ha='right', va='center', fontsize=8.5)
                    for j, term in enumerate(columns):
                        not_in_library = term not in applicable
                        ax.add_patch(Rectangle((j, y), 1, 1, facecolor='#f4f4f4' if not_in_library else 'none',
                                               edgecolor='#e4e4e4', hatch='////' if not_in_library else None,
                                               linewidth=0.6, zorder=0))
                    coefficients = mechanism_coefficients(module, core['core'], seed_results, matrices_per_seed)
                    distances = join_distances(module, core['core'], seed_results)
                    for signed, values in coefficients.items():
                        j = columns.index(generic_term(signed.rsplit(' (', 1)[0], module))
                        color = PHI_CMAP(PHI_NORM(1 - distances[signed]))
                        ax.add_patch(Rectangle((j, y), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5))
                        if annotated:
                            ax.text(j + 0.5, y + 0.5, format_coefficient(np.median(values)), ha='center',
                                    va='center', fontsize=8, color=text_color(color))
                    y += 1
            y += BLOCK_GAP

        ax.set_xlim(label_x, n_columns)
        ax.set_ylim(y - BLOCK_GAP, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        add_horizontal_colorbar(fig, ax, 0, n_columns, PHI_CMAP, PHI_NORM,
                                'φ at which the term joins the mechanism' if annotated else None)
        save_figure(fig, output_path if annotated else f'{output_path}_clean')


# ── Correlation between mechanisms ──────────────────────────────────

def mechanism_presence(cores, matrices_per_seed, votes):
    """(participants x mechanisms) presence by consensus: all terms carried in >= `votes` seeds."""
    consensus = consensus_presence(matrices_per_seed, votes)
    columns = []
    for core in cores:
        module = consensus[core['module']]
        columns.append(module['presence'][:, [module['terms'].index(t) for t in core['core']]].all(axis=1))
    return np.column_stack(columns).astype(int)


def mechanism_cooccurrence(presence):
    """Pairwise phi, co-occurrence counts and BH-corrected Fisher exact p-values between mechanisms."""
    from scipy.stats import fisher_exact, false_discovery_control
    n_participants, n_mechanisms = presence.shape
    counts = presence.T @ presence
    with np.errstate(invalid='ignore', divide='ignore'):
        phi = np.corrcoef(presence.T)
    pairs = [(a, b) for a in range(n_mechanisms) for b in range(a + 1, n_mechanisms)]
    p_values = []
    for a, b in pairs:
        both = counts[a, b]
        table = [[both, counts[a, a] - both], [counts[b, b] - both, n_participants - counts[a, a] - counts[b, b] + both]]
        p_values.append(fisher_exact(table)[1])
    q = np.ones((n_mechanisms, n_mechanisms))
    for (a, b), value in zip(pairs, false_discovery_control(p_values) if pairs else []):
        q[a, b] = q[b, a] = value
    return phi, counts, q


def plot_mechanism_correlations(modules, cores, matrices_per_seed, votes, output_path, clean=False):
    """Phi between the mechanisms' presence across participants (consensus: >= `votes` seeds), as one lower-triangular matrix: rows M2..Mn,
    columns M1..Mn-1. Thin lines: module boundaries; dashed box: both mechanisms in one module; * = Fisher exact
    test, BH-corrected, q < .05. Saved annotated (<output_path>) and, with `clean`, without values and colourbar
    label (<output_path>_clean); <output_path>.csv holds carriers, co-occurrence, phi and q per pair."""
    if len(cores) < 2:
        print(f'Fewer than two recurring mechanisms; skipped {output_path}')
        return
    cores = sort_cores(cores, modules)
    codes = mechanism_codes(cores)
    module_of = [c['module'] for c in cores]
    phi, counts, q = mechanism_cooccurrence(mechanism_presence(cores, matrices_per_seed, votes))
    n = len(cores)

    for annotated in ((True, False) if clean else (True,)):
        fig, ax = plt.subplots(figsize=(CELL * (n - 1) + 0.8, CELL * (n - 1) + 1.2))
        for a in range(1, n):          # mechanism of the row
            for b in range(a):         # mechanism of the column
                x, y = b, a - 1
                color = PHI_CMAP(PHI_NORM(np.nan_to_num(phi[a, b])))
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white', linewidth=1.5))
                if module_of[a] == module_of[b]:
                    ax.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, facecolor='none', edgecolor=text_color(color),
                                           linewidth=1.2, linestyle=(0, (2, 1.5)), zorder=3))
                if annotated:
                    text = format_coefficient(phi[a, b]) + ('*' if q[a, b] < .05 else '')
                    ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', fontsize=7.5,
                            color=text_color(color))

        # thin lines where the module changes: a vertical one below the column, a horizontal one left of the row
        for k in range(1, n):
            if module_of[k] != module_of[k - 1]:
                ax.plot([k, k], [k - 1, n - 1], color='#555555', linewidth=0.7, zorder=2)
                ax.plot([0, k], [k - 1, k - 1], color='#555555', linewidth=0.7, zorder=2)
        # thick outline of the triangle: top-left corner, the staircase down the right, bottom edge, left edge
        outline = [(0, 0)] + [point for k in range(1, n) for point in ((k, k - 1), (k, k))]
        outline[-1] = (n - 1, n - 1)
        outline += [(0, n - 1), (0, 0)]
        ax.plot(*zip(*outline), color='#1f1f1f', linewidth=2.0, zorder=4, solid_capstyle='projecting')

        ax.set_xlim(-0.05, n - 1 + 0.05)
        ax.set_ylim(n - 1 + 0.05, -0.05)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(n - 1) + 0.5)
        ax.set_xticklabels(codes[:-1], fontsize=8.5)
        ax.set_yticks(np.arange(n - 1) + 0.5)
        ax.set_yticklabels(codes[1:], fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_horizontal_colorbar(fig, ax, 0, n - 1, PHI_CMAP, PHI_NORM,
                                'φ (presence across participants)' if annotated else None, gap=0.12)
        save_figure(fig, output_path if annotated else f'{output_path}_clean')

    rows = [dict(mechanism_a=codes[a], mechanism_b=codes[b], module_a=module_of[a], module_b=module_of[b],
                 n_a=counts[a, a], n_b=counts[b, b], n_both=counts[a, b], phi=phi[a, b], q_fisher_bh=q[a, b])
            for a in range(n) for b in range(a + 1, n)]
    pd.DataFrame(rows).to_csv(f'{output_path}.csv', index=False)

# ── Dendrogram ──────────────────────────────────────────────────────

def group_merge_distances(module, group, seed_results):
    """Per seed carrying `group` as a significant group: the height (1 - phi) at which all its terms have joined."""
    heights = []
    for result in seed_results:
        r = result[module]
        if any(group <= found for found in r['groups']):
            position = {t: i for i, t in enumerate(r['terms'])}
            members = [position[t] for t in group]
            heights.append(r['cophenetic'][np.ix_(members, members)].max())
    return np.array(heights)


def node_positions(linkage_matrix, n_leaves, leaf_order):
    """y position (scipy's left-orientation leaf coordinates) and members of every tree node."""
    y = {leaf: 5.0 + 10.0 * rank for rank, leaf in enumerate(leaf_order)}
    members = {leaf: {leaf} for leaf in range(n_leaves)}
    for row, (left, right, _, _) in enumerate(linkage_matrix):
        node = n_leaves + row
        y[node] = (y[int(left)] + y[int(right)]) / 2
        members[node] = members[int(left)] | members[int(right)]
    return y, members


def plot_mechanism_dendrogram(modules, seed_sets, seed_results, cores, votes, output_path, clean=False):
    """Per module, the joint-support tree (``joint_tree``) of the terms some other term is grouped with in >= `votes`
    seeds; x-axis: joint support of each branch. Mechanisms are coloured and (annotated version) labelled with their
    joint support; dashed line: the cut at `votes` seeds. Returns a table per mechanism: joint support, merge distance
    (1 - phi) in the seeds carrying it and the mean per-seed critical distance."""
    n_seeds = len(seed_sets)
    trees = {m: joint_tree(seed_sets, m, min_support=votes) for m in modules}
    modules = [m for m in modules if trees[m][1] is not None]   # modules with < 2 recurring terms have no tree
    if not modules:
        print(f'No recurring terms; skipped {output_path}')
        return pd.DataFrame()
    rows_out = []
    for annotated in ((True, False) if clean else (True,)):   # clean: no support labels, titles, axis label, legend
        height = [max(1.6, 0.36 * len(trees[m][0]) + 0.6) for m in modules]
        fig, axes = plt.subplots(len(modules), 1, figsize=(7.5, sum(height) + 0.8), squeeze=False,
                                 gridspec_kw={'height_ratios': height})

        for ax, module in zip(axes.ravel(), modules):
            terms, linkage_matrix = trees[module]
            n_leaves = len(terms)
            module_cores = sorted([c for c in cores if c['module'] == module], key=lambda c: -c['support'])
            group_of = {term: k for k, core in enumerate(module_cores) for term in core['core']}

            order = dendrogram(linkage_matrix, no_plot=True)['leaves']
            y, members = node_positions(linkage_matrix, n_leaves, order)

            def color(link):
                groups = {group_of.get(terms[i]) for i in members[link]}
                return GROUP_COLORS[groups.pop() % len(GROUP_COLORS)] if len(groups) == 1 and None not in groups else NEUTRAL

            dendrogram(linkage_matrix, labels=terms, orientation='left', link_color_func=color, ax=ax)
            ax.axvline(1 - votes / n_seeds, color='#777777', linestyle='--', linewidth=1, zorder=1)

            for k, core in enumerate(module_cores):
                leaves = {terms.index(t) for t in core['core']}
                node = min((n for n, m in members.items() if leaves <= m), key=lambda n: len(members[n]))
                height_node = linkage_matrix[node - n_leaves, 2] if node >= n_leaves else 0.0
                if annotated:
                    ax.annotate(f'{core["support"]}/{n_seeds}', (height_node, y[node]), xytext=(-3, 4),
                                textcoords='offset points', ha='right', fontsize=7.5,
                                color=GROUP_COLORS[k % len(GROUP_COLORS)], fontweight='bold')
                heights = group_merge_distances(module, core['core'], seed_results)
                critical = np.array([r[module]['critical_distance'] for r in seed_results])
                critical = critical[np.isfinite(critical)]
                if not annotated:
                    continue
                rows_out.append({'module': module, 'group': ', '.join(sorted(core['core'])),
                                 'joint_support': core['support'],
                                 'merge_distance_mean': heights.mean() if heights.size else np.nan,
                                 'merge_distance_sd': heights.std() if heights.size else np.nan,
                                 'critical_distance_mean': critical.mean() if critical.size else np.nan})

            for label in ax.get_yticklabels():
                k = group_of.get(label.get_text())
                if k is not None:
                    label.set_color(GROUP_COLORS[k % len(GROUP_COLORS)])
                    label.set_fontweight('bold')
            ticks = [0, 2, 4, 6, 8, 10] if n_seeds == 10 else list(range(n_seeds + 1))
            ax.set_xticks([1 - t / n_seeds for t in ticks])
            ax.set_xticklabels([str(t) for t in ticks])
            ax.set_xlim(1.02, -0.04)   # room right of 10/10 so merges supported by every seed stay visible
            if annotated:
                ax.set_title(module, fontsize=10, fontweight='bold')
            if annotated and ax is axes.ravel()[len(modules) - 1]:
                ax.set_xlabel(f'seeds with all terms of the branch in one group (of {n_seeds})', fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)

        handles = [plt.Line2D([], [], color=GROUP_COLORS[0], linewidth=2.5,
                              label='recurring mechanism (one colour each), labelled with its joint support'),
                   plt.Line2D([], [], color=NEUTRAL, linewidth=2.5, label='other merges'),
                   plt.Line2D([], [], color='#777777', linestyle='--', label=f'cut: all terms grouped together in ≥ {votes}/{n_seeds} seeds')]
        if annotated:
            fig.legend(handles=handles, loc='upper center', ncol=1, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.9 / fig.get_figheight()])
        save_figure(fig, output_path if annotated else f'{output_path}_clean')
    return pd.DataFrame(rows_out)


# ── Pipeline ────────────────────────────────────────────────────────

def recurring_mechanisms(matrices_per_seed, votes=6, cache_path=None, n_jobs=8, **clustering):
    """Per-seed significant groups and the mechanisms recurring in >= `votes` seeds, in figure order (M1, M2, ...).
    `clustering`: n_permutations, grouping_alpha, min_carriers of ``cluster_seed``."""
    modules = list(matrices_per_seed[0])
    seed_results = cluster_seeds(matrices_per_seed, cache_path, n_jobs, **clustering)
    seed_sets = [{(m, g) for m in modules for g in result[m]['groups']} for result in seed_results]
    cores = sort_cores(core_mechanisms(seed_sets, votes), modules)
    return seed_results, seed_sets, cores


def equation_discovery(matrices_per_seed, output_dir, votes=6, n_jobs=8, criteria=(), data=None,
                       participant_col='participant', min_count=5, alpha=0.05, clean=False, **clustering):
    """The whole analysis on the seeds' coefficient matrices (``seed_matrices``): mechanisms, their figures and,
    for every criterion in `criteria` (keyword arguments of ``analysis_mechanism_betas``, e.g.
    ``dict(criterion_col='reward')``), their beta effects on `data` (CSV path or DataFrame). With `clean`, every
    figure is also saved without annotations. Returns the mechanisms."""
    os.makedirs(output_dir, exist_ok=True)
    modules = list(matrices_per_seed[0])
    library = {m: matrices_per_seed[0][m]['terms'] for m in modules}
    seed_results, seed_sets, cores = recurring_mechanisms(
        matrices_per_seed, votes, os.path.join(output_dir, 'seed_clusters.pkl'), n_jobs, **clustering)

    plot_equation_matrix(modules, library, seed_results, matrices_per_seed, cores,
                         os.path.join(output_dir, 'figure_equation_matrix'), clean)
    plot_mechanism_correlations(modules, cores, matrices_per_seed, votes,
                                os.path.join(output_dir, 'figure_mechanism_correlations'), clean)
    table = plot_mechanism_dendrogram(modules, seed_sets, seed_results, cores, votes,
                                      os.path.join(output_dir, 'figure_mechanism_dendrogram'), clean)
    table.to_csv(os.path.join(output_dir, 'mechanisms.csv'), index=False)

    mechanisms = [(c['module'], c['core']) for c in cores]
    for criterion in criteria:
        analysis_mechanism_betas(matrices_per_seed, mechanisms, data, output_dir, votes=votes,
                                 participant_col=participant_col, min_count=min_count, alpha=alpha,
                                 prefix='mechanisms', figure_prefix='figure_mechanism_betas', clean=clean,
                                 **criterion)
    return cores


# ── Command line ────────────────────────────────────────────────────

def parse_value(text):
    """A command-line value as int or float when it is one (data codes such as 1, 3, 4), else as string."""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models-dir', required=True, help='directory with the fits (seeds); other files are skipped')
    parser.add_argument('--module', required=True, help='dotted module path exposing SpiceModel and CONFIG')
    parser.add_argument('--data', required=True, help='behavioural CSV the models were fitted to')
    parser.add_argument('--output-dir', default='equation_discovery')
    parser.add_argument('--clean', action='store_true', help='also save every figure without annotations')

    data = parser.add_argument_group('data columns (passed on to csv_to_dataset)')
    data.add_argument('--df-participant-id', default='participant')
    data.add_argument('--df-block', default='block')
    data.add_argument('--df-experiment-id', default='experiment')
    data.add_argument('--df-choice', nargs='+', default=['choice'])
    data.add_argument('--df-feedback', nargs='*', default=['reward'], help='none for no feedback column')
    data.add_argument('--additional-inputs', nargs='*', default=None)
    data.add_argument('--timeshift-additional-inputs', nargs='*', type=int, default=None)
    data.add_argument('--continuous-action', action='store_true')
    data.add_argument('--polynomial-degree', type=int, default=2)

    mechanisms = parser.add_argument_group('mechanisms')
    mechanisms.add_argument('--votes', type=int, default=6, help='seeds a mechanism (and a participant\'s term) needs')
    mechanisms.add_argument('--n-permutations', type=int, default=10000)
    mechanisms.add_argument('--grouping-alpha', type=float, default=0.05)
    mechanisms.add_argument('--min-carriers', type=int, default=5, help='participants a term needs to be clustered')
    mechanisms.add_argument('--n-jobs', type=int, default=8)

    betas = parser.add_argument_group('beta effects (only with --criterion)')
    betas.add_argument('--criterion', default=None, help='per-trial or per-participant column of --data')
    betas.add_argument('--criterion-type', choices=['continuous', 'discrete'], default='continuous')
    betas.add_argument('--criterion-aggregate', default='mean', help='per-participant aggregate of a continuous one')
    betas.add_argument('--block-col', default=None, help='sum the criterion per block before aggregating')
    betas.add_argument('--criterion-map', nargs='*', default=None, metavar='VALUE=LABEL',
                       help='relabel (or merge) discrete groups, e.g. Depression=Patients Bipolar=Patients')
    betas.add_argument('--comparison', nargs='*', default=None, metavar='GROUP:REFERENCE')
    betas.add_argument('--reference', default=None, help='compare every group with this one')
    betas.add_argument('--min-count', type=int, default=5, help='participants needed on each side of a test')
    betas.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    spice_module = importlib.import_module(args.module)
    feedback = args.df_feedback if args.df_feedback and args.df_feedback != ['none'] else None
    frame, dataset = load_dataset(
        args.data, spice_module,
        df_participant_id=args.df_participant_id,
        df_block=args.df_block,
        df_experiment_id=args.df_experiment_id,
        df_choice=args.df_choice[0] if len(args.df_choice) == 1 else args.df_choice,
        df_feedback=feedback[0] if feedback and len(feedback) == 1 else feedback,
        additional_inputs=args.additional_inputs,
        timeshift_additional_inputs=args.timeshift_additional_inputs,
        continuous_action=args.continuous_action,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    matrices_per_seed = seed_matrices(args.models_dir, model_loader(spice_module, dataset, args.polynomial_degree),
                                      os.path.join(args.output_dir, 'equation_matrices.pkl'))

    criteria = []
    if args.criterion:
        criteria.append(dict(
            criterion_col=args.criterion,
            criterion_type=args.criterion_type,
            criterion_aggregate=args.criterion_aggregate,
            block_col=args.block_col,
            reference=parse_value(args.reference) if args.reference else None,
            comparisons=[tuple(parse_value(v) for v in pair.split(':', 1)) for pair in args.comparison]
            if args.comparison else None,
            criterion_map={parse_value(k): v for k, v in (item.split('=', 1) for item in args.criterion_map)}
            if args.criterion_map else None,
        ))
    equation_discovery(matrices_per_seed, args.output_dir, votes=args.votes, n_jobs=args.n_jobs, criteria=criteria,
                       data=frame, participant_col=args.df_participant_id, min_count=args.min_count,
                       alpha=args.alpha, clean=args.clean, n_permutations=args.n_permutations,
                       grouping_alpha=args.grouping_alpha, min_carriers=args.min_carriers)


if __name__ == '__main__':
    main()

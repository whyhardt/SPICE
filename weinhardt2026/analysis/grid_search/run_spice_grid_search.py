"""Evaluate and dissect a multi-hyperparameter grid search (slurm_jobs/spice_stability_studies.sh).

Checkpoints are named `spice_<study>_th.._sw.._al.._er.._gp.._fp.._rs...pkl`;
every `<tag><value>` segment becomes a column. Each checkpoint is scored with
`evaluate_checkpoint` (in-sample BIC/AIC + trial likelihood, hold-out trial
likelihood, RNN-only reference likelihoods), and the cell table is then
dissected for redundancy between hyperparameters:

  - `variance_decomposition`: two-way ANOVA (main effects + pairwise
    interactions, categorical levels) per metric, with partial eta^2 and the
    seed (residual) share as the noise floor.
  - `sparsity_mediation`: how much of each metric the active-coefficient count
    alone explains, and how much each hyperparameter adds on top of it. A
    hyperparameter whose effect on BIC vanishes once `n_params_mean` is known
    is just another knob on the same sparsity dial -- redundant with any other
    such knob.
  - `pairwise_substitution`: for each pair of swept hyperparameters, the
    correlation of their main-effect profiles on n_params and the sign of their
    interaction on it. Two knobs that move sparsity the same way and saturate
    each other (sub-additive interaction) are substitutes.

Usage:
    python weinhardt2026/analysis/analysis_grid_search.py
"""

import importlib
import os
import re
import sys
from glob import glob
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'weinhardt2026'))

from spice import csv_to_dataset, split_data_along_blockdim
from weinhardt2026.analysis.analysis_sparsity_hpscan import evaluate_checkpoint, prepare_split


HP_TAGS = {
    'th': 'pruning_threshold',
    'sw': 'sindy_weight',
    'al': 'sindy_alpha',
    'er': 'ensemble_ratio',
    'gp': 'gate_penalty',
    'fp': 'feature_penalty',
    'rs': 'seed',
}
SELECTION_METRICS = ['BIC', 'AIC']
KEY_METRICS = ('BIC', 'AIC', 'trial_likelihood_test', 'trial_likelihood_test_rnn')
DIAGNOSTIC_METRICS = ['n_params_mean', 'trial_likelihood', 'trial_likelihood_test',
                      'generalization_gap', 'trial_likelihood_rnn', 'trial_likelihood_test_rnn']


# ── Checkpoint discovery ─────────────────────────────────────────────

def parse_grid_tags(path):
    """`..._th0.05_sw0.001_..._rs2.pkl` → {'th': 0.05, 'sw': 0.001, ..., 'rs': 2.0}."""
    basename = os.path.basename(path).removesuffix('.pkl')
    pattern = r'_(' + '|'.join(HP_TAGS) + r')(-?\d+(?:\.\d+)?(?:e-?\d+)?)(?=_|$)'
    return {tag: float(value) for tag, value in re.findall(pattern, basename)}


def find_grid_checkpoints(pkl_pattern):
    """Final (non-stage1) checkpoints whose filenames carry the full tag set of the grid."""
    paths = [p for p in sorted(glob(pkl_pattern)) if 'stage1' not in p]
    tags = {p: parse_grid_tags(p) for p in paths}
    n_tags = max((len(t) for t in tags.values()), default=0)
    skipped = [p for p, t in tags.items() if len(t) < n_tags]
    for p in skipped:
        print(f"  Skipping (incomplete tag set, older grid): {os.path.basename(p)}")
    return {p: t for p, t in tags.items() if len(t) == n_tags}


# ── Study loading (mirrors weinhardt2026/run.py hooks) ────────────────

def load_study_splits(module, data_path, test_blocks, data_kwargs=None, device=None):
    """Import a study module and build its train/test splits exactly as run.py does."""
    spice_module = importlib.import_module(module)
    data = data_path
    if hasattr(spice_module, 'prepare_dataframe'):
        data = spice_module.prepare_dataframe(pd.read_csv(data_path))
    dataset = csv_to_dataset(file=data, **(data_kwargs or {}))
    if getattr(spice_module, 'NORMALIZE_REWARDS', True):
        dataset.normalize_rewards()
    if hasattr(spice_module, 'prepare_dataset'):
        dataset = spice_module.prepare_dataset(dataset)
    dataset_train, dataset_test = split_data_along_blockdim(dataset, list(test_blocks))
    n_actions = getattr(spice_module, 'ESTIMATOR_KWARGS', {}).get('n_actions', dataset_train.ys.shape[-1])
    return spice_module, prepare_split(dataset_train, device), prepare_split(dataset_test, device), n_actions


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_grid(pkl_pattern, module, data_path, test_blocks, cache_path,
                  data_kwargs=None, model_kwargs=None, polynomial_degree=2, device=None):
    """Score every grid checkpoint; rows already in `cache_path` are reused, so reruns only add new runs."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cached = pd.read_csv(cache_path) if os.path.exists(cache_path) else pd.DataFrame(columns=['path'])
    checkpoints = find_grid_checkpoints(pkl_pattern)
    todo = {p: t for p, t in checkpoints.items() if os.path.basename(p) not in set(cached['path'])}
    print(f"{len(checkpoints)} grid checkpoints, {len(todo)} to evaluate")
    if not todo:
        return cached

    spice_module, split_train, split_test, n_actions = load_study_splits(
        module, data_path, test_blocks, data_kwargs=data_kwargs, device=device)

    rows = []
    for index, (path, tags) in enumerate(todo.items()):
        print(f"  [{index + 1}/{len(todo)}] {os.path.basename(path)}")
        rows.append({**tags, **evaluate_checkpoint(
            path, spice_module.SpiceModel, spice_module.CONFIG, n_actions, split_train, split_test,
            polynomial_degree=polynomial_degree, model_kwargs=model_kwargs, device=device,
        )})
        # checkpoint the cache so an interrupted run loses at most one evaluation
        pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).to_csv(cache_path, index=False)

    return pd.read_csv(cache_path)


def swept_tags(df):
    """Hyperparameter tags that actually vary across the grid (seed excluded)."""
    return [t for t in HP_TAGS if t in df and t != 'rs' and df[t].nunique() > 1]


# ── Redundancy analysis ──────────────────────────────────────────────

def summarize_cells(df, metrics):
    """Seed-aggregated mean, std and run count per grid cell."""
    tags = swept_tags(df)
    grouped = df.groupby(tags)[metrics]
    summary = grouped.mean().add_suffix('_mean').join(grouped.std().add_suffix('_seed_std'))
    summary['n_seeds'] = df.groupby(tags).size()
    return summary.reset_index()


def _anova_table(df, metric, tags, covariate=None):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    main = ' + '.join(f'C({t})' for t in tags)
    pairs = ' + '.join(f'C({a}):C({b})' for a, b in combinations(tags, 2))
    formula = f'{metric} ~ ' + (f'{covariate} + ' if covariate else '') + main + (f' + {pairs}' if pairs else '')
    return anova_lm(smf.ols(formula, data=df).fit(), typ=2)


def variance_decomposition(df, metrics):
    """Share of variance (eta^2 of the type-II sum of squares) per effect and metric; `Residual` = seed noise."""
    tags = swept_tags(df)
    shares = {}
    for metric in metrics:
        table = _anova_table(df, metric, tags)
        shares[metric] = table['sum_sq'] / table['sum_sq'].sum()
    out = pd.DataFrame(shares)
    out.index = [i.replace('C(', '').replace(')', '') for i in out.index]
    return out


def sparsity_mediation(df, metrics):
    """R² of metric ~ n_params alone, and the extra R² each hyperparameter adds on top of it."""
    import statsmodels.formula.api as smf

    tags = swept_tags(df)
    rows = []
    for metric in metrics:
        base = smf.ols(f'{metric} ~ n_params_mean + I(n_params_mean**2)', data=df).fit()
        row = {'metric': metric, 'R2_n_params_only': base.rsquared}
        for tag in tags:
            extended = smf.ols(f'{metric} ~ n_params_mean + I(n_params_mean**2) + C({tag})', data=df).fit()
            row[f'dR2_{tag}'] = extended.rsquared - base.rsquared
        full = smf.ols(f'{metric} ~ n_params_mean + I(n_params_mean**2) + '
                       + ' + '.join(f'C({t})' for t in tags), data=df).fit()
        row['dR2_all_hps'] = full.rsquared - base.rsquared
        rows.append(row)
    return pd.DataFrame(rows).set_index('metric')


def pairwise_substitution(df, metric='n_params_mean'):
    """Per pair: direction of each knob's effect on `metric` and whether they saturate each other.

    `effect_<a>` is the change of the cell mean from the lowest to the highest
    level of a (averaged over everything else). `interaction` is the
    difference-in-differences at the grid corners (high,high) − (high,low) −
    (low,high) + (low,low): same-signed effects with an interaction of the
    opposite sign mean the knobs partly do the same job (substitutes).
    """
    tags = swept_tags(df)
    rows = []
    for a, b in combinations(tags, 2):
        corners = df.groupby([a, b])[metric].mean()
        lo_a, hi_a, lo_b, hi_b = df[a].min(), df[a].max(), df[b].min(), df[b].max()
        effect_a = df[df[a] == hi_a][metric].mean() - df[df[a] == lo_a][metric].mean()
        effect_b = df[df[b] == hi_b][metric].mean() - df[df[b] == lo_b][metric].mean()
        interaction = (corners[(hi_a, hi_b)] - corners[(hi_a, lo_b)]
                       - corners[(lo_a, hi_b)] + corners[(lo_a, lo_b)])
        same_direction = np.sign(effect_a) == np.sign(effect_b)
        saturating = same_direction and np.sign(interaction) == -np.sign(effect_a)
        rows.append({'pair': f'{a}×{b}', f'effect_{metric}_first': effect_a, f'effect_{metric}_second': effect_b,
                     'interaction': interaction,
                     'saturation_ratio': abs(interaction) / min(abs(effect_a), abs(effect_b)) if saturating else 0.0,
                     'substitutes': bool(saturating)})
    return pd.DataFrame(rows).set_index('pair')


def best_cells(df, metric='BIC', n=10):
    """Grid cells ranked by seed-mean `metric`, with the seed std alongside for judging ties."""
    summary = summarize_cells(df, ['BIC', 'AIC', 'n_params_mean', 'trial_likelihood',
                                   'trial_likelihood_test', 'generalization_gap', 'trial_likelihood_test_rnn'])
    return summary.sort_values(f'{metric}_mean').head(n)


# ── Plotting ─────────────────────────────────────────────────────────

INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ORDINAL_BLUES = ['#86b6ef', '#2a78d6', '#104281']                    # ordinal sw levels, light → dark
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']  # fixed slot order
NEUTRAL = '#b5b4ae'                                                   # seed noise
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
METRIC_LABELS = {
    'BIC': 'SINDy BIC (train) ↓', 'AIC': 'SINDy AIC (train) ↓', 'n_params_mean': 'active coefficients / participant',
    'trial_likelihood': 'SINDy trial likelihood (train)', 'trial_likelihood_test': 'SINDy trial likelihood (test)',
    'generalization_gap': 'generalization gap', 'trial_likelihood_rnn': 'RNN trial likelihood (train)',
    'trial_likelihood_test_rnn': 'RNN trial likelihood (test)',
}


def _style():
    plt.rcParams.update({
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK_MUTED,
        'axes.labelcolor': INK, 'axes.titlecolor': INK, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.8,
        'legend.frameon': False, 'legend.fontsize': 8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
        'font.size': 9, 'axes.axisbelow': True,
    })


def _save(fig, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_main_effects(df, output_path,
                      metrics=KEY_METRICS):
    """Rows = metrics, columns = swept HPs; dots are runs, lines are seed+grid means ± SE per sw level."""
    _style()
    tags = swept_tags(df)
    sw_levels = sorted(df['sw'].unique()) if 'sw' in tags else [None]
    fig, axes = plt.subplots(len(metrics), len(tags), figsize=(2.9 * len(tags), 2.3 * len(metrics)),
                             sharey='row', squeeze=False)
    rng = np.random.default_rng(0)
    for col, tag in enumerate(tags):
        levels = sorted(df[tag].unique())
        x_of = {level: i for i, level in enumerate(levels)}
        # the sw column has only one line (sw itself is the x-axis)
        series = [(None, INK)] if tag == 'sw' else list(zip(sw_levels, ORDINAL_BLUES))
        offsets = np.linspace(-0.15, 0.15, len(series)) if len(series) > 1 else [0.0]
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for (sw, color), offset in zip(series, offsets):
                subset = df if sw is None else df[df['sw'] == sw]
                x_runs = subset[tag].map(x_of) + offset + rng.uniform(-0.04, 0.04, len(subset))
                ax.scatter(x_runs, subset[metric], s=9, color=color, alpha=0.35, linewidths=0)
                stats = subset.groupby(tag)[metric].agg(['mean', 'sem'])
                xs = [x_of[level] + offset for level in stats.index]
                ax.errorbar(xs, stats['mean'], yerr=stats['sem'], color=color, lw=2, marker='o', ms=5,
                            capsize=0, label=None if sw is None else f'sw = {sw:g}')
            ax.set_xticks(range(len(levels)), [f'{level:g}' for level in levels])
            ax.set_xlim(-0.5, len(levels) - 0.5)
            if row == 0:
                ax.set_title(f'{tag} ({HP_TAGS[tag]})')
            if row == len(metrics) - 1:
                ax.set_xlabel(tag)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Main effects per hyperparameter (dots: runs; lines: mean ± SE, split by sindy_weight)',
                 y=1.05, fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, output_path)


def plot_interaction_heatmaps(df, output_path, row_tag='al', col_tag='gp', facet_tag='er',
                              fixed=None, metrics=KEY_METRICS):
    """row_tag × col_tag heatmaps of seed-mean metrics, one figure row per facet_tag level.

    Darker = better for every panel (the ramp is flipped for metrics where lower is better).
    """
    _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    data = df.copy()
    for tag, value in (fixed or {}).items():
        data = data[data[tag] == value]
    facets = sorted(data[facet_tag].unique())
    fig, axes = plt.subplots(len(facets), len(metrics), figsize=(3.6 * len(metrics), 2.9 * len(facets)), squeeze=False)
    for col, metric in enumerate(metrics):
        lower_is_better = metric in ('BIC', 'AIC', 'n_params_mean', 'generalization_gap')
        vmin, vmax = data.groupby([facet_tag, row_tag, col_tag])[metric].mean().agg(['min', 'max'])
        for row, facet in enumerate(facets):
            ax = axes[row, col]
            ax.grid(False)
            subset = data[data[facet_tag] == facet]
            mean = subset.groupby([row_tag, col_tag])[metric].mean().unstack(col_tag)
            sem = subset.groupby([row_tag, col_tag])[metric].sem().unstack(col_tag)
            im = ax.imshow(mean.values, cmap=cmap.reversed() if lower_is_better else cmap,
                           vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
            for i in range(mean.shape[0]):
                for j in range(mean.shape[1]):
                    value = mean.values[i, j]
                    shade = (value - vmin) / (vmax - vmin + 1e-12)
                    dark_cell = shade > 0.55 if not lower_is_better else shade < 0.45
                    fmt = '{:.1f}' if abs(value) > 10 else ('{:.2f}' if abs(value) > 1 else '{:.4f}')
                    ax.text(j, i, fmt.format(value) + f'\n±{fmt.format(sem.values[i, j])}', ha='center',
                            va='center', fontsize=7.5, color='#ffffff' if dark_cell else INK)
            ax.set_xticks(range(mean.shape[1]), [f'{v:g}' for v in mean.columns])
            ax.set_yticks(range(mean.shape[0]), [f'{v:g}' for v in mean.index])
            ax.set_xlabel(col_tag)
            ax.set_ylabel(f'{row_tag}   ({facet_tag} = {facet:g})')
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
    fixed_text = ', '.join(f'{k} = {v:g}' for k, v in (fixed or {}).items())
    fig.suptitle(f'{row_tag} × {col_tag}' + (f' at {fixed_text}' if fixed_text else '')
                 + ' — mean ± SE over seeds, darker = better', y=1.02, fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, output_path)


def plot_tradeoffs(df, output_path, n_failed=0):
    """(a) SINDy BIC and (b) AIC over sparsity, (c) SINDy vs RNN test likelihood (extraction loss)."""
    _style()
    sw_levels = sorted(df['sw'].unique())
    markers = {level: m for level, m in zip(sorted(df['er'].unique()), ['o', '^', 's'])}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    panels = [('n_params_mean', 'BIC'), ('n_params_mean', 'AIC'),
              ('trial_likelihood_test_rnn', 'trial_likelihood_test')]
    for ax, (x, y) in zip(axes, panels):
        ax.grid(True, axis='both')
        for sw, color in zip(sw_levels, ORDINAL_BLUES):
            for er, marker in markers.items():
                subset = df[(df['sw'] == sw) & (df['er'] == er)]
                ax.scatter(subset[x], subset[y], s=30, marker=marker, color=color,
                           edgecolors='#fcfcfb', linewidths=0.8, label=f'sw = {sw:g}, er = {er:g}')
        ax.set_xlabel(METRIC_LABELS[x])
        ax.set_ylabel(METRIC_LABELS[y])
    lo = min(df['trial_likelihood_test'].min(), df['trial_likelihood_test_rnn'].min())
    hi = max(df['trial_likelihood_test'].max(), df['trial_likelihood_test_rnn'].max())
    axes[2].plot([lo, hi], [lo, hi], color=INK_MUTED, lw=1, ls='--', zorder=0)
    axes[2].text(hi, hi, ' SINDy = RNN', color=INK_MUTED, fontsize=8, va='bottom', ha='right')
    loss = df['trial_likelihood_test_rnn'] - df['trial_likelihood_test']
    axes[2].text(0.03, 0.95, f'RNN − SINDy = {loss.mean():.4f} ± {loss.std():.4f}',
                 transform=axes[2].transAxes, color=INK, fontsize=8.5, va='top')
    axes[0].set_title('(a) SINDy BIC vs. sparsity')
    axes[1].set_title('(b) SINDy AIC vs. sparsity')
    axes[2].set_title('(c) Test likelihood: SINDy vs. RNN')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.07))
    if n_failed:
        fig.text(0.995, -0.02, f'{n_failed} runs with diverged SINDy refit excluded', ha='right',
                 color=INK_MUTED, fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)


# colour follows the hyperparameter, so panels with different swept sets stay comparable
HP_COLORS = {'sw': CATEGORICAL[0], 'al': CATEGORICAL[1], 'er': CATEGORICAL[2], 'gp': CATEGORICAL[3],
             'interactions': CATEGORICAL[4], 'seed noise': NEUTRAL}


LOWER_IS_BETTER = {'BIC', 'AIC', 'n_params_mean', 'generalization_gap'}


def effect_directions(df, metrics):
    """Per swept HP and metric: change of the mean from the HP's lowest to its highest grid value."""
    return pd.DataFrame({
        metric: {tag: df[df[tag] == df[tag].max()][metric].mean() - df[df[tag] == df[tag].min()][metric].mean()
                 for tag in swept_tags(df)}
        for metric in metrics
    })


def _variance_shares(decomposition):
    """Per metric: main-effect shares, pooled pairwise interactions and the residual (seed noise)."""
    tags = [t for t in decomposition.index if ':' not in t and t != 'Residual']
    shares = pd.DataFrame({t: decomposition.loc[t] for t in tags})
    shares['interactions'] = decomposition.loc[[i for i in decomposition.index if ':' in i]].sum()
    shares['seed noise'] = decomposition.loc['Residual']
    return shares


def plot_variance_decomposition(panels, output_path, title=None):
    """Stacked horizontal bars per metric, one panel per {title: (decomposition, directions)}; seed noise in grey.

    `directions` comes from `effect_directions`. Each main-effect segment is
    labelled with an arrow for the direction the metric moves when the HP goes
    from its lowest to its highest value, and hatched when that move makes the
    criterion worse.
    """
    from matplotlib.patches import Patch
    _style()
    plt.rcParams['hatch.color'] = '#fcfcfb'
    plt.rcParams['hatch.linewidth'] = 1.2
    panel_shares = {name: _variance_shares(decomposition) for name, (decomposition, _) in panels.items()}
    metrics = next(iter(panel_shares.values())).index
    height = 0.45 * len(metrics) + 1.6
    fig, axes = plt.subplots(1, len(panels), figsize=(1.6 + 4.4 * len(panels), height), sharey=True, squeeze=False)
    y = np.arange(len(metrics))[::-1]

    legend = {}
    for ax, (name, shares) in zip(axes[0], panel_shares.items()):
        directions = panels[name][1]
        ax.grid(True, axis='x')
        ax.grid(False, axis='y')
        left = np.zeros(len(metrics))
        for component in shares.columns:
            color = HP_COLORS[component]
            width = shares[component].reindex(metrics).values
            has_direction = component in directions.index
            for yi, li, wi, metric in zip(y, left, width, metrics):
                delta = directions.loc[component, metric] if has_direction else 0.0
                worse = has_direction and (delta > 0) == (metric in LOWER_IS_BETTER)
                # 2px surface gap between adjacent segments; hatching = higher HP value worsens the criterion
                bar = ax.barh(yi, wi, left=li, color=color, height=0.62, edgecolor='#fcfcfb', linewidth=1.5,
                              hatch='////' if worse else None)
                legend.setdefault(component, Patch(facecolor=color, edgecolor='#fcfcfb'))
                arrow = (' ↑' if delta > 0 else ' ↓') if has_direction else ''
                if wi >= 0.08:
                    ax.text(li + wi / 2, yi, f'{wi:.0%}{arrow}', ha='center', va='center', fontsize=7.5,
                            color='#ffffff' if color == CATEGORICAL[0] else INK,
                            bbox=dict(boxstyle='round,pad=0.15', fc=color, ec='none') if worse else None)
                elif has_direction and wi >= 0.03:
                    ax.text(li + wi / 2, yi, arrow.strip(), ha='center', va='center', fontsize=7.5,
                            color='#ffffff' if color == CATEGORICAL[0] else INK)
            left += width
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_title(name)
        ax.spines['left'].set_visible(False)
    axes[0, 0].set_yticks(y, [METRIC_LABELS.get(m, m) for m in metrics])
    fig.supxlabel('share of variance across runs (type-II sum of squares)', fontsize=9, color=INK)

    order = [c for c in HP_COLORS if c in legend]
    labels = [f'{c} ({HP_TAGS[c]})' if c in HP_TAGS else
              {'interactions': 'interactions (pairwise)', 'seed noise': 'seed noise (residual)'}[c] for c in order]
    fig.tight_layout()
    n_columns = min(len(order), 3 * len(panels))
    handles = [legend[c] for c in order] + [
        Patch(facecolor=INK_MUTED, edgecolor='#fcfcfb', label='higher HP value improves the criterion'),
        Patch(facecolor=INK_MUTED, edgecolor='#fcfcfb', hatch='////', label='higher HP value worsens the criterion'),
    ]
    labels = labels + [h.get_label() for h in handles[-2:]]
    n_legend_rows = int(np.ceil(len(handles) / n_columns))
    fig.legend(handles, labels, loc='lower center', ncol=n_columns, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.01, '↑/↓: the metric rises/falls from the lowest to the highest grid value '
             'of the hyperparameter (difference of means over all other settings and seeds)',
             ha='center', va='top', fontsize=8, color=INK_MUTED)
    if title:
        fig.suptitle(title, y=1.0 + (0.3 + 0.25 * n_legend_rows) / height, fontsize=11, color=INK)
    _save(fig, output_path)


def plot_single_hp(df, tag, output_path, line_tag=None, title=None,
                   metrics=('BIC', 'AIC', 'trial_likelihood', 'trial_likelihood_test',
                            'trial_likelihood_rnn', 'trial_likelihood_test_rnn')):
    """One panel per metric: `tag` on the x-axis, dots = runs, lines = mean ± SE per `line_tag` level."""
    _style()
    levels = sorted(df[tag].unique())
    x_of = {level: i for i, level in enumerate(levels)}
    series = [(None, INK)] if line_tag is None else list(zip(sorted(df[line_tag].unique()), ORDINAL_BLUES))
    offsets = np.linspace(-0.12, 0.12, len(series)) if len(series) > 1 else [0.0]
    n_cols = 2
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.7 * n_rows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes.ravel(), metrics):
        for (level, color), offset in zip(series, offsets):
            subset = df if level is None else df[df[line_tag] == level]
            ax.scatter(subset[tag].map(x_of) + offset + rng.uniform(-0.03, 0.03, len(subset)), subset[metric],
                       s=12, color=color, alpha=0.45, linewidths=0)
            stats = subset.groupby(tag)[metric].agg(['mean', 'sem'])
            ax.errorbar([x_of[v] + offset for v in stats.index], stats['mean'], yerr=stats['sem'], color=color,
                        lw=2, marker='o', ms=5, label=None if level is None else f'{line_tag} = {level:g}')
        overall = df.groupby(tag)[metric].mean()
        ax.plot([x_of[v] for v in overall.index], overall.values, color=INK, lw=1, ls='--', label='mean over all')
        ax.set_xticks(range(len(levels)), [f'{v:g}' for v in levels])
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xlabel(f'{tag} ({HP_TAGS[tag]})')
        ax.set_title(METRIC_LABELS.get(metric, metric))
    for ax in axes.ravel()[len(metrics):]:
        ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, y=1.0 + 0.45 / (2.7 * n_rows), fontsize=11, color=INK)
    _save(fig, output_path)


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == '__main__':

    study = 'dezfouli2019'
    module = 'studies.dezfouli2019.spice_dezfouli2019'
    test_blocks = (3, 6, 9)
    data_kwargs = {}

    study_dir = f'weinhardt2026/studies/{study}'
    results_dir = os.path.join(study_dir, 'results', 'grid_search')
    os.makedirs(results_dir, exist_ok=True)

    df = evaluate_grid(
        pkl_pattern=os.path.join(study_dir, 'params', 'params_grid_search', f'spice_{study}_*.pkl'),
        module=module,
        data_path=os.path.join(study_dir, 'data', f'{study}.csv'),
        test_blocks=test_blocks,
        data_kwargs=data_kwargs,
        cache_path=os.path.join(results_dir, 'grid_search_runs.csv'),
    )

    pd.set_option('display.width', 200)
    metrics = SELECTION_METRICS + DIAGNOSTIC_METRICS

    failures = df.groupby(swept_tags(df))['sindy_failed'].agg(['sum', 'size'])
    print('\nDiverged SINDy refits (excluded from everything below):')
    print(failures[failures['sum'] > 0].to_string())
    df = df[~df['sindy_failed']]

    print('\nBest cells by BIC (mean over seeds):')
    print(best_cells(df, 'BIC').to_string(index=False, float_format='{:.4f}'.format))
    print('\nBest cells by AIC (mean over seeds):')
    print(best_cells(df, 'AIC').to_string(index=False, float_format='{:.4f}'.format))

    decomposition = variance_decomposition(df, metrics)
    print('\nVariance share per effect (Residual = seed noise):')
    print(decomposition.to_string(float_format='{:.3f}'.format))

    mediation = sparsity_mediation(df, metrics)
    print('\nVariance explained by sparsity alone, and what each HP adds on top:')
    print(mediation.to_string(float_format='{:.3f}'.format))

    substitution = pairwise_substitution(df, 'n_params_mean')
    print('\nPairwise substitution on coefficient count:')
    print(substitution.to_string(float_format='{:.3f}'.format))

    key_table = summarize_cells(df, list(KEY_METRICS)).sort_values('BIC_mean')
    key_table.to_csv(os.path.join(results_dir, 'grid_search_key_metrics.csv'), index=False)
    print('\nKey metrics per cell (sorted by SINDy BIC):')
    print(key_table.to_string(index=False, float_format='{:.4f}'.format))
    summarize_cells(df, metrics).to_csv(os.path.join(results_dir, 'grid_search_cells.csv'), index=False)
    decomposition.to_csv(os.path.join(results_dir, 'grid_search_variance_decomposition.csv'))
    mediation.to_csv(os.path.join(results_dir, 'grid_search_sparsity_mediation.csv'))
    substitution.to_csv(os.path.join(results_dir, 'grid_search_substitution.csv'))
    plot_main_effects(df, os.path.join(results_dir, 'fig_main_effects.png'))
    plot_interaction_heatmaps(df, os.path.join(results_dir, 'fig_al_x_gp.png'), fixed={'sw': df['sw'].min()})
    plot_tradeoffs(df, os.path.join(results_dir, 'fig_tradeoffs.png'), n_failed=int(failures['sum'].sum()))
    plot_variance_decomposition({'all runs': (decomposition[list(KEY_METRICS)],
                                              effect_directions(df, list(KEY_METRICS)))},
                                os.path.join(results_dir, 'fig_variance_decomposition.png'))
    by_sw_metrics = ['BIC', 'AIC', 'trial_likelihood', 'trial_likelihood_test', 'trial_likelihood_rnn',
                     'trial_likelihood_test_rnn']
    plot_variance_decomposition(
        {f'sw = {sw:g}  (n = {(df["sw"] == sw).sum()})': (variance_decomposition(df[df['sw'] == sw], by_sw_metrics),
                                                         effect_directions(df[df['sw'] == sw], by_sw_metrics))
         for sw in sorted(df['sw'].unique())},
        os.path.join(results_dir, 'fig_variance_decomposition_by_sw.png'),
        title='What drives each metric at fixed sindy_weight')
    pd.concat({f'sw={sw:g}': effect_directions(df[df['sw'] == sw], by_sw_metrics) for sw in sorted(df['sw'].unique())},
              names=['sw', 'hp']).to_csv(os.path.join(results_dir, 'grid_search_effect_directions_by_sw.csv'))

    # er dominates the variance but is settled at 0.5; decompose what remains
    fixed_er = df[df['er'] == 0.5]
    plot_variance_decomposition(
        {f'sw = {sw:g}  (n = {(fixed_er["sw"] == sw).sum()})': (
            variance_decomposition(fixed_er[fixed_er['sw'] == sw], by_sw_metrics),
            effect_directions(fixed_er[fixed_er['sw'] == sw], by_sw_metrics))
         for sw in sorted(fixed_er['sw'].unique())},
        os.path.join(results_dir, 'fig_variance_decomposition_by_sw_er0.5.png'),
        title='What drives each metric at fixed sindy_weight, er = 0.5')
    plot_single_hp(df[(df['sw'] == 0) & (df['er'] == 0.5)], 'gp',
                   os.path.join(results_dir, 'fig_gate_penalty_sw0_er0.5.png'), line_tag='al',
                   title='Influence of gate_penalty at sw = 0, er = 0.5 (dots: seeds; lines: mean ± SE)')
    print(f'\nSaved tables and figures to {results_dir}')

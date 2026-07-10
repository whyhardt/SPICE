"""
Prototype: Dumbbell / connected dot plot as alternative to figure2a bar chart.

One row per study, dots for each model connected by a thin line.
The gap between SPICE and competitors is immediately visible.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from figures.create_figure2a import STUDIES, _load_study_data


# ── Palette ────────────────────────────────────────────────────────
COLORS = {
    'Benchmark': '#7f7f7f',   # grey
    'GRU':       '#aaaaaa',   # light grey
    'SPICE-RNN': '#e07b54',   # warm coral
    'SPICE-EQ':  '#d62728',   # vivid red
}
MARKERS = {
    'Benchmark': 's',
    'GRU':       'D',
    'SPICE-RNN': '^',
    'SPICE-EQ':  'o',
}
MODEL_ORDER = ['GRU', 'Benchmark', 'SPICE-RNN', 'SPICE-EQ']


def _dumbbell_panel(ax, study_names, study_values, xlabel, highlight_best=True):
    """Draw one dumbbell panel.

    study_values: list of dicts {model_name: value}
    """
    n = len(study_names)
    y_positions = np.arange(n)

    for i, (name, vals) in enumerate(zip(study_names, study_values)):
        if not vals:
            continue
        all_vals = list(vals.values())
        vmin, vmax = min(all_vals), max(all_vals)

        # Connecting line (range bar)
        ax.plot([vmin, vmax], [i, i], color='#cccccc', linewidth=1.5, zorder=1)

        # Dots per model
        for model in MODEL_ORDER:
            if model not in vals:
                continue
            v = vals[model]
            ax.scatter(v, i, color=COLORS[model], marker=MARKERS[model],
                       s=70, zorder=3, edgecolors='white', linewidths=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(study_names, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_figure2a_dumbbell(studies=STUDIES, output_path='figures/prototype_figure2a_dumbbell'):
    # Separate discrete and continuous
    discrete = [s for s in studies if s['type'] == 'discrete']
    continuous = [s for s in studies if s['type'] == 'continuous']
    all_studies = discrete + continuous

    # Load data
    names_d, names_c = [], []
    pred_d, pred_c = [], []
    bic_d, bic_c = [], []
    spr_d, spr_c = [], []

    for s in discrete:
        pp, db, sp = _load_study_data(s['results_dir'], s['type'],
                                       gru_embedding=s.get('gru_embedding', False))
        names_d.append(s['name'].replace('\n', ' '))
        pred_d.append(pp)
        bic_d.append(db)
        spr_d.append(sp)

    for s in continuous:
        pp, db, sp = _load_study_data(s['results_dir'], s['type'],
                                       gru_embedding=s.get('gru_embedding', False))
        names_c.append(s['name'].replace('\n', ' '))
        pred_c.append(pp)
        bic_c.append(db)
        spr_c.append(sp)

    # ── Figure layout: 2 columns (discrete | continuous) × 3 rows ──
    has_continuous = len(continuous) > 0
    n_cols = 2 if has_continuous else 1
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 8),
                             gridspec_kw={'width_ratios': [len(discrete), max(len(continuous), 1)]} if has_continuous else {})
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    # Row 1: Predictive Performance
    _dumbbell_panel(axes[0, 0], names_d, pred_d, xlabel='Trial Likelihood')
    axes[0, 0].set_title('Predictive Performance', fontsize=11, fontweight='bold', loc='left')
    if has_continuous:
        _dumbbell_panel(axes[0, 1], names_c, pred_c, xlabel='R²')

    # Row 2: Model Selection (ΔBIC)
    _dumbbell_panel(axes[1, 0], names_d, bic_d, xlabel='ΔBIC / trial')
    axes[1, 0].set_title('Model Selection', fontsize=11, fontweight='bold', loc='left')
    if has_continuous:
        _dumbbell_panel(axes[1, 1], names_c, bic_c, xlabel='ΔBIC / trial')

    # Row 3: Individual Differences Recovery
    _dumbbell_panel(axes[2, 0], names_d, spr_d, xlabel='Spearman ρ')
    axes[2, 0].set_title('Individual Differences Recovery', fontsize=11, fontweight='bold', loc='left')
    if has_continuous:
        _dumbbell_panel(axes[2, 1], names_c, spr_c, xlabel='Spearman ρ')

    # Shared legend
    handles = [plt.Line2D([0], [0], marker=MARKERS[m], color='w',
               markerfacecolor=COLORS[m], markeredgecolor='white',
               markersize=8, label=m) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(f'{output_path}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{output_path}.pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}.png/.pdf")


if __name__ == '__main__':
    plot_figure2a_dumbbell()

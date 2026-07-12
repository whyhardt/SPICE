"""
Figure 3 Prototype: Performance + Structural Variance

Panels (saved individually):
  a) Performance: 3 metric rows (Predictive Performance, Delta BIC, Spearman rho)
     across all studies (one column per study)
  b) Structural variance: Term presence rates per module (one column per study)

Each panel saved as detailed + clean versions in figures/figure3/.
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from weinhardt2026.figures.create_figure2a import (
    STUDIES, COLORS, SLOT_ORDER, MAX_MODELS, BAR_WIDTH, SLOT_OFFSETS,
    _load_study_data, _plot_bars_cell,
)
from weinhardt2026.figures.panel_utils import save_panel


# ── Study model configurations for structural variance ──────────────
STUDY_MODELS = {
    'dezfouli2019': {
        'model_module': 'spice.precoded.workingmemory',
        'pkl': 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl',
        'n_actions': 2,
    },
    'ganesh2024a': {
        'model_module': 'weinhardt2026.studies.ganesh2024a.spice_ganesh2024a',
        'pkl': 'weinhardt2026/studies/ganesh2024a/params/spice_ganesh2024a.pkl',
        'n_actions': 2,
    },
    'eckstein2026': {
        'model_module': 'weinhardt2026.studies.eckstein2026.spice_eckstein2026',
        'pkl': 'weinhardt2026/studies/eckstein2026/params/spice_eckstein2026.pkl',
        'n_actions': 4,
    },
    'braun2018': {
        'model_module': 'weinhardt2026.studies.braun2018.spice_braun2018',
        'pkl': 'weinhardt2026/studies/braun2018/params/spice_braun2018.pkl',
        'n_actions': 2,
    },
    'bustamante2023': {
        'model_module': 'weinhardt2026.studies.bustamante2023.spice_bustamante2023',
        'pkl': 'weinhardt2026/studies/bustamante2023/params/spice_bustamante2023.pkl',
        'n_actions': 2,
    },
    'kolff2025': {
        'model_module': 'weinhardt2026.studies.kolff2025.spice_kolff2025',
        'pkl': 'weinhardt2026/studies/kolff2025/params/spice_kolff2025.pkl',
        'n_actions': 2,
    },
    'bruckner2025': {
        'model_module': 'weinhardt2026.studies.bruckner2025.spice_bruckner2025',
        'pkl': 'weinhardt2026/studies/bruckner2025/params/spice_bruckner2025.pkl',
        'n_actions': 1,
    },
    'weber2024': {
        'model_module': 'weinhardt2026.studies.weber2024.spice_weber2024',
        'pkl': 'weinhardt2026/studies/weber2024/params/spice_weber2024_continuous.pkl',
        'n_actions': 2,
    },
}


def _extract_study_key(results_dir):
    """Extract study key from results directory path."""
    for key in STUDY_MODELS:
        if key in results_dir:
            return key
    return None


def _load_presence_rates(study_key):
    """Load SPICE model and compute per-module term presence rates."""
    info = STUDY_MODELS[study_key]
    pkl_path = info['pkl']

    if not os.path.isfile(pkl_path):
        print(f"  Warning: {pkl_path} not found, skipping")
        return None, None, 0

    ckpt = torch.load(pkl_path, map_location='cpu')
    state = ckpt['model']

    module_presence = {}
    term_presence = {}
    n_participants = 0

    for key, tensor in state.items():
        if key.startswith('sindy_coefficients.'):
            mod_name = key.replace('sindy_coefficients.', '')
            coefs = tensor.float()
            n_participants = coefs.shape[1]

            is_nonzero = (coefs.abs() > 1e-8).float()
            ensemble_majority = (is_nonzero.mean(dim=0) > 0.5).float()
            avg_presence = ensemble_majority.mean(dim=1)

            term_rates = (avg_presence > 0).float().mean(dim=0).numpy()
            term_presence[mod_name] = term_rates
            module_presence[mod_name] = float(term_rates.mean())

    return module_presence, term_presence, n_participants


def _plot_panel_a(ordered_studies, study_data, n_discrete, n_continuous):
    """Panel a: Performance metrics (3 rows x N study columns)."""
    n_total = len(ordered_studies)
    has_sep = n_discrete > 0 and n_continuous > 0

    if has_sep:
        width_ratios = [1] * n_discrete + [0.15] + [1] * n_continuous
        n_cols = n_total + 1
        sep_col = n_discrete
    else:
        width_ratios = [1] * n_total
        n_cols = n_total
        sep_col = None

    def _col(study_idx):
        if has_sep and study_idx >= n_discrete:
            return study_idx + 1
        return study_idx

    n_rows = 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.2 * n_total + 1, 7),
        gridspec_kw={'width_ratios': width_ratios},
    )
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    if has_sep:
        for row_idx in range(n_rows):
            axes[row_idx, sep_col].set_visible(False)

    perf_rows = [
        ('Predictive Performance', 'Predictive\nPerformance', 0),
        ('Model Selection', '$\\Delta$BIC\n(per trial)', 1),
        ('Individual Differences', 'Spearman $\\rho$', 2),
    ]

    for row_idx, (title, ylabel, data_idx) in enumerate(perf_rows):
        for study_idx in range(n_total):
            col = _col(study_idx)
            ax = axes[row_idx, col]
            cell_data = study_data[study_idx][data_idx]

            if cell_data:
                _plot_bars_cell(ax, cell_data)
                vals = list(cell_data.values())
                vmin, vmax = min(vals), max(vals)
                span = vmax - vmin
                margin = span * 0.15 if span > 0 else max(abs(vmax) * 0.15, 0.05)
                ax.set_ylim(max(0, vmin - margin), vmax + margin)

            ax.set_xticks([])
            half = MAX_MODELS / 2 * BAR_WIDTH
            ax.set_xlim(-half - 0.05, half + 0.05)
            ax.grid(axis='y', alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            if study_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if study_idx == 0 and row_idx == 0:
                ax.set_title('a) Model Performance', fontsize=11,
                             fontweight='bold', loc='left')

            # Study name below bottom row
            if row_idx == n_rows - 1:
                study_name = ordered_studies[study_idx]['name'].replace('\n', ' ')
                ax.set_xlabel(study_name, fontsize=7.5)

    # Metric type annotations
    if has_sep and n_continuous > 0:
        mid_d = _col(n_discrete // 2)
        ax_d = axes[0, mid_d]
        ax_d.text(0, ax_d.get_ylim()[1], 'Avg. Trial Likelihood',
                  ha='center', va='bottom', fontsize=8, fontstyle='italic',
                  color='#555555')
        mid_c = _col(n_discrete + n_continuous // 2)
        ax_c = axes[0, mid_c]
        ax_c.text(0, ax_c.get_ylim()[1], '$R^2$',
                  ha='center', va='bottom', fontsize=8, fontstyle='italic',
                  color='#555555')

    # Shared legend
    all_handles, all_labels = {}, {}
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles[label] = handle
                all_labels[label] = label

    legend_order = [m for m in SLOT_ORDER if m in all_handles]
    if legend_order:
        fig.legend(
            [all_handles[m] for m in legend_order], legend_order,
            loc='upper center', ncol=len(legend_order), fontsize=9,
            framealpha=0.7, bbox_to_anchor=(0.5, 0.99),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_panel_b(ordered_studies, structural_data, n_discrete, n_continuous):
    """Panel b: Structural variance (strip/jitter plot, one column per study)."""
    n_total = len(ordered_studies)
    has_sep = n_discrete > 0 and n_continuous > 0

    if has_sep:
        width_ratios = [1] * n_discrete + [0.15] + [1] * n_continuous
        n_cols = n_total + 1
        sep_col = n_discrete
    else:
        width_ratios = [1] * n_total
        n_cols = n_total
        sep_col = None

    def _col(study_idx):
        if has_sep and study_idx >= n_discrete:
            return study_idx + 1
        return study_idx

    fig, axes = plt.subplots(
        1, n_cols, figsize=(2.2 * n_total + 1, 3),
        gridspec_kw={'width_ratios': width_ratios},
    )
    if n_cols == 1:
        axes = np.array([axes])

    if has_sep:
        axes[sep_col].set_visible(False)

    module_cmap = plt.cm.Set2
    for study_idx in range(n_total):
        col = _col(study_idx)
        ax = axes[col]

        sd = structural_data[study_idx]
        if sd is None or sd['term_presence'] is None:
            ax.text(0.5, 0.5, 'No model', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='grey')
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        term_pres = sd['term_presence']
        modules = list(term_pres.keys())
        n_modules = len(modules)

        for mod_idx, mod_name in enumerate(modules):
            rates = term_pres[mod_name]
            color = module_cmap(mod_idx / max(n_modules - 1, 1))

            jitter = np.random.RandomState(42 + mod_idx).uniform(
                -0.25, 0.25, size=len(rates)
            )
            x = np.full(len(rates), mod_idx) + jitter

            ax.scatter(x, rates, c=[color], s=18, alpha=0.75,
                       edgecolors='white', linewidths=0.3, zorder=3)

        ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.axhline(0.0, color='grey', linestyle=':', linewidth=0.5, alpha=0.4)

        ax.set_ylim(-0.08, 1.12)
        ax.set_xlim(-0.5, max(n_modules - 0.5, 0.5))

        short_names = []
        for mod_name in modules:
            short = mod_name.replace('value_', '').replace('_', '\n')
            short_names.append(short)
        ax.set_xticks(range(n_modules))
        ax.set_xticklabels(short_names, fontsize=5.5, rotation=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.15)

        if study_idx == 0:
            ax.set_ylabel('Term\nPresence Rate', fontsize=9)
            ax.set_title('b) Structural Variance (SPICE-EQ)', fontsize=11,
                         fontweight='bold', loc='left')

        study_name = ordered_studies[study_idx]['name'].replace('\n', ' ')
        ax.set_xlabel(study_name, fontsize=7.5)

    fig.tight_layout()
    return fig


def plot_figure3(
    studies=None,
    output_dir='figures/figure3',
):
    """Create Figure 3 panels: Performance + Structural Variance.

    Each panel saved as detailed + clean versions in output_dir.
    """
    if studies is None:
        studies = STUDIES

    discrete_studies = [s for s in studies if s['type'] == 'discrete']
    continuous_studies = [s for s in studies if s['type'] == 'continuous']
    ordered_studies = discrete_studies + continuous_studies
    n_discrete = len(discrete_studies)
    n_continuous = len(continuous_studies)

    # Load performance data
    study_data = []
    for study in ordered_studies:
        study_data.append(_load_study_data(
            study['results_dir'], study['type'],
            gru_embedding=study.get('gru_embedding', False),
        ))

    # Load structural variance data
    structural_data = []
    for study in ordered_studies:
        key = _extract_study_key(study['results_dir'])
        if key is not None:
            mod_pres, term_pres, n_part = _load_presence_rates(key)
            structural_data.append({
                'module_presence': mod_pres,
                'term_presence': term_pres,
                'n_participants': n_part,
            })
        else:
            structural_data.append(None)

    # Panel a: Performance
    fig_a = _plot_panel_a(ordered_studies, study_data, n_discrete, n_continuous)
    save_panel(fig_a, output_dir, 'panel_a_performance')

    # Panel b: Structural variance
    fig_b = _plot_panel_b(ordered_studies, structural_data, n_discrete, n_continuous)
    save_panel(fig_b, output_dir, 'panel_b_structural_variance')

    print(f"\nAll Figure 3 panels saved to {output_dir}/")

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
 

# ── Default study configuration ────────────────────────────────────
STUDIES = [
    {
        'name': 'Two-Armed Bandit\n(Dezfouli 2019)',
        'results_dir': 'weinhardt2026/studies/dezfouli2019/results',
        'type': 'discrete',
        'gru_embedding': False,
    },
    {
        'name': 'Perceptual\nTwo-Armed Bandit\n(Ganesh 2024)',
        'results_dir': 'weinhardt2026/studies/ganesh2024a/results',
        'type': 'discrete',
        'gru_embedding': False,
    },
    {
        'name': 'Four-Armed Bandit\n(Eckstein 2026)',
        'results_dir': 'weinhardt2026/studies/eckstein2026/results',
        'type': 'discrete',
        'gru_embedding': False,
    },
    {
        'name': 'Task Switching\n(Braun 2018)',
        'results_dir': 'weinhardt2026/studies/braun2018/results',
        'type': 'discrete',
        'gru_embedding': False,
    },
    {
        'name': 'Foraging\n(Bustamante 2023)',
        'results_dir': 'weinhardt2026/studies/bustamante2023/results',
        'type': 'discrete',
        'gru_embedding': False,
    },
    {
        'name': 'Chimpanzee\nCommunication\n(Kolff 2025)',
        'results_dir': 'weinhardt2026/studies/kolff2025/results',
        'type': 'discrete',
        'gru_embedding': False,
    },
    {
        'name': 'Predictive Inference\n(Bruckner 2025)',
        'results_dir': 'weinhardt2026/studies/bruckner2025/results',
        'type': 'continuous',
        'gru_embedding': False,
    },
    {
        'name': 'Continuous\nPredictive Inference\n(Weber 2024)',
        'results_dir': 'weinhardt2026/studies/weber2024/results',
        'type': 'continuous',
        'gru_embedding': False,
    },
]


# ── Colors ──────────────────────────────────────────────────────────
COLORS = {
    'Benchmark': '#1a3a5c',   # dark navy
    'GRU':       '#5b8db8',   # muted steel blue
    'SPICE-RNN': '#c44e52',   # muted brick red
    'SPICE-EQ':  '#8b2529',   # dark crimson
}

# ── Layout constants ────────────────────────────────────────────────
SLOT_ORDER = ['GRU', 'Benchmark', 'SPICE-EQ', 'SPICE-RNN']
MAX_MODELS = 4
BAR_WIDTH = 0.8 / MAX_MODELS

SLOT_OFFSETS = {
    'GRU':       -1.5 * BAR_WIDTH,
    'Benchmark': -0.5 * BAR_WIDTH,
    'SPICE-EQ':  +0.5 * BAR_WIDTH,
    'SPICE-RNN': +1.5 * BAR_WIDTH,
}

# Map CSV model-name variants → display name
_MODEL_DISPLAY = {
    'Benchmark': 'Benchmark',
    'GRU':       'GRU',
    'SPICE-RNN': 'SPICE-RNN',
    'SPICE-EQ':  'SPICE-EQ',
    'benchmark': 'Benchmark',
    'gru':       'GRU',
    'spice_rnn': 'SPICE-RNN',
    'spice':     'SPICE-EQ',
    'spice_eq':  'SPICE-EQ',
}


def _load_csv(path):
    """Load a CSV, return DataFrame or None if the file doesn't exist."""
    if path is not None and os.path.isfile(path):
        return pd.read_csv(path, index_col=0)
    return None


def _extract_column(df, column, models=None):
    """Extract a column from a DataFrame, mapping model names to display names.

    Returns dict {display_name: value} for the requested models.
    """
    if df is None or column not in df.columns:
        return {}

    result = {}
    for idx_name in df.index:
        display = _MODEL_DISPLAY.get(str(idx_name))
        if display is None:
            continue
        if models is not None and display not in models:
            continue
        val = df.loc[idx_name, column]
        if pd.notna(val):
            try:
                result[display] = float(val)
            except (ValueError, TypeError):
                pass
    return result


def _load_study_data(results_dir, study_type, gru_embedding=False):
    """Load all metric values for one study from its results directory.

    Returns:
        pred_perf: {model: value} for Row 1
        delta_bic: {model: value} for Row 2
        spearman:  {model: value} for Row 3
    """
    # Row 1 + Row 2: model evaluation CSV
    if study_type == 'discrete':
        eval_df = _load_csv(os.path.join(results_dir, 'model_evaluation.csv'))
        pred_perf = _extract_column(eval_df, 'Trial Lik.')
    else:
        eval_df = _load_csv(os.path.join(results_dir, 'model_evaluation_mse.csv'))
        pred_perf = _extract_column(eval_df, 'R²')

    # Skip models with perfect scores (fake entries from missing models)
    pred_perf = {m: v for m, v in pred_perf.items() if not np.isclose(v, 1.0)}

    delta_bic = _extract_column(eval_df, 'ΔBIC/trial', models={'Benchmark', 'SPICE-EQ'})
    # Remove models that were already filtered from pred_perf
    delta_bic = {m: v for m, v in delta_bic.items() if m in pred_perf or m not in {'Benchmark', 'GRU', 'SPICE-RNN', 'SPICE-EQ'}}

    # Row 3: Spearman rho
    spr_df = _load_csv(os.path.join(results_dir, 'generative_spearman.csv'))
    spearman = _extract_column(spr_df, 'Mean')

    # Only show GRU Spearman if the GRU uses participant embeddings
    if not gru_embedding:
        spearman.pop('GRU', None)

    return pred_perf, delta_bic, spearman


def _plot_bars_cell(ax, cell_data):
    """Plot grouped bars for a single study in one axes."""
    for model in SLOT_ORDER:
        if model not in cell_data:
            continue
        ax.bar(SLOT_OFFSETS[model], cell_data[model], BAR_WIDTH,
               label=model, color=COLORS[model], alpha=0.85,
               edgecolor='white', linewidth=0.5)



def plot_figure2a(
    studies: list[dict],
    output_path: str = 'figures/figure2a',
):
    """Create Figure 2a from per-study result CSVs.

    Each study gets its own column (and thus its own y-axis scale).
    Only the leftmost column displays y-axis tick labels.

    Args:
        studies: List of study dicts, each with keys:
            - 'name': Display label (may include newlines).
            - 'results_dir': Path to the study's results directory.
            - 'type': ``'discrete'`` or ``'continuous'``.
            - 'gru_embedding': ``True`` if the GRU uses participant embeddings
              (default ``False``). When ``False``, the GRU is excluded from
              the Spearman row since it cannot capture individual differences.
        output_path: Base path for output files (without extension).
    """
    # Separate discrete and continuous studies (preserving order)
    discrete_studies = [s for s in studies if s['type'] == 'discrete']
    continuous_studies = [s for s in studies if s['type'] == 'continuous']
    ordered_studies = discrete_studies + continuous_studies
    n_discrete = len(discrete_studies)
    n_continuous = len(continuous_studies)
    n_total = len(ordered_studies)

    study_names = [s['name'] for s in ordered_studies]

    # Load per-study data as list of 3-tuples
    study_data = []  # list of (pred_perf, delta_bic, spearman) dicts
    for study in ordered_studies:
        study_data.append(_load_study_data(
            study['results_dir'], study['type'],
            gru_embedding=study.get('gru_embedding', False),
        ))

    # Row definitions: (title, ylabel, data-index into 3-tuple)
    rows = [
        ('Predictive Performance',          'Predictive Performance',    0),
        ('Model Selection',                 '$\\Delta$BIC (per trial)',  1),
        ('Individual Differences Recovery', 'Spearman $\\rho$',         2),
    ]

    # ── Figure: 3 rows × n_total columns ────────────────────────────
    # Insert a narrow separator column between discrete and continuous
    has_sep = n_discrete > 0 and n_continuous > 0
    if has_sep:
        width_ratios = [1] * n_discrete + [0.15] + [1] * n_continuous
        n_cols = n_total + 1
        sep_col = n_discrete  # index of the separator column
    else:
        width_ratios = [1] * n_total
        n_cols = n_total
        sep_col = None

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2 * n_total + 1, 7),
        gridspec_kw={'width_ratios': width_ratios},
    )
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Map study index → column index (accounting for separator)
    def _col(study_idx):
        if has_sep and study_idx >= n_discrete:
            return study_idx + 1  # skip separator column
        return study_idx

    # Hide separator columns
    if has_sep:
        for row_idx in range(n_rows):
            axes[row_idx, sep_col].set_visible(False)

    # ── Plot each cell ──────────────────────────────────────────────
    for row_idx, (title, ylabel, data_idx) in enumerate(rows):
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

            # Formatting
            ax.set_xticks([])
            half_bar_group = MAX_MODELS / 2 * BAR_WIDTH
            ax.set_xlim(-half_bar_group - 0.05, half_bar_group + 0.05)
            ax.grid(axis='y', alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

            # Y-axis text label only on leftmost column
            if study_idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            else:
                ax.set_ylabel('')

            # Row title above leftmost column
            if study_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

            # Study name below bottom row
            if row_idx == len(rows) - 1:
                ax.set_xlabel(study_names[study_idx], fontsize=9)

    # Row 1: annotate discrete vs continuous metric regions
    if has_sep and n_continuous > 0:
        # "Avg. Trial Likelihood" centered over discrete columns
        mid_discrete = _col(n_discrete // 2)
        ax_d = axes[0, mid_discrete]
        ax_d.text(0, ax_d.get_ylim()[1], 'Avg. Trial Likelihood',
                  ha='center', va='bottom', fontsize=8.5, fontstyle='italic',
                  color='#555555')
        # "R²" centered over continuous columns
        mid_continuous = _col(n_discrete + n_continuous // 2)
        ax_c = axes[0, mid_continuous]
        ax_c.text(0, ax_c.get_ylim()[1], '$R^2$',
                  ha='center', va='bottom', fontsize=8.5, fontstyle='italic',
                  color='#555555')

    # ── Shared legend ───────────────────────────────────────────────
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

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Saved to {output_path}.pdf and .png")
    return fig


if __name__ == '__main__':
    plot_figure2a(STUDIES, output_path='figures/figure2a')

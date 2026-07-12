"""
Figure 4 Prototype: Individual Differences

Panels (saved individually):
  a) PCA scatter (PC1 vertical) — behavioral phenotype clusters
  b) Equation fingerprint heatmap (rows sorted by PC1, aligned with panel a)
  c) Equation differences by cluster (grouped bars)
  d) Diagnostic group effects (beta forest plot)

Each panel saved as detailed + clean (no text) versions in figures/figure4/.
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from spice import SpiceEstimator
from weinhardt2026.analysis.analysis_behavioral_clustering import (
    _extract_equation_features,
    _cluster_participants,
    _test_equation_differences,
)
from weinhardt2026.figures.panel_utils import save_panel


# ── Color palettes ──
CLUSTER_COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']


def _significance_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _compute_pca(df_metrics):
    """Compute PCA on behavioral metrics. Returns (coords, pca_obj, X_std, metric_cols)."""
    metric_cols = [c for c in df_metrics.columns if c not in ('participant_id', 'behavioral_cluster')]
    X = df_metrics[metric_cols].values.copy()

    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_std)

    return coords, pca, X_std, metric_cols


# ---------------------------------------------------------------------------
# Panel a: PCA scatter (PC1 vertical)
# ---------------------------------------------------------------------------

def _plot_panel_a(coords, labels, nearest, pc1_values):
    """PCA scatter with participants on y-axis (ranked by PC1), PC2 on x-axis."""
    n = len(pc1_values)
    sort_order = np.argsort(pc1_values)[::-1]
    rank = np.empty(n)
    for row_idx, orig_idx in enumerate(sort_order):
        rank[orig_idx] = row_idx

    x = coords[:, 1]  # PC2
    y = rank

    fig, ax = plt.subplots(figsize=(5, 7))

    unique_labels = np.sort(np.unique(labels))
    for k in unique_labels:
        mask = labels == k
        color = CLUSTER_COLORS[int(k) - 1]
        ax.scatter(x[mask], y[mask],
                   c=color, s=30, alpha=0.6, edgecolors='white', linewidths=0.3,
                   label=f'Cluster {k} (n={mask.sum()})', zorder=3)

    # Highlight centroids
    for k, pid_idx in nearest.items():
        color = CLUSTER_COLORS[int(k) - 1]
        ax.scatter(x[pid_idx], y[pid_idx],
                   c=color, s=140, marker='*', edgecolors='black',
                   linewidths=0.8, zorder=5)

    ax.set_xlabel('PC2', fontsize=10)
    ax.set_ylabel('PC1 (highest variance)', fontsize=10)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    tick_positions = np.linspace(0, n - 1, 5, dtype=int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f'{pc1_values[sort_order[i]]:.1f}' for i in tick_positions], fontsize=8)
    ax.legend(fontsize=8, loc='lower left', framealpha=0.7)
    ax.set_title('a) Behavioral Phenotype Clusters', fontsize=12,
                 fontweight='bold', loc='left')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel b: Equation fingerprint heatmap (sorted by PC1)
# ---------------------------------------------------------------------------

def _plot_panel_b(coeff_df, presence_df, labels, modules_list, pc1_values):
    """Heatmap: participants (rows, sorted by PC1) x terms (cols)."""
    sort_order = np.argsort(pc1_values)[::-1]

    coeff_vals = coeff_df.values[sort_order]
    presence_vals = presence_df.values[sort_order]
    ordered_labels = labels[sort_order]

    active = coeff_vals[presence_vals > 0]
    vmax = np.percentile(np.abs(active), 95) if len(active) > 0 else 1.0

    display = np.where(presence_vals > 0, coeff_vals, 0)
    cmap = plt.cm.RdBu_r.copy()

    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(
        display, aspect='auto', interpolation='nearest',
        cmap=cmap, vmin=-vmax, vmax=vmax,
    )

    # White overlay on absent terms
    absent_mask = presence_vals == 0
    absent_overlay = np.where(absent_mask, 1.0, np.nan)
    ax.imshow(
        absent_overlay, aspect='auto', interpolation='nearest',
        cmap=mcolors.ListedColormap(['white']), vmin=0, vmax=1,
        alpha=0.9,
    )

    # Module separation lines + labels
    col_idx = 0
    mod_names = list(coeff_df.columns)
    for mod_idx, mod_terms in enumerate(modules_list):
        mid = col_idx + len(mod_terms) / 2 - 0.5
        if len(mod_terms) > 0:
            first_col = mod_names[col_idx] if col_idx < len(mod_names) else ''
            mod_label = first_col.split(':')[0] if ':' in first_col else ''
            mod_short = mod_label.replace('value_', '').replace('_', '\n')
            ax.text(mid, -1.5, mod_short, ha='center', va='bottom',
                    fontsize=6.5, color='#333333', fontweight='bold', clip_on=False)
        col_idx += len(mod_terms)
        if col_idx < coeff_df.shape[1]:
            ax.axvline(col_idx - 0.5, color='black', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('SINDy Terms', fontsize=10)
    ax.set_ylabel('Participants (sorted by PC1)', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Coefficient', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('b) Equation Fingerprints', fontsize=12, fontweight='bold', loc='left')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel c: Equation differences per cluster
# ---------------------------------------------------------------------------

def _plot_panel_c(df_eq_tests, labels):
    """Grouped horizontal bars showing equation terms that differ between clusters."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sig = df_eq_tests[df_eq_tests['kw_p_corrected'] < 0.05].copy()
    if len(sig) == 0:
        ax.text(0.5, 0.5, 'No significant\ncluster differences',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('c) Equation Differences by Cluster', fontsize=12,
                     fontweight='bold', loc='left')
        fig.tight_layout()
        return fig

    sig = sig.sort_values('kw_stat', ascending=True).reset_index(drop=True)
    if len(sig) > 15:
        sig = sig.tail(15).reset_index(drop=True)

    unique_labels = np.sort(np.unique(labels))
    n_clusters = len(unique_labels)
    y_pos = np.arange(len(sig))
    bar_width = 0.8 / n_clusters

    for k_idx, k in enumerate(unique_labels):
        col_name = f'mean_c{k}'
        if col_name not in sig.columns:
            continue
        color = CLUSTER_COLORS[int(k) - 1]
        offsets = y_pos + (k_idx - n_clusters / 2 + 0.5) * bar_width
        ax.barh(offsets, sig[col_name], height=bar_width, color=color,
                alpha=0.75, edgecolor='white', linewidth=0.3,
                label=f'Cluster {k}')

    ax.axvline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_yticks(y_pos)

    term_labels = []
    for t in sig['term']:
        short = t.split(':')[-1] if ':' in t else t
        mod = t.split(':')[0] if ':' in t else ''
        mod_short = mod.replace('value_', '').replace('_', ' ')
        term_labels.append(f'{mod_short}: {short}')
    ax.set_yticklabels(term_labels, fontsize=7)

    ax.set_xlabel('Mean Coefficient', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.8)

    for i, (_, row) in enumerate(sig.iterrows()):
        sig_label = _significance_label(row['kw_p_corrected'])
        ax.text(ax.get_xlim()[1] * 0.98, i, sig_label,
                ha='right', va='center', fontsize=7, color='#555555')

    ax.set_title('c) Equation Differences by Cluster', fontsize=12,
                 fontweight='bold', loc='left')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel d: Forest plot of beta effects (diagnosis)
# ---------------------------------------------------------------------------

def _plot_panel_d(results_csv_path):
    """Forest plot from the discrete_odds_ratio_results.csv."""
    fig, ax = plt.subplots(figsize=(8, 6))

    df = pd.read_csv(results_csv_path)

    comparisons = []
    for col in df.columns:
        if col.endswith('_beta'):
            comparisons.append(col.replace('_beta', ''))

    if not comparisons:
        ax.text(0.5, 0.5, 'No beta data', ha='center', va='center',
                transform=ax.transAxes)
        fig.tight_layout()
        return fig

    rows = []
    for comp in comparisons:
        beta_col = f'{comp}_beta'
        se_col = f'{comp}_se'
        p_col = f'{comp}_p'
        sig_col = f'{comp}_sig'

        for _, row in df.iterrows():
            if pd.isna(row.get(beta_col)) or pd.isna(row.get(se_col)):
                continue
            if row.get(sig_col, 'ns') in ('ns', 'no_variation', 'always_present'):
                continue
            rows.append({
                'term': row['coefficient_clean'],
                'comparison': comp.replace('_vs_', ' vs '),
                'beta': float(row[beta_col]),
                'se': float(row[se_col]),
                'p': float(row.get(p_col, 1.0)),
                'sig': row.get(sig_col, 'ns'),
            })

    if not rows:
        ax.text(0.5, 0.5, 'No significant effects', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='grey')
        ax.set_title('d) Diagnostic Group Effects', fontsize=12,
                     fontweight='bold', loc='left')
        fig.tight_layout()
        return fig

    df_plot = pd.DataFrame(rows)
    df_plot = df_plot.sort_values('beta', key=abs, ascending=True).reset_index(drop=True)

    if len(df_plot) > 20:
        df_plot = df_plot.nlargest(20, 'beta', keep='all').sort_values(
            'beta', key=abs, ascending=True
        ).reset_index(drop=True)

    y_pos = np.arange(len(df_plot))
    comp_colors = plt.cm.Dark2.colors
    unique_comps = df_plot['comparison'].unique()
    comp_color_map = {c: comp_colors[i % len(comp_colors)] for i, c in enumerate(unique_comps)}

    for y, idx in zip(y_pos, range(len(df_plot))):
        row = df_plot.iloc[idx]
        color = comp_color_map[row['comparison']]

        if not np.isnan(row['se']):
            lo = row['beta'] - 1.96 * row['se']
            hi = row['beta'] + 1.96 * row['se']
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.5,
                    solid_capstyle='round', alpha=0.7)

        ax.scatter(row['beta'], y, color=color, s=40, zorder=5,
                   edgecolors='black', linewidths=0.3)

    ax.axvline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['term'], fontsize=7.5)
    ax.set_xlabel('Effect (beta)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles = [Line2D([0], [0], marker='o', color=comp_color_map[c], linestyle='-',
                       markersize=5, markeredgecolor='black', markeredgewidth=0.3)
               for c in unique_comps]
    ax.legend(handles, unique_comps, fontsize=7.5, loc='lower right',
              framealpha=0.8, borderpad=0.3)

    ax.set_title('d) Diagnostic Group Effects', fontsize=12,
                 fontweight='bold', loc='left')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_figure4(
    estimator,
    path_behavioral_metrics,
    results_csv_path,
    n_clusters=3,
    output_dir='figures/figure4',
):
    """Create Figure 4 panels: individual differences.

    Each panel saved as detailed + clean versions in output_dir.
    """
    df_metrics = pd.read_csv(path_behavioral_metrics)
    coeff_df, presence_df, term_names = _extract_equation_features(estimator)
    labels, Z, centroids, nearest = _cluster_participants(df_metrics, n_clusters)
    df_eq_tests = _test_equation_differences(labels, coeff_df, presence_df)
    coords, pca_obj, X_std, metric_cols = _compute_pca(df_metrics)
    pc1_values = coords[:, 0]

    modules = estimator.get_modules()
    candidate_terms = estimator.get_candidate_terms()
    modules_list = [candidate_terms[m] for m in modules]

    print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    sig_count = (df_eq_tests['kw_p_corrected'] < 0.05).sum()
    print(f"Significant terms: {sig_count} / {len(df_eq_tests)}")
    print(f"PCA variance explained: PC1={pca_obj.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca_obj.explained_variance_ratio_[1]:.1%}")

    # Panel a: PCA scatter
    fig_a = _plot_panel_a(coords, labels, nearest, pc1_values)
    save_panel(fig_a, output_dir, 'panel_a_scatter')

    # Panel b: Equation fingerprint heatmap
    fig_b = _plot_panel_b(coeff_df, presence_df, labels, modules_list, pc1_values)
    save_panel(fig_b, output_dir, 'panel_b_fingerprint')

    # Panel c: Equation differences by cluster
    fig_c = _plot_panel_c(df_eq_tests, labels)
    save_panel(fig_c, output_dir, 'panel_c_equation_diffs')

    # Panel d: Beta effects
    fig_d = _plot_panel_d(results_csv_path)
    save_panel(fig_d, output_dir, 'panel_d_beta_effects')

    print(f"\nAll Figure 4 panels saved to {output_dir}/")

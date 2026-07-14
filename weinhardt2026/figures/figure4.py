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
                   c=color, s=200, alpha=0.6, edgecolors='white', linewidths=0.3,
                   label=f'Cluster {k} (n={mask.sum()})', zorder=3)

    # Highlight centroids
    for k, pid_idx in nearest.items():
        color = CLUSTER_COLORS[int(k) - 1]
        ax.scatter(x[pid_idx], y[pid_idx],
                   c=color, s=400, marker='*', edgecolors='black',
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

def _plot_panel_b(coeff_df, presence_df, labels, modules_list, sort_values, sort_label='PC1',
                  diagnosis_per_participant=None, participant_ids=None):
    """Heatmap: participants (rows, sorted by sort_values) x terms (cols sorted by |corr|),
    with sidebars showing sort variable, diagnosis density, and diagnosis labels."""
    from scipy.stats import spearmanr
    DIAG_COLORS = {'Control': '#4daf4a', 'Bipolar': '#ff7f00', 'Depression': '#e41a1c'}

    # ── Sort participants (rows) by sort_values ──
    sort_order = np.argsort(sort_values)[::-1]
    n = len(sort_values)
    y_positions = np.arange(n)
    ordered_sort_values = sort_values[sort_order]

    # ── Split terms into two groups (FDR-corrected) ──
    # Group 1 (structural): terms whose presence significantly depends on sort_values (logistic β)
    # Group 2 (parametric): remaining terms, sorted by Spearman |ρ| of coefficient values
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import chi2
    from statsmodels.stats.multitest import multipletests

    col_names = list(coeff_df.columns)
    sv_std = StandardScaler().fit_transform(sort_values.reshape(-1, 1))

    # Pass 1: compute raw p-values for all terms
    structural_candidates = []  # (col_idx, |β|, raw_p)
    parametric_only = []        # (col_idx, |ρ|, raw_p) — terms with no structural variation
    for i, col in enumerate(col_names):
        present = (presence_df[col].values > 0).astype(int)
        if present.std() == 0:
            # Always or never present — no structural variation, test parametric only
            vals = coeff_df[col].values
            if vals.std() > 0:
                r, p = spearmanr(vals, sort_values)
                if not np.isnan(r):
                    parametric_only.append((i, abs(r), p))
        else:
            # Logistic regression: presence ~ standardized sort_values
            try:
                lr_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=0)
                lr_model.fit(sv_std, present)
                beta = float(lr_model.coef_[0][0])
                p_hat = np.clip(lr_model.predict_proba(sv_std)[:, 1], 1e-10, 1 - 1e-10)
                p0 = np.clip(present.mean(), 1e-10, 1 - 1e-10)
                ll = np.sum(present * np.log(p_hat) + (1 - present) * np.log(1 - p_hat))
                ll0 = np.sum(present * np.log(p0) + (1 - present) * np.log(1 - p0))
                lr_stat = -2 * (ll0 - ll)
                p_val = 1 - chi2.cdf(lr_stat, df=1)
            except Exception:
                p_val, beta = 1.0, 0.0
            structural_candidates.append((i, abs(beta), p_val))

    # Pass 2: FDR correction on structural p-values
    structural_terms = []  # (col_idx, |β|)
    structural_nonsig_indices = []  # indices that failed structural test → try parametric
    if structural_candidates:
        raw_ps = np.array([p for _, _, p in structural_candidates])
        reject, _, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')
        for j, (col_idx, effect, _) in enumerate(structural_candidates):
            if reject[j]:
                structural_terms.append((col_idx, effect))
            else:
                structural_nonsig_indices.append(col_idx)

    # Pass 3: collect parametric candidates (non-structural + failed structural)
    parametric_candidates = list(parametric_only)  # already have (col_idx, |ρ|, raw_p)
    for col_idx in structural_nonsig_indices:
        col = col_names[col_idx]
        vals = coeff_df[col].values
        if vals.std() > 0:
            r, p = spearmanr(vals, sort_values)
            if not np.isnan(r):
                parametric_candidates.append((col_idx, abs(r), p))

    # Pass 4: FDR correction on parametric p-values
    parametric_terms = []  # (col_idx, |ρ|)
    if parametric_candidates:
        raw_ps = np.array([p for _, _, p in parametric_candidates])
        reject, _, _, _ = multipletests(raw_ps, alpha=0.05, method='fdr_bh')
        for j, (col_idx, effect, _) in enumerate(parametric_candidates):
            if reject[j]:
                parametric_terms.append((col_idx, effect))

    # Sort each group by descending effect size
    structural_terms.sort(key=lambda x: x[1], reverse=True)
    parametric_terms.sort(key=lambda x: x[1], reverse=True)

    col_order = np.array([idx for idx, _ in structural_terms] + [idx for idx, _ in parametric_terms])
    n_structural = len(structural_terms)

    # Build original-column-index → module-index mapping
    orig_col_to_mod = {}
    module_names = []
    offset = 0
    for mod_idx, mod_terms in enumerate(modules_list):
        for _ in mod_terms:
            orig_col_to_mod[offset] = mod_idx
            offset += 1
        if mod_terms and sum(len(m) for m in modules_list[:mod_idx]) < len(col_names):
            first_col = col_names[sum(len(m) for m in modules_list[:mod_idx])]
            mod_name = first_col.split(':')[0] if ':' in first_col else f'Module {mod_idx}'
            module_names.append(mod_name)
        else:
            module_names.append(f'Module {mod_idx}')
    reordered_mod_indices = [orig_col_to_mod.get(i, 0) for i in col_order]

    # Apply both sorts (rows by sort_values, columns by |corr|)
    coeff_vals = coeff_df.values[:, col_order][sort_order]
    presence_vals = presence_df.values[:, col_order][sort_order]

    active = coeff_vals[presence_vals > 0]
    vmax = np.percentile(np.abs(active), 95) if len(active) > 0 else 1.0
    display = np.where(presence_vals > 0, coeff_vals, np.nan)

    # Custom diverging colormap that reserves white for absent terms only.
    # Start from visible red/blue so even small coefficients are distinguishable.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('fingerprint', [
        (0.0,  '#2166ac'),   # strong blue (most negative)
        (0.35, '#67a9cf'),   # medium blue
        (0.5,  '#d9d9d9'),   # light grey (zero — not white)
        (0.65, '#ef8a62'),   # medium red
        (1.0,  '#b2182b'),   # strong red (most positive)
    ])
    cmap.set_bad('white')  # NaN (absent) → white

    # Build diagnosis data if available
    has_diag = diagnosis_per_participant is not None and participant_ids is not None
    if has_diag:
        from scipy.ndimage import gaussian_filter1d
        ordered_pids = participant_ids[sort_order]
        ordered_diag = [diagnosis_per_participant.get(int(pid), 'Unknown') for pid in ordered_pids]
        diag_types = sorted(set(ordered_diag))

        # Smoothed density curves
        sigma = max(2, n // 15)
        indicators = np.zeros((n, len(diag_types)))
        for i, d in enumerate(ordered_diag):
            indicators[i, diag_types.index(d)] = 1.0
        smoothed = np.stack([gaussian_filter1d(indicators[:, j].astype(float), sigma)
                             for j in range(len(diag_types))], axis=1)

        # Categorical color array for diagnosis label strip
        diag_color_rgb = np.array([list(mcolors.to_rgb(DIAG_COLORS.get(d, '#999999')))
                                   for d in ordered_diag])

    # ── Layout: [avg_reward | density | diag_labels | structural | gap | parametric] ──
    has_both = n_structural > 0 and len(parametric_terms) > 0
    n_param = len(parametric_terms)

    if has_diag and has_both:
        fig, (ax_side, ax_diag, ax_labels, ax_struct, ax_gap, ax_param) = plt.subplots(
            1, 6, figsize=(16, 7), sharey=True,
            gridspec_kw={'width_ratios': [2, 2, 0.8, n_structural, 0.3, n_param],
                         'wspace': 0.02})
        ax_gap.set_visible(False)
    elif has_diag:
        fig, (ax_side, ax_diag, ax_labels, ax_single) = plt.subplots(
            1, 4, figsize=(14, 7), sharey=True,
            gridspec_kw={'width_ratios': [2, 2, 0.8, 10], 'wspace': 0.02})
        ax_struct = ax_single if n_structural > 0 else None
        ax_param = ax_single if n_param > 0 else None
    elif has_both:
        fig, (ax_side, ax_struct, ax_gap, ax_param) = plt.subplots(
            1, 4, figsize=(15, 7), sharey=True,
            gridspec_kw={'width_ratios': [2, n_structural, 0.3, n_param],
                         'wspace': 0.02})
        ax_gap.set_visible(False)
    else:
        fig, (ax_side, ax_single) = plt.subplots(
            1, 2, figsize=(13, 7), sharey=True,
            gridspec_kw={'width_ratios': [2, 10], 'wspace': 0.02})
        ax_struct = ax_single if n_structural > 0 else None
        ax_param = ax_single if n_param > 0 else None

    # ── 1. Sort variable sidebar (avg_reward bars, pointing left) ──
    ax_side.barh(y_positions, ordered_sort_values, height=1.0, color='#555555',
                 edgecolor='none', alpha=0.6)
    ax_side.set_ylim(-0.5, n - 0.5)
    ax_side.invert_yaxis()
    v_min, v_max = ordered_sort_values.min(), ordered_sort_values.max()
    v_margin = (v_max - v_min) * 0.05
    ax_side.set_xlim(v_max + v_margin, v_min - v_margin)
    ax_side.set_xlabel(sort_label, fontsize=9)
    ax_side.set_ylabel(f'Participants (sorted by {sort_label})', fontsize=10)
    ax_side.set_yticks([])
    ax_side.spines['top'].set_visible(False)
    ax_side.spines['right'].set_visible(False)

    # ── 2. Diagnosis density curves ──
    if has_diag:
        for j, d in enumerate(diag_types):
            color = DIAG_COLORS.get(d, '#999999')
            ax_diag.fill_betweenx(y_positions, 0, smoothed[:, j],
                                  color=color, alpha=0.3, linewidth=0)
            ax_diag.plot(smoothed[:, j], y_positions, color=color,
                         linewidth=1.2, label=d)
        ax_diag.set_xlim(0, None)
        ax_diag.invert_xaxis()
        ax_diag.set_xlabel('Density', fontsize=8)
        ax_diag.set_yticks([])
        ax_diag.spines['top'].set_visible(False)
        ax_diag.spines['right'].set_visible(False)
        ax_diag.legend(fontsize=6.5, loc='upper left', framealpha=0.8)

    # ── 3. Diagnosis label strip ──
    if has_diag:
        label_img = diag_color_rgb.reshape(n, 1, 3)
        ax_labels.imshow(label_img, aspect='auto', interpolation='nearest',
                         extent=[-0.5, 0.5, n - 0.5, -0.5])
        ax_labels.set_xticks([])
        ax_labels.set_xlabel('Diag', fontsize=7)
        ax_labels.spines['top'].set_visible(False)
        ax_labels.spines['right'].set_visible(False)
        ax_labels.spines['left'].set_visible(False)
        ax_labels.spines['bottom'].set_visible(False)

    # ── 4. Heatmaps (structural + parametric) ──
    hm_configs = []
    if n_structural > 0 and ax_struct is not None:
        hm_configs.append((ax_struct, display[:, :n_structural],
                           reordered_mod_indices[:n_structural], 'Structural'))
    if n_param > 0 and ax_param is not None:
        hm_configs.append((ax_param, display[:, n_structural:],
                           reordered_mod_indices[n_structural:], 'Parametric'))

    im = None
    for hm_ax, hm_display, hm_mod_indices, hm_title in hm_configs:
        cur_im = hm_ax.imshow(hm_display, aspect='auto', interpolation='nearest',
                              cmap=cmap, vmin=-vmax, vmax=vmax)
        if im is None:
            im = cur_im

        hm_ax.set_xlabel(hm_title, fontsize=9, fontweight='bold')
        hm_ax.set_xticks([])
        hm_ax.set_yticks([])

    # Colorbar on rightmost heatmap
    rightmost_ax = ax_param if (ax_param is not None and n_param > 0) else ax_struct
    if im is not None and rightmost_ax is not None:
        cbar = plt.colorbar(im, ax=rightmost_ax, fraction=0.02, pad=0.01)
        cbar.set_label('Coefficient', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Title on leftmost heatmap
    leftmost_ax = ax_struct if (ax_struct is not None and n_structural > 0) else ax_param
    if leftmost_ax is not None:
        leftmost_ax.set_title('b) Equation Fingerprints', fontsize=12,
                              fontweight='bold', loc='left')

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

def _plot_panel_d(results_csv_path, excluded_terms=None):
    """Forest plot from the discrete_odds_ratio_results.csv.

    Args:
        excluded_terms: set of raw coefficient names to exclude (e.g. low effect size).
    """
    if excluded_terms is None:
        excluded_terms = set()
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
            if row['coefficient'] in excluded_terms:
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

    # Group by unique terms, sorted by max |beta|
    unique_comps = df_plot['comparison'].unique()
    n_comps = len(unique_comps)
    term_max_beta = (df_plot.groupby('term')['beta']
                     .apply(lambda x: x.abs().max())
                     .sort_values(ascending=True))
    term_list = term_max_beta.index.tolist()
    if len(term_list) > 20:
        term_list = term_list[-20:]

    y_pos = np.arange(len(term_list))
    comp_colors = plt.cm.Dark2.colors
    comp_color_map = {c: comp_colors[i % len(comp_colors)] for i, c in enumerate(unique_comps)}
    bar_height = 0.4 / n_comps

    for c_idx, comp in enumerate(unique_comps):
        subset = df_plot[df_plot['comparison'] == comp]
        color = comp_color_map[comp]
        for i, term in enumerate(term_list):
            match = subset[subset['term'] == term]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            y_offset = i + (c_idx - n_comps / 2 + 0.5) * bar_height

            if not np.isnan(row['se']):
                lo = row['beta'] - 1.96 * row['se']
                hi = row['beta'] + 1.96 * row['se']
                ax.plot([lo, hi], [y_offset, y_offset], color=color, linewidth=1.5,
                        solid_capstyle='round', alpha=0.7)

            ax.scatter(row['beta'], y_offset, color=color, s=40, zorder=5,
                       edgecolors='black', linewidths=0.3)

    ax.axvline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(term_list, fontsize=7.5)
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
# Panel e: Merged cluster + diagnosis effects (shared y-axis from panel d)
# ---------------------------------------------------------------------------

def _plot_panel_e(df_eq_tests, results_csv_path, labels, excluded_terms=None):
    """Side-by-side: cluster mean coefficients (left) + diagnosis beta effects (right),
    sharing y-axis defined by significant diagnosis terms."""
    if excluded_terms is None:
        excluded_terms = set()

    # ── Right side data: significant diagnosis terms ──
    df_beta = pd.read_csv(results_csv_path)

    comparisons = [col.replace('_beta', '') for col in df_beta.columns if col.endswith('_beta')]
    if not comparisons:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No beta data', ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        return fig

    beta_rows = []
    for comp in comparisons:
        for _, row in df_beta.iterrows():
            beta_col, se_col = f'{comp}_beta', f'{comp}_se'
            p_col, sig_col = f'{comp}_p', f'{comp}_sig'
            if pd.isna(row.get(beta_col)) or pd.isna(row.get(se_col)):
                continue
            if row.get(sig_col, 'ns') in ('ns', 'no_variation', 'always_present'):
                continue
            if row['coefficient'] in excluded_terms:
                continue
            beta_rows.append({
                'coefficient': row['coefficient'],
                'term_clean': row['coefficient_clean'],
                'comparison': comp.replace('_vs_', ' vs '),
                'beta': float(row[beta_col]),
                'se': float(row[se_col]),
                'p': float(row.get(p_col, 1.0)),
            })

    if not beta_rows:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No significant diagnosis effects', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='grey')
        ax.set_title('e) Cluster & Diagnosis Effects', fontsize=12, fontweight='bold', loc='left')
        fig.tight_layout()
        return fig

    df_diag = pd.DataFrame(beta_rows)

    # Unique terms from diagnosis, sorted by max |beta|
    term_order = (df_diag.groupby('coefficient')['beta']
                  .apply(lambda x: x.abs().max())
                  .sort_values(ascending=True).index.tolist())
    if len(term_order) > 20:
        term_order = term_order[-20:]

    # ── Map panel d coefficient names to panel c term names ──
    # Panel c: "module:term", panel d: "module_term" (first colon → underscore)
    eq_term_lookup = {}
    for t in df_eq_tests['term'].values:
        key = t.replace(':', '_', 1)
        eq_term_lookup[key] = t

    # ── Build shared y-axis ──
    n_terms = len(term_order)
    y_pos = np.arange(n_terms)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, max(4, n_terms * 0.35)),
                                             sharey=True, gridspec_kw={'width_ratios': [1, 1],
                                                                       'wspace': 0.05})

    # ── Left panel: cluster mean coefficients ──
    unique_labels = np.sort(np.unique(labels))
    n_clusters = len(unique_labels)
    bar_width = 0.8 / n_clusters

    for k_idx, k in enumerate(unique_labels):
        col_name = f'mean_c{k}'
        vals = []
        for coeff_raw in term_order:
            eq_term = eq_term_lookup.get(coeff_raw)
            if eq_term is not None and col_name in df_eq_tests.columns:
                match = df_eq_tests[df_eq_tests['term'] == eq_term]
                vals.append(float(match[col_name].values[0]) if len(match) > 0 else 0.0)
            else:
                vals.append(0.0)

        color = CLUSTER_COLORS[int(k) - 1]
        offsets = y_pos + (k_idx - n_clusters / 2 + 0.5) * bar_width
        ax_left.barh(offsets, vals, height=bar_width, color=color,
                     alpha=0.75, edgecolor='white', linewidth=0.3,
                     label=f'Cluster {k}')

    ax_left.axvline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax_left.set_xlabel('Mean Coefficient', fontsize=10)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.legend(fontsize=7.5, loc='lower left', framealpha=0.8)
    ax_left.set_title('Cluster Effects', fontsize=11, fontweight='bold', loc='left')

    # Mark terms that are also cluster-significant
    for i, coeff_raw in enumerate(term_order):
        eq_term = eq_term_lookup.get(coeff_raw)
        if eq_term is not None:
            match = df_eq_tests[df_eq_tests['term'] == eq_term]
            if len(match) > 0 and match['kw_p_corrected'].values[0] < 0.05:
                sig_label = _significance_label(match['kw_p_corrected'].values[0])
                ax_left.text(ax_left.get_xlim()[0] * 0.02 if ax_left.get_xlim()[0] < 0 else -0.01,
                             i, sig_label, ha='right', va='center', fontsize=6.5, color='#555555')

    # ── Right panel: diagnosis beta forest plot ──
    comp_colors = plt.cm.Dark2.colors
    unique_comps = df_diag['comparison'].unique()
    n_comps = len(unique_comps)
    comp_color_map = {c: comp_colors[i % len(comp_colors)] for i, c in enumerate(unique_comps)}
    forest_bar_height = 0.4 / n_comps

    for c_idx, comp in enumerate(unique_comps):
        subset = df_diag[df_diag['comparison'] == comp]
        color = comp_color_map[comp]
        for i, coeff_raw in enumerate(term_order):
            match = subset[subset['coefficient'] == coeff_raw]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            y_offset = i + (c_idx - n_comps / 2 + 0.5) * forest_bar_height

            lo = row['beta'] - 1.96 * row['se']
            hi = row['beta'] + 1.96 * row['se']
            ax_right.plot([lo, hi], [y_offset, y_offset], color=color, linewidth=1.5,
                          solid_capstyle='round', alpha=0.7)
            ax_right.scatter(row['beta'], y_offset, color=color, s=40, zorder=5,
                             edgecolors='black', linewidths=0.3)

    ax_right.axvline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax_right.set_xlabel('Effect (beta)', fontsize=10)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)

    handles = [Line2D([0], [0], marker='o', color=comp_color_map[c], linestyle='-',
                       markersize=5, markeredgecolor='black', markeredgewidth=0.3)
               for c in unique_comps]
    ax_right.legend(handles, unique_comps, fontsize=7.5, loc='lower right',
                    framealpha=0.8, borderpad=0.3)
    ax_right.set_title('Diagnosis Effects', fontsize=11, fontweight='bold', loc='left')

    # ── Shared y-axis labels ──
    term_labels = []
    for coeff_raw in term_order:
        eq_term = eq_term_lookup.get(coeff_raw)
        if eq_term is not None:
            short = eq_term.split(':')[-1] if ':' in eq_term else eq_term
            mod = eq_term.split(':')[0] if ':' in eq_term else ''
            mod_short = mod.replace('value_', '').replace('_', ' ')
            term_labels.append(f'{mod_short}: {short}')
        else:
            term_labels.append(coeff_raw)

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(term_labels, fontsize=7)

    fig.suptitle('e) Cluster & Diagnosis Effects', fontsize=12, fontweight='bold',
                 x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Panel f: Behavioral metrics + diagnosis composition by cluster
# ---------------------------------------------------------------------------

def _plot_panel_f(df_metrics, labels, diagnosis_per_participant):
    """Single grouped horizontal bar chart: behavioral metrics + diagnosis ratios on y-axis,
    clusters as grouped bars.

    Args:
        df_metrics: Behavioral metrics DataFrame with participant_id column.
        labels: Array of cluster labels (1-indexed) per participant.
        diagnosis_per_participant: dict mapping participant_id (int) → diagnosis string.
    """
    unique_labels = np.sort(np.unique(labels))
    n_clusters = len(unique_labels)

    metric_cols = [c for c in df_metrics.columns if c not in ('participant_id', 'behavioral_cluster')]

    # Standardise behavioral metrics so they share the same scale
    metric_vals = df_metrics[metric_cols].values.copy()
    for j in range(metric_vals.shape[1]):
        col_vals = metric_vals[:, j]
        valid = ~np.isnan(col_vals)
        if valid.sum() > 1:
            mu, sd = np.nanmean(col_vals), np.nanstd(col_vals)
            if sd > 0:
                metric_vals[:, j] = (col_vals - mu) / sd
            else:
                metric_vals[:, j] = 0.0

    # Compute diagnosis proportions per cluster
    diag_labels = [diagnosis_per_participant.get(int(pid), 'Unknown')
                   for pid in df_metrics['participant_id'].values]
    diag_arr = np.array(diag_labels)
    diag_types = sorted(set(diag_arr))

    # Build combined y-axis: behavioral metrics + diagnosis proportions
    all_names = [c.replace('_', ' ') for c in metric_cols] + [f'% {d}' for d in diag_types]
    n_rows = len(all_names)

    fig, ax = plt.subplots(figsize=(8, max(4, n_rows * 0.45)))
    y_pos = np.arange(n_rows)
    bar_width = 0.8 / n_clusters

    for k_idx, k in enumerate(unique_labels):
        mask = labels == k
        # Behavioral metric means (standardised)
        metric_means = np.nanmean(metric_vals[mask], axis=0).tolist()
        # Diagnosis proportions
        cluster_diag = diag_arr[mask]
        n_in_cluster = mask.sum()
        diag_ratios = [
            (cluster_diag == d).sum() / n_in_cluster if n_in_cluster > 0 else 0.0
            for d in diag_types
        ]
        vals = metric_means + diag_ratios

        color = CLUSTER_COLORS[int(k) - 1]
        offsets = y_pos + (k_idx - n_clusters / 2 + 0.5) * bar_width
        ax.barh(offsets, vals, height=bar_width, color=color,
                alpha=0.75, edgecolor='white', linewidth=0.3,
                label=f'Cluster {k} (n={mask.sum()})')

    # Separator between metrics and diagnosis
    sep_y = len(metric_cols) - 0.5
    ax.axhline(sep_y, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.axvline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel('Standardised Mean / Proportion', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.8)
    ax.set_title('f) Behavioral Profile by Cluster', fontsize=12, fontweight='bold', loc='left')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel g: Model morphing — coefficient evolution along avg_reward direction
# ---------------------------------------------------------------------------

MODULE_DISPLAY_NAMES = {
    'value_reward_chosen': 'Value (chosen)',
    'value_reward_not_chosen': 'Value (unchosen)',
    'value_choice_chosen': 'Choice (chosen)',
    'value_choice_not_chosen': 'Choice (unchosen)',
}

MODULE_COLORS_MORPH = {
    'value_reward_chosen': '#1f77b4',
    'value_reward_not_chosen': '#ff7f0e',
    'value_choice_chosen': '#2ca02c',
    'value_choice_not_chosen': '#d62728',
}


def _load_morphing_data(results_dir):
    """Load morphing coefficients and validation data from .npz files."""
    coeffs_path = os.path.join(results_dir, 'morphing_coefficients.npz')
    val_path = os.path.join(results_dir, 'morphing_validation.npz')

    if not os.path.exists(coeffs_path):
        print(f"  Morphing data not found: {coeffs_path}")
        return None, None

    coeffs_data = np.load(coeffs_path, allow_pickle=True)
    val_data = np.load(val_path) if os.path.exists(val_path) else None

    return coeffs_data, val_data


def _select_varying_terms(coeffs_data, modules, min_coeff_range=0.01, min_ip_range=0.01):
    """Select terms that vary meaningfully across morphing steps.

    Returns list of (module, term_idx, term_name, variation_type) tuples,
    sorted by module then by total coefficient range descending.
    """
    selected = []
    for module in modules:
        mc = coeffs_data[f'{module}_mean_coefficients']  # (n_steps, n_terms)
        ip = coeffs_data[f'{module}_inclusion_probability']  # (n_steps, n_terms)
        names = coeffs_data[f'{module}_term_names']

        for j in range(len(names)):
            coeff_range = mc[:, j].max() - mc[:, j].min()
            ip_range = ip[:, j].max() - ip[:, j].min()

            if ip_range >= min_ip_range:
                selected.append((module, j, str(names[j]), 'structural', coeff_range))
            elif coeff_range >= min_coeff_range:
                selected.append((module, j, str(names[j]), 'parametric', coeff_range))

    # Sort: by module order, then by variation magnitude descending
    module_order = {m: i for i, m in enumerate(modules)}
    selected.sort(key=lambda x: (module_order.get(x[0], 99), -x[4]))

    return selected


def _plot_panel_g_heatmap(coeffs_data, val_data, modules):
    """Morphing panel: heatmap of coefficient values + validation line plot.

    Left: avg_reward vs morphing step (validation).
    Right: heatmap with terms on y-axis (grouped by module), morphing steps
           on x-axis. Color = coefficient value, opacity = inclusion probability.
    """
    step_values = coeffs_data['step_values']
    n_steps = int(coeffs_data['n_steps'])

    # Select varying terms
    selected = _select_varying_terms(coeffs_data, modules)
    if not selected:
        print("  No varying terms found for morphing heatmap.")
        return None

    n_terms = len(selected)

    # Build heatmap data
    coeff_matrix = np.zeros((n_terms, n_steps))
    ip_matrix = np.ones((n_terms, n_steps))
    term_labels = []
    module_boundaries = []
    module_names = []
    prev_module = None

    for i, (module, j, name, vtype, _) in enumerate(selected):
        coeff_matrix[i] = coeffs_data[f'{module}_mean_coefficients'][:, j]
        ip_matrix[i] = coeffs_data[f'{module}_inclusion_probability'][:, j]
        term_labels.append(name)

        if module != prev_module:
            if prev_module is not None:
                module_boundaries.append(i - 0.5)
            module_names.append((i, MODULE_DISPLAY_NAMES.get(module, module)))
            prev_module = module

    # Layout
    has_validation = val_data is not None
    if has_validation:
        fig, (ax_val, ax_hm) = plt.subplots(
            1, 2, figsize=(14, max(6, n_terms * 0.3)),
            gridspec_kw={'width_ratios': [3, n_steps], 'wspace': 0.3})
    else:
        fig, ax_hm = plt.subplots(figsize=(10, max(6, n_terms * 0.3)))
        ax_val = None

    # ── Validation line plot ──
    if ax_val is not None and has_validation:
        mean = val_data['avg_reward_mean']
        se = val_data['avg_reward_se']
        ax_val.fill_between(step_values, mean - se, mean + se, alpha=0.3, color='#1f77b4')
        ax_val.plot(step_values, mean, 'o-', color='#1f77b4', markersize=4, linewidth=1.5)
        ax_val.set_xlabel('Embedding projection', fontsize=9)
        ax_val.set_ylabel('Avg. reward (generated)', fontsize=9)
        ax_val.spines['top'].set_visible(False)
        ax_val.spines['right'].set_visible(False)
        ax_val.tick_params(labelsize=8)

        # Add correlation annotation
        from scipy.stats import pearsonr
        r, p = pearsonr(step_values, mean)
        ax_val.text(0.05, 0.95, f'r = {r:.3f}', transform=ax_val.transAxes,
                    fontsize=8, verticalalignment='top')

    # ── Coefficient heatmap ──
    vmax = np.percentile(np.abs(coeff_matrix), 95)
    cmap = plt.cm.RdBu_r

    # Apply inclusion probability as alpha
    rgba = cmap((coeff_matrix / vmax + 1) / 2)  # normalize to [0, 1] for cmap
    rgba[..., 3] = np.clip(ip_matrix, 0.15, 1.0)  # min alpha for visibility

    ax_hm.imshow(rgba, aspect='auto', interpolation='nearest')

    # Module boundaries
    for b in module_boundaries:
        ax_hm.axhline(b, color='white', linewidth=2)

    # Labels
    ax_hm.set_yticks(np.arange(n_terms))
    ax_hm.set_yticklabels(term_labels, fontsize=7, fontfamily='monospace')

    # X-axis: show a few projection values
    n_xticks = min(5, n_steps)
    xtick_idx = np.linspace(0, n_steps - 1, n_xticks, dtype=int)
    ax_hm.set_xticks(xtick_idx)
    ax_hm.set_xticklabels([f'{step_values[i]:+.1f}' for i in xtick_idx], fontsize=7)
    ax_hm.set_xlabel('Embedding projection (low → high avg. reward)', fontsize=9)

    # Module name annotations on right side
    for y_start, mod_name in module_names:
        # Find end of this module's block
        y_end = n_terms - 1
        for b in module_boundaries:
            if b > y_start:
                y_end = b - 0.5
                break
        y_mid = (y_start + y_end) / 2
        ax_hm.text(n_steps + 0.5, y_mid, mod_name, fontsize=7, va='center',
                   fontweight='bold', color='#444444')

    # Colorbar (for coefficient magnitude — ignoring alpha)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_hm, fraction=0.02, pad=0.15)
    cbar.set_label('Coefficient', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    title = 'g) Model Morphing: Coefficient Evolution'
    if ax_val is not None:
        ax_val.set_title(title, fontsize=12, fontweight='bold', loc='left')
    else:
        ax_hm.set_title(title, fontsize=12, fontweight='bold', loc='left')

    fig.tight_layout()
    return fig


def _plot_panel_g_ridgeline(coeffs_data, val_data, modules):
    """Morphing panel: ridgeline plot of coefficient evolution.

    Each row shows one term's coefficient value as a filled curve along the
    morphing axis, stacked vertically and grouped by module. Color encodes
    the module. Inclusion probability modulates opacity.
    """
    step_values = coeffs_data['step_values']
    n_steps = int(coeffs_data['n_steps'])

    # Select varying terms
    selected = _select_varying_terms(coeffs_data, modules)
    if not selected:
        print("  No varying terms found for morphing ridgeline.")
        return None

    n_terms = len(selected)
    has_validation = val_data is not None

    # Layout: optional validation on top, ridgeline below
    if has_validation:
        fig, (ax_val, ax_ridge) = plt.subplots(
            2, 1, figsize=(10, max(8, 2 + n_terms * 0.5)),
            gridspec_kw={'height_ratios': [1, max(3, n_terms * 0.4)]})
    else:
        fig, ax_ridge = plt.subplots(figsize=(10, max(6, n_terms * 0.5)))
        ax_val = None

    # ── Validation line plot ──
    if ax_val is not None and has_validation:
        mean = val_data['avg_reward_mean']
        se = val_data['avg_reward_se']
        ax_val.fill_between(step_values, mean - se, mean + se, alpha=0.3, color='#1f77b4')
        ax_val.plot(step_values, mean, 'o-', color='#1f77b4', markersize=3, linewidth=1.5)
        ax_val.set_ylabel('Avg. reward', fontsize=9)
        ax_val.spines['top'].set_visible(False)
        ax_val.spines['right'].set_visible(False)
        ax_val.tick_params(labelsize=7)
        ax_val.set_xticklabels([])

        from scipy.stats import pearsonr
        r, _ = pearsonr(step_values, mean)
        ax_val.text(0.02, 0.9, f'r = {r:.3f}', transform=ax_val.transAxes, fontsize=8)
        ax_val.set_title('g) Model Morphing along Avg. Reward', fontsize=12,
                         fontweight='bold', loc='left')

    # ── Ridgeline plot ──
    row_height = 1.0
    overlap = 0.4
    has_se = f'{modules[0]}_se_coefficients' in coeffs_data

    for i, (module, j, name, vtype, _) in enumerate(reversed(selected)):
        idx = n_terms - 1 - i  # plot bottom-to-top
        y_base = idx * (row_height - overlap)

        coeffs = coeffs_data[f'{module}_mean_coefficients'][:, j]
        ip = coeffs_data[f'{module}_inclusion_probability'][:, j]

        # SE across ensemble members (if available)
        if has_se:
            se = coeffs_data[f'{module}_se_coefficients'][:, j]
        else:
            se = np.zeros_like(coeffs)

        # Scale coefficient curve to fit within row_height
        # Use a shared scale factor for mean and SE
        c_range = coeffs.max() - coeffs.min()
        if c_range > 0:
            scale_factor = row_height * 0.8 / c_range
            c_min = coeffs.min()
            scaled = (coeffs - c_min) * scale_factor
            scaled_lo = ((coeffs - se) - c_min) * scale_factor
            scaled_hi = ((coeffs + se) - c_min) * scale_factor
        else:
            scaled = np.full_like(coeffs, row_height * 0.4)
            scaled_lo = scaled.copy()
            scaled_hi = scaled.copy()

        color = MODULE_COLORS_MORPH.get(module, '#888888')
        mean_ip = ip.mean()

        # Ensemble SE band (lighter, behind main fill)
        if has_se and se.max() > 0:
            ax_ridge.fill_between(step_values,
                                  y_base + np.clip(scaled_lo, 0, None),
                                  y_base + scaled_hi,
                                  alpha=max(0.08, mean_ip * 0.2), color=color,
                                  linewidth=0)

        # Fill under the curve (mean)
        ax_ridge.fill_between(step_values, y_base, y_base + scaled,
                              alpha=max(0.15, mean_ip * 0.6), color=color,
                              linewidth=0)
        ax_ridge.plot(step_values, y_base + scaled, color=color,
                      linewidth=1.0, alpha=max(0.3, mean_ip))

        # Baseline
        ax_ridge.axhline(y_base, color='grey', linewidth=0.3, alpha=0.3)

        # Term label
        ax_ridge.text(step_values[0] - (step_values[-1] - step_values[0]) * 0.02,
                      y_base + row_height * 0.3, name,
                      fontsize=6, ha='right', va='center', fontfamily='monospace')

    # Module boundaries and legend
    prev_module = None
    boundary_positions = []
    for i, (module, _, _, _, _) in enumerate(selected):
        idx = n_terms - 1 - i
        if module != prev_module and prev_module is not None:
            y_boundary = idx * (row_height - overlap) + row_height
            boundary_positions.append(y_boundary)
        prev_module = module

    for yb in boundary_positions:
        ax_ridge.axhline(yb, color='black', linewidth=0.8, alpha=0.4, linestyle='--')

    ax_ridge.set_xlabel('Embedding projection (low → high avg. reward)', fontsize=9)
    ax_ridge.set_yticks([])
    ax_ridge.spines['top'].set_visible(False)
    ax_ridge.spines['right'].set_visible(False)
    ax_ridge.spines['left'].set_visible(False)

    # Module legend
    from matplotlib.patches import Patch
    legend_elements = []
    seen = set()
    for module, _, _, _, _ in selected:
        if module not in seen:
            seen.add(module)
            legend_elements.append(
                Patch(facecolor=MODULE_COLORS_MORPH.get(module, '#888'),
                      label=MODULE_DISPLAY_NAMES.get(module, module), alpha=0.6))
    ax_ridge.legend(handles=legend_elements, fontsize=7, loc='upper right', framealpha=0.7)

    if ax_val is None:
        ax_ridge.set_title('g) Model Morphing along Avg. Reward', fontsize=12,
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
    path_raw_data=None,
    results_dir=None,
):
    """Create Figure 4 panels: individual differences.

    Each panel saved as detailed + clean versions in output_dir.

    Args:
        path_raw_data: Path to raw CSV with a 'diag' column for diagnosis info.
            Required for panel f (behavioral profile by cluster).
    """
    df_metrics = pd.read_csv(path_behavioral_metrics)
    if results_dir is None:
        results_dir = os.path.dirname(path_behavioral_metrics)
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

    # Cluster ↔ behavioral metrics correlation
    from scipy.stats import kruskal, spearmanr
    metric_cols_clean = [c for c in df_metrics.columns if c not in ('participant_id', 'behavioral_cluster')]
    print(f"\nCluster ↔ Behavioral Metrics (Kruskal-Wallis + Spearman r with cluster label):")
    print(f"  {'Metric':<30s} {'H-stat':>8s} {'p':>10s} {'Spearman r':>10s} {'sig':>5s}")
    print(f"  {'-'*68}")
    for col in metric_cols_clean:
        vals = df_metrics[col].values
        valid = ~np.isnan(vals)
        groups = [vals[valid & (labels == k)] for k in np.sort(np.unique(labels))]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        h_stat, h_p = kruskal(*groups)
        r, _ = spearmanr(labels[valid], vals[valid])
        sig = '***' if h_p < 0.001 else '**' if h_p < 0.01 else '*' if h_p < 0.05 else 'ns'
        print(f"  {col:<30s} {h_stat:>8.2f} {h_p:>10.4g} {r:>+10.3f} {sig:>5s}")

    # Variance filter: exclude terms with near-constant coefficients across participants
    # (low std → no individual variation to explain by cluster/diagnosis)
    term_stds = {}
    for term in coeff_df.columns:
        active = coeff_df[term].values[presence_df[term].values > 0]
        if len(active) > 1:
            term_stds[term] = np.std(active)
        elif len(active) == 1:
            term_stds[term] = 0.0
    std_values = np.array(list(term_stds.values()))
    std_threshold = np.percentile(std_values, 10) if len(std_values) > 0 else 0.0
    # Convert to raw coefficient format used in beta CSV (colon → underscore)
    excluded_terms = {t.replace(':', '_', 1) for t, s in term_stds.items() if s < std_threshold}
    print(f"\nVariance filter: std threshold={std_threshold:.5f} (10th percentile)")
    print(f"  Excluded {len(excluded_terms)} near-constant terms")

    # Load diagnosis mapping (used by panels b and f)
    diagnosis_per_participant = None
    if path_raw_data is not None:
        df_raw = pd.read_csv(path_raw_data)
        if 'diag' in df_raw.columns and 'participant' in df_raw.columns:
            id_map = {val: idx for idx, val in enumerate(df_raw['participant'].unique())}
            diagnosis_per_participant = {
                id_map[p]: df_raw[df_raw['participant'] == p]['diag'].iloc[0]
                for p in id_map
            }

    # Panel a: PCA scatter
    fig_a = _plot_panel_a(coords, labels, nearest, pc1_values)
    save_panel(fig_a, output_dir, 'panel_a_scatter')

    # Panel b: Equation fingerprint heatmap (sorted by avg_reward)
    sort_values = df_metrics['avg_reward'].values
    sort_label = 'Avg Reward'
    participant_ids = df_metrics['participant_id'].values
    fig_b = _plot_panel_b(coeff_df, presence_df, labels, modules_list, sort_values, sort_label,
                          diagnosis_per_participant=diagnosis_per_participant,
                          participant_ids=participant_ids)
    save_panel(fig_b, output_dir, 'panel_b_fingerprint')

    # Panel c: Equation differences by cluster
    fig_c = _plot_panel_c(df_eq_tests, labels)
    save_panel(fig_c, output_dir, 'panel_c_equation_diffs')

    # Panel d: Beta effects
    fig_d = _plot_panel_d(results_csv_path, excluded_terms=excluded_terms)
    save_panel(fig_d, output_dir, 'panel_d_beta_effects')

    # Panel e: Merged cluster + diagnosis effects
    fig_e = _plot_panel_e(df_eq_tests, results_csv_path, labels, excluded_terms=excluded_terms)
    save_panel(fig_e, output_dir, 'panel_e_cluster_diagnosis')

    # Panel f: Behavioral profile by cluster
    if diagnosis_per_participant is not None:
        fig_f = _plot_panel_f(df_metrics, labels, diagnosis_per_participant)
        save_panel(fig_f, output_dir, 'panel_f_behavioral_profile')
    else:
        print("Note: diagnosis data not available — skipping panel f")

    # Panel g: Model morphing
    coeffs_data, val_data = _load_morphing_data(results_dir)
    if coeffs_data is not None:
        fig_g1 = _plot_panel_g_heatmap(coeffs_data, val_data, modules)
        if fig_g1 is not None:
            save_panel(fig_g1, output_dir, 'panel_g_morphing_heatmap')

        fig_g2 = _plot_panel_g_ridgeline(coeffs_data, val_data, modules)
        if fig_g2 is not None:
            save_panel(fig_g2, output_dir, 'panel_g_morphing_ridgeline')
    else:
        print("Note: morphing data not available — skipping panel g")

    print(f"\nAll Figure 4 panels saved to {output_dir}/")

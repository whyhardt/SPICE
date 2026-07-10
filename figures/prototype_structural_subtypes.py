"""
Prototype: Structural subtypes visualization.

Clusters participants by their equation *structure* (which terms are
present/absent), not just parameter values. Shows that participants
fall into structurally distinct cognitive subtypes.

Layout:
  a) Clustered binary presence heatmap (participants × terms)
  b) Representative equations for each cluster
  c) Cluster-level summary (size, key distinguishing terms)
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import SpiceEstimator, SpiceConfig, BaseModel


def load_coefficients(model_path, model_module, n_actions=2):
    """Load model and extract ensemble-averaged coefficients."""
    mod = importlib.import_module(model_module)
    rnn_class = mod.SpiceModel
    spice_config = mod.CONFIG

    ckpt = torch.load(model_path, map_location="cpu")
    first_module = next(iter(spice_config.library_setup))
    ensemble_size = ckpt["model"][f"sindy_coefficients.{first_module}"].shape[0]
    n_participants = ckpt["model"][f"sindy_coefficients.{first_module}"].shape[1]
    del ckpt

    estimator = SpiceEstimator(
        spice_class=rnn_class,
        spice_config=spice_config,
        n_actions=n_actions,
        n_participants=n_participants,
        sindy_library_polynomial_degree=2,
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(model_path)

    coefficients = estimator.get_sindy_coefficients(aggregate=True)
    candidate_terms = estimator.get_candidate_terms()
    modules = estimator.get_modules()

    return coefficients, candidate_terms, modules, n_participants, estimator


def build_presence_matrix(coefficients, candidate_terms, modules, n_participants,
                          threshold=1e-10):
    """Build binary presence matrix and coefficient value matrix."""
    col_names = []
    col_modules = []
    presence_data = []
    value_data = []

    for module in modules:
        coefs = coefficients[module].detach().cpu().numpy()  # (P, X, T)
        terms = candidate_terms[module]
        coef_avg = coefs.mean(axis=1)  # (P, T)

        for t_idx, term in enumerate(terms):
            col_name = f"{module}:{term}"
            col_names.append(col_name)
            col_modules.append(module)
            presence_data.append((np.abs(coef_avg[:, t_idx]) > threshold).astype(float))
            value_data.append(coef_avg[:, t_idx])

    presence_df = pd.DataFrame(
        np.column_stack(presence_data),
        columns=col_names,
        index=[f"P{i}" for i in range(n_participants)],
    )
    value_df = pd.DataFrame(
        np.column_stack(value_data),
        columns=col_names,
        index=[f"P{i}" for i in range(n_participants)],
    )

    return presence_df, value_df, col_modules


def plot_structural_subtypes(
    model_path,
    model_module,
    n_actions=2,
    n_clusters=3,
    output_path='figures/prototype_structural_subtypes',
    title='Structural Subtypes of Cognitive Dynamics',
    filter_zero_cols=True,
):
    """Create structural subtypes visualization."""
    coefficients, candidate_terms, modules, n_participants, estimator = \
        load_coefficients(model_path, model_module, n_actions=n_actions)

    presence_df, value_df, col_modules = build_presence_matrix(
        coefficients, candidate_terms, modules, n_participants
    )

    # Filter columns with no variation
    if filter_zero_cols:
        varying_mask = presence_df.std() > 0
        # Also keep columns that are always present (interesting for equations)
        always_present = presence_df.mean() == 1.0
        keep = varying_mask | always_present
        presence_df = presence_df.loc[:, keep]
        value_df = value_df.loc[:, keep]
        col_modules = [m for m, k in zip(col_modules, keep) if k]

    # Hierarchical clustering on presence patterns
    if len(presence_df) < 3:
        print("Too few participants for clustering.")
        return

    # Use Jaccard distance for binary presence
    dist_matrix = pdist(presence_df.values, metric='jaccard')
    # Handle NaN distances (when all values are identical)
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
    Z = linkage(dist_matrix, method='ward')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    presence_df['cluster'] = cluster_labels

    # Sort by cluster
    presence_df = presence_df.sort_values('cluster')
    value_df = value_df.loc[presence_df.index]
    cluster_col = presence_df.pop('cluster')

    # ── Figure layout ──
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3, 1],
                           height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # Panel a: Presence heatmap with cluster annotation
    ax_heat = fig.add_subplot(gs[0, 0])

    # Module color map
    unique_modules = list(dict.fromkeys(col_modules))
    module_palette = sns.color_palette("husl", len(unique_modules))
    module_cmap = dict(zip(unique_modules, module_palette))

    # Cluster color map
    cluster_palette = sns.color_palette("Set2", n_clusters)

    # Plot presence heatmap
    im = ax_heat.imshow(
        presence_df.values, aspect='auto', cmap='Blues',
        vmin=0, vmax=1, interpolation='nearest',
    )

    # Cluster color bar on left
    for i, (idx, cl) in enumerate(cluster_col.items()):
        ax_heat.plot([-1.5, -0.8], [i, i], color=cluster_palette[cl - 1],
                     linewidth=3, solid_capstyle='butt', clip_on=False)

    # Module color bar on top
    for j, mod in enumerate(col_modules):
        ax_heat.plot([j, j], [-0.8, -1.5], color=module_cmap[mod],
                     linewidth=4, solid_capstyle='butt', clip_on=False)

    ax_heat.set_xticks(range(len(presence_df.columns)))
    ax_heat.set_xticklabels(presence_df.columns, rotation=90, fontsize=5, ha='center')
    n_parts = len(presence_df)
    if n_parts <= 60:
        ax_heat.set_yticks(range(n_parts))
        ax_heat.set_yticklabels(presence_df.index, fontsize=5)
    else:
        ax_heat.set_yticks([])
    ax_heat.set_xlabel('SINDy Terms', fontsize=10)
    ax_heat.set_ylabel('Participants (clustered)', fontsize=10)
    ax_heat.set_title('a) Equation Structure (term presence)', fontsize=11,
                       fontweight='bold', loc='left')

    # Panel b: Cluster summary
    ax_summary = fig.add_subplot(gs[0, 1])
    ax_summary.axis('off')
    ax_summary.set_title('b) Cluster Summary', fontsize=11, fontweight='bold', loc='left')

    summary_text = ""
    for cl in range(1, n_clusters + 1):
        cl_mask = cluster_col == cl
        cl_count = cl_mask.sum()
        cl_presence = presence_df[cl_mask].mean()
        # Distinguish terms: present in this cluster but rare in others
        other_presence = presence_df[~cl_mask].mean()
        diff = cl_presence - other_presence
        distinguishing = diff.nlargest(3)
        missing = diff.nsmallest(3)

        summary_text += f"Cluster {cl} (n={cl_count})\n"
        summary_text += f"  Distinctive terms:\n"
        for term, d in distinguishing.items():
            if d > 0.1:
                short_term = term.split(':')[-1] if ':' in term else term
                summary_text += f"    + {short_term} ({cl_presence[term]:.0%})\n"
        for term, d in missing.items():
            if d < -0.1:
                short_term = term.split(':')[-1] if ':' in term else term
                summary_text += f"    − {short_term} ({cl_presence[term]:.0%})\n"
        summary_text += "\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8',
                              edgecolor='#cccccc'))

    # Panel c: Mean coefficient values per cluster
    ax_coefs = fig.add_subplot(gs[1, :])

    # Show top varying terms (by cross-cluster variance)
    cluster_means = {}
    for cl in range(1, n_clusters + 1):
        cl_mask = cluster_col == cl
        cluster_means[f'Cluster {cl}'] = value_df[cl_mask].mean()

    means_df = pd.DataFrame(cluster_means)
    # Select terms with highest between-cluster variance
    between_var = means_df.var(axis=1)
    top_terms = between_var.nlargest(min(15, len(between_var))).index
    means_df = means_df.loc[top_terms]

    x = np.arange(len(top_terms))
    width = 0.8 / n_clusters
    for cl in range(n_clusters):
        offsets = x + (cl - n_clusters / 2 + 0.5) * width
        ax_coefs.bar(offsets, means_df.iloc[:, cl], width,
                     color=cluster_palette[cl], alpha=0.85,
                     label=f'Cluster {cl + 1}', edgecolor='white', linewidth=0.5)

    ax_coefs.set_xticks(x)
    short_labels = [t.split(':')[-1] if ':' in t else t for t in top_terms]
    ax_coefs.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
    ax_coefs.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_coefs.set_ylabel('Mean coefficient', fontsize=10)
    ax_coefs.set_title('c) Coefficient values by cluster (top distinguishing terms)',
                        fontsize=11, fontweight='bold', loc='left')
    ax_coefs.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax_coefs.spines['top'].set_visible(False)
    ax_coefs.spines['right'].set_visible(False)

    # Legends for panel a
    mod_handles = [mpatches.Patch(color=module_cmap[m], label=m) for m in unique_modules]
    cl_handles = [mpatches.Patch(color=cluster_palette[i], label=f'Cluster {i+1}')
                  for i in range(n_clusters)]

    fig.legend(handles=mod_handles, loc='lower left', bbox_to_anchor=(0.02, 0.92),
               ncol=len(unique_modules), fontsize=7, title='Module', title_fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(f'{output_path}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{output_path}.pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}.png/.pdf")


if __name__ == '__main__':
    plot_structural_subtypes(
        model_path='weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl',
        model_module='spice.precoded.workingmemory',
        n_actions=2,
        n_clusters=3,
        output_path='figures/prototype_structural_subtypes_dezfouli2019',
        title='Structural Subtypes — Dezfouli 2019 (Two-Armed Bandit)',
    )

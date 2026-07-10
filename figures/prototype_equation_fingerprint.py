"""
Prototype: Equation fingerprint heatmap.

Combined participants × ALL active SINDy terms across ALL modules.
Shows both sparsity and individual differences in one shot.
Rows = participants, columns = SINDy terms grouped by module.
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spice import SpiceEstimator, SpiceConfig, BaseModel


def load_coefficients(model_path, model_module):
    """Load a SPICE model and extract ensemble-averaged coefficients."""
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
        n_actions=2,
        n_participants=n_participants,
        sindy_library_polynomial_degree=2,
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(model_path)

    coefficients = estimator.get_sindy_coefficients(aggregate=True)
    candidate_terms = estimator.get_candidate_terms()
    modules = estimator.get_modules()

    return coefficients, candidate_terms, modules, n_participants


def build_fingerprint_df(coefficients, candidate_terms, modules, n_participants):
    """Build a DataFrame: rows=participants, columns=module:term."""
    data = {}
    col_modules = []  # track which module each column belongs to

    for module in modules:
        coefs = coefficients[module].detach().cpu().numpy()  # (P, X, T)
        terms = candidate_terms[module]
        coef_avg = coefs.mean(axis=1)  # (P, T) average over experiments

        for t_idx, term in enumerate(terms):
            col_name = f"{module}\n{term}"
            data[col_name] = coef_avg[:, t_idx]
            col_modules.append(module)

    df = pd.DataFrame(data, index=[f"P{i}" for i in range(n_participants)])
    return df, col_modules


def plot_equation_fingerprint(
    model_path,
    model_module,
    output_path='figures/prototype_equation_fingerprint',
    title='Equation Fingerprint',
    max_participants=80,
    filter_zero_cols=True,
):
    """Create combined equation fingerprint heatmap."""
    coefficients, candidate_terms, modules, n_participants = load_coefficients(
        model_path, model_module
    )

    df, col_modules = build_fingerprint_df(
        coefficients, candidate_terms, modules, n_participants
    )

    # Optionally filter columns that are zero for all participants
    if filter_zero_cols:
        nonzero_mask = (df.abs() > 1e-10).any(axis=0)
        df = df.loc[:, nonzero_mask]
        col_modules = [m for m, keep in zip(col_modules, nonzero_mask) if keep]

    # Limit participants for readability
    if len(df) > max_participants:
        df = df.iloc[:max_participants]

    n_terms = len(df.columns)
    n_parts = len(df)

    # Color limits
    vmax = max(abs(df.values.min()), abs(df.values.max()))
    if vmax < 1e-10:
        vmax = 1.0

    # Module color map for column annotations
    unique_modules = list(dict.fromkeys(col_modules))  # preserve order
    module_palette = sns.color_palette("husl", len(unique_modules))
    module_color_map = dict(zip(unique_modules, module_palette))

    # ── Plot ──
    fig_width = max(10, n_terms * 0.35)
    fig_height = max(6, n_parts * 0.15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Main heatmap
    im = ax.imshow(
        df.values, aspect='auto', cmap='RdBu_r',
        vmin=-vmax, vmax=vmax, interpolation='nearest',
    )

    # Module color bar on top
    for j, mod in enumerate(col_modules):
        ax.plot([j, j], [-0.5, -1.5], color=module_color_map[mod],
                linewidth=6, solid_capstyle='butt', clip_on=False)

    # Axes
    ax.set_xticks(range(n_terms))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=6, ha='center')
    if n_parts <= 50:
        ax.set_yticks(range(n_parts))
        ax.set_yticklabels(df.index, fontsize=6)
    else:
        ax.set_yticks([])

    ax.set_xlabel('SINDy Terms (grouped by module)', fontsize=10)
    ax.set_ylabel('Participants', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    # Module legend
    legend_handles = [mpatches.Patch(color=module_color_map[m], label=m)
                      for m in unique_modules]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              title='Module', fontsize=7, title_fontsize=8, framealpha=0.9)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.15)
    cbar.set_label('Coefficient value', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(f'{output_path}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{output_path}.pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}.png/.pdf")


if __name__ == '__main__':
    # Example: run on dezfouli2019
    plot_equation_fingerprint(
        model_path='weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl',
        model_module='spice.precoded.workingmemory',
        output_path='figures/prototype_equation_fingerprint_dezfouli2019',
        title='Equation Fingerprint — Dezfouli 2019 (Two-Armed Bandit)',
    )

"""
Equation stability: how reproducibly the final SPICE seeds identify the same equations.

Per study, all final SPICE seeds (weinhardt2026/utils/final_models.py); within a seed a participant carries a
term if the majority of its ensemble members do (analysis_coefficient_dendrogram.get_module_matrices).
Per module:
  - agreement (heatmap): per term, the mean over participants of max(f, 1 - f), with f the share of seeds in
    which the participant carries the term; 1 = all seeds agree, 0.5 = maximal disagreement
  - coefficients (right): per seed, mean ± SD across participants of the term's coefficient (0 where absent)
Terms that no seed gives to any participant (e.g. prior-excluded ones) are left out of the figure; they are
in the CSV (agreement 1, presence 0).

Written to weinhardt2026/studies/<study>/results/equation_stability/:
    equation_stability.pdf/.png
    equation_stability.csv   per (module, term): agreement, presence rate (over seeds and participants),
                             mean and SD over seeds of the per-seed mean coefficient

The per-seed matrices come from utils/final_models.seed_matrices (cached, shared with figure_mechanisms.py).

Usage:
    python weinhardt2026/figures/figure_equation_stability.py all
    python weinhardt2026/figures/figure_equation_stability.py dezfouli2019 kolff2025
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026', 'figures'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import figure_model_evaluation as evaluation
from weinhardt2026.utils.final_models import seed_matrices

MODULES_PER_ROW = 3
CMAP_AGREEMENT = plt.cm.RdYlGn
NORM_AGREEMENT = Normalize(vmin=0.5, vmax=1.0)


def stability_table(matrices):
    """Per (module, term): agreement over seeds and the per-seed coefficient summaries."""
    seeds = sorted(matrices)
    rows = []
    for module in matrices[seeds[0]]:
        terms = matrices[seeds[0]][module]['terms']
        presence = np.stack([matrices[seed][module]['presence'] for seed in seeds])                          # (S, P, T)
        values = np.stack([matrices[seed][module]['values'] * matrices[seed][module]['presence'] for seed in seeds])
        share = presence.mean(axis=0)                                                                         # (P, T)
        agreement = np.maximum(share, 1 - share).mean(axis=0)
        seed_means, seed_sds = values.mean(axis=1), values.std(axis=1)                                       # (S, T)
        for index, term in enumerate(terms):
            rows.append(dict(module=module, term=term, agreement=agreement[index],
                             presence_rate=presence[:, :, index].mean(),
                             coefficient_mean=seed_means[:, index].mean(), coefficient_sd_over_seeds=seed_means[:, index].std(),
                             seed_means=seed_means[:, index], seed_sds=seed_sds[:, index]))
    return pd.DataFrame(rows)


def plot_stability(table, study, output_path):
    table = table[table['presence_rate'] > 0]
    modules = list(dict.fromkeys(table['module']))
    rows = [modules[i:i + MODULES_PER_ROW] for i in range(0, len(modules), MODULES_PER_ROW)]
    heights = [0.28 * max((table['module'] == m).sum() for m in row) + 1.2 for row in rows]
    fig = plt.figure(figsize=(5.2 * MODULES_PER_ROW + 0.6, sum(heights) + 0.6))
    subfigures = np.atleast_1d(fig.subfigures(len(rows), 1, height_ratios=heights))
    seed_colors = plt.cm.tab10

    for subfigure, row in zip(subfigures, rows):
        # per module: [blank column for the term labels, agreement, coefficients]
        axes = subfigure.subplots(1, 3 * MODULES_PER_ROW, gridspec_kw={'width_ratios': [1.3, 0.45, 3.2] * MODULES_PER_ROW,
                                                                        'wspace': 0.05})
        for slot in range(MODULES_PER_ROW):
            axes[3 * slot].set_visible(False)
            ax_agreement, ax_coefficients = axes[3 * slot + 1], axes[3 * slot + 2]
            if slot >= len(row):
                ax_agreement.set_visible(False)
                ax_coefficients.set_visible(False)
                continue
            module_table = table[table['module'] == row[slot]]
            n_terms, y = len(module_table), np.arange(len(module_table))

            ax_agreement.imshow(module_table['agreement'].to_numpy()[:, None], aspect='auto', cmap=CMAP_AGREEMENT,
                                norm=NORM_AGREEMENT, interpolation='nearest')
            for i, value in enumerate(module_table['agreement']):
                ax_agreement.text(0, i, f'{value:.2f}', ha='center', va='center', fontsize=6,
                                  color='black' if value > 0.65 else 'white')
            ax_agreement.set_yticks(y)
            ax_agreement.set_yticklabels(module_table['term'], fontsize=6.5)
            ax_agreement.set_xticks([])
            ax_agreement.set_title(row[slot], fontsize=8, fontweight='bold', loc='left')

            seed_means = np.stack(module_table['seed_means'].to_numpy())   # (T, S)
            seed_sds = np.stack(module_table['seed_sds'].to_numpy())
            n_seeds = seed_means.shape[1]
            for seed in range(n_seeds):
                offset = (seed - (n_seeds - 1) / 2) * 0.6 / n_seeds
                color = seed_colors(seed % 10)
                ax_coefficients.errorbar(seed_means[:, seed], y + offset, xerr=seed_sds[:, seed], fmt='o', markersize=2.5,
                                         color=color, elinewidth=0.6, capsize=1, alpha=0.75, zorder=3)
            ax_coefficients.axvline(0, color='#404040', linestyle='--', linewidth=0.8, zorder=1)
            ax_coefficients.set_yticks(y)
            ax_coefficients.set_yticklabels([])
            for ax in (ax_agreement, ax_coefficients):
                ax.set_ylim(n_terms - 0.5, -0.5)
            ax_coefficients.tick_params(axis='x', labelsize=6.5)
            ax_coefficients.grid(axis='x', alpha=0.15)
            ax_coefficients.set_xlabel('coefficient (per seed: mean ± SD over participants)', fontsize=6.5)
            for side in ('top', 'right'):
                ax_coefficients.spines[side].set_visible(False)

    colorbar = fig.colorbar(ScalarMappable(norm=NORM_AGREEMENT, cmap=CMAP_AGREEMENT), cax=fig.add_axes([0.995, 0.3, 0.008, 0.4]))
    colorbar.set_label('agreement across seeds', fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(f'{evaluation.LABELS[study].replace(chr(10), " ")}: equation stability across seeds', fontsize=10, y=1.03)
    for extension in ('pdf', 'png'):
        fig.savefig(f'{output_path}.{extension}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {output_path}.pdf/.png')


def run_equation_stability(study):
    output_dir = evaluation._study_path(study, 'results', 'equation_stability')
    os.makedirs(output_dir, exist_ok=True)
    table = stability_table(seed_matrices(study))
    table.drop(columns=['seed_means', 'seed_sds']).to_csv(os.path.join(output_dir, 'equation_stability.csv'), index=False)
    plot_stability(table, study, os.path.join(output_dir, 'equation_stability'))
    active = table[table['presence_rate'] > 0]
    print(f'{study}: mean agreement {active["agreement"].mean():.3f} over {len(active)} terms ever selected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('studies', nargs='+', help=f"study names or 'all' ({', '.join(evaluation.ALL_STUDIES)})")
    args = parser.parse_args()
    for study in (evaluation.ALL_STUDIES if args.studies == ['all'] else args.studies):
        run_equation_stability(study)

"""Figure 5: Across-run stability analysis.

Row 1 — Structural agreement heatmap: mean agreement rate per term (averaged
across participants).  1.0 = all runs agree, 0.5 = maximum disagreement.

Row 2 — Coefficient distributions across runs: per-run mean ± SD (across
participants), showing both across-run reproducibility and within-run
participant heterogeneity.

Both rows share the same term y-axis per module column.  Prior-masked terms
(e.g. binary^2) are excluded.

Additional panel — Hyperparameter sensitivity heatmaps: trial likelihood and
mean number of parameters across pruning_threshold × pruning_test grid.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from spice import SpiceEstimator
from weinhardt2026.figures.panel_utils import save_panel


# ── Helpers ──────────────────────────────────────────────────────────


def _load_stability_runs(pkl_paths, spice_class, spice_config, n_actions,
                         polynomial_degree=2, model_kwargs=None):
    """Load all stability run checkpoints and extract coefficients + presence.

    Returns
    -------
    runs : list[dict]
        Each entry has keys per module:
        - 'presence': dict[module, ndarray (P, C)] binary
        - 'coefficients': dict[module, ndarray (P, C)] ensemble-aggregated
    candidate_terms : dict[module, list[str]]
    modules : list[str]
    prior_mask : dict[module, ndarray (C,)] — 1=allowed, 0=prior-excluded
    """
    runs = []
    candidate_terms = None
    modules = None
    prior_mask = None

    for path in sorted(pkl_paths):
        ckpt = torch.load(path, map_location='cpu')
        first_mod = next(iter(spice_config.library_setup))
        ensemble_size = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[0]
        n_participants = ckpt['model'][f'sindy_coefficients.{first_mod}'].shape[1]
        del ckpt

        estimator = SpiceEstimator(
            spice_class=spice_class,
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=n_participants,
            sindy_library_polynomial_degree=polynomial_degree,
            ensemble_size=ensemble_size,
            use_sindy=True,
            kwargs_spice_class=model_kwargs or {},
        )
        estimator.load_spice(path)

        if candidate_terms is None:
            candidate_terms = estimator.get_candidate_terms()
            modules = estimator.get_modules()
            prior_mask = {}
            for m in modules:
                pm = estimator.model.sindy_coefficients_prior_mask[m][0, 0, 0].detach().cpu().float().numpy()
                prior_mask[m] = pm

        coefficients_agg = estimator.get_sindy_coefficients(aggregate=True)
        presence = {}
        coefficients = {}
        for m in modules:
            p = estimator.model.sindy_coefficients_presence[m][0].detach().cpu().float()
            p = p.squeeze(1) if p.ndim == 3 else p
            presence[m] = p.numpy()

            c = coefficients_agg[m].detach().cpu()
            c = c.squeeze(1) if c.ndim == 3 else c
            coefficients[m] = c.numpy()

        runs.append({'presence': presence, 'coefficients': coefficients})

    return runs, candidate_terms, modules, prior_mask


def _short_term(term):
    """Abbreviate term names for display."""
    return (term
            .replace('value_reward_chosen', 'V_r(ch)')
            .replace('value_reward_not_chosen', 'V_r(unch)')
            .replace('value_choice_chosen', 'V_c(ch)')
            .replace('value_choice_not_chosen', 'V_c(unch)')
            .replace('reward', 'r')
            .replace('choice', 'c')
            .replace('[t]', '')
            .replace('[t-1]', '₋₁')
            .replace('[t-2]', '₋₂')
            .replace('[t-3]', '₋₃'))


def _short_module(module):
    """Abbreviate module names for subplot titles."""
    return (module
            .replace('value_reward_chosen', 'V_reward (chosen)')
            .replace('value_reward_not_chosen', 'V_reward (unchosen)')
            .replace('value_choice_chosen', 'V_choice (chosen)')
            .replace('value_choice_not_chosen', 'V_choice (unchosen)'))


# ── Combined figure ──────────────────────────────────────────────────


def _plot_figure5(runs, candidate_terms, modules, prior_mask, output_dir):
    """Combined figure: agreement heatmap (row 1) + coefficient distributions (row 2).

    Both rows share the same term y-axis per module column.
    Prior-masked terms are excluded.
    """
    n_runs = len(runs)
    n_modules = len(modules)

    # ── Pre-compute data per module, filtering out prior-masked terms ──
    module_data = {}
    for m in modules:
        all_terms = candidate_terms[m]
        pm = prior_mask[m]
        kept_idx = np.where(pm > 0)[0]
        terms = [all_terms[i] for i in kept_idx]

        # Agreement rate: (P, C_kept)
        presence_stack = np.stack([run['presence'][m][:, kept_idx] for run in runs], axis=0)
        frac_present = presence_stack.mean(axis=0)  # (P, C_kept)
        agreement = np.maximum(frac_present, 1 - frac_present)
        mean_agreement = agreement.mean(axis=0)  # (C_kept,)

        # Coefficient stats per run: all participants (absent terms contribute zero)
        run_means = np.full((n_runs, len(kept_idx)), np.nan)
        run_sds = np.full((n_runs, len(kept_idx)), np.nan)
        for r, run in enumerate(runs):
            coefs = run['coefficients'][m][:, kept_idx]  # already zero where absent
            for j in range(len(kept_idx)):
                run_means[r, j] = coefs[:, j].mean()
                run_sds[r, j] = coefs[:, j].std()

        module_data[m] = {
            'terms': terms,
            'mean_agreement': mean_agreement,
            'run_means': run_means,
            'run_sds': run_sds,
        }

    # ── Figure layout: n_modules groups, each [narrow heatmap | wide coef plot] ──
    max_terms = max(len(md['terms']) for md in module_data.values())
    fig_height = 0.3 * max_terms + 1.5

    # Width ratios: per module → [heatmap=1, coefs=5], with small gaps between modules
    width_ratios = []
    for i in range(n_modules):
        if i > 0:
            width_ratios.append(0.3)  # gap column
        width_ratios.extend([0.5, 5])
    n_cols = len(width_ratios)

    fig, all_axes = plt.subplots(1, n_cols,
                                 figsize=(2.8 * n_modules + 1.5, fig_height),
                                 gridspec_kw={'width_ratios': width_ratios})

    cmap_agreement = plt.cm.RdYlGn
    norm = Normalize(vmin=0.5, vmax=1.0)
    cmap_runs = plt.cm.tab10

    for col, m in enumerate(modules):
        md = module_data[m]
        terms = md['terms']
        n_terms = len(terms)
        y_positions = np.arange(n_terms)
        term_labels = [_short_term(t) for t in terms]

        # Column indices (accounting for gap columns)
        base = col * 3 if col > 0 else 0
        if col > 0:
            base = col * 2 + col  # each module = 2 cols + 1 gap before it
        idx_heatmap = base
        idx_coefs = base + 1

        # Hide gap columns
        if col > 0:
            gap_idx = base - 1
            all_axes[gap_idx].set_visible(False)

        ax_a = all_axes[idx_heatmap]
        ax_b = all_axes[idx_coefs]

        # ── Heatmap (agreement) ──
        agreement_2d = md['mean_agreement'].reshape(-1, 1)
        ax_a.imshow(agreement_2d, aspect='auto', cmap=cmap_agreement, norm=norm,
                    interpolation='nearest')
        for i, val in enumerate(md['mean_agreement']):
            ax_a.text(0, i, f'{val:.2f}', ha='center', va='center',
                      fontsize=6.5, color='black' if val > 0.65 else 'white')

        ax_a.set_yticks(y_positions)
        ax_a.set_yticklabels(term_labels, fontsize=7)
        ax_a.set_xticks([])
        ax_a.set_title(_short_module(m), fontsize=8.5, fontweight='bold')
        ax_a.set_ylim(n_terms - 0.5, -0.5)

        # ── Coefficient distributions ──
        run_means = md['run_means']
        run_sds = md['run_sds']

        jitter_width = 0.3
        for r in range(n_runs):
            jitter = (r - n_runs / 2 + 0.5) * (jitter_width * 2 / n_runs)
            color = cmap_runs(r / max(n_runs - 1, 1))
            ax_b.errorbar(
                run_means[r], y_positions + jitter, xerr=run_sds[r],
                fmt='o', markersize=3, color=color, ecolor=color,
                elinewidth=0.6, capsize=1, alpha=0.7, zorder=3,
            )

        ax_b.axvline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)
        ax_b.set_yticks(y_positions)
        ax_b.set_yticklabels([])
        ax_b.set_ylim(n_terms - 0.5, -0.5)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
        ax_b.grid(axis='x', alpha=0.15)
        ax_b.set_xlabel('Coefficient value', fontsize=7.5)

    # Colorbar for agreement
    sm = ScalarMappable(norm=norm, cmap=cmap_agreement)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=all_axes.tolist(), shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label('Agreement rate', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_panel(fig, output_dir, 'figure5_stability')


# ── HP scan heatmaps ─────────────────────────────────────────────────


def _plot_hpscan_heatmaps(hpscan_df, output_dir):
    """Three side-by-side heatmaps: trial likelihood, BIC, and mean n_params.

    Parameters
    ----------
    hpscan_df : pd.DataFrame
        Must contain columns: threshold, test, trial_likelihood, BIC,
        n_params_mean.
    output_dir : str
        Directory to save figure panels.
    """
    df = hpscan_df.copy()

    thresholds = sorted(df['threshold'].unique())
    tests = sorted(df['test'].unique())
    n_thresh = len(thresholds)
    n_test = len(tests)

    # Build 2D grids
    lik_grid = np.full((n_thresh, n_test), np.nan)
    bic_grid = np.full((n_thresh, n_test), np.nan)
    params_grid = np.full((n_thresh, n_test), np.nan)

    for i, thr in enumerate(thresholds):
        for j, tst in enumerate(tests):
            row = df[(df['threshold'] == thr) & (df['test'] == tst)]
            if len(row) == 1:
                lik_grid[i, j] = row['trial_likelihood'].values[0]
                params_grid[i, j] = row['n_params_mean'].values[0]
                if 'BIC' in row.columns:
                    bic_grid[i, j] = row['BIC'].values[0]

    # Axis labels
    thresh_labels = [f'{t:g}' for t in thresholds]
    test_labels = [f'{t:g}' for t in tests]

    has_bic = not np.all(np.isnan(bic_grid))
    n_panels = 3 if has_bic else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 2.5))

    # Custom colormap: black (bad) → dark green → bright green (good)
    from matplotlib.colors import LinearSegmentedColormap
    cmap_good = LinearSegmentedColormap.from_list(
        'black_green', ['#000000', "#146D14", '#1a7a1a', '#3cb43c',
                         '#6fd36f', '#a8e6a8', '#d0f5d0'], N=256)

    def _annotated_heatmap(ax, grid, title, fmt, higher_is_better=True):
        # For "lower is better" metrics, invert the grid for coloring
        # so that low values map to light green (good) and high to black (bad)
        display_grid = grid if higher_is_better else -grid
        vals = display_grid[~np.isnan(display_grid)].ravel()
        sorted_vals = np.sort(vals)
        # Set scale from second-worst to best so the single worst
        # outlier clips to black without compressing the rest
        vmin_data = sorted_vals[1] if len(sorted_vals) > 2 else sorted_vals[0]
        vmax = sorted_vals[-1]
        data_range = vmax - vmin_data
        # Pad below so the second-worst still maps to dark green, not black
        vmin = vmin_data - data_range * 0.2
        vmax = vmax + data_range * 0.05
        norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(display_grid, aspect='auto', cmap=cmap_good, norm=norm,
                       interpolation='nearest')
        # Annotate with original values; text color based on mapped brightness
        for i in range(n_thresh):
            for j in range(n_test):
                val = grid[i, j]
                if not np.isnan(val):
                    mapped = norm(display_grid[i, j])
                    text_color = 'white' if mapped < 0.5 else 'black'
                    ax.text(j, i, fmt.format(val), ha='center', va='center',
                            fontsize=8, color=text_color)
        ax.set_xticks(range(n_test))
        ax.set_xticklabels(test_labels, fontsize=8)
        ax.set_yticks(range(n_thresh))
        ax.set_yticklabels(thresh_labels, fontsize=8)
        ax.set_xlabel('Pruning test', fontsize=9)
        ax.set_ylabel('Pruning threshold', fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.ax.tick_params(labelsize=7)

    # ── Trial likelihood (higher = better) ──
    _annotated_heatmap(axes[0], lik_grid,
                       'Trial likelihood (hold-out)', '{:.3f}',
                       higher_is_better=True)

    # ── BIC (lower = better) ──
    if has_bic:
        _annotated_heatmap(axes[1], bic_grid,
                           'BIC (hold-out)', '{:.0f}',
                           higher_is_better=False)

    # ── N params (lower = better → sparser) ──
    ax_par_idx = 2 if has_bic else 1
    _annotated_heatmap(axes[ax_par_idx], params_grid,
                       'Mean parameters per participant', '{:.1f}',
                       higher_is_better=False)

    fig.tight_layout()
    save_panel(fig, output_dir, 'figure5_hpscan')


# ── Main entry point ─────────────────────────────────────────────────


def plot_figure5(stability_pkl_paths, spice_class, spice_config, n_actions,
                 output_dir, polynomial_degree=2, model_kwargs=None,
                 hpscan_csv=None):
    """Generate Figure 5: across-run stability analysis.

    Parameters
    ----------
    stability_pkl_paths : list[str]
        Paths to the stability run .pkl files.
    spice_class : type
        Model class (e.g. workingmemory.SpiceModel).
    spice_config : SpiceConfig
        Model configuration.
    n_actions : int
        Number of actions.
    output_dir : str
        Directory to save figure panels.
    polynomial_degree : int
        SINDy library polynomial degree.
    model_kwargs : dict, optional
        Extra kwargs for the model constructor (e.g. {"reward_binary": True}).
    hpscan_csv : str, optional
        Path to HP scan results CSV (from analysis_sparsity_hpscan).
        If provided, generates the HP sensitivity heatmap panel.
    """
    print("Loading stability runs...")
    runs, candidate_terms, modules, prior_mask = _load_stability_runs(
        stability_pkl_paths, spice_class, spice_config, n_actions,
        polynomial_degree=polynomial_degree, model_kwargs=model_kwargs,
    )
    print(f"  Loaded {len(runs)} runs, {len(modules)} modules")

    print("Plotting figure 5 stability panel...")
    _plot_figure5(runs, candidate_terms, modules, prior_mask, output_dir)

    if hpscan_csv and os.path.isfile(hpscan_csv):
        print("Plotting figure 5 HP scan panel...")
        hpscan_df = pd.read_csv(hpscan_csv)
        _plot_hpscan_heatmaps(hpscan_df, output_dir)
    elif hpscan_csv:
        print(f"  HP scan CSV not found: {hpscan_csv}, skipping hpscan panel.")

    print(f"Figure 5 complete → {output_dir}/")


# ── Standalone execution ─────────────────────────────────────────────

if __name__ == '__main__':
    from spice.precoded import workingmemory

    params_dir = 'weinhardt2026/studies/dezfouli2019/params_array'
    pkl_paths = sorted(glob(os.path.join(params_dir, 'spice_dezfouli2019_stability_[0-9].pkl')))

    if not pkl_paths:
        print(f"No stability runs found in {params_dir}")
    else:
        plot_figure5(
            stability_pkl_paths=pkl_paths,
            spice_class=workingmemory.SpiceModel,
            spice_config=workingmemory.CONFIG,
            n_actions=2,
            output_dir='weinhardt2026/studies/dezfouli2019/figures/figure5',
            model_kwargs={'reward_binary': True},
            hpscan_csv='weinhardt2026/studies/dezfouli2019/results/hpscan_results.csv',
        )

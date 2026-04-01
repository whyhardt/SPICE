"""
SINDy coefficient distribution analysis pipeline.

Analyzes coefficient distributions across ensemble members,
participants, and experiments from a trained SPICE model.

Two analysis blocks:
  1. Ensemble Consistency – How consistent are coefficients across
     ensemble members for each (participant, experiment) pair?
  2. Coefficient Distributions – How do ensemble-averaged coefficients
     distribute across the participant x experiment grid?

Usage examples:

  # From CLI:
  python analysis_coefficients_distributions.py \\
      --model weinhardt2026/params/dezfouli2019/spice_dezfouli2019.pkl \\
      --data weinhardt2026/data/dezfouli2019/dezfouli2019.csv \\
      --model-module spice.precoded.workingmemory_rewardbinary \\
      --output weinhardt2026/analysis/output/distributions

  # From Python:
  from weinhardt2026.analysis.analysis_coefficients_distributions import (
      analysis_coefficients_distributions,
  )
  analysis_coefficients_distributions(spice_model=estimator, output_dir="output/")
"""

import argparse
import importlib
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spice import SpiceEstimator, csv_to_dataset, BaseRNN, SpiceConfig, SpiceDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_significance(p):
    if pd.isna(p):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


SIG_COLORS = {"***": "#FF0000", "**": "#FFA500", "*": "#FFD700", "ns": "#999999", "na": "#CCCCCC"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def prepare_spice(
    path_model: str,
    dataset: SpiceDataset = None,
    model_module: str = None,
    model_class: BaseRNN = None,
    model_config: SpiceConfig = None,
    n_reward_features: int = None,
) -> SpiceEstimator:
    """Load a trained SPICE model from a checkpoint file.

    Accepts either *model_module* (dotted import path) or
    *model_class* + *model_config*.  Infers n_actions and
    n_participants from the dataset when provided.
    """
    if model_module is not None and model_class is None and model_config is None:
        mod = importlib.import_module(model_module)
        rnn_class = mod.SpiceModel
        spice_config = mod.CONFIG
    elif model_module is None and model_class is not None and model_config is not None:
        rnn_class = model_class
        spice_config = model_config
    else:
        raise ValueError("Provide either (model_module) OR (model_class AND model_config).")

    # Peek at checkpoint to infer dimensions
    ckpt = torch.load(path_model, map_location="cpu")
    first_module = next(iter(spice_config.library_setup))
    coef_shape = ckpt["model"][f"sindy_coefficients.{first_module}"].shape
    ensemble_size = coef_shape[0]
    n_participants_ckpt = coef_shape[1]

    if dataset is not None:
        n_actions = dataset.ys.shape[-1]
        n_participants = int(dataset.xs[..., -1].unique().shape[0])
    else:
        n_participants = n_participants_ckpt
        # infer n_actions from output layer
        for key in ckpt["model"]:
            if "output" in key and "weight" in key:
                n_actions = ckpt["model"][key].shape[0]
                break
        else:
            n_actions = 2  # fallback
    del ckpt

    estimator = SpiceEstimator(
        spice_class=rnn_class,
        spice_config=spice_config,
        n_actions=n_actions,
        n_participants=n_participants,
        n_reward_features=n_reward_features,
        sindy_library_polynomial_degree=2,
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(path_model)
    estimator.model.eval()
    return estimator


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_coefficient_data(
    estimator: SpiceEstimator,
) -> Tuple[
    Dict[str, torch.Tensor],   # raw coefficients:  {module: (E, P, X, T)}
    Dict[str, torch.Tensor],   # aggregated coeffs:  {module: (P, X, T)}
    Dict[str, torch.Tensor],   # presence masks:     {module: (E, P, X, T)}
    Dict[str, List[str]],      # candidate terms:    {module: [str, ...]}
    List[str],                  # module names
]:
    """Extract all coefficient data from a loaded SpiceEstimator."""
    modules = estimator.get_modules()
    candidate_terms = estimator.get_candidate_terms()
    raw = estimator.get_sindy_coefficients(aggregate=False)
    agg = estimator.get_sindy_coefficients(aggregate=True)

    presence = {}
    for m in modules:
        presence[m] = estimator.model.sindy_coefficients_presence[m].detach().cpu()

    # Move tensors to CPU
    for m in modules:
        raw[m] = raw[m].detach().cpu()
        agg[m] = agg[m].detach().cpu()

    return raw, agg, presence, candidate_terms, modules


# ---------------------------------------------------------------------------
# Analysis 1: Ensemble Consistency
# ---------------------------------------------------------------------------

def compute_ensemble_consistency(
    raw_coefficients: Dict[str, torch.Tensor],
    presence_masks: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
) -> pd.DataFrame:
    """Compute per-term ensemble consistency statistics.

    Returns a DataFrame with one row per (module, term) containing
    presence agreement, presence rate, coefficient of variation, and
    ensemble spread metrics.
    """
    rows = []
    for module in modules:
        coefs = raw_coefficients[module].numpy()   # (E, P, X, T)
        pres = presence_masks[module].numpy()       # (E, P, X, T)
        terms = candidate_terms[module]
        E, P, X, T = coefs.shape

        for t_idx, term in enumerate(terms):
            c = coefs[:, :, :, t_idx]   # (E, P, X)
            p = pres[:, :, :, t_idx]    # (E, P, X)

            # Per (participant, experiment) metrics
            frac_active = p.mean(axis=0)        # (P, X)
            agreement = np.maximum(frac_active, 1 - frac_active)  # 1.0 = unanimous

            # CV and std across ensemble (only active members)
            cvs = []
            stds = []
            means = []
            for pi in range(P):
                for xi in range(X):
                    active = p[:, pi, xi].astype(bool)
                    n_active = active.sum()
                    if n_active >= 2:
                        vals = c[active, pi, xi]
                        m_val = np.mean(vals)
                        s_val = np.std(vals, ddof=1)
                        stds.append(s_val)
                        means.append(m_val)
                        if abs(m_val) > 1e-10:
                            cvs.append(s_val / abs(m_val))
                    elif n_active == 1:
                        stds.append(0.0)
                        means.append(float(c[active, pi, xi][0]))

            rows.append({
                "module": module,
                "term": term,
                "term_index": t_idx,
                "presence_agreement_mean": float(np.mean(agreement)),
                "presence_agreement_std": float(np.std(agreement)),
                "presence_rate_mean": float(np.mean(frac_active)),
                "presence_rate_std": float(np.std(frac_active)),
                "cv_mean": float(np.mean(cvs)) if cvs else np.nan,
                "cv_median": float(np.median(cvs)) if cvs else np.nan,
                "ensemble_std_mean": float(np.mean(stds)) if stds else np.nan,
                "ensemble_mean_mean": float(np.mean(means)) if means else np.nan,
            })

    return pd.DataFrame(rows)


def plot_ensemble_spread(
    raw_coefficients: Dict[str, torch.Tensor],
    presence_masks: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
    output_dir: str,
    max_participants: int = 30,
) -> None:
    """Plot ensemble member values as strip markers per participant.

    One figure per module with one subplot row per term.
    Falls back to a CV heatmap if P > max_participants.
    """
    for module in modules:
        coefs = raw_coefficients[module].numpy()   # (E, P, X, T)
        pres = presence_masks[module].numpy()
        terms = candidate_terms[module]
        E, P, X, n_terms = coefs.shape

        if P > max_participants:
            # Fall back to CV heatmap for this module
            _plot_ensemble_cv_heatmap_single(
                coefs, pres, terms, module, output_dir,
            )
            continue

        fig, axs = plt.subplots(
            nrows=n_terms, ncols=1,
            figsize=(max(8, P * 0.4), 2.5 * n_terms),
            sharex=True, squeeze=False,
        )

        for t_idx, term in enumerate(terms):
            ax = axs[t_idx, 0]
            for pi in range(P):
                # Average across experiments for each ensemble member
                vals = coefs[:, pi, :, t_idx].mean(axis=1)   # (E,)
                active = pres[:, pi, :, t_idx].any(axis=1)   # (E,)
                # Active ensemble members
                ax.plot(
                    np.full(active.sum(), pi), vals[active],
                    "x", color="tab:blue", markersize=4, alpha=0.7,
                )
                # Pruned ensemble members (if any active exist)
                n_pruned = (~active).sum()
                if n_pruned > 0 and active.sum() > 0:
                    ax.plot(
                        np.full(n_pruned, pi), vals[~active],
                        "x", color="tab:red", markersize=3, alpha=0.3,
                    )
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.set_ylabel(term, fontsize=8, rotation=0, ha="right")
            ax.set_yticks([])

        axs[0, 0].set_title(f"Ensemble spread: {module}")
        axs[-1, 0].set_xlabel("Participant")
        axs[-1, 0].set_xticks(range(P))
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"ensemble_spread_{module}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()


def _plot_ensemble_cv_heatmap_single(
    coefs: np.ndarray,
    pres: np.ndarray,
    terms: List[str],
    module: str,
    output_dir: str,
) -> None:
    """CV heatmap for a single module (helper for fallback)."""
    E, P, X, n_terms = coefs.shape
    cv_matrix = np.full((n_terms, P), np.nan)

    for t_idx in range(n_terms):
        for pi in range(P):
            # Average across experiments
            active = pres[:, pi, :, t_idx].any(axis=1)
            n_active = active.sum()
            if n_active >= 2:
                vals = coefs[active, pi, :, t_idx].mean(axis=1)
                m = np.mean(vals)
                s = np.std(vals, ddof=1)
                if abs(m) > 1e-10:
                    cv_matrix[t_idx, pi] = s / abs(m)

    fig, ax = plt.subplots(figsize=(max(8, P * 0.3), max(4, n_terms * 0.4)))
    sns.heatmap(
        cv_matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=2.0,
        xticklabels=range(P), yticklabels=terms,
        cbar_kws={"label": "Coefficient of Variation"},
    )
    ax.set_xlabel("Participant")
    ax.set_title(f"Ensemble CV: {module}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"ensemble_cv_heatmap_{module}.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()


def plot_ensemble_cv_heatmap(
    raw_coefficients: Dict[str, torch.Tensor],
    presence_masks: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
    output_dir: str,
) -> None:
    """CV heatmap per module: rows=terms, cols=participants, color=CV."""
    for module in modules:
        coefs = raw_coefficients[module].numpy()
        pres = presence_masks[module].numpy()
        terms = candidate_terms[module]
        _plot_ensemble_cv_heatmap_single(coefs, pres, terms, module, output_dir)


# ---------------------------------------------------------------------------
# Analysis 2: Coefficient Distributions (participants x experiments)
# ---------------------------------------------------------------------------

def compute_distribution_statistics(
    aggregated_coefficients: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
) -> pd.DataFrame:
    """Compute summary statistics for ensemble-averaged coefficients.

    Flattens across the (P, X) grid and computes mean, std, median,
    IQR, presence rate, etc. for each (module, term).
    """
    rows = []
    for module in modules:
        coefs = aggregated_coefficients[module].numpy()  # (P, X, T)
        terms = candidate_terms[module]
        P, X, T = coefs.shape

        for t_idx, term in enumerate(terms):
            flat = coefs[:, :, t_idx].ravel()  # (P*X,)
            nonzero = flat[np.abs(flat) > 1e-10]
            q25, q75 = np.percentile(flat, [25, 75]) if flat.size else (np.nan, np.nan)

            rows.append({
                "module": module,
                "term": term,
                "term_index": t_idx,
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "median": float(np.median(flat)),
                "iqr": float(q75 - q25),
                "min_val": float(np.min(flat)),
                "max_val": float(np.max(flat)),
                "presence_rate": float(len(nonzero) / len(flat)) if len(flat) > 0 else 0.0,
                "n_nonzero": int(len(nonzero)),
                "n_total": int(len(flat)),
            })

    return pd.DataFrame(rows)


def plot_coefficient_violins(
    aggregated_coefficients: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
    output_dir: str,
) -> None:
    """Violin plots of coefficient values per term, one figure per module.

    Only nonzero values are shown in the violins.  Individual data
    points are overlaid as jittered dots.
    """
    for module in modules:
        coefs = aggregated_coefficients[module].numpy()  # (P, X, T)
        terms = candidate_terms[module]
        P, X, T = coefs.shape

        # Build a DataFrame for seaborn
        records = []
        for t_idx, term in enumerate(terms):
            flat = coefs[:, :, t_idx].ravel()
            nonzero = flat[np.abs(flat) > 1e-10]
            for v in nonzero:
                records.append({"term": term, "value": float(v)})

        if not records:
            continue

        df_plot = pd.DataFrame(records)
        n_terms = len(terms)
        fig, ax = plt.subplots(figsize=(max(8, n_terms * 1.2), 5))

        # Only plot terms that have nonzero values
        order = [t for t in terms if t in df_plot["term"].values]
        if not order:
            plt.close()
            continue

        sns.violinplot(
            data=df_plot, x="term", y="value", order=order,
            ax=ax, inner=None, color="lightblue", alpha=0.6, cut=0,
        )
        sns.stripplot(
            data=df_plot, x="term", y="value", order=order,
            ax=ax, color="tab:blue", size=2, alpha=0.4, jitter=True,
        )
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(f"Coefficient distributions: {module}")
        ax.set_xlabel("Library term")
        ax.set_ylabel("Coefficient value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"distribution_violins_{module}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()


def plot_presence_rate_bar(
    stats_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Grouped bar chart of presence rates by module and term."""
    modules = stats_df["module"].unique()
    n_modules = len(modules)

    fig, axes = plt.subplots(
        1, n_modules, figsize=(max(6, 3 * n_modules), 5),
        sharey=True, squeeze=False,
    )

    for i, module in enumerate(modules):
        ax = axes[0, i]
        sub = stats_df[stats_df["module"] == module].copy()
        bars = ax.bar(range(len(sub)), sub["presence_rate"], color="steelblue", edgecolor="black")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["term"], rotation=45, ha="right", fontsize=8)
        ax.set_title(module)
        ax.set_ylim(0, 1.05)

        # Annotate with rates
        for bar, rate in zip(bars, sub["presence_rate"]):
            if rate > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=7,
                )

    axes[0, 0].set_ylabel("Presence rate")
    fig.suptitle("Coefficient presence rates", y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "presence_rates.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()


def plot_experiment_comparison(
    aggregated_coefficients: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
    output_dir: str,
) -> Optional[pd.DataFrame]:
    """Per-experiment box plots and repeated-measures tests per term.

    Runs Friedman test (X >= 3) or Wilcoxon signed-rank (X == 2).
    Returns None if X == 1.
    """
    # Check experiment dimension
    first_module = modules[0]
    X = aggregated_coefficients[first_module].shape[1]
    if X <= 1:
        return None

    test_rows = []

    for module in modules:
        coefs = aggregated_coefficients[module].numpy()  # (P, X, T)
        terms = candidate_terms[module]
        P, X_mod, T = coefs.shape
        n_terms = len(terms)

        ncols = min(4, n_terms)
        nrows = math.ceil(n_terms / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

        for t_idx, term in enumerate(terms):
            ax = axes[t_idx // ncols, t_idx % ncols]

            # Build per-experiment data
            data_by_exp = []
            for xi in range(X_mod):
                data_by_exp.append(coefs[:, xi, t_idx])

            # Box plot
            ax.boxplot(data_by_exp, labels=[f"Exp {xi}" for xi in range(X_mod)])
            ax.set_title(term, fontsize=9)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

            # Statistical test
            # Only include participants with at least one nonzero coefficient
            valid_mask = np.any(np.abs(coefs[:, :, t_idx]) > 1e-10, axis=1)
            n_valid = valid_mask.sum()

            stat, p_val, test_name = np.nan, np.nan, "none"
            if n_valid >= 5:
                valid_data = [coefs[valid_mask, xi, t_idx] for xi in range(X_mod)]
                if X_mod >= 3:
                    try:
                        stat, p_val = friedmanchisquare(*valid_data)
                        test_name = "friedman"
                    except Exception:
                        pass
                elif X_mod == 2:
                    try:
                        stat, p_val = wilcoxon(valid_data[0], valid_data[1])
                        test_name = "wilcoxon"
                    except Exception:
                        pass

            sig = get_significance(p_val)
            ax.set_xlabel(f"{test_name} p={p_val:.3f} {sig}" if not np.isnan(p_val) else "", fontsize=7)

            test_rows.append({
                "module": module,
                "term": term,
                "test_name": test_name,
                "statistic": float(stat) if not np.isnan(stat) else np.nan,
                "p_value": float(p_val) if not np.isnan(p_val) else np.nan,
                "significance": sig,
                "n_valid_participants": int(n_valid),
            })

        # Hide unused subplots
        for idx in range(n_terms, nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")

        fig.suptitle(f"Experiment comparison: {module}", y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"experiment_comparison_{module}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    return pd.DataFrame(test_rows)


def plot_sparsity_heatmap(
    aggregated_coefficients: Dict[str, torch.Tensor],
    candidate_terms: Dict[str, List[str]],
    modules: List[str],
    output_dir: str,
    cluster: bool = True,
) -> None:
    """Participants x terms heatmap of coefficient values.

    Averages across experiments.  Optionally applies hierarchical
    clustering via seaborn clustermap.
    """
    for module in modules:
        coefs = aggregated_coefficients[module].numpy()  # (P, X, T)
        terms = candidate_terms[module]
        P, X, T = coefs.shape

        # Average across experiments
        coef_avg = coefs.mean(axis=1)  # (P, T)

        df_heat = pd.DataFrame(
            coef_avg,
            index=[f"P{pi}" for pi in range(P)],
            columns=terms,
        )

        vmax = max(abs(df_heat.values.min()), abs(df_heat.values.max()))
        if vmax < 1e-10:
            vmax = 1.0

        if cluster and P >= 3 and T >= 2:
            g = sns.clustermap(
                df_heat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                figsize=(max(8, T * 0.6), max(6, P * 0.3)),
                cbar_kws={"label": "Coefficient value"},
                xticklabels=True, yticklabels=(P <= 50),
            )
            g.fig.suptitle(f"Sparsity heatmap (clustered): {module}", y=1.02)
            g.savefig(
                os.path.join(output_dir, f"sparsity_heatmap_{module}.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(max(8, T * 0.6), max(6, P * 0.3)))
            sns.heatmap(
                df_heat, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "Coefficient value"},
                xticklabels=True, yticklabels=(P <= 50),
            )
            ax.set_title(f"Sparsity heatmap: {module}")
            ax.set_xlabel("Library term")
            ax.set_ylabel("Participant")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"sparsity_heatmap_{module}.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def analysis_coefficients_distributions(
    # Accept pre-loaded model
    spice_model: SpiceEstimator = None,

    # Or path-based loading
    model_path: str = None,
    model_module: str = None,
    model_class: BaseRNN = None,
    model_config: SpiceConfig = None,
    dataset: SpiceDataset = None,
    n_reward_features: int = None,

    # Output
    output_dir: str = "analysis_coefficient_distributions",

    # Options
    max_participants_strip: int = 30,
    cluster_heatmap: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Run the full coefficient distribution analysis.

    Parameters
    ----------
    spice_model : SpiceEstimator, optional
        Pre-loaded estimator.  If None, loads from *model_path*.
    model_path : str, optional
        Path to the SPICE checkpoint (.pkl).
    model_module : str, optional
        Dotted module path (e.g. ``spice.precoded.workingmemory_rewardbinary``).
    model_class : BaseRNN, optional
        RNN class (alternative to *model_module*).
    model_config : SpiceConfig, optional
        SPICE config (required when *model_class* is given).
    dataset : SpiceDataset, optional
        Used to infer n_actions / n_participants when loading from path.
    output_dir : str
        Directory to save plots and CSVs.
    max_participants_strip : int
        Max participants for strip plots before switching to heatmap.
    cluster_heatmap : bool
        Use hierarchical clustering in sparsity heatmaps.
    verbose : bool
        Print progress information.

    Returns
    -------
    ensemble_df : pd.DataFrame
        Ensemble consistency statistics.
    stats_df : pd.DataFrame
        Distribution summary statistics.
    experiment_df : pd.DataFrame or None
        Experiment comparison test results (None if only 1 experiment).
    """

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    if spice_model is None:
        if model_path is None:
            raise ValueError("Provide either spice_model or model_path.")
        if verbose:
            print("Loading SPICE model...")
        spice_model = prepare_spice(
            path_model=model_path,
            dataset=dataset,
            model_module=model_module,
            model_class=model_class,
            model_config=model_config,
            n_reward_features=n_reward_features,
        )

    # ------------------------------------------------------------------
    # 2. Extract data
    # ------------------------------------------------------------------
    if verbose:
        print("Extracting coefficient data...")
    raw, agg, presence, candidate_terms, modules = extract_coefficient_data(spice_model)

    E = raw[modules[0]].shape[0]
    P = raw[modules[0]].shape[1]
    X = raw[modules[0]].shape[2]
    n_terms_total = sum(len(candidate_terms[m]) for m in modules)
    if verbose:
        print(f"  Ensemble={E}, Participants={P}, Experiments={X}, "
              f"Modules={len(modules)}, Total terms={n_terms_total}")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Ensemble Consistency (Analysis 1)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS 1: Ensemble Consistency")
        print("=" * 60)

    ensemble_df = compute_ensemble_consistency(raw, presence, candidate_terms, modules)
    ensemble_df.to_csv(os.path.join(output_dir, "ensemble_consistency.csv"), index=False)

    if verbose:
        print(f"  Mean presence agreement: {ensemble_df['presence_agreement_mean'].mean():.3f}")
        print(f"  Mean presence rate:      {ensemble_df['presence_rate_mean'].mean():.3f}")
        cv_valid = ensemble_df["cv_mean"].dropna()
        if len(cv_valid):
            print(f"  Mean CV:                 {cv_valid.mean():.3f}")

    plot_ensemble_spread(raw, presence, candidate_terms, modules, output_dir, max_participants_strip)
    if verbose:
        print("  Ensemble spread plots saved.")

    plot_ensemble_cv_heatmap(raw, presence, candidate_terms, modules, output_dir)
    if verbose:
        print("  Ensemble CV heatmaps saved.")

    # ------------------------------------------------------------------
    # 4. Coefficient Distributions (Analysis 2)
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS 2: Coefficient Distributions")
        print("=" * 60)

    stats_df = compute_distribution_statistics(agg, candidate_terms, modules)
    stats_df.to_csv(os.path.join(output_dir, "distribution_statistics.csv"), index=False)

    if verbose:
        print(f"  Terms with >50% presence: "
              f"{(stats_df['presence_rate'] > 0.5).sum()} / {len(stats_df)}")
        print(f"  Terms with 0% presence:   "
              f"{(stats_df['presence_rate'] == 0).sum()} / {len(stats_df)}")

    plot_coefficient_violins(agg, candidate_terms, modules, output_dir)
    if verbose:
        print("  Violin plots saved.")

    plot_presence_rate_bar(stats_df, output_dir)
    if verbose:
        print("  Presence rate bar chart saved.")

    experiment_df = plot_experiment_comparison(agg, candidate_terms, modules, output_dir)
    if experiment_df is not None:
        experiment_df.to_csv(os.path.join(output_dir, "experiment_comparison.csv"), index=False)
        sig_count = (experiment_df["significance"].isin(["*", "**", "***"])).sum()
        if verbose:
            print(f"  Experiment comparison: {sig_count} significant terms out of {len(experiment_df)}.")
    elif verbose:
        print("  Experiment comparison skipped (X=1).")

    plot_sparsity_heatmap(agg, candidate_terms, modules, output_dir, cluster=cluster_heatmap)
    if verbose:
        print("  Sparsity heatmaps saved.")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nAll results saved to: {output_dir}")

    return ensemble_df, stats_df, experiment_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SINDy coefficient distribution analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                    help="Path to the trained SPICE model (.pkl)")
    p.add_argument("--model-module", default="spice.precoded.workingmemory_rewardbinary",
                    help="Name of the SPICE model module "
                         "(default: spice.precoded.workingmemory_rewardbinary)")
    p.add_argument("--data", default=None,
                    help="Path to the experiment data CSV (for n_actions inference)")
    p.add_argument("--output", default=None,
                    help="Output directory (default: auto-generated next to model)")
    p.add_argument("--max-participants", type=int, default=30,
                    help="Max participants for strip plots (default: 30)")
    p.add_argument("--no-cluster", action="store_true",
                    help="Disable hierarchical clustering in sparsity heatmaps")
    args = p.parse_args()

    output_dir = args.output or os.path.join(
        os.path.dirname(args.model), "analysis_coefficient_distributions",
    )

    dataset = None
    if args.data is not None:
        dataset = csv_to_dataset(file=args.data)

    analysis_coefficients_distributions(
        model_path=args.model,
        model_module=args.model_module,
        dataset=dataset,
        output_dir=output_dir,
        max_participants_strip=args.max_participants,
        cluster_heatmap=not args.no_cluster,
        verbose=True,
    )

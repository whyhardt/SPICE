"""Batch NMF-per-module coefficient-compression hyperparameter search across
all dezfouli2019 params_array checkpoints (the sindy_threshold_pruning x
sindy_ensemble_pruning training HP scan).

For each checkpoint in params_array/, grid-searches (alpha_W, alpha_H) at a
fixed K_per_module and records train/test NLL/BIC via
analysis_coefficient_compression.run_nmf_per_module_hyperparameter_search,
selecting per-model by train ΔBIC/trial (never test -- see that module's
docstring). Results from all models are concatenated into one CSV/plot set
so the best compression setting can be compared across the training HP scan.

Usage:
    python weinhardt2026/studies/dezfouli2019/analysis_compression_hpscan_array.py
"""

import os
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice.precoded.workingmemory import SpiceModel, CONFIG
from weinhardt2026.analysis.analysis_coefficient_compression import (
    run_nmf_per_module_hyperparameter_search,
)


def parse_hp(path):
    basename = os.path.basename(path).replace(".pkl", "")
    match = re.search(r"_(\d+\.?\d*)_(\d+\.?\d*)$", basename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def load_estimator(path, n_actions=2):
    ckpt = torch.load(path, map_location="cpu")
    first_mod = next(iter(CONFIG.library_setup))
    ensemble_size = ckpt["model"][f"sindy_coefficients.{first_mod}"].shape[0]
    n_participants = ckpt["model"][f"sindy_coefficients.{first_mod}"].shape[1]
    del ckpt

    estimator = SpiceEstimator(
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        kwargs_spice_class={"reward_binary": True},
        sindy_library_polynomial_degree=2,
        ensemble_size=ensemble_size,
        use_sindy=True,
    )
    estimator.load_spice(path)
    estimator.model.eval()
    return estimator


def plot_across_models(df: pd.DataFrame, output_dir: str, edge_alpha_w, edge_alpha_h):
    """Per-model best (alpha_W, alpha_H) by train ΔBIC/trial, plus the full grid."""
    best_rows = df.loc[df.groupby("model_path")["train_dbic_per_trial"].idxmax()]
    best_rows = best_rows.sort_values(["train_hp1", "train_hp2"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    labels = [f"{r.train_hp1:g}/{r.train_hp2:g}" for r in best_rows.itertuples()]
    ax.bar(range(len(best_rows)), best_rows["train_dbic_per_trial"], color="tab:blue")
    ax.set_xticks(range(len(best_rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("model (threshold_pruning / ensemble_pruning)")
    ax.set_ylabel("best train ΔBIC/trial")
    ax.set_title("Best compression fit per model")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    sc = ax.scatter(best_rows["alpha_W"], best_rows["alpha_H"], c=best_rows["train_dbic_per_trial"],
                     cmap="viridis", s=80, edgecolor="k")
    fig.colorbar(sc, ax=ax, label="best train ΔBIC/trial")
    ax.set_xlabel("alpha_W")
    ax.set_ylabel("alpha_H")
    ax.set_title("Selected (alpha_W, alpha_H) per model")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(best_rows["train_dbic_per_trial"], best_rows["test_trial_lik"], s=60, alpha=0.7)
    ax.set_xlabel("train ΔBIC/trial (selection)")
    ax.set_ylabel("test trial likelihood (confirmation)")
    ax.set_title("Train/test calibration across models")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hyperparameter_search_array_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Heatmap of train ΔBIC/trial averaged across models, over the (alpha_W, alpha_H) grid
    pivot = df.groupby(["alpha_W", "alpha_H"])["train_dbic_per_trial"].mean().unstack("alpha_H")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("alpha_H")
    ax.set_ylabel("alpha_W")
    ax.set_title("Mean train ΔBIC/trial across all models")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean train ΔBIC/trial")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hyperparameter_search_array_grid_mean.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return best_rows


def main():
    study = "dezfouli2019"
    data_path = f"weinhardt2026/studies/{study}/data/{study}.csv"
    params_dir = f"weinhardt2026/studies/{study}/params_array"
    results_dir = f"weinhardt2026/studies/{study}/results/compression"
    os.makedirs(results_dir, exist_ok=True)

    K_PER_MODULE = 8
    # Widened from the original [0.001..0.005] x [0.0..0.0005] grid: the first pass showed
    # 14/16 models best at alpha_W=0.005 (upper edge) and 16/16 at alpha_H=0.0005 (upper edge).
    ALPHA_W = [0.003, 0.005, 0.008, 0.012, 0.02]
    ALPHA_H = [0.0005, 0.001, 0.002, 0.004, 0.008]

    dataset = csv_to_dataset(file=data_path)
    dataset.normalize_rewards()
    dataset_train, dataset_test = split_data_along_blockdim(dataset, (3, 6, 9))

    pkl_paths = sorted(glob(os.path.join(params_dir, f"spice_{study}_*.pkl")))
    pkl_paths = [p for p in pkl_paths if "stage1" not in p]
    print(f"Found {len(pkl_paths)} checkpoints in {params_dir}")

    all_rows = []
    for i, path in enumerate(pkl_paths):
        p1, p2 = parse_hp(path)
        print(f"\n=== [{i + 1}/{len(pkl_paths)}] {os.path.basename(path)} "
              f"(sindy_threshold_pruning={p1}, sindy_ensemble_pruning={p2}) ===")
        estimator = load_estimator(path)
        df = run_nmf_per_module_hyperparameter_search(
            estimator, dataset_train, dataset_test,
            k_per_module_values=[K_PER_MODULE],
            alpha_w_values=ALPHA_W,
            alpha_h_values=ALPHA_H,
            center=False, verbose=True,
        )
        df.insert(0, "model_path", os.path.basename(path))
        df.insert(1, "train_hp1", p1)
        df.insert(2, "train_hp2", p2)
        all_rows.append(df)

        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(os.path.join(results_dir, "hyperparameter_search_array.csv"), index=False)

        best_idx = df["train_dbic_per_trial"].idxmax()
        best = df.loc[best_idx]
        print(f"  -> best this model: alpha_W={best['alpha_W']:g}, alpha_H={best['alpha_H']:g}, "
              f"train_dbic/trial={best['train_dbic_per_trial']:.4f}, test_lik={best['test_trial_lik']:.4f}")

        del estimator

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(os.path.join(results_dir, "hyperparameter_search_array.csv"), index=False)
    print(f"\nSaved combined results to {os.path.join(results_dir, 'hyperparameter_search_array.csv')}")

    # --- Edge check: does the per-model best alpha_W / alpha_H sit at grid edges? ---
    best_rows = combined.loc[combined.groupby("model_path")["train_dbic_per_trial"].idxmax()]
    aw_min, aw_max = min(ALPHA_W), max(ALPHA_W)
    ah_min, ah_max = min(ALPHA_H), max(ALPHA_H)
    n_aw_edge = ((best_rows["alpha_W"] == aw_min) | (best_rows["alpha_W"] == aw_max)).sum()
    n_ah_edge = ((best_rows["alpha_H"] == ah_min) | (best_rows["alpha_H"] == ah_max)).sum()
    print(f"\nEdge check: {n_aw_edge}/{len(best_rows)} models have best alpha_W at grid edge "
          f"({aw_min:g} or {aw_max:g}); {n_ah_edge}/{len(best_rows)} models have best alpha_H at grid edge "
          f"({ah_min:g} or {ah_max:g}).")

    plot_across_models(combined, results_dir, (aw_min, aw_max), (ah_min, ah_max))
    print(f"Saved summary plots to {results_dir}")

    return combined, best_rows


if __name__ == "__main__":
    main()

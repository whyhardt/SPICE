"""
Coefficient compression analysis pipeline.

A fitted SPICE model gives each participant its own free coefficient for
every active SINDy term, e.g. ~47 free coefficients per participant in
dezfouli2019 (spread across 4 modules, 15-21 terms each). Many of these
terms co-vary strongly across the population -- they are not independent
axes of individual difference, just different symptoms of the same
underlying trait, and the raw equations are too large to read.

After comparing dense SVD, sparse dictionary learning, sign-split NMF, and
a hand-classified term-family basis (each jointly across all modules and/or
per-module -- see `spice.resources.sindy_compression` for all of them), the
adopted method is: **sign-split NMF fit independently within each module's
own term block** (`fit_nmf_signsplit_per_module`). It gives the best
combination of the four things that mattered:
  - predictive cost: competitive with the best unconstrained methods,
  - usage sparsity: participants genuinely lack some mechanisms (nonzero
    loadings only on a handful), not just small-everywhere,
  - mechanism compactness: a handful of terms per mechanism, not ~20,
  - localization: every mechanism confined to one module by construction,
  - no hand-classification: which terms combine is learned from data.

This pipeline:
  1. Grid-searches (K_per_module, alpha_W, alpha_H) and evaluates NLL/BIC on
     both train and test via the same scoring machinery as
     `analysis_model_evaluation`.
  2. Selects the best setting **by training-set ΔBIC/trial only** -- test
     performance is reported purely as an out-of-sample confirmation of
     that choice, never used to pick it. Selecting by test performance
     (which every exploratory pass in this file's history did before this
     version) is leakage: it overfits the hyperparameters to the test set
     and reports an optimistic number.
  3. Saves plots/CSVs/symbolic equations (population baseline, mechanisms,
     example participant equations) for the selected setting.

Note on the ensemble: writing a reconstruction back requires collapsing
the model's ensemble (one coefficient set instead of E slightly different
ones, normally averaged in probability space). This costs real
performance by itself, so the correct baseline for judging the *marginal*
cost of compression is a full-rank reconstruction evaluated the same way,
not the original ensemble-based evaluation -- the gap between those two
can be large and study-dependent (small on dezfouli2019, large on
eckstein2026) and should be checked before trusting compressed-vs-original
comparisons.

Usage examples:

  # From Python:
  from weinhardt2026.analysis.analysis_coefficient_compression import (
      analysis_coefficient_compression,
  )
  search_df, loadings_df, compressed_model = analysis_coefficient_compression(
      spice_model=estimator,
      dataset_train=dataset_train,
      dataset_test=dataset_test,
      output_dir="results/compression",
  )
  compressed_model.print_mechanisms()
  compressed_model.print_participant(participant_id=0)

  # From CLI:
  python analysis_coefficient_compression.py \\
      --model weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl \\
      --model-module spice.precoded.workingmemory \\
      --data weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv \\
      --output weinhardt2026/studies/dezfouli2019/results/compression
"""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spice import SpiceEstimator, csv_to_dataset, SpiceDataset, CompressedSpiceModel
from spice.resources.sindy_compression import extract_joint_coefficient_matrix, fit_nmf_signsplit_per_module
from weinhardt2026.analysis.analysis_coefficients_distributions import prepare_spice
from weinhardt2026.analysis.analysis_model_evaluation import (
    get_choice_probs, log_likelihood, get_participant_experiment_groups, grouped_information_criteria,
)


# ---------------------------------------------------------------------------
# 1. Evaluate a compressed model on held-out data
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_compressed_model(
    spice_model: SpiceEstimator,
    dataset: SpiceDataset,
    compressed_model: CompressedSpiceModel,
) -> Dict[str, float]:
    """Score predictive performance of a coefficient reconstruction on ``dataset``.

    Uses grouped (per participant x experiment) BIC/AIC, matching
    `analysis_model_evaluation.grouped_information_criteria`, not a single
    dataset-pooled BIC -- pooling scales the BIC penalty with total trial
    count while using one global parameter count, which unfairly penalizes
    per-subject-parameter models (like this one) relative to shared-weight
    models purely as dataset size grows, independent of actual fit quality.
    Each (participant, experiment) group's own trial count and own active-
    mechanism count are used, then averaged across groups.
    """
    with compressed_model.apply(spice_model):
        spice_model.model.init_state(batch_size=dataset.xs.shape[0])
        preds, _ = spice_model(dataset.xs.to(spice_model.device))
        probs = get_choice_probs(preds.mean(dim=0)).detach().cpu()

    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])
    targets_eval = dataset.ys.clone()
    targets_eval[~valid_mask] = float("nan")
    considered_trials_participant = valid_mask.sum(dim=1).float()
    considered_trials = considered_trials_participant.sum()

    unique_pairs, group_index = get_participant_experiment_groups(dataset)
    n_groups = unique_pairs.shape[0]

    # Per (participant, experiment) active-mechanism count -- compressed_model.loadings
    # is row-ordered (participant, experiment) matching extract_joint_coefficient_matrix
    # (row = participant * X + experiment), same convention count_sindy_coefficients() uses.
    active_per_pair = (np.abs(compressed_model.loadings) > 1e-6).sum(axis=1)  # (P*X,)
    active_per_pair = torch.tensor(active_per_pair, dtype=torch.float32).reshape(compressed_model.P, compressed_model.X)
    n_parameters_per_group = active_per_pair[unique_pairs[:, 0], unique_pairs[:, 1]]

    nll = -log_likelihood(data=targets_eval, probs=probs)
    nll_per_session = nll.sum(dim=-1).nansum(dim=1)[..., 0]
    trial_lik = torch.exp(-nll_per_session.sum() / considered_trials).item()

    info = grouped_information_criteria(
        nll_per_session=nll_per_session,
        n_trials_per_session=considered_trials_participant,
        group_index=group_index,
        n_groups=n_groups,
        n_parameters_per_group=n_parameters_per_group,
        n_actions_baseline=dataset.n_actions,
    )

    return dict(
        nll=info["nll_total"], aic=info["aic_mean"], bic=info["bic_mean"],
        trial_lik=trial_lik,
        dbic_per_trial=info["delta_bic_per_trial_mean"],
    )


# ---------------------------------------------------------------------------
# 2. Hyperparameter search
# ---------------------------------------------------------------------------

def run_nmf_per_module_hyperparameter_search(
    spice_model: SpiceEstimator,
    dataset_train: SpiceDataset,
    dataset_test: SpiceDataset,
    k_per_module_values: List[int],
    alpha_w_values: List[float],
    alpha_h_values: List[float],
    center: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Grid-search (K_per_module, alpha_W, alpha_H) for `fit_nmf_signsplit_per_module`.

    ``center`` selects ``MODEL = U @ H`` (``center=False``, the default --
    every participant's model is fully described by their loadings against
    the shared dictionary, nothing held outside that count) vs
    ``MODEL = mean + U @ H`` (``center=True``).

    Reports both train and test metrics for every setting (for transparency
    and diagnostic plots), but `analysis_coefficient_compression` below only
    ever *selects* using the train columns -- treat the test columns here as
    read-only context, not a menu to pick from.
    """
    C, col_labels, col_slices, P, X = extract_joint_coefficient_matrix(spice_model)

    rows = []
    for K_per_module in k_per_module_values:
        for alpha_W in alpha_w_values:
            for alpha_H in alpha_h_values:
                mean_vec, components, loadings, mechanism_names = fit_nmf_signsplit_per_module(
                    C, col_labels, col_slices, K_per_module=K_per_module, alpha_W=alpha_W, alpha_H=alpha_H, center=center,
                )
                compressed = CompressedSpiceModel(mean_vec, components, loadings, col_labels, col_slices, P, X, mechanism_names=mechanism_names)
                C_hat = compressed.reconstructed_coefficients()
                recon_r2 = 1 - ((C - C_hat) ** 2).sum() / ((C - C.mean(axis=0)) ** 2).sum()
                n_active_mean = float((np.abs(loadings) > 1e-6).sum(axis=1).mean())

                res_train = evaluate_compressed_model(spice_model, dataset_train, compressed)
                res_test = evaluate_compressed_model(spice_model, dataset_test, compressed)

                row = dict(
                    K_per_module=K_per_module, alpha_W=alpha_W, alpha_H=alpha_H, K=compressed.K,
                    coef_recon_r2=float(recon_r2), sparsity=compressed.sparsity(), n_active_mean=n_active_mean,
                    mean_terms_per_mechanism=float(compressed.n_terms_above_threshold(0.15).mean()),
                    mean_modules_per_mechanism=float(compressed.n_modules_touched(0.15).mean()),
                )
                row.update({f"train_{k}": v for k, v in res_train.items()})
                row.update({f"test_{k}": v for k, v in res_test.items()})
                rows.append(row)
                if verbose:
                    print(f"  K/mod={K_per_module} aW={alpha_W:<7g} aH={alpha_H:<7g} K={compressed.K:>3d} "
                          f"active={n_active_mean:5.2f} terms/mech={row['mean_terms_per_mechanism']:4.2f} "
                          f"train_dbic/trial={res_train['dbic_per_trial']:.4f} test_dbic/trial={res_test['dbic_per_trial']:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

def plot_hyperparameter_search(df: pd.DataFrame, output_dir: str, chosen_idx: int) -> None:
    """Two diagnostics: train-vs-test calibration, and the sparsity/fit frontier.

    Both mark the chosen (train-selected) setting, and the left panel in
    particular is the check that train-based selection was reasonable --
    if train and test ΔBIC/trial were uncorrelated across the grid, that
    would be a red flag that the search is overfitting train.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    chosen = df.loc[chosen_idx]

    ax = axes[0]
    ax.scatter(df["train_dbic_per_trial"], df["test_dbic_per_trial"], alpha=0.5, s=20, color="tab:blue")
    ax.scatter([chosen["train_dbic_per_trial"]], [chosen["test_dbic_per_trial"]], color="tab:red", s=80,
               marker="*", zorder=5, label="chosen (by train)")
    lims = [
        min(df["train_dbic_per_trial"].min(), df["test_dbic_per_trial"].min()),
        max(df["train_dbic_per_trial"].max(), df["test_dbic_per_trial"].max()),
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlabel("train ΔBIC/trial (selection criterion)")
    ax.set_ylabel("test ΔBIC/trial (confirmation only)")
    ax.set_title("Train/test calibration across the grid")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(df["n_active_mean"], df["train_dbic_per_trial"], c=df["mean_terms_per_mechanism"],
                     cmap="viridis", alpha=0.7, s=25)
    ax.scatter([chosen["n_active_mean"]], [chosen["train_dbic_per_trial"]], color="tab:red", s=80,
               marker="*", zorder=5, label="chosen (by train)")
    fig.colorbar(sc, ax=ax, label="terms/mechanism")
    ax.set_xlabel("active mechanisms / participant")
    ax.set_ylabel("train ΔBIC/trial")
    ax.set_title("Sparsity / fit frontier")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hyperparameter_search.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main orchestrator
# ---------------------------------------------------------------------------

def analysis_coefficient_compression(
    # Accept pre-loaded model
    spice_model: SpiceEstimator = None,

    # Or path-based loading
    model_path: str = None,
    model_module: str = None,
    dataset: SpiceDataset = None,
    n_reward_features: int = None,

    # Data to evaluate on
    dataset_train: SpiceDataset = None,
    dataset_test: SpiceDataset = None,

    # Output
    output_dir: str = "analysis_coefficient_compression",

    # Hyperparameter search grid
    k_per_module_values: Optional[List[int]] = None,
    alpha_w_values: Optional[List[float]] = None,
    alpha_h_values: Optional[List[float]] = None,

    # Or skip the search and use an explicit setting
    chosen_K_per_module: Optional[int] = None,
    chosen_alpha_W: Optional[float] = None,
    chosen_alpha_H: Optional[float] = None,

    # MODEL = U @ H (default) vs MODEL = mean + U @ H
    center: bool = False,

    # Optional concise, hand-assigned names (same order as the auto-generated
    # "module: mechanism N" list -- module-by-module, index 0..K_per_module-1
    # within each module) to use everywhere instead of the generic ones.
    mechanism_names_override: Optional[List[str]] = None,

    mechanism_threshold_ratio: float = 0.15,
    n_example_participants: int = 3,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, CompressedSpiceModel]:
    """Run the full coefficient-compression analysis (see module docstring).

    Parameters
    ----------
    spice_model : SpiceEstimator, optional
        Pre-loaded estimator. If None, loads from *model_path*.
    dataset_train, dataset_test : SpiceDataset
        Datasets to score held-out predictive performance on.
    output_dir : str
        Directory to save plots and CSVs.
    k_per_module_values, alpha_w_values, alpha_h_values : list, optional
        Grid to search. Defaults to a moderate grid around what worked on
        dezfouli2019 -- widen it for a more thorough search, narrow it for
        large datasets where each fit is slow (e.g. eckstein2026).
    chosen_K_per_module, chosen_alpha_W, chosen_alpha_H : optional
        Skip the grid search and use this exact setting instead (e.g. once
        you've already run the search once and know what you want).
    center : bool
        ``MODEL = U @ H`` (default, ``False``) -- every participant's model
        is fully described by their loadings against the shared dictionary,
        nothing held outside that count. ``True`` uses
        ``MODEL = mean + U @ H`` instead (see `fit_nmf_signsplit_per_module`).
    mechanism_threshold_ratio : float
        Relative-magnitude threshold for sparsifying mechanism definitions
        (see `spice.resources.sindy_compression.format_mechanism_terms`).
    n_example_participants : int
        Number of participants to print compressed equations for.

    Returns
    -------
    search_df : pd.DataFrame
        One row per grid setting, with train *and* test NLL/BIC -- selection
        used only the train columns; see module docstring.
    loadings_df : pd.DataFrame
        Per-(participant, experiment) loadings for the chosen setting.
    compressed_model : CompressedSpiceModel
        Reusable object exposing `.print_population()`, `.print_mechanisms()`,
        and `.print_participant(participant_id)` for the chosen setting.
    """
    if spice_model is None:
        if model_path is None:
            raise ValueError("Provide either spice_model or model_path.")
        if verbose:
            print("Loading SPICE model...")
        spice_model = prepare_spice(
            path_model=model_path, dataset=dataset, model_module=model_module,
            n_reward_features=n_reward_features,
        )
    if dataset_train is None or dataset_test is None:
        raise ValueError("Provide dataset_train and dataset_test to evaluate held-out performance.")

    os.makedirs(output_dir, exist_ok=True)

    explicit_setting = chosen_K_per_module is not None and chosen_alpha_W is not None and chosen_alpha_H is not None
    if explicit_setting:
        if verbose:
            print(f"Using explicit setting K_per_module={chosen_K_per_module}, "
                  f"alpha_W={chosen_alpha_W}, alpha_H={chosen_alpha_H} (skipping search).")
        search_df = run_nmf_per_module_hyperparameter_search(
            spice_model, dataset_train, dataset_test,
            k_per_module_values=[chosen_K_per_module], alpha_w_values=[chosen_alpha_W], alpha_h_values=[chosen_alpha_H],
            center=center, verbose=verbose,
        )
        chosen_idx = search_df.index[0]
    else:
        if k_per_module_values is None:
            k_per_module_values = [4, 6, 8]
        if alpha_w_values is None:
            alpha_w_values = [0.001, 0.002, 0.003, 0.005]
        if alpha_h_values is None:
            alpha_h_values = [0.0, 0.0002, 0.0005]

        if verbose:
            print("\n" + "=" * 60)
            print("NMF-per-module hyperparameter search "
                  f"({len(k_per_module_values)}x{len(alpha_w_values)}x{len(alpha_h_values)} grid, "
                  "selecting by TRAIN ΔBIC/trial)")
            print("=" * 60)

        search_df = run_nmf_per_module_hyperparameter_search(
            spice_model, dataset_train, dataset_test,
            k_per_module_values=k_per_module_values, alpha_w_values=alpha_w_values, alpha_h_values=alpha_h_values,
            center=center, verbose=verbose,
        )
        # Selection criterion is TRAIN performance only -- test is confirmation, not a menu.
        chosen_idx = search_df["train_dbic_per_trial"].idxmax()

    search_df.to_csv(os.path.join(output_dir, "hyperparameter_search.csv"), index=False)
    chosen = search_df.loc[chosen_idx]
    chosen_K_per_module = int(chosen["K_per_module"])
    chosen_alpha_W = float(chosen["alpha_W"])
    chosen_alpha_H = float(chosen["alpha_H"])

    if verbose:
        print(f"\nChosen (by train ΔBIC/trial={chosen['train_dbic_per_trial']:.4f}): "
              f"K_per_module={chosen_K_per_module}, alpha_W={chosen_alpha_W}, alpha_H={chosen_alpha_H}")
        print(f"  Out-of-sample confirmation: test ΔBIC/trial={chosen['test_dbic_per_trial']:.4f} "
              f"(never used for selection)")
        print(f"  {chosen['n_active_mean']:.1f}/{int(chosen['K'])} mechanisms active/participant on average, "
              f"{chosen['mean_terms_per_mechanism']:.1f} terms/mechanism, "
              f"{chosen['mean_modules_per_mechanism']:.1f} modules/mechanism")

    if len(search_df) > 1:
        plot_hyperparameter_search(search_df, output_dir, chosen_idx)
        if verbose:
            print("  Hyperparameter search plot saved.")

    # ------------------------------------------------------------------
    # Refit the chosen setting standalone (search rows share nothing to
    # reuse across settings, unlike SVD's truncation trick) and build the
    # CompressedSpiceModel + symbolic outputs.
    # ------------------------------------------------------------------
    C, col_labels, col_slices, P, X = extract_joint_coefficient_matrix(spice_model)
    mean_vec, components, loadings, mechanism_names = fit_nmf_signsplit_per_module(
        C, col_labels, col_slices, K_per_module=chosen_K_per_module, alpha_W=chosen_alpha_W, alpha_H=chosen_alpha_H,
        center=center,
    )
    if mechanism_names_override is not None:
        if len(mechanism_names_override) != len(mechanism_names):
            raise ValueError(
                f"mechanism_names_override has {len(mechanism_names_override)} names but the fit produced "
                f"{len(mechanism_names)} mechanisms -- names are order-dependent on (K_per_module, alpha_W, "
                f"alpha_H, center); a stale override list from a previous setting won't line up."
            )
        mechanism_names = list(mechanism_names_override)
    compressed_model = CompressedSpiceModel(mean_vec, components, loadings, col_labels, col_slices, P, X, mechanism_names=mechanism_names)

    pop_eq = pd.DataFrame(
        [{"module": m, "term": t, "mean_coefficient": v} for (m, t), v in zip(col_labels, mean_vec)]
    )
    pop_eq.to_csv(os.path.join(output_dir, "population_equation.csv"), index=False)
    with open(os.path.join(output_dir, "population_equations.txt"), "w") as f:
        f.write(compressed_model.population_string())

    mechanisms_text = compressed_model.mechanisms_string(threshold_ratio=mechanism_threshold_ratio)
    with open(os.path.join(output_dir, "mechanisms.txt"), "w") as f:
        f.write(mechanisms_text)

    p_idx, x_idx = np.divmod(np.arange(P * X), X)
    loadings_df = pd.DataFrame(loadings, columns=[m for m in mechanism_names])
    loadings_df.insert(0, "experiment_index", x_idx)
    loadings_df.insert(0, "participant_index", p_idx)
    loadings_df.to_csv(os.path.join(output_dir, "participant_loadings.csv"), index=False)

    example_lines = []
    for pid in range(min(n_example_participants, P)):
        example_lines.append(f"--- participant {pid} ---")
        example_lines.append(compressed_model.participant_string(pid))
        example_lines.append("")
    with open(os.path.join(output_dir, "compressed_equations_example.txt"), "w") as f:
        f.write("\n".join(example_lines))

    if verbose:
        print(f"\nPopulation baseline (shared across all participants):\n")
        print(compressed_model.population_string())
        print(f"\nMechanisms (K={compressed_model.K}, threshold_ratio={mechanism_threshold_ratio}):\n")
        print(mechanisms_text)
        print(f"\nExample compressed participant equations:\n")
        print("\n".join(example_lines))
        print(f"\nAll results saved to: {output_dir}")

    return search_df, loadings_df, compressed_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Coefficient compression analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, help="Path to the trained SPICE model (.pkl)")
    p.add_argument("--model-module", default="spice.precoded.workingmemory",
                    help="Dotted module path of the SPICE model (default: spice.precoded.workingmemory)")
    p.add_argument("--data", required=True, help="Path to the experiment data CSV")
    p.add_argument("--test-blocks", type=int, nargs="*", default=None,
                    help="Block IDs held out as test data (default: none, evaluates on full data as both splits)")
    p.add_argument("--output", default=None, help="Output directory (default: auto-generated next to model)")
    p.add_argument("--k-per-module", type=int, nargs="*", default=None, help="K_per_module grid to search")
    p.add_argument("--alpha-w", type=float, nargs="*", default=None, help="alpha_W grid to search")
    p.add_argument("--alpha-h", type=float, nargs="*", default=None, help="alpha_H grid to search")
    args = p.parse_args()

    output_dir = args.output or os.path.join(
        os.path.dirname(args.model), "analysis_coefficient_compression",
    )

    from spice.utils.convert_dataset import split_data_along_blockdim

    dataset = csv_to_dataset(file=args.data)
    dataset.normalize_rewards()
    if args.test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks=args.test_blocks)
    else:
        dataset_train = dataset_test = dataset

    analysis_coefficient_compression(
        model_path=args.model,
        model_module=args.model_module,
        dataset=dataset,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        output_dir=output_dir,
        k_per_module_values=args.k_per_module,
        alpha_w_values=args.alpha_w,
        alpha_h_values=args.alpha_h,
        verbose=True,
    )

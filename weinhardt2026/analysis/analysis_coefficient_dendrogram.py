"""Hierarchical clustering of SINDy coefficients — one dendrogram per module.

Each SPICE submodule holds one equation per participant. Across the population,
its candidate-term coefficients co-vary: terms that move together (or in exact
opposition, e.g. ``c_reward = -c_value`` = a learning rate) form a *tie* that
can be read as a single cognitive mechanism rather than as free parameters.

This analysis clusters the terms *within* each module by how their
per-participant coefficients covary across the population, and draws one
dendrogram per module so those candidate mechanisms become visible.

Two views are available via ``mode``:

- ``'coefficients'`` — correlation distance between coefficient values.
  ``metric='abs'`` uses ``1 - |r|`` so sign-flipped ties (``c_a = -c_b``) join
  early; ``metric='signed'`` uses ``1 - r`` so only same-sign terms join.
- ``'presence'`` — Jaccard distance between binary presence masks, i.e. which
  terms are switched on/off together by pruning (the "gate pattern" view).

Usage:

    python -m weinhardt2026.analysis.analysis_coefficient_dendrogram \
        --model weinhardt2026/studies/dezfouli2019/params_array/spice_dezfouli2019_0.05_0.5.pkl \
        --module spice.precoded.workingmemory \
        --n-participants 101 \
        --output-dir weinhardt2026/studies/dezfouli2019/results
"""

import argparse
import importlib
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spice import BaseModel, SpiceConfig, SpiceEstimator
from weinhardt2026.utils.checkpoints import load_estimator


# ---------------------------------------------------------------------------
# Coefficient extraction
# ---------------------------------------------------------------------------

def get_module_matrices(
    estimator: SpiceEstimator,
    experiment: int = 0,
) -> Dict[str, dict]:
    """Per module: coefficient matrix (P, T), presence matrix (P, T), term names."""
    coefficients = estimator.get_sindy_coefficients(aggregate=True)  # (P, X, T)
    candidate_terms = estimator.get_candidate_terms()
    presence_all = estimator.model.sindy_coefficients_presence  # (E, P, X, T)

    matrices = {}
    for module in estimator.get_modules():
        values = coefficients[module].detach().cpu().numpy()[:, experiment]  # (P, T)
        presence = presence_all[module].float().mean(dim=0).detach().cpu().numpy()[:, experiment]
        matrices[module] = {
            "values": values,
            "presence": (presence > 0.5).astype(float),
            "terms": list(candidate_terms[module]),
        }
    return matrices


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def coefficient_distance(
    matrix: np.ndarray,
    presence: Optional[np.ndarray] = None,
    metric: str = "abs",
    min_overlap: int = 30,
) -> np.ndarray:
    """Correlation distance between the columns (terms) of a (P, T) matrix.

    A pruned term is stored as an exact zero, so correlating full columns mixes
    the parametric question ("do these magnitudes move together?") with the
    structural one ("are these terms absent in the same participants?"). When
    *presence* is given, each pair is therefore correlated only over the
    participants in which **both** terms are present. Pairs with fewer than
    *min_overlap* such participants cannot be judged and are set to r = 0, i.e.
    maximally distant, rather than silently counted as unrelated evidence.

    The returned correlation keeps its sign, so a tie of the form c_a = -c_b
    stays readable regardless of which *metric* the tree is built on.
    """
    n_terms = matrix.shape[1]
    if presence is None:
        correlation = np.nan_to_num(np.corrcoef(matrix.T), nan=0.0)
        untestable = np.zeros((n_terms, n_terms), dtype=bool)
    else:
        correlation = np.eye(n_terms)
        untestable = np.zeros((n_terms, n_terms), dtype=bool)
        for i in range(n_terms):
            for j in range(i + 1, n_terms):
                both = (presence[:, i] > 0) & (presence[:, j] > 0)
                a, b = matrix[both, i], matrix[both, j]
                if both.sum() >= min_overlap and a.std() > 1e-12 and b.std() > 1e-12:
                    value = np.corrcoef(a, b)[0, 1]
                else:
                    value, untestable[i, j] = 0.0, True
                    untestable[j, i] = True
                correlation[i, j] = correlation[j, i] = np.nan_to_num(value, nan=0.0)

    if metric == "abs":
        distance = 1.0 - np.abs(correlation)
    elif metric == "signed":
        distance = 1.0 - correlation
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'abs' or 'signed'.")
    np.fill_diagonal(distance, 0.0)
    return np.clip(distance, 0.0, 2.0), correlation, untestable


def presence_distance(matrix: np.ndarray) -> np.ndarray:
    """Jaccard distance between the binary presence columns of a (P, T) matrix."""
    n_terms = matrix.shape[1]
    distance = np.zeros((n_terms, n_terms))
    similarity = np.eye(n_terms)
    for i in range(n_terms):
        for j in range(i + 1, n_terms):
            intersection = np.sum((matrix[:, i] > 0) & (matrix[:, j] > 0))
            union = np.sum((matrix[:, i] > 0) | (matrix[:, j] > 0))
            jaccard = intersection / union if union > 0 else 0.0
            similarity[i, j] = similarity[j, i] = jaccard
            distance[i, j] = distance[j, i] = 1.0 - jaccard
    return distance, similarity, np.zeros_like(distance, dtype=bool)


def term_distance(
    matrix: np.ndarray,
    presence: Optional[np.ndarray] = None,
    mode: str = "coefficients",
    metric: str = "abs",
    min_overlap: int = 30,
):
    """Dispatch to the distance that matches *mode*."""
    if mode == "coefficients":
        return coefficient_distance(matrix, presence=presence, metric=metric,
                                    min_overlap=min_overlap)
    if mode == "presence":
        return presence_distance(matrix)
    raise ValueError(f"Unknown mode '{mode}'. Use 'coefficients' or 'presence'.")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def cluster_module(
    values: np.ndarray,
    presence: np.ndarray,
    terms: List[str],
    mode: str = "coefficients",
    metric: str = "abs",
    linkage_method: str = "average",
    distance_threshold: float = 0.5,
    min_active_fraction: float = 0.05,
    min_overlap: int = 30,
    pairwise_complete: bool = True,
) -> dict:
    """Cluster one module's candidate terms. Returns linkage, labels and table."""
    active_fraction = presence.mean(axis=0)
    variance = values.std(axis=0)

    # Terms that are pruned for (almost) everyone carry no across-participant
    # structure — they are part of the discovered equation form, not of it.
    keep = (active_fraction >= min_active_fraction) & (variance > 1e-12)
    kept_terms = [t for t, k in zip(terms, keep) if k]

    dropped = pd.DataFrame({
        "term": [t for t, k in zip(terms, keep) if not k],
        "active_fraction": active_fraction[~keep],
        "mean": values[:, ~keep].mean(axis=0),
    })

    if len(kept_terms) < 2:
        return {"terms": kept_terms, "keep": keep, "linkage": None, "clusters": None,
                "similarity": None, "untestable": None, "table": pd.DataFrame(),
                "dropped": dropped}

    matrix = (values if mode == "coefficients" else presence)[:, keep]
    distance, similarity, untestable = term_distance(
        matrix,
        presence=presence[:, keep] if pairwise_complete else None,
        mode=mode, metric=metric, min_overlap=min_overlap,
    )

    linkage_matrix = linkage(squareform(distance, checks=False), method=linkage_method)
    clusters = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

    table = pd.DataFrame({
        "term": kept_terms,
        "cluster": clusters,
        "active_fraction": active_fraction[keep],
        "mean": values[:, keep].mean(axis=0),
        "sd": values[:, keep].std(axis=0),
    }).sort_values(["cluster", "term"]).reset_index(drop=True)

    return {"terms": kept_terms, "keep": keep, "linkage": linkage_matrix,
            "clusters": clusters, "similarity": similarity, "distance": distance,
            "untestable": untestable, "table": table, "dropped": dropped}


def bootstrap_stability(
    values: np.ndarray,
    presence: np.ndarray,
    result: dict,
    mode: str = "coefficients",
    metric: str = "abs",
    linkage_method: str = "average",
    distance_threshold: float = 0.5,
    n_bootstrap: int = 200,
    seed: int = 0,
    min_overlap: int = 30,
    pairwise_complete: bool = True,
) -> dict:
    """Resample participants with replacement and recluster each draw.

    Two complementary numbers come out of it:

    - a term-by-term co-assignment probability, i.e. how often two terms end up
      in the same cluster, which says whether a tie is carried by the whole
      population or by a handful of participants;
    - per reference cluster, the Hennig clusterwise Jaccard: for every draw, the
      best-matching bootstrap cluster's Jaccard with the reference cluster.
      Its mean is the cluster's stability; >= 0.75 is conventionally "stable",
      <= 0.6 "dissolved".
    """
    keep, terms = result["keep"], result["terms"]
    if result["linkage"] is None:
        return {}

    matrix = (values if mode == "coefficients" else presence)[:, keep]
    presence_kept = presence[:, keep]
    n_participants, n_terms = matrix.shape
    reference_clusters = result["clusters"]
    reference_sets = {c: set(np.flatnonzero(reference_clusters == c))
                      for c in np.unique(reference_clusters)}

    generator = np.random.default_rng(seed)
    coassignment = np.zeros((n_terms, n_terms))
    jaccards = {c: [] for c in reference_sets}

    for _ in range(n_bootstrap):
        rows = generator.integers(0, n_participants, size=n_participants)
        distance, _, _ = term_distance(
            matrix[rows],
            presence=presence_kept[rows] if pairwise_complete else None,
            mode=mode, metric=metric, min_overlap=min_overlap,
        )
        clusters = fcluster(linkage(squareform(distance, checks=False), method=linkage_method),
                            t=distance_threshold, criterion="distance")
        coassignment += (clusters[:, None] == clusters[None, :]).astype(float)

        bootstrap_sets = [set(np.flatnonzero(clusters == c)) for c in np.unique(clusters)]
        for cluster_id, reference_set in reference_sets.items():
            best = max(len(reference_set & b) / len(reference_set | b) for b in bootstrap_sets)
            jaccards[cluster_id].append(best)

    coassignment /= n_bootstrap

    # Stability of each reference cluster, plus the weakest link inside it: the
    # lowest co-assignment probability among its member pairs.
    rows_out = []
    for cluster_id, reference_set in reference_sets.items():
        members = sorted(reference_set)
        if len(members) > 1:
            block = coassignment[np.ix_(members, members)]
            min_pair = block[~np.eye(len(members), dtype=bool)].min()
        else:
            min_pair = np.nan
        draws = np.asarray(jaccards[cluster_id])
        rows_out.append({
            "cluster": cluster_id,
            "size": len(members),
            "terms": " | ".join(terms[i] for i in members),
            "mean_jaccard": draws.mean(),
            "frac_stable": float(np.mean(draws >= 0.75)),
            "frac_dissolved": float(np.mean(draws <= 0.5)),
            "min_pair_coassignment": min_pair,
        })

    return {"coassignment": coassignment,
            "clusters": pd.DataFrame(rows_out).sort_values("cluster").reset_index(drop=True)}


def plot_coassignment(
    results: Dict[str, dict],
    stabilities: Dict[str, dict],
    output_path: str,
):
    """Bootstrap co-assignment probability per module, in dendrogram leaf order."""
    modules = [m for m in results if stabilities.get(m)]
    if not modules:
        return

    n_cols = min(2, len(modules))
    n_rows = int(np.ceil(len(modules) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, module in zip(axes, modules):
        result = results[module]
        order = dendrogram(result["linkage"], no_plot=True)["leaves"]
        probability = stabilities[module]["coassignment"][np.ix_(order, order)]
        labels = [result["terms"][i] for i in order]
        image = axis.imshow(probability, cmap="viridis", vmin=0, vmax=1)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_title(f"{module} — co-assignment", fontsize=11, fontweight="bold")
        figure.colorbar(image, ax=axis, fraction=0.046)

    for axis in axes[len(modules):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def describe_clusters(result: dict, mode: str) -> List[str]:
    """Human-readable summary of every multi-term cluster of one module."""
    lines = []
    table, similarity, terms = result["table"], result["similarity"], result["terms"]
    if table.empty:
        return ["  (fewer than two terms with across-participant variance)"]

    index_of = {t: i for i, t in enumerate(terms)}
    for cluster_id, group in table.groupby("cluster"):
        members = list(group["term"])
        if len(members) == 1:
            continue
        indices = [index_of[t] for t in members]
        block = similarity[np.ix_(indices, indices)]
        off_diagonal = block[~np.eye(len(indices), dtype=bool)]
        lines.append(f"  cluster {cluster_id}: mean |similarity| = {np.abs(off_diagonal).mean():.2f}")
        for term in members:
            row = group[group["term"] == term].iloc[0]
            # Sign relative to the first member makes ties like c_a = -c_b readable.
            relation = similarity[index_of[members[0]], index_of[term]]
            sign = "+" if relation >= 0 else "-"
            lines.append(f"      [{sign}] {term:40s} act={row['active_fraction']:.2f} "
                         f"mean={row['mean']:+.3f} sd={row['sd']:.3f}")
    if not lines:
        lines.append("  (no multi-term cluster below the distance threshold)")
    return lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dendrograms(
    results: Dict[str, dict],
    output_path: str,
    distance_threshold: float = 0.5,
    title: str = "",
):
    """One dendrogram panel per module."""
    modules = [m for m in results if results[m]["linkage"] is not None]
    n_modules = len(modules)
    if n_modules == 0:
        return

    n_cols = min(2, n_modules)
    n_rows = int(np.ceil(n_modules / n_cols))
    height = max(3.0, 0.32 * max(len(results[m]["terms"]) for m in modules))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, height * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, module in zip(axes, modules):
        result = results[module]
        dendrogram(
            result["linkage"],
            labels=result["terms"],
            orientation="left",
            color_threshold=distance_threshold,
            above_threshold_color="0.6",
            ax=axis,
        )
        axis.axvline(distance_threshold, color="crimson", linestyle="--", linewidth=1)
        axis.set_title(module, fontsize=11, fontweight="bold")
        axis.set_xlabel("distance")
        axis.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    for axis in axes[n_modules:]:
        axis.axis("off")

    if title:
        figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_similarity_heatmaps(results: Dict[str, dict], output_path: str):
    """Term-by-term similarity per module, reordered by the dendrogram leaves."""
    modules = [m for m in results if results[m]["linkage"] is not None]
    if not modules:
        return

    n_cols = min(2, len(modules))
    n_rows = int(np.ceil(len(modules) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, module in zip(axes, modules):
        result = results[module]
        order = dendrogram(result["linkage"], no_plot=True)["leaves"]
        similarity = result["similarity"][np.ix_(order, order)]
        labels = [result["terms"][i] for i in order]
        image = axis.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_title(module, fontsize=11, fontweight="bold")
        figure.colorbar(image, ax=axis, fraction=0.046)

    for axis in axes[len(modules):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analysis_coefficient_dendrogram(
    output_dir: str,
    spice_model=None,
    model_path: Optional[str] = None,
    spice_class: Optional[BaseModel] = None,
    spice_config: Optional[SpiceConfig] = None,
    n_actions: int = 2,
    n_participants: int = 1,
    n_experiments: int = 1,
    experiment: int = 0,
    polynomial_degree: int = 2,
    model_kwargs: Optional[dict] = None,
    mode: str = "coefficients",
    metric: str = "abs",
    linkage_method: str = "average",
    distance_threshold: float = 0.5,
    min_active_fraction: float = 0.05,
    min_overlap: int = 30,
    pairwise_complete: bool = True,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
    prefix: Optional[str] = None,
) -> Dict[str, dict]:
    """Cluster each module's SINDy terms and write dendrograms + a cluster table."""
    os.makedirs(output_dir, exist_ok=True)
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(model_path))[0] if model_path else "spice"

    if spice_model is not None:
        estimator = spice_model
    elif model_path is not None:
        estimator = load_estimator(
            model_path=model_path,
            spice_class=spice_class,
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=n_participants,
            n_experiments=n_experiments,
            polynomial_degree=polynomial_degree,
            model_kwargs=model_kwargs,
        )
    else:
        raise ValueError("Provide either spice_model or model_path.")
    matrices = get_module_matrices(estimator, experiment=experiment)

    results, tables, stabilities, stability_tables = {}, [], {}, []
    for module, data in matrices.items():
        result = cluster_module(
            values=data["values"],
            presence=data["presence"],
            terms=data["terms"],
            mode=mode,
            metric=metric,
            linkage_method=linkage_method,
            distance_threshold=distance_threshold,
            min_active_fraction=min_active_fraction,
            min_overlap=min_overlap,
            pairwise_complete=pairwise_complete,
        )
        results[module] = result

        n_untestable = (int(np.triu(result["untestable"], k=1).sum())
                        if result["untestable"] is not None else 0)
        print(f"\n=== {module} ({len(result['terms'])} terms with variance, "
              f"{len(result['dropped'])} dropped, "
              f"{n_untestable} pairs untestable at overlap>={min_overlap}) ===")
        for line in describe_clusters(result, mode=mode):
            print(line)

        if not result["table"].empty:
            table = result["table"].copy()
            table.insert(0, "module", module)
            tables.append(table)

        if n_bootstrap > 0 and result["linkage"] is not None:
            stability = bootstrap_stability(
                values=data["values"],
                presence=data["presence"],
                result=result,
                mode=mode,
                metric=metric,
                linkage_method=linkage_method,
                distance_threshold=distance_threshold,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
                min_overlap=min_overlap,
                pairwise_complete=pairwise_complete,
            )
            stabilities[module] = stability
            summary = stability["clusters"]
            print(f"  -- bootstrap stability over {n_bootstrap} participant resamples --")
            for _, row in summary[summary["size"] > 1].iterrows():
                print(f"      cluster {row['cluster']} (n={row['size']}): "
                      f"jaccard={row['mean_jaccard']:.2f} "
                      f"stable={row['frac_stable']:.2f} dissolved={row['frac_dissolved']:.2f} "
                      f"weakest pair={row['min_pair_coassignment']:.2f}")
            summary = summary.copy()
            summary.insert(0, "module", module)
            stability_tables.append(summary)

    suffix = mode if mode == "presence" else f"{mode}_{metric}"
    if tables:
        table_path = os.path.join(output_dir, f"dendrogram_clusters_{prefix}_{suffix}.csv")
        pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
        print(f"\nSaved cluster table to {table_path}")

    dendrogram_path = os.path.join(output_dir, f"dendrogram_{prefix}_{suffix}.png")
    plot_dendrograms(results, dendrogram_path, distance_threshold=distance_threshold,
                     title=f"{prefix} — {mode} ({metric if mode == 'coefficients' else 'jaccard'}, "
                           f"{linkage_method} linkage)")
    print(f"Saved dendrograms to {dendrogram_path}")

    heatmap_path = os.path.join(output_dir, f"dendrogram_heatmaps_{prefix}_{suffix}.png")
    plot_similarity_heatmaps(results, heatmap_path)
    print(f"Saved similarity heatmaps to {heatmap_path}")

    if stability_tables:
        stability_path = os.path.join(output_dir, f"dendrogram_stability_{prefix}_{suffix}.csv")
        pd.concat(stability_tables, ignore_index=True).to_csv(stability_path, index=False)
        print(f"Saved bootstrap stability to {stability_path}")

        coassignment_path = os.path.join(output_dir, f"dendrogram_coassignment_{prefix}_{suffix}.png")
        plot_coassignment(results, stabilities, coassignment_path)
        print(f"Saved co-assignment heatmaps to {coassignment_path}")

    return {"clusters": results, "stability": stabilities}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to the SPICE checkpoint (.pkl)")
    parser.add_argument("--module", required=True,
                        help="Dotted module path exposing SpiceModel and CONFIG")
    parser.add_argument("--n-actions", type=int, default=2)
    parser.add_argument("--n-participants", type=int, required=True)
    parser.add_argument("--n-experiments", type=int, default=1)
    parser.add_argument("--experiment", type=int, default=0)
    parser.add_argument("--polynomial-degree", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["coefficients", "presence"], default="coefficients")
    parser.add_argument("--metric", choices=["abs", "signed"], default="abs")
    parser.add_argument("--linkage", dest="linkage_method",
                        choices=["average", "complete", "single", "weighted"], default="average")
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--min-active-fraction", type=float, default=0.05)
    parser.add_argument("--min-overlap", type=int, default=30,
                        help="Minimum co-present participants for a pair to be correlated")
    parser.add_argument("--full-columns", action="store_true",
                        help="Correlate full columns, mixing structural zeros into the "
                             "parametric correlation (the old behaviour)")
    parser.add_argument("--n-bootstrap", type=int, default=0,
                        help="Participant resamples for cluster stability (0 disables)")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--model-kwarg", action="append", default=[],
                        help="Extra kwarg for the SpiceModel, as key=value (repeatable)")
    args = parser.parse_args()

    model_kwargs = {}
    for item in args.model_kwarg:
        key, value = item.split("=", 1)
        model_kwargs[key] = {"true": True, "false": False}.get(value.lower(), value)

    imported = importlib.import_module(args.module)
    analysis_coefficient_dendrogram(
        model_path=args.model,
        spice_class=imported.SpiceModel,
        spice_config=imported.CONFIG,
        n_actions=args.n_actions,
        n_participants=args.n_participants,
        n_experiments=args.n_experiments,
        experiment=args.experiment,
        polynomial_degree=args.polynomial_degree,
        output_dir=args.output_dir,
        mode=args.mode,
        metric=args.metric,
        linkage_method=args.linkage_method,
        distance_threshold=args.distance_threshold,
        min_active_fraction=args.min_active_fraction,
        min_overlap=args.min_overlap,
        pairwise_complete=not args.full_columns,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()

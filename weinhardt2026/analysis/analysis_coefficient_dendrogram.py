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
- ``'presence'`` — 1 - phi between binary presence masks, i.e. which terms are
  switched on/off together by pruning beyond what their frequency alone implies.

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


def split_by_sign(data: dict) -> dict:
    """Replace every term by one pseudo-term per sign: ``term (+)`` and ``term (-)``.

    A pseudo-term is present in a participant who has the term with that sign.
    A term whose coefficient points one way in some participants and the other
    way in others describes two different mechanisms (e.g. repeating vs
    switching), so for structural analyses the direction is part of the
    structure. The two signs of one term exclude each other (phi < 0) and never
    group together. A sign nobody carries is dropped here; rare ones fall to
    ``min_active_fraction`` later.
    """
    values, present, terms = data["values"], data["presence"] > 0, data["terms"]
    columns_values, columns_presence, names = [], [], []
    for index, term in enumerate(terms):
        for label, has_sign in (("+", values[:, index] > 0), ("-", values[:, index] < 0)):
            carriers = present[:, index] & has_sign
            if carriers.any():
                names.append(f"{term} ({label})")
                columns_presence.append(carriers.astype(float))
                columns_values.append(np.where(carriers, values[:, index], 0.0))

    n_participants = values.shape[0]
    empty = np.empty((n_participants, 0))
    return {
        "values": np.column_stack(columns_values) if names else empty,
        "presence": np.column_stack(columns_presence) if names else empty,
        "terms": names,
    }


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
    """Co-occurrence distance between the binary presence columns of a (P, T) matrix.

    Uses phi — the correlation of the two 0/1 columns — so it measures whether
    two terms occur together *more than chance*. Jaccard does not: two terms that
    are each present in most participants overlap heavily even when entirely
    independent, so Jaccard merges common terms rather than co-occurring ones.

    The distance is 1 - phi, so only terms that switch on together join; terms
    that exclude each other (phi < 0) are pushed apart. A term present in all or
    no participants has no variation and gets phi = 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.nan_to_num(np.corrcoef(matrix.T), nan=0.0)
    np.fill_diagonal(similarity, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
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

# (stars, alpha) from strictest to loosest, shared by tests, cuts and colours.
STAR_TIERS = (("***", 0.001), ("**", 0.01), ("*", 0.05))


def significance_stars(p: float) -> str:
    """The star convention used across the beta analyses."""
    if not np.isfinite(p):
        return "na"
    for stars, alpha in STAR_TIERS:
        if p < alpha:
            return stars
    return "ns"


def merge_significance(
    matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    presence: Optional[np.ndarray] = None,
    mode: str = "coefficients",
    metric: str = "abs",
    linkage_method: str = "average",
    min_overlap: int = 30,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Permutation p-value for every merge of one module's tree.

    Each term's column is shuffled across participants independently, which
    keeps how common every term is but destroys any link between terms. The
    tree is rebuilt from the shuffled data and its tightest merge — the best
    coincidence anywhere in the module — is recorded. A real merge's p-value is
    the share of shuffled trees whose tightest merge was at least as tight.

    Comparing every merge against the module-wide best coincidence controls the
    family-wise error rate per module: the chance of declaring even one false
    group in the module is at most *alpha*. It is conservative for looser
    merges higher up the tree.

    Returns the p-value per linkage row and the critical distance: a merge is
    significant exactly when its height is below it, so cutting the tree there
    yields the significant groups.
    """
    generator = np.random.default_rng(seed)
    n_participants, n_terms = matrix.shape

    null_tightest = np.empty(n_permutations)
    for index in range(n_permutations):
        order = np.column_stack([generator.permutation(n_participants) for _ in range(n_terms)])
        shuffled = np.take_along_axis(matrix, order, axis=0)
        # A term's presence travels with its values, so co-presence stays per term.
        shuffled_presence = (np.take_along_axis(presence, order, axis=0)
                             if presence is not None else None)
        distance, _, _ = term_distance(shuffled, presence=shuffled_presence, mode=mode,
                                       metric=metric, min_overlap=min_overlap)
        null_linkage = linkage(squareform(distance, checks=False), method=linkage_method)
        null_tightest[index] = null_linkage[:, 2].min()

    null_sorted = np.sort(null_tightest)
    heights = linkage_matrix[:, 2]
    at_least_as_tight = np.searchsorted(null_sorted, heights, side="right")
    p_values = (1 + at_least_as_tight) / (n_permutations + 1)

    return {"p_values": p_values,
            "critical_distance": _critical_distance(null_sorted, alpha),
            "tier_cuts": {tier: _critical_distance(null_sorted, tier_alpha)
                          for tier, tier_alpha in STAR_TIERS},
            "null_tightest": null_tightest}


def _critical_distance(null_sorted: np.ndarray, alpha: float) -> float:
    """Merge height below which p < alpha, from the sorted null tightest merges.

    p < alpha  <=>  fewer than alpha*(N+1) - 1 null trees were at least as tight
               <=>  height below the k-th tightest null merge. When N is too small
    to resolve alpha, nothing can be significant and the cut is -inf.
    """
    k = int(np.ceil(alpha * (len(null_sorted) + 1) - 1))
    return null_sorted[k - 1] if k >= 1 else -np.inf


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
    n_permutations: int = 0,
    grouping_alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Cluster one module's candidate terms. Returns linkage, labels and table.

    With ``n_permutations > 0`` the groups are the merges whose co-occurrence is
    significant under a per-module permutation test (see ``merge_significance``)
    instead of the merges below a fixed ``distance_threshold``.
    """
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
                "dropped": dropped, "significance": None,
                "distance_threshold": distance_threshold}

    matrix = (values if mode == "coefficients" else presence)[:, keep]
    distance, similarity, untestable = term_distance(
        matrix,
        presence=presence[:, keep] if pairwise_complete else None,
        mode=mode, metric=metric, min_overlap=min_overlap,
    )

    linkage_matrix = linkage(squareform(distance, checks=False), method=linkage_method)

    significance = None
    if n_permutations > 0:
        significance = merge_significance(
            matrix, linkage_matrix,
            presence=presence[:, keep] if pairwise_complete else None,
            mode=mode, metric=metric, linkage_method=linkage_method,
            min_overlap=min_overlap, n_permutations=n_permutations,
            alpha=grouping_alpha, seed=seed,
        )
        # Heights never decrease up the tree, so the significant merges are
        # exactly those below the critical distance; cutting just under it
        # keeps them and nothing else.
        distance_threshold = np.nextafter(significance["critical_distance"], -np.inf)
    clusters = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

    table = pd.DataFrame({
        "term": kept_terms,
        "cluster": clusters,
        "active_fraction": active_fraction[keep],
        "mean": values[:, keep].mean(axis=0),
        "sd": values[:, keep].std(axis=0),
    })

    # Activity of the group each term belongs to. A tie between terms only
    # exists in participants who have all of them, so "all" is the primary
    # number; "any" shows how much wider the group's footprint is.
    present = presence[:, keep] > 0
    for cluster_id in np.unique(clusters):
        members = clusters == cluster_id
        rows = table["cluster"] == cluster_id
        table.loc[rows, "group_size"] = int(members.sum())
        table.loc[rows, "group_active_all"] = present[:, members].all(axis=1).mean()
        table.loc[rows, "group_active_any"] = present[:, members].any(axis=1).mean()
    table["group_size"] = table["group_size"].astype(int)
    table = table.sort_values(["cluster", "term"]).reset_index(drop=True)

    return {"terms": kept_terms, "keep": keep, "linkage": linkage_matrix,
            "clusters": clusters, "similarity": similarity, "distance": distance,
            "untestable": untestable, "table": table, "dropped": dropped,
            "significance": significance, "distance_threshold": distance_threshold}


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

# Significance tiers, dark = strongest, shared by every panel.
STAR_COLORS = {"***": "#08306b", "**": "#2171b5", "*": "#6baed6", "ns": "0.75"}


def plot_dendrograms(
    results: Dict[str, dict],
    output_path: str,
    distance_threshold: float = 0.5,
    title: str = "",
):
    """One dendrogram panel per module.

    When a module carries merge p-values, every merge — the bracket joining its
    two children — is coloured by its significance tier and the dashed line
    marks the module's critical distance. Otherwise merges below
    *distance_threshold* are coloured by cluster.
    """
    modules = [m for m in results if results[m]["linkage"] is not None]
    n_modules = len(modules)
    if n_modules == 0:
        return

    n_cols = min(2, n_modules)
    n_rows = int(np.ceil(n_modules / n_cols))
    height = max(3.0, 0.32 * max(len(results[m]["terms"]) for m in modules))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, height * n_rows))
    axes = np.atleast_1d(axes).ravel()

    any_significance = False
    for axis, module in zip(axes, modules):
        result = results[module]
        significance = result.get("significance")
        n_leaves = len(result["terms"])

        if significance is not None:
            any_significance = True
            stars = [significance_stars(p) for p in significance["p_values"]]
            # Link ids above n_leaves index the linkage rows in order.
            dendrogram(result["linkage"], labels=result["terms"], orientation="left",
                       link_color_func=lambda link: STAR_COLORS[stars[link - n_leaves]],
                       ax=axis)
            cuts = {tier: significance["tier_cuts"][tier] for tier, _ in STAR_TIERS}
        else:
            dendrogram(result["linkage"], labels=result["terms"], orientation="left",
                       color_threshold=distance_threshold, above_threshold_color="0.6",
                       ax=axis)
            cuts = {None: distance_threshold}

        finite = [cut for cut in cuts.values() if np.isfinite(cut)]
        if finite:
            # The axis runs right-to-left; widen it when a cut lies past the root.
            left, right = axis.get_xlim()
            if max(finite) > left:
                axis.set_xlim(max(finite) * 1.05, right)
        for tier, cut in cuts.items():
            if np.isfinite(cut):
                axis.axvline(cut, color=STAR_COLORS[tier] if tier else "crimson",
                             linestyle="--", linewidth=1.2)
        axis.set_title(module, fontsize=11, fontweight="bold")
        axis.set_xlabel("distance")
        axis.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    for axis in axes[n_modules:]:
        axis.axis("off")

    if any_significance:
        handles = [plt.Line2D([], [], color=STAR_COLORS[tier], linewidth=3,
                              label=f"merge p < {tier_alpha:g}")
                   for tier, tier_alpha in STAR_TIERS]
        handles.append(plt.Line2D([], [], color=STAR_COLORS["ns"], linewidth=3,
                                  label="merge not significant"))
        handles += [plt.Line2D([], [], color=STAR_COLORS[tier], linestyle="--",
                               linewidth=1.2, label=f"cut p = {tier_alpha:g}")
                    for tier, tier_alpha in STAR_TIERS]
        figure.legend(handles=handles, loc="upper right", fontsize=8, frameon=False,
                      title="co-occurrence (permutation,\nFWER per module)",
                      title_fontsize=8)

    if title:
        figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 0.9 if any_significance else 1, 1))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def merge_table(result: dict, presence: np.ndarray, module: str) -> pd.DataFrame:
    """One row per merge of the tree: every grouping level, from pairs to root."""
    if result["linkage"] is None:
        return pd.DataFrame()

    terms = result["terms"]
    present = presence[:, result["keep"]] > 0
    significance = result.get("significance")
    n_leaves = len(terms)

    # Members of every node: leaves first, then one new node per linkage row.
    members = [[i] for i in range(n_leaves)]
    rows = []
    for index, (left, right, height, _) in enumerate(result["linkage"]):
        node = members[int(left)] + members[int(right)]
        members.append(node)
        columns = present[:, node]
        p_value = significance["p_values"][index] if significance is not None else np.nan
        rows.append({
            "module": module,
            "merge": index + 1,
            "size": len(node),
            "terms": " | ".join(terms[i] for i in node),
            "distance": height,
            "similarity": 1.0 - height,
            "p": p_value,
            "stars": significance_stars(p_value),
            "active_all": columns.all(axis=1).mean(),
            "active_any": columns.any(axis=1).mean(),
        })
    return pd.DataFrame(rows)


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
    n_permutations: int = 0,
    grouping_alpha: float = 0.05,
    permutation_seed: int = 0,
    split_signs: bool = True,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
    prefix: Optional[str] = None,
) -> Dict[str, dict]:
    """Cluster each module's SINDy terms and write dendrograms + a cluster table.

    With ``n_permutations > 0`` every merge gets a permutation p-value for
    co-occurrence (FWER per module), groups are the significant merges, and a
    per-merge table covering every grouping level is written alongside.

    In presence mode with ``split_signs`` (default), every term is split into
    ``term (+)`` and ``term (-)`` first, so groups carry the direction of each
    term (see ``split_by_sign``).
    """
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

    results, tables, stabilities, stability_tables, merge_tables = {}, [], {}, [], []
    for module, data in matrices.items():
        if mode == "presence" and split_signs:
            data = split_by_sign(data)
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
            n_permutations=n_permutations,
            grouping_alpha=grouping_alpha,
            seed=permutation_seed,
        )
        results[module] = result

        merges = merge_table(result, data["presence"], module)
        if not merges.empty:
            merge_tables.append(merges)


        n_untestable = (int(np.triu(result["untestable"], k=1).sum())
                        if result["untestable"] is not None else 0)
        print(f"\n=== {module} ({len(result['terms'])} terms with variance, "
              f"{len(result['dropped'])} dropped, "
              f"{n_untestable} pairs untestable at overlap>={min_overlap}) ===")
        if result["significance"] is not None:
            print(f"  critical distance (p < {grouping_alpha}): "
                  f"{result['significance']['critical_distance']:.3f}")
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
                distance_threshold=result["distance_threshold"],
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

    if merge_tables:
        merges_path = os.path.join(output_dir, f"dendrogram_merges_{prefix}_{suffix}.csv")
        pd.concat(merge_tables, ignore_index=True).to_csv(merges_path, index=False)
        print(f"Saved per-merge table to {merges_path}")

    dendrogram_path = os.path.join(output_dir, f"dendrogram_{prefix}_{suffix}.png")
    plot_dendrograms(results, dendrogram_path, distance_threshold=distance_threshold,
                     title=f"{prefix} — {mode} ({metric if mode == 'coefficients' else 'phi'}, "
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
    parser.add_argument("--n-permutations", type=int, default=0,
                        help="Permutations for merge significance; groups become the "
                             "significant merges (0 keeps the fixed distance threshold)")
    parser.add_argument("--grouping-alpha", type=float, default=0.05)
    parser.add_argument("--no-split-signs", action="store_true",
                        help="Presence mode: treat a term as one unit regardless of its sign")
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
        n_permutations=args.n_permutations,
        grouping_alpha=args.grouping_alpha,
        split_signs=not args.no_split_signs,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()

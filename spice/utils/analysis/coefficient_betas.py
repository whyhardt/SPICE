"""Structural beta effects: does term presence relate to a behavioral metric?

SPICE's claim is about structural individual differences — which terms exist
in a participant's equation — so this tests **presence**, never coefficient
magnitude: one Firth-penalized logistic regression per node, presence ~ z(metric), with
Benjamini-Hochberg correction across all tested nodes. A discrete criterion
(e.g. diagnosis) is compared group by group against a reference, each contrast
over the participants of its two groups with its own correction; beta is then
the log odds ratio of presence in the group vs the reference.

By default every term is split by sign into ``term (+)`` and ``term (-)``: a
term that points one way in some participants and the other way in others
describes two different mechanisms, so the direction is part of the structure.

A *leaf* is one node of the per-module presence dendrogram: either a group of
terms whose co-occurrence is significant — they switch on and off together more
than their frequencies alone would produce, judged by a permutation test with
family-wise error control per module — or a term that co-occurs significantly
with nothing and stays on its own. A group counts as present in a participant
only when all of its terms are. Because group members co-occur by construction, one test per group
replaces several near-identical single-term tests. The share of participants
with *any* member present is reported next to it as a check: when it is far
above the *all* share, the group is looser than its label suggests.

A group whose effect on the metric is not significant is broken up into its two
subgroups (its children in the tree), and so on down to single terms. So per
branch the analysis reports the largest group that is significant, or else the
finest level it can. Every node inside every group is tested up front and the
Benjamini-Hochberg family is all of them; the break-up rule only decides which
result is *reported*. Counting only the nodes the rule visits would make the
family shrink exactly when a group looks significant (its subgroups would never
be counted), so groups that look good would face a milder correction than those
that don't. A fixed family keeps the correction independent of the results.

Nodes present in almost no one or almost everyone carry no structural variation
and are not tested; like a non-significant group, they are broken up.

Metrics that scale with how much a participant played (e.g. rewards per block)
are not suitable: term survival depends on data volume through pruning, so such
a metric produces spurious presence effects. The analysis does not adjust for
trial count — it checks for this, and warns when the metric correlates with it.

Usage:

    python -m spice.utils.analysis.coefficient_betas \
        --model weinhardt2026/studies/dezfouli2019/params_array/spice_dezfouli2019_0.1_0.7.pkl \
        --module spice.precoded.workingmemory \
        --data weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv \
        --criterion reward \
        --output-dir weinhardt2026/studies/dezfouli2019/results
"""

import argparse
import importlib
import os
import textwrap
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit
from scipy.stats import chi2, kruskal, pearsonr
from statsmodels.stats.multitest import multipletests

from spice import BaseModel, SpiceConfig
from spice.utils.analysis.coefficient_dendrogram import (
    cluster_module,
    get_module_matrices,
    split_by_sign,
)
from spice.utils.analysis.loading import load_estimator


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------

def get_criterion(
    data_path,
    criterion_col: str = "reward",
    participant_col: str = "participant",
    aggregate: str = "mean",
    block_col: Optional[str] = None,
) -> pd.DataFrame:
    """Per-participant criterion plus log trial count for the volume check.

    *data_path* is a CSV path or the data as a DataFrame. With *block_col*, the
    criterion is first summed per block and then aggregated over the participant's
    blocks (e.g. the block-averaged total reward).

    Participant order follows ``df[participant_col].unique()``, which is the same
    order ``csv_to_dataset`` assigns integer participant indices in, so row *i*
    of the returned frame is participant index *i* of the model.
    """
    raw = data_path if isinstance(data_path, pd.DataFrame) else pd.read_csv(data_path)
    order = list(raw[participant_col].unique())

    if block_col is None:
        criterion = raw.groupby(participant_col)[criterion_col].agg(aggregate)
    else:
        criterion = raw.groupby([participant_col, block_col])[criterion_col].sum().groupby(level=0).agg(aggregate)
    n_trials = raw.groupby(participant_col).size().rename("n_trials")

    frame = pd.concat([criterion, n_trials], axis=1).loc[order].reset_index()
    frame = frame.rename(columns={participant_col: "participant_id"})
    frame["log_n_trials"] = np.log(frame["n_trials"])
    return frame


def check_trial_count(criterion: np.ndarray, log_n_trials: np.ndarray,
                      criterion_col: str, criterion_type: str = "continuous",
                      alpha: float = 0.05) -> float:
    """Warn when the criterion moves with how much a participant played.

    Continuous: Pearson correlation with log trial count. Discrete: Kruskal-Wallis
    test of log trial count across the groups.
    """
    if criterion_type == "discrete":
        groups = [log_n_trials[criterion == label] for label in np.unique(criterion)]
        p = kruskal(*groups).pvalue
        print(f"Trial-count check: log trials across '{criterion_col}' groups, "
              f"Kruskal-Wallis p = {p:.3g}")
    else:
        defined = np.isfinite(criterion.astype(float))
        r, p = pearsonr(criterion[defined], log_n_trials[defined])
        print(f"Trial-count check: {criterion_col} ~ log trials, r = {r:+.2f} (p = {p:.3g})")
    if p < alpha:
        print(f"  WARNING: '{criterion_col}' is related to trial count. Term survival "
              f"depends on data volume, so presence effects against it can be spurious. "
              f"Use a criterion that does not scale with trial count.")
    return p


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------

def get_candidates(presence: np.ndarray, result: dict) -> List[dict]:
    """Every node inside every co-occurrence group, plus ungrouped single terms.

    A group is a maximal significant subtree of the presence dendrogram; its
    sub-merges are all tighter, so every node within it is a valid co-occurring
    group too. Each node records its parent and children within the group, which
    is what the break-up rule walks. Node ids follow scipy: leaves are
    0..n-1, the merge in linkage row k is n+k.
    """
    terms = result["terms"]
    present = presence[:, result["keep"]] > 0
    n_leaves = len(terms)

    members = [[i] for i in range(n_leaves)]
    children = [()] * n_leaves
    if result["linkage"] is not None:
        for left, right, _, _ in result["linkage"]:
            members.append(members[int(left)] + members[int(right)])
            children.append((int(left), int(right)))

    clusters = (result["clusters"] if result["clusters"] is not None
                else np.arange(1, n_leaves + 1))
    roots = []
    for cluster_id in np.unique(clusters):
        member_set = set(np.flatnonzero(clusters == cluster_id))
        roots.append(next(node for node in range(len(members))
                          if set(members[node]) == member_set))

    candidates = []

    def visit(node, parent, group, depth):
        columns = present[:, members[node]]
        names = [terms[i] for i in members[node]]
        candidates.append({
            "group": group,
            "node": node,
            "parent": parent,
            "depth": depth,
            "children": children[node],
            "members": names,
            "label": names[0] if len(names) == 1 else "{" + ", ".join(names) + "}",
            "presence": columns.all(axis=1).astype(float),
            "presence_any": columns.any(axis=1).astype(float),
        })
        for child in children[node]:
            visit(child, node, group, depth + 1)

    for group, root in enumerate(roots, start=1):
        visit(root, -1, group, 0)
    return candidates


def consensus_presence(seed_matrices: List[dict], votes: int) -> dict:
    """Per module: a participant carries ``term (±)`` when >= *votes* seeds carry it.

    *seed_matrices* holds one ``get_module_matrices`` result per seed; terms are
    split by sign first, so the vote is on the signed term.
    """
    consensus = {}
    for module in seed_matrices[0]:
        splits = [split_by_sign(matrices[module]) for matrices in seed_matrices]
        terms = sorted({term for split in splits for term in split["terms"]})
        n_participants = splits[0]["presence"].shape[0]
        counts = np.zeros((n_participants, len(terms)))
        for split in splits:
            for index, term in enumerate(split["terms"]):
                counts[:, terms.index(term)] += split["presence"][:, index] > 0
        consensus[module] = {"presence": (counts >= votes).astype(float), "terms": terms}
    return consensus


def get_mechanism_candidates(consensus: dict, mechanisms: List[Tuple[str, frozenset]],
                             linkage_method: str = "average") -> List[dict]:
    """Recurring mechanisms (one step of break-up each) plus every other signed term.

    A mechanism is the root of its own group; its two children are the top split
    of its members (1 - phi on consensus presence, as in the presence dendrogram;
    a two-term mechanism splits into its terms). Children have no children, so a
    non-significant mechanism is broken up by exactly one step. Every signed term
    outside all mechanisms of its module is a group of its own.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from spice.utils.analysis.coefficient_dendrogram import presence_distance

    candidates = []

    def node(module, group, node_id, parent, depth, children, names, present):
        columns = present[:, [consensus[module]["terms"].index(t) for t in names]] > 0
        return {"module": module, "group": group, "node": node_id, "parent": parent,
                "depth": depth, "children": children, "members": names,
                "label": names[0] if len(names) == 1 else "{" + ", ".join(names) + "}",
                "presence": columns.all(axis=1).astype(float),
                "presence_any": columns.any(axis=1).astype(float)}

    for module, data in consensus.items():
        present, group = data["presence"], 0
        module_mechanisms = [sorted(terms) for m, terms in mechanisms if m == module]
        for members in module_mechanisms:
            group += 1
            missing = [t for t in members if t not in data["terms"]]
            if missing:
                raise ValueError(f"{module}: mechanism terms {missing} absent from consensus presence")
            if len(members) == 2:
                halves = [[members[0]], [members[1]]]
            else:
                columns = present[:, [data["terms"].index(t) for t in members]]
                distance, _, _ = presence_distance(columns)
                top = linkage(squareform(distance, checks=False), method=linkage_method)
                label = np.zeros(len(members), dtype=int)
                # the last merge joins two clusters; assign leaves by membership of its left child
                clusters = {i: [i] for i in range(len(members))}
                for row, (left, right, _, _) in enumerate(top):
                    clusters[len(members) + row] = clusters[int(left)] + clusters[int(right)]
                left_leaves = clusters[int(top[-1, 0])]
                label[left_leaves] = 1
                halves = [[members[i] for i in range(len(members)) if label[i] == side] for side in (1, 0)]
            candidates.append(node(module, group, 0, -1, 0, (1, 2), members, present))
            for child_id, half in zip((1, 2), halves):
                candidates.append(node(module, group, child_id, 0, 1, (), half, present))
        in_mechanism = {t for members in module_mechanisms for t in members}
        for term in data["terms"]:
            if term not in in_mechanism:
                group += 1
                candidates.append(node(module, group, 0, -1, 0, (), [term], present))
    return candidates


def select_reported(table: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Per group, the largest significant node; otherwise break it up.

    Walks each group from its root. A node that is significant is reported and
    its subgroups are not. A node that is not — non-significant, too few or
    not estimable — hands over to its two children, down to single terms, which
    are reported whatever their result. The reported nodes partition the terms.

    Returns the reported nodes and the groups that were broken up on the way.
    """
    reported = pd.Series(False, index=table.index)
    broken = pd.Series(False, index=table.index)
    for _, nodes in table.groupby(["module", "group"]):
        index_of = dict(zip(nodes["node"], nodes.index))
        stack = list(nodes.loc[nodes["parent"] == -1, "node"])
        while stack:
            index = index_of[stack.pop()]
            children = table.at[index, "children"]
            if table.at[index, "significant"] or len(children) == 0:
                reported[index] = True
            else:
                broken[index] = True
                stack.extend(children)
    return reported, broken


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _penalized_loglik(beta: np.ndarray, design: np.ndarray,
                      presence: np.ndarray) -> Tuple[float, np.ndarray]:
    """Firth's penalized log-likelihood (log-likelihood + 1/2 log det of the Fisher
    information) and its gradient, the Firth-modified score (Heinze & Schemper 2002)."""
    linear = design @ beta
    probability = expit(linear)
    weights = probability * (1 - probability)
    information = design.T @ (weights[:, None] * design)
    _, logdet = np.linalg.slogdet(information)
    leverage = weights * np.einsum("ij,jk,ik->i", design, np.linalg.inv(information), design)
    score = design.T @ (presence - probability + leverage * (0.5 - probability))
    return np.sum(presence * linear - np.logaddexp(0, linear)) + 0.5 * logdet, score


def firth_profile(design: np.ndarray, presence: np.ndarray, slope: float,
                  intercept_bound: float = 40.0) -> float:
    """Firth-penalized log-likelihood maximized over the intercept at a fixed slope.

    The penalized likelihood is unimodal in the intercept, so a bounded scalar
    search is enough, even where fitted probabilities saturate.
    """
    result = minimize_scalar(lambda intercept: -_penalized_loglik(np.array([intercept, slope]),
                                                                  design, presence)[0],
                             bounds=(-intercept_bound, intercept_bound), method="bounded",
                             options={"xatol": 1e-8})
    return -result.fun


def firth_fit(design: np.ndarray, presence: np.ndarray,
              slope_grid: np.ndarray = np.linspace(-15, 15, 121)) -> Tuple[float, float]:
    """Firth-penalized logistic regression of presence on [intercept, slope].

    The penalized likelihood is not concave in the slope — a high-leverage
    predictor value adds a second mode near zero — so a local optimizer can stop
    in the wrong mode. Its profile over the intercept is well behaved, so the
    slope is found globally: the profile on *slope_grid*, refined around the best
    grid point. Returns the slope and the maximum penalized log-likelihood.
    """
    profile = [firth_profile(design, presence, slope) for slope in slope_grid]
    best = int(np.argmax(profile))
    if best in (0, len(slope_grid) - 1):
        raise np.linalg.LinAlgError("Firth slope outside the search grid")
    result = minimize_scalar(lambda slope: -firth_profile(design, presence, slope),
                             bounds=(slope_grid[best - 1], slope_grid[best + 1]), method="bounded",
                             options={"xatol": 1e-6})
    return result.x, -result.fun


def presence_beta(presence: np.ndarray, predictor: np.ndarray,
                  standardize: bool = True, ci_level: float = 0.95,
                  max_ci_width: float = 50.0) -> dict:
    """Firth-penalized logistic regression of presence on one predictor.

    With a standardized continuous predictor, beta is the change in log-odds of
    presence per SD. With a 0/1 group indicator (standardize=False), beta is the
    log odds ratio of presence in the group vs the reference. Firth's penalty
    keeps beta finite under (quasi-)separation — a group in which nobody, or
    everybody, has the term — so the strongest group differences are tested
    rather than dropped. The p-value is the penalized likelihood-ratio test and
    the interval the profile penalized-likelihood interval, both valid under
    separation where the Wald statistics are not; an interval bound beyond
    *max_ci_width* from beta is reported as infinite.
    """
    empty = {"beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "estimable": False}
    design = np.column_stack([np.ones(len(predictor)),
                              zscore(predictor) if standardize else predictor])
    try:
        slope, maximum = firth_fit(design, presence)

        def profile(value: float) -> float:
            return firth_profile(design, presence, value)

        statistic = max(2 * (maximum - profile(0.0)), 0.0)
        threshold = maximum - chi2.ppf(ci_level, 1) / 2

        def bound(direction: int) -> float:
            width = 1.0
            while profile(slope + direction * width) > threshold:
                width *= 2
                if width > max_ci_width:
                    return direction * np.inf
            ends = sorted((slope, slope + direction * width))
            return brentq(lambda slope: profile(slope) - threshold, *ends)

        ci_low, ci_high = bound(-1), bound(+1)
    except np.linalg.LinAlgError:
        return empty
    return {"beta": slope, "ci_low": ci_low, "ci_high": ci_high,
            "p": chi2.sf(statistic, 1), "estimable": True}


def add_fdr(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg q-values over the rows with a p-value, as one family."""
    table = table.copy()
    valid = table["p"].notna()
    table["q"] = np.nan
    table["significant"] = False
    if valid.any():
        rejected, q_values, _, _ = multipletests(table.loc[valid, "p"], alpha=alpha, method="fdr_bh")
        table.loc[valid, "q"] = q_values
        table.loc[valid, "significant"] = rejected
    return table


# ---------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------

def contrast_suffix(criterion_col: str, criterion_type: str, name: str) -> str:
    """File-name suffix of one contrast: the criterion, plus the compared groups for a discrete one."""
    return criterion_col if criterion_type == "continuous" else f"{criterion_col}_{name}"


def contrast_xlabel(criterion_col: str, criterion_type: str, name: str) -> str:
    """Axis label of one contrast's betas."""
    if criterion_type == "continuous":
        return f"log-odds of presence per SD of {criterion_col} (95% CI)"
    label, reference = name.split("_vs_")
    return f"log odds ratio of presence, {label} vs {reference} (95% CI)"


def get_contrasts(criterion: np.ndarray, criterion_col: str,
                  criterion_type: str = "continuous",
                  reference: Optional[str] = None,
                  comparisons: Optional[List[Tuple[str, str]]] = None) -> List[dict]:
    """The comparisons to run, each over a subset of participants.

    Continuous: one contrast over everyone, predictor = z(criterion). Discrete:
    one contrast per group against *reference* — or one per (group, reference)
    pair in *comparisons* — over the participants of those two groups, predictor
    = 1 for the group and 0 for the reference, so beta is a log odds ratio. Each
    contrast is its own analysis with its own FDR family.
    """
    if criterion_type == "continuous":
        defined = np.isfinite(criterion.astype(float))   # participants without the criterion are left out
        if not defined.all():
            print(f"'{criterion_col}' undefined for {(~defined).sum()} participant(s): left out of the contrast")
        return [{"name": criterion_col,
                 "mask": defined,
                 "predictor": criterion[defined].astype(float),
                 "standardize": True,
                 "xlabel": contrast_xlabel(criterion_col, criterion_type, criterion_col)}]

    if criterion_type != "discrete":
        raise ValueError(f"Unknown criterion_type '{criterion_type}'.")
    labels = sorted(np.unique(criterion))
    if comparisons is None:
        if reference not in labels:
            raise ValueError(f"Reference '{reference}' not in {labels}.")
        comparisons = [(label, reference) for label in labels if label != reference]
    contrasts = []
    for label, reference in comparisons:
        for name in (label, reference):
            if name not in labels:
                raise ValueError(f"Group '{name}' not in {labels}.")
        mask = np.isin(criterion, [reference, label])
        contrasts.append({"name": f"{label}_vs_{reference}",
                          "mask": mask,
                          "predictor": (criterion[mask] == label).astype(float),
                          "standardize": False,
                          "xlabel": contrast_xlabel(criterion_col, criterion_type, f"{label}_vs_{reference}")})
    return contrasts


def test_candidates(candidates: List[dict], contrast: dict,
                    min_count: int = 5,
                    alpha: float = 0.05) -> pd.DataFrame:
    """Test every node for one contrast, correct once, apply the break-up rule.

    A node is tested only when at least *min_count* of the contrast's participants
    carry it and at least *min_count* do not (Vittinghoff & McCulloch 2007: 5-9
    cases per predictor suffice; Firth's penalty removes most of the small-sample
    bias); otherwise it is marked "too few". A count, not a share, so the rule is
    the same for samples of any size. Presence and counts are computed on the
    contrast's own participants, since that is the sample the regression sees.
    """
    mask = contrast["mask"]
    rows = []
    for node in candidates:
        presence = node["presence"][mask]
        row = {"contrast": contrast["name"], "module": node["module"],
               "group": node["group"], "node": node["node"], "parent": node["parent"],
               "depth": node["depth"], "children": node["children"], "leaf": node["label"],
               "size": len(node["members"]),
               "active_all": presence.mean(),
               "active_any": node["presence_any"][mask].mean()}
        n_present = int(presence.sum())
        if min(n_present, len(presence) - n_present) >= min_count:
            row.update(presence_beta(presence, contrast["predictor"], contrast["standardize"]))
            row["status"] = "tested" if row["estimable"] else "not estimable"
        else:
            row.update({"beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "estimable": False,
                        "status": "too few"})
        rows.append(row)

    table = add_fdr(pd.DataFrame(rows), alpha=alpha)
    table["odds_ratio"] = np.exp(table["beta"])
    table["reported"], table["broken_up"] = select_reported(table)
    return table


def report_contrast(table: pd.DataFrame, contrast_name: str, alpha: float = 0.05):
    """Print the reported leaves, the untested ones, and the groups broken up."""
    tested = table[table["status"] == "tested"]
    reported = table[table["reported"]]
    reported_tested = reported[reported["status"] == "tested"]
    print(f"\n=== structural beta effects — {contrast_name} ===")
    print(f"  {len(tested)} nodes tested (FDR family), {len(reported)} reported leaves, "
          f"{int(reported_tested['significant'].sum())} significant at FDR {alpha}")
    for _, row in reported_tested.sort_values("p").iterrows():
        marker = "*" if row["significant"] else " "
        print(f"  {marker} {row['module'].replace('value_', ''):20s} {row['leaf'][:60]:60s} "
              f"all={row['active_all']:.2f} any={row['active_any']:.2f} "
              f"beta={row['beta']:+.3f} OR={row['odds_ratio']:.2f} q={row['q']:.3f}")
    skipped = reported[reported["status"] != "tested"]
    if not skipped.empty:
        print(f"\n  reported but not tested ({len(skipped)}):")
        for _, row in skipped.iterrows():
            print(f"    {row['module'].replace('value_', ''):20s} {row['leaf'][:60]:60s} "
                  f"all={row['active_all']:.2f}  [{row['status']}]")
    broken = table[table["broken_up"]]
    if not broken.empty:
        print(f"\n  groups broken up ({len(broken)}):")
        for _, row in broken.iterrows():
            reason = row["status"] if row["status"] != "tested" else f"n.s., q={row['q']:.3f}"
            print(f"    {row['module'].replace('value_', ''):20s} {row['leaf'][:60]:60s} [{reason}]")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_forest(table: pd.DataFrame, output_path: str, title: str,
                label_col: str = "leaf",
                xlabel: str = "log-odds of presence per SD of metric (95% CI)",
                module_order: Optional[List[str]] = None):
    """Beta with 95% CI per leaf, grouped by module, significant ones filled.

    Modules run top to bottom in *module_order* (e.g. the SPICE config's), else
    alphabetically.
    """
    table = table.dropna(subset=["beta"]).copy()
    if table.empty:
        return

    order = module_order or sorted(table["module"].unique())
    table["module_rank"] = table["module"].map({m: i for i, m in enumerate(order)})
    table = table.sort_values(["module_rank", "beta"]).reset_index(drop=True)
    labels = [textwrap.fill(f"{row['module'].replace('value_', '')} | {row[label_col]}", 46)
              for _, row in table.iterrows()]
    # Wrapped labels need vertical room, so rows grow with the tallest label.
    row_height = 0.22 * max(label.count("\n") + 1 for label in labels) + 0.14
    figure, axis = plt.subplots(figsize=(10, max(3.0, row_height * len(table))))

    palette = plt.get_cmap("tab10")
    modules = list(dict.fromkeys(table["module"]))
    # colour follows the module's position in *order*, so it is the same in every plot
    color_of = {module: palette(order.index(module) % 10) for module in modules}

    # An infinite profile bound (separation) runs to the plot edge and ends in an arrow.
    bounds = table[["beta", "ci_low", "ci_high"]].to_numpy()
    finite = bounds[np.isfinite(bounds)]
    margin = 0.1 * max(finite.max() - finite.min(), 1.0)
    edges = (min(finite.min(), 0) - margin, max(finite.max(), 0) + margin)
    for position, row in table.iterrows():
        color = color_of[row["module"]]
        low, high = np.clip([row["ci_low"], row["ci_high"]], *edges)
        axis.errorbar(row["beta"], position, xerr=[[row["beta"] - low], [high - row["beta"]]],
                      fmt="o", color=color, markersize=6, capsize=3,
                      markerfacecolor=color if row["significant"] else "white")
        for bound, edge, marker in ((row["ci_low"], low, "<"), (row["ci_high"], high, ">")):
            if np.isinf(bound):
                axis.plot(edge, position, marker=marker, color=color, markersize=5)
    axis.set_xlim(*edges)

    axis.axvline(0, color="0.3", linewidth=1)
    axis.set_yticks(range(len(table)))
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_ylim(len(table) - 0.5, -0.5)
    axis.set_xlabel(xlabel)
    axis.set_title(title, fontsize=11, fontweight="bold")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color_of[m], label=m)
               for m in modules]
    if not table["significant"].all():
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color="0.3",
                                  markerfacecolor="white", label="not significant (FDR)"))
    axis.legend(handles=handles, fontsize=7, frameon=False,
                loc="upper left", bbox_to_anchor=(1.02, 1.0))

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_mechanism_betas(table: pd.DataFrame, mechanisms: List[Tuple[str, frozenset]], output_path: str,
                         xlabel: str, clean: bool = False):
    """Beta and 95% CI of the mechanisms alone, for one contrast.

    Rows are M1, M2, ... in the order of *mechanisms*. Black marks are significant
    (FDR q below the analysis' alpha), white ones are not; a mechanism that was not
    tested (too few carriers or non-carriers, or not estimable) is marked n.t. An
    infinite profile bound runs to the plot edge and ends in an arrow. Saved
    annotated (<output_path>.pdf/.png: n.t. marks and x-label) and, with *clean*,
    without annotations (<output_path>_clean.pdf/.png).
    """
    roots = table[table["depth"] == 0]
    row_of = {(row["module"], frozenset(row["leaf"].strip("{}").split(", "))): row for _, row in roots.iterrows()}
    rows = [row_of.get((module, frozenset(core))) for module, core in mechanisms]
    tested = [row for row in rows if row is not None and row["status"] == "tested"]

    bounds = np.array([[row["beta"], row["ci_low"], row["ci_high"]] for row in tested]).ravel()
    finite = bounds[np.isfinite(bounds)] if len(bounds) else np.zeros(1)
    margin = 0.1 * max(finite.max() - finite.min(), 1.0)
    edges = (min(finite.min(), 0) - margin, max(finite.max(), 0) + margin)

    for annotated in ((True, False) if clean else (True,)):
        figure, axis = plt.subplots(figsize=(3.0, 0.32 * len(mechanisms) + 0.8))
        axis.axvline(0, color="0.6", linewidth=0.8, zorder=0)
        for position, row in enumerate(rows):
            if row is None or row["status"] != "tested":
                if annotated:
                    axis.text(0, position, "n.t.", ha="center", va="center", fontsize=7, color="0.5", style="italic")
                continue
            low, high = np.clip([row["ci_low"], row["ci_high"]], *edges)
            axis.plot([low, high], [position, position], color="black", linewidth=1.3, zorder=1)
            for bound, edge, marker in ((row["ci_low"], low, "<"), (row["ci_high"], high, ">")):
                if np.isinf(bound):
                    axis.plot(edge, position, marker=marker, color="black", markersize=4)
            axis.plot(row["beta"], position, marker="o", markersize=6, markeredgecolor="black", markeredgewidth=1.2,
                      markerfacecolor="black" if row["significant"] else "white", zorder=2)

        axis.set_xlim(*edges)
        axis.set_yticks(range(len(mechanisms)))
        axis.set_yticklabels([f"M{i + 1}" for i in range(len(mechanisms))], fontsize=8.5)
        axis.set_ylim(len(mechanisms) - 0.5, -0.5)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelsize=7.5)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        if annotated:
            axis.set_xlabel(xlabel, fontsize=7.5)

        path = output_path if annotated else f"{output_path}_clean"
        figure.savefig(f"{path}.pdf", bbox_inches="tight")
        figure.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {path}.pdf/.png")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analysis_coefficient_betas(
    data_path: str,
    output_dir: str,
    spice_model=None,
    model_path: Optional[str] = None,
    spice_class: Optional[BaseModel] = None,
    spice_config: Optional[SpiceConfig] = None,
    n_actions: int = 2,
    criterion_col: str = "reward",
    criterion_type: str = "continuous",
    reference: Optional[str] = None,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    participant_col: str = "participant",
    criterion_aggregate: str = "mean",
    n_experiments: int = 1,
    experiment: int = 0,
    polynomial_degree: int = 2,
    model_kwargs: Optional[dict] = None,
    linkage_method: str = "average",
    min_carriers: int = 5,
    n_permutations: int = 10000,
    grouping_alpha: float = 0.05,
    permutation_seed: int = 0,
    split_signs: bool = True,
    min_count: int = 5,
    alpha: float = 0.05,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Structural beta effects of every presence-dendrogram node on a criterion.

    *criterion_type* is "continuous" (e.g. reward rate) or "discrete" (e.g. a
    diagnosis column, compared group by group against *reference*). A discrete
    criterion is read as each participant's first value, whatever
    *criterion_aggregate* says. With *split_signs* (default) every term is split
    into ``term (+)`` and ``term (-)``, so presence carries the term's direction.
    """
    os.makedirs(output_dir, exist_ok=True)
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(model_path))[0] if model_path else "spice"

    aggregate = "first" if criterion_type == "discrete" else criterion_aggregate
    criterion_frame = get_criterion(data_path, criterion_col, participant_col, aggregate)
    criterion = criterion_frame[criterion_col].to_numpy()
    check_trial_count(criterion, criterion_frame["log_n_trials"].to_numpy(), criterion_col,
                      criterion_type=criterion_type, alpha=alpha)

    if spice_model is not None:
        estimator = spice_model
    elif model_path is not None:
        estimator = load_estimator(
            model_path=model_path,
            spice_class=spice_class,
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=len(criterion_frame),
            n_experiments=n_experiments,
            polynomial_degree=polynomial_degree,
            model_kwargs=model_kwargs,
        )
    else:
        raise ValueError("Provide either spice_model or model_path.")

    # Grouping never looks at the criterion, so it is computed once, on everyone,
    # and shared by every contrast.
    candidates = []
    for module, data in get_module_matrices(estimator, experiment=experiment).items():
        if split_signs:
            data = split_by_sign(data)
        result = cluster_module(
            values=data["values"],
            presence=data["presence"],
            terms=data["terms"],
            mode="presence",
            linkage_method=linkage_method,
            min_carriers=min_carriers,
            n_permutations=n_permutations,
            grouping_alpha=grouping_alpha,
            seed=permutation_seed,
        )
        for node in get_candidates(data["presence"], result):
            node["module"] = module
            candidates.append(node)

    return run_contrasts(candidates, criterion, criterion_col, criterion_type, reference,
                         comparisons, output_dir, prefix, min_count, alpha)


def run_contrasts(candidates: List[dict], criterion: np.ndarray, criterion_col: str,
                  criterion_type: str, reference: Optional[str],
                  comparisons: Optional[List[Tuple[str, str]]], output_dir: str, prefix: str,
                  min_count: int = 5,
                  alpha: float = 0.05, plot_significant_only: bool = False,
                  forest_plot: bool = True) -> pd.DataFrame:
    """Test the candidates for every contrast; print, save and (with *forest_plot*) plot each one.

    The forest plot shows the reported, tested leaves — only the significant ones
    with *plot_significant_only*; the CSV always holds every node.
    """
    tables = []
    for contrast in get_contrasts(criterion, criterion_col, criterion_type, reference,
                                  comparisons):
        table = test_candidates(candidates, contrast, min_count, alpha)
        report_contrast(table, contrast["name"], alpha)

        suffix = contrast_suffix(criterion_col, criterion_type, contrast["name"])
        if forest_plot:
            plot_path = os.path.join(output_dir, f"betas_structural_{prefix}_{suffix}.png")
            shown = table["reported"] & (table["status"] == "tested")
            if plot_significant_only:
                shown &= table["significant"]
            if os.path.exists(plot_path):
                os.remove(plot_path)  # nothing to show leaves no stale plot behind
            plot_forest(table[shown], plot_path,
                        module_order=list(dict.fromkeys(node["module"] for node in candidates)),
                        title=f"{prefix} — presence, {contrast['name']}",
                        xlabel=contrast["xlabel"])
            print(f"Saved {plot_path}")

        # One file per contrast, so runs with other comparisons never overwrite it.
        saved = table.copy()
        saved["children"] = saved["children"].map(lambda c: " ".join(map(str, c)))
        table_path = os.path.join(output_dir, f"betas_structural_{prefix}_{suffix}.csv")
        saved.to_csv(table_path, index=False)
        print(f"Saved {table_path}")
        tables.append(table)

    table = pd.concat(tables, ignore_index=True)
    return table


def analysis_mechanism_betas(
    seed_matrices: List[dict],
    mechanisms: List[Tuple[str, frozenset]],
    data_path,
    output_dir: str,
    votes: int,
    criterion_col: str = "reward",
    criterion_type: str = "continuous",
    reference: Optional[str] = None,
    comparisons: Optional[List[Tuple[str, str]]] = None,
    participant_col: str = "participant",
    criterion_aggregate: str = "mean",
    min_count: int = 5,
    alpha: float = 0.05,
    prefix: str = "mechanisms",
    criterion_map: Optional[dict] = None,
    block_col: Optional[str] = None,
    figure_prefix: str = "figure_mechanism_betas",
    clean: bool = False,
) -> pd.DataFrame:
    """Structural beta effects of mechanisms recurring across seeds, plus all other terms.

    Instead of one model's dendrogram groups, the leaves are *mechanisms* found
    across seeds (e.g. ``spice.utils.analysis.equation_discovery.core_mechanisms``) and every
    signed term outside them. Presence is the seed vote of ``consensus_presence``
    (>= *votes* seeds). A non-significant mechanism is broken up by one step only
    (``get_mechanism_candidates``); as in ``analysis_coefficient_betas``, every
    node is tested up front and forms one FDR family per contrast.

    *criterion_map* relabels the criterion values (discrete: names groups, or merges
    them by mapping several values onto one label). *data_path* (CSV or DataFrame)
    and *block_col* as in ``get_criterion``. Per contrast, the table of every node is
    saved as ``betas_structural_<prefix>_<criterion>[_<group>_vs_<reference>].csv``
    and the mechanisms alone are plotted (``plot_mechanism_betas``, rows M1, M2, ...
    in the order of *mechanisms*) as ``<figure_prefix>_<criterion>[...]``, with
    *clean* also without annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    aggregate = "first" if criterion_type == "discrete" else criterion_aggregate
    criterion_frame = get_criterion(data_path, criterion_col, participant_col, aggregate, block_col=block_col)
    if criterion_map is not None:
        criterion_frame[criterion_col] = criterion_frame[criterion_col].map(criterion_map)
    criterion = criterion_frame[criterion_col].to_numpy()
    check_trial_count(criterion, criterion_frame["log_n_trials"].to_numpy(), criterion_col,
                      criterion_type=criterion_type, alpha=alpha)

    candidates = get_mechanism_candidates(consensus_presence(seed_matrices, votes), mechanisms)
    table = run_contrasts(candidates, criterion, criterion_col, criterion_type, reference,
                          comparisons, output_dir, prefix, min_count, alpha, forest_plot=False)

    if mechanisms:
        for name, contrast_table in table.groupby("contrast", sort=False):
            suffix = contrast_suffix(criterion_col, criterion_type, name)
            plot_mechanism_betas(contrast_table, mechanisms, os.path.join(output_dir, f"{figure_prefix}_{suffix}"),
                                 contrast_xlabel(criterion_col, criterion_type, name), clean=clean)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--module", required=True,
                        help="Dotted module path exposing SpiceModel and CONFIG")
    parser.add_argument("--data", required=True)
    parser.add_argument("--criterion", default="reward")
    parser.add_argument("--criterion-type", choices=["continuous", "discrete"],
                        default="continuous")
    parser.add_argument("--reference", default=None,
                        help="Reference group for a discrete criterion")
    parser.add_argument("--comparison", action="append", default=None,
                        help="GROUP:REFERENCE pair for a discrete criterion (repeatable); "
                             "overrides --reference")
    parser.add_argument("--participant-col", default="participant")
    parser.add_argument("--criterion-aggregate", default="mean")
    parser.add_argument("--n-actions", type=int, default=2)
    parser.add_argument("--polynomial-degree", type=int, default=2)
    parser.add_argument("--min-carriers", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--grouping-alpha", type=float, default=0.05)
    parser.add_argument("--no-split-signs", action="store_true",
                        help="Treat a term as one unit regardless of its sign")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-kwarg", action="append", default=[],
                        help="Extra kwarg for the SpiceModel, as key=value (repeatable)")
    args = parser.parse_args()

    model_kwargs = {}
    for item in args.model_kwarg:
        key, value = item.split("=", 1)
        model_kwargs[key] = {"true": True, "false": False}.get(value.lower(), value)

    imported = importlib.import_module(args.module)
    analysis_coefficient_betas(
        data_path=args.data,
        output_dir=args.output_dir,
        model_path=args.model,
        spice_class=imported.SpiceModel,
        spice_config=imported.CONFIG,
        n_actions=args.n_actions,
        criterion_col=args.criterion,
        criterion_type=args.criterion_type,
        reference=args.reference,
        comparisons=([tuple(pair.split(":", 1)) for pair in args.comparison]
                     if args.comparison else None),
        participant_col=args.participant_col,
        criterion_aggregate=args.criterion_aggregate,
        polynomial_degree=args.polynomial_degree,
        min_carriers=args.min_carriers,
        n_permutations=args.n_permutations,
        grouping_alpha=args.grouping_alpha,
        split_signs=not args.no_split_signs,
        alpha=args.alpha,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()

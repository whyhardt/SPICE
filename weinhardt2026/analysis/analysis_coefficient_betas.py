"""Structural beta effects: does term presence relate to a behavioral metric?

SPICE's claim is about structural individual differences — which terms exist
in a participant's equation — so this tests **presence**, never coefficient
magnitude: one logistic regression per node, presence ~ z(metric), with
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

    python -m weinhardt2026.analysis.analysis_coefficient_betas \
        --model weinhardt2026/studies/dezfouli2019/params_array/spice_dezfouli2019_0.1_0.7.pkl \
        --module spice.precoded.workingmemory \
        --data weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv \
        --criterion reward \
        --output-dir weinhardt2026/studies/dezfouli2019/results
"""

import argparse
import importlib
import os
import sys
import textwrap
import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal, pearsonr
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spice import BaseModel, SpiceConfig
from weinhardt2026.analysis.analysis_coefficient_dendrogram import (
    cluster_module,
    get_module_matrices,
    split_by_sign,
)
from weinhardt2026.utils.checkpoints import load_estimator


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------

def get_criterion(
    data_path: str,
    criterion_col: str = "reward",
    participant_col: str = "participant",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Per-participant criterion plus log trial count for the volume check.

    Participant order follows ``df[participant_col].unique()``, which is the same
    order ``csv_to_dataset`` assigns integer participant indices in, so row *i*
    of the returned frame is participant index *i* of the model.
    """
    raw = pd.read_csv(data_path)
    order = list(raw[participant_col].unique())

    criterion = raw.groupby(participant_col)[criterion_col].agg(aggregate)
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
        r, p = pearsonr(criterion, log_n_trials)
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


def select_reported(table: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Per group, the largest significant node; otherwise break it up.

    Walks each group from its root. A node that is significant is reported and
    its subgroups are not. A node that is not — non-significant, invariant or
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

def presence_beta(presence: np.ndarray, predictor: np.ndarray,
                  standardize: bool = True) -> dict:
    """Logistic regression of presence on one predictor.

    With a standardized continuous predictor, beta is the change in log-odds of
    presence per SD. With a 0/1 group indicator (standardize=False), beta is the
    log odds ratio of presence in the group vs the reference. Fits that separate
    perfectly or fail to converge are returned as not estimable rather than as a
    spuriously huge effect.
    """
    empty = {"beta": np.nan, "se": np.nan, "p": np.nan, "estimable": False}

    # Quasi-separation: a group in which nobody (or everybody) has the term makes
    # the odds ratio infinite. statsmodels only flags *complete* separation, so
    # the fit would drift to a huge, meaningless beta; check the 2x2 cells instead.
    if not standardize and np.isin(predictor, (0.0, 1.0)).all():
        for level in (0.0, 1.0):
            share = presence[predictor == level].mean()
            if share in (0.0, 1.0):
                return empty

    design = sm.add_constant(zscore(predictor) if standardize else predictor)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        try:
            fit = sm.Logit(presence, design).fit(disp=0)
        except (PerfectSeparationError, PerfectSeparationWarning, np.linalg.LinAlgError):
            return empty
    if not fit.mle_retvals.get("converged", False):
        return empty
    return {"beta": fit.params[1], "se": fit.bse[1], "p": fit.pvalues[1], "estimable": True}


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
        return [{"name": criterion_col,
                 "mask": np.ones(len(criterion), dtype=bool),
                 "predictor": criterion.astype(float),
                 "standardize": True,
                 "xlabel": f"log-odds of presence per SD of {criterion_col} (95% CI)"}]

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
                          "xlabel": f"log odds ratio of presence, {label} vs {reference} (95% CI)"})
    return contrasts


def test_candidates(candidates: List[dict], contrast: dict,
                    invariant_bounds: Tuple[float, float] = (0.05, 0.95),
                    alpha: float = 0.05) -> pd.DataFrame:
    """Test every node for one contrast, correct once, apply the break-up rule.

    Presence shares and the invariance check are computed on the contrast's own
    participants, since that is the sample the regression sees.
    """
    lower, upper = invariant_bounds
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
        if lower <= row["active_all"] <= upper:
            row.update(presence_beta(presence, contrast["predictor"], contrast["standardize"]))
            row["status"] = "tested" if row["estimable"] else "not estimable"
        else:
            row.update({"beta": np.nan, "se": np.nan, "p": np.nan, "estimable": False,
                        "status": "invariant"})
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
                xlabel: str = "log-odds of presence per SD of metric (95% CI)"):
    """Beta with 95% CI per leaf, grouped by module, significant ones filled."""
    table = table.dropna(subset=["beta"]).copy()
    if table.empty:
        return

    table = table.sort_values(["module", "beta"]).reset_index(drop=True)
    labels = [textwrap.fill(f"{row['module'].replace('value_', '')} | {row[label_col]}", 46)
              for _, row in table.iterrows()]
    # Wrapped labels need vertical room, so rows grow with the tallest label.
    row_height = 0.22 * max(label.count("\n") + 1 for label in labels) + 0.14
    figure, axis = plt.subplots(figsize=(10, max(3.0, row_height * len(table))))

    palette = plt.get_cmap("tab10")
    modules = list(dict.fromkeys(table["module"]))
    color_of = {module: palette(i % 10) for i, module in enumerate(modules)}

    for position, row in table.iterrows():
        color = color_of[row["module"]]
        axis.errorbar(row["beta"], position, xerr=1.96 * row["se"], fmt="o", color=color,
                      markersize=6, capsize=3,
                      markerfacecolor=color if row["significant"] else "white")

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
    handles.append(plt.Line2D([], [], marker="o", linestyle="", color="0.3",
                              markerfacecolor="white", label="not significant (FDR)"))
    axis.legend(handles=handles, fontsize=7, frameon=False,
                loc="upper left", bbox_to_anchor=(1.02, 1.0))

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
    min_active_fraction: float = 0.05,
    n_permutations: int = 10000,
    grouping_alpha: float = 0.05,
    permutation_seed: int = 0,
    split_signs: bool = True,
    invariant_bounds: Tuple[float, float] = (0.05, 0.95),
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
            min_active_fraction=min_active_fraction,
            n_permutations=n_permutations,
            grouping_alpha=grouping_alpha,
            seed=permutation_seed,
        )
        for node in get_candidates(data["presence"], result):
            node["module"] = module
            candidates.append(node)

    tables = []
    for contrast in get_contrasts(criterion, criterion_col, criterion_type, reference,
                                  comparisons):
        table = test_candidates(candidates, contrast, invariant_bounds, alpha)
        report_contrast(table, contrast["name"], alpha)

        suffix = (criterion_col if criterion_type == "continuous"
                  else f"{criterion_col}_{contrast['name']}")
        plot_path = os.path.join(output_dir, f"betas_structural_{prefix}_{suffix}.png")
        plot_forest(table[table["reported"] & (table["status"] == "tested")], plot_path,
                    title=f"{prefix} — presence, {contrast['name']}",
                    xlabel=contrast["xlabel"])

        # One file per contrast, so runs with other comparisons never overwrite it.
        saved = table.copy()
        saved["children"] = saved["children"].map(lambda c: " ".join(map(str, c)))
        table_path = os.path.join(output_dir, f"betas_structural_{prefix}_{suffix}.csv")
        saved.to_csv(table_path, index=False)
        print(f"Saved {table_path}\nSaved {plot_path}")
        tables.append(table)

    table = pd.concat(tables, ignore_index=True)
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
    parser.add_argument("--min-active-fraction", type=float, default=0.05)
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
        min_active_fraction=args.min_active_fraction,
        n_permutations=args.n_permutations,
        grouping_alpha=args.grouping_alpha,
        split_signs=not args.no_split_signs,
        alpha=args.alpha,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()

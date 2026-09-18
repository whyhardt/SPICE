"""Standardized effect of SINDy coefficients on a behavioral criterion.

Two regressions, both on the same per-participant criterion (e.g. average
reward per block):

1. *single coefficients* — one univariate model per candidate term, giving the
   standardized slope beta of that coefficient on the criterion;
2. *clusters* — the same, but on the cluster scores discovered by
   ``analysis_coefficient_dendrogram``. A cluster score is the sign-aligned mean
   of its z-scored member coefficients, so a tie that acts as one mechanism is
   tested as one regressor instead of as several correlated ones.

Both families are Benjamini-Hochberg corrected. Because the cluster members are
correlated by construction, the cluster models are additionally run jointly, so
the partial betas show what each mechanism explains beyond the others.

Usage:

    python -m weinhardt2026.analysis.analysis_coefficient_betas \
        --model weinhardt2026/studies/dezfouli2019/params_array/spice_dezfouli2019_0.1_0.7.pkl \
        --module spice.precoded.workingmemory \
        --data weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv \
        --criterion mean_reward \
        --output-dir weinhardt2026/studies/dezfouli2019/results
"""

import argparse
import importlib
import os
import sys
import textwrap
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spice import BaseModel, SpiceConfig
from weinhardt2026.analysis.analysis_coefficient_dendrogram import (
    cluster_module,
    get_module_matrices,
)
from weinhardt2026.utils.checkpoints import load_estimator


# ---------------------------------------------------------------------------
# Criterion and regressors
# ---------------------------------------------------------------------------

def get_criterion(
    data_path: str,
    criterion_col: str = "mean_reward",
    participant_col: str = "participant",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Per-participant criterion plus the data-volume nuisance covariate.

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


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)


def cluster_scores(values: np.ndarray, result: dict,
                   include_singletons: bool = True) -> Dict[int, dict]:
    """One regressor per cluster: the sign-aligned mean of its z-scored members.

    With *include_singletons*, a term that joined no cluster enters as itself, so
    the regressors partition the kept terms — every term is represented exactly
    once, as part of a tie or on its own.
    """
    table, similarity, terms = result["table"], result["similarity"], result["terms"]
    matrix = values[:, result["keep"]]
    index_of = {term: i for i, term in enumerate(terms)}

    scores = {}
    for cluster_id, group in table.groupby("cluster"):
        members = list(group["term"])
        if len(members) < 2:
            if include_singletons:
                index = index_of[members[0]]
                scores[cluster_id] = {"score": zscore(matrix[:, index]),
                                      "members": members, "signs": [1.0],
                                      "label": members[0]}
            continue
        anchor = index_of[members[0]]
        columns, signs = [], []
        for term in members:
            index = index_of[term]
            sign = 1.0 if similarity[anchor, index] >= 0 else -1.0
            columns.append(sign * zscore(matrix[:, index]))
            signs.append(sign)
        scores[cluster_id] = {
            "score": np.mean(columns, axis=0),
            "members": members,
            "signs": signs,
            "label": " ".join(f"{'+' if s > 0 else '-'}{t}" for s, t in zip(signs, members)),
        }
    return scores


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def univariate_beta(
    regressor: np.ndarray,
    criterion: np.ndarray,
    covariate: Optional[np.ndarray] = None,
) -> dict:
    """Standardized slope of *criterion* on *regressor*, optionally adjusted."""
    x = zscore(regressor)
    y = zscore(criterion)
    if np.allclose(x, 0):
        return {"beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "r2": np.nan}

    design = sm.add_constant(np.column_stack([x] + ([zscore(covariate)] if covariate is not None else [])))
    fit = sm.OLS(y, design).fit()
    return {"beta": fit.params[1], "se": fit.bse[1], "t": fit.tvalues[1],
            "p": fit.pvalues[1], "r2": fit.rsquared}


def add_fdr(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg q-values over all rows of *table* as one family."""
    table = table.copy()
    valid = table["p"].notna()
    table["q"] = np.nan
    table["significant"] = False
    if valid.any():
        rejected, q_values, _, _ = multipletests(table.loc[valid, "p"], alpha=alpha, method="fdr_bh")
        table.loc[valid, "q"] = q_values
        table.loc[valid, "significant"] = rejected
    return table


def joint_betas(scores: Dict[str, np.ndarray], criterion: np.ndarray) -> pd.DataFrame:
    """All cluster scores in one model — partial betas, mutually adjusted."""
    if not scores:
        return pd.DataFrame()
    names = list(scores)
    design = sm.add_constant(np.column_stack([zscore(scores[n]) for n in names]))
    fit = sm.OLS(zscore(criterion), design).fit()
    return pd.DataFrame({
        "regressor": names,
        "beta_partial": fit.params[1:],
        "se": fit.bse[1:],
        "p": fit.pvalues[1:],
        "model_r2": fit.rsquared,
        "model_r2_adj": fit.rsquared_adj,
    })


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_forest(table: pd.DataFrame, output_path: str, title: str, label_col: str = "term"):
    """Beta with 95% CI per regressor, grouped by module, significant ones filled."""
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
    axis.set_xlabel("standardized beta (95% CI)")
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
    criterion_col: str = "mean_reward",
    participant_col: str = "participant",
    criterion_aggregate: str = "mean",
    n_experiments: int = 1,
    experiment: int = 0,
    polynomial_degree: int = 2,
    model_kwargs: Optional[dict] = None,
    metric: str = "abs",
    linkage_method: str = "average",
    distance_threshold: float = 0.5,
    min_active_fraction: float = 0.05,
    alpha: float = 0.05,
    adjust_for_volume: bool = True,
    include_singletons: bool = True,
    prefix: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Beta effects of single coefficients and of discovered clusters."""
    os.makedirs(output_dir, exist_ok=True)
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(model_path))[0] if model_path else "spice"

    criterion_frame = get_criterion(data_path, criterion_col, participant_col, criterion_aggregate)
    criterion = criterion_frame[criterion_col].to_numpy()
    covariate = criterion_frame["log_n_trials"].to_numpy() if adjust_for_volume else None

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
    matrices = get_module_matrices(estimator, experiment=experiment)

    term_rows, cluster_rows, joint_scores = [], [], {}
    for module, data in matrices.items():
        values = data["values"]
        result = cluster_module(
            values=values,
            presence=data["presence"],
            terms=data["terms"],
            mode="coefficients",
            metric=metric,
            linkage_method=linkage_method,
            distance_threshold=distance_threshold,
            min_active_fraction=min_active_fraction,
        )

        # --- single coefficients ---
        for term in result["terms"]:
            column = values[:, data["terms"].index(term)]
            statistics = univariate_beta(column, criterion, covariate=None)
            adjusted = univariate_beta(column, criterion, covariate=covariate)
            term_rows.append({"module": module, "term": term, **statistics,
                              "beta_adjusted": adjusted["beta"], "p_adjusted": adjusted["p"]})

        # --- clusters ---
        for cluster_id, score in cluster_scores(
                values, result, include_singletons=include_singletons).items():
            statistics = univariate_beta(score["score"], criterion, covariate=None)
            adjusted = univariate_beta(score["score"], criterion, covariate=covariate)
            name = f"{module}::cluster{cluster_id}"
            cluster_rows.append({"module": module, "cluster": cluster_id,
                                 "term": score["label"], "size": len(score["members"]),
                                 **statistics,
                                 "beta_adjusted": adjusted["beta"], "p_adjusted": adjusted["p"]})
            joint_scores[name] = score["score"]

    terms_table = add_fdr(pd.DataFrame(term_rows), alpha=alpha)
    clusters_table = add_fdr(pd.DataFrame(cluster_rows), alpha=alpha)
    joint_table = joint_betas(joint_scores, criterion)

    # --- report ---
    for label, table in (("single coefficients", terms_table), ("clusters", clusters_table)):
        print(f"\n=== beta effects on '{criterion_col}' — {label} "
              f"({int(table['significant'].sum())}/{len(table)} significant at FDR {alpha}) ===")
        ranked = table.reindex(table["beta"].abs().sort_values(ascending=False).index)
        for _, row in ranked.iterrows():
            marker = "*" if row["significant"] else " "
            print(f"  {marker} {row['module'].replace('value_', ''):24s} {row['term'][:58]:58s} "
                  f"beta={row['beta']:+.3f} p={row['p']:.4f} q={row['q']:.3f} "
                  f"beta_adj={row['beta_adjusted']:+.3f}")

    if not joint_table.empty:
        print(f"\n=== clusters, joint model (R2={joint_table['model_r2'].iloc[0]:.3f}, "
              f"adj={joint_table['model_r2_adj'].iloc[0]:.3f}) ===")
        for _, row in joint_table.iterrows():
            print(f"    {row['regressor']:44s} beta_partial={row['beta_partial']:+.3f} "
                  f"p={row['p']:.4f}")

    terms_path = os.path.join(output_dir, f"betas_terms_{prefix}_{criterion_col}.csv")
    clusters_path = os.path.join(output_dir, f"betas_clusters_{prefix}_{criterion_col}.csv")
    terms_table.to_csv(terms_path, index=False)
    clusters_table.to_csv(clusters_path, index=False)
    print(f"\nSaved {terms_path}\nSaved {clusters_path}")

    if not joint_table.empty:
        joint_path = os.path.join(output_dir, f"betas_clusters_joint_{prefix}_{criterion_col}.csv")
        joint_table.to_csv(joint_path, index=False)
        print(f"Saved {joint_path}")

    plot_forest(terms_table, os.path.join(output_dir, f"betas_terms_{prefix}_{criterion_col}.png"),
                title=f"{prefix} — single coefficients vs {criterion_col}")
    plot_forest(clusters_table,
                os.path.join(output_dir, f"betas_clusters_{prefix}_{criterion_col}.png"),
                title=f"{prefix} — cluster scores vs {criterion_col}")
    print("Saved forest plots")

    return {"terms": terms_table, "clusters": clusters_table, "joint": joint_table}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--criterion", default="mean_reward")
    parser.add_argument("--participant-col", default="participant")
    parser.add_argument("--criterion-aggregate", default="mean")
    parser.add_argument("--n-actions", type=int, default=2)
    parser.add_argument("--polynomial-degree", type=int, default=2)
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--min-active-fraction", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-volume-covariate", action="store_true")
    parser.add_argument("--clusters-only", action="store_true",
                        help="Drop single terms that joined no cluster from the cluster family")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-kwarg", action="append", default=[])
    args = parser.parse_args()

    model_kwargs = {}
    for item in args.model_kwarg:
        key, value = item.split("=", 1)
        model_kwargs[key] = {"true": True, "false": False}.get(value.lower(), value)

    imported = importlib.import_module(args.module)
    analysis_coefficient_betas(
        model_path=args.model,
        spice_class=imported.SpiceModel,
        spice_config=imported.CONFIG,
        n_actions=args.n_actions,
        data_path=args.data,
        output_dir=args.output_dir,
        criterion_col=args.criterion,
        participant_col=args.participant_col,
        criterion_aggregate=args.criterion_aggregate,
        polynomial_degree=args.polynomial_degree,
        distance_threshold=args.distance_threshold,
        min_active_fraction=args.min_active_fraction,
        alpha=args.alpha,
        adjust_for_volume=not args.no_volume_covariate,
        include_singletons=not args.clusters_only,
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    main()

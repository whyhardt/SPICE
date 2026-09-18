"""
Mechanism-level structural group-difference analysis.

`analysis_coefficients_individuals.py` asks "which raw SINDy terms are more
or less *present* in one clinical group than another" -- the presence/absence
story that motivates SPICE's structural-differences claim. This module asks
the same question one level up, on the compressed representation from
`analysis_coefficient_compression.py`: which *mechanisms* (data-discovered,
per-module NMF concepts, e.g. "value_reward_chosen: mechanism 4") are more or
less active per clinical group. Same statistical machinery (pairwise logistic
regression of presence, forest plots), reused directly from
`analysis_coefficients_individuals` rather than reimplemented.

Usage:

    from weinhardt2026.analysis.analysis_mechanism_individuals import (
        analysis_mechanism_individuals,
    )
    analysis_mechanism_individuals(
        loadings_df=loadings_df,          # from analysis_coefficient_compression
        path_data=path_data,
        reference="Control",
        criterion="diag",
        output_dir="results/compression",
    )
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from weinhardt2026.analysis.analysis_coefficients_individuals import (
    _logistic_beta, get_significance, _plot_forest, _forest_legend,
)


def prepare_mechanism_data(
    loadings_df: pd.DataFrame, path_data: str, criterion_col: str,
    df_participant_id: str = "participant", active_threshold: float = 1e-6,
) -> tuple:
    """Merge mechanism loadings with a group/criterion column from the raw data CSV.

    ``loadings_df`` must have a ``participant_index`` column (0-indexed,
    matching the SPICE model's internal participant order -- exactly what
    `analysis_coefficient_compression` returns) plus one column per
    mechanism. Maps back to original participant IDs the same way
    `analysis_coefficients_individuals.prepare` does: integer index i ->
    ``raw_df[df_participant_id].unique()[i]``, since `csv_to_dataset` builds
    that same mapping via ``enumerate(df[participant_col].unique())``.

    Returns
    -------
    df : pd.DataFrame
        One row per participant, mechanism columns + criterion column.
    mechanism_cols : list[str]
        Names of the mechanism columns.
    """
    raw_df = pd.read_csv(path_data)
    original_pids = raw_df[df_participant_id].unique()

    mechanism_cols = [c for c in loadings_df.columns if c not in ("participant_index", "experiment_index")]

    df = loadings_df.copy()
    df["participant_id"] = df["participant_index"].map(lambda i: original_pids[int(i)])

    crit_df = raw_df.groupby(df_participant_id).first().reset_index()
    crit_df = crit_df.rename(columns={df_participant_id: "participant_id"})
    crit_df = crit_df[["participant_id", criterion_col]]

    df = df.merge(crit_df, on="participant_id", how="inner")
    df = df.dropna(subset=[criterion_col])
    print(f"Mechanism data: {len(df)} participants with criterion '{criterion_col}', {len(mechanism_cols)} mechanisms.")
    return df, mechanism_cols


def run_discrete_mechanisms(
    df: pd.DataFrame, mechanism_cols: list, criterion_col: str, reference_group: str,
    output_dir: str, active_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Pairwise logistic regression of mechanism *activation* (|loading| > threshold)
    between the reference group and every other group. Mirrors
    `analysis_coefficients_individuals.run_discrete` exactly, just on
    mechanism loadings instead of raw SINDy coefficients.
    """
    groups = sorted(df[criterion_col].unique())
    if reference_group not in groups:
        raise ValueError(f"Reference group '{reference_group}' not in {groups}.")
    comparison_groups = [g for g in groups if g != reference_group]
    print(f"\nDiscrete mechanism analysis: reference='{reference_group}', comparisons={comparison_groups}")
    for g in groups:
        print(f"  {g}: n={len(df[df[criterion_col] == g])}")

    results = []
    for col in mechanism_cols:
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            continue
        presence = (np.abs(vals[mask]) > active_threshold).astype(int)
        rate = presence.mean()
        if rate == 0 or rate == 1.0:
            continue

        crit_vals = df[criterion_col].values[mask]
        result = {"mechanism": col, "n_total": int(mask.sum()), "activation_rate": rate}

        for cg in comparison_groups:
            pair_mask = (crit_vals == reference_group) | (crit_vals == cg)
            if pair_mask.sum() < 10:
                result[f"{reference_group}_vs_{cg}_beta"] = np.nan
                result[f"{reference_group}_vs_{cg}_se"] = np.nan
                result[f"{reference_group}_vs_{cg}_p"] = np.nan
                result[f"{reference_group}_vs_{cg}_sig"] = "insufficient_data"
                continue
            pair_presence = presence[pair_mask]
            is_ref = (crit_vals[pair_mask] == reference_group).astype(int)
            if pair_presence.std() == 0:
                result[f"{reference_group}_vs_{cg}_beta"] = np.nan
                result[f"{reference_group}_vs_{cg}_se"] = np.nan
                result[f"{reference_group}_vs_{cg}_p"] = np.nan
                result[f"{reference_group}_vs_{cg}_sig"] = "no_variation"
                continue
            beta, se, p_val = _logistic_beta(pair_presence, is_ref)
            result[f"{reference_group}_vs_{cg}_beta"] = beta
            result[f"{reference_group}_vs_{cg}_se"] = se
            result[f"{reference_group}_vs_{cg}_p"] = p_val
            result[f"{reference_group}_vs_{cg}_sig"] = get_significance(p_val)

        results.append(result)

    res_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    res_df.to_csv(os.path.join(output_dir, "mechanism_group_differences.csv"), index=False)

    print(f"\nAnalysed {len(res_df)} mechanisms.")
    for cg in comparison_groups:
        sig_col = f"{reference_group}_vs_{cg}_sig"
        if sig_col not in res_df.columns:
            continue
        sig = res_df[res_df[sig_col].isin(["*", "**", "***"])]
        print(f"\n{reference_group} vs {cg}: {len(sig)} significant mechanisms")
        for _, row in sig.iterrows():
            beta = row[f"{reference_group}_vs_{cg}_beta"]
            p_val = row[f"{reference_group}_vs_{cg}_p"]
            direction = f"more active in {reference_group}" if beta > 0 else f"less active in {reference_group}"
            print(f"  {row['mechanism']}: β={beta:.3f}, p={p_val:.4f} {row[sig_col]} ({direction})")

    _plot_mechanism_forest(res_df, reference_group, comparison_groups, output_dir)
    _plot_mechanism_activation_rates(res_df, df, criterion_col, reference_group, comparison_groups, output_dir, active_threshold)
    return res_df


def _plot_mechanism_forest(res_df: pd.DataFrame, ref: str, comparisons: list, output_dir: str) -> None:
    import matplotlib.pyplot as plt

    n_plots = len(comparisons)
    n_rows = max(len(res_df), 1)
    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(9 * max(n_plots, 1), max(6, n_rows * 0.3)))
    if n_plots == 1:
        axes = [axes]

    for i, cg in enumerate(comparisons):
        beta_col = f"{ref}_vs_{cg}_beta"
        se_col = f"{ref}_vs_{cg}_se"
        sig_col = f"{ref}_vs_{cg}_sig"
        if beta_col not in res_df.columns:
            continue
        valid = res_df.dropna(subset=[beta_col]).copy()
        valid = valid[valid[sig_col].isin(["***", "**", "*", "ns"])]
        df_plot = pd.DataFrame({
            "beta": valid[beta_col].values,
            "se": valid[se_col].values,
            "coefficient_clean": valid["mechanism"].values,
            "significance": valid[sig_col].values,
        })
        _plot_forest(df_plot, axes[i], title=f"{ref} vs {cg}",
                     xlabel="Effect β (>0: more active in reference)")

    _forest_legend(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mechanism_forest_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Mechanism forest plot saved.")


def _plot_mechanism_activation_rates(
    res_df: pd.DataFrame, df: pd.DataFrame, criterion_col: str, ref: str, comparisons: list,
    output_dir: str, active_threshold: float,
) -> None:
    """Bar chart of activation rate per group, for mechanisms significant in
    at least one comparison (falls back to the 8 most active mechanisms if
    none are significant). Mirrors
    `analysis_coefficients_individuals._plot_presence_rates`.
    """
    import matplotlib.pyplot as plt

    sig_mechanisms = []
    for _, row in res_df.iterrows():
        for cg in comparisons:
            sig_col = f"{ref}_vs_{cg}_sig"
            if sig_col in row and row.get(sig_col) in ["*", "**", "***"]:
                sig_mechanisms.append(row["mechanism"])
                break
    if not sig_mechanisms:
        sig_mechanisms = res_df.nlargest(8, "activation_rate")["mechanism"].tolist()

    if not sig_mechanisms:
        return

    all_groups = [ref] + comparisons
    rate_data = []
    for mech in sig_mechanisms:
        for grp in all_groups:
            vals = df[df[criterion_col] == grp][mech].dropna()
            if len(vals) > 0:
                rate_data.append({
                    "Mechanism": mech, "Group": grp,
                    "Activation_Rate": float((vals.abs() > active_threshold).mean()),
                })
    if not rate_data:
        return

    rate_df = pd.DataFrame(rate_data)
    pivot = rate_df.pivot(index="Mechanism", columns="Group", values="Activation_Rate")
    col_order = [ref] + [g for g in pivot.columns if g != ref]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("Activation rate")
    ax.set_title("Mechanism activation rates by group (significant mechanisms)")
    ax.legend(title=criterion_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mechanism_activation_rates.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Mechanism activation-rate plot saved.")


def analysis_mechanism_individuals(
    loadings_df: pd.DataFrame, path_data: str, reference: str, criterion: str,
    output_dir: str, df_participant_id: str = "participant", active_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Run the full mechanism-level structural group-difference analysis.

    Parameters
    ----------
    loadings_df : pd.DataFrame
        From `analysis_coefficient_compression` -- `participant_index` +
        one column per mechanism.
    path_data : str
        Path to the raw experiment CSV (for the group/criterion column).
    reference : str
        Reference group for pairwise comparisons (e.g. "Control").
    criterion : str
        Column name in the raw CSV holding the group label (e.g. "diag").
    output_dir : str
        Directory to save the CSV and forest plot.

    Returns
    -------
    res_df : pd.DataFrame
        One row per mechanism, with beta/SE/p-value/significance per
        reference-vs-group comparison.
    """
    df, mechanism_cols = prepare_mechanism_data(
        loadings_df, path_data, criterion, df_participant_id=df_participant_id, active_threshold=active_threshold,
    )
    return run_discrete_mechanisms(df, mechanism_cols, criterion, reference, output_dir, active_threshold=active_threshold)

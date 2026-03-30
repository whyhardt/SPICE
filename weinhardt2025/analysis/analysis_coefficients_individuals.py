"""
Unified SINDy coefficient analysis pipeline.

Extracts ensemble-averaged SINDy coefficients from a trained SPICE model,
merges them with participant-level data, and runs either:
  - discrete analysis  (e.g. diagnosis groups → odds ratios)
  - continuous analysis (e.g. age → logistic regression effect β)

Usage examples:

  # Discrete (diagnosis-based, odds ratios):
  python analysis_coefficients.py \
      --model weinhardt2025/params/dezfouli2019/spice_dezfouli2019_a0_05.pkl \
      --data  weinhardt2025/data/dezfouli2019/dezfouli2019.csv \
      --analysis discrete \
      --criterion diag \
      --reference Healthy

  # Continuous (age-based, effect sizes):
  python analysis_coefficients.py \
      --model weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl \
      --data  weinhardt2025/data/eckstein2022/eckstein2022.csv \
      --analysis continuous \
      --criterion age 
"""

import argparse
import importlib
import os
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kruskal, norm, chi2, f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# Ensure project root is importable
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


def clean_name(col):
    return (col.replace("x_", "").replace("_", " ").title())[:50]


SIG_COLORS = {"***": "#FF0000", "**": "#FFA500", "*": "#FFD700", "ns": "#999999"}


# ---------------------------------------------------------------------------
# 1. Preparation – extract coefficients
# ---------------------------------------------------------------------------

def prepare(criterion_col, data_path: str, dataset_kwargs: dict = {}, spice_model: SpiceEstimator = None, model_path: str = None, model_module: str = None, model_class: BaseRNN = None, model_config: SpiceConfig = None):
    """Load a trained SPICE model, extract ensemble-averaged SINDy coefficients
    per participant and merge with the data file.

    Parameters
    ----------
    data_path : str
        Path to the experiment data CSV.
    criterion_col : str
        Column name for the regression criterion.
    dataset_kwargs : dict
        Extra keyword arguments passed to ``csv_to_dataset``.
    spice_model : SpiceEstimator, optional
        Pre-loaded estimator.  If None, loads from *model_path*.
    model_path : str, optional
        Path to the SPICE checkpoint (.pkl).  Required when *spice_model* is None.
    model_module : str, optional
        Dotted module path (e.g. ``spice.precoded.workingmemory_rewardbinary``).
    model_class : BaseRNN, optional
        RNN class (alternative to *model_module*).
    model_config : SpiceConfig, optional
        SPICE config (required when *model_class* is given).

    Returns
    -------
    df : pd.DataFrame
        One row per participant with columns for every SINDy coefficient
        (prefixed ``x_``) and the criterion column.
    sindy_cols : list[str]
        Names of the SINDy coefficient columns.
    """
    # --- load data to infer dimensions ---
    dataset = csv_to_dataset(file=data_path, **dataset_kwargs)
    raw_df = pd.read_csv(data_path)
    
    n_actions = dataset.ys.shape[-1]
    unique_sessions = dataset.xs[..., -1].int().unique().tolist()
    n_participants = len(unique_sessions)
    
    # --- load or reuse SPICE model ---
    if spice_model is None:
        if model_path is None:
            raise ValueError("Provide either spice_model or model_path.")
        if model_module is not None and model_class is None and model_config is None:
            mod = importlib.import_module(model_module)
            rnn_class = mod.SpiceModel
            spice_config = mod.CONFIG
        elif model_module is None and model_class is not None and model_config is not None:
            rnn_class = model_class
            spice_config = model_config
        else:
            raise ValueError("You have to give either (model_module) OR (model_class AND model_config).")

        # Peek at saved checkpoint to infer ensemble_size
        _ckpt = torch.load(model_path, map_location="cpu")
        _first_module = next(iter(spice_config.library_setup))
        ensemble_size = _ckpt["model"][f"sindy_coefficients.{_first_module}"].shape[0]
        del _ckpt

        estimator = SpiceEstimator(
            spice_class=rnn_class,
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=n_participants,
            sindy_library_polynomial_degree=2,
            ensemble_size=ensemble_size,
            use_sindy=True,
        )
        estimator.load_spice(model_path)
    else:
        estimator = spice_model

    # --- extract ensemble-averaged coefficients ---
    # Returns Dict[module_name, np.ndarray] with shape (P, X, T)
    coefficients = estimator.get_sindy_coefficients(aggregate=True)
    candidate_terms = estimator.get_candidate_terms()
    modules = estimator.get_modules()
    
    sindy_cols = []
    for m in modules:
        for c in candidate_terms[m]:
            sindy_cols.append(m+"_"+c)

    # Map integer indices back to original participant labels from the CSV.
    # csv_to_dataset maps participants via enumerate(df["participant"].unique()),
    # so the integer order matches the unique() order of the raw DataFrame.
    original_pids = raw_df["participant"].unique()
    index_to_session = {i: original_pids[i] for i in range(n_participants)}

    rows = []
    for p_idx in range(n_participants):
        pid = index_to_session[p_idx]
        row = {"participant_id": pid}
        for module in modules:
            coefs = coefficients[module].detach().cpu().numpy()  # (P, X, T)
            terms = candidate_terms[module]
            # TODO: change this to respect each participant/experiment combination as a single module
            # Average across experiments (X dimension) for each participant
            p_coefs = coefs[p_idx].mean(axis=0)  # (T,)
            for t_idx, term in enumerate(terms):
                col_name = f"{module}_{term}"
                row[col_name] = float(p_coefs[t_idx])
            row[f"params_{module}"] = int(np.sum(np.abs(p_coefs) > 1e-10))
        row["total_params"] = sum(row[f"params_{m}"] for m in modules)
        rows.append(row)

    sindy_df = pd.DataFrame(rows)
    print(f"Extracted {len(sindy_cols)} SINDy coefficient columns for {len(sindy_df)} participants.")
    
    # --- build criterion column per participant from raw data ---
    crit_df = raw_df.groupby("participant").first().reset_index()
    crit_df = crit_df.rename(columns={"participant": "participant_id"})
    crit_df = crit_df[["participant_id", criterion_col]]
    
    # Merge
    df = sindy_df.merge(crit_df, on="participant_id", how="inner")
    df = df.dropna(subset=[criterion_col])
    print(f"After merge: {len(df)} participants with criterion '{criterion_col}'.")
    return df, sindy_cols


# ---------------------------------------------------------------------------
# 2a. Discrete analysis – logistic regression per group-pair → odds ratios
# ---------------------------------------------------------------------------

def _logistic_or(presence, is_reference):
    """Fit logistic regression, return (odds_ratio, p_value)."""
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(is_reference.reshape(-1, 1)).flatten()
        model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=0)
        model.fit(X.reshape(-1, 1), presence)
        or_val = np.exp(model.coef_[0][0])

        # Likelihood ratio test
        p_hat = model.predict_proba(X.reshape(-1, 1))[:, 1]
        eps = 1e-15
        ll = np.sum(
            presence * np.log(np.clip(p_hat, eps, 1 - eps))
            + (1 - presence) * np.log(np.clip(1 - p_hat, eps, 1 - eps))
        )
        p0 = presence.mean()
        ll0 = np.sum(presence * np.log(p0) + (1 - presence) * np.log(1 - p0))
        lr = -2 * (ll0 - ll)
        p_val = 1 - chi2.cdf(max(0, lr), df=1)
        return or_val, p_val
    except Exception:
        return np.nan, np.nan


def run_discrete(df, sindy_cols, criterion_col, reference_group, output_dir):
    """Pairwise logistic regression of coefficient presence between
    the reference group and every other group.  Produces odds-ratio
    bar-charts and presence-rate plots."""

    groups = sorted(df[criterion_col].unique())
    if reference_group not in groups:
        raise ValueError(
            f"Reference group '{reference_group}' not in {groups}."
        )
    comparison_groups = [g for g in groups if g != reference_group]
    print(f"\nDiscrete analysis: reference='{reference_group}', "
          f"comparisons={comparison_groups}")

    for g in groups:
        print(f"  {g}: n={len(df[df[criterion_col] == g])}")

    results = []
    for col in sindy_cols:
        vals = df[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            continue
        presence = (vals[mask] != 0).astype(int)
        rate = presence.mean()
        if rate == 0 or rate == 1.0:
            continue

        crit_vals = df[criterion_col].values[mask]
        result = {
            "coefficient": col,
            "coefficient_clean": clean_name(col),
            "n_total": int(mask.sum()),
            "presence_rate": rate,
        }

        for cg in comparison_groups:
            pair_mask = (crit_vals == reference_group) | (crit_vals == cg)
            if pair_mask.sum() < 10:
                result[f"{reference_group}_vs_{cg}_OR"] = np.nan
                result[f"{reference_group}_vs_{cg}_p"] = np.nan
                result[f"{reference_group}_vs_{cg}_sig"] = "insufficient_data"
                continue
            pair_presence = presence[pair_mask]
            is_ref = (crit_vals[pair_mask] == reference_group).astype(int)
            if pair_presence.std() == 0:
                result[f"{reference_group}_vs_{cg}_OR"] = np.nan
                result[f"{reference_group}_vs_{cg}_p"] = np.nan
                result[f"{reference_group}_vs_{cg}_sig"] = "no_variation"
                continue
            or_val, p_val = _logistic_or(pair_presence, is_ref)
            result[f"{reference_group}_vs_{cg}_OR"] = or_val
            result[f"{reference_group}_vs_{cg}_p"] = p_val
            result[f"{reference_group}_vs_{cg}_sig"] = get_significance(p_val)

        results.append(result)

    res_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    res_df.to_csv(os.path.join(output_dir, "discrete_odds_ratio_results.csv"), index=False)

    # ---- print summary ----
    print(f"\nAnalysed {len(res_df)} coefficients.")
    for cg in comparison_groups:
        sig_col = f"{reference_group}_vs_{cg}_sig"
        if sig_col not in res_df.columns:
            continue
        sig = res_df[res_df[sig_col].isin(["*", "**", "***"])]
        print(f"\n{reference_group} vs {cg}: {len(sig)} significant coefficients")
        for _, row in sig.head(5).iterrows():
            or_val = row[f"{reference_group}_vs_{cg}_OR"]
            p_val = row[f"{reference_group}_vs_{cg}_p"]
            direction = f"higher in {reference_group}" if or_val > 1 else f"lower in {reference_group}"
            print(f"  {row['coefficient_clean']}: OR={or_val:.3f}, "
                  f"p={p_val:.4f} {row[sig_col]} ({direction})")

    # ---- plots ----
    _plot_odds_ratios(res_df, reference_group, comparison_groups, output_dir)
    _plot_presence_rates(res_df, df, criterion_col, reference_group,
                         comparison_groups, output_dir)
    return res_df


def _plot_odds_ratios(res_df, ref, comparisons, output_dir):
    n_plots = len(comparisons)
    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(10 * max(n_plots, 1), 8))
    if n_plots == 1:
        axes = [axes]

    for i, cg in enumerate(comparisons):
        or_col = f"{ref}_vs_{cg}_OR"
        sig_col = f"{ref}_vs_{cg}_sig"
        if or_col not in res_df.columns:
            continue
        valid = res_df.dropna(subset=[or_col]).copy()
        valid = valid[valid[sig_col].isin(["***", "**", "*", "ns"])]
        if valid.empty:
            axes[i].text(0.5, 0.5, "No valid data", ha="center", va="center",
                         transform=axes[i].transAxes)
            continue
        valid = valid.sort_values(or_col)
        colors = valid[sig_col].map(SIG_COLORS)
        y_pos = np.arange(len(valid))
        axes[i].barh(y_pos, valid[or_col], color=colors, edgecolor="black")
        axes[i].axvline(x=1, color="black", linestyle="--", alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(valid["coefficient_clean"], fontsize=8)
        axes[i].set_xlabel(f"Odds Ratio (OR > 1: more in {ref})")
        axes[i].set_title(f"{ref} vs {cg}")
        axes[i].set_xscale("log")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SIG_COLORS[s], edgecolor="black")
               for s in ["***", "**", "*", "ns"]]
    axes[-1].legend(handles, ["p<0.001", "p<0.01", "p<0.05", "ns"],
                    title="Significance", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "odds_ratios.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Odds-ratio plot saved.")


def _plot_presence_rates(res_df, df, criterion_col, ref, comparisons, output_dir):
    # Pick significant coefficients, or top-6 by presence rate
    sig_coeffs = []
    for _, row in res_df.iterrows():
        for cg in comparisons:
            sig_col = f"{ref}_vs_{cg}_sig"
            if sig_col in row and row.get(sig_col) in ["*", "**", "***"]:
                sig_coeffs.append(row)
                break
    if not sig_coeffs:
        sig_coeffs = res_df.nlargest(6, "presence_rate").to_dict("records")
    else:
        sig_coeffs = sig_coeffs[:6]

    if not sig_coeffs:
        return

    all_groups = [ref] + comparisons
    presence_data = []
    for coef in sig_coeffs:
        col = coef["coefficient"]
        for grp in all_groups:
            data = df[df[criterion_col] == grp][col].dropna()
            if len(data) > 0:
                presence_data.append({
                    "Coefficient": coef["coefficient_clean"],
                    "Group": grp,
                    "Presence_Rate": (data != 0).mean(),
                })
    if not presence_data:
        return

    pres_df = pd.DataFrame(presence_data)
    pivot = pres_df.pivot(index="Coefficient", columns="Group", values="Presence_Rate")
    col_order = [ref] + [g for g in pivot.columns if g != ref]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Presence Rate")
    ax.set_title("Presence Rates for Significant Coefficients")
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend(title=criterion_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "presence_rates.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Presence-rate plot saved.")


# ---------------------------------------------------------------------------
# 2b. Continuous analysis – logistic regression β (effect) per coefficient
# ---------------------------------------------------------------------------

def run_continuous(df, sindy_cols, criterion_col, output_dir):
    """For each SINDy coefficient, fit logistic regression predicting
    presence/absence from the continuous criterion (e.g. age).
    Produces β bar-charts and logistic-curve plots."""

    df_clean = df[df[criterion_col].notna()].copy()
    if df_clean.empty:
        raise ValueError(f"No valid values in '{criterion_col}'.")

    crit_min = df_clean[criterion_col].min()
    crit_max = df_clean[criterion_col].max()
    print("\nContinuous analysis on '"+criterion_col+"': "+f"range [{crit_min:.1f}, {crit_max:.1f}], n={len(df_clean)}")

    scaler = StandardScaler()
    crit_std = scaler.fit_transform(df_clean[[criterion_col]]).flatten()

    results = []
    skipped = []

    for col in sindy_cols:
        vals = df_clean[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            skipped.append((col, f"<10 obs"))
            continue
        y = (vals[mask] != 0).astype(int)
        rate = y.mean()

        if rate == 0:
            skipped.append((col, "all zero"))
            continue
        if rate == 1.0:
            results.append({
                "coefficient": col,
                "coefficient_clean": clean_name(col),
                "beta": np.nan,
                "p_value": np.nan,
                "n_nonzero": int(y.sum()),
                "n_total": int(len(y)),
                "significance": "ns",
                "note": "always_present",
            })
            continue

        solver = "saga" if rate < 0.1 else "liblinear"
        max_iter = 2000 if rate < 0.1 else 1000
        model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=0)
        model.fit(crit_std[mask].reshape(-1, 1), y)
        beta = model.coef_[0][0]

        # Likelihood ratio test
        p_hat = model.predict_proba(crit_std[mask].reshape(-1, 1))[:, 1]
        eps = 1e-15
        ll = np.sum(
            y * np.log(np.clip(p_hat, eps, 1 - eps))
            + (1 - y) * np.log(np.clip(1 - p_hat, eps, 1 - eps))
        )
        p0 = y.mean()
        ll0 = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        lr = -2 * (ll0 - ll)
        p_val = 1 - chi2.cdf(max(0, lr), df=1)

        results.append({
            "coefficient": col,
            "coefficient_clean": clean_name(col),
            "beta": beta,
            "p_value": p_val,
            "n_nonzero": int(y.sum()),
            "n_total": int(len(y)),
            "significance": get_significance(p_val),
        })

    if skipped:
        print(f"Skipped {len(skipped)} coefficients (e.g. {skipped[:3]})")

    res_df = pd.DataFrame(results)
    if res_df.empty:
        raise ValueError("No valid regressions.")

    os.makedirs(output_dir, exist_ok=True)

    # Separate always-present from actual regressions
    has_note = "note" in res_df.columns
    if has_note:
        mask_reg = res_df["note"].isna()
    else:
        mask_reg = pd.Series(True, index=res_df.index)

    res_df.to_csv(os.path.join(output_dir, "continuous_effect_results_all.csv"), index=False)

    reg_df = res_df[mask_reg].copy()
    reg_df["abs_beta"] = reg_df["beta"].abs()
    reg_df = reg_df.sort_values("abs_beta", ascending=False).drop(columns="abs_beta")
    reg_df.to_csv(os.path.join(output_dir, "continuous_effect_results_variable.csv"), index=False)

    # ---- print summary ----
    never = [clean_name(c) for c, n in skipped if n == "all zero"]
    always = res_df.loc[res_df.get("note") == "always_present", "coefficient_clean"].tolist() if has_note else []
    print(f"\nAnalysed {len(reg_df)} variable coefficients.")
    if never:
        print(f"Never-present ({len(never)}): {', '.join(never[:5])}")
    if always:
        print(f"Always-present ({len(always)}): {', '.join(always[:5])}")

    sig = reg_df[reg_df["significance"].isin(["*", "**", "***"])]
    print(f"Significant effects: {len(sig)}")
    for _, row in sig.head(10).iterrows():
        direction = "increases" if row["beta"] > 0 else "decreases"
        print(f"  {row['coefficient_clean']}: β={row['beta']:.3f}, "
              f"p={row['p_value']:.4f} {row['significance']} "
              f"(presence {direction} with {criterion_col})")

    # ---- plots ----
    if not reg_df.empty:
        _plot_beta_bars(reg_df, criterion_col, output_dir)
        _plot_logistic_curves(reg_df, criterion_col, crit_min, crit_max, output_dir)
    return res_df


def _plot_beta_bars(df, criterion_col, output_dir):
    colors = df["significance"].map(SIG_COLORS)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.5), 5))
    ax.bar(range(len(df)), df["beta"], color=colors, edgecolor="black")
    ax.axhline(0, linestyle="--", color="black")
    ax.set_ylabel(f"{criterion_col} Effect (β)")
    ax.set_title(f"{criterion_col} Effect (β) by SINDy Coefficient")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["coefficient_clean"], rotation=45, ha="right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SIG_COLORS[s], edgecolor="black")
               for s in ["***", "**", "*", "ns"]]
    ax.legend(handles, ["p<0.001", "p<0.01", "p<0.05", "ns"],
              title="Significance", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beta_effects.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  β bar-plot saved.")


def _plot_logistic_curves(df, criterion_col, crit_min, crit_max, output_dir):
    n = min(len(df), 12)
    top = df.head(n)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    xs = np.linspace(crit_min, crit_max, 200)
    xs_std = (xs - xs.mean()) / xs.std()

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (_, row) in zip(axes_flat, top.iterrows()):
        p = 1 / (1 + np.exp(-row["beta"] * xs_std))
        ax.plot(xs, p, color=SIG_COLORS[row["significance"]], linewidth=2)
        ax.set_title(row["coefficient_clean"], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlabel(criterion_col)
        ax.set_ylabel("Prob(present)")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Presence probability vs {criterion_col}", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "logistic_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Logistic-curve plot saved.")


# ---------------------------------------------------------------------------
# 3. Magnitude analysis – Spearman / Kruskal-Wallis / Jonckheere-Terpstra
# ---------------------------------------------------------------------------

def jonckheere_terpstra(groups):
    k = len(groups)
    n_pairs = sum(len(a) * len(b) for i, a in enumerate(groups) for b in groups[i + 1:])
    if k < 3 or n_pairs == 0:
        return np.nan, np.nan
    U = sum(sum(y > x for y in b for x in a)
            for i, a in enumerate(groups) for b in groups[i + 1:])
    N = sum(len(g) for g in groups)
    var = (N ** 2 * (2 * N + 3) - sum(len(g) ** 2 * (2 * len(g) + 3) for g in groups)) / 72
    if var <= 0:
        return np.nan, np.nan
    z = (U - n_pairs / 2) / np.sqrt(var)
    return z, 2 * (1 - norm.cdf(abs(z)))


def run_magnitude_analysis(df, sindy_cols, criterion_col, analysis_type,
                           output_dir, group_labels=None):
    """Non-parametric magnitude analysis (Spearman, Kruskal-Wallis,
    Jonckheere-Terpstra) of SINDy coefficients vs the criterion.

    For continuous criteria the data is binned into groups first.
    For discrete criteria the groups are the unique values.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if analysis_type == "disc":
        unique_vals = sorted(df[criterion_col].unique())
        group_col = "__group__"
        df[group_col] = df[criterion_col].map({v: i for i, v in enumerate(unique_vals)})
        if group_labels is None:
            group_labels = [str(v) for v in unique_vals]
        n_groups = len(unique_vals)
    else:
        # Bin continuous criterion into quantile-based groups
        n_groups = min(6, len(df) // 10)
        if n_groups < 3:
            print("Not enough data for magnitude analysis grouping. Skipping.")
            return None
        group_col = "__group__"
        df[group_col] = pd.qcut(df[criterion_col], q=n_groups, labels=False, duplicates="drop")
        n_groups = df[group_col].nunique()
        if group_labels is None:
            bounds = pd.qcut(df[criterion_col], q=n_groups, duplicates="drop").cat.categories
            group_labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in bounds]

    keep = [c for c in sindy_cols
            if (nz := df.loc[df[c] != 0, c]).size > 10 and nz.std() > 1e-10]
    if not keep:
        print("No variable coefficients for magnitude analysis.")
        return None

    print(f"\nMagnitude analysis: {len(keep)} coefficients, {n_groups} groups")

    results = []
    for c in keep:
        nz_mask = df[c] != 0
        vals = df.loc[nz_mask, c]
        groups_vals = df.loc[nz_mask, group_col]
        raw = [vals[groups_vals == g].values for g in range(n_groups)]

        rho, p_s = spearmanr(groups_vals, vals) if vals.size > 10 else (np.nan, np.nan)
        valid = [x for x in raw if x.size and x.std() > 1e-10]
        kw, p_kw = kruskal(*valid) if len(valid) >= 3 else (np.nan, np.nan)
        z, p_jt = jonckheere_terpstra(raw)

        trend = ("Increasing" if rho > 0 else "Decreasing") if not np.isnan(rho) else "Undetermined"

        grp_stats = []
        for g_idx in range(n_groups):
            grp = vals[groups_vals == g_idx]
            grp_stats.append({
                "group": g_idx,
                "label": group_labels[g_idx] if g_idx < len(group_labels) else str(g_idx),
                "count": len(grp),
                "mean": grp.mean() if grp.size else np.nan,
                "median": grp.median() if grp.size else np.nan,
                "std": grp.std() if grp.size else np.nan,
            })

        results.append({
            "coefficient": c,
            "n_nonzero": vals.size,
            "mean": vals.mean(),
            "std": vals.std(),
            "spearman_rho": rho, "spearman_p": p_s,
            "kruskal_stat": kw, "kruskal_p": p_kw,
            "jt_z": z, "jt_p": p_jt,
            "trend": trend,
            "group_stats": grp_stats,
        })

    res = pd.DataFrame(results).sort_values("jt_p")

    # FDR correction
    for col in ["spearman_p", "kruskal_p", "jt_p"]:
        m = res[col].notna()
        if m.sum() > 0:
            res.loc[m, f"{col}_fdr"] = multipletests(res.loc[m, col], method="fdr_bh")[1]

    # Save CSV (flatten group stats)
    flat = []
    for _, row in res.iterrows():
        rec = {k: row[k] for k in res.columns if k != "group_stats"}
        for g in row["group_stats"]:
            gid = g["group"]
            for k in ("count", "mean", "median", "std"):
                rec[f"group_{gid}_{k}"] = g[k]
        flat.append(rec)
    pd.DataFrame(flat).to_csv(out / "magnitude_analysis_results.csv", index=False)

    # Heatmap of top coefficients
    top15 = res.head(min(15, len(res)))
    heat = pd.DataFrame(
        {row["coefficient"]: [g["mean"] for g in row["group_stats"]]
         for _, row in top15.iterrows()}
    ).T
    heat.columns = group_labels[:n_groups]

    plt.figure(figsize=(12, max(8, len(top15) * 0.5)))
    sns.heatmap(heat, annot=True, cmap="RdBu_r", center=0, fmt=".3f",
                cbar_kws={"label": "Mean Coefficient"})
    plt.title(f"SINDy coefficient means by {criterion_col} group (top-{len(top15)} JT)")
    plt.tight_layout()
    plt.savefig(out / "magnitude_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summary
    print(f"Significant Spearman (p<0.05):       {(res.spearman_p < 0.05).sum()}")
    print(f"Significant Kruskal-Wallis (p<0.05): {(res.kruskal_p < 0.05).sum()}")
    print(f"Significant Jonckheere-Terpstra:     {(res.jt_p < 0.05).sum()}")
    print("\nTop 10 by JT p-value:")
    print(res[["coefficient", "spearman_rho", "spearman_p", "jt_p", "trend"]]
          .head(10).to_string(index=False))

    # Clean up temp column
    if group_col in df.columns:
        df.drop(columns=[group_col], inplace=True)

    return res


def analysis_coefficients_individuals(
    criterion: str,
    analysis: str,
    path_data: str,
    spice_model: SpiceEstimator = None,
    path_model: str = None,
    model_module: str = None,
    model_class: BaseRNN = None,
    model_config: SpiceConfig = None,
    reference: str = None,
    dataset_kwargs: dict = {},
    dir_output: str = None,
    ):
    """Run the full individual-level SINDy coefficient analysis pipeline.

    Parameters
    ----------
    path_data : str
        Path to the experiment data CSV.
    criterion : str
        Column name for the regression criterion.
    analysis : str
        ``"disc"`` for discrete (odds-ratio) or ``"cont"`` for continuous.
    spice_model : SpiceEstimator, optional
        Pre-loaded estimator.  If None, loads from *path_model*.
    path_model : str, optional
        Path to the SPICE checkpoint (.pkl).  Required when *spice_model* is None.
    model_module : str, optional
        Dotted module path (e.g. ``spice.precoded.workingmemory_rewardbinary``).
    model_class : BaseRNN, optional
        RNN class (alternative to *model_module*).
    model_config : SpiceConfig, optional
        SPICE config (required when *model_class* is given).
    reference : str, optional
        Reference group for discrete analysis.  Required when *analysis* is ``"disc"``.
    dir_output : str, optional
        Output directory (default: auto-generated next to data).
    """

    if analysis == "disc" and reference is None:
        raise ValueError("--reference-group is required for discrete analysis.")

    output_dir = dir_output or os.path.join(
        os.path.dirname(path_data),
        f"analysis_{criterion}_{analysis}",
    )

    # 1. Preparation
    print("=" * 70)
    print("STEP 1: Preparing data")
    print("=" * 70)
    df, sindy_cols = prepare(
        data_path=path_data,
        criterion_col=criterion,
        spice_model=spice_model,
        model_path=path_model,
        model_module=model_module,
        model_class=model_class,
        model_config=model_config,
        dataset_kwargs=dataset_kwargs,
    )

    # 2. Regression analysis
    print("\n" + "=" * 70)
    print("STEP 2: Regression analysis")
    print("=" * 70)
    if analysis == "disc":
        res = run_discrete(df, sindy_cols, criterion,
                           reference, output_dir)
    else:
        res = run_continuous(df, sindy_cols, criterion, output_dir)

    # 3. Magnitude analysis (Spearman / KW / JT)
    print("\n" + "=" * 70)
    print("STEP 3: Magnitude analysis")
    print("=" * 70)
    run_magnitude_analysis(df, sindy_cols, criterion, analysis, output_dir)

    print(f"\nAll results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    p = argparse.ArgumentParser(
        description="Unified SINDy coefficient analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="Path to the trained SPICE model (.pkl)")
    p.add_argument("--data", required=True,
                   help="Path to the experiment data CSV")
    p.add_argument("--criterion", required=True,
                   help="Column name for the regression criterion "
                        "(e.g. 'Diagnosis', 'Age', 'diag')")
    p.add_argument("--analysis", required=True, choices=["disc", "cont"],
                   help="'discrete' for odds-ratio analysis, "
                        "'continuous' for effect-size analysis")
    p.add_argument("--model-module", default="spice.precoded.workingmemory_rewardbinary",
                   help="Name of the SPICE model module "
                        "(default: spice.precoded.workingmemory_rewardbinary)")
    p.add_argument("--reference", default=None,
                   help="Reference group for discrete analysis "
                        "(e.g. 'Healthy'). Required for --analysis discrete.")
    p.add_argument("--output", default=None,
                   help="Output directory (default: auto-generated next to data)")
    args  = p.parse_args()
    
    analysis_coefficients_individuals(
        path_data=args.data,
        criterion=args.criterion,
        analysis=args.analysis,
        path_model=args.model,
        model_module=args.model_module,
        reference=args.reference,
        dir_output=args.output,
    )

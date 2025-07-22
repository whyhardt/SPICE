from __future__ import annotations
import argparse
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

warnings.filterwarnings("ignore", category=FutureWarning)

# Updated output directory for saving plots and summaries
DEFAULT_OUT = Path(
    "/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/plots/behaviorals_by_age"
)

AGE_BINS   = [0, 10, 13, 15, 17, 24, 100]
AGE_LABELS = ["8-10", "10-13", "13-15", "15-17", "18-24", "25-30"]
PALETTE    = ["#E3F2FD", "#B6D7FF", "#7BB3F0",  
              "#4A90E2", "#2E5BBA", "#1A237E"]

BEHAVIOURS = [
    "stay_after_reward", "switch_rate", "perseveration",
    "stay_after_plus_plus", "stay_after_plus_minus",
    "stay_after_minus_plus", "stay_after_minus_minus",
    "avg_reward", "avg_rt",
]

def load_data(csv_path: str | Path = "final_df_sindy_analysis_with_metrics.csv") -> pd.DataFrame:
    """Read CSV and add six-level age group if missing."""
    df = pd.read_csv(csv_path)
    if "age_group_six" not in df.columns:
        df["age_group_six"] = (
            pd.cut(df.Age, bins=AGE_BINS,
                   labels=range(1, 7), include_lowest=True)
            .astype(int)
        )
    return df


def behavioural_boxplots(
    df: pd.DataFrame,
    out_dir: Path,
    annotate_anova: bool = True,
) -> None:
    """
    Draw all behavioural measures stratified by age group and save the figure.

    Parameters
    ----------
    annotate_anova : bool
        If True, show the ANOVA p-value box in each subplot.
    """
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    bmeas = [c for c in BEHAVIOURS if c in df.columns]
    if not bmeas:
        print("No behavioural columns found – nothing plotted.")
        return

    n_cols, n_meas = 3, len(bmeas)
    n_rows = ceil(n_meas / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        "Behavioural measures across age groups",
        fontsize=22, y=0.97,
    )
    axes = np.ravel([axes])

    for ax, beh in zip(axes, bmeas):
        data = [
            df.loc[df.age_group_six == g, beh].dropna()
            for g in range(1, 7)
        ]
        bp = ax.boxplot(
            data,
            positions=range(1, 7),
            patch_artist=True,
            showmeans=False,
        )

        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(PALETTE[i])
            box.set_alpha(0.6)
            box.set_edgecolor("black")
            box.set_linewidth(1)

        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(2.1)

        for part in ("whiskers", "caps"):
            for ln in bp[part]:
                ln.set_color("black")
                ln.set_linewidth(1)

        for i, grp in enumerate(data):
            ax.scatter(
                np.random.normal(i + 1, 0.04, size=len(grp)),
                grp, alpha=0.25, s=16, color="black", zorder=3,
            )

        if annotate_anova and any(len(g) for g in data):
            p = f_oneway(*(g for g in data if len(g)))[1]
            stars = (
                "***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else
                "ns"
            )
            ax.text(
                0.02, 0.98,
                f"ANOVA  p={p:.3g}  {stars}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                fontsize=10,
            )

        all_values = np.concatenate([grp for grp in data if len(grp)])
        if len(all_values) > 0:
            data_min, data_max = all_values.min(), all_values.max()
            data_range = data_max - data_min
            padding = data_range * 0.1 if data_range > 0 else 0.1
            if data_min >= 0:
                ax.set_ylim(0, data_max + padding)
            else:
                ax.set_ylim(data_min - padding, data_max + padding)

        ax.set_title(beh.replace("_", " ").title(), fontsize=13, pad=10)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(AGE_LABELS, rotation=45)
        ax.set_xlabel("Age group")
        ax.set_ylabel(beh.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.3)
        ax.locator_params(axis='y', nbins=6)

    for ax in axes[len(bmeas):]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      fc=PALETTE[i], ec="black", lw=1,
                      label=AGE_LABELS[i])
        for i in range(6)
    ]
    fig.legend(
        handles=handles,
        title="Age groups",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=12,
        title_fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    suffix = "with_anova" if annotate_anova else "no_anova"
    fpath = out_dir / f"behavioural_measures_by_age_{suffix}.png"
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved →", fpath)


def summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for beh in BEHAVIOURS:
        if beh not in df.columns:
            continue
        for g in range(1, 7):
            grp = df.loc[df.age_group_six == g, beh].dropna()
            if len(grp):
                rows.append(dict(
                    Measure=beh.replace("_", " ").title(),
                    Age_Group=AGE_LABELS[g - 1],
                    Mean=grp.mean(),
                    SEM=grp.std(ddof=1) / np.sqrt(len(grp)),
                    SD=grp.std(ddof=1),
                    N=len(grp),
                    Min=grp.min(),
                    Max=grp.max(),
                ))
    if rows:
        tab = pd.DataFrame(rows)
        fpath = out_dir / "behavioural_measures_by_age_summary.csv"
        tab.to_csv(fpath, index=False)
        print("Summary table written →", fpath)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate behavioural box-plots by age group "
                    "(with and without ANOVA annotation)."
    )
    parser.add_argument(
        "-o", "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for plots and summary table"
    )
    parser.add_argument(
        "--csv",
        type=str, default="final_df_sindy_analysis_with_metrics.csv",
        help="Path to input CSV file"
    )

    args = parser.parse_args()
    out: Path = args.out
    csv_path: Path | str = args.csv

    print("Output dir :", out)
    df = load_data(csv_path)
    print("Dataset shape:", df.shape)

    behavioural_boxplots(df, out, annotate_anova=True)
    behavioural_boxplots(df, out, annotate_anova=False)
    summary_table(df, out)


if __name__ == "__main__":
    main()
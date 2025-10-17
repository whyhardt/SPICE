import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def compute_block_choice_reward_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-(session, block) reward probabilities for choice==0 and choice==1.

    Expected input columns: 'session', 'block', 'choice', 'reward', 'best_action'.
    - 'choice' is assumed to be 0/1
    - 'reward' is assumed to be numeric (0/1)
    - 'best_action' indicates whether the row's action is the block's best action

    Returns a copy of df with new columns:
      - 'p_reward_choice0_block'
      - 'p_reward_choice1_block'
      - 'p_reward_for_row_choice_block' (probability for the row's actual choice)
    """
    required_cols = {"session", "block", "choice", "reward", "best_action"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric types
    df = df.copy()
    df["choice"] = df["choice"].astype(int)
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0).astype(float)
    # Normalize best_action to boolean
    if df["best_action"].dtype != bool:
        # Accept strings like 'True'/'False' or 1/0
        df["best_action"] = df["best_action"].map({True: True, False: False, "True": True, "False": False, 1: True, 0: False}).fillna(False).astype(bool)

    # Compute counts and reward sums per (session, block, choice)
    grouped = df.groupby(["session", "block", "choice"], as_index=False).agg(
        n_trials=("reward", "size"),
        reward_sum=("reward", "sum"),
    )
    grouped["p_reward"] = grouped["reward_sum"] / grouped["n_trials"].where(grouped["n_trials"] > 0, other=1)

    # Determine the best choice per (session, block) using best_action labels.
    # We compute which choice most frequently has best_action==True within each block.
    best_choice_df = (
        df.groupby(["session", "block", "choice"], as_index=False)["best_action"].mean()
        .sort_values(["session", "block", "best_action"], ascending=[True, True, False])
    )
    # Keep the top choice per (session, block)
    best_choice_df = best_choice_df.groupby(["session", "block"], as_index=False).first()
    best_choice_df = best_choice_df.rename(columns={"choice": "best_choice", "best_action": "best_action_rate"})

    # Merge empirical p_reward for both choices to facilitate snapping
    pivot_probs = (
        grouped.pivot(index=["session", "block"], columns="choice", values="p_reward")
        .rename(columns={0: "emp_p_choice0", 1: "emp_p_choice1"})
        .reset_index()
    )

    blocks = best_choice_df.merge(pivot_probs, on=["session", "block"], how="left")

    # Define allowed probabilities
    BEST_PROBS = [0.25, 0.125, 0.08]
    OTHER_PROB = 0.05

    def snap_best_prob(row: pd.Series) -> Tuple[float, float]:
        # Empirical p for the best choice in the block
        emp_best = row["emp_p_choice1"] if row["best_choice"] == 1 else row["emp_p_choice0"]
        # Snap to nearest allowed best prob
        best_prob = min(BEST_PROBS, key=lambda v: abs(v - (emp_best if pd.notnull(emp_best) else 0.0)))
        # Assign other prob
        other_prob = OTHER_PROB
        return best_prob, other_prob

    snapped = blocks.apply(snap_best_prob, axis=1, result_type="expand")
    blocks["best_prob"], blocks["other_prob"] = snapped[0], snapped[1]

    # Turn into explicit per-choice probabilities
    def assign_choice_probs(row: pd.Series) -> Tuple[float, float]:
        if row["best_choice"] == 1:
            return row["other_prob"], row["best_prob"]  # choice0, choice1
        else:
            return row["best_prob"], row["other_prob"]

    choice_probs = blocks.apply(assign_choice_probs, axis=1, result_type="expand")
    blocks["p_reward_choice0_block"], blocks["p_reward_choice1_block"] = choice_probs[0], choice_probs[1]
    blocks = blocks[["session", "block", "p_reward_choice0_block", "p_reward_choice1_block"]]

    # Merge back onto original rows
    df = df.merge(blocks, on=["session", "block"], how="left")

    # Probability corresponding to the row's actual choice
    df["p_reward_for_row_choice_block"] = df.apply(
        lambda r: r["p_reward_choice1_block"] if r["choice"] == 1 else r["p_reward_choice0_block"], axis=1
    )

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-block reward probabilities for each choice (0/1).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data" / "dezfouli2019.csv",
        help="Path to input CSV (expects columns: session, block, choice, reward, best_action).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "dezfouli2019_with_probs.csv",
        help="Path to write the augmented CSV.",
    )
    return parser.parse_args()


def main() -> Tuple[pd.DataFrame, Path]:
    args = parse_args()
    df = pd.read_csv(args.input)
    out_df = compute_block_choice_reward_probabilities(df)
    out_df.to_csv(args.output, index=False)
    return out_df, args.output


if __name__ == "__main__":
    out_df, out_path = main()
    # Print a brief confirmation and preview
    print(f"Wrote augmented CSV to: {out_path}")
    print(out_df.head(10).to_string(index=False))



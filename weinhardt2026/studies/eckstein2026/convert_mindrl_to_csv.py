"""Convert MindRL JSONL dataset to SPICE-compatible CSV.

Usage:
    python weinhardt2026/studies/eckstein2026/convert_mindrl_to_csv.py \
        --input weinhardt2026/studies/eckstein2026/data/public_train.jsonl \
        --output weinhardt2026/studies/eckstein2026/data/public_train.csv

JSONL format (one trajectory per line):
    {
        "context": {"subject_id": "sub_000848", "metadata": {"block_id": "block_003816"}, ...},
        "trials": [{"trial_index": 0, "action": 1, "reward": 8.0, "info": {"rt": 440.0}}, ...]
    }

Output CSV format (matching eckstein2024.csv):
    participant, block, trial_id, choice, reward, rt
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_mindrl_jsonl_to_csv(input_path: str, output_path: str) -> pd.DataFrame:
    rows = []

    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            ctx = data["context"]

            subject_id = int(ctx["subject_id"].replace("sub_", ""))
            block_id = int(ctx["metadata"]["block_id"].replace("block_", ""))

            for trial in data["trials"]:
                rows.append(
                    {
                        "participant": subject_id,
                        "block": block_id,
                        "trial_id": trial["trial_index"],
                        "choice": trial["action"],
                        "reward": trial["reward"] / 100.0,
                        "rt": trial["info"].get("rt"),
                    }
                )

    df = pd.DataFrame(rows)
    # Renumber blocks sequentially (1, 2, 3, ...) per participant
    df["block"] = df.groupby("participant")["block"].transform(
        lambda s: pd.factorize(s, sort=True)[0] + 1
    )
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows ({df['participant'].nunique()} participants, "
          f"{df.groupby('participant')['block'].nunique().median():.0f} median blocks/participant) "
          f"to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MindRL JSONL to SPICE CSV")
    parser.add_argument("--input", type=str,
                        default=str(Path(__file__).parent / "data" / "public_train.jsonl"))
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "data" / "public_train.csv"))
    args = parser.parse_args()

    convert_mindrl_jsonl_to_csv(args.input, args.output)

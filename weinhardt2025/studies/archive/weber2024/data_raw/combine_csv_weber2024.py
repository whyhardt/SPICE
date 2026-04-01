import re
import pandas as pd
import numpy as np
from pathlib import Path

def combine_and_process_csv_files():
    data_dir = Path(__file__).parent
    output_file = Path(__file__).parent.parent / "weber2024" / "weber2024.csv"

    # Get template columns from a baseline file
    baseline_file = next(data_dir.rglob("*baseline*.csv"))
    template_cols = pd.read_csv(baseline_file, nrows=0).columns.tolist()

    all_dfs = []

    for csv_file in data_dir.rglob("*.csv"):
        filename = csv_file.name

        # Extract participant_id from sub-xxx
        match = re.search(r"sub-(\d+)", filename)
        if match:
            participant_id = int(match.group(1))
        else:
            continue

        # Set experiment_id based on whether "infusion" is in filename
        experiment_id = 1 if "infusion" in filename else 0

        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except pd.errors.EmptyDataError:
            continue

        cols_to_keep = [c for c in template_cols if c in df.columns]
        df = df[cols_to_keep]
        df["participant"] = participant_id
        df["experiment"] = experiment_id
        df = df.iloc[::10]

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined {len(all_dfs)} files, total rows: {len(combined_df)}")

    # Rename blockID to block
    combined_df = combined_df.rename(columns={'blockID': 'block'})

    # Sort by participant, experiment to ensure proper grouping
    combined_df = combined_df.sort_values(['participant', 'experiment']).reset_index(drop=True)

    # Create incrementing block numbers per participant/experiment
    combined_df['block_inc'] = 0
    for (participant, experiment), group in combined_df.groupby(['participant', 'experiment']):
        idx = group.index
        blocks = group['block'].values
        block_inc = np.ones(len(blocks), dtype=int)
        current_block = 1
        for i in range(1, len(blocks)):
            if blocks[i] != blocks[i-1]:
                current_block += 1
            block_inc[i] = current_block
        combined_df.loc[idx, 'block_inc'] = block_inc

    # Compute reward and choice using vectorized operations where possible
    combined_df['reward'] = combined_df['totalReward'].diff()
    combined_df['shield_diff'] = combined_df['shieldRotation'].diff()
    combined_df['choice'] = np.where(combined_df['shield_diff'] == 0, 0,
                                      np.where(combined_df['shield_diff'] > 0, 1, 2))

    # Create mask for valid rows (same block, participant, experiment as previous)
    same_block = combined_df['block'] == combined_df['block'].shift(1)
    same_participant = combined_df['participant'] == combined_df['participant'].shift(1)
    same_experiment = combined_df['experiment'] == combined_df['experiment'].shift(1)
    valid_mask = same_block & same_participant & same_experiment

    # Filter to valid rows only
    result_df = combined_df[valid_mask].copy()

    # Replace block with incrementing block
    result_df['block'] = result_df['block_inc']
    result_df = result_df.drop(columns=['block_inc', 'shield_diff'])

    # Save
    result_df.to_csv(output_file, index=False)
    print(f"Processed rows: {len(result_df)}")
    print(f"\nBlocks per participant stats:")
    blocks_per_participant = result_df.groupby('participant')['block'].max()
    print(f"  Mean: {blocks_per_participant.mean():.2f}")
    print(f"  Std:  {blocks_per_participant.std():.2f}")
    print(f"  Min:  {blocks_per_participant.min()}")
    print(f"  Max:  {blocks_per_participant.max()}")
    print(f"\nChoice distribution:")
    print(result_df['choice'].value_counts())

if __name__ == "__main__":
    combine_and_process_csv_files()

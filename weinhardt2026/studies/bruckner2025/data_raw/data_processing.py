"""
Data processing pipeline for Bruckner et al. (2025) helicopter task.

Reads per-subject TSV files from data_raw/sub_*/,
consolidates into a single CSV compatible with csv_to_dataset().

Columns produced:
    participant  – subject ID (from subj_num)
    experiment   – condition label (stable_high, stable_low, push_high, push_low)
    block        – sequential block number per participant (derived from new_block)
    trial        – trial index within block
    b_t          – participant prediction (bucket position)
    x_t          – outcome (helicopter drop position)
    mu_t         – true mean of outcome distribution
    c_t          – change point indicator (0/1)
    r_t          – cumulative reward
    sigma        – outcome noise (std)
    z_t          – initial bucket position at trial start
    age_group    – age group code
"""

import os
import pandas as pd

# paths
DIR_RAW = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(os.path.dirname(DIR_RAW), 'data')
OUTPUT_FILE = os.path.join(DIR_DATA, 'bruckner2025.csv')


def process_data() -> pd.DataFrame:
    """Load all subject TSVs, derive block numbers, return unified DataFrame."""

    frames = []
    for entry in sorted(os.listdir(DIR_RAW)):
        sub_dir = os.path.join(DIR_RAW, entry)
        if not os.path.isdir(sub_dir) or not entry.startswith('sub_'):
            continue

        tsv_file = os.path.join(sub_dir, f'{entry}_task-helicopter_behav.tsv')
        if not os.path.isfile(tsv_file):
            print(f'Warning: missing TSV for {entry}, skipping.')
            continue

        df = pd.read_csv(tsv_file, sep='\t')
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # derive block number per participant from cumulative sum of new_block
    df_all['block'] = df_all.groupby('subj_num')['new_block'].cumsum().astype(int)

    # rename for csv_to_dataset compatibility
    df_all = df_all.rename(columns={'subj_num': 'participant', 'cond': 'experiment'})

    # drop the now-redundant binary indicator
    df_all = df_all.drop(columns=['new_block'])

    # reorder: identifiers first, then task variables
    col_order = [
        'participant', 'experiment', 'block', 'trial',
        'b_t', 'x_t', 'mu_t', 'c_t', 'r_t',
        'sigma', 'z_t', 'age_group',
    ]
    df_all = df_all[col_order]

    return df_all


def main():
    os.makedirs(DIR_DATA, exist_ok=True)
    df = process_data()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f'Saved {len(df)} rows ({df["participant"].nunique()} participants) → {OUTPUT_FILE}')


if __name__ == '__main__':
    main()

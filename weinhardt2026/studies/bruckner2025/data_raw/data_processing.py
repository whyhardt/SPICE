"""
Data processing pipeline for Bruckner et al. (2025) helicopter task.

Reads per-subject TSV files from data_raw/sub_*/behav/,
consolidates into a single CSV compatible with csv_to_dataset().

Columns produced:
    participant  – subject ID (from subj_num)
    experiment   – condition label (main_noPush, main_push)
    block        – sequential block number per participant (derived from new_block)
    trial        – 0-based trial index within block
    b_t          – participant prediction (bucket position)
    x_t          – outcome (helicopter drop position)
    mu_t         – true mean of outcome distribution
    c_t          – change point indicator (0/1)
    r_t          – reward value (coin type: 1 = gold/high, 0.25 = stone/low)
    catch        – bag caught flag (1 if |b_t - x_t| <= sigma/2, else 0)
    sigma        – outcome noise (std)
    v_t          – helicopter visibility (0 = hidden, 1 = visible / catch trial)
    z_t          – initial bucket position at trial start
    age_group    – age group code (1 = children, 3 = younger adults, 4 = older adults)
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

        # Raw data layout: sub_<num>/behav/sub-<num>_task-helicopter_behav.tsv
        num = entry.split('_', 1)[1]
        tsv_file = os.path.join(sub_dir, 'behav', f'sub-{num}_task-helicopter_behav.tsv')
        if not os.path.isfile(tsv_file):
            print(f'Warning: missing TSV for {entry}, skipping.')
            continue

        df = pd.read_csv(tsv_file, sep='\t')
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # derive block number per participant from cumulative sum of new_block
    df_all['block'] = df_all.groupby('subj_num')['new_block'].cumsum().astype(int)

    # derive 0-based trial index within each block
    df_all['trial'] = df_all.groupby(['subj_num', 'block']).cumcount()

    # derive catch flag: bag caught if |b_t - x_t| <= bucket_half_width (= sigma/2)
    df_all['catch'] = (abs(df_all['b_t'] - df_all['x_t']) <= df_all['sigma'] / 2).astype(int)

    # rename for csv_to_dataset compatibility
    df_all = df_all.rename(columns={'subj_num': 'participant', 'cond': 'experiment'})

    # reorder: identifiers first, then task variables
    col_order = [
        'participant', 'experiment', 'block', 'trial',
        'b_t', 'x_t', 'mu_t', 'c_t', 'r_t', 'catch',
        'sigma', 'v_t', 'z_t', 'age_group',
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

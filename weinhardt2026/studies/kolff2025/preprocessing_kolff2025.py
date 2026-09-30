"""Recode the element-level kolff2025 data into the agreed behavioural categories.

Writes data/kolff2025_categories.csv. All steps and the reasoning behind them are documented
in data/PREPROCESSING.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / 'data'
PATH_ELEMENTS = DATA_DIR / 'Original_data_with_dominance_rank - with dominance rank_ turntaking_df copy.csv'
PATH_OUT = DATA_DIR / 'kolff2025_categories.csv'

CATEGORY_OF_ELEMENT = {
    'groom': 'Groom',
    'reposition': 'Self_Reposition',
    'touch': 'Reposition_Body',
    'grab-pull limb': 'Reposition_Body',
    'push': 'Reposition_Body',
    'touch hold': 'Reposition_Body',
    'maintain contact': 'Grooming_Process',
    'directed scratch': 'Directed_Scratch',
    'hold': 'Grooming_Solicitation',
    'kiss': 'Grooming_Solicitation',
}
SOLICITATION_PREFIXES = ('present', 'raise', 'extend')

EXCLUDED_ELEMENTS = {
    'approach', 'follow', 'mount',   # socially directed actions, not negotiation
    'leave', 'move away',            # exits from the interaction, not negotiation
    'handclasp',                     # span around mutual grooming that is already coded per ape
    'peer', 'leaf groom',            # function unclear
    'display', 'drumming',           # not covered by the categorisation
}


def element_to_category(element) -> str:
    """Element name -> category. NaN (no act) stays NaN, excluded elements map to None,
    unknown elements raise."""
    if pd.isna(element):
        return np.nan
    if element in CATEGORY_OF_ELEMENT:
        return CATEGORY_OF_ELEMENT[element]
    if element.startswith(SOLICITATION_PREFIXES):
        return 'Grooming_Solicitation'
    if element in EXCLUDED_ELEMENTS:
        return None
    raise ValueError(f'element {element!r} has no category -- extend the mapping')


def recode(df: pd.DataFrame) -> pd.DataFrame:
    """Map elements to categories and delete every row that contains an excluded element."""
    df = df.copy()
    df['Category_ID1'] = df['SigAct_ID1'].map(element_to_category)
    df['Category_ID2'] = df['SigAct_ID2'].map(element_to_category)

    # `None` marks an excluded element; NaN marks "no act" and is kept
    excluded = df['SigAct_ID1'].notna() & df['Category_ID1'].isna()
    excluded |= df['SigAct_ID2'].notna() & df['Category_ID2'].isna()

    return df[~excluded]


def main():
    df = pd.read_csv(PATH_ELEMENTS).rename(columns={
        'Dominance rank_ID1': 'rank_ID1',
        'Dominance rank_ID2': 'rank_ID2',
    })

    # row order within an interaction is event order and is preserved as is
    df_out = recode(df)[[
        'interaction_id', 'community_id', 'ID1', 'ID2', 'rank_ID1', 'rank_ID2',
        'SigAct_ID1', 'SigAct_ID2', 'Category_ID1', 'Category_ID2',
    ]]
    df_out.to_csv(PATH_OUT, index=False)

    categories = pd.concat([df_out['Category_ID1'], df_out['Category_ID2']]).value_counts()
    print(f'{PATH_OUT.name}: {len(df_out)} rows, {df_out["interaction_id"].nunique()} interactions')
    print(categories.to_string())


if __name__ == '__main__':
    sys.exit(main())

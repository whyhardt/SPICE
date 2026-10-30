"""Selecting and labelling fitted SPICE checkpoints (loading them: spice.utils.analysis.loading)."""

import os
from typing import Optional

import pandas as pd
def select_lowest_bic_checkpoint(hpscan_csv: str, params_dir: str):
    """Return (path, row) of the checkpoint with the lowest BIC in a scan table."""
    scan = pd.read_csv(hpscan_csv)
    row = scan.loc[scan["BIC"].idxmin()]
    path = os.path.join(params_dir, row["path"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint referenced by {hpscan_csv} not found: {path}")
    return path, row


def participant_label_map(data_path: str, label_col: str,
                          participant_col: str = "participant") -> dict:
    """Map the 0-indexed participant index used by SPICE to a label column.

    csv_to_dataset indexes participants by ``df[participant_col].unique()``
    order, so that same order defines the mapping here.
    """
    raw = pd.read_csv(data_path)
    order = list(raw[participant_col].unique())
    return {index: raw.loc[raw[participant_col] == participant, label_col].iloc[0]
            for index, participant in enumerate(order)}


def log_trials_per_participant(data_path: str,
                               participant_col: str = "participant"):
    """log trial count per participant, in SPICE's participant index order.

    Presence regressions adjust for this: sparsity is decided by data-driven
    pruning, so a participant with more trials supports more surviving terms.
    """
    import numpy as np

    raw = pd.read_csv(data_path)
    counts = raw.groupby(participant_col).size()
    order = list(raw[participant_col].unique())
    return np.log(np.array([counts[participant] for participant in order], dtype=float))

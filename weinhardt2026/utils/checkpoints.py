"""Loading fitted SPICE checkpoints without the data they were trained on.

Analyses that only read coefficients need the model, not the dataset, so the
ensemble size and participant count are inferred from the saved tensors.
"""

import os
from typing import Optional

import pandas as pd
import torch

from spice import BaseModel, SpiceConfig, SpiceEstimator


def load_estimator(
    model_path: str,
    spice_class: BaseModel,
    spice_config: SpiceConfig,
    n_actions: int = 2,
    n_participants: Optional[int] = None,
    n_experiments: int = 1,
    polynomial_degree: int = 2,
    model_kwargs: Optional[dict] = None,
    eval_mode: bool = True,
) -> SpiceEstimator:
    """Rebuild a SpiceEstimator around a saved checkpoint.

    *n_participants* is read off the checkpoint when not given.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    first_module = next(iter(spice_config.library_setup))
    shape = checkpoint["model"][f"sindy_coefficients.{first_module}"].shape
    ensemble_size = shape[0]
    if n_participants is None:
        n_participants = shape[1]
    del checkpoint

    estimator = SpiceEstimator(
        spice_class=spice_class,
        spice_config=spice_config,
        n_actions=n_actions,
        n_participants=n_participants,
        n_experiments=n_experiments,
        sindy_library_polynomial_degree=polynomial_degree,
        ensemble_size=ensemble_size,
        use_sindy=True,
        kwargs_spice_class=model_kwargs or {},
    )
    estimator.load_spice(model_path)
    if eval_mode:
        estimator.model.eval()
    return estimator


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

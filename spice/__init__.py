"""
SPICE: Sparse and Interpretable Cognitive Equations
"""

from .resources.estimator import SpiceEstimator
from .resources.model import BaseModel
from .resources.spice_utils import SpiceConfig, SpiceDataset, SpiceSignals
from .resources.spice_training import cross_entropy_loss
from .utils.convert_dataset import csv_to_dataset, dataset_to_csv, split_data_along_sessiondim, split_data_along_timedim
from .utils.plotting import plot_session

__version__ = "0.1.0"
__all__ = [
    "SpiceEstimator",
    "SpiceConfig",
    "BaseModel",
    "SpiceDataset",
    "SpiceSignals",
    "csv_to_dataset",
    "dataset_to_csv",
    "split_data_along_sessiondim",
    "split_data_along_timedim",
    "plot_session",
    "cross_entropy_loss",
]
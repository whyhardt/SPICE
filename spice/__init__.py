"""
SPICE: Sparse and Interpretable Cognitive Equations
"""

from .resources.estimator import SpiceEstimator
from .resources.rnn import BaseRNN
from .resources.spice_utils import SpiceConfig, SpiceDataset, SpiceSignals
from .utils.convert_dataset import csv_to_dataset, dataset_to_csv, split_data_along_sessiondim, split_data_along_timedim
from .utils.plotting import plot_session
from .utils.agent import Agent, get_update_dynamics

__version__ = "0.1.0"
__all__ = [
    "SpiceEstimator",
    "SpiceConfig",
    "BaseRNN",
    "SpiceConfig",
    "SpiceDataset",
    "SpiceSignals",
    "csv_to_dataset",
    "dataset_to_csv",
    "split_data_along_sessiondim",
    "split_data_along_timedim",
    "plot_session",
    "setup_agent",
    "Agent",
    "get_update_dynamics",
]
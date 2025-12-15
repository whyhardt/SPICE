"""
SPICE: Sparse and Interpretable Cognitive Equations
"""

from .estimator import SpiceEstimator
from .precoded import (
    rescorlawagner,
    forgetting,
    learningrate,
    embedding,
    choice,
    interaction,
    workingmemory,
)
from .resources.rnn import BaseRNN
from .resources.spice_utils import SpiceConfig, SpiceDataset, SpiceSignals
from .utils.convert_dataset import convert_dataset, split_data_along_sessiondim, split_data_along_timedim
from .utils.plotting import plot_session
from .utils.setup_agents import setup_agent


__version__ = "0.1.0"
__all__ = [
    "SpiceEstimator",
    "SpiceConfig",
    "BaseRNN",
    "SpiceConfig",
    "SpiceDataset",
    "SpiceSignals",
    "convert_dataset",
    "split_data_along_sessiondim",
    "split_data_along_timedim",
    "plot_session",
    "setup_agent",
]
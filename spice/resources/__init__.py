"""
Base resources for SPICE package.
"""

from .rnn import BaseRNN
from .spice_training import fit_spice, cross_entropy_loss, mse_loss
from .spice_utils import SpiceConfig, SpiceDataset, SpiceSignals

__all__ = [
    'BaseRNN',
    'fit_spice',
    'SpiceDataset',
    'SpiceConfig',
    'SpiceSignals',
    'cross_entropy_loss',
    'mse_loss',
]
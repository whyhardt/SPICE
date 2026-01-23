"""
Utilities for SPICE package.
"""

from .convert_dataset import convert_dataset, split_data_along_sessiondim, split_data_along_timedim, reshape_data_along_participantdim
from .plotting import plot_session, plot_dynamics
from .agent import Agent, get_update_dynamics


__all__ = [
    'convert_dataset',
    'split_data_along_sessiondim',
    'split_data_along_timedim',
    'reshape_data_along_participantdim',
    'plot_session',
    'plot_dynamics',
    'Agent',
    'get_update_dynamics',
]
"""
SPICE: Sparse and Interpretable Cognitive Equations
"""

from .estimator import SpiceEstimator
from .precoded import (
    RescorlaWagnerRNN,
    ForgettingRNN,
    LearningRateRNN,
    ParticipantEmbeddingRNN,
    ChoiceRNN
)
from .resources.rnn import BaseRNN
from .resources.spice_utils import SpiceConfig, SpiceDataset, SpiceSignals

__version__ = "0.1.0"
__all__ = [
    "SpiceEstimator",
    "SpiceConfig",
    "RescorlaWagnerRNN",
    "ForgettingRNN",
    "LearningRateRNN",
    "ParticipantEmbeddingRNN",
    "ChoiceRNN",
    "BaseRNN",
    "SpiceConfig",
    "SpiceDataset",
    "SpiceSignals",
]
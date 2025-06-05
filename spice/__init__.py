"""
SPICE: Sparse and Interpretable Cognitive Equations
"""

from .estimator import SpiceEstimator, SpiceConfig
from .precoded import (
    RescorlaWagnerRNN,
    ForgettingRNN,
    LearningRateRNN,
    ParticipantEmbeddingRNN,
    Weinhardt2024RNN
)
from .resources.rnn import BaseRNN

__version__ = "0.1.0"
__all__ = [
    "SpiceEstimator",
    "SpiceConfig",
    "RescorlaWagnerRNN",
    "ForgettingRNN",
    "LearningRateRNN",
    "ParticipantEmbeddingRNN",
    "Weinhardt2024RNN",
    "BaseRNN",
] 
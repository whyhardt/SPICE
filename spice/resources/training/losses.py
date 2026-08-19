"""Loss functions and schedule helpers shared by every training stage."""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple, Union, Optional
import shutil
from torch.nn.functional import mse_loss

from ..model import BaseModel
from ..spice_utils import SpiceDataset


def cross_entropy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_smoothing=0.,
    weight: torch.Tensor = None,
    ) -> torch.Tensor:
    """Wrapper for torch's cross entropy loss which does all the reshaping when getting SpiceDataset.ys tensors as predicitons and targets.

    weight: Optional per-category loss weighting, shape (C,) or (1, C) -- passed straight
        through to torch.nn.functional.cross_entropy's own `weight` argument.
    """
    n_actions = target.shape[-1]

    prediction = prediction.reshape(-1, n_actions)
    target = torch.argmax(target.reshape(-1, n_actions), dim=1)

    if weight is not None:
        weight = weight.reshape(-1).to(device=prediction.device, dtype=prediction.dtype)

    return torch.nn.functional.cross_entropy(prediction, target, weight=weight, label_smoothing=label_smoothing)


def _setup_warmup_scaler(n_warmup_steps: int, exp_max: float = 1) -> torch.Tensor:
    """Create exponential warmup scaler for SINDy weight."""
    if n_warmup_steps <= 0:
        return None
    warmup_scaler = torch.exp(torch.linspace(0, exp_max, n_warmup_steps))
    warmup_scaler = (warmup_scaler - warmup_scaler.min()) / (warmup_scaler.max() - warmup_scaler.min()) + 1e-4
    return warmup_scaler

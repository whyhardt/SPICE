"""SPICE training pipelines.

The two-stage pipeline is split by responsibility:

  fit.py           orchestration (fit_spice)
  stage1.py        joint RNN + SINDy training against behaviour
  stage2.py        SINDy refit on the frozen RNN's trajectories (2.1 + 2.2)
  shooting.py      multi-step rollouts against those trajectories
  trajectories.py  collecting / reshaping hidden-state trajectories
  ridge.py         closed-form ridge initialization
  pruning.py       gate pruning and cross-ensemble consensus
  losses.py        loss functions and schedule helpers
  reporting.py     terminal output
"""

from torch.nn.functional import mse_loss

from .fit import fit_spice
from .losses import cross_entropy_loss
from .reporting import _get_terminal_width
from .pruning import _ensemble_ratio_test
from .ridge import _ridge_solve_sindy
from .trajectories import _vectorize_state_sequential

__all__ = ['fit_spice', 'cross_entropy_loss', 'mse_loss']

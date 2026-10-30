"""Loading fitted SPICE checkpoints for analyses.

Analyses that only read coefficients need the model, not the dataset it was trained
on: the ensemble size and participant count are read off the saved tensors.
"""

from typing import Optional

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
    **estimator_kwargs,
) -> SpiceEstimator:
    """Rebuild a SpiceEstimator around a saved checkpoint.

    *n_participants* is read off the checkpoint when not given. *model_kwargs* go to
    the SPICE model class, *estimator_kwargs* (e.g. ``n_items``, ``n_reward_features``)
    to the SpiceEstimator.
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
        **estimator_kwargs,
    )
    estimator.load_spice(model_path)
    if eval_mode:
        estimator.model.eval()
    return estimator

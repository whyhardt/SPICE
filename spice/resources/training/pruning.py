"""Pruning decisions: which concepts a unit keeps, and cross-member consensus."""

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
from .reporting import _get_terminal_width


def _ensemble_pruning(
    model: BaseModel,
    sindy_ensemble_pruning: float,
    sindy_threshold_pruning: float,
    verbose: bool,
    n_terms_pruning: int = None,
):
    """Ensemble-based closing of concept gates, with optional rate limiting.

    Operates on concept gates, not on terms: members agree on the concept
    dictionary by construction, so the cross-member vote is now about whether
    enough members *load* on a concept rather than about which terms they support.
    That makes sindy_threshold_pruning a loading magnitude measured against
    unit-norm directions -- a different scale from the raw coefficient threshold it
    used to be, so previously tuned values do not carry over.

    Args:
        n_terms_pruning: Max concepts to close per event (across all modules).
            When set, only the smallest-loading candidates are closed.
            None = no limit (close all that fail the test).
    """
    pruned = False
    module_list = list(model.submodules_rnn.keys())

    ensemble_test_kwargs = dict(
        threshold=sindy_threshold_pruning or 0.0,
        ratio=sindy_ensemble_pruning,
    )

    confidence_masks = _compute_pruning_masks(
        model,
        ensemble_test_kwargs=ensemble_test_kwargs,
        verbose=verbose,
    )

    # Update patience counters for all modules
    for module in module_list:
        mask = confidence_masks[module].to(model.device)  # (P, X, C)
        still_active = model.sindy_concept_gates[module].any(dim=0)  # (P, X, C)
        failed = ~mask & still_active

        counters = model.sindy_pruning_patience_counters[module]
        failed_e = failed.unsqueeze(0).expand_as(counters)
        counters.data = torch.where(failed_e, counters + 1, torch.zeros_like(counters))

    # Collect candidates across all modules: terms with counters >= 2
    all_prune_candidates = torch.cat(
        [model.sindy_pruning_patience_counters[m][0] >= 2 for m in module_list], dim=-1
    )  # (P, X, total_concepts)

    if all_prune_candidates.any():
        # Rate-limit: only prune the n_terms_pruning smallest-magnitude candidates
        if n_terms_pruning is not None:
            all_coeffs_abs = torch.cat([
                model.effective_loadings(m).detach().mean(dim=0)
                for m in module_list
            ], dim=-1)  # (P, X, total_concepts)

            # Set non-candidates to inf so they won't be selected
            temp_coeffs = all_coeffs_abs.clone()
            temp_coeffs[~all_prune_candidates] = torch.inf

            k = min(n_terms_pruning, all_prune_candidates.sum(dim=-1).max().item())
            if k > 0:
                _, indices = torch.topk(temp_coeffs, k, dim=-1, largest=False)
                limited_mask = torch.zeros_like(all_prune_candidates)
                limited_mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
                all_prune_candidates = all_prune_candidates & limited_mask

        # Split back to modules and apply pruning
        start_idx = 0
        for module in module_list:
            n_concepts = model.sindy_concept_gates[module].shape[-1]
            prune = all_prune_candidates[..., start_idx:start_idx + n_concepts]  # (P, X, C)
            if prune.any():
                prune_e = prune.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1)
                model.sindy_concept_gates[module] &= ~prune_e
                model.sindy_concept_loadings[module].data *= model.sindy_concept_gates[module].float()
                model.sindy_pruning_patience_counters[module].data *= (~prune_e).int()
                model.retire_dead_concepts(module)
                pruned = True
            start_idx += n_concepts

    return model, pruned


def _ensemble_ratio_test(
    coefficients: torch.Tensor,
    presence: torch.Tensor,
    threshold: float = 0.0,
    ratio: float = 0.6,
) -> torch.Tensor:
    """
    Ensemble ratio test: a term survives iff a sufficient fraction of
    ensemble members have |coefficient| > threshold.

    Args:
        coefficients: [E, P, X, terms] raw coefficient values
        presence: [E, P, X, terms] boolean presence mask
        threshold: minimum absolute coefficient value for a member to
                   count as supporting the term (default: 0.0)
        ratio: minimum fraction of ensemble members that must exceed
               the threshold for the term to survive (default: 0.6)

    Returns:
        [P, X, terms] boolean mask — True where term passes the ratio test
    """
    effective_coeffs = (coefficients * presence.float()).detach()
    E = effective_coeffs.shape[0]

    # Count how many ensemble members exceed threshold per (P, X, term)
    n_above = (effective_coeffs.abs() > threshold).float().sum(dim=0)  # [P, X, terms]

    # Term survives if ratio of members above threshold >= required ratio
    significant = (n_above / E) >= ratio

    return significant


def _compute_ensemble_masks(
    model: BaseModel,
    verbose: bool = True,
    **test_fn_kwargs,
) -> dict:
    """
    Per-participant ensemble confidence filtering via the ensemble ratio test.

    For each (participant, experiment, term), tests whether a sufficient
    fraction of ensemble members agree the term is non-zero.

    Args:
        model: trained model with SINDy coefficients
        verbose: print filtering results
        **test_fn_kwargs: keyword arguments passed to _ensemble_ratio_test
            (threshold, ratio)

    Returns:
        Dict mapping module names to [P, X, terms] boolean masks
    """
    ensemble_masks = {}

    if verbose:
        print("Ensemble confidence filtering:")

    for module in model.submodules_rnn:
        coeffs = model.effective_loadings(module).detach()
        presence = model.sindy_concept_gates[module]

        mask = _ensemble_ratio_test(coeffs, presence, **test_fn_kwargs)
        ensemble_masks[module] = mask

        if verbose:
            n_before = presence.any(dim=0).sum().item()
            n_after = mask.sum().item()
            total = mask.numel()
            print(f"\t{module}: {n_before} -> {n_after} / {total} (participant, experiment, term) slots")

    return ensemble_masks


def _compute_pruning_masks(
    model: BaseModel,
    ensemble_test_kwargs: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Ensemble ratio-test filtering per (participant, experiment, term).

    Args:
        model: trained model with SINDy coefficients
        ensemble_test_kwargs: keyword arguments passed to _ensemble_ratio_test
            (threshold, ratio). None skips ensemble filtering.
        verbose: print filtering results

    Returns:
        Dict mapping module names to [P, X, terms] boolean masks
    """
    if ensemble_test_kwargs is not None:
        return _compute_ensemble_masks(model, verbose=verbose, **ensemble_test_kwargs)

    # No ensemble filtering — term is present for (P, X) if any ensemble member has it
    return {
        module: model.sindy_concept_gates[module].any(dim=0)
        for module in model.submodules_rnn
    }

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


def compute_pruning_budgets(
    model: BaseModel,
    epochs: int,
    n_warmup_steps: int,
    sindy_pruning_frequency: Optional[int],
    override: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """How many gates / support entries a single pruning event may remove.

    Both halves of the factorization are pruned on their own budget: the gate budget is
    per unit over the concatenated concept axis (that is the axis prune_concept_gates()
    runs topk over), the support budget is per module over its (C, T) direction matrix.
    Sizing them separately is what lets each half reach 0 within the expected number of
    pruning events -- a single shared budget would starve whichever half has more items.

    Computed once before training from the *initial* factorization, so the budget stays
    a fixed rate rather than shrinking along with the structure it prunes.
    """
    if sindy_pruning_frequency is None:
        return None, None
    if override is not None:
        return int(override), int(override)

    n_pruning_events = max(1, (epochs - n_warmup_steps) // max(1, sindy_pruning_frequency))

    n_gates = sum(model.sindy_concept_gates[m].shape[-1] for m in model.submodules_rnn)
    n_support = max(
        (model.sindy_concept_support[m].numel() for m in model.submodules_rnn),
        default=0,
    )

    n_prune_z = max(1, math.ceil(n_gates / n_pruning_events))
    n_prune_v = max(1, math.ceil(n_support / n_pruning_events))
    return n_prune_z, n_prune_v


def _ensemble_pruning(
    model: BaseModel,
    sindy_ensemble_pruning: float,
    sindy_threshold_pruning: float,
    n_prune_z: int = None,
    verbose: bool = True,
):
    """Ensemble-based closing of concept gates.

    Operates on concept gates, not on terms: members agree on the concept
    dictionary by construction, so the cross-member vote is now about whether
    enough members *load* on a concept rather than about which terms they support.
    That makes sindy_threshold_pruning a loading magnitude measured against
    unit-norm directions -- a different scale from the raw coefficient threshold it
    used to be, so previously tuned values do not carry over.

    No patience counter: the ensemble vote already is the test for statistical
    uncertainty about |z| >= threshold, so a gate that fails it is a candidate now.
    Patience is the single-member substitute for that vote, and lives in
    prune_concept_gates() instead.

    The vote decides *candidacy*; n_prune_z decides how many candidates actually close
    this event. Without that rate limit a single event closes every gate that lost its
    vote at once -- and since retire_dead_concepts() zeroes the whole support row of a
    concept nobody loads on any more, the collapse propagates into V and routes around
    its budget too.

    Candidates are ranked by the *median* loading across ensemble members, so a gate
    kept alive by one outlier member does not outrank one the members agree is near
    zero. At ratio = 0.5 that ranking is the same statement as the vote itself. Gates
    whose median is exactly 0 (the majority zeroed them) tie, and topk breaks those ties
    arbitrarily -- every one of them is a legitimate prune, and the rest come back as
    candidates at the next event.
    """
    confidence_masks = _compute_pruning_masks(
        model,
        ensemble_test_kwargs=dict(
            threshold=sindy_threshold_pruning or 0.0,
            ratio=sindy_ensemble_pruning,
        ),
        verbose=verbose,
    )

    module_list = list(model.submodules_rnn.keys())

    # Concatenate over modules: n_prune_z is a per-unit budget over the *whole* concept
    # axis, the same axis prune_concept_gates() spends it on.
    all_candidates = torch.cat(
        [
            (~confidence_masks[m].to(model.device)) & model.sindy_concept_gates[m].any(dim=0)
            for m in module_list
        ],
        dim=-1,
    )                                                                   # (P, X, C_total)
    if not all_candidates.any():
        return model, False

    all_scores = torch.cat(
        [model.effective_loadings(m).detach().median(dim=0).values for m in module_list],
        dim=-1,
    )                                                                   # (P, X, C_total)
    all_scores = all_scores.clone()
    all_scores[~all_candidates] = torch.inf

    k = int(all_candidates.sum(dim=-1).max().item())
    if n_prune_z is not None:
        k = min(k, int(n_prune_z))
    if k == 0:
        return model, False

    _, indices = torch.topk(all_scores, k, dim=-1, largest=False)
    pruning_mask = torch.zeros_like(all_candidates)
    pruning_mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
    pruning_mask &= all_candidates                # Safety: only close actual candidates

    if verbose:
        print(f"\tclosing {int(pruning_mask.sum().item())} of "
              f"{int(all_candidates.sum().item())} candidate gates (budget {k}/unit)")

    pruned = False
    start_idx = 0
    for module in module_list:
        n_concepts = model.sindy_concept_gates[module].shape[-1]
        prune = pruning_mask[..., start_idx:start_idx + n_concepts]     # (P, X, C)
        start_idx += n_concepts
        if not prune.any():
            continue

        prune_e = prune.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1)
        model.sindy_concept_gates[module] &= ~prune_e
        model.sindy_concept_loadings[module].data *= model.sindy_concept_gates[module].float()
        model.retire_dead_concepts(module)
        pruned = True

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
            print(f"\t{module}: {n_before} -> {n_after} / {total} (participant, experiment, concept) slots pass the vote")

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

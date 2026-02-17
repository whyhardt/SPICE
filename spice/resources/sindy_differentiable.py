"""
Differentiable SINDy implementation in PyTorch for end-to-end training.
Based on SINDy-SHRED approach: sparse coefficients learned during RNN training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Iterable
from itertools import combinations_with_replacement


def compute_library_size(n_features: int, degree: int) -> int:
    """
    Compute the number of terms in a polynomial library.

    Args:
        n_features: Number of input features (state + controls)
        degree: Maximum polynomial degree

    Returns:
        Number of library terms
    """
    size = 0
    for d in range(degree + 1):
        # Number of d-degree monomials from n_features
        size += len(list(combinations_with_replacement(range(n_features), d)))
    return size


def get_library_feature_names(feature_names: List[str], degree: int) -> List[str]:
    """
    Generate feature names for polynomial library terms.

    Args:
        feature_names: Names of input features
        degree: Maximum polynomial degree

    Returns:
        List of library term names (e.g., ['1', 'x', 'x^2', 'x*u', ...])
    """
    n_features = len(feature_names)
    library_names = []

    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            if len(combo) == 0:
                library_names.append('1')
            else:
                term_parts = []
                feature_counts = {}
                for idx in combo:
                    feature_counts[idx] = feature_counts.get(idx, 0) + 1

                for idx, count in sorted(feature_counts.items()):
                    if count == 1:
                        term_parts.append(feature_names[idx])
                    else:
                        term_parts.append(f"{feature_names[idx]}^{count}")

                library_names.append('*'.join(term_parts))

    return library_names


def get_polynomial_degree_from_term(term: str) -> int:
    """
    Extract the polynomial degree from a library term string.

    Args:
        term: Library term string (e.g., '1', 'x', 'x^2', 'x*u', 'x^2*u', etc.)

    Returns:
        Polynomial degree of the term
    """
    # Constant term
    if term == '1':
        return 0

    # Default degree for single variable terms without exponent
    degree = 1

    # Count multiplicative terms (separated by '*')
    if '*' in term:
        parts = term.split('*')
        degree = 0
        for part in parts:
            # Check if this part has an exponent
            if '^' in part:
                # Extract the number after '^'
                exponent = int(part.split('^')[1])
                degree += exponent
            else:
                # Single variable without exponent contributes degree 1
                degree += 1
    else:
        # Single term, check for exponent
        if '^' in term:
            degree = int(term.split('^')[1])
        else:
            degree = 1

    return degree


def get_library_term_degrees(library_names: List[str]) -> List[int]:
    """
    Compute the polynomial degree for each term in the library from term names.

    Args:
        library_names: List of library term names

    Returns:
        List of degrees for each library term (e.g., [0, 1, 1, 2, 2, 2, ...])
    """
    return [get_polynomial_degree_from_term(term) for term in library_names]


def compute_polynomial_library(
    x: torch.Tensor,
    controls: torch.Tensor,
    degree: int,
    feature_names: Iterable[str],
    library: Iterable[str],
) -> torch.Tensor:
    """
    Compute polynomial library features in PyTorch (fully differentiable).

    Args:
        x: State tensor [batch, n_actions] or [batch, time, n_actions]
        controls: Control tensor [batch, n_actions, n_controls] or [batch, time, n_actions, n_controls]
        degree: Maximum polynomial degree
        include_bias: Whether to include constant term
        interaction_only: Whether to inlude only interaction terms (with '*') of polynomial degree > 1

    Returns:
        Library tensor [batch, n_actions, n_library_terms] or [batch, time, n_actions, n_library_terms]
    """

    # x: [batch, time, n_actions]
    # controls: [batch, time, n_actions, n_controls]

    x_expanded = x.unsqueeze(-1)  # [batch, time, n_actions, 1]
    if len(feature_names) == controls.shape[-1]:
        features = controls
    elif len(feature_names) > controls.shape[-1]:
        if len(feature_names) == controls.shape[-1]+1:
            features = torch.cat([x_expanded, controls], dim=-1)  # [batch, time, n_actions, 1+n_controls]
        else:
            raise ValueError(f"Size of feature names ({len(feature_names)}) must be size of control features ({controls.shape[-1]}) + 1")
    else:
        raise ValueError(f"Size of feature names ({len(feature_names)}) must be at least of size of control features ({controls.shape[-1]})")
    library_values = torch.zeros((*features.shape[:-1], len(library)), device=x.device)
    
    # compute library values from x and controls
    if degree > 0:
        # loop through the whole library to compute each term value
        for index_term, term in enumerate(library):
            if term == '1':
                value = 1
            else:
                # first split into single product terms
                terms_products = term.split('*')
                value = 1
                # loop through each product term and aggregate values
                for index_tp, tp in enumerate(terms_products):
                    # find correct feature to aggregate
                    tp_pow = tp.split('^')
                    index_feature = feature_names.index(tp_pow[0])
                    exp = float(tp_pow[-1]) if len(tp_pow) > 1 else 1
                    value *= torch.pow(features[..., index_feature], exp)
            library_values[..., index_term] += value

    # # Generate all polynomial terms
    # for d in range(degree + 1):
    #     for combo in combinations_with_replacement(range(n_features), d):
    #         if len(combo) == 0 and include_bias:
    #             # Constant term
    #             term = torch.ones_like(features[..., :1])
    #             library_terms.append(term)
    #         elif len(combo) > 0:
    #             # Check if we should skip this term based on interaction_only
    #             if interaction_only and d > 1:
    #                 # For degree > 1: only include if it's an interaction term
    #                 # An interaction has at least 2 different feature indices
    #                 unique_indices = len(set(combo))
    #                 if unique_indices == 1:
    #                     # Pure power term (e.g., x^2, x^3), skip it
    #                     continue

    #             # Polynomial term
    #             term = torch.ones_like(features[..., :1])
    #             for idx in combo:
    #                 term = term * features[..., idx:idx+1]
    #             library_terms.append(term)

    # library = torch.cat(library_terms, dim=-1)  # [batch, time, n_actions, n_library_terms]

    return library_values


def solve_sindy_stlsq(
    library: torch.Tensor,
    delta_h: torch.Tensor,
    threshold: float = 0.05,
    max_iter: int = 10,
    presence_mask: torch.Tensor = None,
    ridge_alpha: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Sequentially Thresholded Least Squares (STLSQ) for SINDy with ridge regularization.

    Solves ξ = argmin ||Θ·ξ - Δh||² + α||ξ||² with iterative hard thresholding.
    Supports batched inputs to solve K independent systems in a single lstsq call.

    Inactive columns (from presence_mask) are zeroed in the library matrix.
    Ridge regularization on those columns drives their coefficients to exactly 0,
    giving identical results to column removal while enabling batched solves
    with divergent active sets.

    Args:
        library: Library matrix Θ [N, n_terms] or batched [K, N, n_terms]
        delta_h: Target state changes [N] or batched [K, N]
        threshold: Coefficient magnitude threshold for pruning
        max_iter: Maximum STLSQ iterations
        presence_mask: Active terms mask [n_terms] or [K, n_terms] (bool). If None, all active.
        ridge_alpha: Ridge regularization strength.

    Returns:
        Tuple of (coefficients, presence_mask) with same batch dims as input.
    """
    unbatched = library.dim() == 2
    if unbatched:
        library = library.unsqueeze(0)
        delta_h = delta_h.unsqueeze(0)
        if presence_mask is not None:
            presence_mask = presence_mask.unsqueeze(0)

    K, N, n_terms = library.shape
    device = library.device
    dtype = library.dtype

    if presence_mask is None:
        active = torch.ones(K, n_terms, dtype=torch.bool, device=device)
    else:
        active = presence_mask.clone().bool()

    coeffs = torch.zeros(K, n_terms, dtype=dtype, device=device)

    for _ in range(max_iter):
        if not active.any():
            break

        # Zero out inactive columns: [K, N, n_terms]
        library_masked = library * active.unsqueeze(1).float()

        if ridge_alpha > 0:
            # Ridge via augmented least squares: [Θ_masked; √α·I] ξ ≈ [Δh; 0]
            # Full identity (not masked) ensures full rank — inactive columns
            # have zero data so regularization drives their coefficients to 0.
            sqrt_alpha = ridge_alpha ** 0.5
            ridge_rows = sqrt_alpha * torch.eye(n_terms, device=device, dtype=dtype).unsqueeze(0).expand(K, -1, -1)
            Theta_aug = torch.cat([library_masked, ridge_rows], dim=1)  # [K, N+n_terms, n_terms]
            dh_aug = torch.cat([
                delta_h,
                torch.zeros(K, n_terms, dtype=dtype, device=device),
            ], dim=1)  # [K, N+n_terms]
            result = torch.linalg.lstsq(Theta_aug, dh_aug.unsqueeze(-1))
        else:
            result = torch.linalg.lstsq(library_masked, delta_h.unsqueeze(-1))

        coeffs = result.solution.squeeze(-1)  # [K, n_terms]
        coeffs *= active.float()  # explicitly zero inactive

        # Threshold small coefficients
        small = coeffs.abs() < threshold
        newly_removed = small & active
        if not newly_removed.any():
            break
        active &= ~small

    if unbatched:
        return coeffs.squeeze(0), active.squeeze(0)
    return coeffs, active


def threshold_coefficients(coefficients: torch.Tensor, masks: torch.Tensor,
                           threshold: float) -> torch.Tensor:
    """
    Apply hard thresholding to coefficients (non-differentiable operation).

    Args:
        coefficients: Coefficient tensor [n_participants, n_library_terms]
        masks: Current mask tensor [n_participants, n_library_terms]
        threshold: Threshold value

    Returns:
        Updated mask tensor
    """
    with torch.no_grad():
        new_mask = (torch.abs(coefficients) > threshold).float()
        # Keep at least one term active (avoid degeneracy)
        for i in range(coefficients.shape[0]):
            if new_mask[i].sum() == 0:
                # Keep the largest coefficient
                max_idx = torch.argmax(torch.abs(coefficients[i]))
                new_mask[i, max_idx] = 1.0

    return new_mask


def extract_sparse_coefficients(coefficients: torch.Tensor, masks: torch.Tensor,
                                feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract sparse coefficients and corresponding feature names.

    Args:
        coefficients: Coefficient tensor [n_library_terms]
        masks: Mask tensor [n_library_terms]
        feature_names: All library feature names

    Returns:
        Sparse coefficients array and corresponding feature names
    """
    coef = (coefficients * masks).detach().cpu().numpy()
    active_idx = np.where(masks.cpu().numpy() > 0)[0]

    sparse_coef = coef[active_idx]
    sparse_names = [feature_names[i] for i in active_idx]

    return sparse_coef, sparse_names

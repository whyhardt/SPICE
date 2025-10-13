"""
Differentiable SINDy implementation in PyTorch for end-to-end training.
Based on SINDy-SHRED approach: sparse coefficients learned during RNN training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
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


def compute_polynomial_library(
    x: torch.Tensor,
    controls: torch.Tensor,
    degree: int,
    include_bias: bool = True
) -> torch.Tensor:
    """
    Compute polynomial library features in PyTorch (fully differentiable).

    Args:
        x: State tensor [batch, n_actions] or [batch, time, n_actions]
        controls: Control tensor [batch, n_actions, n_controls] or [batch, time, n_actions, n_controls]
        degree: Maximum polynomial degree
        include_bias: Whether to include constant term

    Returns:
        Library tensor [batch, n_actions, n_library_terms] or [batch, time, n_actions, n_library_terms]
    """
    # Handle different input shapes
    if x.dim() == 2:  # [batch, n_actions]
        x = x.unsqueeze(1)  # [batch, 1, n_actions]
        controls = controls.unsqueeze(1) if controls.dim() == 3 else controls
        squeeze_time = True
    else:
        squeeze_time = False

    # Combine state and controls
    if controls.dim() == 3:  # [batch, n_actions, n_controls]
        controls = controls.unsqueeze(1)  # [batch, 1, n_actions, n_controls]

    # x: [batch, time, n_actions]
    # controls: [batch, time, n_actions, n_controls]

    x_expanded = x.unsqueeze(-1)  # [batch, time, n_actions, 1]
    features = torch.cat([x_expanded, controls], dim=-1)  # [batch, time, n_actions, 1+n_controls]

    n_features = features.shape[-1]
    library_terms = []

    # Generate all polynomial terms
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            if len(combo) == 0 and include_bias:
                # Constant term
                term = torch.ones_like(features[..., :1])
            elif len(combo) > 0:
                # Polynomial term
                term = torch.ones_like(features[..., :1])
                for idx in combo:
                    term = term * features[..., idx:idx+1]
            else:
                continue

            library_terms.append(term)

    library = torch.cat(library_terms, dim=-1)  # [batch, time, n_actions, n_library_terms]

    if squeeze_time:
        library = library.squeeze(1)  # [batch, n_actions, n_library_terms]

    return library


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

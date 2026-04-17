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


def build_library_structure(n_features: int, degree: int) -> dict:
    """Build multiplication table for recursive polynomial expansion.

    Generates the term structure used for unfolding polynomial layer weights
    into monomial-basis coefficients. Term ordering matches
    ``combinations_with_replacement`` (same as ``get_library_feature_names``).

    Args:
        n_features: Number of input features (state + controls)
        degree: Maximum polynomial degree

    Returns:
        dict with keys:
            terms: list of tuples (sorted feature index multisets)
            mult_table: (n_terms, n_features) long tensor — maps
                (term, feature) to product term index (-1 if exceeds degree)
            linear_indices: (n_features,) long tensor — indices of degree-1 terms
            bias_index: int — index of the constant term
            n_terms: int — total number of library terms
    """
    terms = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            terms.append(combo)

    term_to_idx = {term: idx for idx, term in enumerate(terms)}
    n_terms = len(terms)

    # Build multiplication table: mult_table[t, f] = index of (term_t * x_f)
    # -1 means the product exceeds the maximum degree
    mult_table = torch.full((n_terms, n_features), -1, dtype=torch.long)
    for t_idx, term in enumerate(terms):
        if len(term) < degree:
            for f in range(n_features):
                product = tuple(sorted(term + (f,)))
                if product in term_to_idx:
                    mult_table[t_idx, f] = term_to_idx[product]

    bias_index = term_to_idx[()]
    linear_indices = torch.tensor(
        [term_to_idx[(f,)] for f in range(n_features)], dtype=torch.long
    )

    return {
        'terms': terms,
        'mult_table': mult_table,
        'linear_indices': linear_indices,
        'bias_index': bias_index,
        'n_terms': n_terms,
    }


def precompute_library_structure(
    feature_names: List[str],
    library: List[str],
) -> dict:
    """
    Parse library term strings once and cache the numeric structure.

    Groups terms by type for vectorized computation:
        - bias: constant '1' terms
        - linear: single feature with exponent 1 (just indexing, no pow)
        - power: single feature with exponent != 1
        - interaction: product of two or more features (each with exp 1)
        - general: anything else (mixed exponents in products)
    """
    bias_indices = []

    # Vectorizable groups: (library_index, feature_index) pairs
    linear_lib_idx = []
    linear_feat_idx = []

    power_terms = []       # (lib_idx, feat_idx, exponent)
    interaction_terms = [] # (lib_idx, [feat_idx1, feat_idx2, ...])
    general_terms = []     # (lib_idx, [(feat_idx, exp), ...])

    for index_term, term in enumerate(library):
        if term == '1':
            bias_indices.append(index_term)
            continue

        factors = []
        for tp in term.split('*'):
            tp_pow = tp.split('^')
            feat_idx = feature_names.index(tp_pow[0])
            exp = float(tp_pow[-1]) if len(tp_pow) > 1 else 1.0
            factors.append((feat_idx, exp))

        if len(factors) == 1:
            fi, exp = factors[0]
            if exp == 1.0:
                linear_lib_idx.append(index_term)
                linear_feat_idx.append(fi)
            else:
                power_terms.append((index_term, fi, exp))
        else:
            # Check if all exponents are 1 (pure interaction)
            if all(exp == 1.0 for _, exp in factors):
                interaction_terms.append((index_term, [fi for fi, _ in factors]))
            else:
                general_terms.append((index_term, factors))

    return {
        'n_terms': len(library),
        'bias_indices': bias_indices,
        'linear_lib_idx': linear_lib_idx,
        'linear_feat_idx': linear_feat_idx,
        'power_terms': power_terms,
        'interaction_terms': interaction_terms,
        'general_terms': general_terms,
    }


# Module-level cache: (tuple(feature_names), tuple(library)) -> precomputed structure
_library_structure_cache: dict = {}


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
        x: State tensor [W, B*E, I]
        controls: Control tensor [W, B*E, I, n_controls]
        degree: Maximum polynomial degree
        feature_names: Names of input features
        library: Library term strings

    Returns:
        Library tensor [W, B*E, I, n_library_terms]
    """

    x_expanded = x.unsqueeze(-1)
    if len(feature_names) == controls.shape[-1]:
        features = controls
    elif len(feature_names) > controls.shape[-1]:
        if len(feature_names) == controls.shape[-1]+1:
            features = torch.cat([x_expanded, controls], dim=-1)
        else:
            raise ValueError(f"Size of feature names ({len(feature_names)}) must be size of control features ({controls.shape[-1]}) + 1")
    else:
        raise ValueError(f"Size of feature names ({len(feature_names)}) must be at least of size of control features ({controls.shape[-1]})")

    # Look up or build the cached term structure (avoids per-call string parsing)
    cache_key = (tuple(feature_names), tuple(library))
    if cache_key not in _library_structure_cache:
        _library_structure_cache[cache_key] = precompute_library_structure(
            list(feature_names), list(library),
        )
    structure = _library_structure_cache[cache_key]

    library_values = torch.zeros((*features.shape[:-1], structure['n_terms']), device=x.device)

    if degree > 0:
        # Bias terms: single write
        for idx in structure['bias_indices']:
            library_values[..., idx] = 1.0

        # Linear terms: one batched gather (e.g. x, c1, c2, c3 → 1 kernel)
        if structure['linear_lib_idx']:
            library_values[..., structure['linear_lib_idx']] = features[..., structure['linear_feat_idx']]

        # Power terms: individual pow calls (rare — only x^2, c^3, etc.)
        for lib_idx, feat_idx, exp in structure['power_terms']:
            library_values[..., lib_idx] = torch.pow(features[..., feat_idx], exp)

        # Interaction terms (exp=1 products): vectorized where possible
        # Group by number of factors for batched computation
        if structure['interaction_terms']:
            # Degree-2 interactions: one gather-multiply (e.g. x*c1, x*c2, c1*c2 → 2 kernels)
            deg2_lib = [t[0] for t in structure['interaction_terms'] if len(t[1]) == 2]
            deg2_left = [t[1][0] for t in structure['interaction_terms'] if len(t[1]) == 2]
            deg2_right = [t[1][1] for t in structure['interaction_terms'] if len(t[1]) == 2]
            if deg2_lib:
                library_values[..., deg2_lib] = features[..., deg2_left] * features[..., deg2_right]

            # Higher-degree interactions: loop (rare with degree≤2)
            for lib_idx, feat_indices in structure['interaction_terms']:
                if len(feat_indices) != 2:
                    val = features[..., feat_indices[0]]
                    for fi in feat_indices[1:]:
                        val = val * features[..., fi]
                    library_values[..., lib_idx] = val

        # General terms (mixed exponents in products): loop
        for lib_idx, factors in structure['general_terms']:
            fi0, exp0 = factors[0]
            val = torch.pow(features[..., fi0], exp0) if exp0 != 1.0 else features[..., fi0]
            for fi, exp in factors[1:]:
                val = val * (torch.pow(features[..., fi], exp) if exp != 1.0 else features[..., fi])
            library_values[..., lib_idx] = val

    return library_values


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

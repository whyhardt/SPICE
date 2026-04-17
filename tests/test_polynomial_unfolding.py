"""
Tests for polynomial unfolding correctness.

Verifies that for random weights at degree 1, 2, 3:
  library @ unfold() matches W_out @ polynomial_layer(x) + b_out

Also verifies that forward_polynomial() matches the standard forward().
"""

import torch
import pytest
from spice.resources.model import EnsembleRNNModule
from spice.resources.sindy_differentiable import build_library_structure


# --------------------------------------------------------------------------
# Test build_library_structure
# --------------------------------------------------------------------------

def test_build_library_structure_degree1():
    """Degree 1 with 2 features: terms = [1, x0, x1]."""
    lib = build_library_structure(n_features=2, degree=1)
    assert lib['n_terms'] == 3
    assert lib['bias_index'] == 0
    assert lib['linear_indices'].tolist() == [1, 2]
    assert lib['terms'] == [(), (0,), (1,)]
    mt = lib['mult_table']
    # Bias term (degree 0) can be multiplied: () * x0 = (0,) = idx 1, () * x1 = (1,) = idx 2
    assert mt[0, 0].item() == 1
    assert mt[0, 1].item() == 2
    # Degree-1 terms: multiplying exceeds degree
    assert (mt[1:] == -1).all()


def test_build_library_structure_degree2():
    """Degree 2 with 2 features: terms = [1, x0, x1, x0^2, x0*x1, x1^2]."""
    lib = build_library_structure(n_features=2, degree=2)
    assert lib['n_terms'] == 6
    assert lib['bias_index'] == 0
    assert lib['linear_indices'].tolist() == [1, 2]
    assert lib['terms'] == [(), (0,), (1,), (0, 0), (0, 1), (1, 1)]

    # Check multiplication table for degree-0 and degree-1 terms
    mt = lib['mult_table']
    # () * x0 = (0,) = idx 1
    assert mt[0, 0].item() == 1
    # () * x1 = (1,) = idx 2
    assert mt[0, 1].item() == 2
    # (0,) * x0 = (0,0) = idx 3
    assert mt[1, 0].item() == 3
    # (0,) * x1 = (0,1) = idx 4
    assert mt[1, 1].item() == 4
    # (1,) * x0 = (0,1) = idx 4
    assert mt[2, 0].item() == 4
    # (1,) * x1 = (1,1) = idx 5
    assert mt[2, 1].item() == 5
    # Degree-2 terms: multiplication exceeds degree
    assert (mt[3:] == -1).all()


def test_build_library_structure_degree3():
    """Degree 3 with 2 features should have 10 terms."""
    lib = build_library_structure(n_features=2, degree=3)
    # C(2+0-1,0) + C(2+1-1,1) + C(2+2-1,2) + C(2+3-1,3) = 1+2+3+4 = 10
    assert lib['n_terms'] == 10


# --------------------------------------------------------------------------
# Test unfold_polynomial_coefficients
# --------------------------------------------------------------------------

def _make_rnn_module(input_size, degree, ensemble_size=2):
    """Create an EnsembleRNNModule with random weights."""
    torch.manual_seed(42)
    module = EnsembleRNNModule(
        ensemble_size=ensemble_size,
        input_size=input_size,
        embedding_size=0,
        dropout=0.,
        compiled_forward=False,
        polynomial_degree=degree,
    )
    return module


def _verify_unfolding(module, input_size, degree, atol=1e-5):
    """Verify that library @ unfold() matches the direct polynomial computation.

    For random input x, computes:
    1. Direct: w_out @ [Π_d (W_d @ x + b_d) / sqrt(D)] + b_out
    2. Unfolded: Σ_t θ_t * monomial_t(x)

    Checks they match to atol.
    """
    E = module.ensemble_size
    n_features = input_size + 1  # state + controls
    B = 5  # batch size

    # Random features: [state, controls]
    x = torch.randn(E, B, n_features) * 0.5

    # --- Direct computation ---
    # Polynomial projection
    gi = module.projection(x)  # (E, B, proj_size)
    # Output projection: n = w_out @ gi + b_out
    direct = torch.einsum('ego,ebo->ebg', module.weight_n, gi) + module.bias_n.unsqueeze(1)
    direct = direct.squeeze(-1)  # (E, B)

    # --- Unfolded computation ---
    theta = module.unfold_polynomial_coefficients()  # (E, n_terms)
    library = module._compute_library(x)  # (E, B, n_terms)
    unfolded = torch.einsum('ebt,et->eb', library, theta)  # (E, B)

    # Compare
    assert torch.allclose(direct, unfolded, atol=atol), (
        f"Unfolding mismatch for degree={degree}, input_size={input_size}.\n"
        f"Max abs diff: {(direct - unfolded).abs().max().item():.2e}\n"
        f"Direct: {direct[0, :3]}\n"
        f"Unfolded: {unfolded[0, :3]}"
    )


@pytest.mark.parametrize("input_size,degree", [
    (1, 1), (2, 1), (3, 1),   # degree 1 (linear)
    (1, 2), (2, 2), (3, 2),   # degree 2 (bilinear)
    (1, 3), (2, 3),           # degree 3 (trilinear)
])
def test_unfolding_correctness(input_size, degree):
    """Unfolded polynomial matches direct computation."""
    module = _make_rnn_module(input_size, degree)
    _verify_unfolding(module, input_size, degree)


# --------------------------------------------------------------------------
# Test forward_polynomial matches standard forward (no mask)
# --------------------------------------------------------------------------

def _verify_forward_match(input_size, degree, within_ts=1, atol=1e-4):
    """Verify that forward_polynomial without mask matches standard forward."""
    module = _make_rnn_module(input_size, degree)
    E = module.ensemble_size
    B = 3
    I = 2
    W = within_ts
    F = input_size

    inputs = torch.randn(W, E, B, I, F) * 0.3
    state = torch.randn(W, E, B, I) * 0.1

    # Standard forward
    out_standard = module._uncompiled_forward(inputs, state)

    # Polynomial forward (no mask)
    out_polynomial = module._uncompiled_forward_polynomial(inputs, state)

    assert torch.allclose(out_standard, out_polynomial, atol=atol), (
        f"Forward mismatch for degree={degree}, input_size={input_size}, W={within_ts}.\n"
        f"Max abs diff: {(out_standard - out_polynomial).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("input_size,degree", [
    (1, 1), (2, 1),
    (1, 2), (2, 2),
    (1, 3),
])
def test_forward_polynomial_matches_standard(input_size, degree):
    """forward_polynomial without mask should match standard forward."""
    _verify_forward_match(input_size, degree, within_ts=1)


@pytest.mark.parametrize("input_size,degree", [
    (1, 2), (2, 2),
])
def test_forward_polynomial_matches_standard_multi_step(input_size, degree):
    """forward_polynomial should match standard forward for multiple within-trial steps."""
    _verify_forward_match(input_size, degree, within_ts=3)


# --------------------------------------------------------------------------
# Test masking behavior
# --------------------------------------------------------------------------

def test_masked_coefficient_zero_contribution():
    """Masked coefficients should produce zero contribution."""
    module = _make_rnn_module(input_size=1, degree=2)
    E = module.ensemble_size
    n_terms = module._n_library_terms

    # Set larger weights so masking has a visible effect
    with torch.no_grad():
        for w in module.projection.weights:
            w.normal_(0, 0.5)
        module.weight_n.normal_(0, 0.5)

    inputs = torch.randn(1, E, 3, 2, 1) * 0.5
    state = torch.randn(1, E, 3, 2) * 0.5

    # Full mask (no masking)
    mask_full = torch.ones(E, n_terms)
    out_full = module._uncompiled_forward_polynomial(inputs, state, mask=mask_full)

    # Mask out all terms except bias
    mask_bias_only = torch.zeros(E, n_terms)
    mask_bias_only[:, module._bias_index] = 1.0
    out_bias_only = module._uncompiled_forward_polynomial(inputs, state, mask=mask_bias_only)

    # With only bias active, the update is state-independent → outputs should differ
    assert not torch.allclose(out_full, out_bias_only, atol=1e-3), (
        "Full and bias-only outputs should differ for non-trivial inputs."
    )


def test_gradient_flows_through_unfolding():
    """Gradients should flow through the unfolded coefficients back to RNN weights."""
    module = _make_rnn_module(input_size=1, degree=2)
    E = module.ensemble_size

    inputs = torch.randn(1, E, 2, 2, 1) * 0.3
    state = torch.randn(1, E, 2, 2) * 0.1

    out = module._uncompiled_forward_polynomial(inputs, state)
    loss = out.sum()
    loss.backward()

    # Check gradients exist for all projection weights
    for w in module.projection.weights:
        assert w.grad is not None, "No gradient for projection weight"
        assert w.grad.abs().sum() > 0, "Zero gradient for projection weight"
    for b in module.projection.biases:
        assert b.grad is not None, "No gradient for projection bias"
    assert module.weight_n.grad is not None, "No gradient for output weight"
    assert module.bias_n.grad is not None, "No gradient for output bias"


# --------------------------------------------------------------------------
# Test per-participant unfolding (with hypernetwork)
# --------------------------------------------------------------------------

def test_per_participant_unfolding():
    """Different embeddings should produce different unfolded coefficients."""
    torch.manual_seed(42)
    module = EnsembleRNNModule(
        ensemble_size=2,
        input_size=1,
        embedding_size=8,
        dropout=0.,
        compiled_forward=False,
        polynomial_degree=2,
    )

    E = 2
    B = 4
    I = 2

    # Set non-zero hypernetwork output weights so offsets are meaningful
    with torch.no_grad():
        module.hypernet_out.weight.normal_(0, 0.1)

    embedding = torch.randn(E, B, 8) * 0.5
    offsets = module.precompute_offsets(embedding, n_items=I)

    # Unfold with offsets: should give (E, B*I, n_terms)
    theta_pp = module.unfold_polynomial_coefficients(offsets)
    assert theta_pp.dim() == 3, f"Expected 3D tensor, got {theta_pp.dim()}D"
    assert theta_pp.shape == (E, B * I, module._n_library_terms)

    # Different participants should have different coefficients
    # Check first two participants (items 0 and I are different participants)
    assert not torch.allclose(theta_pp[0, 0], theta_pp[0, I], atol=1e-6), (
        "Per-participant coefficients should differ across participants"
    )


def test_forward_polynomial_with_offsets():
    """forward_polynomial should work with hypernetwork offsets."""
    torch.manual_seed(42)
    module = EnsembleRNNModule(
        ensemble_size=2,
        input_size=1,
        embedding_size=8,
        dropout=0.,
        compiled_forward=False,
        polynomial_degree=2,
    )

    E = 2
    B = 3
    I = 2
    W = 1
    F = 1

    inputs = torch.randn(W, E, B, I, F) * 0.3
    state = torch.randn(W, E, B, I) * 0.1
    embedding = torch.randn(E, B, 8) * 0.5
    offsets = module.precompute_offsets(embedding, n_items=I)

    # Standard forward with offsets
    out_standard = module._uncompiled_forward(inputs, state, precomputed_offsets=offsets)

    # Polynomial forward with offsets (no mask)
    out_polynomial = module._uncompiled_forward_polynomial(inputs, state, precomputed_offsets=offsets)

    assert torch.allclose(out_standard, out_polynomial, atol=1e-4), (
        f"Forward with offsets mismatch.\n"
        f"Max abs diff: {(out_standard - out_polynomial).abs().max().item():.2e}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

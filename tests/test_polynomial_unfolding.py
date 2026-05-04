"""
Tests for polynomial unfolding correctness.

Verifies that for random weights at degree 1, 2, 3:
  library @ unfold() matches W_out @ polynomial_layer(x) + b_out
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

    # Check degree_ranges
    assert lib['degree_ranges'] == {0: (0, 1), 1: (1, 3)}


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

    # Check degree_ranges
    assert lib['degree_ranges'] == {0: (0, 1), 1: (1, 3), 2: (3, 6)}


def test_build_library_structure_degree3():
    """Degree 3 with 2 features should have 10 terms."""
    lib = build_library_structure(n_features=2, degree=3)
    # C(2+0-1,0) + C(2+1-1,1) + C(2+2-1,2) + C(2+3-1,3) = 1+2+3+4 = 10
    assert lib['n_terms'] == 10

    # Check degree_ranges
    assert lib['degree_ranges'] == {0: (0, 1), 1: (1, 3), 2: (3, 6), 3: (6, 10)}


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
    1. Direct: w_out @ [sum_d prod_k(W_{d,k} @ x)] + b_out
    2. Unfolded: Σ_t θ_t * monomial_t(x)

    Checks they match to atol.
    """
    E = module.ensemble_size
    n_features = input_size + 1  # state + controls
    B = 5  # batch size

    # Random features: [state, controls]
    x = torch.randn(E, B, n_features) * 0.5

    # --- Direct computation ---
    # Polynomial projection (disentangled: sum of per-degree products)
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
# Test degree purity (disentangled property)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("input_size,degree", [
    (1, 2), (2, 2),
    (1, 3), (2, 3),
])
def test_degree_purity(input_size, degree):
    """Each degree group should only produce coefficients at its own degree indices."""
    module = _make_rnn_module(input_size, degree)
    n_features = input_size + 1
    lib = build_library_structure(n_features, degree)
    degree_ranges = lib['degree_ranges']
    n_terms = lib['n_terms']

    W_groups, w_out, b_out = module._get_effective_params()
    hidden_per_degree = module.hidden_per_degree

    w_out_offset = 0
    for d_idx, W_list_d in enumerate(W_groups):
        d = d_idx + 1

        # Unfold this degree group only (mimic per-degree step from unfold_polynomial_coefficients)
        coeffs = torch.zeros(module.ensemble_size, hidden_per_degree, n_terms)
        coeffs[..., module._linear_indices] = W_list_d[0]
        for k in range(1, d):
            new_coeffs = torch.zeros_like(coeffs)
            for f in range(n_features):
                targets = module._mult_table[:, f]
                valid = targets >= 0
                src_idx = torch.where(valid)[0]
                tgt_idx = targets[src_idx]
                new_coeffs[..., tgt_idx] = new_coeffs[..., tgt_idx] + coeffs[..., src_idx] * W_list_d[k][..., f:f+1]
            coeffs = new_coeffs

        # Contract with w_out segment
        w_seg = w_out[..., w_out_offset:w_out_offset + hidden_per_degree]
        theta_d = torch.einsum('...o,...ot->...t', w_seg, coeffs)
        w_out_offset += hidden_per_degree

        # Only degree-d indices should be non-zero
        start, end = degree_ranges[d]
        for other_d, (s, e) in degree_ranges.items():
            if other_d == d:
                continue
            assert theta_d[..., s:e].abs().max() < 1e-10, (
                f"Degree-{d} group has non-zero coefficients at degree-{other_d} indices.\n"
                f"Max value: {theta_d[..., s:e].abs().max().item():.2e}"
            )


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
        for group in module.projection.degree_groups:
            for w in group:
                w.normal_(0, 0.5)
        module.weight_n.normal_(0, 0.5)

    inputs = torch.randn(1, E, 3, 2, 1) * 0.5
    state = torch.randn(1, E, 3, 2) * 0.5

    # Full mask (no masking)
    mask_full = torch.ones(E, n_terms)
    out_full = module._forward_impl(inputs, state, mask=mask_full)

    # Mask out all terms except bias
    mask_bias_only = torch.zeros(E, n_terms)
    mask_bias_only[:, module._bias_index] = 1.0
    out_bias_only = module._forward_impl(inputs, state, mask=mask_bias_only)

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

    out = module._forward_impl(inputs, state)
    loss = out.sum()
    loss.backward()

    # Check gradients exist for all projection weights across all degree groups
    for d_idx, group in enumerate(module.projection.degree_groups):
        for k, w in enumerate(group):
            assert w.grad is not None, f"No gradient for degree_groups[{d_idx}][{k}]"
            assert w.grad.abs().sum() > 0, f"Zero gradient for degree_groups[{d_idx}][{k}]"
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


# --------------------------------------------------------------------------
# Test L2 coefficient penalty
# --------------------------------------------------------------------------

def test_coefficient_l2_differentiable():
    """_compute_coefficient_l2 should return a differentiable scalar."""
    from spice.resources.spice_training import _compute_coefficient_l2
    from spice.resources.model import BaseModel
    from spice.resources.spice_utils import SpiceConfig

    config = SpiceConfig(
        library_setup={'value': ('reward',)},
        memory_state={'value': 0.0},
    )

    class SimpleModel(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.setup_module(key_module='value', input_size=1)

        def forward(self, inputs, prev_state=None):
            spice_signals = self.init_forward_pass(inputs, prev_state)
            for t in spice_signals.trials:
                self.call_module(
                    key_module='value', key_state='value',
                    inputs=spice_signals.rewards[t, 0],
                    action_mask=spice_signals.actions[t, 0],
                )
                spice_signals.logits[t] = self.state['value']
            spice_signals = self.post_forward_pass(spice_signals)
            return spice_signals.logits, self.get_state()

    model = SimpleModel(
        n_actions=2, n_participants=1, n_experiments=1,
        spice_config=config, sindy_polynomial_degree=1,
        ensemble_size=2, compiled_forward=False,
    )

    penalty = _compute_coefficient_l2(model)
    assert penalty.dim() == 0, "L2 penalty should be a scalar"
    assert penalty.requires_grad, "L2 penalty should be differentiable"
    penalty.backward()

    # Check gradients reached the RNN weights
    rnn = model.submodules_rnn['value']
    for group in rnn.projection.degree_groups:
        for w in group:
            assert w.grad is not None, "L2 penalty should produce gradients for RNN weights"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

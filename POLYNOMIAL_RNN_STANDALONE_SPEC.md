# Technical Specification: Polynomial RNN for Sparse Nonlinear Dynamics Discovery

## Purpose of This Document

This document specifies every technical detail needed to build a **standalone Python package** that implements a multilinear/polynomial RNN with ensemble-based sparse pruning for discovering interpretable dynamical systems from sequential data.

**Target audience:** An LLM (Claude) that will use these instructions to generate the entire codebase from scratch.

**What to build:** A well-structured GitHub repository (working name: `sindy-rnn`) containing:
1. Core polynomial RNN architecture (`PolynomialRNN` — pure `torch.nn.Module`)
2. Simple training loop with ensemble bootstrap and pruning
3. Polynomial coefficient unfolding (the key contribution)
4. Reproducible examples on benchmark dynamical systems (Lorenz, Duffing, Lotka-Volterra)

**Scope:** This is deliberately minimal. No hierarchical models, no individual differences, no modular architecture. One RNN, one state vector, one set of equations. The contributions are:
1. Multilinear RNN cell = exact polynomial
2. Analytical coefficient unfolding
3. Ensemble CI pruning

---

## 1. Motivation and Positioning

### 1.1 The Problem

Given sequential observations of a dynamical system `x[t+1] = f(x[t], u[t])`, discover a **sparse polynomial** approximation of `f` directly from data. This is the core problem of SINDy (Sparse Identification of Nonlinear Dynamics, Brunton et al. 2016), but existing approaches:
- Require clean state derivative estimates (noise-sensitive)
- Use a two-stage pipeline: first fit, then sparsify (information loss)
- Lack principled uncertainty quantification for term selection

### 1.2 Our Approach

Train an ensemble of **polynomial RNNs** end-to-end on prediction loss. Each RNN cell is architecturally constrained to compute an exact degree-D polynomial, whose coefficients can be **analytically extracted** from the weight matrices (no approximation). Ensemble disagreement provides a natural statistical test for pruning: terms that aren't consistently identified across ensemble members are removed.

### 1.3 Key Technical Contributions

1. **Multilinear RNN cell** — product of D independent linear projections = exact degree-D polynomial. Weights encode polynomial coefficients implicitly.
2. **Analytical coefficient unfolding** — recursive algorithm to expand the multilinear product into monomial-basis coefficients. Fully differentiable.
3. **Ensemble CI pruning** — minimum-effect confidence interval test across ensemble members with patience-based stability.

---

## 2. High-Level Architecture

### 2.1 Overview

The model maintains a hidden state vector `h ∈ R^n` and receives control inputs `u ∈ R^m` at each timestep. A **single** polynomial RNN operates on the full state+control vector:

```
h[t+1] = (1 - α) * h[t] + α_n * P(h[t], u[t])
```

where:
- `P: R^{n+m} → R^n` is an exact degree-D polynomial, parameterized as a product of D linear forms followed by a per-output-dimension linear readout
- `α = sigmoid(damping_coefficient)` — learned scalar gate, shared across all ensemble members
- `α_n = α` (default, `scale_candidate=True`) or `α_n = 1` (`scale_candidate=False`)
- The gated update is **element-wise** — the same `α` applies to every state dimension

The polynomial library is constructed from `n + m` features `[h_1, ..., h_n, u_1, ..., u_m]`. Each output dimension `i` gets its own polynomial coefficients via the linear readout, so the model discovers `n` independent equations — one per state variable — but they share the underlying polynomial feature layer.

The model outputs predictions for the next state: `ŷ[t] = h[t]`.

### 2.2 PolynomialRNN (Top-Level Model)

```python
class PolynomialRNN(nn.Module):
    """Polynomial RNN for sparse dynamics discovery.

    A single polynomial RNN operating on the full hidden state h ∈ R^n.
    The polynomial layer maps [h, u] → R^{proj_size} (shared), then a
    per-dimension linear readout produces n independent update candidates.

    Args:
        n_states: Number of state variables (hidden state dimensionality)
        n_controls: Number of external control/forcing inputs
        ensemble_size: Number of independent ensemble members (for pruning)
        polynomial_degree: Maximum polynomial degree (1=linear, 2=bilinear, etc.)
        dropout: Dropout rate
        compiled_forward: Use torch.compile for the forward pass
        initial_state: Initial hidden state values, shape (n_states,) or scalar
    """
    def __init__(
        self,
        n_states: int,
        n_controls: int = 0,
        ensemble_size: int = 1,
        polynomial_degree: int = 2,
        dropout: float = 0.,
        compiled_forward: bool = True,
        initial_state: Union[float, Tensor] = 0.,
    ):
        # Single RNN cell operating on full [h, u] vector
        # input_size = n_states + n_controls (the full state IS the input — no "own state" separation)
        self.rnn = EnsembleRNNModule(
            ensemble_size=ensemble_size,
            n_states=n_states,
            n_controls=n_controls,
            dropout=dropout,
            compiled_forward=compiled_forward,
            polynomial_degree=polynomial_degree,
        )

        # Sparsity mask: (E, n_states, n_terms) — per output dimension
        self.coefficient_masks = torch.ones(
            ensemble_size, n_states, n_library_terms, dtype=torch.bool
        )
        self.pruning_patience = torch.zeros(
            ensemble_size, n_states, n_library_terms, dtype=torch.int32
        )

        # Library term names from features [h_1, ..., h_n, u_1, ..., u_m]
        feature_names = state_names + control_names
        self.library_terms = get_library_feature_names(feature_names, polynomial_degree)

    def forward(self, x, state=None):
        """Forward pass over a sequence.

        Args:
            x: Observations, shape (E, B, T, n_states + n_controls) or (B, T, ...)
                First n_states columns are states, remaining are controls.
            state: Initial hidden state (optional), shape (E, B, n_states)

        Returns:
            predictions: (E, B, T, n_states)
            final_state: (E, B, n_states)
        """
        # Promote to 4D if needed, split states/controls, loop over timesteps
        # For each timestep t:
        #   h[t+1] = rnn.forward_polynomial(h[t], u[t], mask=coefficient_masks)
        #   predictions[t] = h[t+1]
        ...

    def get_equations(self) -> str:
        """Return discovered equations as formatted string."""

    def get_coefficients(self, aggregate=True) -> Dict[str, Tensor]:
        """Return effective polynomial coefficients per state dimension.
        Each value is (n_terms,) if aggregate, else (E, n_terms)."""

    def count_active_terms(self) -> Dict[str, int]:
        """Count active polynomial terms per state dimension."""

    def print_equations(self):
        """Print discovered equations to stdout."""
        print(self.get_equations())

    def save(self, path: str):
        """Save model weights + sparsity masks."""
        torch.save({
            'state_dict': self.state_dict(),
            'coefficient_masks': self.coefficient_masks,
            'pruning_patience': self.pruning_patience,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load saved model."""
        checkpoint = torch.load(path)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.coefficient_masks = checkpoint['coefficient_masks']
        model.pruning_patience = checkpoint['pruning_patience']
        return model
```

### 2.3 EnsemblePolynomialLayer

The core building block. Computes the **element-wise product of D independent affine projections** of the input, yielding an exact degree-D polynomial.

```
output = Π_{d=0}^{D-1} (W_d @ x + b_d) / √D
```

where `W_d ∈ R^{E×O×I}`, `b_d ∈ R^{E×O}`, `x ∈ R^{E×B×I}`, E = ensemble size, O = projection dimension, I = input size, B = batch.

```python
class EnsemblePolynomialLayer(nn.Module):
    def __init__(self, ensemble_size: int, input_size: int, output_size: int, degree: int = 2):
        self.degree = degree
        # D independent weight/bias pairs
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
            for _ in range(degree)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(ensemble_size, output_size))
            for _ in range(degree)
        ])
        # Xavier normal init with small gain (0.01) to keep initial magnitudes small
        for w in self.weights:
            nn.init.xavier_normal_(w, gain=0.01)

    def forward(self, x):
        # x: (E, B, I) → (E, B, O)
        result = torch.einsum('eoi,ebi->ebo', self.weights[0], x) + self.biases[0].unsqueeze(1)
        for d in range(1, self.degree):
            result = result * (torch.einsum('eoi,ebi->ebo', self.weights[d], x) + self.biases[d].unsqueeze(1))
        if self.degree > 1:
            result *= 1.0 / self.degree ** 0.5
        return result
```

**Why this works:** The product of D linear forms in I variables is an exact degree-D polynomial. No activation functions needed — the polynomial nonlinearity emerges from the multiplicative structure.

### 2.4 EnsembleRNNModule

A gated recurrent cell built on EnsemblePolynomialLayer. It maintains the **full** hidden state `h ∈ R^n` and updates it via a shared polynomial feature layer with per-dimension linear readouts:

```
x_t = concat(h[t], u[t])           # (E, B, n+m) — full state + controls
g = PolynomialLayer(x_t)           # (E, B, proj_size) — shared multilinear projection
n = W_out @ g + b_out              # (E, B, n) — per-dimension candidate (linear readout)
h[t+1] = (1 - α) * h[t] + α_n * n # element-wise gated update
```

```python
class EnsembleRNNModule(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        n_states: int,            # hidden state dimensionality (n)
        n_controls: int = 0,      # external control inputs (m)
        dropout: float = 0.,
        compiled_forward: bool = True,
        polynomial_degree: int = 2,
    ):
        n_features = n_states + n_controls  # polynomial operates on [h, u]
        proj_size = max(n_features, 1) * polynomial_degree + 1

        self.projection = EnsemblePolynomialLayer(
            ensemble_size=ensemble_size,
            input_size=n_features,
            output_size=proj_size,
            degree=polynomial_degree,
        )

        # Per-dimension output readout: (E, n_states, proj_size)
        self.weight_n = nn.Parameter(torch.empty(ensemble_size, n_states, proj_size))
        self.bias_n = nn.Parameter(torch.zeros(ensemble_size, n_states))
        nn.init.xavier_uniform_(self.weight_n)

        self.damping_coefficient = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047
        self.scale_candidate = True

        self.dropout = nn.Dropout(p=dropout)
        self.n_states = n_states
        self.n_controls = n_controls

        # Precompute library structure for polynomial unfolding
        lib = build_library_structure(n_features, polynomial_degree)
        self._library_terms = lib['terms']
        self._n_library_terms = lib['n_terms']
        self._bias_index = lib['bias_index']
        self.register_buffer('_mult_table', lib['mult_table'])
        self.register_buffer('_linear_indices', lib['linear_indices'])

        if compiled_forward:
            self._compiled_forward = torch.compile(self._forward_impl, dynamic=True)
```

**Forward pass (standard — gradients flow through this):**

```python
def forward(self, h, u=None):
    # h: (E, B, n) — current hidden state
    # u: (E, B, m) — controls, or None
    x_t = torch.cat([h, u], dim=-1) if u is not None else h  # (E, B, n+m)
    g = self.projection(x_t)                                   # (E, B, proj_size)
    n = torch.einsum('eno,ebo->ebn', self.weight_n, g) + self.bias_n.unsqueeze(1)  # (E, B, n)
    alpha = torch.sigmoid(self.damping_coefficient)
    alpha_n = alpha if self.scale_candidate else 1.0
    h_next = (1 - alpha) * h + alpha_n * n                    # (E, B, n)
    return h_next
```

**Forward pass (polynomial — with sparsity mask):**

```python
def forward_polynomial(self, h, u=None, mask=None):
    # mask: (E, n_states, n_terms) — per-dimension sparsity mask
    theta = self.unfold_polynomial_coefficients()  # (E, n_states, n_terms)
    if mask is not None:
        theta = theta * mask.float()
    x_t = torch.cat([h, u], dim=-1) if u is not None else h  # (E, B, n+m)
    library = self._compute_library(x_t)                       # (E, B, n_terms)
    n = torch.einsum('ebt,ent->ebn', library, theta)          # (E, B, n)
    alpha = torch.sigmoid(self.damping_coefficient)
    alpha_n = alpha if self.scale_candidate else 1.0
    h_next = (1 - alpha) * h + alpha_n * n                    # (E, B, n)
    return h_next
```

---

## 3. Analytical Coefficient Unfolding (KEY CONTRIBUTION)

The method `unfold_polynomial_coefficients()` converts the implicit polynomial (product of D linear forms contracted with the output projection) into explicit monomial-basis coefficients.

### 3.1 Library Structure

First, build a multiplication table at init time:

```python
def build_library_structure(n_features: int, degree: int) -> dict:
    """Enumerate monomials and build multiplication table for recursive expansion.

    Term ordering matches combinations_with_replacement.

    Example for n_features=2, degree=2:
        terms = [(), (0,), (1,), (0,0), (0,1), (1,1)]
        names = ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']

    Returns:
        terms: list of tuples (sorted feature index multisets)
        mult_table: (n_terms, n_features) long tensor — mult_table[t, f] = index
            of (term_t * x_f), or -1 if product exceeds degree
        linear_indices: (n_features,) long tensor — indices of degree-1 terms
        bias_index: int — index of the constant term '1'
        n_terms: int — total number of library terms
    """
    terms = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            terms.append(combo)

    term_to_idx = {term: idx for idx, term in enumerate(terms)}
    n_terms = len(terms)

    mult_table = torch.full((n_terms, n_features), -1, dtype=torch.long)
    for t_idx, term in enumerate(terms):
        if len(term) < degree:
            for f in range(n_features):
                product = tuple(sorted(term + (f,)))
                if product in term_to_idx:
                    mult_table[t_idx, f] = term_to_idx[product]

    bias_index = term_to_idx[()]
    linear_indices = torch.tensor([term_to_idx[(f,)] for f in range(n_features)])

    return {'terms': terms, 'mult_table': mult_table, 'linear_indices': linear_indices,
            'bias_index': bias_index, 'n_terms': n_terms}
```

### 3.2 Recursive Unfolding Algorithm

```python
def unfold_polynomial_coefficients(self):
    """Unfold weight matrices into polynomial coefficients via recursive multiplication.

    Expands the product of D independent linear projections
    ``Π_{d=0}^{D-1} (W_d @ x + b_d) / sqrt(D)``
    into the monomial basis, then contracts with the output projection
    ``w_out`` to get final polynomial coefficients.

    Works for arbitrary polynomial degree. Fully differentiable.
    Uses ellipsis (...) indexing so the same code works regardless of
    leading batch dimensions.

    Returns:
        theta: (E, n_states, n_terms) — polynomial coefficients in monomial basis.
    """
    W_list = list(self.projection.weights)    # D tensors of (E, O, I_in)
    b_list = list(self.projection.biases)     # D tensors of (E, O)
    w_out = self.weight_n                     # (E, n_states, proj_size)
    b_out = self.bias_n                       # (E, n_states)
    degree = self.projection.degree
    n_terms = self._n_library_terms
    n_features = self._mult_table.shape[1]

    # Initialize with first linear form (d=0):
    # coeffs[..., j, t] is the monomial-t coefficient for hidden unit j
    coeffs = torch.zeros(*W_list[0].shape[:-1], n_terms,
                         device=W_list[0].device, dtype=W_list[0].dtype)
    coeffs[..., self._linear_indices] = W_list[0]
    coeffs[..., self._bias_index] = b_list[0]

    # Recursively multiply by remaining linear forms (d=1 to D-1)
    for d in range(1, degree):
        # Multiply all existing terms by the bias of the d-th linear form
        new_coeffs = coeffs * b_list[d].unsqueeze(-1)
        # For each feature, multiply existing terms and accumulate into product terms
        for f in range(n_features):
            targets = self._mult_table[:, f]
            valid = targets >= 0
            src_idx = torch.where(valid)[0]
            tgt_idx = targets[src_idx]
            new_coeffs[..., tgt_idx] = new_coeffs[..., tgt_idx] + coeffs[..., src_idx] * W_list[d][..., f:f+1]
        coeffs = new_coeffs

    # Apply degree scaling (matches EnsemblePolynomialLayer)
    if degree > 1:
        coeffs = coeffs * (1.0 / degree ** 0.5)

    # Contract with output projection: θ_t = Σ_j w_out_j * coeffs_j_t + b_out
    theta = torch.einsum('...o,...ot->...t', w_out, coeffs)
    theta[..., self._bias_index] = theta[..., self._bias_index] + b_out

    return theta  # (E, n_states, n_terms)
```

### 3.3 Library Evaluation

For the polynomial forward path, compute monomial values from features:

```python
def _compute_library(self, features):
    """Compute polynomial library values for all monomial terms.

    The library is shared across all output dimensions — the same monomial
    values are used for every state equation. Per-dimension differentiation
    comes from the output readout (weight_n) and sparsity masks.

    Args:
        features: (E, B, n_features) tensor of [h_0, ..., h_{n-1}, u_0, ..., u_{m-1}]

    Returns:
        library: (E, B, n_terms) tensor of monomial values
    """
    n_terms = self._n_library_terms
    library = torch.ones(*features.shape[:-1], n_terms, device=features.device)

    # Linear terms: direct gather
    library[..., self._linear_indices] = features

    # Higher-degree terms: products of features
    for t_idx, term in enumerate(self._library_terms):
        if len(term) >= 2:
            val = features[..., term[0]]
            for f_idx in term[1:]:
                val = val * features[..., f_idx]
            library[..., t_idx] = val

    return library
```

### 3.4 Effective Coefficients and the Gated Update

The gated update `h[t+1] = (1-α)*h[t] + α_n * P(h, u)` means the **effective** polynomial governing the full state transition has per-dimension self-terms. For output dimension `i`:

- For all terms except `h_i`: `c_eff[i, t] = α_n * θ_raw[i, t]`
- For the self-term `h_i[t]`: `c_eff[i, h_i_idx] = (1-α) + α_n * θ_raw[i, h_i_idx]`

Each output dimension `i` has a **different** self-term index — it is the library index of the linear monomial `h_i`. Note that `h_i` also appears in the polynomial for output `j ≠ i` as a cross-state term, where it does NOT get the `(1-α)` contribution.

All reporting, printing, and pruning operates on **effective coefficients**. The `(1-α)` contribution to the state self-term is always present regardless of the sparsity mask — it comes from the gated architecture, not from a learned coefficient.

### 3.5 Key Invariant

The standard forward path (through `EnsemblePolynomialLayer`) and the polynomial forward path (through unfolded coefficients + library) compute **exactly the same function**. This is by construction — the unfolding is an analytical identity, not an approximation.

Critical test: `forward(x, h)` must equal `forward_polynomial(x, h, mask=ones)` to numerical precision.

---

## 4. Ensemble Infrastructure

### 4.1 EnsembleLinear

```python
class EnsembleLinear(nn.Module):
    """Linear layer with independent parameters per ensemble member.
    weight: (E, out_features, in_features), bias: (E, out_features)
    """
    def __init__(self, ensemble_size, in_features, out_features):
        self.weight = nn.Parameter(torch.empty(ensemble_size, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.einsum('eoi,...ei->...eo', self.weight, x) + self.bias
```

### 4.2 Bootstrapping

Each ensemble member trains on a different bootstrap sample of the data:

```python
if E > 1:
    indices = torch.randint(0, B, (E, B))
    xs_train = xs[indices]  # (E, B, T, F)
else:
    xs_train = xs.unsqueeze(0)  # (1, B, T, F)
```

Bootstrap indices are generated once at the start of training, not per-epoch.

---

## 5. Pruning System

### 5.1 Sparsity Masks

The model maintains a single sparsity mask and patience counter, both shaped per output dimension:
- `coefficient_masks`: boolean tensor `(E, n_states, n_terms)` — which polynomial terms are active per ensemble member, per output dimension
- `pruning_patience`: int tensor `(E, n_states, n_terms)` — consecutive pruning failures

**Initial state:** All terms active (True), all patience counters at 0.

**Optional initial exclusions:**
- `include_bias=False` → mask out the constant term '1'
- `interaction_only=True` → mask out pure power terms (e.g., `x^2` but keep `x*y`)

### 5.2 Primary Pruning: Ensemble CI Test

Minimum-effect confidence interval test across ensemble members:

```python
def minimum_effect_ci_test(
    coefficients: Tensor,  # (E, n_states, n_terms) — effective polynomial coefficients
    presence: Tensor,      # (E, n_states, n_terms) — boolean mask
    alpha: float = 0.05,   # confidence level
    delta: float = 0.0,    # minimum effect size threshold
) -> Tensor:              # (n_states, n_terms) — True where term survives
    """
    A term survives iff: |mean| - t_crit * SE > delta

    When delta=0: equivalent to standard t-test for non-zero mean.
    Pruned coefficients (presence=False) are treated as zero, so terms
    only found by a few ensemble members are naturally penalized.

    The test is applied independently per output dimension (n_states),
    so term t may survive for state i but be pruned for state j.
    """
    effective = (coefficients * presence.float()).detach()
    E = effective.shape[0]

    mean = effective.mean(dim=0)             # (n_states, n_terms)
    std = effective.std(dim=0, correction=1)
    se = std / (E ** 0.5)

    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df=E - 1)

    ci_lower = mean.abs() - t_crit * se
    significant = ci_lower > delta

    # Require at least 2 active ensemble members for valid test
    n_active = presence.float().sum(dim=0)
    significant = significant & (n_active >= 2)

    return significant
```

### 5.3 Patience Mechanism

Terms must fail the CI test **2 consecutive times** before permanent removal:

```python
def ensemble_prune(model, alpha, delta):
    # Get effective coefficients from the single RNN
    theta = model.rnn.unfold_polynomial_coefficients()  # (E, n_states, n_terms)
    alpha_gate = sigmoid(model.rnn.damping_coefficient).item()
    alpha_n = alpha_gate if model.rnn.scale_candidate else 1.0
    theta_eff = theta * alpha_n

    mask = model.coefficient_masks                # (E, n_states, n_terms)
    significant = minimum_effect_ci_test(theta_eff, mask, alpha, delta)  # (n_states, n_terms)

    still_active = mask.any(dim=0)                # (n_states, n_terms)
    failed = ~significant & still_active

    # Update patience: increment on failure, reset on success
    counters = model.pruning_patience              # (E, n_states, n_terms)
    counters = torch.where(
        failed.unsqueeze(0).expand_as(counters),
        counters + 1,
        torch.zeros_like(counters),
    )

    # Prune terms with counter >= 2
    prune = counters[0] >= 2                       # (n_states, n_terms)
    if prune.any():
        model.coefficient_masks &= ~prune.unsqueeze(0).expand_as(mask)
        counters *= (~prune.unsqueeze(0).expand_as(counters)).int()

    model.pruning_patience = counters
```

### 5.4 Fallback: Per-Member Threshold Pruning

For single ensemble member (`E=1`), per-member hard thresholding with patience accumulation:

```python
def threshold_patience_update(model, threshold):
    """Increment patience for terms with |effective_coef| < threshold. Reset on success."""
    theta_eff = get_effective_coefficients(model.rnn)  # (E, n_states, n_terms)
    below = (theta_eff.abs() < threshold) & model.coefficient_masks
    model.pruning_patience = torch.where(
        below, model.pruning_patience + 1, torch.zeros_like(model.pruning_patience)
    )

def threshold_prune(model, patience_limit):
    """Prune terms that exceeded patience_limit."""
    candidates = (model.pruning_patience >= patience_limit) & model.coefficient_masks
    model.coefficient_masks &= ~candidates
    model.pruning_patience *= (~candidates).int()
```

---

## 6. Training

### 6.1 Overview

Single-stage training. No two-stage fitting, no separate SINDy optimization, no L1 penalty. No sklearn estimator — `PolynomialRNN` is a pure `torch.nn.Module`.

1. **MSE loss** on next-state prediction
2. **L2 weight decay** via AdamW (implicitly penalizes higher-degree terms more)
3. **Periodic pruning** of polynomial terms using ensemble CI test

### 6.2 Data Format

Plain tensors — no dataset class needed:
- `xs`: `(B, T, n_states)` or `(B, T, n_states + n_controls)` — state observations (+ controls)
- `ys`: `(B, T, n_states)` — next-state targets

For a single trajectory, `B=1`. NaN-pad for variable-length sequences.

### 6.3 Training Loop

```python
def fit(model, xs, ys, xs_test=None, ys_test=None,
        epochs=500, warmup_steps=None, batch_size=None,
        learning_rate=1e-2, l2=1e-4,
        pruning_frequency=1, pruning_threshold=None,
        ensemble_pruning_alpha=0.05,
        verbose=True):
    """Train the polynomial RNN.

    Args:
        model: PolynomialRNN instance
        xs: (B, T, n_states + n_controls) — state observations + optional controls
        ys: (B, T, n_states) — next-state targets
        xs_test, ys_test: optional validation data (same shapes)
        epochs: number of training epochs
        warmup_steps: epochs before pruning begins (default: epochs // 4)
        batch_size: mini-batch size (None = full batch)
        learning_rate: AdamW learning rate
        l2: AdamW weight decay
        pruning_frequency: epochs between pruning events
        pruning_threshold: minimum effect size delta for CI test
        ensemble_pruning_alpha: confidence level for ensemble CI test
        verbose: print progress
    """
    if warmup_steps is None:
        warmup_steps = epochs // 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)

    E = model.ensemble_size
    B, T = xs.shape[0], xs.shape[1]

    # Bootstrap: (B, T, F) → (E, B, T, F)
    if E > 1:
        indices = torch.randint(0, B, (E, B))
        xs_train = xs[indices]
        ys_train = ys[indices]
    else:
        xs_train = xs.unsqueeze(0)
        ys_train = ys.unsqueeze(0)

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        if batch_size and batch_size < B:
            batch_idx = torch.randperm(B)[:batch_size]
            xb, yb = xs_train[:, batch_idx], ys_train[:, batch_idx]
        else:
            xb, yb = xs_train, ys_train

        ys_pred, _ = model(xb)  # (E, B, T, n_states)

        # NaN masking (variable-length sequences)
        mask = ~torch.isnan(yb.sum(dim=-1))
        loss = F.mse_loss(ys_pred[mask], yb[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Pruning (after warmup, every pruning_frequency epochs)
        if epoch >= warmup_steps and epoch % pruning_frequency == 0:
            if ensemble_pruning_alpha and E > 1:
                ensemble_prune(model, ensemble_pruning_alpha, pruning_threshold or 0.0)
            elif pruning_threshold and pruning_threshold > 0:
                threshold_patience_update(model, pruning_threshold)
                threshold_prune(model, patience_limit=2)

        # Logging
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            active = model.count_active_terms()
            total = sum(active.values())
            msg = f"Epoch {epoch:4d} | loss {loss.item():.6f} | active terms: {total}"
            if xs_test is not None:
                with torch.no_grad():
                    model.eval()
                    ys_test_pred, _ = model(xs_test.unsqueeze(0).expand(E, -1, -1, -1))
                    mask_t = ~torch.isnan(ys_test.sum(dim=-1))
                    loss_test = F.mse_loss(ys_test_pred[:, mask_t], ys_test[mask_t])
                    msg += f" | test {loss_test.item():.6f}"
            print(msg)
```

---

## 7. Equation Extraction and Printing

### 7.1 Effective Coefficients

```python
def get_coefficients(model, aggregate=True):
    """Return effective polynomial coefficients for each state dimension.

    Effective = gate-absorbed:
    - Non-state terms: c_eff = α_n * θ_unfolded
    - State self-term h_i: c_eff[i, h_i_idx] = (1-α) + α_n * θ[i, h_i_idx]
    (1-α) is ALWAYS added to the self-term regardless of mask.

    Returns: dict mapping state_name → Tensor
        If aggregate: (n_terms,) — ensemble mean (NaN-aware)
        Else: (E, n_terms) — per-member
    """
    alpha = torch.sigmoid(model.rnn.damping_coefficient).item()
    alpha_n = alpha if model.rnn.scale_candidate else 1.0

    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    mask = model.coefficient_masks.float()                        # (E, n_states, n_terms)

    # Apply mask and gate scaling
    c_eff = theta * mask * alpha_n                                # (E, n_states, n_terms)

    # Add (1-α) to each dimension's self-term
    # The library features are [h_0, h_1, ..., h_{n-1}, u_0, ...], so
    # the linear index of h_i is model.rnn._linear_indices[i]
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        c_eff[:, i, self_idx] = c_eff[:, i, self_idx] + (1 - alpha)

    results = {}
    for i in range(model.n_states):
        c_i = c_eff[:, i, :]          # (E, n_terms)
        mask_i = mask[:, i, :]        # (E, n_terms)
        self_idx = model.rnn._linear_indices[i].item()

        if aggregate:
            # NaN-aware mean: treat pruned terms as NaN
            c_agg = c_i.clone()
            non_self = torch.ones(c_i.shape[-1], dtype=torch.bool)
            non_self[self_idx] = False
            c_agg[:, non_self] = torch.where(
                mask_i[:, non_self] == 0,
                torch.tensor(float('nan')),
                c_agg[:, non_self],
            )
            c_i = torch.nanmean(c_agg, dim=0)
            c_i = torch.nan_to_num(c_i, nan=0.0)

        results[model.state_names[i]] = c_i

    return results
```

### 7.2 Equation Printing

```python
def get_equations(model):
    """Return discovered equations as formatted string.

    Example:
        x[t+1] = 0.900*x[t] + 0.100*y
        y[t+1] = 0.280*x - 0.100*x*z + 0.900*y[t]
        z[t+1] = 0.100*x*y + 0.973*z[t]
    """
    coefs = model.get_coefficients(aggregate=True)
    lines = []
    for state_name in model.state_names:
        c = coefs[state_name]
        terms = model.library_terms
        equation = f"{state_name}[t+1] = "
        active_parts = []
        for j, term in enumerate(terms):
            if abs(c[j]) > 0:
                coef_str = f"{abs(c[j]):.3f}"
                sign = "+" if c[j] >= 0 else "-"
                if term == '1':
                    active_parts.append((sign, coef_str))
                elif term == state_name:
                    active_parts.append((sign, f"{coef_str}*{state_name}[t]"))
                else:
                    active_parts.append((sign, f"{coef_str}*{term}"))
        # Format with signs
        ...
        lines.append(equation)
    return "\n".join(lines)
```

---

## 8. Polynomial Library Utilities

```python
def compute_library_size(n_features: int, degree: int) -> int:
    """Number of monomials up to degree D from n features.
    Example: (n=2, d=2) → 6 terms {1, x0, x1, x0², x0*x1, x1²}
    """

def get_library_feature_names(feature_names: List[str], degree: int) -> List[str]:
    """Generate human-readable monomial names.
    Example: ['h', 'u'], degree=2 → ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']
    """

def get_polynomial_degree_from_term(term: str) -> int:
    """Extract degree from term string. '1'→0, 'x'→1, 'x^2'→2, 'x*y'→2"""
```

---

## 9. Repository Structure

```
sindy-rnn/
├── sindy_rnn/
│   ├── __init__.py                    # Public API: PolynomialRNN, fit
│   ├── model.py                       # PolynomialRNN, EnsembleRNNModule,
│   │                                  #   EnsemblePolynomialLayer, EnsembleLinear
│   ├── training.py                    # fit(), bootstrap
│   ├── pruning.py                     # minimum_effect_ci_test, ensemble_prune,
│   │                                  #   threshold_prune
│   ├── polynomial_library.py          # compute_library_size, get_library_feature_names,
│   │                                  #   build_library_structure
│   └── equations.py                   # get_coefficients, get_equations
├── examples/
│   ├── lorenz.py                      # Lorenz attractor discovery
│   ├── duffing.py                     # Duffing oscillator
│   ├── lotka_volterra.py              # Predator-prey
│   └── linear_system.py              # Simple linear system (sanity check)
├── tests/
│   ├── test_polynomial_layer.py       # forward == forward_polynomial (mask=ones)
│   ├── test_unfolding.py              # Known polynomial recovery
│   ├── test_pruning.py                # CI test, patience, mask updates
│   └── test_training.py              # Simple system recovery end-to-end
├── requirements.txt                   # torch, numpy, scipy
└── README.md
```

---

## 10. Implementation Priorities

### Phase 1: Core (must have)
1. `EnsemblePolynomialLayer` with forward pass
2. `EnsembleRNNModule` with standard and polynomial forward paths
3. `unfold_polynomial_coefficients()` — the key algorithm
4. `build_library_structure()` and `get_library_feature_names()`
5. `PolynomialRNN` — the top-level model (with save/load, get_equations, get_coefficients)
6. Training loop (`fit()`) with MSE loss, gradient clipping, NaN masking, bootstrap
7. Ensemble CI pruning with patience
8. Equation extraction and printing

### Phase 2: Polish
9. `EnsembleLinear`
10. `torch.compile` support
11. Fallback threshold pruning (single ensemble member)

### Phase 3: Examples & Tests
12. Lorenz system discovery example
13. Duffing oscillator example
14. Linear system sanity check
15. Unit tests for unfolding correctness
16. Integration test: known polynomial system recovery

---

## 11. Critical Implementation Details

### 11.1 Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Training data (after bootstrap) | `(E, B, T, F)` | F = n_states + n_controls |
| Targets | `(E, B, T, n_states)` | |
| Hidden state | `(E, B, n_states)` | Full state vector |
| RNN input | `(E, B, n_states + n_controls)` | [h, u] concatenated |
| Shared projection output | `(E, B, proj_size)` | From EnsemblePolynomialLayer |
| Polynomial coefficients (unfolded) | `(E, n_states, n_terms)` | Per output dimension |
| Sparsity mask | `(E, n_states, n_terms)` | Per output dimension |
| Patience counter | `(E, n_states, n_terms)` | Per output dimension |
| Monomial library values | `(E, B, n_terms)` | Shared across output dims |
| Predictions | `(E, B, T, n_states)` | |

### 11.2 Dimension Conventions

| Symbol | Meaning |
|--------|---------|
| E | Ensemble members |
| B | Batch (sequences) |
| T | Timesteps |
| n | n_states |
| m | n_controls |
| C | Candidate polynomial terms (n_terms) |

### 11.3 Common Pitfalls

1. **`tensor.to(device)` returns a new tensor** — must reassign
2. **Masks and patience counters** are not `nn.Parameter` — need manual `.to(device)` in the model's `to()` method (or use `register_buffer`)
3. **`(1-α)` state decay** is added to effective coefficients regardless of mask — it's architectural
4. **NaN masking** before loss computation — variable-length sequences are NaN-padded
5. **Gradient clipping** (`max_norm=1.0`) is essential for stability
6. **Bootstrap indices** are generated once at start of training, not per-epoch
7. **The polynomial forward with mask** is used during training, not just eval
8. **`proj_size = max(n_features, 1) * polynomial_degree + 1`** where `n_features = n_states + n_controls` — intermediate projection dimension, not library size
9. **Xavier init with gain=0.01** for polynomial layer weights — small initial coefficients
10. **Damping init `torch.tensor(-3.0)`** → `sigmoid(-3) ≈ 0.047` — nearly closed gate, very persistent state
11. **Per-dimension self-terms** — output dim `i` gets `(1-α)` added to its `h_i` coefficient only. The same monomial `h_i` appearing in output dim `j ≠ i` is a cross-state term and does NOT get `(1-α)`

### 11.4 Feature Ordering

The single RNN operates on the full concatenated vector `[h, u]`:
```
features = [h_0, h_1, ..., h_{n-1}, u_0, u_1, ..., u_{m-1}]
```
This is both the RNN input and the polynomial library's feature space. The linear monomial index of state `h_i` is `_linear_indices[i]` (equal to `i` for the standard ordering). Library terms are shared across all output dimensions — sparsity determines which terms each dimension uses.

---

## 12. Example: Lorenz System Discovery

```python
import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit

# Generate Lorenz data
def lorenz_rk4(x, dt=0.01, sigma=10, rho=28, beta=8/3):
    def f(x):
        return np.array([sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]])
    k1 = f(x); k2 = f(x+dt/2*k1); k3 = f(x+dt/2*k2); k4 = f(x+dt*k3)
    return x + dt/6*(k1+2*k2+2*k3+k4)

x = np.array([1., 1., 1.])
trajectory = [x]
for _ in range(5000):
    x = lorenz_rk4(x)
    trajectory.append(x)
trajectory = np.array(trajectory)  # (5001, 3)

# Prepare data: (B=1, T, n_states)
xs = torch.tensor(trajectory[:-1], dtype=torch.float32).unsqueeze(0)
ys = torch.tensor(trajectory[1:], dtype=torch.float32).unsqueeze(0)

# Create model
model = PolynomialRNN(
    n_states=3,
    n_controls=0,
    polynomial_degree=2,
    ensemble_size=10,
    state_names=['x', 'y', 'z'],
)

# Train
fit(model, xs, ys,
    epochs=500,
    warmup_steps=100,
    ensemble_pruning_alpha=0.05,
    pruning_threshold=0.01,
    learning_rate=1e-2,
    l2=1e-4,
    verbose=True,
)

model.print_equations()

# Expected output (approximately):
# x[t+1] = 0.900*x[t] + 0.100*y
# y[t+1] = 0.280*x - 0.100*x*z + 0.900*y[t]
# z[t+1] = 0.100*x*y + 0.973*z[t]
```

---

## 13. Dependencies

```
torch >= 2.0
numpy
scipy
```

Optional: `matplotlib` for examples.

---

## 14. Testing Strategy

### Unit Tests
1. **Unfolding correctness:** `forward(x, h) == forward_polynomial(x, h, mask=ones)` for random weights
2. **Library structure:** Verify multiplication table for small cases (n=2, d=2)
3. **Feature names:** Verify name generation matches manual enumeration
4. **CI test:** Term with consistent nonzero mean passes; term with zero mean fails
5. **Patience:** Counter increments on failure, resets on success, prunes at 2

### Integration Tests
6. **Linear system recovery:** Known `x[t+1] = 0.9*x + 0.1*u` → verify coefficients
7. **Sparsity:** Many irrelevant terms get pruned
8. **Ensemble consistency:** Members discover similar equations

---

## 15. Notes for the Implementing Agent

- **Keep it minimal.** No module system, no config objects, no per-entity logic, no sklearn estimator, no dataset class. One polynomial RNN, one state vector, plain tensors in and out.
- **The core novelty** is the multilinear → monomial unfolding. Get this right and test it thoroughly.
- **`PolynomialRNN`** is the main class — a pure `torch.nn.Module`. It contains a **single** `EnsembleRNNModule` with multi-dimensional output (`n_states` output dimensions via per-dimension linear readout from a shared polynomial feature layer). It has `get_equations()`, `get_coefficients()`, `save()`, `load()` methods directly.
- **`fit()`** is a standalone function that takes a `PolynomialRNN` and plain tensors. MSE loss only.
- **Feature names** are `[h_0, h_1, ..., h_{n-1}, u_0, ..., u_{m-1}]` — the same for all output dimensions. The library is shared; sparsity masks are per output dimension `(E, n_states, n_terms)`.
- **Use `torch.compile` with `dynamic=True`** by default, with fallback.

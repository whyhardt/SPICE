"""Concept discovery: block-structured reduction of per-participant SINDy coefficients.

`sindy_compression` factorizes the coefficient matrix into K shared mechanisms
(rotationally ambiguous, signed and additive so mechanisms can cancel, and
unable to tell a structural zero from a small coefficient). `sindy_ties`
searches for shared low-dimensional constraints, but from a hand-written
candidate list (term pairs, lag families, geometric templates) accepted greedily
with a full behavioural refit per candidate.

This module replaces both with one object. Per module, the candidate terms are
*partitioned* into K concepts. Concept k owns a term set S_k and a unit-norm
direction v_k over those terms; a participant either has the concept or does not
(a binary gate), and if they have it they contribute one number:

    c_p[S_k] = z_pk * theta_pk * v_k          z in {0,1},  ||v_k|| = 1

Everything the other two modules do is a special case:

  * |S_k| = 2 is exactly a rank-1 tie (`sindy_ties`' `kind="free", rank=1`).
  * |S_k| = 1 is an ordinary free coefficient.
  * a lag family with v_k ~ (1, g, g^2, ...) is a decaying memory kernel --
    discovered, not templated.
  * `fit_term_family_basis`' hand-classified families are a *fixed* partition;
    here the partition is learned from column collinearity.

Disjointness is what makes it interpretable. Two concepts never share a term, so
(i) nothing can cancel, (ii) the basis is identified up to permutation only --
there is no rotation to be ambiguous about, and (iii) the presence mask is
exactly a Boolean factorization `mask_pj = OR_k (z_pk AND [j in S_k])`, so a
participant's support is a union of whole concepts rather than an arbitrary
subset. That last point is where the description-length saving lives: naming an
arbitrary support costs `log C(T, k_p)` nats *per participant*, naming a
partition costs `T log K` *once*.

Everything here runs on the one-step state-space objective, which is quadratic
in the coefficients:

    RSS_u(c) = c^T G_u c - 2 b_u^T c + btb_u

with G_u = Theta^T Theta and b_u = Theta^T y accumulated per unit u = (ensemble
member, participant, experiment) by `BaseModel.sindy_ridge_accumulate`. Those
sufficient statistics are a few MB for a whole study, so once collected, the
entire search -- clustering, alternating least squares, gate tests, partition
refinement, the MDL path over K -- touches no data and needs no forward passes.
A candidate partition costs milliseconds to score, against ~3 minutes for one
candidate tie under `analysis_coefficient_ties`.

Two things this deliberately does *not* do:

  * It does not fit against behavioural NLL. Fitting choice likelihood directly
    lets the coefficients drift away from the RNN's dynamics to compensate for
    approximation error at the readout, which is only a gain when the SINDy/RNN
    state discrepancy is already large -- and when it is, the equations have
    stopped describing the RNN. Behavioural likelihood is *measured* here, never
    optimized.
  * It does not use the K-step shooting objective. Shooting applies the
    coefficients K times, so the loss stops being quadratic and the
    sufficient-statistic algebra no longer holds. Discovery runs one-step (where
    it does hold); estimating theta and v within the discovered structure is
    left to the existing Stage 2.2 shooting refit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------

@dataclass
class ModuleStats:
    """One-step normal equations for one module, per unit u = (member, participant, experiment).

    ``term_indices`` maps the reduced term axis used everywhere in this module
    back to the model's full candidate-term axis: terms excluded by the model's
    theory prior mask (e.g. squares of binary signals) are dropped up front
    rather than carried as permanently-zero columns, which would otherwise make
    every Gram matrix exactly singular.
    """

    module: str
    G: torch.Tensor              # (U, T, T)
    b: torch.Tensor              # (U, T)
    btb: torch.Tensor            # (U,)
    n_rows: torch.Tensor         # (U,)
    term_names: List[str]
    term_indices: np.ndarray     # (T,) indices into the model's full term axis
    E: int
    P: int
    X: int

    @property
    def U(self) -> int:
        return self.G.shape[0]

    @property
    def T(self) -> int:
        return self.G.shape[1]

    def unit_participant(self) -> np.ndarray:
        """(U,) participant index of each unit."""
        return np.repeat(np.arange(self.P), self.X)[None, :].repeat(self.E, 0).reshape(-1)

    def unit_member(self) -> np.ndarray:
        """(U,) ensemble-member index of each unit."""
        return np.repeat(np.arange(self.E), self.P * self.X)


def collect_sufficient_statistics(
    estimator, xs: torch.Tensor, ys: torch.Tensor, verbose: bool = True,
) -> Dict[str, ModuleStats]:
    """Accumulate one-step normal equations on the **unmasked** library.

    `sindy_ridge_accumulate` zeroes library columns for pruned terms
    ([model.py], `sample_mask`), which is right for a ridge solve on a fixed
    support and wrong here: concept discovery re-decides the support, so it
    needs the statistics for every candidate term including the ones Stage 2.1
    happened to prune. Presence is therefore forced to the theory prior mask for
    the duration of the pass and restored afterwards.

    ``xs``/``ys`` are 5D (E, B, T, W, F); pass the full non-bootstrapped data so
    every ensemble member sees the same trials.
    """
    from spice.resources.spice_training import _ridge_solve_sindy

    model = getattr(estimator, 'model', estimator)
    modules = model.get_modules()

    saved_presence = {m: model.sindy_coefficients_presence[m].clone() for m in modules}
    saved_coefficients = {m: model.sindy_coefficients[m].data.clone() for m in modules}
    try:
        for m in modules:
            model.sindy_coefficients_presence[m].fill_(True)
            model.sindy_coefficients_presence[m] &= model.sindy_coefficients_prior_mask[m]
        # _ridge_solve_sindy runs the accumulation pass and then finalizes (which
        # overwrites the coefficients). We only want the accumulators; the
        # coefficients are restored below, so the solve itself is harmless.
        _ridge_solve_sindy(model, xs, ys)
        accumulators = {m: model._ridge_accumulators[m] for m in modules}

        stats: Dict[str, ModuleStats] = {}
        E = model.ensemble_size
        P, X = model.n_participants, model.n_experiments
        for m in modules:
            accum = accumulators[m]
            if 'btb' not in accum:
                raise RuntimeError(
                    "Ridge accumulators lack 'btb'/'n_rows'. sindy_concepts needs the "
                    "absolute residual (for the Gaussian noise scale), not just the "
                    "relative one -- update BaseModel.sindy_ridge_accumulate."
                )
            T_full = accum['AtA'].shape[-1]
            allowed = model.sindy_coefficients_prior_mask[m].reshape(-1, T_full).any(dim=0)
            keep = torch.where(allowed)[0]

            G = accum['AtA'].reshape(E * P * X, T_full, T_full)[:, keep][:, :, keep]
            b = accum['Atb'].reshape(E * P * X, T_full)[:, keep]
            stats[m] = ModuleStats(
                module=m,
                G=G.double().cpu(),
                b=b.double().cpu(),
                btb=accum['btb'].reshape(-1).double().cpu(),
                n_rows=accum['n_rows'].reshape(-1).double().cpu(),
                term_names=[model.sindy_candidate_terms[m][i] for i in keep.tolist()],
                term_indices=keep.cpu().numpy(),
                E=E, P=P, X=X,
            )
            if verbose:
                print(f"  {m:32s} U={stats[m].U:5d}  T={stats[m].T:3d} "
                      f"(of {T_full})  rows/unit={stats[m].n_rows.mean():.0f}")
        return stats
    finally:
        for m in modules:
            model.sindy_coefficients_presence[m] = saved_presence[m]
            model.sindy_coefficients[m].data = saved_coefficients[m]
        model.reset_ridge_accumulators()


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def _ridged(G: torch.Tensor, rel: float = 1e-8) -> torch.Tensor:
    """G + eps*I with eps scaled to G's own magnitude.

    Condition numbers of 1e8-1e13 are routine for these libraries (see the
    float64 note in `sindy_ridge_accumulate`), so an absolute epsilon is either
    negligible or dominant depending on the module. Scaling by the mean diagonal
    keeps the regularization at a fixed *relative* level across modules.
    """
    T = G.shape[-1]
    scale = torch.diagonal(G, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-30)
    eye = torch.eye(T, dtype=G.dtype, device=G.device)
    return G + (rel * scale)[..., None, None] * eye


def _solve(A: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Batched solve with a Cholesky fast path and an lstsq fallback."""
    try:
        return torch.linalg.solve(A, r.unsqueeze(-1)).squeeze(-1)
    except Exception:
        return torch.linalg.lstsq(A, r.unsqueeze(-1)).solution.squeeze(-1)


def full_support_rss(stats: ModuleStats) -> torch.Tensor:
    """(U,) residual sum of squares with every allowed term free.

    The reference point for the noise scale: RSS = btb - b^T G^-1 b.
    """
    c = _solve(_ridged(stats.G), stats.b)
    return (stats.btb - (stats.b * c).sum(-1)).clamp_min(0.0)


def noise_scale(stats: ModuleStats) -> float:
    """Gaussian sigma^2 for this module, from the full-support residual.

    A Gaussian BIC on the state-space objective needs a variance to make
    `RSS/sigma^2` commensurable with a `k log n` penalty in nats; without one,
    the trade-off "is this parameter worth log(n)?" compares a squared residual
    to a count and is dimensionally meaningless. Estimated once, at full
    support, and held fixed for every candidate structure so that comparisons
    between structures are not also comparisons between noise models.
    """
    rss = full_support_rss(stats)
    dof = (stats.n_rows - stats.T).clamp_min(1.0)
    return float(rss.sum() / dof.sum())


# ---------------------------------------------------------------------------
# Column clustering: the initial partition
# ---------------------------------------------------------------------------

def column_distances(
    coefficients: np.ndarray, support: np.ndarray, min_overlap: int = 5,
) -> np.ndarray:
    """(T, T) projective distance between coefficient columns.

    Terms i and j belong to the same rank-1 concept exactly when their columns
    are proportional across participants -- `c_pi = lam c_pj` for all p -- so the
    quantity to minimize is one minus the squared cosine between the columns:

        d(i, j) = 1 - (c_i . c_j)^2 / (||c_i||^2 ||c_j||^2)

    Squared, so a sign flip does not split a concept; scale-free, so no term
    standardization is needed; and critically it never divides by an individual
    coefficient. (A ratio penalty like `1 - |c_i/c_j|` both explodes as c_j -> 0
    and pins the ratio at 1, when the ratio should be a free shared parameter --
    here it comes out as the quotient of two entries of v_k.)

    Computed **pairwise-complete**: only over participants where both terms are
    present. This is the step a factorization of the coefficient matrix cannot
    do, and the reason it confounds "term absent for this participant" with
    "term present but small". Pairs with fewer than ``min_overlap`` shared
    participants get distance 1 (never merged) rather than a correlation
    estimated from nothing.
    """
    N, T = coefficients.shape
    D = np.ones((T, T))
    np.fill_diagonal(D, 0.0)
    for i in range(T):
        for j in range(i + 1, T):
            both = support[:, i] & support[:, j]
            if both.sum() < min_overlap:
                continue
            ci, cj = coefficients[both, i], coefficients[both, j]
            nii, njj = float(ci @ ci), float(cj @ cj)
            if nii < 1e-24 or njj < 1e-24:
                continue
            d = 1.0 - float(ci @ cj) ** 2 / (nii * njj)
            D[i, j] = D[j, i] = max(0.0, d)
    return D


def initial_partition(D: np.ndarray, K: int) -> np.ndarray:
    """Agglomerative clustering of columns on the precomputed distance matrix.

    Average linkage, deterministic -- no seed, so the same checkpoint always
    yields the same starting partition and any instability observed downstream
    is a property of the data rather than of the initialization.
    """
    from sklearn.cluster import AgglomerativeClustering

    T = D.shape[0]
    if K >= T:
        return np.arange(T)
    if K <= 1:
        return np.zeros(T, dtype=int)
    model = AgglomerativeClustering(n_clusters=K, metric='precomputed', linkage='average')
    return model.fit_predict(D)


# ---------------------------------------------------------------------------
# The concept model
# ---------------------------------------------------------------------------

@dataclass
class ConceptFit:
    """A fitted partition for one module."""

    module: str
    labels: np.ndarray            # (T,) block id per term
    directions: List[np.ndarray]  # per block, (|S_k|,) unit norm
    theta: torch.Tensor           # (U, K)
    gates: torch.Tensor           # (U, K) bool
    rss: torch.Tensor             # (U,)
    stats: ModuleStats = field(repr=False)
    sigma2: float = 1.0
    complexity_weight: float = 1.0

    @property
    def K(self) -> int:
        return len(self.directions)

    def blocks(self) -> List[np.ndarray]:
        return [np.where(self.labels == k)[0] for k in range(self.K)]

    def coefficients(self) -> torch.Tensor:
        """(U, T) reconstruction from gates, loadings and directions."""
        out = torch.zeros(self.stats.U, self.stats.T, dtype=torch.float64)
        for k, idx in enumerate(self.blocks()):
            if len(idx) == 0:
                continue
            amp = self.theta[:, k] * self.gates[:, k].double()      # (U,)
            v = torch.tensor(self.directions[k], dtype=torch.float64)
            out[:, idx] = amp[:, None] * v[None, :]
        return out

    # -- parameter accounting ------------------------------------------------

    def n_loadings_per_unit(self) -> torch.Tensor:
        """(U,) open gates -- the per-participant free parameters."""
        return self.gates.sum(dim=1).double()

    def n_shared(self) -> int:
        """Direction values shared by the whole population (unit norm removes one per block)."""
        return int(sum(max(0, len(idx) - 1) for idx in self.blocks()))

    def structure_nats(self) -> float:
        """Description length of the structure itself.

        Two parts. The partition: each of T terms carries a block label,
        `T log K`, paid **once** for the population -- against `log C(T, k_p)`
        *per participant* for an arbitrary support, which is the fragmentation
        cost an ordinary BIC cannot see. And the gate pattern, coded at each
        concept's empirical prevalence, so a concept everyone has (or nobody
        has) is nearly free while a concept splitting the population 50/50 costs
        one bit per participant -- which is exactly when it is a real structural
        individual difference.
        """
        T, K, U = self.stats.T, self.K, self.stats.U
        partition = T * math.log(max(K, 1)) + math.log(max(T, 1))
        gate = 0.0
        for k in range(K):
            n_open = float(self.gates[:, k].sum())
            pi = min(max(n_open / U, 1e-6), 1 - 1e-6)
            gate += -(n_open * math.log(pi) + (U - n_open) * math.log(1 - pi))
        gate += K * 0.5 * math.log(max(U, 2))   # coding the prevalences themselves
        return partition + gate

    def mdl(self) -> Dict[str, float]:
        """Gaussian state-space BIC plus the structure code, in BIC units (2*nats).

        ``complexity_weight`` scales the whole complexity side against the fit
        side. It is not a free knob for its own sake: the nominal row count
        badly overstates the *effective* sample size of this objective. Each
        trial contributes one row per item, and an item's rows are strongly
        autocorrelated across consecutive trials (a value trace is smooth by
        construction), so the residuals are nowhere near the `n_rows`
        independent observations a textbook Gaussian BIC assumes. On
        dezfouli2019, `n_rows` is exactly twice the trial count before
        autocorrelation is even considered. A weight of w is equivalent to
        declaring the effective sample size to be `n_rows / w`, or the true
        noise variance to be `w * sigma2`. Sweeping it and selecting on the
        *behavioural* criterion is the honest way to set it; a per-sample
        Gauss-Newton weighting of the state residual would fix it from first
        principles, and is the intended replacement.
        """
        n_rows = self.stats.n_rows.clamp_min(1.0)
        w = self.complexity_weight
        fit = float((self.rss / self.sigma2).sum())
        loadings = w * float((self.n_loadings_per_unit() * torch.log(n_rows)).sum())
        shared = w * self.n_shared() * math.log(float(n_rows.sum()))
        structure = w * 2.0 * self.structure_nats()
        return dict(total=fit + loadings + shared + structure, fit=fit,
                    loadings=loadings, shared=shared, structure=structure,
                    rss=float(self.rss.sum()))


def _build_embed(labels: np.ndarray, directions: List[np.ndarray], T: int) -> torch.Tensor:
    """(T, K) block-sparse matrix whose column k is v_k embedded in the full term space."""
    K = len(directions)
    V = torch.zeros(T, K, dtype=torch.float64)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx):
            V[idx, k] = torch.tensor(directions[k], dtype=torch.float64)
    return V


def _init_directions(stats: ModuleStats, labels: np.ndarray, coefficients: np.ndarray) -> List[np.ndarray]:
    """Leading right singular vector of each block's coefficient sub-matrix.

    Sign convention: the largest-magnitude entry is made positive, so a concept's
    direction is reproducible across runs and `theta >= 0` means "has the
    concept in its canonical orientation".
    """
    directions = []
    for k in range(labels.max() + 1 if labels.size else 0):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            directions.append(np.zeros(0))
            continue
        block = coefficients[:, idx]
        if np.abs(block).max() < 1e-12:
            v = np.zeros(len(idx))
            v[0] = 1.0
        else:
            _, _, Vt = np.linalg.svd(block, full_matrices=False)
            v = Vt[0]
            if v[np.argmax(np.abs(v))] < 0:
                v = -v
            v = v / max(np.linalg.norm(v), 1e-12)
        directions.append(v)
    return directions


def als(
    stats: ModuleStats, labels: np.ndarray, directions: List[np.ndarray],
    gates: torch.Tensor, n_iter: int = 12, sigma2: float = 1.0,
    complexity_weight: float = 1.0,
) -> ConceptFit:
    """Alternating least squares for loadings and directions.

    Both half-steps are exact linear solves on the quadratic state-space
    objective, so this is bi-convex and descends monotonically:

      theta step (per unit):  (V^T G_u V) theta_u = V^T b_u          -- K x K
      v step (all blocks):    [sum_u diag(a_u) G_u diag(a_u)] w = sum_u a_u * b_u

    where a_u broadcasts each unit's loading onto the terms of its block, and w
    stacks every block's direction (disjointness is what keeps this one T x T
    system rather than a coupled mess). Directions are renormalized to unit norm
    after each sweep with the scale folded back into theta, which fixes the
    (v -> cv, theta -> theta/c) gauge freedom that would otherwise let the two
    half-steps drift against each other.
    """
    T, U = stats.T, stats.U
    directions = [d.copy() for d in directions]
    gate_d = gates.double()

    theta = torch.zeros(U, len(directions), dtype=torch.float64)
    for _ in range(max(1, n_iter)):
        # -- theta step ------------------------------------------------------
        V = _build_embed(labels, directions, T)                      # (T, K)
        GV = stats.G @ V                                             # (U, T, K)
        A = V.T @ GV                                                 # (U, K, K)
        r = torch.einsum('tk,ut->uk', V, stats.b)                    # (U, K)
        mask = gate_d                                                # (U, K)
        A = A * mask[:, :, None] * mask[:, None, :]
        # Closed gates get a unit diagonal and zero rhs so the batched solve
        # stays non-singular and returns exactly zero for them.
        closed = (mask < 0.5)
        A = A + torch.diag_embed(closed.double())
        theta = _solve(_ridged(A, rel=1e-10), r * mask)
        theta = theta * mask

        # -- direction step --------------------------------------------------
        block_of = torch.tensor(labels, dtype=torch.long)
        a = (theta * mask)[:, block_of]                              # (U, T)
        M = (a[:, :, None] * a[:, None, :] * stats.G).sum(dim=0)     # (T, T)
        rhs = (a * stats.b).sum(dim=0)                               # (T,)
        w = _solve(_ridged(M[None], rel=1e-10)[0], rhs)
        for k, idx in enumerate([np.where(labels == kk)[0] for kk in range(len(directions))]):
            if len(idx) == 0:
                continue
            v = w[idx].numpy()
            norm = float(np.linalg.norm(v))
            if norm < 1e-12:
                continue
            v = v / norm
            if v[np.argmax(np.abs(v))] < 0:
                v, norm = -v, -norm
            directions[k] = v
            theta[:, k] = theta[:, k] * norm

    coefficients = torch.zeros(U, T, dtype=torch.float64)
    for k, idx in enumerate([np.where(labels == kk)[0] for kk in range(len(directions))]):
        if len(idx) == 0:
            continue
        amp = theta[:, k] * gate_d[:, k]
        coefficients[:, idx] = amp[:, None] * torch.tensor(directions[k], dtype=torch.float64)[None, :]
    rss = (stats.btb
           + torch.einsum('ut,utv,uv->u', coefficients, stats.G, coefficients)
           - 2 * (coefficients * stats.b).sum(-1)).clamp_min(0.0)

    return ConceptFit(module=stats.module, labels=labels, directions=directions,
                      theta=theta, gates=gates.clone(), rss=rss, stats=stats, sigma2=sigma2,
                      complexity_weight=complexity_weight)


def update_gates(fit: ConceptFit, min_prevalence: int = 2) -> torch.Tensor:
    """Close every gate whose loading does not pay for itself, in closed form.

    Dropping loading k from a unit's K-dimensional quadratic raises the residual
    by the standard restriction cost `theta_k^2 / (A^-1)_kk` -- no refit, no
    forward pass. It is worth keeping exactly when that exceeds the BIC price of
    one parameter, `sigma^2 log(n_rows)`. Directions the unit's own data never
    constrained have a large `(A^-1)_kk` and so close automatically, which is the
    intended behaviour: a parameter the likelihood cannot see should not be
    charged for, nor reported as an individual difference.

    Concepts left with fewer than ``min_prevalence`` users are closed entirely
    (they are then dropped by `refine_partition`, which merges their terms
    elsewhere).
    """
    stats = fit.stats
    V = _build_embed(fit.labels, fit.directions, stats.T)
    A = V.T @ (stats.G @ V)                                          # (U, K, K)
    r = torch.einsum('tk,ut->uk', V, stats.b)                        # (U, K)
    A_reg = _ridged(A, rel=1e-10)
    theta_free = _solve(A_reg, r)                                    # unconstrained
    A_inv_diag = torch.diagonal(torch.linalg.pinv(A_reg), dim1=-2, dim2=-1).clamp_min(1e-30)
    delta_rss = theta_free ** 2 / A_inv_diag                         # (U, K)

    price = (fit.complexity_weight * fit.sigma2
             * torch.log(stats.n_rows.clamp_min(2.0))[:, None])
    gates = delta_rss > price

    keep = gates.sum(dim=0) >= min_prevalence
    gates = gates & keep[None, :]
    return gates


# ---------------------------------------------------------------------------
# Partition refinement
# ---------------------------------------------------------------------------

def refine_partition(
    stats: ModuleStats, labels: np.ndarray, directions: List[np.ndarray], gates: torch.Tensor,
    sigma2: float, n_sweeps: int = 3, als_iter: int = 6, verbose: bool = False,
    complexity_weight: float = 1.0,
) -> ConceptFit:
    """Coordinate descent on the MDL score over term-to-block assignments.

    Each sweep offers every term a move to every other block (and to a block of
    its own), rescoring with a short ALS + gate update. Moves are accepted
    greedily on the total MDL. This is a local search over a combinatorial space,
    so it finds a local optimum -- the deterministic column-clustering start is
    what keeps it a reproducible one.
    """
    cw = complexity_weight
    labels = labels.copy()
    best = als(stats, labels, directions, gates, n_iter=als_iter, sigma2=sigma2, complexity_weight=cw)
    best.gates = update_gates(best)
    best = als(stats, best.labels, best.directions, best.gates, n_iter=als_iter,
               sigma2=sigma2, complexity_weight=cw)
    best_score = best.mdl()['total']

    for sweep in range(n_sweeps):
        improved = False
        for term in range(stats.T):
            current = labels[term]
            targets = sorted(set(labels.tolist()) | {int(labels.max()) + 1})
            for target in targets:
                if target == current:
                    continue
                trial_labels = labels.copy()
                trial_labels[term] = target
                trial_labels = _compact(trial_labels)
                coefficients = best.coefficients().numpy()
                trial_dirs = _init_directions(stats, trial_labels, coefficients)
                trial_gates = _resize_gates(best.gates, len(trial_dirs))
                trial = als(stats, trial_labels, trial_dirs, trial_gates, n_iter=als_iter,
                            sigma2=sigma2, complexity_weight=cw)
                trial.gates = update_gates(trial)
                trial = als(stats, trial.labels, trial.directions, trial.gates,
                            n_iter=als_iter, sigma2=sigma2, complexity_weight=cw)
                score = trial.mdl()['total']
                if score < best_score - 1e-6:
                    best, best_score, labels = trial, score, trial.labels.copy()
                    improved = True
        if verbose:
            print(f"    sweep {sweep + 1}: MDL {best_score:.1f}  K={best.K}")
        if not improved:
            break
    return best


def _compact(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to 0..K-1 with no gaps (a move can empty a block)."""
    unique = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.array([unique[v] for v in labels.tolist()])


def _resize_gates(gates: torch.Tensor, K: int) -> torch.Tensor:
    """Grow or shrink the gate matrix to K columns, new concepts starting open."""
    U, K_old = gates.shape
    if K == K_old:
        return gates.clone()
    if K < K_old:
        return gates[:, :K].clone()
    extra = torch.ones(U, K - K_old, dtype=torch.bool)
    return torch.cat([gates, extra], dim=1)


# ---------------------------------------------------------------------------
# Top-level discovery
# ---------------------------------------------------------------------------

def discover_concepts(
    stats: ModuleStats, coefficients: np.ndarray, support: np.ndarray,
    K_grid: Optional[Sequence[int]] = None, n_sweeps: int = 3, verbose: bool = True,
    complexity_weight: float = 1.0, sigma2: Optional[float] = None,
) -> ConceptFit:
    """Full discovery for one module: cluster, refine, select K by MDL.

    ``coefficients``/``support`` are the checkpoint's ensemble-aggregated
    coefficients and consensus presence mask, (N, T) with N = P*X, used only to
    seed the partition; the fit itself is against the sufficient statistics.
    """
    sigma2 = noise_scale(stats) if sigma2 is None else sigma2
    D = column_distances(coefficients, support)
    if K_grid is None:
        # Both directions of search: a coarse start that the sweeps can split,
        # and an all-singleton start that they can merge. Refinement is a local
        # search, so the two paths do not always meet at the same optimum.
        K_grid = sorted(set([max(2, stats.T // 3), stats.T]))

    best, best_score = None, math.inf
    for K in K_grid:
        labels = _compact(initial_partition(D, K))
        directions = _init_directions(stats, labels, coefficients)
        gates = torch.ones(stats.U, len(directions), dtype=torch.bool)
        fit = refine_partition(stats, labels, directions, gates, sigma2,
                               n_sweeps=n_sweeps, verbose=False,
                               complexity_weight=complexity_weight)
        score = fit.mdl()['total']
        if verbose:
            print(f"    K0={K:3d} -> K={fit.K:3d}  MDL={score:12.1f}  "
                  f"RSS={fit.mdl()['rss']:.4f}  loadings/unit={float(fit.n_loadings_per_unit().mean()):.2f}")
        if score < best_score:
            best, best_score = fit, score
    # Final polish at the selected structure
    best = als(stats, best.labels, best.directions, best.gates, n_iter=60, sigma2=sigma2,
               complexity_weight=complexity_weight)
    return best


# ---------------------------------------------------------------------------
# Installing a fit back into a model
# ---------------------------------------------------------------------------

def install_concepts(estimator, fits: Dict[str, ConceptFit]) -> None:
    """Overwrite the model's SINDy coefficients with the concept reconstruction.

    Presence is set to the reconstruction's own nonzeros -- the Boolean
    factorization `mask_pj = OR_k (z_pk AND [j in S_k])`. Note that
    `count_sindy_coefficients()` will then report the number of *terms* a
    participant carries, which is no longer their number of free parameters
    (that is their number of open gates); pass `dof_per_participant` to any
    BIC-style scoring rather than relying on the term count.
    """
    model = getattr(estimator, 'model', estimator)
    for module, fit in fits.items():
        stats = fit.stats
        recon = fit.coefficients()                                    # (U, T_reduced)
        full = torch.zeros(stats.U, model.sindy_coefficients[module].shape[-1], dtype=torch.float64)
        full[:, torch.tensor(stats.term_indices, dtype=torch.long)] = recon
        target = model.sindy_coefficients[module]
        shaped = full.reshape(stats.E, stats.P, stats.X, -1)
        target.data = shaped.to(target.dtype).to(target.device)
        model.sindy_coefficients_presence[module] = (
            (shaped != 0).to(model.sindy_coefficients_presence[module].dtype)
            .to(model.sindy_coefficients_presence[module].device)
        )


class ConceptParameterization(torch.nn.Module):
    """Differentiable map from (theta, directions) to full coefficient tensors.

    The partition and the gates are frozen buffers; only the per-participant
    loadings and the shared directions are trainable. That is the whole point of
    the handoff: Stage 2.1 (one-step, where the objective is quadratic and the
    sufficient-statistic algebra holds) decides *structure*, and Stage 2.2
    (K-step shooting, where it does not) only estimates the numbers inside it.

    Directions are re-normalized to unit length inside `build_coefficients`
    rather than by projection after each step, so the (v -> cv, theta -> theta/c)
    gauge freedom is removed from the optimization instead of being fought by it.
    """

    def __init__(self, fits: Dict[str, ConceptFit], n_terms_full: Dict[str, int], device=None):
        super().__init__()
        self.module_names = list(fits.keys())
        self.shapes = {}
        self._blocks = {}
        for name, fit in fits.items():
            stats = fit.stats
            self.shapes[name] = (stats.E, stats.P, stats.X, n_terms_full[name])
            theta0 = (fit.theta * fit.gates.double()).to(torch.float32)
            self.register_parameter(f"theta_{name}", torch.nn.Parameter(theta0.to(device)))
            packed = torch.zeros(stats.T, dtype=torch.float32)
            blocks = []
            for k, idx in enumerate(fit.blocks()):
                if len(idx) == 0:
                    blocks.append(None)
                    continue
                packed[idx] = torch.tensor(fit.directions[k], dtype=torch.float32)
                blocks.append(torch.tensor(idx, dtype=torch.long, device=device))
            self._blocks[name] = blocks
            self.register_parameter(f"w_{name}", torch.nn.Parameter(packed.to(device)))
            self.register_buffer(f"gates_{name}", fit.gates.to(torch.float32).to(device))
            self.register_buffer(f"cols_{name}",
                                 torch.tensor(stats.term_indices, dtype=torch.long, device=device))

    def build_coefficients(self) -> Dict[str, torch.Tensor]:
        out = {}
        for name in self.module_names:
            E, P, X, T_full = self.shapes[name]
            theta = getattr(self, f"theta_{name}")
            w = getattr(self, f"w_{name}")
            gates = getattr(self, f"gates_{name}")
            cols = getattr(self, f"cols_{name}")
            reduced = torch.zeros(theta.shape[0], cols.shape[0],
                                  dtype=theta.dtype, device=theta.device)
            for k, idx in enumerate(self._blocks[name]):
                if idx is None:
                    continue
                v = w[idx]
                v = v / v.norm().clamp_min(1e-8)
                amp = theta[:, k] * gates[:, k]
                reduced = reduced.index_add(
                    1, idx, amp[:, None] * v[None, :])
            full = torch.zeros(theta.shape[0], T_full, dtype=theta.dtype, device=theta.device)
            full = full.index_add(1, cols, reduced)
            out[name] = full.reshape(E, P, X, T_full)
        return out


def _shooting_windows(T: int, shooting_steps: int) -> Tuple[List[int], int]:
    """Window starts and effective K, identical to Stage 2.2's construction."""
    K = shooting_steps
    if K > 1:
        n_windows = T // K
        window_starts = [i * K for i in range(n_windows)]
        if T % K > 0 and T > K:
            window_starts.append(T - K)
        elif T % K > 0 and T <= K:
            window_starts = [0]
            K = T
    else:
        window_starts = list(range(T))
    return window_starts, K


def _shooting_loss(
    model, xs_train, state_trajectories, nan_mask, window_starts, K,
    batch_sessions, state_noise_std: float,
) -> Tuple[torch.Tensor, int]:
    """Stage 2.2's shooting objective for one batch of sessions.

    Mirrors `_run_shooting_epoch_vectorized`: windows folded into the batch
    dimension, K autoregressive SINDy steps from the RNN's recorded state, MSE
    against the recorded next states of every module in ``states_in_logit``,
    averaged over valid steps. Reimplemented rather than called because that
    function zeroes `model.zero_grad()` (which would miss parameters living
    outside the model) and, after stepping, writes `sindy_coefficients[m].data
    *= presence` -- under `functional_coefficients` that assigns into a computed
    temporary and silently does nothing. Returns the loss and the step count so
    the caller owns the backward pass.
    """
    T = nan_mask.shape[1]
    B_batch = len(batch_sessions)
    n_windows = len(window_starts)
    if B_batch * n_windows == 0:
        return torch.zeros((), device=model.device), 0

    ws = torch.tensor(window_starts, dtype=torch.long)
    session_idx = batch_sessions.repeat(n_windows)
    time_idx = ws.unsqueeze(1).expand(-1, B_batch).reshape(-1)

    current_state = {
        s: state_trajectories[s][:, :, session_idx, time_idx].to(model.device)
        for s in state_trajectories
    }

    total_loss = torch.zeros((), device=model.device)
    n_valid_steps = 0
    for k in range(K):
        t_k = time_idx + k
        in_bounds = t_k < T
        if not in_bounds.any():
            break
        t_k_safe = torch.clamp(t_k, max=T - 1)
        valid = (nan_mask[session_idx, t_k_safe] & in_bounds).to(model.device)
        if not valid.any():
            continue

        xs_step = xs_train[:, session_idx, t_k_safe].unsqueeze(2).to(model.device)
        if state_noise_std > 0:
            current_state = {s: v + state_noise_std * torch.randn_like(v)
                             for s, v in current_state.items()}

        _, next_state = model(xs_step, current_state)

        step_loss = torch.zeros((), device=model.device)
        for s_key in model.spice_config.states_in_logit:
            target = state_trajectories[s_key][:, :, session_idx, t_k_safe + 1].to(model.device)
            pred = next_state[s_key]
            mask = valid.view(1, 1, -1, 1).expand_as(pred)
            step_loss = step_loss + ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)

        total_loss = total_loss + step_loss
        n_valid_steps += 1
        current_state = next_state

    if n_valid_steps == 0:
        return torch.zeros((), device=model.device), 0
    return total_loss / n_valid_steps, n_valid_steps


def refit_concepts_shooting(
    estimator, fits: Dict[str, ConceptFit], xs: torch.Tensor, ys: torch.Tensor,
    shooting_steps: int = 20, epochs: int = 300, lr: float = 1e-2,
    patience: int = 30, min_delta: float = 1e-7, state_noise_std: float = 0.05,
    batch_size_sessions: Optional[int] = None, verbose: bool = True,
) -> Tuple[ConceptParameterization, Dict[str, float]]:
    """Stage 2.2 inside a frozen concept structure.

    Concept discovery runs one-step because that is where the objective is
    quadratic; the coefficients it produces are therefore fit to a different
    objective than the pipeline's Stage 2.2 uses. This closes that gap: the
    partition and gates stay exactly as discovered, and only the loadings and
    directions are re-estimated against the same K-step shooting loss on the
    frozen RNN's state trajectories that Stage 2.2 optimizes.

    Nothing here touches behavioural likelihood -- same reasoning as the rest of
    the module. The RNN is frozen throughout; only the concept parameters train.

    Returns the fitted parameterization and a small history dict. The model is
    left holding the best-scoring coefficients.
    """
    from spice.resources.sindy_ties import functional_coefficients
    from spice.resources.spice_training import _vectorize_state_sequential

    model = getattr(estimator, 'model', estimator)
    device = model.device
    E = model.ensemble_size

    # Freeze everything the RNN owns; only concept parameters are trainable.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Presence is the Boolean factorization of the frozen gates, not the
    # numerical nonzeros of the current reconstruction -- a direction entry that
    # happens to sit near zero must stay *inside* its block, or the refit could
    # never move it back.
    for name, fit in fits.items():
        stats = fit.stats
        mask = torch.zeros(stats.U, model.sindy_coefficients[name].shape[-1], dtype=torch.bool)
        for k, idx in enumerate(fit.blocks()):
            if len(idx) == 0:
                continue
            cols = torch.tensor(stats.term_indices[idx], dtype=torch.long)
            mask[:, cols] |= fit.gates[:, k][:, None]
        target = model.sindy_coefficients_presence[name]
        model.sindy_coefficients_presence[name] = (
            mask.reshape(stats.E, stats.P, stats.X, -1).to(target.dtype).to(target.device))

    model.eval(use_sindy=False)
    state_trajectories, nan_mask = _vectorize_state_sequential(model, xs, ys, verbose=verbose)
    T = nan_mask.shape[1]
    window_starts, K = _shooting_windows(T, shooting_steps)
    B = xs.shape[1]
    batch_size_sessions = batch_size_sessions or B

    n_terms_full = {m: model.sindy_coefficients[m].shape[-1] for m in fits}
    parameterization = ConceptParameterization(fits, n_terms_full, device=device).to(device)
    optimizer = torch.optim.Adam(parameterization.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

    def _evaluate() -> float:
        """Noise-free shooting loss over all sessions -- the selection criterion."""
        parameterization.eval()
        with torch.no_grad():
            built = parameterization.build_coefficients()
            for name, values in built.items():
                holder[name] = values
            model.eval(use_sindy=True)
            total, batches = 0.0, 0
            for start in range(0, B, batch_size_sessions):
                sessions = torch.arange(start, min(start + batch_size_sessions, B))
                loss, steps = _shooting_loss(model, xs, state_trajectories, nan_mask,
                                             window_starts, K, sessions, 0.0)
                if steps:
                    total += float(loss)
                    batches += 1
        return total / max(batches, 1)

    history = {}
    with functional_coefficients(model) as holder:
        best = dict(loss=_evaluate(), epoch=0,
                    state={k: v.detach().clone() for k, v in parameterization.state_dict().items()})
        history['initial'] = best['loss']
        if verbose:
            print(f"Stage 2.2 (concept-constrained, K={K}): initial shooting loss {best['loss']:.7f}")

        stale = 0
        pbar = range(1, epochs + 1)
        for epoch in pbar:
            parameterization.train()
            model.train(mode=True, use_sindy=True)
            for rnn_module in model.submodules_rnn.values():
                rnn_module.eval()

            perm = torch.randperm(B)
            epoch_loss, batches = 0.0, 0
            for start in range(0, B, batch_size_sessions):
                sessions = perm[start:start + batch_size_sessions]
                optimizer.zero_grad(set_to_none=True)
                # Rebuild every batch: the graph is consumed by each backward.
                for name, values in parameterization.build_coefficients().items():
                    holder[name] = values
                loss, steps = _shooting_loss(model, xs, state_trajectories, nan_mask,
                                             window_starts, K, sessions, state_noise_std)
                if steps == 0:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite shooting loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameterization.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += float(loss)
                batches += 1

            current = _evaluate()
            scheduler.step(current)
            if current < best['loss'] - min_delta:
                best = dict(loss=current, epoch=epoch,
                            state={k: v.detach().clone()
                                   for k, v in parameterization.state_dict().items()})
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:4d}  train {epoch_loss / max(batches, 1):.7f}  "
                      f"eval {current:.7f}  best {best['loss']:.7f} @{best['epoch']}  "
                      f"lr {optimizer.param_groups[0]['lr']:.1e}")

        parameterization.load_state_dict(best['state'])
        final = parameterization.build_coefficients()

    # Outside the context manager sindy_coefficients is a real ParameterDict
    # again, so write the values in for good.
    with torch.no_grad():
        for name, values in final.items():
            model.sindy_coefficients[name].data.copy_(
                values.detach().to(model.sindy_coefficients[name].dtype))

    history.update(best_loss=best['loss'], best_epoch=best['epoch'], K=K)
    if verbose:
        print(f"Stage 2.2 done: shooting loss {history['initial']:.7f} -> "
              f"{best['loss']:.7f} (epoch {best['epoch']})")
    return parameterization, history


def dof_per_participant(fits: Dict[str, ConceptFit], amortize_shared: bool = True) -> np.ndarray:
    """(P, X) free parameters per participant under the concept model.

    Per participant: one loading per open gate, summed over modules, averaged
    across ensemble members (each member is a valid model; this reports the
    typical one, matching `score_member`'s convention).

    ``amortize_shared`` adds each module's shared direction values spread over
    the population. Those are genuine population parameters, but dividing by P
    makes them nearly free, which biases any search toward pooling -- so the
    caller should report both settings rather than picking one.

    The structure code is deliberately *not* folded in here. It is a description
    length in nats, not a parameter count, and converting it would need the
    per-group trial count this function does not see; report it alongside
    (`ConceptFit.structure_nats`) instead of smuggling it into a `k`.
    """
    any_fit = next(iter(fits.values()))
    P, X, E = any_fit.stats.P, any_fit.stats.X, any_fit.stats.E
    dof = np.zeros((P, X))
    for fit in fits.values():
        per_unit = fit.n_loadings_per_unit().reshape(E, P, X).mean(dim=0).numpy()
        dof += per_unit
        if amortize_shared:
            dof += fit.n_shared() / (P * X)
    return dof


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def concept_string(fit: ConceptFit, threshold: float = 0.05) -> str:
    """Human-readable concepts: term set, direction, prevalence, mean amplitude."""
    stats = fit.stats
    lines = []
    order = sorted(range(fit.K), key=lambda k: -float(fit.gates[:, k].sum()))
    for k in order:
        idx = fit.blocks()[k]
        if len(idx) == 0:
            continue
        n_open = int(fit.gates[:, k].sum())
        prevalence = n_open / stats.U
        if n_open == 0:
            lines.append(f"  [dropped]  {{{', '.join(stats.term_names[i] for i in idx)}}}")
            continue
        amp = fit.theta[fit.gates[:, k], k]
        v = fit.directions[k]
        profile = ' '.join(
            f"{v[m]:+.3f} {stats.term_names[i]}"
            for m, i in enumerate(idx) if abs(v[m]) > threshold
        )
        sign_split = float((amp < 0).double().mean())
        flag = '  [SIGN CONFLICT]' if 0.15 < sign_split < 0.85 else ''
        lines.append(
            f"  C{k}  used by {n_open:4d}/{stats.U} ({prevalence:5.1%})  "
            f"theta = {float(amp.mean()):+.3f} +/- {float(amp.std()):.3f}{flag}\n"
            f"       {profile}"
        )
    return "\n".join(lines)


def summary_row(fit: ConceptFit) -> Dict[str, float]:
    mdl = fit.mdl()
    sizes = [len(b) for b in fit.blocks() if len(b) > 0]
    active = [k for k in range(fit.K) if int(fit.gates[:, k].sum()) > 0]
    return dict(
        module=fit.module, T=fit.stats.T, K=len(active),
        mean_block_size=float(np.mean(sizes)) if sizes else 0.0,
        max_block_size=int(max(sizes)) if sizes else 0,
        loadings_per_unit=float(fit.n_loadings_per_unit().mean()),
        n_shared=fit.n_shared(), rss=mdl['rss'], mdl=mdl['total'],
        structure_nats=fit.structure_nats(),
    )

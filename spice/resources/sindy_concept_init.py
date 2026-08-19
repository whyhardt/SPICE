"""Data-driven seeding for the concept dictionary.

Salvaged from the post-hoc `sindy_concepts` module, which was retired once the
factorization moved into training itself. These pieces stay useful for two things
the sign-split initialization does not cover:

  * seeding V from a *fitted* coefficient matrix (stage 2.1, where a ridge solution
    already exists) instead of from the term-wise sign split, by clustering the
    coefficient columns and taking each block's leading singular vector;
  * the one-step sufficient statistics (G_u = Theta^T Theta, b_u = Theta^T y), which
    make the K=1 objective exactly quadratic in the coefficients and are the basis
    for a closed-form alternating least squares refit.

Nothing here is wired into the training loop yet.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


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
    from spice.resources.training.ridge import _ridge_solve_sindy

    model = getattr(estimator, 'model', estimator)
    modules = model.get_modules()

    saved_gates = {m: model.sindy_concept_gates[m].clone() for m in modules}
    saved_support = {m: model.sindy_concept_support[m].clone() for m in modules}
    saved_loadings = {m: model.sindy_concept_loadings[m].data.clone() for m in modules}
    saved_directions = {m: model.sindy_concept_directions[m].data.clone() for m in modules}
    try:
        # Accumulate on the *unmasked* library: reset to the sign-split so derived
        # presence covers every theory-allowed term.
        model.reset_concepts()
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
                    "Ridge accumulators lack 'btb'/'n_rows'. This needs the "
                    "absolute residual (for the Gaussian noise scale), not just the "
                    "relative one -- update BaseModel.sindy_ridge_accumulate."
                )
            T_full = accum['AtA'].shape[-1]
            allowed = model.sindy_term_prior_mask[m]
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
            model.sindy_concept_gates[m] = saved_gates[m]
            model.sindy_concept_support[m] = saved_support[m]
            model.sindy_concept_loadings[m].data = saved_loadings[m]
            model.sindy_concept_directions[m].data = saved_directions[m]
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

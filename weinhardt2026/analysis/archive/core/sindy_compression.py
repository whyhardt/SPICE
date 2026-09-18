"""
Low-rank compression of per-participant SINDy coefficients.

A fitted SPICE model gives every participant a free coefficient for each
active SINDy term across all modules. Many of these terms co-vary strongly
across the population -- they are not independent axes of individual
difference, just different symptoms of the same underlying trait. This
module reparameterizes the per-participant coefficient set as:

    coefficients[participant] ~= population_mean + loadings[participant] @ components

i.e. a population-average equation shared by everyone, plus K numbers per
participant (their loading on K shared "mechanisms", each mechanism being a
fixed direction in coefficient space). Two ways to fit the K components:

  - ``method="svd"``: dense PCA/SVD. Optimal reconstruction fidelity for a
    given K, but every participant gets a nonzero loading on every
    mechanism -- individual differences show up as continuous shifts along
    dense axes, not as presence/absence of a mechanism.
  - ``method="sparse"``: sparse dictionary learning (L1-penalized codes).
    Each participant activates only a handful of the K mechanisms, mirroring
    the structural, presence/absence individual differences that SPICE's
    own SINDy pruning already produces at the raw-term level -- at some
    cost to reconstruction fidelity (see `fit_sparse_dictionary`).

This module only fits and symbolically prints the reparameterization. It
does not evaluate predictive performance (that needs train/test datasets
and study-specific scoring, e.g. NLL/BIC) -- see
`weinhardt2026/analysis/analysis_coefficient_compression.py` for a sweep
over K (and, for the sparse method, alpha) that measures held-out
predictive cost.
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_joint_coefficient_matrix(spice_model) -> Tuple[np.ndarray, List[Tuple[str, str]], List[Tuple[str, int, int]], int, int]:
    """Stack ensemble-averaged coefficients from all modules into one matrix.

    ``spice_model`` may be a `SpiceEstimator` or a `BaseModel` -- anything
    exposing `get_modules()`, `get_candidate_terms()`, and
    `get_sindy_coefficients(aggregate=True)`.

    Returns
    -------
    C : (P*X, T_total) array -- rows are (participant, experiment) pairs,
        columns are every candidate term from every module (incl. intercept).
    col_labels : list of (module, term) per column.
    col_slices : list of (module, start, end) column ranges, for writing
        reconstructions back into the model's per-module tensors.
    P, X : number of participants / experiments in the model.
    """
    modules = spice_model.get_modules()
    terms = spice_model.get_candidate_terms()
    coefs = spice_model.get_sindy_coefficients(aggregate=True)  # module -> (P, X, T)

    P, X = next(iter(coefs.values())).shape[:2]

    col_labels: List[Tuple[str, str]] = []
    col_slices: List[Tuple[str, int, int]] = []
    blocks = []
    offset = 0
    for module in modules:
        c = coefs[module].reshape(P * X, -1).detach().cpu().numpy()  # (P*X, T_m)
        T_m = c.shape[1]
        blocks.append(c)
        col_labels += [(module, t) for t in terms[module]]
        col_slices.append((module, offset, offset + T_m))
        offset += T_m

    C = np.concatenate(blocks, axis=1)  # (P*X, T_total)
    return C, col_labels, col_slices, P, X


def extract_joint_coefficient_matrix_per_ensemble_member(
    spice_model,
) -> Tuple[np.ndarray, List[Tuple[str, str]], List[Tuple[str, int, int]], int, int, int]:
    """Like `extract_joint_coefficient_matrix`, but one row per (ensemble
    member, participant, experiment) instead of ensemble-averaged rows.

    `.commit()`/`.apply()` normally broadcast one reconstructed coefficient
    set to every ensemble member -- fine for evaluating the *cost of
    compression*, but it throws away whatever real predictive benefit the
    ensemble's diversity was providing (can be large: on eckstein2026,
    collapsing the ensemble alone drops test ΔBIC/trial from ~1.76 to
    ~0.33, before any compression at all). Fitting on ensemble-member-level
    rows instead lets the factorization discover a *shared* dictionary
    across all members while keeping genuinely different loadings *per
    member* -- `commit_full_ensemble` writes those back without collapsing,
    restoring ensemble diversity in the committed model. For inspection/
    naming/group-analysis purposes (where one number per participant is
    wanted), average the returned loadings over the ensemble axis to get
    something directly usable by `CompressedSpiceModel`, which expects
    (P*X, K)-shaped loadings.

    Returns
    -------
    C : (E*P*X, T_total) array -- row order matches ``.reshape(E, P, X, -1)``
        (C-contiguous, i.e. ensemble-member-major).
    col_labels, col_slices : same as `extract_joint_coefficient_matrix`.
    P, X, E : number of participants / experiments / ensemble members.
    """
    modules = spice_model.get_modules()
    terms = spice_model.get_candidate_terms()
    coefs = spice_model.get_sindy_coefficients(aggregate=False)  # module -> (E, P, X, T), presence-masked

    E, P, X = next(iter(coefs.values())).shape[:3]

    col_labels: List[Tuple[str, str]] = []
    col_slices: List[Tuple[str, int, int]] = []
    blocks = []
    offset = 0
    for module in modules:
        c = coefs[module].reshape(E * P * X, -1).detach().cpu().numpy()  # (E*P*X, T_m)
        T_m = c.shape[1]
        blocks.append(c)
        col_labels += [(module, t) for t in terms[module]]
        col_slices.append((module, offset, offset + T_m))
        offset += T_m

    C = np.concatenate(blocks, axis=1)  # (E*P*X, T_total)
    return C, col_labels, col_slices, P, X, E


def commit_full_ensemble(
    spice_model, mean_vec: np.ndarray, components: np.ndarray, loadings_full: np.ndarray,
    col_slices: List[Tuple[str, int, int]], P: int, X: int, E: int,
) -> None:
    """Write a per-ensemble-member compressed reconstruction back into
    ``spice_model`` *without* collapsing the ensemble -- the direct-write
    counterpart to `CompressedSpiceModel.commit`, for use with
    `extract_joint_coefficient_matrix_per_ensemble_member`-fitted results
    (``loadings_full`` has ``E*P*X`` rows, not ``P*X``).

    Restores genuine ensemble diversity in the committed model (each member
    gets its own reconstruction from its own loadings against the shared
    dictionary), recovering most of the predictive benefit `.commit()`
    otherwise gives up. Marks every term present, same convention as
    `CompressedSpiceModel._write`.
    """
    C_hat = mean_vec + loadings_full @ components  # (E*P*X, T_total)
    model = spice_model.model
    for module, start, end in col_slices:
        orig_coef = model.sindy_coefficients[module].data
        vals = torch.tensor(C_hat[:, start:end], dtype=orig_coef.dtype, device=orig_coef.device)
        vals = vals.reshape(E, P, X, -1)
        model.sindy_coefficients[module].data = vals.clone()
        model.sindy_coefficients_presence[module].data = torch.ones_like(model.sindy_coefficients_presence[module].data)


# ---------------------------------------------------------------------------
# Fitting: dense (SVD) and sparse (dictionary learning)
# ---------------------------------------------------------------------------

def fit_low_rank_basis(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SVD of the population-centered coefficient matrix.

    Returns mean_vec (T_total,), U (N, r), S (r,), Vt (r, T_total) such that
    ``C ~= mean_vec + (U * S) @ Vt``. Computed once at full rank; truncate to
    any K afterwards with `svd_components_and_loadings` without refitting.
    """
    mean_vec = C.mean(axis=0)
    U, S, Vt = np.linalg.svd(C - mean_vec, full_matrices=False)
    return mean_vec, U, S, Vt


def svd_components_and_loadings(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate a full SVD to the top K components: (components, loadings)."""
    if K <= 0:
        return Vt[:0], np.zeros((U.shape[0], 0))
    return Vt[:K, :], U[:, :K] * S[:K]


def fit_sparse_dictionary(
    C: np.ndarray, K: int, alpha: float = 1.0, max_iter: int = 1000, random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse dictionary learning: ``C ~= mean_vec + codes @ dictionary``, with
    an L1 penalty on ``codes`` so each participant activates only a handful of
    the K dictionary atoms ("mechanisms") instead of all of them (as dense
    SVD/PCA does). Handles signed coefficients directly, unlike NMF (which
    requires non-negativity).

    ``alpha`` controls the sparsity penalty strength: higher alpha -> fewer
    active mechanisms per participant, at some cost to reconstruction
    fidelity. Unlike SVD, refitting is required for every (K, alpha) pair.

    Returns mean_vec (T_total,), dictionary (K, T_total), codes (N, K).
    """
    from sklearn.decomposition import DictionaryLearning

    mean_vec = C.mean(axis=0)
    centered = C - mean_vec
    model = DictionaryLearning(
        n_components=K, alpha=alpha, max_iter=max_iter, random_state=random_state,
        fit_algorithm="lars", transform_algorithm="lasso_lars",
    )
    codes = model.fit_transform(centered)  # (N, K)
    dictionary = model.components_  # (K, T_total)
    return mean_vec, dictionary, codes


def fit_sparse_dictionary_per_module(
    C: np.ndarray, col_labels: List[Tuple[str, str]], col_slices: List[Tuple[str, int, int]],
    K_per_module: int, alpha: float = 1.0, max_iter: int = 1000, random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Sparse dictionary learning fit independently within each module's own
    term block, instead of jointly across all modules (`fit_sparse_dictionary`).

    Guarantees every mechanism is confined to one module by construction --
    the same localization property as `fit_term_family_basis` -- but *which*
    terms combine into a mechanism, and in what proportion, is learned from
    data via sparse coding, not asserted by a hand-designed term taxonomy.
    An L1 penalty on the codes (``alpha``) still drives sparse per-participant
    usage, same role as in `fit_sparse_dictionary`.

    Returns mean_vec (T_total,), components (K_total, T_total, module-block-
    sparse by construction), loadings (N, K_total), mechanism_names (e.g.
    "value_reward_chosen: mechanism 3" -- unnamed since composition is
    data-driven, not a fixed category; inspect `mechanisms_string()` for
    what each one actually contains).
    """
    from sklearn.decomposition import DictionaryLearning

    mean_vec = C.mean(axis=0)
    residual = C - mean_vec
    T_total = C.shape[1]

    components_blocks = []
    loadings_blocks = []
    mechanism_names = []
    for module, start, end in col_slices:
        block = residual[:, start:end]  # (N, T_m)
        k = min(K_per_module, block.shape[1])
        model = DictionaryLearning(
            n_components=k, alpha=alpha, max_iter=max_iter, random_state=random_state,
            fit_algorithm="lars", transform_algorithm="lasso_lars",
        )
        codes = model.fit_transform(block)  # (N, k)
        dictionary = model.components_      # (k, T_m)
        for j in range(k):
            full_component = np.zeros(T_total)
            full_component[start:end] = dictionary[j]
            components_blocks.append(full_component)
            loadings_blocks.append(codes[:, j])
            mechanism_names.append(f"{module}: mechanism {j}")

    components = np.stack(components_blocks, axis=0) if components_blocks else np.zeros((0, T_total))
    loadings = np.stack(loadings_blocks, axis=1) if loadings_blocks else np.zeros((C.shape[0], 0))
    return mean_vec, components, loadings, mechanism_names


def _normalize_atoms(codes: np.ndarray, dictionary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale (codes, dictionary) so every dictionary atom has unit L2 norm.

    ``sklearn.decomposition.NMF`` (unlike ``DictionaryLearning``, which
    projects atoms onto the unit ball at every iteration) does not constrain
    atom scale. When only the codes are penalized (``alpha_W > 0``,
    ``alpha_H == 0``), the optimizer can trivially game that penalty by
    scaling a dictionary atom up by some factor and the matching code column
    down by the same factor -- ``codes @ dictionary`` (and therefore
    reconstruction/NLL) is exactly invariant to this rescaling, but it makes
    the L1 penalty on codes nearly free and any fixed sparsity threshold on
    codes meaningless, while the printed coefficients blow up to physically
    nonsensical magnitudes. This is *not* a hypothetical: it happened during
    hyperparameter search on dezfouli2019 (a small ``alpha_W`` produced
    coefficients like -335, 208 instead of the true O(0.1-1) scale). Fixing
    every atom to unit norm removes the ambiguity without changing
    reconstruction at all.

    Raises if either input already contains NaN: a degenerate/near-singular
    fit (e.g. a module whose coefficient matrix has fewer independent
    directions than the requested K, more likely with a heavily-pruned
    input) can make sklearn's NMF solver emit NaN silently. Left unchecked,
    that NaN propagates through `.commit()`/evaluation and (via
    `nansum`-style skipping deeper in the scoring code) can produce a
    *numerically better-looking* score than any valid fit -- this happened
    during a real sweep (K_per_module=8 on a heavily-pruned dezfouli2019
    checkpoint scored ~1.38, far above every legitimate configuration,
    purely from NaN predictions). Don't remove this check to "fix" a NaN
    error -- fix the fit (different K, alpha, or random_state) instead.
    """
    if np.isnan(codes).any() or np.isnan(dictionary).any():
        raise RuntimeError(
            "NMF fit produced NaN (codes or dictionary) -- likely a degenerate/"
            "near-singular fit for this K_per_module on this coefficient matrix "
            "(common with heavily-pruned inputs where a module's rank is less "
            "than K). Try a different K_per_module, alpha, or random_state rather "
            "than suppressing this error -- a NaN reconstruction can silently "
            "score as spuriously excellent downstream."
        )
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return codes * norms.T, dictionary / norms


def fit_nmf_signsplit(
    C: np.ndarray, K: int, alpha_W: float = 0.0, alpha_H: float = 0.0, max_iter: int = 1000, random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NMF on sign-split coefficients, as an alternative to `fit_sparse_dictionary`.

    NMF requires non-negative input and non-negative factors, but SINDy
    coefficients are signed. Each population-centered residual is split into
    a non-negative positive part and a non-negative negative part
    (``residual = pos - neg``, both >= 0) before fitting; the resulting
    dictionary is recombined back into signed components afterward
    (``components = H_pos - H_neg``), so the returned components/loadings
    are drop-in compatible with `CompressedSpiceModel` and
    `fit_sparse_dictionary`. ``alpha_W`` penalizes the (non-negative)
    loadings, same role as `fit_sparse_dictionary`'s ``alpha`` -- driving
    sparse per-participant mechanism *usage*. ``alpha_H`` penalizes the
    dictionary itself, driving sparse mechanism *definitions* (fewer terms
    per mechanism) -- nothing else tried so far constrains this; SVD's
    orthogonality doesn't shrink term-count, and plain sparse dictionary
    learning only penalizes the codes.

    Caveat (2026-07-21): non-negativity here applies to the *sign-split*
    codes/dictionary, not to the recombined signed components actually used
    for prediction -- two different mechanisms can and empirically do land
    on opposite signs for the same term, which cancel when both are active
    for a participant. This is *not* the classic NMF "parts-based, no
    cancellation" guarantee; that guarantee doesn't survive the
    split-then-recombine step. See `fit_nmf_signsplit_per_module_exclusive`
    for a soft (not guaranteed) penalty against this.

    Returns mean_vec (T_total,), components (K, T_total, signed), loadings (N, K, >= 0).
    """
    from sklearn.decomposition import NMF

    mean_vec = C.mean(axis=0)
    residual = C - mean_vec
    T_total = C.shape[1]
    split = np.concatenate([np.clip(residual, 0, None), np.clip(-residual, 0, None)], axis=1)  # (N, 2*T_total)

    model = NMF(
        n_components=K, init="nndsvda", alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=1.0,
        max_iter=max_iter, random_state=random_state,
    )
    loadings = model.fit_transform(split)  # (N, K), >= 0
    H = model.components_  # (K, 2*T_total), >= 0
    loadings, H = _normalize_atoms(loadings, H)
    components = H[:, :T_total] - H[:, T_total:]  # (K, T_total), signed
    return mean_vec, components, loadings


def fit_nmf_signsplit_per_module(
    C: np.ndarray, col_labels: List[Tuple[str, str]], col_slices: List[Tuple[str, int, int]],
    K_per_module: int = 6, alpha_W: float = 0.001, alpha_H: float = 0.0, max_iter: int = 1000, random_state: int = 0,
    center: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Sign-split NMF fit independently within each module's own term block,
    combining `fit_sparse_dictionary_per_module`'s module-localization
    (guaranteed by construction) with `fit_nmf_signsplit`'s dual sparsity
    control (``alpha_W`` for usage, ``alpha_H`` for how many terms each
    mechanism touches). See `fit_nmf_signsplit`'s docstring for the
    cross-mechanism cancellation caveat -- non-negativity here does not
    guarantee mechanisms never disagree in sign on a shared term.

    ``center`` controls whether a population-mean offset is subtracted
    before factorizing (``MODEL = mean + U @ H``) or not
    (``MODEL = U @ H`` directly, ``center=False``, the default). Centering
    keeps mechanisms from having to spend capacity reconstructing the
    population-shared baseline (see module docstring discussion), but folds
    part of the model into an uncounted "free" mean vector that sits outside
    the loading-based parameter count -- every participant's model is fully
    described by their ``U`` row against the shared ``H``, with no separate
    offset held outside that count.

    Returns mean_vec (T_total,, zeros when ``center=False``), components
    (K_total, T_total, module-block-sparse by construction, signed),
    loadings (N, K_total, >= 0), mechanism_names (e.g.
    "value_reward_chosen: mechanism 3").
    """
    from sklearn.decomposition import NMF

    T_total = C.shape[1]
    mean_vec = C.mean(axis=0) if center else np.zeros(T_total)
    residual = C - mean_vec

    components_blocks = []
    loadings_blocks = []
    mechanism_names = []
    for module, start, end in col_slices:
        block = residual[:, start:end]  # (N, T_m)
        T_m = end - start
        k = min(K_per_module, T_m)
        split = np.concatenate([np.clip(block, 0, None), np.clip(-block, 0, None)], axis=1)  # (N, 2*T_m)

        model = NMF(
            n_components=k, init="nndsvda", alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=1.0,
            max_iter=max_iter, random_state=random_state,
        )
        codes = model.fit_transform(split)  # (N, k)
        H = model.components_  # (k, 2*T_m)
        codes, H = _normalize_atoms(codes, H)
        dictionary = H[:, :T_m] - H[:, T_m:]  # (k, T_m), signed

        for j in range(k):
            full_component = np.zeros(T_total)
            full_component[start:end] = dictionary[j]
            components_blocks.append(full_component)
            loadings_blocks.append(codes[:, j])
            mechanism_names.append(f"{module}: mechanism {j}")

    components = np.stack(components_blocks, axis=0) if components_blocks else np.zeros((0, T_total))
    loadings = np.stack(loadings_blocks, axis=1) if loadings_blocks else np.zeros((C.shape[0], 0))
    return mean_vec, components, loadings, mechanism_names


def fit_nmf_signsplit_per_module_exclusive(
    C: np.ndarray, col_labels: List[Tuple[str, str]], col_slices: List[Tuple[str, int, int]],
    K_per_module: int = 6, alpha_W: float = 0.001, alpha_H: float = 0.0, alpha_exclusive: float = 0.0,
    n_steps: int = 3000, lr: float = 0.05, random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """`fit_nmf_signsplit_per_module`, custom-trained (sklearn's NMF has no
    hook for a custom penalty term) to add an exclusive-overlap penalty on
    top of the same sign-split NMF formulation: ``alpha_exclusive`` penalizes
    ``sum_{k<m} sum_term |component_k,term| * |component_m,term|`` on the
    *recombined signed* components (not the split halves -- the split
    representation can't see the cancellation this targets, since one
    mechanism using the positive half and another the negative half of the
    same term looks unrelated there). This is a soft bias against two
    mechanisms both loading heavily on the same term, not a hard guarantee
    (see conversation, 2026-07-21, for why no known variant that keeps
    signed, combinable mechanisms gives a hard guarantee).

    Trained via proximal gradient (Adam on the smooth reconstruction +
    exclusive-penalty terms; L1 sparsity enforced by an explicit
    soft-threshold step each iteration, not by adding |x| to the loss --
    plain gradient descent on an L1 term does not reliably reach exact
    zeros). At ``alpha_exclusive=0`` this should closely match
    `fit_nmf_signsplit_per_module` at comparable sparsity, modulo this being
    a hand-rolled optimizer against sklearn's mature, well-converged one --
    check that before trusting a comparison with the penalty turned on.

    Returns mean_vec (T_total,, always zeros -- no population offset here),
    components (K_total, T_total, module-block-sparse by construction,
    signed), loadings (N, K_total, >= 0), mechanism_names.
    """
    torch.manual_seed(random_state)
    T_total = C.shape[1]
    N = C.shape[0]

    components_blocks = []
    loadings_blocks = []
    mechanism_names = []
    for module, start, end in col_slices:
        block = torch.tensor(C[:, start:end], dtype=torch.float64)  # (N, T_m)
        T_m = block.shape[1]
        k = min(K_per_module, T_m)
        split = torch.cat([block.clamp(min=0), (-block).clamp(min=0)], dim=1)  # (N, 2*T_m), >= 0

        G_raw = (torch.rand(N, k, dtype=torch.float64) * 0.1).requires_grad_(True)
        H_raw = (torch.rand(k, 2 * T_m, dtype=torch.float64) * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([G_raw, H_raw], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            G = torch.clamp(G_raw, min=0)
            H = torch.clamp(H_raw, min=0)
            recon = G @ H
            loss = ((split - recon) ** 2).mean()
            if alpha_exclusive > 0 and k > 1:
                comp = H[:, :T_m] - H[:, T_m:]  # (k, T_m), signed, recombined
                absC = comp.abs()
                gram = absC @ absC.T  # (k, k)
                loss = loss + alpha_exclusive * (gram.sum() - gram.diagonal().sum()) / (k * (k - 1))
            loss.backward()
            opt.step()
            with torch.no_grad():
                G_raw.copy_(torch.clamp(G_raw - lr * alpha_W, min=0))
                H_raw.copy_(torch.clamp(H_raw - lr * alpha_H, min=0))
                # G @ H is invariant to (G -> G/c, H -> H*c): without removing
                # this gauge freedom every step, the optimizer can evade the L1
                # penalty on G by inflating H instead of ever shrinking G,
                # silently defeating alpha_W regardless of its value.
                norms = H_raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
                H_raw.copy_(H_raw / norms)
                G_raw.copy_(G_raw * norms.T)

        with torch.no_grad():
            G = torch.clamp(G_raw, min=0).numpy()
            H = torch.clamp(H_raw, min=0).numpy()

        codes, H_norm = _normalize_atoms(G, H)
        dictionary = H_norm[:, :T_m] - H_norm[:, T_m:]  # (k, T_m), signed

        for j in range(k):
            full_component = np.zeros(T_total)
            full_component[start:end] = dictionary[j]
            components_blocks.append(full_component)
            loadings_blocks.append(codes[:, j])
            mechanism_names.append(f"{module}: mechanism {j}")

    components = np.stack(components_blocks, axis=0) if components_blocks else np.zeros((0, T_total))
    loadings = np.stack(loadings_blocks, axis=1) if loadings_blocks else np.zeros((N, 0))
    mean_vec = np.zeros(T_total)
    return mean_vec, components, loadings, mechanism_names


# ---------------------------------------------------------------------------
# Semi-NMF (Ding, Li & Jordan, 2010): X ~= G @ F.T, G >= 0, F unconstrained
# ---------------------------------------------------------------------------
#
# Fits signed coefficients directly -- no sign-split/recombine hack, so no
# artificial doubling of dimensionality. Still does NOT guarantee mechanisms
# never disagree in sign on the same term (that guarantee needs
# non-negativity on the *same* factor as any orthogonality constraint, which
# would have to be G -- forcing one-mechanism-per-participant hard
# clustering -- not F; see conversation, 2026-07-21). Optionally adds an
# exclusive-overlap penalty (a soft bias against two mechanisms both loading
# heavily on the same term, not a hard constraint) on top of the standard
# L1 sparsity penalties.

def fit_semi_nmf_per_module(
    C: np.ndarray, col_labels: List[Tuple[str, str]], col_slices: List[Tuple[str, int, int]],
    K_per_module: int = 6, alpha_W: float = 0.001, alpha_H: float = 0.0, alpha_exclusive: float = 0.0,
    n_steps: int = 3000, lr: float = 0.05, random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Semi-NMF fit independently within each module's own term block.

    ``alpha_W`` / ``alpha_H`` are L1 penalties on the loadings / dictionary,
    same role as in `fit_nmf_signsplit_per_module`. ``alpha_exclusive``
    penalizes ``sum_{k<m} |F_k,j| * |F_m,j|`` (summed over terms j) --
    an "exclusive lasso" style soft bias against two mechanisms both loading
    heavily on the same term, encouraging (not guaranteeing) disjoint
    support. Fit via projected gradient descent (Adam + clamp G >= 0 each
    step) rather than the classic multiplicative-update rule, since the
    multiplicative updates aren't a natural fit for L1/exclusive penalties.

    Returns mean_vec (T_total,, always zeros -- semi-NMF has no separate
    population offset here), components (K_total, T_total, module-block-
    sparse by construction, signed), loadings (N, K_total, >= 0),
    mechanism_names (e.g. "value_reward_chosen: mechanism 3").
    """
    torch.manual_seed(random_state)
    T_total = C.shape[1]
    N = C.shape[0]

    components_blocks = []
    loadings_blocks = []
    mechanism_names = []
    for module, start, end in col_slices:
        block = torch.tensor(C[:, start:end], dtype=torch.float64)  # (N, T_m)
        T_m = block.shape[1]
        k = min(K_per_module, T_m)

        G_raw = torch.randn(N, k, dtype=torch.float64) * 0.1
        F = torch.randn(T_m, k, dtype=torch.float64) * 0.1
        G_raw.clamp_(min=0)
        G_raw.requires_grad_(True)
        F.requires_grad_(True)
        opt = torch.optim.Adam([G_raw, F], lr=lr)

        # Proximal gradient: Adam handles the smooth reconstruction (+ exclusive
        # penalty) term only; L1 sparsity is enforced via an explicit
        # soft-threshold step after each Adam update, not by adding |x| to the
        # loss and hoping gradient descent finds exact zeros (it generally
        # won't -- that's the standard ISTA/proximal-gradient construction).
        # For G (constrained >= 0), soft-threshold-then-clamp is the proximal
        # operator of (alpha*|x| + indicator{x>=0}) jointly: relu(x - lr*alpha).
        for _ in range(n_steps):
            opt.zero_grad()
            G = torch.clamp(G_raw, min=0)
            recon = G @ F.T
            loss = ((block - recon) ** 2).mean()
            if alpha_exclusive > 0 and k > 1:
                absF = F.abs()  # (T_m, k)
                gram = absF.T @ absF  # (k, k), off-diagonal = sum_j |F_kj||F_mj|
                loss = loss + alpha_exclusive * (gram.sum() - gram.diagonal().sum()) / (k * (k - 1))
            loss.backward()
            opt.step()
            with torch.no_grad():
                G_raw.copy_(torch.clamp(G_raw - lr * alpha_W, min=0))
                F.copy_(torch.sign(F) * torch.clamp(F.abs() - lr * alpha_H, min=0))

        with torch.no_grad():
            G = torch.clamp(G_raw, min=0).numpy()
            F_np = F.numpy()

        codes, dictionary_T = _normalize_atoms(G, F_np.T)  # dictionary_T: (k, T_m)
        for j in range(k):
            full_component = np.zeros(T_total)
            full_component[start:end] = dictionary_T[j]
            components_blocks.append(full_component)
            loadings_blocks.append(codes[:, j])
            mechanism_names.append(f"{module}: mechanism {j}")

    components = np.stack(components_blocks, axis=0) if components_blocks else np.zeros((0, T_total))
    loadings = np.stack(loadings_blocks, axis=1) if loadings_blocks else np.zeros((N, 0))
    mean_vec = np.zeros(T_total)
    return mean_vec, components, loadings, mechanism_names


# ---------------------------------------------------------------------------
# Term-family (block-diagonal) fitting
# ---------------------------------------------------------------------------
#
# SVD, sparse dictionary learning, and sign-split NMF all only penalize the
# *loadings* for sparsity -- nothing stops a single mechanism from spreading
# across every module and ~a third of all terms, which is exactly as hard to
# name as the raw equations. This alternative instead constrains *where a
# mechanism is allowed to have nonzero support* by construction: classify
# every term into a semantic family from its syntactic form (does it involve
# the module's own state? one control signal at one lag? a product of two
# lags of the same signal?), fit a small basis *within* each family block
# only, and never let a mechanism mix across families or modules. Every
# mechanism is then trivially nameable ("reward history", "reward
# consistency", ...) because the family membership *is* the name.

import re


def classify_term(term: str, module: str) -> Tuple[str, str]:
    """Classify a SINDy candidate term into a (family_key, family_label) pair
    from its syntactic form, relative to its owning ``module``.

    Families: bias ('1'), persistence (module's own linear state term),
    self-nonlinearity (module^2), sensitivity (a control signal at lag 0),
    history (a control signal at lag >= 1), value-gated update (module
    times a control signal), consistency (product of two lags of the same
    control signal, including its square). Anything unrecognized falls into
    "other" rather than being silently mis-grouped.
    """
    if term == "1":
        return "bias", "baseline"
    if term == module:
        return "persistence", "persistence"
    if term == f"{module}^2":
        return "self_nl", "self-nonlinearity"
    m = re.match(r"^(\w+)\[t\]$", term)
    if m:
        signal = m.group(1)
        return f"sens_{signal}", f"{signal} sensitivity (immediate)"
    m = re.match(r"^(\w+)\[t-(\d+)\]$", term)
    if m:
        signal = m.group(1)
        return f"hist_{signal}", f"{signal} history"
    m = re.match(rf"^{re.escape(module)}\*(\w+)\[t(?:-\d+)?\]$", term)
    if m:
        signal = m.group(1)
        return f"gated_{signal}", f"value-gated {signal} update"
    m = re.match(r"^(\w+)\[t(?:-\d+)?\]\^2$", term)
    if m:
        signal = m.group(1)
        return f"consist_{signal}", f"{signal} consistency"
    m = re.match(r"^(\w+)\[t(?:-\d+)?\]\*(\w+)\[t(?:-\d+)?\]$", term)
    if m and m.group(1) == m.group(2):
        signal = m.group(1)
        return f"consist_{signal}", f"{signal} consistency"
    return "other", "other"


def fit_term_family_basis(
    C: np.ndarray, col_labels: List[Tuple[str, str]], col_slices: List[Tuple[str, int, int]],
    rank_per_family: int = 1, alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Block-diagonal PCA: one small SVD per (module, term-family) block.

    Every term is assigned to exactly one family via `classify_term`, so
    families partition the term space -- no mechanism ever mixes across
    families or modules, unlike SVD/sparse-dict/NMF which only constrain the
    loadings. ``rank_per_family`` components are fit per family (1 by
    default: "how much of this concept does this participant show").
    Families smaller than ``rank_per_family`` use their full size instead
    (e.g. a single-term family like "persistence" just becomes that raw
    coefficient, unchanged).

    ``alpha`` soft-thresholds each family's loadings (``sign(x) *
    max(|x| - alpha, 0)``), giving sparse, presence/absence usage on top of
    the block-diagonal localization -- a participant "lacks" a mechanism
    once its projection falls below alpha. Because each family's basis
    directions are orthonormal (right singular vectors from that family's
    own SVD), this soft-threshold is the *exact* solution to the
    L1-penalized (Lasso) fit within that block, not an approximation -- for
    an orthonormal design matrix, Lasso decouples into independent
    per-coordinate soft-thresholding. No iterative optimization or
    random-seed sensitivity, unlike `fit_sparse_dictionary`/`fit_nmf_signsplit`.

    Returns mean_vec (T_total,), components (K, T_total, block-sparse by
    construction), loadings (N, K), and mechanism_names (len K, e.g.
    "value_reward_chosen: reward history").
    """
    mean_vec = C.mean(axis=0)
    residual = C - mean_vec
    T_total = C.shape[1]

    # group column indices by (module, family_key)
    families: dict = {}
    family_order: List[Tuple[str, str]] = []
    for col_idx, (module, term) in enumerate(col_labels):
        family_key, family_label = classify_term(term, module)
        bucket = (module, family_key)
        if bucket not in families:
            families[bucket] = {"label": family_label, "indices": []}
            family_order.append(bucket)
        families[bucket]["indices"].append(col_idx)

    components_blocks = []
    loadings_blocks = []
    mechanism_names = []
    for module, family_key in family_order:
        info = families[(module, family_key)]
        idx = info["indices"]
        block = residual[:, idx]  # (N, n_terms_in_family)
        r = min(rank_per_family, len(idx))
        if np.abs(block).max() < 1e-10:
            continue  # family entirely inactive across the population
        U, S, Vt = np.linalg.svd(block, full_matrices=False)
        block_components = Vt[:r, :]         # (r, n_terms_in_family)
        block_loadings = U[:, :r] * S[:r]    # (N, r)
        if alpha > 0:
            block_loadings = np.sign(block_loadings) * np.maximum(np.abs(block_loadings) - alpha, 0.0)
        for k in range(r):
            full_component = np.zeros(T_total)
            full_component[idx] = block_components[k]
            components_blocks.append(full_component)
            loadings_blocks.append(block_loadings[:, k])
            suffix = f" [{k+1}/{r}]" if r > 1 else ""
            mechanism_names.append(f"{module}: {info['label']}{suffix}")

    components = np.stack(components_blocks, axis=0) if components_blocks else np.zeros((0, T_total))
    loadings = np.stack(loadings_blocks, axis=1) if loadings_blocks else np.zeros((C.shape[0], 0))
    return mean_vec, components, loadings, mechanism_names


# ---------------------------------------------------------------------------
# Symbolic formatting
# ---------------------------------------------------------------------------

def format_equation_string(
    coef_vector: np.ndarray, terms: List[str], module: str,
    threshold: float = 1e-3, add_identity: bool = False,
) -> str:
    """Format a coefficient vector as ``module[t+1] = c1 term1 + c2 term2 ...``.

    Matches the formatting convention of ``BaseModel.get_spice_model_string``.
    With ``add_identity=True``, the module's own linear state term has +1
    folded in (showing the effective total multiplier on the current state,
    as ``print_spice_model`` does). With ``add_identity=False`` (default),
    coefficients are shown raw, matching the literal residual update
    ``h_next = h_current + library @ coeffs`` used by ``forward_sindy``.
    """
    coef_vector = np.asarray(coef_vector, dtype=float).copy()
    equation_str = f"{module}[t+1] = "
    for i, term in enumerate(terms):
        c = coef_vector[i]
        if add_identity and term == module:
            c = c + 1
        if abs(c) > threshold:
            if not equation_str.endswith("= "):
                equation_str += "+ "
            equation_str += f"{round(float(c), 3)} {term}"
            equation_str += "[t] " if term == module else " "
    if equation_str.endswith("= "):
        equation_str += "0"
    return equation_str.strip()


def format_mechanism_terms(row: np.ndarray, terms: List[str], threshold_ratio: float = 0.15) -> Optional[str]:
    """Sparse symbolic expression for one component's contribution to one module.

    Keeps only terms whose loading exceeds ``threshold_ratio`` of this row's
    max absolute loading, so each mechanism reads as a short expression
    instead of a dense 15-20 term list.
    """
    max_abs = np.abs(row).max()
    if max_abs < 1e-8:
        return None
    parts = []
    for term, v in zip(terms, row):
        if abs(v) > threshold_ratio * max_abs:
            parts.append(f"{round(float(v), 3)} {term}")
    if not parts:
        return None
    return " + ".join(parts).replace("+ -", "- ")


# ---------------------------------------------------------------------------
# CompressedSpiceModel
# ---------------------------------------------------------------------------

class CompressedSpiceModel:
    """A fitted (components, loadings) reparameterization of a SPICE model's
    coefficients, from either `method="svd"` or `method="sparse"`.

    Every participant's transition equations become a shared population
    baseline plus a weighted sum of K shared "mechanisms" -- K numbers per
    participant instead of the full per-term coefficient set. Mirrors
    ``SpiceEstimator.print_spice_model`` but in the compressed basis.
    """

    def __init__(
        self,
        mean_vec: np.ndarray, components: np.ndarray, loadings: np.ndarray,
        col_labels: List[Tuple[str, str]], col_slices: List[Tuple[str, int, int]],
        P: int, X: int, mechanism_names: Optional[List[str]] = None,
    ):
        self.mean_vec = mean_vec
        self.components = components  # (K, T_total)
        self.loadings = loadings      # (P*X, K)
        self.col_labels = col_labels
        self.col_slices = col_slices
        self.P = P
        self.X = X
        self.K = components.shape[0]
        self.mechanism_names = mechanism_names if mechanism_names is not None else [f"M{k}" for k in range(self.K)]

    def _module_terms(self, module: str) -> List[str]:
        start, end = next((s, e) for m, s, e in self.col_slices if m == module)
        return [t for _, t in self.col_labels[start:end]]

    def reconstructed_coefficients(self) -> np.ndarray:
        """Full (P*X, T_total) reconstruction at this K, in the original term basis."""
        if self.K <= 0:
            return np.tile(self.mean_vec, (self.P * self.X, 1))
        return self.mean_vec + self.loadings @ self.components

    def sparsity(self, threshold: float = 1e-6) -> float:
        """Mean fraction of near-zero loadings per participant (0 = dense, 1 = all-zero).

        This is usage sparsity -- how many mechanisms a participant activates.
        It says nothing about how concentrated each mechanism itself is; see
        `n_effective_terms` / `n_terms_above_threshold` / `n_modules_touched`
        for that -- a method can have sparse usage (few active mechanisms per
        participant) while every mechanism is still a dense blob spanning
        every module, which is just as hard to name as the raw equations.
        """
        if self.K == 0:
            return float("nan")
        return float((np.abs(self.loadings) <= threshold).mean())

    def n_effective_terms(self) -> np.ndarray:
        """Per-mechanism effective number of contributing terms (participation
        ratio (sum|c|)^2 / sum(c^2), over all terms globally). 1 = concentrated
        on a single term, T_total = spread uniformly across every term.
        Threshold-free, unlike `n_terms_above_threshold`.
        """
        l1 = np.abs(self.components).sum(axis=1)
        l2sq = (self.components ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(l2sq > 0, l1 ** 2 / l2sq, 0.0)

    def n_terms_above_threshold(self, threshold_ratio: float = 0.15) -> np.ndarray:
        """Per-mechanism count of terms exceeding threshold_ratio * that
        mechanism's global max |loading| (across all modules, unlike the
        per-module-relative threshold `mechanisms_string` uses for display).
        """
        max_abs = np.abs(self.components).max(axis=1, keepdims=True)
        max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
        return (np.abs(self.components) > threshold_ratio * max_abs).sum(axis=1)

    def n_modules_touched(self, threshold_ratio: float = 0.15) -> np.ndarray:
        """Per-mechanism count of modules containing at least one term above
        the global threshold (see `n_terms_above_threshold`). Low values mean
        a mechanism is localized to one or two modules, not smeared across
        the whole equation set.
        """
        max_abs = np.abs(self.components).max(axis=1, keepdims=True)
        max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
        above = np.abs(self.components) > threshold_ratio * max_abs  # (K, T_total)
        counts = np.zeros(self.K, dtype=int)
        for module, start, end in self.col_slices:
            counts += above[:, start:end].any(axis=1).astype(int)
        return counts

    # -- symbolic strings -------------------------------------------------

    def population_string(self) -> str:
        lines = []
        for module, start, end in self.col_slices:
            terms = self._module_terms(module)
            lines.append(format_equation_string(self.mean_vec[start:end], terms, module, add_identity=True))
        return "\n".join(lines)

    def mechanisms_string(self, threshold_ratio: float = 0.15) -> str:
        lines = []
        for k in range(self.K):
            lines.append(f"Mechanism {self.mechanism_names[k]}:")
            any_module = False
            for module, start, end in self.col_slices:
                terms = self._module_terms(module)
                expr = format_mechanism_terms(self.components[k, start:end], terms, threshold_ratio)
                if expr is not None:
                    lines.append(f"  {module:<28s} += {expr}")
                    any_module = True
            if not any_module:
                lines.append("  (negligible contribution to all modules)")
        return "\n".join(lines)

    def participant_string(self, participant_id: int, experiment_id: int = 0, threshold_ratio: float = 0.1) -> str:
        """Compact per-participant equation: the module's own state term,
        plus (when the population baseline carries any information --
        i.e. wasn't fit with ``center=False``) `POP_module` (printed once
        via `population_string()`), plus this participant's active
        mechanisms.

        Note: when a nonzero population baseline is present, `POP_module`
        already includes the module's own-state term with the identity
        ("+1") folded into its coefficient -- the same effective-multiplier
        convention `population_string()` prints (and that
        `SpiceEstimator.print_spice_model()` uses everywhere else in this
        codebase) -- so no separate `module[t]` term is added alongside it,
        to avoid double-counting that persistence. When there's no
        population baseline (``center=False``, e.g. the default
        `fit_nmf_signsplit_per_module` used for the current SPICE-EQ model),
        `POP_module` would just be the identity itself, so this prints the
        equivalent, more legible `module[t]` directly instead. Mechanisms
        are pure directions with no such folding (see `mechanisms_string()`).
        """
        row_idx = participant_id * self.X + experiment_id
        loadings_row = self.loadings[row_idx]
        active_k = [k for k in range(self.K) if abs(loadings_row[k]) > 1e-6]
        lines = [
            f"Participant {participant_id} loadings ({len(active_k)}/{self.K} active): "
            + ", ".join(f"{self.mechanism_names[k]}={round(float(loadings_row[k]), 3)}" for k in active_k)
        ]
        for module, start, end in self.col_slices:
            module_max = np.abs(self.components[:, start:end]).max() if self.K > 0 else 0.0
            relevant_k = [
                k for k in active_k
                if module_max > 1e-8 and np.abs(self.components[k, start:end]).max() > threshold_ratio * module_max
            ]
            terms_str = " + ".join(self.mechanism_names[k] for k in relevant_k) if relevant_k else "0"
            has_population_baseline = np.abs(self.mean_vec[start:end]).max() > 1e-10
            baseline_term = f"POP_{module}" if has_population_baseline else f"{module}[t]"
            lines.append(f"{module}[t+1] = {baseline_term} + {terms_str}")
        return "\n".join(lines)

    def print_population(self) -> None:
        print(self.population_string())

    def print_mechanisms(self, threshold_ratio: float = 0.15) -> None:
        print(self.mechanisms_string(threshold_ratio=threshold_ratio))

    def print_participant(self, participant_id: int, experiment_id: int = 0, threshold_ratio: float = 0.1) -> None:
        print(self.participant_string(participant_id, experiment_id, threshold_ratio))

    # -- applying the compression to a live model --------------------------

    def _write(self, spice_model) -> None:
        """Overwrite ``spice_model``'s SINDy coefficients in place with this
        compression's reconstruction. Broadcasts the same (participant,
        experiment) coefficient to every ensemble member (so inference uses
        one deduplicated equation set instead of E diverse ones -- see
        module docstring caveat in
        `weinhardt2026/analysis/analysis_coefficient_compression.py`) and
        marks every term as present, since zeros already encode "inactive".
        """
        C_hat = self.reconstructed_coefficients()
        model = spice_model.model
        E = model.ensemble_size
        for module, start, end in self.col_slices:
            orig_coef = model.sindy_coefficients[module].data
            vals = torch.tensor(C_hat[:, start:end], dtype=orig_coef.dtype, device=orig_coef.device)
            vals = vals.reshape(self.P, self.X, -1)
            model.sindy_coefficients[module].data = vals.unsqueeze(0).expand(E, -1, -1, -1).clone()
            model.sindy_coefficients_presence[module].data = torch.ones_like(model.sindy_coefficients_presence[module].data)

    @contextmanager
    def apply(self, spice_model):
        """Temporarily overwrite ``spice_model``'s SINDy coefficients with this
        compression's reconstruction, for running inference/evaluation.
        Restores the original coefficients on exit -- use `commit` instead
        to make the change permanent (e.g. before `estimator.save_spice(...)`).
        """
        modules = [m for m, _, _ in self.col_slices]
        originals = {
            m: (spice_model.model.sindy_coefficients[m].data.clone(),
                spice_model.model.sindy_coefficients_presence[m].data.clone())
            for m in modules
        }
        try:
            self._write(spice_model)
            yield spice_model
        finally:
            for module in modules:
                spice_model.model.sindy_coefficients[module].data = originals[module][0]
                spice_model.model.sindy_coefficients_presence[module].data = originals[module][1]

    def commit(self, spice_model) -> None:
        """Permanently overwrite ``spice_model``'s SINDy coefficients with this
        compression's reconstruction (no restore) -- use this to bake the
        compressed model in as the model you actually ship, e.g. before
        `estimator.save_spice(...)`.

        Important caveat for reporting model complexity afterward: this
        writes the reconstruction back in the *original* per-term basis, and
        the population-average equation is dense across nearly every term,
        so `spice_model.count_sindy_coefficients()` / `print_spice_model()`
        on the committed model will show ~all raw terms as "present" and
        report an inflated parameter count -- the true degrees of freedom is
        `self.K` shared mechanisms times the per-participant active count
        (see `.sparsity()` / the `n_active_mean` column from the
        hyperparameter search), not a term count in this view. Report *that*
        number for BIC/AIC purposes, not `count_sindy_coefficients()`'s.
        """
        self._write(spice_model)


def compress_sindy_equations(spice_model, K: int = None, method: str = "nmf_per_module", **method_kwargs) -> CompressedSpiceModel:
    """Fit a reparameterization of ``spice_model``'s SINDy coefficients.

    Parameters
    ----------
    K : number of shared mechanisms. Ignored (pass via ``method_kwargs``
        instead) for ``method="family"`` and ``method="nmf_per_module"``/
        ``"sparse_per_module"``, which take a *per-module* ``K_per_module``
        instead of one global K.
    method : after comparing all of these on real studies, **"nmf_per_module"
        (the default)** -- sign-split NMF fit independently within each
        module's own term block -- won out on every axis that mattered:
        predictive cost, genuine usage sparsity (participants lack some
        mechanisms outright, not just small-everywhere), compact mechanisms
        (a handful of terms, not ~20), guaranteed one-module localization,
        and no hand-classification. The others remain available for
        comparison: "svd" (dense, optimal reconstruction fidelity but every
        participant loads on every mechanism, and mechanisms span every
        module), "sparse" (dictionary learning, L1-penalized codes, joint
        across all modules -- sparse usage but still spans every module),
        "sparse_per_module" (dictionary learning per module -- localized,
        but nothing constrains terms-per-mechanism the way NMF's alpha_H
        does), "nmf" (sign-split NMF, joint across all modules), and
        "family" (block-diagonal PCA within hand-classified term families --
        the most legible names, but the families are asserted, not learned,
        and it costs more predictive performance than nmf_per_module at
        matched sparsity).
    method_kwargs : forwarded to the fitting function for ``method``:
        `fit_nmf_signsplit_per_module` (``method="nmf_per_module"``, e.g.
        ``K_per_module``, ``alpha_W``, ``alpha_H``), `fit_sparse_dictionary`
        (``method="sparse"``, e.g. ``alpha``), `fit_sparse_dictionary_per_module`
        (``method="sparse_per_module"``, e.g. ``K_per_module``, ``alpha``),
        `fit_nmf_signsplit` (``method="nmf"``, e.g. ``alpha_W``, ``alpha_H``),
        or `fit_term_family_basis` (``method="family"``, e.g. ``rank_per_family``,
        ``alpha``).

    See `CompressedSpiceModel` for what you can do with the result, and
    `weinhardt2026.analysis.analysis_coefficient_compression` for a proper
    train-selected/test-confirmed hyperparameter search instead of guessing
    ``method_kwargs`` by hand.
    """
    C, col_labels, col_slices, P, X = extract_joint_coefficient_matrix(spice_model)
    mechanism_names = None
    if method == "svd":
        mean_vec, U, S, Vt = fit_low_rank_basis(C)
        components, loadings = svd_components_and_loadings(U, S, Vt, K)
    elif method == "sparse":
        mean_vec, components, loadings = fit_sparse_dictionary(C, K, **method_kwargs)
    elif method == "sparse_per_module":
        mean_vec, components, loadings, mechanism_names = fit_sparse_dictionary_per_module(C, col_labels, col_slices, **method_kwargs)
    elif method == "nmf":
        mean_vec, components, loadings = fit_nmf_signsplit(C, K, **method_kwargs)
    elif method == "nmf_per_module":
        mean_vec, components, loadings, mechanism_names = fit_nmf_signsplit_per_module(C, col_labels, col_slices, **method_kwargs)
    elif method == "family":
        mean_vec, components, loadings, mechanism_names = fit_term_family_basis(C, col_labels, col_slices, **method_kwargs)
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'svd', 'sparse', 'sparse_per_module', 'nmf', 'nmf_per_module', or 'family'.")
    return CompressedSpiceModel(mean_vec, components, loadings, col_labels, col_slices, P, X, mechanism_names=mechanism_names)

"""Coefficient ties: model reduction by removing redundancy *among* SINDy
coefficients, instead of reparameterizing them into a new basis.

`sindy_compression` factorizes the per-participant coefficient matrix into K
shared "mechanisms" plus loadings. That buys a lower parameter count at the
price of three problems, all of them structural rather than tunable:

  * It reads a structural zero (term pruned away for this participant) and a
    small nonzero coefficient as nearby points in the same Euclidean space,
    so presence/absence differences get confounded with magnitude
    differences. On dezfouli2019 this is not a corner case: every one of the
    101 participants has a *distinct* joint support pattern (and all 10
    ensemble members agree on the mask for every participant), so a
    factorization of that matrix is substantially modelling the zero pattern.
  * The recovered atoms are identifiable only up to a rotation within the
    span they recover, so "which terms belong to which mechanism" -- and in
    particular a term appearing in several mechanisms -- is not fixed by the
    data unless it is shown to be stable across seeds/K/resamples.
  * The mechanisms are signed and additive, so two of them can cancel on a
    shared term.

This module takes the other route. The coefficients stay exactly the
coefficients the equations are written in; what gets removed is redundancy
*between* them. Concretely, a group of terms S is declared to lie on a shared
low-dimensional set:

    c_p[S] = offset + theta_p @ direction          (theta_p in R^rank)

with ``direction``/``offset`` shared across the population and ``theta_p``
the only per-participant freedom. rank=1 over two terms is the classic tie --
``c_i = lam * c_j`` -- which is how Rescorla-Wagner's single learning rate
shows up here: with the residual convention ``Q[t+1] = Q[t] + lib @ c``,
``Q[t+1] = (1-a) Q[t] + a R[t]`` means ``c_Q = -a``, ``c_R = +a``, i.e. the
tie ``c_R = -c_Q``. rank=1 over a lag family with a geometric direction
``(1, gamma, gamma^2, ...)`` is a decaying memory kernel: amplitude per
participant, decay rate shared. rank=0 is "this coefficient is not an
individual difference at all", collapsing P parameters into 1.

Because a tie is only ever *fit on* and *applied to* the subpopulation where
all of its terms are present, structural zeros never enter as numbers -- the
first problem above does not arise. Ties are constraints on named
coefficients rather than a new basis, so there is no rotational ambiguity and
a coefficient participating in two ties just means the constraints intersect.
And nothing is summed, so nothing can cancel.

Usage is two-phase, mirroring the SINDy pipeline itself:

  1. `screen_candidates` proposes ties from the fitted coefficients alone
     (cheap, no forward passes) and ranks them by how tightly the population
     actually obeys them.
  2. The caller accepts them greedily, each acceptance validated by refitting
     the surviving free parameters against behavioural NLL and scoring BIC --
     see `weinhardt2026/analysis/analysis_coefficient_ties.py`. BIC is the
     objective: a tie is worth having exactly when the degrees of freedom it
     removes outweigh the likelihood it costs.

`TieSet.build_coefficients` produces the full coefficient tensors from the
free parameters differentiably, so step 2 optimizes the per-participant
loadings *and* the shared constants (lam, gamma, ...) jointly against NLL.
"""

from __future__ import annotations

import itertools
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_coefficients_and_support(spice_model) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Per-module presence-masked coefficients and presence masks, as numpy.

    ``spice_model`` may be a `SpiceEstimator` or a `BaseModel`.

    Returns (coefficients, presence, terms), each keyed by module:
    coefficients (E, P, X, T) float, presence (E, P, X, T) bool, terms a list
    of T candidate-term names.
    """
    model = getattr(spice_model, 'model', spice_model)
    coefs = model.get_sindy_coefficients(aggregate=False)
    terms = model.get_candidate_terms()
    coefficients = {m: coefs[m].detach().cpu().numpy() for m in model.get_modules()}
    presence = {m: model.sindy_coefficients_presence[m].detach().cpu().numpy().astype(bool)
                for m in model.get_modules()}
    return coefficients, presence, {m: list(terms[m]) for m in model.get_modules()}


def participant_support(presence: np.ndarray) -> np.ndarray:
    """Collapse an (E, P, X, T) presence mask to a per-participant (P, X, T)
    support, requiring agreement across ensemble members.

    A term counts as present for a participant only if *every* member kept it.
    Members that disagree would otherwise make a tie apply to some members of
    the same participant and not others, which is not a statement about that
    participant's model. On dezfouli2019 the members agree everywhere, so this
    is a no-op there; on studies where they do not, it errs toward leaving the
    term free (unconstrained), which is the conservative direction.
    """
    return presence.all(axis=0)


# ---------------------------------------------------------------------------
# Tie groups
# ---------------------------------------------------------------------------

@dataclass
class TieGroup:
    """One group of terms constrained to a shared low-dimensional set.

    ``kind`` determines how ``direction`` (rank, |S|) is parameterized and how
    many shared population parameters that costs:

      * ``"free"``    -- direction learned freely, normalized to unit rows.
                         rank * |S| - rank shared parameters (the scale of
                         each direction row is absorbed into theta, so it is
                         not a free parameter).
      * ``"geometric"`` -- rank 1, direction = (1, g, g^2, ...) for a single
                         shared decay ``g``: 1 shared parameter. Only
                         meaningful when ``term_indices`` is ordered by lag.
      * ``"constant"`` -- rank 0: the terms take a shared value for everyone,
                         no per-participant freedom. |S| shared parameters.

    ``offset`` is nonzero only when the group was fit with affine ties
    enabled; a zero offset is the more interpretable "pure ratio" statement
    (c_i = lam * c_j exactly, no additive part) and is the default.

    ``participants`` is a boolean (P, X) mask: the group applies to a
    (participant, experiment) only where all of its terms are present. Where
    it does not apply, those terms keep their own free coefficients.
    """

    module: str
    term_indices: Tuple[int, ...]
    kind: str
    rank: int
    direction: np.ndarray            # (rank, |S|); empty for kind="constant"
    offset: np.ndarray               # (|S|,)
    participants: np.ndarray         # (P, X) bool
    gamma: Optional[float] = None    # kind="geometric" only
    term_names: Tuple[str, ...] = ()
    fit_error: float = float('nan')  # relative residual at screening time
    use_offset: bool = False

    @property
    def size(self) -> int:
        return len(self.term_indices)

    @property
    def n_applicable(self) -> int:
        return int(self.participants.sum())

    @property
    def n_shared(self) -> int:
        """Population-level parameters this group introduces."""
        if self.kind == 'constant':
            shared = self.size
        elif self.kind == 'geometric':
            shared = 1
        else:
            shared = self.rank * self.size - self.rank  # unit-norm rows
        return shared + (self.size if self.use_offset else 0)

    @property
    def dof_saved(self) -> int:
        """Net free parameters removed: per-participant savings minus the
        shared constants the group introduces."""
        return self.n_applicable * (self.size - self.rank) - self.n_shared

    def describe(self) -> str:
        names = list(self.term_names) or [str(i) for i in self.term_indices]
        head = f"{self.module}: {self.kind} rank-{self.rank} over [{', '.join(names)}]"
        detail = ''
        if self.kind == 'geometric':
            detail = f"  gamma={self.gamma:.3f}"
        elif self.kind == 'constant':
            detail = '  values=[' + ', '.join(f"{v:+.3f}" for v in self.offset) + ']'
        elif self.rank == 1:
            v = self.direction[0]
            ref = v[np.argmax(np.abs(v))]
            ratios = v / ref
            detail = '  ratios=[' + ', '.join(f"{r:+.3f}" for r in ratios) + ']'
        return (f"{head}{detail}\n    applies to {self.n_applicable} participants, "
                f"rel.err {self.fit_error:.3f}, saves {self.dof_saved} parameters")


# ---------------------------------------------------------------------------
# Fitting a group's shared set from coefficients
# ---------------------------------------------------------------------------

def _rows_for(participants: np.ndarray, coefficients: np.ndarray, term_indices: Sequence[int]) -> np.ndarray:
    """(E * n_applicable, |S|) block of coefficients the group applies to."""
    sub = coefficients[:, :, :, list(term_indices)]           # (E, P, X, |S|)
    return sub[:, participants]                                # (E, n_applicable, |S|)


def _relative_error(block: np.ndarray, recon: np.ndarray) -> float:
    """Worst per-term relative residual over a covered block.

    Deliberately *not* the Frobenius error of the whole block. That version
    is normalized by the block's total magnitude, which a large coefficient
    dominates -- so pairing a big term with a near-zero one scores as a
    near-perfect rank-1 tie no matter what the small term does, purely because
    its contribution to the norm is negligible. Screening on it fills the
    candidate list with vacuous ties (measured on dezfouli2019: every one of
    the top eight candidates paired a substantial coefficient with a
    ~0.1-magnitude one). Normalizing each term by its own norm and taking the
    worst forces every member of the group to be genuinely predicted by the
    shared profile; a term that is near-zero for everybody fails this and
    should be pruned, not tied.
    """
    numerator = np.linalg.norm(block - recon, axis=0)
    denominator = np.linalg.norm(block, axis=0)
    ratios = numerator / np.where(denominator > 1e-12, denominator, np.inf)
    return float(ratios.max()) if ratios.size else float('nan')


def fit_group(
    coefficients: np.ndarray, term_indices: Sequence[int], participants: np.ndarray,
    kind: str = 'free', rank: int = 1, use_offset: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], float]:
    """Fit a group's shared (direction, offset) to the coefficients it covers.

    Returns (direction, offset, gamma, relative_error); see `_relative_error`
    for what the error measures. Residuals are taken against the raw
    coefficients rather than population-centered ones (unless ``use_offset``),
    so "the tie explains this coefficient" means what it says.
    """
    block = _rows_for(participants, coefficients, term_indices).reshape(-1, len(term_indices))

    if kind == 'constant':
        offset = block.mean(axis=0)
        return (np.zeros((0, len(term_indices))), offset, None,
                _relative_error(block, np.broadcast_to(offset, block.shape)))

    offset = block.mean(axis=0) if use_offset else np.zeros(len(term_indices))
    centered = block - offset

    if kind == 'geometric':
        gamma, _ = _fit_geometric(centered)
        direction = gamma ** np.arange(len(term_indices))[None, :]
        norm = np.linalg.norm(direction)
        direction = direction / (norm if norm > 1e-12 else 1.0)
        recon = np.outer(centered @ direction[0], direction[0])
        return direction, offset, float(gamma), _relative_error(block, recon + offset)

    # free direction: truncated SVD of the (uncentered unless use_offset) block
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    r = min(rank, Vt.shape[0])
    direction = Vt[:r]
    recon = (centered @ direction.T) @ direction
    return direction, offset, None, _relative_error(block, recon + offset)


def _fit_geometric(block: np.ndarray, n_grid: int = 199) -> Tuple[float, np.ndarray]:
    """Best shared decay ``gamma`` for a rank-1 geometric profile.

    Grid search over gamma in (-1, 1) with the amplitude solved in closed form
    at each candidate -- the objective is smooth but not convex in gamma, and
    the grid is cheap (|S| is a handful of terms). Selection minimizes the
    same worst-per-term relative residual the screening reports rather than
    plain least squares: a decaying profile's later lags are small, and a
    least-squares gamma buys accuracy on the leading lag by abandoning the
    tail it is supposed to describe.
    """
    n_terms = block.shape[1]
    best = (np.inf, 0.0, None)
    for gamma in np.linspace(-0.99, 0.99, n_grid):
        v = gamma ** np.arange(n_terms)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        v = v / norm
        amplitudes = block @ v
        resid = _relative_error(block, np.outer(amplitudes, v))
        if resid < best[0]:
            best = (resid, gamma, amplitudes)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Candidate screening
# ---------------------------------------------------------------------------

_LAG_RE = re.compile(r"^(\w+)\[t(?:-(\d+))?\]$")


def _lag_families(terms: Sequence[str]) -> Dict[str, List[int]]:
    """Group plain (non-product) control-signal terms by signal, ordered by lag.

    ``reward[t], reward[t-1], reward[t-2]`` -> {"reward": [i0, i1, i2]}. Only
    these families are candidates for the geometric template, which assumes
    consecutive lags of one signal.
    """
    families: Dict[str, List[Tuple[int, int]]] = {}
    for idx, term in enumerate(terms):
        m = _LAG_RE.match(term)
        if not m:
            continue
        signal, lag = m.group(1), int(m.group(2) or 0)
        families.setdefault(signal, []).append((lag, idx))
    return {sig: [i for _, i in sorted(v)] for sig, v in families.items() if len(v) >= 3}


def screen_candidates(
    coefficients: Dict[str, np.ndarray], presence: Dict[str, np.ndarray], terms: Dict[str, List[str]],
    min_coverage: float = 0.25, max_error: float = 0.6, use_offset: bool = False,
    include_geometric: bool = True, include_constant: bool = True,
) -> List[TieGroup]:
    """Propose ties from fitted coefficients alone, ranked tightest-first.

    Candidates: every within-module pair of terms (rank-1, i.e. ``c_i = lam *
    c_j``), every lag family under the geometric template and under a free
    rank-1 direction, and -- with ``include_constant`` -- every single term as
    a "not an individual difference" rank-0 group.

    ``min_coverage`` is the minimum fraction of (participant, experiment)
    pairs that must have all the group's terms present; a tie that only
    applies to a handful of people saves little and generalizes poorly.
    ``max_error`` caps the relative residual, so obviously-violated ties never
    reach the (expensive) refit stage.

    Screening is a *ranking* heuristic, not the decision: the caller accepts
    candidates greedily under BIC after refitting. Nothing here looks at the
    likelihood, so a tie along a direction the data barely constrains looks
    exactly like a tie the data insists on -- which is fine for proposing, and
    is why the acceptance test is BIC and not this number.
    """
    candidates: List[TieGroup] = []
    for module, C in coefficients.items():
        support = participant_support(presence[module])       # (P, X, T)
        n_rows = support.shape[0] * support.shape[1]
        term_names = terms[module]

        def _propose(indices: Sequence[int], kind: str, rank: int) -> None:
            indices = tuple(indices)
            applies = support[:, :, list(indices)].all(axis=-1)   # (P, X)
            if applies.sum() < max(2, min_coverage * n_rows):
                return
            block = _rows_for(applies, C, indices)
            if np.abs(block).max() < 1e-8:
                return
            direction, offset, gamma, err = fit_group(C, indices, applies, kind=kind, rank=rank,
                                                      use_offset=use_offset)
            group = TieGroup(module=module, term_indices=indices, kind=kind, rank=rank,
                             direction=direction, offset=offset, participants=applies,
                             gamma=gamma, term_names=tuple(term_names[i] for i in indices),
                             fit_error=err, use_offset=use_offset)
            if err <= max_error and group.dof_saved > 0:
                candidates.append(group)

        n_terms = len(term_names)
        for i, j in itertools.combinations(range(n_terms), 2):
            _propose((i, j), 'free', 1)
        for indices in _lag_families(term_names).values():
            _propose(indices, 'free', 1)
            if include_geometric:
                _propose(indices, 'geometric', 1)
        if include_constant:
            for i in range(n_terms):
                _propose((i,), 'constant', 0)

    candidates.sort(key=lambda g: g.fit_error)
    return candidates


def merge_groups(a: TieGroup, b: TieGroup, coefficients: np.ndarray) -> Optional[TieGroup]:
    """Fuse two overlapping same-module groups into one over the union of
    their terms, refitting the shared set.

    Overlapping ties cannot both be imposed independently -- ``c_i = lam c_j``
    and ``c_j = mu c_k`` are one statement about {i, j, k}, not two -- so when
    a candidate touches a term an accepted group already owns, this is what
    actually gets proposed. Applies to the intersection of the two
    participant sets (everyone with the full union present). Returns None if
    the merge would cover too few participants to be worth anything.
    """
    if a.module != b.module:
        return None
    indices = tuple(sorted(set(a.term_indices) | set(b.term_indices)))
    participants = a.participants & b.participants
    if participants.sum() < 2:
        return None
    rank = max(1, max(a.rank, b.rank))
    kind = 'free'  # a merged group's terms are no longer a clean lag family
    direction, offset, gamma, err = fit_group(coefficients, indices, participants,
                                              kind=kind, rank=rank, use_offset=a.use_offset or b.use_offset)
    # Names must follow `indices` (sorted), not the order the two groups happen
    # to be merged in: `direction` is aligned to `indices`, and `describe()`
    # zips the two, so a mismatch silently prints each ratio against the wrong
    # term.
    lookup = {**dict(zip(a.term_indices, a.term_names)), **dict(zip(b.term_indices, b.term_names))}
    names = tuple(lookup[i] for i in indices)
    return TieGroup(module=a.module, term_indices=indices, kind=kind, rank=rank,
                    direction=direction, offset=offset, participants=participants, gamma=gamma,
                    term_names=names, fit_error=err, use_offset=a.use_offset or b.use_offset)


# ---------------------------------------------------------------------------
# TieSet: the accepted ties, as a differentiable reparameterization
# ---------------------------------------------------------------------------

class TieSet(nn.Module):
    """The accepted ties for a model, as a differentiable map from free
    parameters to the full SINDy coefficient tensors.

    Holds, per module, a free coefficient tensor (E, P, X, T) used wherever no
    group claims a term, plus one ``theta`` (E, P, X, rank) per group and the
    group's shared direction/offset. `build_coefficients` assembles these into
    the tensors the model consumes, so a behavioural refit optimizes the
    per-participant loadings and the shared constants in the same backward
    pass.
    """

    def __init__(self, coefficients: Dict[str, np.ndarray], presence: Dict[str, np.ndarray],
                 terms: Dict[str, List[str]], device=None, dtype=torch.float32):
        super().__init__()
        self.modules_list = list(coefficients.keys())
        self.terms = terms
        self.groups: List[TieGroup] = []
        self._dtype = dtype

        self.free = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(c, dtype=dtype, device=device))
            for m, c in coefficients.items()
        })
        for m, p in presence.items():
            self.register_buffer(f"presence_{m}", torch.tensor(p, dtype=dtype, device=device))
        self._group_params = nn.ParameterList()
        self._group_state: List[dict] = []
        self._base_coefficients = {m: c.copy() for m, c in coefficients.items()}
        self._base_presence = {m: p.copy() for m, p in presence.items()}

    # -- construction ------------------------------------------------------

    def add_group(self, group: TieGroup) -> None:
        """Register a group and initialize its free parameters from the
        current coefficients (so adding a tie starts the refit at the best
        available projection rather than at random)."""
        device = self.free[group.module].device
        C = torch.tensor(self._base_coefficients[group.module], dtype=self._dtype, device=device)
        indices = torch.tensor(group.term_indices, dtype=torch.long, device=device)
        block = C[:, :, :, indices]                                   # (E, P, X, |S|)
        offset = torch.tensor(group.offset, dtype=self._dtype, device=device)

        state = {
            'group': group,
            'indices': indices,
            'applies': torch.tensor(group.participants, dtype=torch.bool, device=device),
        }
        if group.kind == 'constant':
            state['offset'] = nn.Parameter(offset.clone())
            self._group_params.append(state['offset'])
            state['theta'] = None
        else:
            direction = torch.tensor(group.direction, dtype=self._dtype, device=device)  # (rank, |S|)
            theta = torch.einsum('epxs,rs->epxr', block - offset, direction)
            state['theta'] = nn.Parameter(theta.contiguous())
            self._group_params.append(state['theta'])
            if group.kind == 'geometric':
                state['gamma'] = nn.Parameter(torch.tensor(float(group.gamma), dtype=self._dtype, device=device))
                self._group_params.append(state['gamma'])
                state['direction'] = None
            else:
                state['direction'] = nn.Parameter(direction.clone())
                self._group_params.append(state['direction'])
            if group.use_offset:
                state['offset'] = nn.Parameter(offset.clone())
                self._group_params.append(state['offset'])
            else:
                state['offset'] = None
        self._group_state.append(state)
        self.groups.append(group)

    def copy_with(self, group: TieGroup, drop: Sequence[TieGroup] = ()) -> 'TieSet':
        """A new TieSet with ``group`` added and every group in ``drop``
        removed, initialized from the *original* checkpoint coefficients.

        Used by the greedy search to evaluate a candidate without mutating the
        accepted set, and to replace groups by their merge with an overlapping
        candidate.
        """
        device = self.free[self.modules_list[0]].device
        new = TieSet(self._base_coefficients, self._base_presence, self.terms,
                     device=device, dtype=self._dtype)
        dropped = {id(g) for g in drop}
        for existing in self.groups:
            if id(existing) in dropped:
                continue
            new.add_group(existing)
        new.add_group(group)
        return new

    def contains(self, candidate: TieGroup) -> bool:
        """Whether an accepted group already imposes exactly this tie.

        Distinct candidates routinely merge to the same union -- a tie on
        ``reward[t], reward[t-1]`` and one on ``reward[t-1], reward[t-2]``
        both fold into the same lag-family group once the first is accepted --
        and re-evaluating an identical tie set costs a full behavioural refit
        to rediscover a gain of exactly zero (measured: two of the first
        thirteen candidates on dezfouli2019, ~3 minutes each).
        """
        for group in self.groups:
            if (group.module == candidate.module
                    and group.term_indices == candidate.term_indices
                    and group.rank == candidate.rank
                    and np.array_equal(group.participants, candidate.participants)):
                return True
        return False

    def conflicts(self, candidate: TieGroup) -> List[TieGroup]:
        """Every accepted group sharing a (participant, term) cell with
        ``candidate``.

        Returns *all* of them, not the first: a candidate can straddle two
        already-accepted groups (a tie on ``reward[t-1], reward[t-2]`` overlaps
        both an accepted ``reward[t], reward[t-1]`` and an accepted
        ``reward[t-2], reward[t-3]``), and merging with only one leaves the
        other still claiming a shared term. Two groups owning the same cell is
        silently wrong in both directions: `build_coefficients` *sums* their
        contributions, so the coefficient obeys neither declared constraint,
        and `dof_per_participant` subtracts the saving twice, inflating the
        reported BIC gain.
        """
        hits: List[TieGroup] = []
        for group in self.groups:
            if group.module != candidate.module:
                continue
            if not set(group.term_indices) & set(candidate.term_indices):
                continue
            if (group.participants & candidate.participants).any():
                hits.append(group)
        return hits

    def assert_disjoint(self) -> None:
        """Raise if any two groups claim the same (participant, term) cell.

        Cheap invariant check; the failure it guards against produces plausible
        numbers rather than an error, so it is worth asserting rather than
        trusting the search to maintain it.
        """
        for module in self.modules_list:
            claims = np.zeros(self._base_coefficients[module].shape[1:], dtype=int)  # (P, X, T)
            for group in self.groups:
                if group.module != module:
                    continue
                claims[:, :, list(group.term_indices)] += group.participants[:, :, None]
            if (claims > 1).any():
                doubled = np.argwhere(claims > 1)[0]
                raise AssertionError(
                    f"{module}: term {self.terms[module][doubled[2]]} claimed by more than one "
                    f"tie group for participant {doubled[0]}")

    # -- evaluation --------------------------------------------------------

    def _free_mask(self, module: str, device) -> torch.Tensor:
        """(P, X, T) mask: 1 where a term is not claimed by any group for that
        (participant, experiment)."""
        shape = self._base_coefficients[module].shape[1:]
        mask = torch.ones(shape, dtype=self._dtype, device=device)
        for state in self._group_state:
            if state['group'].module != module:
                continue
            applies = state['applies']                                  # (P, X)
            mask[applies[:, :, None].expand(shape) &
                 torch.zeros(shape, dtype=torch.bool, device=device).index_fill_(
                     -1, state['indices'], True)] = 0.0
        return mask

    def _direction(self, state: dict) -> torch.Tensor:
        group = state['group']
        if group.kind == 'geometric':
            powers = torch.arange(group.size, device=state['gamma'].device, dtype=self._dtype)
            v = state['gamma'] ** powers
            return (v / v.norm().clamp_min(1e-8)).unsqueeze(0)
        direction = state['direction']
        return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def build_coefficients(self) -> Dict[str, torch.Tensor]:
        """Assemble the (E, P, X, T) coefficient tensor per module from the
        free parameters, the group loadings and the shared constants.

        Differentiable end to end; the presence mask is applied last, so
        pruned terms stay exactly zero regardless of what the ties do.
        """
        out: Dict[str, torch.Tensor] = {}
        for module in self.modules_list:
            device = self.free[module].device
            coefficients = self.free[module] * self._free_mask(module, device)
            for state in self._group_state:
                group = state['group']
                if group.module != module:
                    continue
                offset = state['offset'] if state['offset'] is not None else \
                    torch.zeros(group.size, dtype=self._dtype, device=device)
                if group.kind == 'constant':
                    contribution = offset.expand(*coefficients.shape[:3], group.size)
                else:
                    contribution = torch.einsum('epxr,rs->epxs', state['theta'], self._direction(state)) + offset
                applies = state['applies'][None, :, :, None].to(self._dtype)   # (1, P, X, 1)
                scatter = torch.zeros_like(coefficients)
                scatter.index_add_(-1, state['indices'], contribution * applies)
                coefficients = coefficients + scatter
            out[module] = coefficients * getattr(self, f"presence_{module}")
        return out

    # -- accounting --------------------------------------------------------

    def dof_per_participant(self, base_count: np.ndarray) -> np.ndarray:
        """Free parameters per (participant, experiment) after the ties.

        ``base_count`` is the untied count, i.e.
        ``model.count_sindy_coefficients().cpu().numpy()`` -- starting from
        that keeps every number directly comparable to the parameter counts
        reported everywhere else in the codebase. Each group subtracts its
        per-participant savings where it applies, and adds its shared
        constants amortized over all participants (a population-level
        parameter is still a parameter; charging each participant an equal
        share is what keeps per-group BIC honest).
        """
        dof = np.asarray(base_count, dtype=float).copy()
        n_cells = max(1, dof.size)   # (participant, experiment) pairs
        for group in self.groups:
            dof -= group.participants * (group.size - group.rank)
            dof += group.n_shared / n_cells
        return dof

    def summary(self) -> str:
        if not self.groups:
            return "(no ties)"
        lines = []
        for group in self.groups:
            lines.append(group.describe())
        total = sum(g.dof_saved for g in self.groups)
        lines.append(f"total parameters removed: {total}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Applying a TieSet to a live model
# ---------------------------------------------------------------------------

@contextmanager
def functional_coefficients(model):
    """Temporarily swap ``model.sindy_coefficients`` for a plain dict, so a
    *computed* (non-leaf) tensor can be installed and gradients flow through
    it back to whatever produced it.

    ``sindy_coefficients`` is an ``nn.ParameterDict``, which only accepts
    ``nn.Parameter`` values, and assigning over the attribute directly is
    refused by ``nn.Module.__setattr__`` because the name is registered as a
    submodule. Popping it out of ``_modules`` and putting a plain dict in
    ``__dict__`` sidesteps both and is exactly reversible. The forward path
    only ever indexes this mapping (see `BaseModel.call_module`), so a plain
    dict of tensors serves it unchanged.

    Yields the dict; assign into it before each forward.
    """
    saved = model._modules.pop('sindy_coefficients')
    holder = {key: value for key, value in saved.items()}
    model.__dict__['sindy_coefficients'] = holder
    try:
        yield holder
    finally:
        model.__dict__.pop('sindy_coefficients', None)
        model._modules['sindy_coefficients'] = saved


def install_coefficients(model, coefficients: Dict[str, torch.Tensor]) -> None:
    """Make ``coefficients`` the ones the model's forward pass will use.

    Works both inside and outside `functional_coefficients`: outside, the
    values are copied into the real ``nn.Parameter`` tensors (so the model can
    be evaluated, saved, or handed to any other analysis script unchanged);
    inside, they replace the dict entries, which keeps the gradient path to
    whatever produced them. Getting this wrong is silent -- copying into
    ``.data`` of an already-installed *computed* tensor updates a temporary
    and leaves the model's actual parameters untouched -- hence the explicit
    branch rather than a single code path.
    """
    holder = model.sindy_coefficients
    for module, values in coefficients.items():
        target = holder[module]
        if isinstance(target, nn.Parameter):
            with torch.no_grad():
                target.data.copy_(values.detach().to(target.dtype).to(target.device))
        else:
            holder[module] = values


# ---------------------------------------------------------------------------
# Symbolic reporting
# ---------------------------------------------------------------------------

def tied_equation_string(tie_set: TieSet, coefficients: Dict[str, torch.Tensor],
                         participant_id: int, experiment_id: int = 0,
                         ensemble_member: int = 0, threshold: float = 1e-3) -> str:
    """One participant's equations with tied terms shown as their shared
    profile times a single free parameter, rather than as independent numbers.

    Terms belonging to a rank-1 group print as ``theta * (r1 term1 + r2
    term2 ...)`` with the ratios ``r`` shared across the population, which is
    the whole point of the reduction: the participant contributes the one
    number in front.
    """
    lines = []
    for module in tie_set.modules_list:
        terms = tie_set.terms[module]
        values = coefficients[module][ensemble_member, participant_id, experiment_id].detach().cpu().numpy()
        claimed = set()
        parts = []
        for state in tie_set._group_state:
            group = state['group']
            if group.module != module or not group.participants[participant_id, experiment_id]:
                continue
            claimed.update(group.term_indices)
            if group.kind == 'constant':
                for index, value in zip(group.term_indices, group.offset):
                    if abs(value) > threshold:
                        parts.append(f"{value:+.3f} {terms[index]}  [shared]")
                continue
            direction = tie_set._direction(state)[0].detach().cpu().numpy()
            theta = float(state['theta'][ensemble_member, participant_id, experiment_id, 0])
            ref = direction[np.argmax(np.abs(direction))]
            ratios = direction / ref
            profile = ' '.join(f"{r:+.3f} {terms[i]}" for i, r in zip(group.term_indices, ratios))
            label = f"gamma={group.gamma:.3f}" if group.kind == 'geometric' else 'tied'
            parts.append(f"{theta * ref:+.3f} * ({profile})  [{label}]")
        for index, value in enumerate(values):
            if index in claimed or abs(value) <= threshold:
                continue
            parts.append(f"{value:+.3f} {terms[index]}")
        body = ' '.join(parts) if parts else '0'
        lines.append(f"{module}[t+1] = {module}[t] + {body}")
    return "\n".join(lines)

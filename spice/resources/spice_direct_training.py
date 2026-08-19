"""RNN-free SPICE training: fit SINDy coefficients straight to behaviour.

The standard pipeline (`spice_training.fit_spice`) is two-stage and RNN-mediated:
an RNN is fitted to behaviour while SINDy equations regularize its submodules,
then Stage 2 refits those equations to reproduce the *frozen RNN's hidden-state
trajectories*. The equations therefore never see the behavioural likelihood
directly -- they are fitted to a surrogate target in state space.

This module removes the RNN from the loop entirely. `model.use_sindy` is held
True for the whole run, so `BaseModel.call_module` takes only its SINDy branch
and `submodules_rnn` is never called; the per-participant SINDy coefficients are
the model, and they are optimized by backprop through the autoregressive rollout
against the same cross-entropy loss the RNN would have been trained on.

Two stages, mirroring the standard pipeline's division of labour but with a
behavioural objective in both:

  Stage 1 (structure discovery) -- coefficients trained with an L1 penalty
    (``sindy_alpha``) plus periodic pruning, so the sparsity pattern is
    discovered rather than assumed. Pruning reuses the existing machinery
    (`BaseModel.concept_gate_patience` / `.prune_concept_gates`
    for per-member thresholding, `_ensemble_ratio_test` for cross-member
    consensus), so the discovered support means the same thing it does
    elsewhere in the codebase.

  Stage 2 (debiased refit) -- the presence masks discovered in Stage 1 are
    frozen, ``sindy_alpha`` is set to 0 and pruning is switched off; the
    surviving coefficients are then refitted from their Stage 1 values.
    This is the standard "relaxed lasso"/debiasing step: an L1 penalty
    shrinks the coefficients it selects, so re-estimating them without the
    penalty removes that bias. Warm-starting (rather than re-initializing,
    as Stage 2.2 of the RNN pipeline does) keeps the refit in the basin the
    structure search actually selected.

What is trained: the concept factorization (`sindy_concept_loadings` and
`sindy_concept_directions`) and, when the model defines them,
`learnable_initial_values` (per-participant initial memory states -- part of the
equation system, not the RNN). `submodules_rnn`, `participant_embedding` and
receive no gradient in SINDy mode and are left frozen;
`participant_embedding` in particular only ever enters through the RNN branch,
so a model fitted this way carries no embedding information.

Ensemble handling: members are bootstrapped *within participant* (sessions are
resampled from that participant's own sessions), not across the whole session
pool as `fit_spice` does. With per-participant coefficients, a global bootstrap
leaves ~37% of (participant, member) slots with no data at all, so their
coefficients never move off their random initialization and the ensemble ratio
test reads that silence as evidence against the term. Stratifying keeps every
participant present in every member.

Usage:

    from spice.resources.spice_direct_training import fit_spice_direct

    model, history = fit_spice_direct(
        model=estimator.model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs_stage1=300,
        epochs_stage2=150,
        sindy_alpha=1e-3,
        sindy_threshold_pruning=0.1,
        sindy_ensemble_pruning=0.7,
    )
"""

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .model import BaseModel
from .spice_utils import SpiceDataset
from .training import cross_entropy_loss
from .training.pruning import _ensemble_ratio_test
from .training.reporting import _get_terminal_width


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def bootstrap_within_participant(
    xs: torch.Tensor, ys: torch.Tensor, ensemble_size: int, seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample sessions with replacement *within each participant*.

    Returns 5D ``(E, B, T, W, F)`` / ``(E, B, T, W, A)`` tensors. Every
    participant keeps exactly its original number of sessions in every
    ensemble member, so no (participant, member) slot is ever left without
    data -- see the module docstring for why a global bootstrap is unsafe
    when coefficients are per-participant.
    """
    if ensemble_size == 1:
        return xs.unsqueeze(0), ys.unsqueeze(0)

    generator = torch.Generator().manual_seed(seed)
    participant_ids = xs[:, 0, 0, -1].long()
    all_rows = torch.arange(xs.shape[0])
    indices = torch.empty(ensemble_size, xs.shape[0], dtype=torch.long)
    for participant in participant_ids.unique():
        rows = all_rows[participant_ids == participant]
        draw = torch.randint(len(rows), (ensemble_size, len(rows)), generator=generator)
        indices[:, rows] = rows[draw]
    return xs[indices], ys[indices]


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------

def _apply_constraints(model: BaseModel, optimizer=None, sindy_alpha: float = 0.0) -> None:
    """Re-establish the factorization's constraints after an optimizer step.

    Non-negativity plus the L1 prox on the loadings, then the unit-norm gauge on the
    concept directions. The L1 is proximal rather than a loss term because Adam never
    drives a penalized parameter to exactly zero, and an exact zero is what makes a
    closed gate mean "this unit does not have this concept". The gauge has to be
    re-fixed every step: Z @ V is invariant under (Z D, D^-1 V), so leaving V free lets
    the optimizer shrink the penalty at no cost to the fit.
    """
    lr = 0.0
    if optimizer is not None:
        lr = max((group.get('lr', 0.0) for group in optimizer.param_groups), default=0.0)
    model.project_loadings(lr=lr, sindy_alpha=sindy_alpha)
    model.normalize_concept_directions()


def run_epoch_direct(
    model: BaseModel,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sindy_alpha: float = 0.0,
    n_steps: Optional[int] = None,
    grad_clip: float = 1.0,
    ensemble_average: bool = False,
) -> float:
    """One pass over ``xs``/``ys`` in pure-SINDy mode; returns mean CE loss.

    ``ensemble_average=False`` (the training default) scores each ensemble
    member separately and averages the losses -- the right training signal,
    since every member fits its own bootstrap resample. ``True`` averages the
    *logits* across members before scoring, which is what
    `analysis_model_evaluation` (and every other evaluation in this repo)
    reports. The two differ by however much ensemble averaging helps, which
    grows with member diversity: ~0.03 trial likelihood on a tightly-pruned
    fit, but >0.10 on a loosely-pruned one whose members disagree. Always use
    ``True`` for anything compared against reported numbers -- a per-member
    curve can trend down while the reported ensemble metric trends up.

    ``optimizer=None`` evaluates without updating (call under `torch.no_grad`).
    ``n_steps`` truncates backprop-through-time into chunks, carrying the state
    across chunk boundaries detached -- the rollout stays continuous, only the
    gradient path is cut.

    Sessions are NaN-padded to the longest session, so a late chunk can contain
    no valid trials at all; cross-entropy over an empty tensor is NaN, so such
    chunks are stepped through (to keep the state rolling) but contribute
    nothing to the loss. The returned value is weighted by each chunk's valid
    trial count, making it a true per-trial mean whose ``exp(-loss)`` is the
    geometric-mean trial likelihood -- an unweighted mean over chunks is not.
    """
    if n_steps is None:
        n_steps = xs.shape[2]

    state = None
    total_loss, total_trials = 0.0, 0
    for t in range(0, xs.shape[2], n_steps):
        chunk = min(xs.shape[2] - t, n_steps)
        xs_step, ys_step = xs[:, :, t:t + chunk], ys[:, :, t:t + chunk]

        predictions, _ = model(xs_step, state)
        state = model.get_state(detach=True)

        # (E, B, T, W) -- drop padded trials before the loss, as _run_batch_training does
        valid = ~torch.isnan(xs_step[..., :model.n_actions].sum(dim=-1))
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue

        if ensemble_average:
            # Average logits over members, then score once -- matches
            # analysis_model_evaluation's get_choice_probs(preds.mean(dim=0)).
            valid_shared = valid.all(dim=0)  # (B, T, W); padding is identical across members
            loss = cross_entropy_loss(predictions.mean(dim=0)[valid_shared], ys_step[0][valid_shared])
        else:
            loss = cross_entropy_loss(predictions[valid], ys_step[valid])

        if optimizer is not None:
            step_loss = loss
            if sindy_alpha > 0:
                step_loss = step_loss + model.compute_constants_penalty(sindy_alpha=sindy_alpha)
            optimizer.zero_grad()
            step_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']], max_norm=grad_clip,
                )
            optimizer.step()
            _apply_constraints(model, optimizer, sindy_alpha or 0.0)

        total_loss += loss.item() * n_valid
        total_trials += n_valid

    if total_trials == 0:
        raise RuntimeError("No valid trials in this pass -- every trial was NaN-padded.")

    mean_loss = total_loss / total_trials
    if not np.isfinite(mean_loss):
        raise RuntimeError(
            f"Non-finite loss ({mean_loss}) in run_epoch_direct. Refusing to continue: a NaN here "
            "propagates silently into nansum-based scoring downstream and can look like a perfect fit."
        )
    return mean_loss


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_step(
    model: BaseModel,
    threshold: float,
    patience: int = 1,
    ensemble_ratio: Optional[float] = None,
    n_terms_pruning: Optional[int] = None,
) -> Tuple[int, int]:
    """One pruning event. Returns (n_active_before, n_active_after) slots.

    Per-member hard thresholding with a patience counter (a term must sit
    below ``threshold`` for ``patience`` consecutive events before removal),
    optionally followed by the ensemble ratio test: a term survives for a
    (participant, experiment) only if at least ``ensemble_ratio`` of members
    support it, and is otherwise removed from *all* members. Theory-driven
    prior masks are re-applied last and are never revived.
    """
    modules = model.get_modules()
    before = int(sum(model.sindy_concept_gates[m].sum().item() for m in modules))

    model.concept_gate_patience(threshold=threshold)
    model.concept_support_patience(threshold=threshold)
    model.prune_concept_gates(patience=patience, n_concepts_pruning=n_terms_pruning)
    model.prune_concept_support(patience=patience, n_terms_pruning=n_terms_pruning)

    if ensemble_ratio is not None:
        with torch.no_grad():
            for module in modules:
                consensus = _ensemble_ratio_test(
                    coefficients=model.effective_loadings(module).detach(),
                    presence=model.sindy_concept_gates[module],
                    threshold=threshold,
                    ratio=ensemble_ratio,
                )  # (P, X, terms)
                model.sindy_concept_gates[module] &= consensus.unsqueeze(0)

    with torch.no_grad():
        for module in modules:
                model.retire_dead_concepts(module)
    _apply_constraints(model)

    after = int(sum(model.sindy_concept_gates[m].sum().item() for m in modules))
    return before, after


# ---------------------------------------------------------------------------
# Scoring the three objects a fitted ensemble contains
# ---------------------------------------------------------------------------

@contextmanager
def averaged_coefficients(model: BaseModel):
    """Temporarily collapse the ensemble to its mean coefficient set.

    This is the model `print_spice_model` / `get_concept_loadings(aggregate=True)`
    describe -- the single equation you would publish. Same convention as
    aggregation: averaged values are already zero wherever a
    term is inactive in every member, so presence is set all-True and the zeros
    carry the 'inactive' meaning.
    """
    modules = model.get_modules()
    saved = {m: (model.sindy_concept_loadings[m].data.clone(),
                 model.sindy_concept_gates[m].clone()) for m in modules}
    try:
        aggregated = model.get_concept_loadings(aggregate=True)  # (P, X, C)
        for module in modules:
            values = aggregated[module].unsqueeze(0).expand(model.ensemble_size, -1, -1, -1).clone()
            model.sindy_concept_loadings[module].data = values.to(saved[module][0].dtype)
            model.sindy_concept_gates[module] = torch.ones_like(saved[module][1])
        yield model
    finally:
        for module in modules:
            model.sindy_concept_loadings[module].data = saved[module][0]
            model.sindy_concept_gates[module] = saved[module][1]


@torch.no_grad()
def score_three_ways(model: BaseModel, xs: torch.Tensor, ys: torch.Tensor) -> Dict[str, float]:
    """Trial likelihood of the three distinct models a fitted ensemble contains.

    They are genuinely different objects and can diverge sharply:
      per_member : mean likelihood of the individual members -- what a single
                   member's equations achieve on their own.
      logit_avg  : softmax of the ensemble-mean logits -- what
                   `analysis_model_evaluation` reports (the headline number),
                   but not expressible as one equation.
      coef_avg   : the ensemble-mean coefficient set -- what `print_spice_model`
                   reports, i.e. the equation actually published.

    ``xs``/``ys`` are 4D ``(B, T, W, F)``; the ensemble dimension is added here.
    """
    model.eval(use_sindy=True)
    E = model.ensemble_size
    valid = ~torch.isnan(xs[:, :, 0, 0])
    targets = torch.nan_to_num(ys, nan=0.0)

    model.init_state(batch_size=xs.shape[0])
    logits, _ = model(xs)                                        # (E, B, T, W, A)

    log_probs_avg = torch.log_softmax(logits.mean(0), dim=-1)
    ll_avg = (targets * log_probs_avg).sum(-1).sum(-1)
    logit_avg = float(torch.exp((ll_avg * valid).sum() / valid.sum()))

    log_probs_member = torch.log_softmax(logits, dim=-1)
    ll_member = (targets.unsqueeze(0) * log_probs_member).sum(-1).sum(-1)
    per_member = float(torch.exp((ll_member * valid.unsqueeze(0)).sum() / (valid.sum() * E)))

    with averaged_coefficients(model):
        model.init_state(batch_size=xs.shape[0])
        logits_coef, _ = model(xs)
        log_probs_coef = torch.log_softmax(logits_coef[0], dim=-1)
        ll_coef = (targets * log_probs_coef).sum(-1).sum(-1)
        coef_avg = float(torch.exp((ll_coef * valid).sum() / valid.sum()))

    return dict(per_member=per_member, logit_avg=logit_avg, coef_avg=coef_avg)


def _snapshot(model: BaseModel) -> dict:
    """Deep copy of everything the refit can change (coefficients, support, initial values)."""
    return dict(
        coefficients={m: model.sindy_concept_loadings[m].data.clone() for m in model.get_modules()},
        presence={m: model.sindy_concept_gates[m].clone() for m in model.get_modules()},
        initial_values={k: p.data.clone() for k, p in model.learnable_initial_values.items()},
    )


def _restore(model: BaseModel, snapshot: dict) -> None:
    for module in model.get_modules():
        model.sindy_concept_loadings[module].data = snapshot['coefficients'][module].clone()
        model.sindy_concept_gates[module] = snapshot['presence'][module].clone()
    for key, parameter in model.learnable_initial_values.items():
        parameter.data = snapshot['initial_values'][key].clone()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _flat_coefficients(model: BaseModel, aggregate: bool = True) -> torch.Tensor:
    """All modules' coefficients concatenated along the term axis (presence-masked)."""
    coefs = model.get_concept_loadings(aggregate=aggregate)
    return torch.cat([coefs[m].reshape(*coefs[m].shape[:-1], -1) for m in model.get_modules()], dim=-1)


def coefficient_diagnostics(model: BaseModel, previous: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Stability/sparsity summary of the current coefficients.

    - ``n_active_mean``  : active coefficients per (participant, experiment)
    - ``coef_drift``     : mean |Δcoefficient| since the last snapshot, over
                           terms active in both -- the direct read on whether
                           the fit has settled
    - ``ensemble_cv``    : median across active terms of std/|mean| over
                           ensemble members; high values mean members disagree
                           on magnitude even where they agree a term exists
    - ``support_agree``  : fraction of (participant, experiment, term) slots on
                           which all ensemble members agree about presence
    - ``max_abs``        : largest |coefficient| (a blow-up canary)
    """
    aggregated = _flat_coefficients(model, aggregate=True)          # (P, X, T_total)
    per_member = _flat_coefficients(model, aggregate=False)         # (E, P, X, T_total)
    presence = torch.cat(
        [model.sindy_concept_gates[m] for m in model.get_modules()], dim=-1,
    )                                                              # (E, P, X, T_total)

    active = aggregated != 0
    diagnostics = {
        'n_active_mean': float(model.count_spice_parameters()['loadings'].mean().item()),
        'max_abs': float(aggregated.abs().max().item()),
        'support_agree': float((presence.all(dim=0) | (~presence.any(dim=0))).float().mean().item()),
    }

    if model.ensemble_size > 1 and active.any():
        member_std = per_member.std(dim=0)
        member_mean = per_member.mean(dim=0).abs()
        cv = member_std[active] / (member_mean[active] + 1e-8)
        diagnostics['ensemble_cv'] = float(cv.median().item())
    else:
        diagnostics['ensemble_cv'] = float('nan')

    if previous is not None:
        both_active = active & (previous != 0)
        diagnostics['coef_drift'] = (
            float((aggregated[both_active] - previous[both_active]).abs().mean().item())
            if both_active.any() else float('nan')
        )
    else:
        diagnostics['coef_drift'] = float('nan')

    return diagnostics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fit_spice_direct(
    model: BaseModel,
    dataset_train: SpiceDataset,
    dataset_test: Optional[SpiceDataset] = None,

    epochs_stage1: int = 300,
    epochs_stage2: int = 150,
    n_warmup_epochs: int = 50,

    learning_rate: float = 0.01,
    learning_rate_stage2: float = 0.003,
    n_steps: Optional[int] = None,
    grad_clip: float = 1.0,

    sindy_alpha: float = 1e-3,
    sindy_threshold_pruning: float = 0.1,
    sindy_ensemble_pruning: Optional[float] = 0.7,
    sindy_pruning_frequency: int = 10,
    sindy_pruning_patience: int = 2,

    check_every: int = 10,
    select_metric: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[BaseModel, pd.DataFrame]:
    """Fit SINDy coefficients directly to behaviour, bypassing the RNN.

    Parameters
    ----------
    model : BaseModel
        The SPICE model whose concept factorization will be fitted. Its RNN
        submodules are never called and never updated.
    dataset_train, dataset_test : SpiceDataset
        Training data, and optional held-out data used *for reporting only* --
        it never influences pruning, stopping, or any other decision here.
    epochs_stage1, epochs_stage2 : int
        Structure-discovery and debiased-refit epoch budgets.
    n_warmup_epochs : int
        Epochs at the start of Stage 1 during which no pruning happens, so
        coefficients can grow away from their ~1e-3 initialization before
        being judged against ``sindy_threshold_pruning``. Pruning from epoch 0
        would remove essentially every term.
    sindy_alpha : float
        L1 penalty strength during Stage 1. Set to 0 in Stage 2 by construction.
    sindy_threshold_pruning : float
        |coefficient| below which a term counts as inactive at a pruning event.
    sindy_ensemble_pruning : float, optional
        Fraction of ensemble members that must support a term for it to
        survive. None disables the consensus test (per-member pruning only).
    check_every : int
        Epoch interval for the diagnostics in `coefficient_diagnostics`.
    select_metric : callable or {'coef_avg', 'logit_avg', 'per_member'}, optional
        When set, keep a snapshot of the best-scoring model on *dataset_test*
        (across both stages) and restore it at the end instead of returning the
        final epoch's model. Higher is better.

        A **callable** ``(model, dataset) -> float`` is the recommended form,
        because the three string options score a *likelihood*, which carries no
        parameter penalty -- across Stage 1 the active-coefficient count falls
        from the full library to a handful, so a likelihood criterion reliably
        prefers the dense pre-pruning model and throws Stage 1's sparsification
        away. Pass a ΔBIC-style scorer instead whenever parsimony matters.
        (`spice` deliberately does not import the study-side BIC machinery, so
        the frontend supplies it -- see `dezfouli2019_direct.py`.)

        **This selects on the evaluation set, so the resulting score is not a
        held-out estimate** -- it is optimistic by however much the peak exceeds
        a typical epoch, and must not be reported as out-of-sample performance.
        It stands in for a proper validation split, which is what this argument
        should eventually point at. Left at ``None`` (the default), nothing is
        selected and training simply returns its final state.

        Which metric matters depends on what you publish: the three peak at
        different epochs (on dezfouli2019 the coefficient-average peaked ~100
        epochs before the logit-average), so selecting on ``logit_avg`` can hand
        back a materially worse *equation*. ``coef_avg`` matches
        `print_spice_model`.

    Returns
    -------
    model : BaseModel
        Fitted in place (returned for convenience), left in eval/SINDy mode.
    history : pd.DataFrame
        One row per check, with stage, losses, trial likelihoods, and the
        stability diagnostics.
    """
    torch.manual_seed(seed)
    device = model.device

    xs_train, ys_train = bootstrap_within_participant(
        dataset_train.xs, dataset_train.ys, model.ensemble_size, seed=seed,
    )
    xs_train, ys_train = xs_train.to(device), ys_train.to(device)

    if dataset_test is not None:
        xs_test = dataset_test.xs.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1, -1).to(device)
        ys_test = dataset_test.ys.unsqueeze(0).expand(model.ensemble_size, -1, -1, -1, -1).to(device)

    # Only the equation system is trainable: SINDy coefficients + per-participant
    # initial memory states. Everything else (RNN submodules, participant
    # embedding, damping) is inert in SINDy mode -- see module docstring.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: List[torch.nn.Parameter] = []
    for module in model.get_modules():
        model.sindy_concept_loadings[module].requires_grad_(True)
        model.sindy_concept_directions[module].requires_grad_(True)
        trainable.append(model.sindy_concept_loadings[module])
        trainable.append(model.sindy_concept_directions[module])
    for parameter in model.learnable_initial_values.values():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    model.fit_sindy = False  # no RNN->SINDy regularization term; there is no RNN here
    model.train(mode=True, use_sindy=True)

    if select_metric is not None:
        if dataset_test is None:
            raise ValueError("select_metric requires dataset_test to score against.")
        if not callable(select_metric) and select_metric not in ('coef_avg', 'logit_avg', 'per_member'):
            raise ValueError(
                f"select_metric must be a callable, 'coef_avg', 'logit_avg' or 'per_member', got {select_metric!r}."
            )
        if not callable(select_metric) and verbose:
            print("  WARNING: selecting on a likelihood ignores parameter count, so the best score "
                  "will tend to land on the dense pre-pruning model and undo Stage 1's sparsification. "
                  "Pass a callable scoring ΔBIC instead if parsimony matters.")

    history: List[dict] = []
    previous_coefficients = None
    best = dict(score=-float('inf'), snapshot=None, epoch=None, stage=None)

    def _check(epoch: int, stage: str, train_loss: float) -> None:
        nonlocal previous_coefficients
        model.eval(use_sindy=True)
        scores = (score_three_ways(model, dataset_test.xs.to(device), dataset_test.ys.to(device))
                  if dataset_test is not None else {})
        diagnostics = coefficient_diagnostics(model, previous_coefficients)
        previous_coefficients = _flat_coefficients(model, aggregate=True).clone()

        row = dict(
            stage=stage, epoch=epoch,
            train_loss=train_loss, train_trial_lik=float(np.exp(-train_loss)),
            **{f'test_{k}': v for k, v in scores.items()},
            **diagnostics,
        )

        if select_metric is not None:
            selection_score = (select_metric(model, dataset_test) if callable(select_metric)
                               else scores[select_metric])
            row['selection_score'] = selection_score
            if selection_score > best['score']:
                best.update(score=selection_score, snapshot=_snapshot(model), epoch=epoch, stage=stage)
                row['is_best'] = True
            else:
                row['is_best'] = False
        else:
            row['is_best'] = False

        history.append(row)
        if verbose:
            marker = ' *BEST*' if row['is_best'] else ''
            test_str = (f"| test member {scores['per_member']:.4f} logit {scores['logit_avg']:.4f} "
                        f"coef {scores['coef_avg']:.4f} " if scores else "")
            print(
                f"  [{stage}] epoch {epoch:4d} | train lik {row['train_trial_lik']:.4f} {test_str}"
                f"| active/ppt {diagnostics['n_active_mean']:5.2f} "
                f"| drift {diagnostics['coef_drift']:.5f} "
                f"| agree {diagnostics['support_agree']:.3f} "
                f"| max|c| {diagnostics['max_abs']:.2f}{marker}"
            )
        model.train(mode=True, use_sindy=True)

    # ------------------------------------------------------------------
    # Stage 1 -- structure discovery (L1 + pruning)
    # ------------------------------------------------------------------
    if verbose:
        width = _get_terminal_width()
        print("\n" + "=" * width)
        print("Direct SINDy training (RNN bypassed)")
        print(f"  Stage 1: {epochs_stage1} epochs, alpha={sindy_alpha}, "
              f"threshold={sindy_threshold_pruning}, ensemble_ratio={sindy_ensemble_pruning}, "
              f"warmup={n_warmup_epochs}")
        print(f"  Stage 2: {epochs_stage2} epochs, alpha=0, no pruning")
        print("=" * width)

    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    for epoch in range(1, epochs_stage1 + 1):
        train_loss = run_epoch_direct(
            model, xs_train, ys_train, optimizer=optimizer,
            sindy_alpha=sindy_alpha, n_steps=n_steps, grad_clip=grad_clip,
        )

        if (epoch > n_warmup_epochs
                and sindy_threshold_pruning is not None
                and epoch % sindy_pruning_frequency == 0):
            before, after = prune_step(
                model, threshold=sindy_threshold_pruning, patience=sindy_pruning_patience,
                ensemble_ratio=sindy_ensemble_pruning,
            )
            if verbose and before != after:
                print(f"  [stage1] epoch {epoch:4d} | pruned {before - after} slots ({before} -> {after})")

        if epoch % check_every == 0 or epoch == epochs_stage1:
            _check(epoch, 'stage1', train_loss)

    # ------------------------------------------------------------------
    # Stage 2 -- freeze the discovered support, refit without the L1 penalty
    # ------------------------------------------------------------------
    frozen_support = {m: model.sindy_concept_gates[m].clone() for m in model.get_modules()}
    if verbose:
        n_active = float(model.count_spice_parameters()['loadings'].mean().item())
        print(f"\nStage 1 complete. Frozen support: {n_active:.2f} active coefficients per participant.")

    optimizer = torch.optim.Adam(trainable, lr=learning_rate_stage2)
    for epoch in range(1, epochs_stage2 + 1):
        train_loss = run_epoch_direct(
            model, xs_train, ys_train, optimizer=optimizer,
            sindy_alpha=0.0, n_steps=n_steps, grad_clip=grad_clip,
        )
        if epoch % check_every == 0 or epoch == epochs_stage2:
            _check(epoch, 'stage2', train_loss)

    # The support must be bit-identical to what Stage 1 selected; if it is not,
    # something in Stage 2 modified presence masks and every parameter count
    # reported downstream would be wrong. Checked before any best-snapshot
    # restore, which may legitimately reinstate a Stage 1 support.
    for module in model.get_modules():
        if not torch.equal(frozen_support[module], model.sindy_concept_gates[module]):
            raise RuntimeError(
                f"Stage 2 changed the sparsity pattern of module {module!r}; it must only refit "
                "coefficients on the support discovered in Stage 1."
            )

    if select_metric is not None and best['snapshot'] is not None:
        _restore(model, best['snapshot'])
        if verbose:
            print(f"\nRestored best-on-test model: {select_metric}={best['score']:.4f} "
                  f"at {best['stage']} epoch {best['epoch']}.")
            print("  NOTE: selected on the evaluation set -- this score is optimistic and is "
                  "not a held-out estimate.")

    model.eval(use_sindy=True)
    return model, pd.DataFrame(history)

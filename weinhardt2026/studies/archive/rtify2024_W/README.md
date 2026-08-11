# rtify2024 — Synthetic DDM Study (Handover Notes)

## What this study is

A synthetic recovery study: can SPICE (RNN + SINDy) recover a classic
two-boundary drift-diffusion model (DDM) from simulated choice/RT data,
without any hand-coded decision rule?

There's no real experiment here — `benchmark_rtify2024.py` generates ground
truth data from a known DDM, and the SPICE pipeline (`rtify2024.py`) tries to
fit it back. The interesting question is whether the fitted model's `drift`
mechanism recovers the true, roughly-constant, stimulus-driven drift rate, or
finds some other (decay/ramp) trajectory that fits the RT/choice distribution
equally well. As of this handover, it's the latter — see "Open problem" below.

## File layout

| File | Role |
|---|---|
| `spice_rtify2024.py` | `CONFIG`, `SpiceDDM` (the fitted model), `make_ddm_loss` |
| `benchmark_rtify2024.py` | `simulate_ddm` (ground truth generator), `get_dataset` |
| `analysis_rtify2024.py` | `decode_choice_rt`, `estimate_non_decision_time`, `evaluate`, `print_spice_models`, `plot_summary` |
| `rtify2024.py` | the pipeline: build dataset → fit → evaluate → plot |

## Model architecture (`spice_rtify2024.py`)

One SINDy-fitted submodule, `drift`, receiving `stimulus` as its only control
signal. `drift` *is* the rate — not the accumulator. Its trajectory is
integrated deterministically, not learned:

```python
self.call_module(key_module='drift', key_state='drift', inputs=(stimulus,), ...)
self.state['evidence'] = self.dt * torch.cumsum(self.state['drift'], dim=0)
```

This split matters:
- `drift` has a genuine learnable initial value per participant
  (`memory_state={'drift': None, ...}`) and is the only thing SINDy fits an
  equation to.
- `evidence` always starts at exactly 0 (`memory_state={'evidence': 0.0}`) and
  is a fixed, non-learnable cumsum of `drift` — never itself an RNN submodule.
  This guarantees "no information accumulated yet" is enforced by
  construction, not learned, and keeps `evidence`'s shape strictly downstream
  of `drift`'s.
- `states_in_logit=['drift']` — critical for Stage 2 (see below). `evidence`
  must NOT be in `states_in_logit`, or Stage 2's shooting/one-step machinery
  will try to treat it as an independently-fitted trajectory, which breaks its
  cumsum (see "Stage 2.1/2.2 loss gap" below).

The decision rule is a closed-form discrete-time hazard/survival model
(RTify-style), not an SDE simulation:

```
p_stop_up[w]    = sigmoid(evidence[w] - threshold)
p_stop_down[w]  = sigmoid(-evidence[w] - threshold)
p_decision_*[w] = p_stop_*[w] * prod_{k<w}(1 - p_stop[k])
```

giving a proper joint distribution over (choice, RT bin) fit by a single
closed-form NLL term — fully differentiable, no custom autograd, no
distribution-matching loss.

`threshold` is `softplus(threshold_raw)`, with `threshold_raw` a
per-participant learnable initial value only (no `setup_module` — it's never
SINDy-fitted, just a free parameter per participant/ensemble member).

Non-decision time (ndt) is applied only to the *output*: the decision process
always runs from t=0 over the full `max_steps` range internally, and ndt just
shifts the reported RT bins by a constant delay at the very end. This was a
deliberate simplification (see "What was done") — earlier versions tried to
truncate the internal computation by ndt, which broke Stage 2's
input-length-agnostic assumptions.

## What was done (chronological)

Roughly in order — see git history / prior conversation for exact diffs:

1. **Stage 2 crash fix**: a `_vectorize_state_sequential` W-mismatch crash in
   the SINDy-refit stage was traced to internal ndt truncation changing
   per-trial state lengths. Fixed by moving ndt to an output-only shift
   (see above) rather than special-casing Stage 2.
2. **Stage 2 generalized**: Stage 2.1 (sparsity discovery) now does genuine
   one-step (`dw=1`) fitting via `_flatten_state_trajectories_onestep`
   (flattens (W,T) trajectories into independent one-step pseudo-sessions);
   Stage 2.2 (coefficient estimation) does genuine multi-step rollout with
   `K=shooting_steps`.
3. **Numerical fixes along the way**: a `sindy_ridge_solve` `h_current`
   indexing bug; float64 accumulation in the ridge solve (precision); a
   misleading Stage 2.2 log message; NaN-guards after ridge solves in both
   Stage 2.1 and 2.2 (ridge succeeding numerically doesn't guarantee a finite
   loss — falls back to small random re-init + SGD if the ridge loss isn't
   finite).
4. **Major framework bug fixed**: `_run_batch_training` in
   `spice/resources/spice_training.py` was always passing a non-`None`
   `prev_state` into `model()`, which meant `init_forward_pass`'s branch that
   applies `learnable_initial_values` never fired — every participant's
   learnable initial `drift`/`threshold_raw` was permanently frozen at 0 with
   zero gradient. Fixed by only passing `state` after the first call within a
   BPTT chunk. Verified directly: before the fix, `threshold_raw` sat frozen
   at exactly 0.0 across 150 epochs; after, it moves and loss decreases
   smoothly instead of plateauing. **This bug is in shared framework code,
   not study-specific — it would have affected every study using
   `learnable_initial_values`.**
5. **Architecture redesign**: split "drift" (rate, SINDy-fitted, learnable
   initial value) from "evidence" (accumulator, deterministic
   `dt*cumsum(drift)`, fixed 0 initial value) — see architecture section
   above. This was the user's own insight.
6. **Stage 2.1 vs 2.2 ~200x loss gap**, root-caused (not just papered over):
   the flattening transform in `_flatten_state_trajectories_onestep` broke
   `evidence`'s cumsum, because a flattened one-step pseudo-session only
   integrates within its own single step, discarding all prior accumulation,
   while the recorded target came from the correct full-length computation.
   Fixed by setting `states_in_logit=['drift']` so Stage 2 never tries to
   independently one-step-fit `evidence` at all (it's not an RNN submodule,
   so it shouldn't be in the shooting/loss machinery in the first place).
7. **A reasoning error, found and reverted**: a `dt`-scaling "fix" was applied
   to `compute_weighted_coefficient_penalty` (the L1 coefficient penalty) by
   flawed analogy to a different, genuinely-correct `dt²` fix in
   `compute_sindy_loss_for_module` (which operates on *state deltas*, which do
   scale with `dt` — coefficients themselves are dt-independent per-second
   rates by design and need no such correction). Reverted. **Lesson for
   future work in this codebase: don't pattern-match a `dt`-fix from one
   context to another without re-deriving whether the quantity in question
   is a rate (dt-independent) or a delta (dt-dependent).**
8. **Experimental design change, tried and reverted**: the original setup had
   exactly one fixed stimulus/drift-rate value per participant — stimulus was
   perfectly confounded with participant identity. Tried a within-participant
   multi-block design (5 stimulus conditions per participant, 200 trials each,
   `drift_rates = torch.rand(n_participants, n_blocks)`) to break the
   confound. **Result: no improvement** — the fitted model produced the same
   decay/ramp drift dynamics as before. Reverted back to the simple one
   condition per participant design (`drift_rates = torch.rand(n_participants)`,
   `n_trials=1000` per participant); the multi-block plumbing in
   `simulate_ddm`/`get_dataset` (`block_id` parameter, 2D `drift_rates`
   support) was removed rather than left dormant.
9. **Drift smoothness regularization removed.** `drift_smoothness_weight` in
   `make_ddm_loss` (and the `loss_fn_kwargs['model']` plumbing it needed) has
   been deleted entirely, not just disabled. Decision going forward: the
   identifiability problem should be addressed through experimental design
   appropriate for flexible model-discovery approaches, not by adding prior
   assumptions/penalties into the loss.

## Open problem: identifiability

This is the main unresolved issue and the reason this study lives in
`archive/` rather than being considered "done."

Even with a clean drift/evidence split and a stimulus-driven `drift` module,
the fitted model repeatedly finds **non-true drift trajectories** — decay or
ramp shapes — that fit the aggregate RT/choice distribution about as well as
the true, roughly-constant drift. This was tested two ways:

- **Fixing `threshold=1`** (removing the multiplicative drift/threshold/gain
  degeneracy) did not resolve it — it just moved the degeneracy to a
  different shape (a linear ramp instead of decay), and produced a *worse*
  and inverted result: participant 0's true-lower drift rate ended up higher
  than participant 2's true-higher drift rate in the recovered model.
- **A smoothness penalty** (`drift_smoothness_weight` in `make_ddm_loss`,
  penalizing `mean((drift[t]-drift[0])**2)`) was tried and removed. It was
  never validated at a meaningful strength (only tested at `1e-2`, three
  orders of magnitude too weak to affect the loss), and the decision was made
  to not pursue this direction further: the identifiability problem should
  be solved through experimental design appropriate for flexible
  model-discovery approaches, not by adding prior-assumption penalties into
  the loss. The code for this has been deleted, not just disabled.
- **A within-participant multi-block redesign** (multiple stimulus/drift
  conditions per participant, breaking the stimulus/participant-identity
  confound) was implemented and tested at full scale. **Result: no
  improvement** — the fitted model produced the same decay/ramp drift
  dynamics as the original single-condition design. This has been reverted;
  the study is back to one fixed drift-rate condition per participant.

Net result: neither of the two approaches tried so far (a loss-side
smoothness prior, a data-side multi-block design) fixed identifiability.
Both are now removed/reverted, and the codebase is back to its simplest form
for whoever picks this up next.

### Suggested next steps for whoever picks this up

1. The stimulus/participant-identity confound is not the only design axis
   worth reconsidering — investigate what specifically makes drift/threshold
   under-determined given the hazard/survival likelihood (e.g., does adding
   trial-level variation *within* the same condition, rather than
   block-level conditions, help more? Does the amount of overlap between
   drift trajectories across participants matter?).
2. Consider whether `threshold_raw`'s learnable per-participant initial value
   should be more constrained (e.g. shared/fixed across participants) — the
   multiplicative degeneracy with `drift` may still be partially open even
   with the intercept/bias fixes already in place.
3. Given both attempted fixes failed, it may be worth stepping back and
   checking whether the decay/ramp trajectories found by the RNN are
   *provably* equally likely under the model (a real identifiability
   result), rather than just an optimization artifact — e.g. by directly
   comparing the fitted model's log-likelihood on ground-truth-trajectory
   constrained drift vs. its own found solution.

## Known framework-level caveat

The `_run_batch_training` fix (item 4 above) lives in
`spice/resources/spice_training.py`, not in this study's files — it's a
shared-framework fix, so any other study relying on `learnable_initial_values`
should be re-checked if it was fit before this fix landed.

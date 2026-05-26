# SPICE Train/Test Gap: Diagnosis & Action Plan

## Context

SPICE training on a 4-armed bandit dataset (5 blocks × 150 trials per participant, middle block held out). The RNN generalizes fine to the held-out block. SINDy, after Stage 2 refit, achieves near-perfect autoregressive training MSE against the RNN but fails badly on the held-out block. Adding regularization (state noise, spectral contraction, L2, state contraction in `forward_sindy`) helped marginally but did not close the gap.

## Diagnosis

The failure is **not** a regularization deficit. It is a **structural property of the fitting objective**: SINDy is fit per-participant against a single long RNN rollout, which constrains the polynomial on a 1-D curve through (h, u)-space but leaves the dynamics under-constrained in the volume around that curve.

Restated mechanically:

- Stage 2 minimizes autoregressive rollout error against the RNN teacher on training trajectories. With per-participant coefficients and a long rollout, the optimizer has enough capacity to match the curve to machine precision.
- Many different ξ values produce the same training rollout but different rollouts on held-out driving sequences (the **null space along the curve**).
- The held-out trajectories visit nearby but not identical regions, and small Jacobian-field mismatches compound exponentially over the rollout.

This is **option (B)** from the diagnostic split — sensitivity / Jacobian-field mismatch — not **option (A)** (coverage). Coverage is unlikely to be the primary problem because:
- The held-out block is the *middle* one, bracketed in time by 4 training blocks.
- The RNN itself generalizes fine, implying training trajectories visit roughly the same (h, u) regions as the test block.

Option (C) — RNN itself not generalizing — was ruled out directly.

## What's Already in Place (working-backup branch)

These are correctly implemented and should be kept:

1. **Learnable damping** in `forward_sindy` (model.py ~L680): `h_{t+1} = σ(α)·h_t + Θ·ξ`, per (module, ensemble, participant, experiment), initialized at σ(5.0) ≈ 0.993. Provides stability on the linear self-loop; correctly addresses one specific failure mode.
2. **Multi-step shooting in Stage 2** (`_run_sindy_training`): full-trajectory autoregressive rollout, gradients through K steps.
3. **Initial-state noise** (spice_training.py L923–928): σ=0.1 added to `current_state` at the start of each rollout window.
4. **Jacobian contraction penalty** (~L976–996): `torch.relu(|J| - 1.0)²` penalizes Jacobian eigenvalues > 1.
5. **Ensemble + CI-based pruning**: correct defense against coefficient instability under data perturbation.
6. **Per-participant SINDy coefficients**: absorbs the RNN's participant-embedding role.

## Why Current Setup Doesn't Close the Gap

The current dithering setup adds noise only at the **window's initial state**; subsequent steps run on SINDy's own predictions. The target at every step is the **clean** RNN trajectory. This is a "trajectory recovery from initial perturbation" test — a stability check — not a Jacobian-field-matching constraint. It does not densely sample the neighborhood of the trajectory or constrain ∂SINDy/∂h ≈ ∂RNN/∂h locally at each step.

The damping + contraction penalty stack pushes the discovered map toward |J| ≤ 1. Useful for stability, but if SINDy fails because its Jacobian field differs from the RNN's (not because |J| > 1 specifically), these regularizers can over-flatten the dynamics rather than aligning them with the RNN.

## Action Plan

### Step 1 (5 min): Sanity-check the RNN generalization claim

Confirm RNN-only train vs test NLL/MSE on the held-out block. We've assumed this is fine. If not, branch (C) is back on the table and the rest of the plan changes.

### Step 2 (15 min): Run the peel-apart plot

For 5–10 held-out test sessions, plot `||h_t^RNN − h_t^SINDy||` over t (per state variable, per session).

- **If trajectories track for ~5–10 steps then peel apart exponentially**: confirms sensitivity / Jacobian-field mismatch (option B). Proceed with Step 4.
- **If trajectories disagree from step 1**: suggests coverage problem (option A). Step 3 becomes more important.
- **If trajectories track perfectly but logits differ**: the issue is in the readout, not the dynamics — different problem.

### Step 3 (30 min): Run the existing diagnostics

From `spice.utils.diagnostics.SpiceDiagnostics`:

```python
diag = SpiceDiagnostics(estimator, dataset)
print(diag.polynomial_adequacy())     # Per-module R² of polynomial fit
print(diag.module_swap_test())        # Which module's RNN→SINDy swap hurts behavioral loss most
print(diag.embedding_dependence())    # emb_frac per module (should be irrelevant given per-participant coefs, but verify)
print(diag.state_range())             # h_t distribution per module
print(diag.residual_structure())      # What polynomial misses
print(diag.summary())                 # Combined table
```

Also: print fitted γ values (`model.sindy_damping_raw` after sigmoid) across participants/modules. If they cluster near 0.993 (init), damping isn't doing much. If they've moved meaningfully (γ < 0.9 for many participants), damping may be over-flattening — possibly part of the problem rather than the solution.

Use the swap test to identify *which* module is most responsible for the held-out gap. If one module dominates, focus subsequent experiments on it.

### Step 4 (~1 hour): Implement per-step input dithering

The main intervention. Replace initial-state-only noise with fresh per-step noise, and re-evaluate the RNN at the perturbed state so both models are compared at the same input.

In `_run_sindy_training`, replace the current rollout body (around L940–1000) with:

```python
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

    # Fresh noise at every step (not just window start)
    sigma_step = 0.02  # tune: ~2% of typical |h| scale, see state_range diagnostic
    perturbed_state = {
        s: current_state[s] + sigma_step * torch.randn_like(current_state[s])
        for s in current_state
    }

    # SINDy from perturbed input
    model.use_sindy = True
    _, next_state_sindy = model(xs_step, perturbed_state)

    # RNN re-evaluated at the SAME perturbed input (this is the key change)
    model.use_sindy = False
    with torch.no_grad():
        # Detach to avoid building gradient graph for RNN forward
        perturbed_detached = {s: v.detach() for s, v in perturbed_state.items()}
        _, next_state_rnn = model(xs_step, perturbed_detached)
    model.use_sindy = True

    # Loss: compare SINDy and RNN at the same perturbed point
    step_loss = torch.tensor(0.0, device=model.device)
    for s_key in model.spice_config.states_in_logit:
        pred = next_state_sindy[s_key]
        target = next_state_rnn[s_key]
        mask = valid.view(1, 1, -1, 1).expand_as(pred)
        diff = (pred - target) ** 2
        step_loss = step_loss + (diff * mask).sum() / mask.sum().clamp(min=1)

    total_loss = total_loss + step_loss
    n_valid_steps += 1

    # Feed clean SINDy prediction forward (do NOT accumulate noise into rollout)
    current_state = next_state_sindy
```

**Implementation notes:**

- Disable or reduce the existing initial-state noise (L923–928) when running this — the two interventions overlap conceptually and stacking them confounds attribution. A clean comparison is: (a) current setup, (b) per-step dithering with clean-input rollout and per-step RNN re-evaluation. Pick one to flip first.
- Keep the Jacobian contraction penalty for now; per-step dithering is independent of it.
- Per-step dithering doubles model forward calls per rollout step (one SINDy with grad + one RNN under no_grad). On A100/H100 this is fine; on smaller GPUs you may want to reduce K or batch size.
- Make sure `model.use_sindy` toggling mid-rollout works cleanly with how `call_module` is structured. Test on a single batch with verbose=True first.

### Step 5: Choose σ for the dithering

Two principles:

- **Much smaller than the typical scale of h.** From `diag.state_range()`, look at the std of each state. Start with σ_step ≈ 0.02 × that std. Probably σ_step ∈ [0.005, 0.05] for most modules.
- **Optionally per-module σ.** If `state_range` shows very different scales across modules (e.g., value_reward has std 0.3 and value_choice has std 0.05), use proportional σ per module rather than a global scalar.

A practical recipe: compute σ once before Stage 2 from the RNN-rollout state buffers:

```python
sigma_per_state = {
    s: 0.02 * state_trajectories[s].std().item()
    for s in state_trajectories
}
```

Then in the rollout loop:

```python
perturbed_state = {
    s: current_state[s] + sigma_per_state[s] * torch.randn_like(current_state[s])
    for s in current_state
}
```

### Step 6: Measure

For each experiment, log:
- Stage 2 final train rollout MSE (RNN vs SINDy on training blocks)
- Stage 2 final test rollout MSE (RNN vs SINDy on held-out block)
- Behavioral NLL on train and test (cross-entropy of SINDy logits against choices)
- Histogram of `||h_t^RNN − h_t^SINDy||` over t on held-out trajectories (the peel-apart plot, quantified)

If per-step dithering closes a meaningful fraction of the gap → confirmed Jacobian-field mismatch was the dominant mechanism. Iterate on σ and consider extensions in Step 7.

If it doesn't move the needle → mechanism is something else. Move to Step 8.

### Step 7 (if Step 4–6 partially helps): Extensions

- **Mini-rollouts from every t**: instead of one long rollout from t=0, start short K-step rollouts from every h_t along the RNN trajectory (sliding-window shooting). This is the full multiple-shooting approach. Existing window iteration in `_run_sindy_training` already does this with overlapping windows; you may already have most of the machinery.
- **Tune the contraction penalty downward**: if damping + contraction over-flatten, reduce `contraction_weight` and see if dithering alone is sufficient.
- **Print fitted γ distribution** — if many participants ended up with γ ≪ 1 (heavy damping), consider initializing the damping at σ(7.0) ≈ 0.999 and using a higher prior toward 1, so it only activates when really needed.

### Step 8 (if Step 4–6 does nothing): Next suspects

If per-step dithering doesn't help, the bottleneck is likely structural — the polynomial library at the current degree cannot represent the RNN's dynamics regardless of how the fit is constrained. Diagnostics to run:

- `diag.polynomial_adequacy()` should reveal this: low R² for some module means even an unrestricted polynomial fit can't capture that module's RNN dynamics.
- `diag.residual_structure()` shows what's missing — high correlation between residuals and a specific input feature suggests adding interaction terms or increasing degree.
- Try `sindy_library_polynomial_degree=3` for the worst-performing module (per swap test).

### Step 9 (parallel validation): Within-training held-out check

Independent of the dithering experiment, run this validation: fit SINDy on 3 of the 4 training blocks, evaluate on the 4th training block. If this within-training held-out also fails badly, the issue is structural and not specific to block 3. This gives a faster iteration loop than re-running with the actual held-out block each time.

## Key Open Questions

- Does the peel-apart plot show exponential divergence after initial agreement (sensitivity) or immediate disagreement (coverage)?
- What's the fitted γ distribution across participants/modules? Is damping doing real work or just sitting at init?
- Per `module_swap_test`, is the gap dominated by one module or spread across all?
- Does within-training-block holdout (Step 9) show the same gap?

## Things to Watch Out For

- **Don't stack the existing initial-state noise + per-step dithering without measuring each in isolation first.** They're conceptually overlapping interventions and confound attribution.
- **σ tuning matters.** Too small (< 1e-3) and floating point swamps the signal. Too large (> 0.1 × state scale) and you're asking SINDy to match the RNN in regions the RNN itself wasn't trained on.
- **The `model.use_sindy` toggling inside the rollout** is the most error-prone part of the implementation. Verify with a single forward pass on a small batch before scaling up.
- **No_grad for the RNN re-evaluation is essential** — otherwise you're building a gradient graph that explodes in memory.
- **Keep gradient flow correct**: `next_state_sindy` (with grad) feeds forward into `current_state` for the next step; the perturbation and the RNN re-eval both detach.

## Conceptual Background (for reference)

The general principle: fitting recurrences from data has three structural difficulties.

1. **Identifiability**: many different equations produce similar short trajectories. Sparsity priors help but don't eliminate the null space.
2. **Ill-conditioning of the polynomial regression**: library columns are correlated (h·u, h², u²...). Ensemble + CI pruning is the right defense.
3. **Sensitive dependence**: discrete maps with small per-step errors can produce wildly different long trajectories. One-step or even K-step rollout loss on a *single* trajectory doesn't constrain the Jacobian field.

Per-step input dithering with RNN re-evaluation directly addresses (3) by making the fitting loss approximate a Jacobian-matching loss via Monte Carlo. Equivalent to penalizing `||∂(RNN)/∂h − ∂(SINDy)/∂h||²` analytically but cheaper and easier to implement.

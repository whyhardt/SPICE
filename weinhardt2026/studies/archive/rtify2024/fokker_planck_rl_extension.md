# Idea: Fokker-Planck value tracking for trial-by-trial RL models

Status: **not implemented** -- design sketch only, written up during the `rtify2024`
Fokker-Planck DDM work so the idea isn't lost. Revisit before starting.

## Context

`spice_rtify2024.py`'s `SpiceDDM` tracks the full probability *density* over evidence
(`evidence_pdf`), evolved via the Fokker-Planck / Kolmogorov forward equation, instead of a
single deterministic evidence trajectory. That was necessary because the DDM's evidence
accumulates through many *unobserved* noisy micro-steps within a trial -- only the final
(RT, choice) is ever observed -- and a deterministic trajectory can't represent that hidden
per-trial noise ("frailty bias": fitting a shared hazard to a population with unobserved
individual-level heterogeneity systematically biases the recovered parameters). See that file's
class docstring for the full chain of reasoning.

That reformulation also inverted which part of the model SINDy is asked to discover:

- **Classic across-trial RL models** (Rescorla-Wagner etc.): the *value* state's update
  equation is what's unknown and interesting (`value[t+1] = f(value[t], reward[t])`); the
  value-to-choice mapping (softmax) is fixed and uninteresting.
- **This DDM model**: the accumulation dynamics are fixed, known theory (that's what makes it a
  DDM); what's unknown is how task variables modulate the *parameters* feeding that fixed
  process (`drift = f(stimulus)`, `threshold = g(time_elapsed)`, both genuine learned
  submodules -- see `library_setup` in `spice_rtify2024.py`).

## The idea

Combine both principles for a trial-by-trial RL model:
- **Deterministic dynamics for the parameters** (learning rate, forgetting, choice bias) --
  submodules, same as `drift`/`threshold_raw` here.
- **Noisy density tracking for the value state itself** -- a `value_pdf`, evolved trial-by-trial
  (not within-trial micro-steps; each RL trial's reward/action IS observed, so this operates at
  the outer trial axis, not an inner discretization axis the way DDM steps do).

### Why this could matter (when it does)

Plain deterministic value tracking (as every current SPICE RL precoded model does) is *correct*
as long as the only stochasticity in the generative process is at the choice stage (softmax
sampling an action from a deterministic value). There's no frailty-bias problem there, because
nothing is hidden -- value updates deterministically from what's actually observed each trial.

Density tracking would only earn its keep if the assumed generative process has genuine
*process noise on value itself* -- value drifts/degrades stochastically between trials in a way
not fully explained by the observed reward sequence (sometimes called "noisy RL"; related to
Kalman-filter/particle-filter Q-learning, representational-noise models in computational
psychiatry). If that's the assumption, deterministic value tracking has the same structural
blind spot the old DDM softmax hazard had, for the same reason.

### Mechanism -- reusing exactly the machinery built for `spice_rtify2024.py`

Per trial, all of the following are either an affine resample (`_shift_pdf`, generalized from a
pure shift to a scale+shift, exactly as sketched for `leak` -- see chat notes / commit history
around the `leak` discussion) or a fixed convolution (`diffusion_kernel`-style `conv1d`):

- **Learning rate (Kalman-style contraction toward reward)**: the delta rule
  `value_new = value + alpha*(reward - value) = (1-alpha)*value + alpha*reward` IS an affine
  map -- contract by `(1-alpha)` toward `reward` instead of toward `0` (as `leak` does toward
  `0`). Same `grid_sample` resampling, same Jacobian correction `1/(1-alpha)` for mass
  conservation under the contraction.
- **Forgetting**: a second, independent affine contraction toward some baseline (0, or a
  learned prior mean). Composable with the learning-rate step into a single combined affine map
  (and thus a single `grid_sample` call) since composition of affine maps is itself affine --
  the two can still be *produced* by separate, separately-interpretable submodules even if
  mathematically combined before the one resampling call.
- **Choice bias / persistence**: a pure additive shift, no contraction -- like `drift` shifting
  `evidence_pdf`.
- **Trial-by-trial value noise**: a fixed `conv1d` spread with some `sigma_value`, exactly like
  `diffusion_kernel` for evidence.

### The circular part (the sharp bit)

Read the density's own variance each trial (`Var[value] = E[value^2] - E[value]^2`, computed
the same way `mean_evidence()` reads the mean) and feed it *forward* into that trial's learning
rate, rather than treating learning rate as a free-floating submodule output only:

```
alpha_t = Var_t / (Var_t + observation_noise_var)
```

This is exactly the Kalman gain equation. It self-regulates: the contraction step shrinks
`Var_t` by `(1-alpha_t)`, the diffusion step reinflates it by the process noise, and the two
settle into a steady-state uncertainty level (the same "absorb then spread" alternation
`evidence_pdf` already does per DDM step, just at trial-scale, with variance now driving
dynamics instead of only being read for reporting).

## Open design questions (unresolved -- decide before implementing)

1. **Pure Kalman-derived `alpha_t`, or Kalman-derived + a SINDy-discoverable deviation term?**
   The latter is arguably the more interesting scientific question -- "does behavior match
   ideal Bayesian updating, or diverge from it in an interpretable way" -- rather than either
   pure hand-derived Kalman or a pure free-floating learned rate.
2. **What drives the trial's choice** -- just `E[value]` through an ordinary softmax (simplest,
   matches every existing precoded RL model), or does `Var[value]` also modulate choice (e.g.
   uncertainty-driven exploration/decision noise)? Note RL trials don't "end" the way a DDM
   step-sequence does (no boundary-absorption analog needed) -- every trial gets a normal
   categorical choice, so this half is simpler than `spice_rtify2024.py`'s `forward()` loop.

## Where to start if/when this gets built

- New precoded model or new study variant (not a modification of `spice_rtify2024.py` --
  different axis semantics: trial-scale updates, not within-trial DDM steps).
- Reuse `_shift_pdf`'s `grid_sample` pattern (generalize from pure shift to scale+shift) and
  `diffusion_kernel`'s `conv1d` pattern directly from `spice_rtify2024.py`.

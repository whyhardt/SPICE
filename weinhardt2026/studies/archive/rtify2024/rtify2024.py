"""
Synthetic DDM study: pure response-time + decision dynamics, no belief updating.

Ground truth: a classic two-boundary drift-diffusion process (spice/precoded/ddm.py's
`simulate_ddm`, adapted). One general `simulate_ddm()` covers the plain DDM plus three
independently-toggleable generalizations (each off by default): a flipping/reversing
stimulus, a leaky (vs. perfect) accumulator, and a collapsing decision boundary. Fitted
model: a single 'drift' RNN submodule that receives the trial's (stimulus, elapsed_time)
repeated across all `max_steps` within-trial timesteps in ONE `call_module` call -- the
underlying EnsembleRNNModule already loops over within-trial timesteps internally
(compiled via torch.compile), so the resulting `self.state['drift']` is a genuinely
time-varying trajectory over the decision.

`drift` *is* the evidence -- there's no separate external integration step. The
module's own state naturally accumulates via its residual update (h[t+1] = h[t] +
dt*n[t], now that `dt` is threaded through `setup_module`/`EnsembleRNNModule`/the SINDy
fit so per-step increments read as per-unit-time rates, not tiny per-step deltas that
shrink as `max_steps` grows). "Leak" isn't a separate hand-designed parameter: it's
whatever self-coefficient SINDy discovers on `drift[t]` in `drift[t+1] = a*drift[t] +
b*stimulus[t] + c*time_elapsed[t]` -- a<1 is decay, a=1 is a perfect integrator, both
expressible by the same equation instead of two different code paths.

That evidence trajectory is turned into a discrete-time two-boundary hazard/survival
distribution (RTify's example2.ipynb idea) instead of ddm.py's noisy SDE simulation +
custom-autograd threshold detection:

    p_stop_up[w]   = sigmoid(drift[w] - threshold)
    p_stop_down[w] = sigmoid(-drift[w] - threshold)
    p_stop[w]      = p_stop_up[w] + p_stop_down[w] - p_stop_up[w] * p_stop_down[w]
    p_decision_*[w] = p_stop_*[w] * prod_{k<w}(1 - p_stop[k])

`p_decision_up` + `p_decision_down` is a proper joint distribution over (choice, RT bin),
so choice and RT are fit by a single closed-form negative log-likelihood term -- fully
differentiable, no custom autograd Function, no distribution-matching loss.

The model itself has no non-decision-time handling: the decision process runs over
the full `max_steps` range from t=0. True RTs (see simulate_ddm) include a real ndt
offset, so a handful of trials with RT < ndt land in bins the model can't reach --
an accepted tradeoff for a model with no input-length-dependent special-casing.

Module layout (matching other studies' convention):
    spice_rtify2024.py     -- CONFIG, DDMRNN, make_ddm_loss (the model)
    benchmark_rtify2024.py -- simulate_ddm (ground truth), get_dataset
    analysis_rtify2024.py  -- decode_choice_rt, estimate_non_decision_time,
                              _sanitize_predictions, evaluate, print_spice_models, plot_summary
    rtify2024.py            -- this file: the pipeline
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator, SpiceDataset

from weinhardt2026.studies.archive.rtify2024.spice_rtify2024 import CONFIG, DDMRNN, make_ddm_loss
from weinhardt2026.studies.archive.rtify2024.benchmark_rtify2024 import get_dataset
from weinhardt2026.studies.archive.rtify2024.analysis_rtify2024 import (
    decode_choice_rt, estimate_non_decision_time, evaluate, print_spice_models, plot_summary,
)


path_spice = 'weinhardt2026/studies/archive/rtify2024/params/rtify2024.pkl'

# simulation settings
max_steps = 100
t_max = 5.0
dt = t_max / max_steps

n_participants = 3
n_trials = 1000

# spice training settings
epochs=1000  # enables stage 1 training; if epochs=0 -> load existing model from path_spice
sindy_refit=True  # enables stage 2 training; if epochs=0 and sindy_refit=False -> skip estimator.fit() and go directly to analysis


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Primary/baseline tests: flip disabled (flip_time_range=None -> constant-sign
# drift), no leak, no collapsing bound. Toggle any of these back on independently:
#   flip_time_range=(0.3, 0.7)  -- reversal paradigm (then drift_rate is a
#                                  magnitude, not a sign, and should stay low
#                                  relative to threshold/flip_time_range, since
#                                  strong drift resolves before the earliest
#                                  possible flip and the flip stops mattering)
#   leak=1.0                    -- leaky (vs. perfect) integration
#   collapsing_bound_rate=0.3   -- urgency-like shrinking boundary
drift_rates = torch.rand(n_participants)

dataset_train, dataset_test, info_dataset = get_dataset(
    drift_rates=drift_rates,
    collapsing_bound_rate=0.,
    leak=0.,
    flip_time_range=None,
    
    n_trials=n_trials,
    max_steps=max_steps,
    t_max=t_max,
    
    device=device,
)

print(f"Estimated non-decision time: {info_dataset['non_decision_time']:.3f}s")

estimator = SpiceEstimator(
    spice_class=DDMRNN,
    spice_config=CONFIG,
    kwargs_spice_class={'dt': dt, 'non_decision_time': 0.2},
    n_reward_features=0,
    
    n_actions=2,
    n_participants=n_participants,

    loss_fn=make_ddm_loss(),
    
    sindy_weight=1e-2,
    sindy_alpha=1e-4,
    sindy_threshold_pruning=0.05,
    sindy_ensemble_pruning=0.7,
    sindy_refit=sindy_refit,
    
    epochs=epochs,
    warmup_steps=500,
    
    device=device,
    verbose=True,
    save_path_spice=path_spice,
    compiled_forward=True,
)

if estimator.epochs == 0:
    estimator.load_spice(path_spice)
if estimator.epochs > 0 or estimator.sindy_refit:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)

print("\n--- Train ---")
print(evaluate(estimator, dataset_train, dt, max_steps))
print("\n--- Test ---")
print(evaluate(estimator, dataset_test, dt, max_steps))

print_spice_models(estimator, participant_ids=(0, 2))

plot_summary(estimator, dataset_test, dt, t_max, max_steps, participant_ids=(0, 2), true_threshold=1.0, output_path='weinhardt2026/studies/archive/rtify2024/results')

print("Learnable initial value for threshold:")
print(torch.nn.functional.softplus(estimator.model.learnable_initial_values['threshold_raw'].mean(dim=0)).detach().cpu().numpy())

print("Learnable initial value for drift:")
print(estimator.model.learnable_initial_values['drift'].mean(dim=0).detach().cpu().numpy())

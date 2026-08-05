"""
Synthetic DDM study -- same ground truth and per-step cross-entropy training as rtify2024_e,
but a structurally different model: the evidence state is an actual (discretized) probability
DENSITY evolved via the Fokker-Planck / Kolmogorov forward equation, not a deterministic
scalar trajectory shared by every trial.

rtify2024_e's Gaussian-CDF hazard was the mathematically correct one-step boundary-crossing
formula, but it was evaluated against a deterministic mean evidence trajectory identical for
every trial with the same drift -- verified directly (see spice_rtify2024_e.py's DDMRNN
docstring) that the TRUE ground-truth parameters scored a *worse* log-likelihood under that
model than the (wrong) values training converged to: the model couldn't represent the true
process at all, a "frailty bias" (unobserved per-trial heterogeneity) problem, not an
optimization bug. Tracking the population density directly sidesteps needing per-trial noise
realizations (and the non-differentiable sampling that would require) while still being exact.
See spice_rtify2024.py's DDMRNN docstring for the full mechanism.

Module layout (matching other studies' convention):
    spice_rtify2024.py     -- CONFIG, DDMRNN (Fokker-Planck density evolution)
    benchmark_rtify2024.py -- simulate_ddm (ground truth), get_dataset
    analysis_rtify2024.py  -- decode_choice_rt, evaluate, print_spice_models,
                                 plot_summary, plot_participant_fit
    rtify2024.py            -- this file: the pipeline
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')

import torch

from spice import SpiceEstimator

from weinhardt2026.studies.archive.rtify2024.spice_rtify2024 import CONFIG, DDMRNN
from weinhardt2026.studies.archive.rtify2024.benchmark_rtify2024 import get_dataset
from weinhardt2026.studies.archive.rtify2024.analysis_rtify2024 import (
    evaluate, print_spice_models, plot_summary, plot_participant_fit
)


path_spice = 'weinhardt2026/studies/archive/rtify2024/params/rtify2024.pkl'
path_results = 'weinhardt2026/studies/archive/rtify2024/results'

# simulation settings
max_steps = 100
t_max = 5.0
dt = t_max / max_steps
non_decision_time = 0.2  # known ground truth for this synthetic identity check
diffusion_rate = 1.0     # must match benchmark_rtify2024.get_dataset's hardcoded diffusion_rate=1.0

n_participants = 3
n_trials = 10000

# spice training settings
epochs = 0  # 0 -> load existing model from path_spice instead of fitting
sindy_refit = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

drift_rates = [0.5]  # , 0.75, 1.0]

dataset_train, dataset_test = get_dataset(
    drift_rates=drift_rates,
    collapsing_bound_rate=0.,
    leak=0.,
    flip_time_range=None,

    n_trials=n_trials,
    max_steps=max_steps,
    t_max=t_max,

    device=device,
)

estimator = SpiceEstimator(
    spice_class=DDMRNN,
    spice_config=CONFIG,
    kwargs_spice_class={'dt': dt, 'diffusion_rate': diffusion_rate},
    n_reward_features=0,

    n_actions=3,  # [no_decision, up, down]
    n_participants=n_participants,

    # loss_fn left at the estimator default (cross_entropy_loss) -- no bespoke ddm_loss needed.

    ensemble_size=1,  # default: 10; only useful with SINDy fitting

    sindy_weight=0,
    sindy_alpha=1e-4,
    sindy_threshold_pruning=0.01,
    sindy_ensemble_pruning=0.5,
    sindy_refit=sindy_refit,

    epochs=epochs,
    warmup_steps=500,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
    compiled_forward=True,  # safe again now that DDMRNN.forward() is fully vectorized (conv1d +
                             # grid_sample, no more data-dependent .unique() participant loop)
)

if estimator.epochs == 0:
    estimator.load_spice(path_spice)
if estimator.epochs > 0 or estimator.sindy_refit:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)

print("\n--- Train ---")
print(evaluate(estimator, dataset_train, dt, max_steps, non_decision_time=non_decision_time))
print("\n--- Test ---")
print(evaluate(estimator, dataset_test, dt, max_steps, non_decision_time=non_decision_time))

print_spice_models(estimator, participant_ids=tuple(range(n_participants)))

plot_summary(
    estimator, dataset_test, dt, t_max, max_steps,
    participant_ids=tuple(range(n_participants)),
    true_threshold=1.0,
    non_decision_time=non_decision_time,
    output_path=os.path.join(path_results, 'results.png'),
)
for pid in range(n_participants):
    plot_participant_fit(estimator, dataset_test, dt, t_max, 
                         participant_id=pid, 
                         non_decision_time=non_decision_time, 
                         use_sindy=False, 
                         output_path=os.path.join(path_results, 'likelihood.png'),
                         )

print("\n")

print("Learnable initial value for threshold:")
print((estimator.model.grid_half_width * torch.sigmoid(estimator.model.learnable_initial_values['threshold_raw'].mean(dim=0))).detach().cpu().numpy())

print("Learnable initial value for drift:")
print(estimator.model.learnable_initial_values['drift'].mean(dim=0).detach().cpu().numpy())

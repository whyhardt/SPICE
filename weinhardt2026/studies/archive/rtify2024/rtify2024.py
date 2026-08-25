"""
Synthetic DDM study -- same ground truth and per-step cross-entropy training as rtify2024_e,
but a structurally different model: the evidence state is an actual (discretized) probability
DENSITY evolved via the Fokker-Planck / Kolmogorov forward equation, not a deterministic
scalar trajectory shared by every trial.

rtify2024_e's Gaussian-CDF hazard was the mathematically correct one-step boundary-crossing
formula, but it was evaluated against a deterministic mean evidence trajectory identical for
every trial with the same drift -- verified directly (see spice_rtify2024_e.py's SpiceDDM
docstring) that the TRUE ground-truth parameters scored a *worse* log-likelihood under that
model than the (wrong) values training converged to: the model couldn't represent the true
process at all, a "frailty bias" (unobserved per-trial heterogeneity) problem, not an
optimization bug. Tracking the population density directly sidesteps needing per-trial noise
realizations (and the non-differentiable sampling that would require) while still being exact.
See spice_rtify2024.py's SpiceDDM docstring for the full mechanism.

Module layout (matching other studies' convention):
    spice_rtify2024.py     -- CONFIG, SpiceDDM (Fokker-Planck density evolution)
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

from weinhardt2026.studies.archive.rtify2024.spice_rtify2024 import CONFIG, SpiceDDM
from weinhardt2026.studies.archive.rtify2024.benchmark_rtify2024 import get_dataset, DDM_PARAMETERS
from weinhardt2026.studies.archive.rtify2024.analysis_rtify2024 import (
    evaluate, print_spice_models, plot_summary, plot_participant_fit, plot_dataset_variables
)


path_spice = 'weinhardt2026/studies/archive/rtify2024/params/rtify2024.pkl'
path_results = 'weinhardt2026/studies/archive/rtify2024/results'

simulation_analysis_only = False

# SPICE model/fitting parameters
epochs = 1000
sindy_refit = False
sindy_weight = 0
grid_points = 241
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

kwargs_ddm = dict(
    drift_rates=[1.0],
    collapsing_bound_rate=0.,
    leak=0.,
    flip_time_range=None,
    drift_update_kwargs={},
    threshold=1,
    non_decision_time = 0.2, # known ground truth for this synthetic identity check
    diffusion_rate = 1.0,     # must match benchmark_rtify2024.get_dataset's hardcoded diffusion_rate=1.0
    n_trials=1000,
    max_steps=100,
    t_max=5.0,
    device=device,
)

dt = kwargs_ddm['t_max'] / kwargs_ddm['max_steps']


dataset_train, dataset_test = get_dataset(**kwargs_ddm)

if simulation_analysis_only:
    plot_dataset_variables(
        dataset_train, dt, kwargs_ddm['t_max'], DDM_PARAMETERS,
        non_decision_time=kwargs_ddm['non_decision_time'],
        output_path=os.path.join(path_results, 'dataset.png'),
    )
else:
    # Inverse-frequency category weights for [no_decision, up, down], computed from the actual
    # training data with the same NaN-masking convention _run_batch_training applies at loss
    # time (mask = ~isnan(xs[..., :n_actions])) -- so post-decision padding rows don't inflate
    # the no_decision count. no_decision dominates by raw count (nearly every trial is still
    # undecided at every early step), which lets the model get up/down (the rare, actually
    # decision-informative classes) systematically wrong at little loss cost.
    # _mask = ~torch.isnan(dataset_train.xs[..., :3].sum(dim=-1))
    # _class_counts = dataset_train.ys[_mask].sum(dim=0)  # (3,) -- [no_decision, up, down]
    # _category_weight = (1. / _class_counts.clamp_min(1.))
    # _category_weight = _category_weight / _category_weight.sum() * 3.  # normalize to mean 1
    
    estimator = SpiceEstimator(
        spice_class=SpiceDDM,
        spice_config=CONFIG,
        kwargs_spice_class={'dt': dt, 'diffusion_rate': kwargs_ddm['diffusion_rate'], 'grid_points': grid_points,},
        n_reward_features=0,

        n_actions=dataset_train.n_actions,  # [no_decision, up, down]
        n_participants=dataset_train.n_participants,

        ensemble_size=10,  # default: 10; only useful with SINDy fitting

        sindy_weight=sindy_weight,
        sindy_lambda_loading=1e-3,
        sindy_threshold_pruning=0.05,
        sindy_ensemble_pruning=0.7,
        sindy_refit=sindy_refit,

        sindy_shooting_steps=10,

        epochs=epochs,
        warmup_steps=epochs//2,
        loss_fn_kwargs={'label_smoothing': 0},#, 'weight': _category_weight},

        device=device,
        verbose=True,
        save_path_spice=path_spice,
        compiled_forward=True,
    )

    if estimator.epochs == 0:
        estimator.load_spice(path_spice)
    if estimator.epochs > 0 or estimator.sindy_refit:
        estimator.fit(dataset_train.xs, dataset_train.ys)#, dataset_test.xs, dataset_test.ys)
        estimator.save_spice(path_spice)

    print("\n--- Train ---")
    print(evaluate(estimator, dataset_train, dt, kwargs_ddm['max_steps'], non_decision_time=kwargs_ddm['non_decision_time']))
    print("\n--- Test ---")
    print(evaluate(estimator, dataset_test, dt, kwargs_ddm['max_steps'], non_decision_time=kwargs_ddm['non_decision_time']))

    print_spice_models(estimator, participant_ids=tuple(range(dataset_train.n_participants)))

    plot_summary(
        estimator, dataset_test, dt, kwargs_ddm['t_max'], kwargs_ddm['max_steps'],
        participant_ids=tuple(range(dataset_train.n_participants)),
        true_threshold=1.0,
        non_decision_time=kwargs_ddm['non_decision_time'],
        output_path=os.path.join(path_results, 'results.png'),
    )
    # Same generative config as dataset_train/dataset_test, but a much larger n_trials --
    # gives plot_participant_fit a low-noise empirical reference density, so histogram sampling
    # noise in dataset_test (n_trials=1000) can be told apart from genuine model misfit.
    kwargs_ddm['n_trials'] = 100000
    dataset_empirical, _ = get_dataset(**kwargs_ddm)
    
    for pid in range(min(dataset_train.n_participants, 3)):
        plot_participant_fit(estimator, dataset_test, dt, kwargs_ddm['t_max'],
                            participant_id=pid,
                            non_decision_time=kwargs_ddm['non_decision_time'],
                            output_path=os.path.join(path_results, 'likelihood.png'),
                            dataset_empirical=dataset_empirical,
                            )

    print("\n")

    print("Learnable initial value for threshold:")
    print((estimator.model.grid_half_width * torch.sigmoid(estimator.model.learnable_initial_values['threshold_raw'].mean(dim=0))).detach().cpu().numpy())

    print("Learnable initial value for drift:")
    print(estimator.model.learnable_initial_values['drift'].mean(dim=0).detach().cpu().numpy())

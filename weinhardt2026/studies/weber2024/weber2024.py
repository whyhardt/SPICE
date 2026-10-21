import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch

from spice import SpiceEstimator, split_data_along_blockdim

from spice_weber2024 import SpiceModel, CONFIG
from spice_weber2024 import clamped_angular_mse
from benchmarking_weber2024 import get_dataset, ChangePointModel

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation_mse


train_spice = False
train_gru = False
train_benchmark = False


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/weber2024/data/weber2024.csv'
test_blocks = (3, 5, 9, 12)

dataset, _ = get_dataset(path_data=path_data)

dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/weber2024/params/spice_weber2024_continuous.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=2,                                    # next shield position: sin, cos
    n_items=1,                                      # belief as one angle
    n_participants=dataset_train.n_participants,
    n_experiments=1,                                # conditions are module inputs: one coefficient set per participant
    n_reward_features=dataset_train.n_reward_features,

    loss_fn=clamped_angular_mse,

    # the penalty defaults are tuned for cross-entropy; against this MSE they switch modules off
    gate_penalty=0.,
    feature_penalty=0.,
    
    epochs=1000,
    warmup_steps=500,
    ensemble_size=1,

    device=device,
    verbose=True,
    compiled_forward=False,
)

if train_spice:
    estimator.fit(
        data=dataset_train.xs,
        targets=dataset_train.ys,
        data_test=dataset_test.xs if dataset_test is not None else None,
        target_test=dataset_test.ys if dataset_test is not None else None,
    )
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(best_checkpoint('weber2024', 'spice'))

# -------------------------------------------------------------------------------------------
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/weber2024/params/gru_weber2024_continuous.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

gru = GRUModel(
    n_actions=2,                                    # output: belief_sin, belief_cos
    additional_inputs=4,                            # laser_caught, volatility, stochasticity, trial_duration_frames (excl. trueMean)
    additional_inputs_scale=(1., 1., 1., 90.),      # trial_duration_frames (1-90 frames) -> (0, 1]
    n_reward_features=dataset_train.n_reward_features,
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    print("Training GRU...")
    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=clamped_angular_mse,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru = load_best('weber2024', 'gru')


# -------------------------------------------------------------------------------------------
# BENCHMARK: BAYESIAN CHANGEPOINT MODEL (Foucault et al., 2026)
# -------------------------------------------------------------------------------------------

path_benchmark = 'weinhardt2026/studies/weber2024/params/benchmark_weber2024.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

benchmark = ChangePointModel(
    n_participants=dataset_train.n_participants,
    h_step=2,               # 2° discretisation (180 grid points)
    variance_inference=True,    # infer which of two fitted noise levels a block has, instead of one fitted s.d.
)

if train_benchmark:
    optimizer = torch.optim.Adam(benchmark.parameters(), lr=0.01)
    print("Training Bayesian changepoint benchmark...")
    benchmark = training(
        model=benchmark,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=clamped_angular_mse,
        scheduler=True,
    )
    torch.save(benchmark.state_dict(), path_benchmark)
    print("Trained benchmark parameters saved to " + path_benchmark)
else:
    benchmark = load_best('weber2024', 'benchmark')

benchmark.to(torch.device('cpu'))

# -------------------------------------------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------------------------------------------

analysis_model_evaluation_mse(
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    spice_selection='best-on-train',
    verbose=True,
    loss_fn=clamped_angular_mse,
)

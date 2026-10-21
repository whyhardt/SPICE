import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from spice import SpiceEstimator, split_data_along_blockdim
from spice_bruckner2025 import SpiceModel, CONFIG, mse_loss
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import (
    get_dataset, RationalResourceModel,
)
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation_mse


train_spice = False
train_gru = False
train_benchmark = False


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/bruckner2025/data/bruckner2025.csv'
test_blocks = (2, 4)   # blocks 1-2 no-push, 3-4 push: hold out one block per condition

dataset, _ = get_dataset(path_data=path_data)

dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/bruckner2025/params/spice_bruckner2025.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,           # 1 (continuous)
    n_items=1,
    n_participants=dataset_train.n_participants,
    n_experiments=dataset_train.n_experiments,
    n_reward_features=dataset_train.n_reward_features,

    loss_fn=mse_loss,

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
    estimator.load_spice(best_checkpoint('bruckner2025', 'spice'))


# -------------------------------------------------------------------------------------------
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/bruckner2025/params/gru_bruckner2025.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

gru = GRUModel(
    n_actions=1,
    additional_inputs=6,  # all observable: z_next, catch, v_t, sigma, r_t, mu_t (excl. c_t)
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
        loss_fn=mse_loss,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru = load_best('bruckner2025', 'gru')


# -------------------------------------------------------------------------------------------
# BENCHMARK MODEL (Reduced Bayesian)
# -------------------------------------------------------------------------------------------

path_benchmark = 'weinhardt2026/studies/bruckner2025/params/benchmark_bruckner2025.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

benchmark = RationalResourceModel(n_participants=dataset_train.n_participants)

if train_benchmark:
    optimizer_bm = torch.optim.Adam(benchmark.parameters(), lr=0.01)
    print("Training benchmark model...")
    benchmark = training(
        model=benchmark,
        optimizer=optimizer_bm,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=mse_loss,
        scheduler=True,
    )
    torch.save(benchmark.state_dict(), path_benchmark)
    print("Trained benchmark parameters saved to " + path_benchmark)
else:
    benchmark = load_best('bruckner2025', 'benchmark')


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
)

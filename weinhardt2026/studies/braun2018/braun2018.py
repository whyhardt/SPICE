import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from spice import SpiceEstimator
from weinhardt2026.studies.braun2018.spice_braun2018 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.studies.braun2018.benchmarking_braun2018 import ExpectedValueControl, get_dataset
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation


train_spice = False
train_benchmark = False
train_gru = False


path_data = 'weinhardt2026/studies/braun2018/data/braun2018.csv'
path_spice = 'weinhardt2026/studies/braun2018/params/spice_braun2018.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
path_benchmark = 'weinhardt2026/studies/braun2018/params/benchmark_braun2018.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
path_gru = 'weinhardt2026/studies/braun2018/params/gru_braun2018.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

# Truncate outlier-length blocks (mean + 2*std)
n_trials_per_block = np.zeros(dataset_train.xs.shape[0])
for i, block in enumerate(dataset_train.xs[:, :, 0, 0]):
    n_trials_per_block[i] = block.shape[0] - block.isnan().sum()
cutoff = int(n_trials_per_block.mean() + 2 * n_trials_per_block.std())
dataset_train.xs = dataset_train.xs[:, :cutoff]
dataset_train.ys = dataset_train.ys[:, :cutoff]
if dataset_test is not None:
    dataset_test.xs = dataset_test.xs[:, :cutoff]
    dataset_test.ys = dataset_test.ys[:, :cutoff]

print(f"Truncated max sequence length to {cutoff}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    n_reward_features=0,

    epochs=0,
    warmup_steps=500,
    ensemble_size=10,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    if estimator.epochs == 0:
        estimator.load_spice(path_spice)
        estimator.model.preprocess_coefficients()
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(best_checkpoint('braun2018', 'spice'))
    estimator.model.preprocess_coefficients()

# -------------------------------------------------------------------------------------------
# EVC BENCHMARK MODEL (Shenhav et al., 2013)
# -------------------------------------------------------------------------------------------

benchmark = ExpectedValueControl(n_participants=dataset_train.n_participants)

if train_benchmark:
    optimizer = torch.optim.Adam(benchmark.parameters(), lr=0.01)
    benchmark = training(
        model=benchmark, 
        optimizer=optimizer,
        dataset_train=dataset_train, 
        dataset_test=dataset_test,
        epochs=1000, 
        device=torch.device('cpu'),
    )
    torch.save(benchmark.state_dict(), path_benchmark)
else:
    benchmark = load_best('braun2018', 'benchmark')

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

gru = GRUModel(dataset_train.n_actions, additional_inputs=4, n_reward_features=0)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    gru = training(
        model=gru, 
        optimizer=optimizer,
        dataset_train=dataset_train, 
        dataset_test=dataset_test,
        epochs=1000,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru = load_best('braun2018', 'gru')

# -------------------------------------------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------------------------------------------

estimator.eval()
benchmark.eval()
gru.eval()


print("\n--- Model evaluation (train: likelihood, AIC, BIC; test: likelihood) ---")
analysis_model_evaluation(
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    spice_selection='best-on-train',
    verbose=True,
)

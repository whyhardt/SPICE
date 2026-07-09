import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from spice import SpiceEstimator
from weinhardt2026.studies.braun2018.spice_braun2018 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.braun2018.benchmarking_braun2018 import ExpectedValueControl, get_dataset, generate_behavior
from weinhardt2026.studies.braun2018.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.utils.generation import generate_repeated


train_spice = True
train_benchmark = False
train_gru = False

N_REPEATS = 100

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/braun2018/data/braun2018.csv'
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

path_spice = 'weinhardt2026/studies/braun2018/params/spice_braun2018.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    n_reward_features=0,

    epochs=1000,
    warmup_steps=500,
    ensemble_size=1,
    sindy_weight=0.01,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(path_spice)

# -------------------------------------------------------------------------------------------
# EVC BENCHMARK MODEL (Shenhav et al., 2013)
# -------------------------------------------------------------------------------------------

path_benchmark = 'weinhardt2026/studies/braun2018/params/benchmark_braun2018.pkl'

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
    benchmark.load_state_dict(torch.load(path_benchmark, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/braun2018/params/gru_braun2018.pkl'

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
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
benchmark.eval()
gru.eval()

for pid in range(3):
    print(f"\nExample SPICE model (participant {pid}):")
    estimator.print_spice_model(participant_id=pid)

print("\n--- Model evaluation (train) ---")
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
))

print("\n--- Model evaluation (test) ---")
print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    output_dir='weinhardt2026/studies/braun2018/results',
))

# # -------------------------------------------------------------------------------------------
# # GENERATIVE BENCHMARKING
# # -------------------------------------------------------------------------------------------

# data_dir = 'weinhardt2026/studies/braun2018/data'
# output_dir = 'weinhardt2026/studies/braun2018/results'

# estimator.use_sindy(False)
# ds_spice_rnn = generate_repeated(
#     generate_behavior,
#     n_repeats=N_REPEATS,
#     dataset=dataset_train,
#     model=estimator,
# )

# estimator.use_sindy(True)
# ds_spice = generate_repeated(
#     generate_behavior,
#     n_repeats=N_REPEATS,
#     dataset=dataset_train,
#     model=estimator,
# )

# ds_benchmark = generate_repeated(
#     generate_behavior,
#     n_repeats=N_REPEATS,
#     dataset=dataset_train,
#     model=benchmark,
# )

# ds_gru = generate_repeated(
#     generate_behavior,
#     n_repeats=N_REPEATS,
#     dataset=dataset_train,
#     model=gru,
# )

# # -------------------------------------------------------------------------------------------
# # ANALYSIS: GENERATIVE BEHAVIOR
# # -------------------------------------------------------------------------------------------

# analysis_generative_behavior(
#     path_data_real=f'{data_dir}/braun2018.csv',
#     path_data_benchmark=ds_benchmark,
#     path_data_gru=ds_gru,
#     path_data_spice_rnn=ds_spice_rnn,
#     path_data_spice=ds_spice,
#     output_dir=output_dir,
# )

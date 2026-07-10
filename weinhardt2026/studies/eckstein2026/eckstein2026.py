import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator
from weinhardt2026.studies.eckstein2026.spice_eckstein2026 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import Castro2025Model, get_dataset, generate_behavior
from weinhardt2026.studies.eckstein2026.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions
from weinhardt2026.analysis.analysis_coefficients_individuals import analysis_coefficients_individuals
from weinhardt2026.utils.generation import generate_repeated


train_spice = True
train_benchmark = False
train_gru = False

generate_data = True
N_REPEATS = 1

path_data = 'weinhardt2026/studies/eckstein2026/data/eckstein2024.csv'
path_spice = 'weinhardt2026/studies/eckstein2026/params/spice_eckstein2026.pkl'
path_benchmark = 'weinhardt2026/studies/eckstein2026/params/benchmark_eckstein2026.pkl'
path_gru = 'weinhardt2026/studies/eckstein2026/params/gru_eckstein2026.pkl'
output_dir='weinhardt2026/studies/eckstein2026/results'

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

test_blocks = (2,)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of actions in dataset: {dataset_train.n_actions}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,

    epochs=0,
    warmup_steps=500,

    verbose=True,
    save_path_spice=path_spice,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
)

if train_spice:
    if estimator.epochs == 0:
        estimator.load_spice(path_spice)
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
else:
    estimator.load_spice(path_spice)

# -------------------------------------------------------------------------------------------
# CASTRO2025 BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

benchmark = Castro2025Model(
    n_participants=dataset_train.n_participants,
    n_actions=dataset_train.n_actions,
    batch_first=True,
)

if train_benchmark:
    optimizer = torch.optim.Adam(params=benchmark.parameters(), lr=0.01)
    benchmark = training(
        model=benchmark, 
        optimizer=optimizer,
        dataset_train=dataset_train, 
        dataset_test=dataset_test,
        epochs=1000, 
        loss_kwargs={'label_smoothing': 0.0},
    )
    torch.save(benchmark.state_dict(), path_benchmark)
else:
    benchmark.load_state_dict(torch.load(path_benchmark, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

gru = GRUModel(
    n_actions=dataset_train.n_actions,
    additional_inputs=2,
    dropout=0.1,
    hidden_size=32,
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    gru = training(
        model=gru, 
        optimizer=optimizer,
        dataset_train=dataset_train, 
        dataset_test=dataset_test,
        epochs=1000, 
        loss_kwargs={'label_smoothing': 0.0},
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
    benchmark_model=benchmark.to(torch.device('cpu')),
    gru_model=gru.eval().to(torch.device('cpu')),
))

print("\n--- Model evaluation (test) ---")
print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    benchmark_model=benchmark.to(torch.device('cpu')),
    gru_model=gru.eval().to(torch.device('cpu')),
    output_dir=output_dir,
))

if generate_data:
    # -------------------------------------------------------------------------------------------
    # GENERATIVE BENCHMARKING
    # -------------------------------------------------------------------------------------------

    estimator.use_sindy(True)
    ds_spice = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=estimator,
        dataset=dataset_train,
    )

    estimator.use_sindy(False)
    ds_spice_rnn = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=estimator,
        dataset=dataset_train,
    )

    gru.eval()
    ds_gru = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=gru,
        dataset=dataset_train,
    )

    ds_benchmark = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=benchmark,
        dataset=dataset_train,
    )

    # -------------------------------------------------------------------------------------------
    # ANALYSIS: GENERATIVE BEHAVIOR
    # -------------------------------------------------------------------------------------------

    analysis_generative_behavior(
        path_data_real=path_data,
        path_data_gru=ds_gru,
        path_data_benchmark=ds_benchmark,
        path_data_spice=ds_spice,
        path_data_spice_rnn=ds_spice_rnn,
        output_dir=output_dir,
    )

# -------------------------------------------------------------------------------------------
# ANALYSIS: INDIVIDUAL DIFFERENCES
# -------------------------------------------------------------------------------------------

# analysis_coefficients_distributions(
#     spice_model=estimator,
#     output_dir=output_dir,
# )

# -------------------------------------------------------------------------------------------
# ANALYSIS: BEHAVIORAL DIFFERENCES
# -------------------------------------------------------------------------------------------

analysis_coefficients_individuals(
    spice_model=estimator,
    path_data=path_data,
    analysis='cont',
    criterion='mean_reward',
    output_dir=output_dir,
)
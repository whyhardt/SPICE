import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator
from weinhardt2026.studies.ganesh2024a.spice_ganesh2024a import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a import BayesianModel, get_dataset, generate_behavior
from weinhardt2026.studies.ganesh2024a.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.utils.generation import generate_repeated


train_spice = False
train_bay = False
train_gru = False

N_REPEATS = 100

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/ganesh2024a/data/ganesh2024a_choice.csv'
test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {info_dataset['n_participants']}")
print(f"Number of actions: {info_dataset['n_actions']}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/ganesh2024a/params/spice_ganesh2024a.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,

    epochs=1000,
    warmup_steps=500,
    ensemble_size=1,

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
# BAYESIAN BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_bay = 'weinhardt2026/studies/ganesh2024a/params/bay_ganesh2024a.pkl'

bay = BayesianModel(n_participants=info_dataset['n_participants'], batch_first=True)

if train_bay:
    optimizer = torch.optim.Adam(bay.parameters(), lr=0.01)
    bay = training(
        model=bay, optimizer=optimizer,
        dataset_train=dataset_train, dataset_test=dataset_test,
        epochs=1000, device=torch.device('cpu'),
    )
    torch.save(bay.state_dict(), path_bay)
else:
    bay.load_state_dict(torch.load(path_bay, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/ganesh2024a/params/gru_ganesh2024a.pkl'

gru = GRUModel(
    n_actions=info_dataset['n_actions'],
    n_participants=info_dataset['n_participants'],
    additional_inputs=2,
    dropout=0.25,
    embedding_size=8,
    hidden_size=8,
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    gru = training(
        model=gru, optimizer=optimizer,
        dataset_train=dataset_train, dataset_test=dataset_test,
        epochs=1000, device=torch.device('cpu'),
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
bay.eval()
gru.eval()

for pid in range(3):
    print(f"\nExample SPICE model (participant {pid}):")
    estimator.print_spice_model(participant_id=pid)

print("\n--- Model evaluation (train) ---")
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=bay,
    gru_model=gru,
))

print("\n--- Model evaluation (test) ---")
print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    benchmark_model=bay,
    gru_model=gru,
    output_dir='weinhardt2026/studies/ganesh2024a/results',
))

# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

data_dir = 'weinhardt2026/studies/ganesh2024a/data'
output_dir = 'weinhardt2026/studies/ganesh2024a/results'

estimator.eval()
estimator.use_sindy(False)
ds_spice_rnn = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    dataset=dataset_train,
    model=estimator,
)

estimator.use_sindy(True)
ds_spice = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    dataset=dataset_train,
    model=estimator,
)

ds_benchmark = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    dataset=dataset_train,
    model=bay,
)

ds_gru = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    dataset=dataset_train,
    model=gru,
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: GENERATIVE BEHAVIOR
# -------------------------------------------------------------------------------------------

analysis_generative_behavior(
    path_data_real=path_data,
    path_data_benchmark=ds_benchmark,
    path_data_gru=ds_gru,
    path_data_spice_rnn=ds_spice_rnn,
    path_data_spice=ds_spice,
    output_dir=output_dir,
)

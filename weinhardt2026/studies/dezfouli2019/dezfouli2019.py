import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator
from spice.precoded import workingmemory
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import GQLModel, get_dataset, generate_behavior
from weinhardt2026.studies.dezfouli2019.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.utils.generation import generate_repeated


train_spice = False
train_gql = False
train_gru = False

N_REPEATS = 100

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'
test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {info_dataset['n_participants']}")
print(f"Number of actions in dataset: {info_dataset['n_actions']}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=workingmemory.SpiceModel,
    spice_config=workingmemory.CONFIG,
    n_actions=info_dataset['n_actions'],
    n_participants=info_dataset['n_participants'],
    kwargs_spice_class={'reward_binary': True},

    epochs=1000,
    warmup_steps=250,

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
# GQL BENCHMARK MODEL (Dezfouli 2019)
# -------------------------------------------------------------------------------------------

path_gql = 'weinhardt2026/studies/dezfouli2019/params/gql_dezfouli2019.pkl'

gql = GQLModel(
    n_participants=info_dataset['n_participants'],
    batch_first=True,
)

if train_gql:
    optimizer = torch.optim.Adam(params=gql.parameters(), lr=0.01)
    gql = training(
        model=gql, optimizer=optimizer,
        dataset_train=dataset_train, dataset_test=dataset_test,
        epochs=1000, device=torch.device('cpu'),
    )
    torch.save(gql.state_dict(), path_gql)
else:
    gql.load_state_dict(torch.load(path_gql, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/dezfouli2019/params/gru_dezfouli2019.pkl'

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
        epochs=1000,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
gql.eval()
gru.eval()

for pid in range(3):
    print(f"\nExample SPICE model (participant {pid}):")
    estimator.print_spice_model(participant_id=pid)

print("\n--- Model evaluation (train) ---")
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=gql.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
))

print("\n--- Model evaluation (test) ---")
print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    benchmark_model=gql.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
    output_dir='weinhardt2026/studies/dezfouli2019/results',
))

# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

data_dir = 'weinhardt2026/studies/dezfouli2019/data'
output_dir = 'weinhardt2026/studies/dezfouli2019/results'

estimator.use_sindy(False)
ds_spice_rnn = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    model=estimator,
    dataset=dataset_train,
)

estimator.use_sindy(True)
ds_spice = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    model=estimator,
    dataset=dataset_train,
)

ds_benchmark = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    model=gql,
    dataset=dataset_train,
)

ds_gru = generate_repeated(
    generate_behavior,
    n_repeats=N_REPEATS,
    model=gru,
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

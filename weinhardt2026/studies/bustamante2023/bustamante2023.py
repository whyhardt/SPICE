import torch

from spice import SpiceEstimator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from weinhardt2026.studies.bustamante2023.spice_bustamante2023 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import (
    MarginalValueTheoremModel, get_dataset, generate_behavior,
)
from weinhardt2026.studies.bustamante2023.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions


train_spice = False
train_mvt = False
train_gru = True

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/bustamante2023/data/bustamante2023.csv'
test_blocks = (3, 6)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of actions in dataset: {dataset_train.n_actions}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/bustamante2023/params/spice_bustamante2023.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,

    epochs=1000,
    warmup_steps=500,
    sindy_weight=0,
    ensemble_size=1,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
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
    estimator.load_spice(path_spice)

# -------------------------------------------------------------------------------------------
# MVT BENCHMARK MODEL (Constantino et al. 2015)
# -------------------------------------------------------------------------------------------

path_mvt = 'weinhardt2026/studies/bustamante2023/params/mvt_bustamante2023.pkl'

mvt = MarginalValueTheoremModel(
    n_participants=dataset_train.n_participants,
    depletion=None,
    baseline_gain=None,
    batch_first=True,
)

if train_mvt:
    print("Training benchmark model...")
    optimizer = torch.optim.Adam(params=mvt.parameters(), lr=0.01)
    mvt = training(
        model=mvt,
        optimizer=optimizer,
        epochs=1000,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        device=torch.device('cpu'),
    )
    torch.save(mvt.state_dict(), path_mvt)
else:
    mvt.load_state_dict(torch.load(path_mvt, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/bustamante2023/params/gru_bustamante2023.pkl'

gru = GRUModel(dataset_train.n_actions)

if train_gru:
    print("Training GRU model...")
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))


# -------------------------------------------------------------------------------------------
# ANALYSIS: EXAMPLE SPICE MODEL
# -------------------------------------------------------------------------------------------

estimator.eval()
mvt.eval()
gru.eval()

for participant_id in range(3):
    print(f"\nExample SPICE model (participant {participant_id}):")
    estimator.print_spice_model(participant_id=participant_id)


# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL EVALUATION (TRAIN)
# -------------------------------------------------------------------------------------------

print("\n--- Model evaluation (train) ---")
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=mvt.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
))

# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL EVALUATION (TEST)
# -------------------------------------------------------------------------------------------

print("\n--- Model evaluation (test) ---")
print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    benchmark_model=mvt.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
))


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

output_dir = 'weinhardt2026/studies/bustamante2023/results'

estimator.use_sindy(False)
generated_dataset_spice_rnn = generate_behavior(
    dataset=dataset_train,
    model=estimator,
    save_dataset='weinhardt2026/studies/bustamante2023/data/bustamante2023_spice_rnn.csv',
)

estimator.use_sindy(True)
generated_dataset_spice = generate_behavior(
    dataset=dataset_train,
    model=estimator,
    save_dataset='weinhardt2026/studies/bustamante2023/data/bustamante2023_spice.csv',
)

generated_dataset_benchmark = generate_behavior(
    dataset=dataset_train,
    model=mvt,
    save_dataset='weinhardt2026/studies/bustamante2023/data/bustamante2023_benchmark.csv',
)

generated_dataset_gru = generate_behavior(
    dataset=dataset_train,
    model=gru,
    save_dataset='weinhardt2026/studies/bustamante2023/data/bustamante2023_gru.csv',
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: GENERATIVE BEHAVIOR
# -------------------------------------------------------------------------------------------

analysis_generative_behavior(
    path_data_real='weinhardt2026/studies/bustamante2023/data/bustamante2023.csv',
    path_data_benchmark='weinhardt2026/studies/bustamante2023/data/bustamante2023_benchmark.csv',
    path_data_gru='weinhardt2026/studies/bustamante2023/data/bustamante2023_gru.csv',
    path_data_spice_rnn='weinhardt2026/studies/bustamante2023/data/bustamante2023_spice_rnn.csv',
    path_data_spice='weinhardt2026/studies/bustamante2023/data/bustamante2023_spice.csv',
)


# -------------------------------------------------------------------------------------------
# ANALYSIS: COEFFICIENT DISTRIBUTIONS
# -------------------------------------------------------------------------------------------

# analysis_coefficients_distributions(
#     spice_model=estimator,
#     output_dir=output_dir,
# )

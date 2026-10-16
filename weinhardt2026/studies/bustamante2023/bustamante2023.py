import torch

from spice import SpiceEstimator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from weinhardt2026.studies.bustamante2023.spice_bustamante2023 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.studies.bustamante2023.benchmarking_bustamante2023 import (
    EffortCostMVTModel, get_dataset,
)
from weinhardt2026.analysis.analysis_generative_comparison import run_generative_analysis
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions


train_spice = False
train_mvt = False
train_gru = False

generate_data = False

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

path_spice = 'weinhardt2026/studies/bustamante2023/params/spice_bustamante2023.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,

    sindy_weight=0.,
    
    epochs=1000,
    warmup_steps=500,

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
    estimator.load_spice(best_checkpoint('bustamante2023', 'spice'))

# -------------------------------------------------------------------------------------------
# MVT BENCHMARK MODEL WITH EFFORT COSTS (Bustamante et al. 2023, Eqs. 2-5)
# -------------------------------------------------------------------------------------------

path_mvt = 'weinhardt2026/studies/bustamante2023/params/effort_mvt_bustamante2023.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

mvt = EffortCostMVTModel(
    n_participants=dataset_train.n_participants,
    path_data=path_data,
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
    mvt = load_best('bustamante2023', 'benchmark')

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/bustamante2023/params/gru_bustamante2023.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

gru = GRUModel(dataset_train.n_actions, additional_inputs=4, additional_inputs_scale=(10., 10., 1., 1.))   # harvest_duration, travel_duration (constant, scaled to [0, 1]), is_msit, is_high

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
    gru = load_best('bustamante2023', 'gru')


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
# ANALYSIS: MODEL EVALUATION
# -------------------------------------------------------------------------------------------

print("\n--- Model evaluation (train: likelihood, AIC, BIC; test: likelihood) ---")
analysis_model_evaluation(
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    spice_model=estimator,
    benchmark_model=mvt.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
    output_dir='weinhardt2026/studies/bustamante2023/results',
    spice_selection='best-on-train',
    verbose=True,
)


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

data_dir = 'weinhardt2026/studies/bustamante2023/data'
output_dir = 'weinhardt2026/studies/bustamante2023/results'

if generate_data:
    # simulate every model in the task and compare with the real behavior; outputs in results/.
    # SPICE: the ensemble member with the lowest training AIC across the grid-search seeds
    # (also runnable on its own: python weinhardt2026/analysis/analysis_generative_comparison.py bustamante2023)
    run_generative_analysis('bustamante2023')   # runs per model: N_REPEATS in analysis_generative.py


# -------------------------------------------------------------------------------------------
# ANALYSIS: COEFFICIENT DISTRIBUTIONS
# -------------------------------------------------------------------------------------------

# analysis_coefficients_distributions(
#     spice_model=estimator,
#     output_dir=output_dir,
# )

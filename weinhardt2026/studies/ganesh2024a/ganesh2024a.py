import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator
from weinhardt2026.studies.ganesh2024a.spice_ganesh2024a import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.studies.ganesh2024a.benchmarking_ganesh2024a import BayesianModel, get_dataset
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation


train_spice = False
train_bay = False
train_gru = False


# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/ganesh2024a/data/ganesh2024a.csv'
test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/ganesh2024a/params/spice_ganesh2024a.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,

    epochs=1000,
    warmup_steps=500,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(best_checkpoint('ganesh2024a', 'spice'))

# -------------------------------------------------------------------------------------------
# BAYESIAN BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_bay = 'weinhardt2026/studies/ganesh2024a/params/bay_ganesh2024a.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

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
    bay = load_best('ganesh2024a', 'benchmark')

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/ganesh2024a/params/gru_ganesh2024a.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

gru = GRUModel(
    n_actions=info_dataset['n_actions'],
    additional_inputs=2,
    dropout=0.25,
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
    gru = load_best('ganesh2024a', 'gru')

# -------------------------------------------------------------------------------------------
# MODEL EVALUATION
# -------------------------------------------------------------------------------------------

estimator.eval()
bay.eval()
gru.eval()


print("\n--- Model evaluation (train: likelihood, AIC, BIC; test: likelihood) ---")
analysis_model_evaluation(
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    spice_model=estimator,
    benchmark_model=bay,
    gru_model=gru,
    spice_selection='best-on-train',
    verbose=True,
)

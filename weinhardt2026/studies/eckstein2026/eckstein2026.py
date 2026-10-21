import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator
from weinhardt2026.studies.eckstein2026.spice_eckstein2026 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.studies.eckstein2026.benchmarking_eckstein2026 import BestRLModel, get_dataset
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation


train_spice = False
train_benchmark = False
train_gru = False


path_data = 'weinhardt2026/studies/eckstein2026/data/eckstein2026.csv'
# path_data = 'weinhardt2026/studies/eckstein2026/data/public_train.csv'
path_spice = 'weinhardt2026/studies/eckstein2026/params/spice_eckstein2026.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
path_benchmark = 'weinhardt2026/studies/eckstein2026/params/benchmark_eckstein2026.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
path_gru = 'weinhardt2026/studies/eckstein2026/params/gru_eckstein2026.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

test_blocks = (2,)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)


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
    estimator.load_spice(best_checkpoint('eckstein2026', 'spice'))

# -------------------------------------------------------------------------------------------
# BENCHMARK MODEL (BestRL)
# -------------------------------------------------------------------------------------------

benchmark = BestRLModel(
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
    benchmark = load_best('eckstein2026', 'benchmark')

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
    gru = load_best('eckstein2026', 'gru')

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
    benchmark_model=benchmark.to(torch.device('cpu')),
    gru_model=gru.eval().to(torch.device('cpu')),
    spice_selection='best-on-train',
    verbose=True,
)

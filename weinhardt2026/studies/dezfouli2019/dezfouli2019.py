import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator

from spice_dezfouli2019 import SpiceModel, CONFIG

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import (
    GQLModel, get_dataset,
)
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation


train_spice = False
train_benchmark = False
train_gru = False

path_data = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'
path_spice = 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019_original_sw_0.001.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
path_benchmark = 'weinhardt2026/studies/dezfouli2019/params/benchmark_dezfouli2019.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
path_gru = 'weinhardt2026/studies/dezfouli2019/params/gru_dezfouli2019.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=info_dataset['n_actions'],
    n_participants=info_dataset['n_participants'],
    kwargs_spice_class={'reward_binary': True},

    epochs=1000,
    warmup_steps=500,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys)#, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(best_checkpoint('dezfouli2019', 'spice'))

# -------------------------------------------------------------------------------------------
# GQL BENCHMARK MODEL (Dezfouli 2019)
# -------------------------------------------------------------------------------------------

benchmark = GQLModel(
    n_participants=info_dataset['n_participants'],
    batch_first=True,
)

if train_benchmark:
    optimizer = torch.optim.Adam(params=benchmark.parameters(), lr=0.01)
    benchmark = training(
        model=benchmark, optimizer=optimizer,
        dataset_train=dataset_train, dataset_test=dataset_test,
        epochs=1000, device=torch.device('cpu'),
    )
    torch.save(benchmark.state_dict(), path_benchmark)
else:
    benchmark = load_best('dezfouli2019', 'benchmark')

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------


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
        epochs=1000,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru = load_best('dezfouli2019', 'gru')

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
    gru_model=gru.to(torch.device('cpu')),
    spice_selection='best-on-train',
    verbose=True,
)

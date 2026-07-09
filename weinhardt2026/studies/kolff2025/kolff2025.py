import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from weinhardt2026.studies.kolff2025.spice_kolff2025 import CONFIG, SpiceModel, cross_entropy_loss_mask_waiting, filter_non_waiting

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import ConditionalFrequencyModel, get_dataset


# Set to False to reuse saved params instead of retraining.
train_spice = False
train_gru = False
train_benchmark = False

# -------------------------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/kolff2025/data/kolff2025.csv'
path_spice = 'weinhardt2026/studies/kolff2025/params/spice_kolff2025.pkl'
path_gru = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025.pkl'
path_benchmark = 'weinhardt2026/studies/kolff2025/params/benchmark_kolff2025.pkl'
data_dir = 'weinhardt2026/studies/kolff2025/data'
output_dir = 'weinhardt2026/studies/kolff2025/results'

for path in (path_spice, path_gru, path_benchmark):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

dataset_train, dataset_test, info = get_dataset(path_data=path_data, test_blocks=(1,))
n_participants = info['n_participants']

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    n_reward_features=dataset_train.n_reward_features,
    
    embedding_size=4,
    loss_fn=cross_entropy_loss_mask_waiting,
    epochs=0,
    warmup_steps=500,
    ensemble_size=10,
    sindy_weight=0.,
    
    device=device,
    verbose=True,
    compiled_forward=True,
)

if train_spice:
    if estimator.epochs == 0:
        estimator.load_spice(path_spice)
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
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

gru = GRUModel(
    n_actions=dataset_train.n_actions,
    additional_inputs=dataset_train.n_additional_inputs,
    n_reward_features=0,
    # for using vanilla GRU -> set both to 1 (not using participant embeddings)
    # n_participants=n_participants,  # to set sender
    # n_experiments=n_participants,   # to set receiver; each receiver is basically an experimental condition
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)

    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=cross_entropy_loss_mask_waiting,
        scheduler=True,
    )

    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))


# -------------------------------------------------------------------------------------------
# BENCHMARK: CONDITIONAL FREQUENCY MODEL
# -------------------------------------------------------------------------------------------

benchmark = ConditionalFrequencyModel(
    n_actions=dataset_train.n_actions,
    n_participants=n_participants,
)

if train_benchmark:
    benchmark.fit(dataset_train)
    torch.save(benchmark.state_dict(), path_benchmark)
    print("Benchmark parameters saved to " + path_benchmark)
else:
    benchmark.load_state_dict(torch.load(path_benchmark, map_location='cpu'))


# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
gru.eval().to(torch.device('cpu'))

# General analysis: model evaluation (average trial likelihood)
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    trial_filter=filter_non_waiting,
    n_actions_random_baseline=dataset_train.n_actions - 1,  # exclude waiting from random baseline
))

print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    output_dir=output_dir,
    trial_filter=filter_non_waiting,
    n_actions_random_baseline=dataset_test.n_actions - 1,  # exclude waiting from random baseline
))


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import generate_behavior_replay
from weinhardt2026.studies.kolff2025.analysis_generative import analysis_generative_behavior

N_REPEATS = 100

# Full dataset (unsplit) for generative benchmarking
dataset_full, _, _ = get_dataset(path_data=path_data, test_blocks=())

# Generate behavior for GRU
gru.eval().to(torch.device('cpu'))
dataset_gen_gru = generate_behavior_replay(
    model=gru,
    dataset=dataset_full,
    n_repeats=N_REPEATS,
    save_csv=f'{data_dir}/gen_gru.csv',
)

# Generate behavior for benchmark
benchmark.eval()
dataset_gen_benchmark = generate_behavior_replay(
    model=benchmark,
    dataset=dataset_full,
    n_repeats=N_REPEATS,
    save_csv=f'{data_dir}/gen_benchmark.csv',
)

# Generate behavior for SPICE-RNN
estimator.eval()
estimator.model.to(torch.device('cpu'))
estimator.use_sindy(False)
dataset_gen_spice_rnn = generate_behavior_replay(
    model=estimator,
    dataset=dataset_full,
    n_repeats=N_REPEATS,
    save_csv=f'{data_dir}/gen_spice_rnn.csv',
)

# Generate behavior for SPICE-EQ
estimator.use_sindy(True)
dataset_gen_spice = generate_behavior_replay(
    model=estimator,
    dataset=dataset_full,
    n_repeats=N_REPEATS,
    save_csv=f'{data_dir}/gen_spice.csv',
)

# Generative analysis: compare real vs generated behavior
analysis_generative_behavior(
    dataset_real=dataset_full,
    dataset_benchmark=dataset_gen_benchmark,
    dataset_gru=dataset_gen_gru,
    dataset_spice_rnn=dataset_gen_spice_rnn,
    dataset_spice=dataset_gen_spice,
    output_dir=output_dir,
)

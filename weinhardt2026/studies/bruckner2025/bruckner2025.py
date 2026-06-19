import torch
import torch.nn as nn

from spice import SpiceEstimator, SpiceConfig, BaseModel, SpiceDataset, split_data_along_blockdim
from spice_bruckner2025 import SpiceModel, CONFIG

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import (
    get_dataset, mse_loss, generate_behavior, RationalResourceModel, POSITION_SCALE,
)
from weinhardt2026.studies.bruckner2025.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation_mse


train_spice = True
train_gru = False
train_benchmark = False

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------
 
path_data = 'weinhardt2026/studies/bruckner2025/data/bruckner2025.csv'
test_blocks = (1,2)   # hold out one block per participant   # use different block split -> anchoring shift was block-wise given

dataset, _ = get_dataset(path_data=path_data)

# Truncate for rapid prototyping
dataset = SpiceDataset(dataset.xs[:, :100], dataset.ys[:, :100], n_reward_features=1)

dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of experiments: {dataset_train.n_experiments}")
print(f"n_actions (continuous): {dataset_train.n_actions}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/bruckner2025/params/spice_bruckner2025.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,           # 1 (continuous)
    n_items=1,
    n_participants=dataset_train.n_participants,
    n_experiments=dataset_train.n_experiments,
    n_reward_features=dataset_train.n_reward_features,

    loss_fn=mse_loss,

    sindy_weight=0,                              # RNN-only for prototyping
    epochs=1000,
    warmup_steps=500,
    ensemble_size=1,

    device=device,
    verbose=True,
    # compiled_forward=False,
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
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/bruckner2025/params/gru_bruckner2025.pkl'

gru = GRUModel(
    n_actions=1,
    additional_inputs=3,  # z_next, catch, v_t only (exclude mu_t, c_t)
    n_reward_features=dataset_train.n_reward_features,
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    print("Training GRU...")
    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=mse_loss,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))


# -------------------------------------------------------------------------------------------
# BENCHMARK MODEL (Reduced Bayesian)
# -------------------------------------------------------------------------------------------

path_benchmark = 'weinhardt2026/studies/bruckner2025/params/benchmark_bruckner2025.pkl'

benchmark = RationalResourceModel(n_participants=dataset_train.n_participants)

if train_benchmark:
    optimizer_bm = torch.optim.Adam(benchmark.parameters(), lr=0.01)
    print("Training benchmark model...")
    benchmark = training(
        model=benchmark,
        optimizer=optimizer_bm,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=mse_loss,
        scheduler=True,
    )
    torch.save(benchmark.state_dict(), path_benchmark)
    print("Trained benchmark parameters saved to " + path_benchmark)
else:
    benchmark.load_state_dict(torch.load(path_benchmark, map_location='cpu'))


# -------------------------------------------------------------------------------------------
# ANALYSIS: EXAMPLE SPICE MODEL
# -------------------------------------------------------------------------------------------
participant_id = 0
print(f"Example SPICE model from participant {participant_id}:")
estimator.print_spice_model(participant_id=participant_id)

# -------------------------------------------------------------------------------------------
# ANALYSIS: PLAUSIBILITY CHECK
# -------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

with torch.no_grad():
    gru.eval().to(torch.device('cpu'))
    predictions_gru, _ = gru(dataset.xs)
    estimator.set_device(torch.device('cpu'))
    estimator.eval()
    predictions_spice = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(False)
    predictions_spice_rnn = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(True)
    predictions_spice_sym = torch.tensor(estimator.predict(dataset.xs))

    benchmark.eval().to(torch.device('cpu'))
    predictions_bm, _ = benchmark(dataset.xs)

# Plot for session 0, trials 5–45
session = 0
t_start, t_end = 5, min(45, dataset.xs.shape[1])
trials = range(t_start, t_end)

actual_bucket = dataset.xs[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
actual_outcome = dataset.xs[session, t_start:t_end, 0, 1].numpy() * POSITION_SCALE
actual_mu = dataset.xs[session, t_start:t_end, 0, 2].numpy() * POSITION_SCALE

pred_gru = predictions_gru[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
pred_spice_rnn = predictions_spice_rnn[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
pred_spice_sym = predictions_spice_sym[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
pred_bm = predictions_bm[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Top: positions
axs[0].plot(trials, actual_mu, 'k--', alpha=0.5, label='True mean (μ)')
axs[0].scatter(trials, actual_outcome, c='gray', s=15, alpha=0.6, label='Outcome (x_t)', zorder=3)
axs[0].plot(trials, actual_bucket, 'b-', label='Human (b_t)')
axs[0].set_ylabel('Position')
axs[0].set_title('Positions: Human vs True Mean')
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

# Bottom: model predictions
axs[1].plot(trials, actual_bucket, 'b-', alpha=0.4, label='Human (b_t)')
axs[1].plot(trials, pred_gru, 'g-', label='GRU')
axs[1].plot(trials, pred_spice_rnn, 'r--', label='SPICE-RNN')
axs[1].plot(trials, pred_spice_sym, 'r-', label='SPICE-SYM')
axs[1].plot(trials, pred_bm, 'm-', label='Benchmark')
axs[1].set_ylabel('Predicted Position')
axs[1].set_xlabel('Trial')
axs[1].set_title('Model Predictions vs Human')
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL EVALUATION (MSE)
# -------------------------------------------------------------------------------------------

df_mse = analysis_model_evaluation_mse(
    dataset=dataset_test if dataset_test is not None else dataset,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    output_dir='weinhardt2026/studies/bruckner2025/results',
    verbose=True,
)


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

output_dir = 'weinhardt2026/studies/bruckner2025/data'

# Generate behavior for GRU
gru.eval().to(torch.device('cpu'))
dataset_gen_gru = generate_behavior(
    model=gru,
    dataset=dataset,
    save_dataset=f'{output_dir}/gen_gru.csv',
)

# Generate behavior for Benchmark
benchmark.eval().to(torch.device('cpu'))
dataset_gen_bm = generate_behavior(
    model=benchmark,
    dataset=dataset,
    save_dataset=f'{output_dir}/gen_benchmark.csv',
)

# Generate behavior for SPICE-RNN
estimator.model.to(torch.device('cpu'))
estimator.use_sindy(False)
estimator.eval()
dataset_gen_spice_rnn = generate_behavior(
    model=estimator,
    dataset=dataset,
    save_dataset=f'{output_dir}/gen_spice_rnn.csv',
)

# Generate behavior for SPICE (SINDy mode)
estimator.use_sindy(True)
estimator.eval()
dataset_gen_spice = generate_behavior(
    model=estimator,
    dataset=dataset,
    save_dataset=f'{output_dir}/gen_spice.csv',
)

# Generative analysis: compare real vs generated behavior
analysis_generative_behavior(
    dataset_real=dataset,
    dataset_benchmark=dataset_gen_bm,
    dataset_gru=dataset_gen_gru,
    dataset_spice_rnn=dataset_gen_spice_rnn,
    dataset_spice=dataset_gen_spice,
    output_dir='weinhardt2026/studies/bruckner2025/results',
)

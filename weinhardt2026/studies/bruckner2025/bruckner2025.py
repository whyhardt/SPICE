import torch

from spice import SpiceEstimator, SpiceDataset, split_data_along_blockdim
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


train_spice = False
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
    compiled_forward=False,
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
    additional_inputs=6,  # all observable: z_next, catch, v_t, sigma, r_t, mu_t (excl. c_t)
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from weinhardt2026.studies.bruckner2025.benchmarking_bruckner2025 import (
    _AI_MU_T, _AI_CATCH, _AI_V_T, _AI_SIGMA,
)

ai = 1 + 1  # n_actions + n_rewards = start of additional inputs block

with torch.no_grad():
    gru.eval().to(torch.device('cpu'))
    predictions_gru, _ = gru(dataset.xs)
    estimator.set_device(torch.device('cpu'))
    estimator.eval()
    predictions_spice = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(False)
    predictions_spice_rnn = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(True)
    predictions_spice_eq = torch.tensor(estimator.predict(dataset.xs))

    benchmark.eval().to(torch.device('cpu'))
    predictions_bm, _ = benchmark(dataset.xs)


# --- Extract SPICE internal states ---

def extract_spice_states(model, xs_session):
    """Run model trial-by-trial and capture per-trial memory states."""
    n_trials = xs_session.shape[1]
    states = {key: [] for key in model.state}
    prev_state = None

    for t in range(n_trials):
        xs_trial = xs_session[:, t:t+1]
        _, prev_state = model(xs_trial, prev_state)
        for key in model.state:
            states[key].append(model.state[key][0, 0, 0, 0].item())

    return states

session = 0

with torch.no_grad():
    model = estimator.model
    model.eval()
    xs_sess = dataset.xs[session:session+1]

    model.use_sindy = False
    states_rnn = extract_spice_states(model, xs_sess)

    model.use_sindy = True
    states_sym = extract_spice_states(model, xs_sess)


# --- Compute derived dynamics ---

def _sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -500, 500)))


def compute_dynamics(states, xs_session, session=0):
    """Compute omega, tau, alpha from raw states and data."""
    n_trials = len(states['belief_value'])
    belief = np.array(states['belief_value'])
    omega = _sigmoid(states['changepoint_value'])
    tau = _sigmoid(states['uncertainty_value'])

    outcome = xs_session[session, :n_trials, 0, 1].numpy()
    bucket = xs_session[session, :n_trials, 0, 0].numpy()
    sigma = xs_session[session, :n_trials, 0, ai + _AI_SIGMA].numpy()

    # PE = outcome - belief (belief is reset to bucket each trial)
    pe = outcome - bucket

    mask_pe_big = (np.abs(pe) > 3 * sigma / 2).astype(float)
    alpha = mask_pe_big * omega + (1 - mask_pe_big) * tau

    return {
        'belief': belief,
        'omega': omega,
        'tau': tau,
        'alpha': alpha,
        'pe': pe,
        'mask_pe_big': mask_pe_big,
    }

dyn_rnn = compute_dynamics(states_rnn, dataset.xs, session)
dyn_sym = compute_dynamics(states_sym, dataset.xs, session)


# --- Plot: 4 panels ---

t_start, t_end = 5, min(45, dataset.xs.shape[1])
trials = range(t_start, t_end)

actual_bucket = dataset.ys[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE  # b_{t+1}: response to trial t
actual_outcome = dataset.xs[session, t_start:t_end, 0, 1].numpy() * POSITION_SCALE
actual_mu = dataset.xs[session, t_start:t_end, 0, ai + _AI_MU_T].numpy() * POSITION_SCALE
catch_vals = dataset.xs[session, t_start:t_end, 0, ai + _AI_CATCH].numpy()
vt_vals = dataset.xs[session, t_start:t_end, 0, ai + _AI_V_T].numpy()

pred_gru = predictions_gru[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
pred_spice_rnn = predictions_spice_rnn[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
pred_spice_eq = predictions_spice_eq[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE
pred_bm = predictions_bm[session, t_start:t_end, 0, 0].numpy() * POSITION_SCALE

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1, 0.8]})

# (1) Actual positions
axs[0].plot(trials, actual_mu, 'k--', alpha=0.5, label='True mean (μ)')
axs[0].scatter(trials, actual_outcome, c='gray', s=15, alpha=0.6, label='Outcome (x_t)', zorder=3)
axs[0].plot(trials, actual_bucket, 'b-', label='Human (b_t)')
axs[0].set_ylabel('Position')
axs[0].set_title('Positions: Human vs True Mean')
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

# (2) Model predictions
axs[1].plot(trials, actual_bucket, 'b-', alpha=0.4, label='Human (b_t)')
axs[1].plot(trials, pred_gru, 'g-', label='GRU')
axs[1].plot(trials, pred_spice_rnn, 'r--', label='SPICE-RNN')
axs[1].plot(trials, pred_spice_eq, 'r-', label='SPICE-EQ')
axs[1].plot(trials, pred_bm, 'm-', label='Benchmark')
axs[1].set_ylabel('Predicted Position')
axs[1].set_title('Model Predictions vs Human')
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

# (3) SPICE internal belief vs outcomes
belief_rnn = np.array(dyn_rnn['belief'][t_start:t_end]) * POSITION_SCALE
belief_sym = np.array(dyn_sym['belief'][t_start:t_end]) * POSITION_SCALE
axs[2].plot(trials, actual_mu, 'k--', alpha=0.5, label='True mean (μ)')
axs[2].scatter(trials, actual_outcome, c='gray', s=15, alpha=0.6, label='Outcome (x_t)', zorder=3)
axs[2].plot(trials, belief_rnn, 'r--', label='Belief (RNN)')
axs[2].plot(trials, belief_sym, 'r-', label='Belief (SYM)')
axs[2].set_ylabel('Position')
axs[2].set_title('SPICE Internal Belief vs Outcomes')
axs[2].legend(fontsize=8)
axs[2].grid(alpha=0.3)

# (4) Dynamic learning rates + event markers
axs[3].plot(list(trials), dyn_rnn['omega'][t_start:t_end],
            'orange', linestyle='--', alpha=0.7, label='ω changepoint (RNN)')
axs[3].plot(list(trials), dyn_sym['omega'][t_start:t_end],
            'orange', linestyle='-', linewidth=2, label='ω changepoint (SYM)')
axs[3].plot(list(trials), dyn_rnn['tau'][t_start:t_end],
            'purple', linestyle='--', alpha=0.7, label='τ uncertainty (RNN)')
axs[3].plot(list(trials), dyn_sym['tau'][t_start:t_end],
            'purple', linestyle='-', linewidth=2, label='τ uncertainty (SYM)')
axs[3].plot(list(trials), dyn_rnn['alpha'][t_start:t_end],
            'r--', alpha=0.5, label='α composite (RNN)')
axs[3].plot(list(trials), dyn_sym['alpha'][t_start:t_end],
            'r-', linewidth=2, label='α composite (SYM)')

axs[3].set_ylabel('Learning Rate')
axs[3].set_ylim([0, 1])
axs[3].set_xlabel('Trial')
axs[3].set_title('Dynamic Learning Rates')
axs[3].grid(alpha=0.3)

# Event markers: catch (green) and helicopter visible (cyan)
caught_trials = [t for t, c in zip(trials, catch_vals) if c > 0.5]
visible_trials = [t for t, v in zip(trials, vt_vals) if v > 0.5]
for ct in caught_trials:
    axs[3].axvline(ct, ymin=0, ymax=0.06, color='green', linewidth=1.5, alpha=0.7)
for vt in visible_trials:
    axs[3].axvline(vt, ymin=0.94, ymax=1.0, color='cyan', linewidth=1.5, alpha=0.7)

handles, labels = axs[3].get_legend_handles_labels()
handles.append(Line2D([0], [0], color='green', lw=1.5))
labels.append('Caught')
handles.append(Line2D([0], [0], color='cyan', lw=1.5))
labels.append('Visible (v_t)')
axs[3].legend(handles=handles, labels=labels, fontsize=7, loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig('weinhardt2026/studies/bruckner2025/results/dynamics_plausibility.png', bbox_inches='tight', dpi=150)
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

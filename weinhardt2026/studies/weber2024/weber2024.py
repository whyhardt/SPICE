import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import math
import torch
import os

from spice import SpiceEstimator, SpiceDataset, split_data_along_blockdim

from spice_weber2024 import SpiceModel, CONFIG
from spice_weber2024 import clamped_angular_mse
from benchmarking_weber2024 import get_dataset, ChangePointModel
from weinhardt2026.analysis.analysis_generative_comparison import run_generative_analysis

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.utils.final_models import best_checkpoint, load_best
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation_mse


train_spice = False
train_gru = False
train_benchmark = False

generate_data = False

truncate_dataset = False   # full blocks; the first-100-trial GRU did not generalize beyond trial 100

output_dir = 'weinhardt2026/studies/weber2024/results'

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/weber2024/data/weber2024.csv'
test_blocks = (3, 5, 9, 12)

dataset, _ = get_dataset(path_data=path_data)

if truncate_dataset:
    # Truncate for rapid prototyping
    dataset = SpiceDataset(dataset.xs[:, :100], dataset.ys[:, :100], n_reward_features=2, continuous_action=True)

dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of experiments (2x2 design): {dataset_train.n_experiments}")
print(f"n_actions (model output): 2 (sin, cos); ys has {dataset_train.ys.shape[-1]} columns (incl. loss metadata)")
print(f"n_reward_features: {dataset_train.n_reward_features}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/weber2024/params/spice_weber2024_continuous.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=2,                                    # next shield position: sin, cos
    n_items=1,                                      # belief as one angle
    n_participants=dataset_train.n_participants,
    n_experiments=1,                                # conditions are module inputs: one coefficient set per participant
    n_reward_features=dataset_train.n_reward_features,

    loss_fn=clamped_angular_mse,

    # the penalty defaults are tuned for cross-entropy; against this MSE they switch modules off
    gate_penalty=0.,
    feature_penalty=0.,
    
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
    estimator.load_spice(best_checkpoint('weber2024', 'spice'))

# -------------------------------------------------------------------------------------------
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

path_gru = 'weinhardt2026/studies/weber2024/params/gru_weber2024_continuous.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

gru = GRUModel(
    n_actions=2,                                    # output: belief_sin, belief_cos
    additional_inputs=4,                            # laser_caught, volatility, stochasticity, trial_duration_frames (excl. trueMean)
    additional_inputs_scale=(1., 1., 1., 90.),      # trial_duration_frames (1-90 frames) -> (0, 1]
    n_reward_features=dataset_train.n_reward_features,
    # n_participants=dataset_train.n_participants,
    # n_experiments=dataset_train.n_experiments,
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
        loss_fn=clamped_angular_mse,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru = load_best('weber2024', 'gru')


# -------------------------------------------------------------------------------------------
# BENCHMARK: BAYESIAN CHANGEPOINT MODEL (Foucault et al., 2026)
# -------------------------------------------------------------------------------------------

path_benchmark = 'weinhardt2026/studies/weber2024/params/benchmark_weber2024.pkl'   # new fits only; otherwise the best final seed is loaded (utils/final_models.py)

benchmark = ChangePointModel(
    n_participants=dataset_train.n_participants,
    h_step=2,               # 2° discretisation (180 grid points)
    variance_inference=True,    # infer which of two fitted noise levels a block has, instead of one fitted s.d.
)

if train_benchmark:
    optimizer = torch.optim.Adam(benchmark.parameters(), lr=0.01)
    print("Training Bayesian changepoint benchmark...")
    benchmark = training(
        model=benchmark,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=clamped_angular_mse,
        scheduler=True,
    )
    torch.save(benchmark.state_dict(), path_benchmark)
    print("Trained benchmark parameters saved to " + path_benchmark)
else:
    benchmark = load_best('weber2024', 'benchmark')

benchmark.to(torch.device('cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS: EXAMPLE SPICE MODEL
# -------------------------------------------------------------------------------------------

participant_id = 0
print(f"Example SPICE model from participant {participant_id}:")
estimator.print_spice_model(participant_id=participant_id)

print(f"\nExample SPICE model from participant {participant_id+1}:")
estimator.print_spice_model(participant_id=participant_id+1)

print(f"\nExample SPICE model from participant {participant_id+2}:")
estimator.print_spice_model(participant_id=participant_id+2)


# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL EVALUATION (MSE)
# -------------------------------------------------------------------------------------------

df_mse = analysis_model_evaluation_mse(
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    spice_selection='best-on-train',
    output_dir='weinhardt2026/studies/weber2024/results',
    verbose=True,
    loss_fn=clamped_angular_mse,
)


# -------------------------------------------------------------------------------------------
# ANALYSIS: PLAUSIBILITY CHECK
# -------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

session = 0
t_start, t_end = 5, min(45, dataset.xs.shape[1])
trials = range(t_start, t_end)

def sincos_to_degrees(sin_vals, cos_vals):
    return torch.atan2(sin_vals, cos_vals) * (180.0 / math.pi) % 360

# --- Extract predictions ---
with torch.no_grad():
    gru.eval().to(torch.device('cpu'))
    predictions_gru, _ = gru(dataset.xs)
    benchmark.eval()
    predictions_benchmark, _ = benchmark(dataset.xs)
    estimator.set_device(torch.device('cpu'))
    estimator.eval()
    estimator.use_sindy(False)
    predictions_spice_rnn = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(True)
    predictions_spice_eq = torch.tensor(estimator.predict(dataset.xs))

# --- Extract per-trial SPICE internal states (belief, learning rate, movement gain) ---
def extract_spice_states(model, xs_session):
    """Run the model trial-by-trial to capture the per-trial belief (degrees), alpha and movement gain."""
    beliefs, learning_rates, movement_gains = [], [], []
    prev_state = None
    for t in range(xs_session.shape[1]):
        _, prev_state = model(xs_session[:, t:t+1], prev_state)
        beliefs.append(model.state['belief_value'][0, 0, 0, 0].item())
        learning_rates.append(torch.sigmoid(model.state['learning_rate_value'])[0, 0, 0, 0].item())
        movement_gains.append(torch.sigmoid(model.state['movement_value'])[0, 0, 0, 0].item())
    return {'belief': beliefs, 'lr': learning_rates, 'movement_gain': movement_gains}

with torch.no_grad():
    model = estimator.model
    model.eval()
    xs_sess = dataset.xs[session:session+1]

    model.use_sindy = False
    states_rnn = extract_spice_states(model, xs_sess)

    model.use_sindy = True
    states_sym = extract_spice_states(model, xs_sess)

# --- Compute degrees for all quantities ---
shield_sin = dataset.xs[session, t_start:t_end, 0, 0]
shield_cos = dataset.xs[session, t_start:t_end, 0, 1]
laser_sin = dataset.xs[session, t_start:t_end, 0, 2]
laser_cos = dataset.xs[session, t_start:t_end, 0, 3]
caught_vals = dataset.xs[session, t_start:t_end, 0, 4].numpy()
true_mean = dataset.xs[session, t_start:t_end, 0, -6].numpy()

actual_shield_deg = sincos_to_degrees(shield_sin, shield_cos)
actual_laser_deg = sincos_to_degrees(laser_sin, laser_cos)

belief_rnn_deg = torch.tensor(states_rnn['belief'][t_start:t_end])
belief_sym_deg = torch.tensor(states_sym['belief'][t_start:t_end])

pred_benchmark_deg = sincos_to_degrees(
    predictions_benchmark[session, t_start:t_end, 0, 0],
    predictions_benchmark[session, t_start:t_end, 0, 1],
)
pred_gru_deg = sincos_to_degrees(
    predictions_gru[session, t_start:t_end, 0, 0],
    predictions_gru[session, t_start:t_end, 0, 1],
)
pred_spice_rnn_deg = sincos_to_degrees(
    predictions_spice_rnn[session, t_start:t_end, 0, 0],
    predictions_spice_rnn[session, t_start:t_end, 0, 1],
)
pred_spice_eq_deg = sincos_to_degrees(
    predictions_spice_eq[session, t_start:t_end, 0, 0],
    predictions_spice_eq[session, t_start:t_end, 0, 1],
)

caught_trials = [t for t, c in zip(trials, caught_vals) if c > 0.5]

# --- Prediction error (laser vs. belief entering that trial's update) ---
def circular_diff_deg(a_deg, b_deg):
    """Signed shortest angular difference a-b, wrapped to [-180, 180]."""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0

laser_sin_full = dataset.xs[session, :, 0, 2]
laser_cos_full = dataset.xs[session, :, 0, 3]
laser_deg_full = sincos_to_degrees(laser_sin_full, laser_cos_full)

belief_sym_deg_full = torch.tensor(states_sym['belief'])
# belief entering trial t's update is the post-update belief from t-1
# (trial 0's belief is initialized to laser[0], per the model's forward pass)
belief_prev_deg_full = torch.cat([laser_deg_full[:1], belief_sym_deg_full[:-1]])

pe_deg_full = circular_diff_deg(laser_deg_full, belief_prev_deg_full)
pe_deg = pe_deg_full[t_start:t_end].numpy()

# --- Fitted equations for the plotted participant/experiment ---
session_participant_id = int(dataset.xs[session, 0, 0, -1].item())
session_experiment_id = int(dataset.xs[session, 0, 0, -2].item())
equation_str = model.get_spice_model_string(
    participant_id=session_participant_id,
    experiment_id=session_experiment_id,
)

# --- Plot: 4 panels + equation text panel ---
fig, axs = plt.subplots(5, 1, figsize=(12, 11.5),
                         gridspec_kw={'height_ratios': [1, 1, 1, 0.6, 0.9]})
for ax in axs[:4]:
    ax.sharex(axs[0])

# (1) Actual positions: laser + shield
axs[0].scatter(list(trials), actual_laser_deg.numpy(), c='blue', s=15, alpha=0.6, label='Laser', zorder=3)
axs[0].plot(list(trials), true_mean, 'b-', label='True Mean')
axs[0].plot(list(trials), actual_shield_deg.numpy(), 'b-', alpha=0.4, label='Shield (human)')
axs[0].set_ylabel('Position (degrees)')
axs[0].set_title('Actual Positions')
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.3)

# (3) Model predictions vs human shield
axs[1].plot(list(trials), actual_shield_deg.numpy(), 'b-', alpha=0.4, label='Shield (human)')
axs[1].plot(list(trials), pred_benchmark_deg.numpy(), 'm-', alpha=0.7, label='Benchmark')
axs[1].plot(list(trials), pred_gru_deg.numpy(), 'g-', label='GRU')
axs[1].plot(list(trials), pred_spice_rnn_deg.numpy(), 'r--', label='SPICE-RNN')
axs[1].plot(list(trials), pred_spice_eq_deg.numpy(), 'r-', label='SPICE-EQ')
axs[1].set_ylabel('Predicted Position (degrees)')
axs[1].set_title('Model Predictions vs Human Shield Position')
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

# (2) SPICE internal belief + predicted shield vs laser observations
axs[2].scatter(list(trials), actual_laser_deg.numpy(), c='blue', s=15, alpha=0.6, label='Laser', zorder=3)
axs[2].plot(list(trials), true_mean, 'b-', label='True Mean')
axs[2].plot(list(trials), belief_rnn_deg.numpy(), 'r--', label='Belief (RNN)')
axs[2].plot(list(trials), belief_sym_deg.numpy(), 'r-', label='Belief (EQ)')
axs[2].plot(list(trials), pred_spice_rnn_deg.numpy(), 'orange', linestyle='--', label='Predicted Shield (RNN)')
axs[2].plot(list(trials), pred_spice_eq_deg.numpy(), 'orange', linestyle='-', label='Predicted Shield (EQ)')
axs[2].set_ylabel('Position (degrees)')
axs[2].set_title('SPICE Internal Belief & Predicted Shield vs Laser')
axs[2].legend(fontsize=8)
axs[2].grid(alpha=0.3)

# (4) Learning rate + movement gain vs. prediction error, catch ticks
axs[3].plot(list(trials), states_rnn['lr'][t_start:t_end], 'r--', label='α (RNN)')
axs[3].plot(list(trials), states_sym['lr'][t_start:t_end], 'r-', label='α (EQ)')
axs[3].plot(list(trials), states_rnn['movement_gain'][t_start:t_end], color='orange', linestyle='--', label='g (RNN)')
axs[3].plot(list(trials), states_sym['movement_gain'][t_start:t_end], color='orange', linestyle='-', label='g (EQ)')
axs[3].set_ylabel('Learning rate (α) / movement gain (g)')
axs[3].set_ylim([0, 1])
axs[3].set_title('Learning Rate & Movement Gain vs. Prediction Error')

ax4_twin = axs[3].twinx()
ax4_twin.plot(list(trials), pe_deg, color='0.5', linewidth=1, alpha=0.5, zorder=1)
ax4_twin.scatter(list(trials), pe_deg, c=['green' if c > 0.5 else 'gray' for c in caught_vals],
                  s=18, alpha=0.8, zorder=2, label='PE (deg)')
ax4_twin.axhline(0, color='0.7', linewidth=0.8, zorder=0)
ax4_twin.set_ylabel('Prediction Error (degrees)', color='0.4')
ax4_twin.tick_params(axis='y', labelcolor='0.4')

axs[3].set_xlabel('Trial')
axs[3].grid(alpha=0.3)

# Catch event markers
for ct in caught_trials:
    axs[3].axvline(ct, ymin=0, ymax=0.06, color='green', linewidth=1.5, alpha=0.7)

handles, labels = axs[3].get_legend_handles_labels()
handles.append(Line2D([0], [0], marker='o', color='0.5', markerfacecolor='green', linestyle='None'))
labels.append('PE, caught')
handles.append(Line2D([0], [0], marker='o', color='0.5', markerfacecolor='gray', linestyle='None'))
labels.append('PE, missed')
axs[3].legend(handles=handles, labels=labels, fontsize=8, loc='upper left')

# (5) Fitted SPICE equations for this participant/experiment
axs[4].axis('off')
axs[4].set_title(
    f'Fitted equations — participant {session_participant_id}, experiment {session_experiment_id}',
    fontsize=10, loc='left',
)
axs[4].text(
    0.0, 1.0, equation_str,
    transform=axs[4].transAxes, ha='left', va='top',
    fontsize=9, family='monospace',
)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'generated_shield_positions.png'), bbox_inches='tight', dpi=150)
plt.show()


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

if generate_data:
    # simulate every model in the task and compare with the real behavior; outputs in results/.
    # SPICE: the ensemble member with the lowest training AIC across the grid-search seeds
    # (also runnable on its own: python weinhardt2026/analysis/analysis_generative_comparison.py weber2024)
    run_generative_analysis('weber2024')   # runs per model: N_REPEATS in analysis_generative.py

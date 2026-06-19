import math
import torch

from spice import SpiceEstimator, SpiceDataset, split_data_along_blockdim

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from weinhardt2026.studies.weber2024.spice_weber2024 import SpiceModel, CONFIG
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.weber2024.benchmarking_weber2024 import (
    get_dataset, clamped_angular_mse, generate_behavior,
)
from weinhardt2026.studies.weber2024.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation_mse


train_spice = False
train_gru = False

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/weber2024/data/weber2024.csv'
test_blocks = (2, 5, 9, 14)

dataset, _ = get_dataset(path_data=path_data)

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

path_spice = 'weinhardt2026/studies/weber2024/params/spice_weber2024_continuous.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=2,                                    # belief_sin, belief_cos
    n_items=2,
    n_participants=dataset_train.n_participants,
    n_experiments=dataset_train.n_experiments,
    n_reward_features=dataset_train.n_reward_features,

    loss_fn=clamped_angular_mse,

    sindy_weight=0,                                 # RNN-only for prototyping
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

path_gru = 'weinhardt2026/studies/weber2024/params/gru_weber2024_continuous.pkl'

gru = GRUModel(
    n_actions=2,                                    # output: belief_sin, belief_cos
    additional_inputs=1,                            # laser_caught, volatility, stochasticity, trial_duration_frames
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
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))


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

# analysis_model_evaluation_mse expects targets and predictions to have the same
# last dimension. Our ys has 5 columns (2 target + 3 loss metadata), so slice to
# the first 2 columns for evaluation.
eval_ds = dataset_test if dataset_test is not None else dataset
eval_ds_sliced = SpiceDataset(eval_ds.xs, eval_ds.ys[:, :, :, :2], n_reward_features=2, continuous_action=True)

df_mse = analysis_model_evaluation_mse(
    dataset=eval_ds_sliced,
    spice_model=estimator,
    gru_model=gru,
    output_dir='weinhardt2026/studies/weber2024/results',
    verbose=True,
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
    estimator.set_device(torch.device('cpu'))
    estimator.eval()
    estimator.use_sindy(False)
    predictions_spice_rnn = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(True)
    predictions_spice_sym = torch.tensor(estimator.predict(dataset.xs))

# --- Extract per-trial SPICE internal states (belief, learning rate) ---
def extract_spice_states(model, xs_session):
    """Run model trial-by-trial to capture per-trial belief and learning rate."""
    beliefs_sin, beliefs_cos, lrs = [], [], []
    prev_state = None
    for t in range(xs_session.shape[1]):
        xs_trial = xs_session[:, t:t+1]
        _, prev_state = model(xs_trial, prev_state)
        belief = model.state['belief_value']   # (W, E, B, I=2)
        lr = torch.sigmoid(model.state['lr_value'])
        beliefs_sin.append(belief[0, 0, 0, 0].item())
        beliefs_cos.append(belief[0, 0, 0, 1].item())
        lrs.append(lr[0, 0, 0, :].mean().item())
    return beliefs_sin, beliefs_cos, lrs

with torch.no_grad():
    model = estimator.model
    model.eval()
    xs_sess = dataset.xs[session:session+1]

    model.use_sindy = False
    bs_rnn, bc_rnn, lr_rnn = extract_spice_states(model, xs_sess)

    model.use_sindy = True
    bs_sym, bc_sym, lr_sym = extract_spice_states(model, xs_sess)

# --- Compute degrees for all quantities ---
shield_sin = dataset.xs[session, t_start:t_end, 0, 0]
shield_cos = dataset.xs[session, t_start:t_end, 0, 1]
laser_sin = dataset.xs[session, t_start:t_end, 0, 2]
laser_cos = dataset.xs[session, t_start:t_end, 0, 3]
caught_vals = dataset.xs[session, t_start:t_end, 0, 4].numpy()
true_mean = dataset.xs[session, t_start:t_end, 0, -6].numpy()

actual_shield_deg = sincos_to_degrees(shield_sin, shield_cos)
actual_laser_deg = sincos_to_degrees(laser_sin, laser_cos)

belief_rnn_deg = sincos_to_degrees(
    torch.tensor(bs_rnn[t_start:t_end]), torch.tensor(bc_rnn[t_start:t_end]),
)
belief_sym_deg = sincos_to_degrees(
    torch.tensor(bs_sym[t_start:t_end]), torch.tensor(bc_sym[t_start:t_end]),
)

pred_gru_deg = sincos_to_degrees(
    predictions_gru[session, t_start:t_end, 0, 0],
    predictions_gru[session, t_start:t_end, 0, 1],
)
pred_spice_rnn_deg = sincos_to_degrees(
    predictions_spice_rnn[session, t_start:t_end, 0, 0],
    predictions_spice_rnn[session, t_start:t_end, 0, 1],
)
pred_spice_sym_deg = sincos_to_degrees(
    predictions_spice_sym[session, t_start:t_end, 0, 0],
    predictions_spice_sym[session, t_start:t_end, 0, 1],
)

caught_trials = [t for t, c in zip(trials, caught_vals) if c > 0.5]

# --- Plot: 4 panels ---
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1, 0.6]})

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
axs[1].plot(list(trials), pred_gru_deg.numpy(), 'g-', label='GRU')
axs[1].plot(list(trials), pred_spice_rnn_deg.numpy(), 'r--', label='SPICE-RNN')
axs[1].plot(list(trials), pred_spice_sym_deg.numpy(), 'r-', label='SPICE-SYM')
axs[1].set_ylabel('Predicted Position (degrees)')
axs[1].set_title('Model Predictions vs Human Shield Position')
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.3)

# (2) SPICE internal belief vs laser observations
axs[2].scatter(list(trials), actual_laser_deg.numpy(), c='blue', s=15, alpha=0.6, label='Laser', zorder=3)
axs[2].plot(list(trials), true_mean, 'b-', label='True Mean')
axs[2].plot(list(trials), belief_rnn_deg.numpy(), 'r--', label='Belief (RNN)')
axs[2].plot(list(trials), belief_sym_deg.numpy(), 'r-', label='Belief (SYM)')
axs[2].set_ylabel('Position (degrees)')
axs[2].set_title('SPICE Internal Belief vs Laser')
axs[2].legend(fontsize=8)
axs[2].grid(alpha=0.3)

# (4) Dynamic learning rate + catch ticks
axs[3].plot(list(trials), lr_rnn[t_start:t_end], 'r--', label='α (RNN)')
axs[3].plot(list(trials), lr_sym[t_start:t_end], 'r-', label='α (SYM)')
axs[3].set_ylabel('Learning Rate (α)')
axs[3].set_xlabel('Trial')
axs[3].set_title('Dynamic Learning Rate')
axs[3].grid(alpha=0.3)
for ct in caught_trials:
    axs[3].axvline(ct, ymin=0, ymax=0.06, color='green', linewidth=1.5, alpha=0.7)
handles, labels = axs[3].get_legend_handles_labels()
handles.append(Line2D([0], [0], color='green', lw=1.5))
labels.append('Caught')
axs[3].legend(handles=handles, labels=labels, fontsize=8)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

output_dir = 'weinhardt2026/studies/weber2024/results'

# Generate behavior for GRU
gru.eval().to(torch.device('cpu'))
dataset_gen_gru = generate_behavior(
    model=gru,
    dataset=dataset,
)

# Generate behavior for SPICE-RNN
estimator.set_device(torch.device('cpu'))
estimator.eval()
estimator.use_sindy(False)
dataset_gen_spice_rnn = generate_behavior(
    model=estimator,
    dataset=dataset,
)

estimator.use_sindy(True)
dataset_gen_spice_sym = generate_behavior(
    model=estimator,
    dataset=dataset,
)


# Generative analysis: compare real vs generated behavior
analysis_generative_behavior(
    dataset_real=dataset,
    dataset_gru=dataset_gen_gru,
    dataset_spice_rnn=dataset_gen_spice_rnn,
    dataset_spice=dataset_gen_spice_sym,
    output_dir=output_dir,
)

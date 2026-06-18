import torch
import torch.nn as nn

from spice import SpiceEstimator, SpiceConfig, BaseModel, csv_to_dataset, SpiceDataset, split_data_along_blockdim
from weinhardt2026.studies.weber2024.archive.spice_weber2024_staymove import SpiceModel, CONFIG

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from weinhardt2026.utils.benchmarking_gru import GRUModel, training


train_spice = False
train_gru = False

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/weber2024/data/weber2024.csv'
test_sessions = 2,5,9,14

dataset = csv_to_dataset(
    file = path_data,
    df_participant_id='participant',
    df_experiment_id='experiment',
    df_choice='action',
    df_feedback=None,
    df_block='block',
    additional_inputs=[
        'shield_distance_initial',
        'shieldRotation',
        'laserRotation',
        'trial_duration_frames',
        'total_movement_degrees',
        'frames_spent_moving',
        'button_press_onsets',
        'reaction_time_frames',
        'laser_caught',
        'volatility',
        'stochasticity',
        ],
)

# --------------------------------------------------------------------------
# make dataset much shorter for rapid prototyping
from spice import SpiceDataset
dataset = SpiceDataset(dataset.xs[:, :100], dataset.ys[:, :100], n_reward_features=0)
# --------------------------------------------------------------------------

if test_sessions is not None:
    dataset_train, dataset_test = split_data_along_blockdim(dataset, test_sessions)
else:
    dataset_train, dataset_test = dataset, None

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of experiments (2x2 design): {dataset_train.n_experiments}")
print(f"Number of actions in dataset: {dataset_train.n_actions}")
print(f"Number of additional inputs: {dataset_train.xs.shape[-1]-2*dataset_train.n_actions-3}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/weber2024/params/spice_weber2024.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    n_experiments=dataset_train.n_experiments,
    n_reward_features=dataset_train.n_reward_features,
    
    sindy_weight=0,
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

path_gru = path_spice.replace('spice', 'gru')

gru = GRUModel(n_actions=dataset_train.n_actions, additional_inputs=dataset_train.n_additional_inputs, n_reward_features=0)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    epochs = 1000

    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=epochs,
        # batch_size=1024,
        scheduler=True,
        )

    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))
    

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation

# estimator.eval()
gru.eval().to(torch.device('cpu'))
estimator.eval()

print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    gru_model=gru,
))

print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    gru_model=gru,
))


# check for plausibility
with torch.no_grad():
    gru.eval()
    predictions_gru, _ = gru(dataset.xs)
    estimator.model.to(torch.device('cpu'))
    estimator.eval()
    predictions_spice = torch.tensor(estimator.predict(dataset.xs))
    estimator.use_sindy(False)
    predictions_spice_rnn = torch.tensor(estimator.predict(dataset.xs))
    
distance = dataset.xs[0, 10:50, 0, 2:3]
stay_probs_gru = torch.softmax(predictions_gru[0, 10:50, 0, :2], dim=-1)
stay_probs_spice = torch.softmax(predictions_spice[0, 10:50, 0, :2], dim=-1)
stay_probs_spice_rnn = torch.softmax(predictions_spice_rnn[0, 10:50, 0, :2], dim=-1)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1)
axs[0].plot(distance)
axs[0].plot(torch.zeros_like(distance)+10, '--r')
axs[0].set_title("distance in degrees between last laser and shield")
axs[1].plot(stay_probs_gru[..., 0], label='gru')
axs[1].plot(stay_probs_spice[..., 0], label='spice')
axs[1].plot(stay_probs_spice_rnn[..., 0], label='spice-rnn')
axs[1].set_title("Predicted stay probability")
plt.legend()
plt.show()


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

from weinhardt2026.studies.weber2024.archive.benchmarking_weber2024 import generate_behavior
from weinhardt2026.studies.weber2024.archive.analysis_generative_staymove import analysis_generative_behavior

output_dir = 'weinhardt2026/studies/weber2024/data'

# Generate behavior for GRU
gru.eval().to(torch.device('cpu'))
dataset_gen_gru = generate_behavior(
    model=gru,
    dataset=dataset,
    save_dataset=f'{output_dir}/gen_gru.csv',
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

# Generate behavior for SPICE-RNN
estimator.use_sindy(True)
estimator.eval()
dataset_gen_spice_rnn = generate_behavior(
    model=estimator,
    dataset=dataset,
    save_dataset=f'{output_dir}/gen_spice.csv',
)

# Generative analysis: compare real vs generated behavior
analysis_generative_behavior(
    path_data_real=path_data,
    path_data_gru=f'{output_dir}/gen_gru.csv',
    path_data_spice_rnn=f'{output_dir}/gen_spice_rnn.csv',
    path_data_spice=f'{output_dir}/gen_spice.csv',
    output_dir=output_dir,
)
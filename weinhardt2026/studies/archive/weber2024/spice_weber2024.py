import torch
import torch.nn as nn

from spice import SpiceEstimator, SpiceConfig, BaseModel, csv_to_dataset, SpiceDataset, split_data_along_sessiondim

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from weinhardt2026.utils.benchmarking_gru import GRUModel, training


train_spice = False
train_gru = False

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/archive/weber2024/data/weber2024.csv'
test_sessions = 2, 5, 9, 14

dataset = csv_to_dataset(
    file = path_data,
    df_participant_id='participant',
    df_experiment_id='experiment',
    df_choice='action',
    df_feedback=None,
    df_block='block',
    additional_inputs=[
        'laserRotation',
        'next_laserRotation',
        'trial_duration_frames',
        'shield_distance_initial',
        # 'reward_change',
        'total_movement_degrees',
        'frames_spent_moving',
        'button_press_onsets',
        'reaction_time_frames',
        'laser_caught',
        # 'hit_occurred',
        ],
)

if test_sessions is not None:
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, test_sessions)
else:
    dataset_train, dataset_test = dataset, None

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of experiments (baseline vs. infusion): {dataset_train.n_experiments}")
print(f"Number of actions in dataset: {dataset_train.n_actions}")
print(f"Number of additional inputs: {dataset_train.xs.shape[-1]-2*dataset_train.n_actions-3}")

# -------------------------------------------------------------------------------------------
# SPICE MODEL
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/archive/weber2024/params/spice_weber2024.pkl'

spice_config = SpiceConfig(
    library_setup={
        'mean_laser': ('laser_rotation',),       # between-trial: observed laser position
    },
    memory_state={
        'mean_laser': 0.,
    },
    states_in_logit=[],                          # logits produced by DDM, not sum-of-states
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=spice_config,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    n_experiments=dataset_train.n_experiments,

    epochs=500,
    warmup_steps=100,
    learning_rate=1e-3,
    device=device,

    verbose=True,
)

if train_spice:
    estimator.fit(
        data=dataset_train.xs,
        targets=dataset_train.ys,
        data_test=dataset_test.xs if dataset_test is not None else None,
        target_test=dataset_test.ys if dataset_test is not None else None,
    )
    estimator.save_spice(path_spice)


# -------------------------------------------------------------------------------------------
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

path_gru = path_spice.replace('spice', 'gru')

gru = GRUModel(n_actions=dataset_train.n_actions, additional_inputs=dataset_train.n_additional_inputs, n_reward_features=0)

if train_gru:
    criterion = nn.CrossEntropyLoss()
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

print(analysis_model_evaluation(
    dataset=dataset_train,
    # spice_model=estimator,
    gru_model=gru,
))

print(analysis_model_evaluation(
    dataset=dataset_test,
    # spice_model=estimator,
    gru_model=gru,
))


# check for plausibility
with torch.no_grad():
    predictions_gru, _ = gru(dataset.xs)
    print(torch.concat((dataset.xs[0, 5:15, 0, 5:6], predictions_gru[0, 5:15, 0, :2]), dim=-1))

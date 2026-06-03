import torch
import torch.nn as nn

from spice import SpiceEstimator, SpiceConfig, BaseModel, csv_to_dataset, SpiceDataset, split_data_along_sessiondim

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))
from weinhardt2026.utils.benchmarking_gru import GRUModel, training


train_spice = True
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
        'shield_distance_initial',
        'laserRotation',
        'trial_duration_frames',
        'total_movement_degrees',
        'frames_spent_moving',
        'button_press_onsets',
        'reaction_time_frames',
        'laser_caught',
        ],
)

# --------------------------------------------------------------------------
# make dataset much shorter for rapid prototyping
from spice import SpiceDataset
dataset = SpiceDataset(dataset.xs[:, :100], dataset.ys[:, :100], n_reward_features=0)
# --------------------------------------------------------------------------

if test_sessions is not None:
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, test_sessions)
else:
    dataset_train, dataset_test = dataset, None

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {dataset_train.n_participants}")
print(f"Number of experiments (2x2 design): {dataset_train.n_experiments}")
print(f"Number of actions in dataset: {dataset_train.n_actions}")
print(f"Number of additional inputs: {dataset_train.xs.shape[-1]-2*dataset_train.n_actions-3}")

# -------------------------------------------------------------------------------------------
# SPICE MODEL
# -------------------------------------------------------------------------------------------

path_spice = 'weinhardt2026/studies/archive/weber2024/params/spice_weber2024.pkl'

spice_config = SpiceConfig(
    library_setup={
        'mean_update':      ('laser_delta',),                                       # slow: tracks laser drift
        'distance_stay':    ('shield_distance', 'laser_drift', 'laser_caught'),     # fast: value update when staying
        'distance_move':    ('shield_distance', 'laser_drift', 'laser_caught'),     # fast: value update when moving (reset/adjust)
        'drift_stay':       ('ddistance',),                                         # depletion analog: distance rate of change
        'drift_move':       (),                                                     # learned reset after moving
        'perseveration':    ('action[t]',),                                         # choice stickiness
    },
    memory_state={
        'mean_laser': None,              # running laser drift estimate (auxiliary)
        'value_distance': None,          # distance-driven move value
        'value_drift': None,             # drift-driven move value (depletion analog)
        'value_perseveration': None,     # choice persistence
        'laserRotation[t-1]': None,      # working memory: previous laser position
        'shield_distance[t-1]': None,    # working memory: previous shield-laser distance
    },
    states_in_logit=['value_distance', 'value_drift', 'value_perseveration'],
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dropout = 0.1

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

        # Slow timescale: laser mean tracking
        self.setup_module(key_module='mean_update', input_size=1)

        # Fast timescale: distance-based value (like reward_patch in foraging)
        self.setup_module(key_module='distance_stay', input_size=3)
        self.setup_module(key_module='distance_move', input_size=3)

        # Drift: depletion analog (distance rate of change when staying)
        self.setup_module(key_module='drift_stay', input_size=1)
        self.setup_module(key_module='drift_move', input_size=0)

        # Choice perseveration
        self.setup_module(key_module='perseveration', input_size=1)

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Fixed mask: only update item 1 (move); item 0 (stay) kept at 0 as reference
        mask_move_value = torch.zeros_like(spice_signals.actions[0])
        mask_move_value[..., 1] = 1

        # Action indicators, expanded to item dimension [T, W, E, B, I]
        action_stay = spice_signals.actions[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        action_move = spice_signals.actions[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)

        # Additional inputs expanded to item dimension [T, W, E, B, I]
        # Indices: 0=shield_distance_initial, 1=laserRotation, 7=laser_caught
        shield_distance_raw = spice_signals.additional_inputs[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        laser_rotation_raw = spice_signals.additional_inputs[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        laser_caught_raw = spice_signals.additional_inputs[..., 7].unsqueeze(-1).expand_as(spice_signals.actions)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            stayed = action_stay[trial] * mask_move_value
            moved  = action_move[trial] * mask_move_value

            # --- Precompute signals ---

            # Laser delta: wrapped change in laser position, normalized to [-1, 1]
            laser_delta = laser_rotation_raw[trial] - self.state['laserRotation[t-1]']
            laser_delta = (laser_delta - 360 * torch.round(laser_delta / 360)) / 180

            # Shield distance, normalized to [0, 1]
            shield_distance = shield_distance_raw[trial] / 180

            # Laser caught (already binary)
            laser_caught = laser_caught_raw[trial]

            # Distance change (drift signal), normalized
            ddistance = (shield_distance_raw[trial] - self.state['shield_distance[t-1]']) / 180

            # --- 1. Mean tracking (slow timescale, not in logits) ---
            self.call_module(
                key_module='mean_update',
                key_state='mean_laser',
                action_mask=None,
                inputs=(laser_delta,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # Laser drift estimate from mean tracking
            laser_drift = self.state['mean_laser']

            # --- 2. Distance processing (fast timescale, like reward_patch in foraging) ---
            self.call_module(
                key_module='distance_stay',
                key_state='value_distance',
                action_mask=stayed,
                inputs=(shield_distance, laser_drift, laser_caught),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='distance_move',
                key_state='value_distance',
                action_mask=moved,
                inputs=(shield_distance, laser_drift, laser_caught),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- 3. Drift processing (depletion analog) ---
            self.call_module(
                key_module='drift_stay',
                key_state='value_drift',
                action_mask=stayed,
                inputs=(ddistance,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='drift_move',
                key_state='value_drift',
                action_mask=moved,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- 4. Perseveration ---
            self.call_module(
                key_module='perseveration',
                key_state='value_perseveration',
                action_mask=None,
                inputs=(spice_signals.actions[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- 5. Working memory updates ---
            self.state['laserRotation[t-1]'] = laser_rotation_raw[trial]
            self.state['shield_distance[t-1]'] = shield_distance_raw[trial]

            # --- 6. Logit computation ---
            spice_signals.logits[trial] = (
                + self.state['value_distance']
                + self.state['value_drift']
                + self.state['value_perseveration']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()


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
    n_reward_features=dataset_train.n_reward_features,
    
    sindy_weight=0,
    epochs=1000,
    warmup_steps=500,
    ensemble_size=1,
    
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
# else:
#     estimator.load_spice(path_spice)

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
axs[0].set_title("distance in degrees between last laser and shield")
axs[1].plot(stay_probs_gru[..., 0], label='gru')
axs[1].plot(stay_probs_spice[..., 0], label='spice')
axs[1].plot(stay_probs_spice_rnn[..., 0], label='spice-rnn')
axs[1].set_title("Predicted stay probability")
plt.legend()
plt.show()
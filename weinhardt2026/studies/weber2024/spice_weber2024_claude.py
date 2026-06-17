import torch
import math

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        'mean_update':      ('laser_trig',),                                              # slow: sin/cos EMA tracker for laser mean
        'distance_stay':    ('shield_laser_distance', 'shield_mean_distance', 'laser_caught'),  # fast: value update when staying
        'distance_move':    ('shield_laser_distance', 'shield_mean_distance', 'laser_caught'),  # fast: value update when moving
        'drift_stay':       ('ddistance',),                                               # depletion analog: distance rate of change
        'drift_move':       (),                                                           # learned reset after moving
        'perseveration':    ('action[t]',),                                               # choice stickiness
    },
    memory_state={
        # environmental estimate (sin/cos components of mean laser position)
        'mean_sin': 0,
        'mean_cos': 0,

        # cognitive variables
        'value_distance': None,          # distance-driven move value
        'value_drift': None,             # drift-driven move value (depletion analog)
        'value_perseveration': None,     # choice persistence

        # working memory
        'shield_laser_distance[t-1]': 0, # previous trig distance for drift computation
    },
    states_in_logit=['mean_sin', 'mean_cos', 'value_distance', 'value_drift', 'value_perseveration'],
    additional_inputs=(
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
    ),
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dropout = 0.1

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size-2,
            dropout=self.dropout,
        )
        
        self.experiment_embedding = self.setup_embedding(
            num_embeddings=self.n_experiments,
            embedding_size=2,
            dropout=self.dropout,
        )

        # Slow timescale: laser mean tracking (shared module for sin and cos)
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

        # Precompute sin/cos of laser and shield positions (degrees → radians → trig)
        laser_rad = spice_signals.additional_inputs['laserRotation'] * (math.pi / 180)
        sin_laser = torch.sin(laser_rad)
        cos_laser = torch.cos(laser_rad)

        shield_rad = spice_signals.additional_inputs['shieldRotation'] * (math.pi / 180)
        sin_shield = torch.sin(shield_rad)
        cos_shield = torch.cos(shield_rad)

        # Initialize mean to first laser observation
        if prev_state is None:
            self.state['mean_sin'] = self.state['mean_sin'] + sin_laser[0]
            self.state['mean_cos'] = self.state['mean_cos'] + cos_laser[0]

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids)
        
        for trial in spice_signals.trials:

            stayed = action_stay[trial] * mask_move_value
            moved  = action_move[trial] * mask_move_value

            # --- Mean tracking (sin/cos EMA via shared module) ---
            self.call_module(
                key_module='mean_update',
                key_state='mean_sin',
                action_mask=None,
                inputs=(sin_laser[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='mean_update',
                key_state='mean_cos',
                action_mask=None,
                inputs=(cos_laser[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            # --- Precompute distance signals (all via trig identity, in [-1, 1]) ---

            # sin(shield - laser): distance to last beam
            shield_laser_distance = (
                sin_shield[trial] * cos_laser[trial]
                - cos_shield[trial] * sin_laser[trial]
            )

            # sin(shield - mean): distance to estimated mean
            shield_mean_distance = (
                sin_shield[trial] * self.state['mean_cos']
                - cos_shield[trial] * self.state['mean_sin']
            )

            # Distance rate of change (drift signal)
            ddistance = shield_laser_distance - self.state['shield_laser_distance[t-1]']

            # --- Distance processing (fast timescale, like reward_patch in foraging) ---
            self.call_module(
                key_module='distance_stay',
                key_state='value_distance',
                action_mask=stayed,
                inputs=(shield_laser_distance, shield_mean_distance, spice_signals.additional_inputs['laser_caught'][trial]),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='distance_move',
                key_state='value_distance',
                action_mask=moved,
                inputs=(shield_laser_distance, shield_mean_distance, spice_signals.additional_inputs['laser_caught'][trial]),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            # --- Drift processing (depletion analog) ---
            self.call_module(
                key_module='drift_stay',
                key_state='value_drift',
                action_mask=stayed,
                inputs=(ddistance,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='drift_move',
                key_state='value_drift',
                action_mask=moved,
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            # --- Perseveration ---
            self.call_module(
                key_module='perseveration',
                key_state='value_perseveration',
                action_mask=None,
                inputs=(spice_signals.actions[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
            )

            # --- Working memory updates ---
            self.state['shield_laser_distance[t-1]'] = shield_laser_distance

            # --- Logit computation ---
            spice_signals.logits[trial] = (
                + self.state['value_distance']
                + self.state['value_drift']
                + self.state['value_perseveration']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

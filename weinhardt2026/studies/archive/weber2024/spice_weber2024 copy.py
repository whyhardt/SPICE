import torch
import torch.nn as nn

from spice import SpiceEstimator, SpiceConfig, BaseModel, csv_to_dataset, SpiceDataset, split_data_along_sessiondim


CONFIG = SpiceConfig(
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
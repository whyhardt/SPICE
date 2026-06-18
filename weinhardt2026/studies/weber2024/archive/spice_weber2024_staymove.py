import torch
import math

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        'mean_update': ('laser_trig',),                # EMA-like tracker for sin/cos of laser mean
        'update_distance': (
            'shield_mean_distance', 
            'shield_laser_distance',
            ),
        'update_action': (
            'stayed', 
            'moved',
            ),
        'update_bias': (
            'trial_duration_frames',
            'total_movement_degrees',
            'frames_spent_moving',
            'button_press_onsets',
            'reaction_time_frames',
            'laser_caught',
        ),
    },
    
    memory_state={
        # environmental estimate (sin/cos components of mean laser position)
        'mean_sin': 0,
        'mean_cos': 0,

        # cognitive variables
        'value_distance': 0,
        'value_action': 0,
        'value_bias': 0,
    },

    states_in_logit=[
        'mean_sin',
        'mean_cos',
        'value_distance',
        'value_action',
        'value_bias',
        ],
    
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

        self.experiment_embedding = self.setup_embedding(num_embeddings=self.n_experiments, embedding_size=2, dropout=self.dropout)
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=self.dropout)#, n_additional_inputs=2)

        # Mean tracking: shared module applied to sin and cos components separately
        self.setup_module(key_module='mean_update', input_size=1)
        self.setup_module(key_module='update_distance', input_size=2)  # NOTE: perhaps split mean-distance and last-laser-distance into two different modules/values
        self.setup_module(key_module='update_action', input_size=2)
        self.setup_module(key_module='update_bias', input_size=6, polynomial_degree=1)

        # for ddm:
        # add new module for e.g. drift with inputs value_{distance, action, bias}
        
        
    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Fixed mask: only update item 1 (move); item 0 (stay) kept at 0 as reference
        mask_move_value = torch.zeros_like(spice_signals.actions[0])
        mask_move_value[..., 1] = 1

        stayed = spice_signals.actions[..., 0].unsqueeze(-1).expand_as(spice_signals.actions)
        moved = spice_signals.actions[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        
        # Precompute sin/cos of laser and shield positions (degrees → radians → trig)
        laser_rad = spice_signals.additional_inputs['laserRotation'] * (math.pi / 180)
        sin_laser = torch.sin(laser_rad)
        cos_laser = torch.cos(laser_rad)

        shield_rad = spice_signals.additional_inputs['shieldRotation'] * (math.pi / 180)
        sin_shield = torch.sin(shield_rad)
        cos_shield = torch.cos(shield_rad)
        
        movement_rad = spice_signals.additional_inputs['total_movement_degrees'] * (math.pi / 180)
        # sin_movement = torch.sin(movement_rad)
        # cos_movement = torch.cos(movement_rad)
        
        # Initialize mean to first laser observation
        if prev_state is None:
            self.state['mean_sin'] = self.state['mean_sin'] + sin_laser[0]
            self.state['mean_cos'] = self.state['mean_cos'] + cos_laser[0]

        # additional inputs for participant embedding
        participant_embedding = self.participant_embedding(
            spice_signals.participant_ids,
            (spice_signals.additional_inputs['volatility'], spice_signals.additional_inputs['stochasticity']),
        )
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None

        for trial in spice_signals.trials:

            # --- Mean tracking (sin/cos EMA via shared module) ---
            self.call_module(
                key_module='mean_update',
                key_state='mean_sin',
                action_mask=None,
                inputs=(sin_laser[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding
            )

            self.call_module(
                key_module='mean_update',
                key_state='mean_cos',
                action_mask=None,
                inputs=(cos_laser[trial],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding
            )

            # --- Distance: sin(shield - mean) via trig identity ---
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            shield_mean_distance = (
                sin_shield[trial] * self.state['mean_cos']
                - cos_shield[trial] * self.state['mean_sin']
            )

            # --- Value update ---
            # Direct beam distance: sin(shield - laser) via trig identity
            shield_laser_distance = (
                sin_shield[trial] * cos_laser[trial]
                - cos_shield[trial] * sin_laser[trial]
            )

            self.call_module(
                key_module='update_distance',
                key_state='value_distance',
                action_mask=mask_move_value,
                inputs=(
                    shield_mean_distance, 
                    shield_laser_distance,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding
            )
            
            self.call_module(
                key_module='update_action',
                key_state='value_action',
                action_mask=mask_move_value,
                inputs=(
                    stayed[trial], 
                    moved[trial],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding
            )
            
            self.call_module(
                key_module='update_bias',
                key_state='value_bias',
                action_mask=mask_move_value,
                inputs=(
                    spice_signals.additional_inputs['trial_duration_frames'][trial],
                    movement_rad[trial],
                    spice_signals.additional_inputs['frames_spent_moving'][trial],
                    spice_signals.additional_inputs['button_press_onsets'][trial],
                    spice_signals.additional_inputs['reaction_time_frames'][trial],
                    spice_signals.additional_inputs['laser_caught'][trial],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding
            )

            # --- Logit computation ---
            spice_signals.logits[trial] = (
                + self.state['value_distance']
                + self.state['value_action']
                + self.state['value_bias']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

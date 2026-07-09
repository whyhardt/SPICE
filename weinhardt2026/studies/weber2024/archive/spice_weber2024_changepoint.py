import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Changepoint LR: update when CP detected, decay when stable
        'changepoint_update': ('laser_caught',),
        'changepoint_decay': ('laser_caught',),
        # Uncertainty LR: update when stable, decay when CP detected
        'uncertainty_update': ('laser_caught',),
        'uncertainty_decay': ('laser_caught',),
        # Belief update: learns f(PE) without alpha constraint (gated output architecture)
        'belief_update_caught': ('prediction_error',),
        'belief_update_missed': ('prediction_error',),
    },

    memory_state={
        'belief_value': 0,           # internal belief about laser position (sin/cos)
        'changepoint_value': 0,   # changepoint-driven learning rate (high alpha on changepoints)
        'uncertainty_value': 0,   # uncertainty-driven learning rate (baseline alpha)
    },

    states_in_logit=[
        'belief_value',
        'changepoint_value',
        'uncertainty_value',
        ],

    additional_inputs=(
        'laser_caught',
        'volatility',
        'stochasticity',
        'trial_duration_frames',
        'trueMean',
    ),
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shield_width = 20/360
        
        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals
        shield = spice_signals.actions       # shield position (sin, cos)
        laser = spice_signals.feedback       # laser position (sin, cos)

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Initialize belief to first laser observation
        if prev_state is None:
            self.state['belief_value'] = self.state['belief_value'] + laser[0]

        for trial in spice_signals.trials:

            # --- Prediction error ---
            prediction_error = laser[trial] - self.state['belief_value']

            # --- Catch mask ---
            caught = spice_signals.additional_inputs['laser_caught'][trial]
            caught_mask = caught.expand_as(self.state['belief_value'])

            # --- Big PE mask (like in the model for bruckner2025) ---
            mask_big_pe = (prediction_error.abs() > 3 * self.shield_width / 2).float()

            # --- Changepoint LR: update when CP detected (p > 0.5), decay when stable ---
            self.call_module(
                key_module='changepoint_update',
                key_state='changepoint_value',
                action_mask=mask_big_pe,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='changepoint_decay',
                key_state='changepoint_value',
                action_mask=1 - mask_big_pe,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Uncertainty LR: update when stable (p <= 0.5), decay when CP detected ---
            self.call_module(
                key_module='uncertainty_update',
                key_state='uncertainty_value',
                action_mask=1 - mask_big_pe,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='uncertainty_decay',
                key_state='uncertainty_value',
                action_mask=mask_big_pe,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Belief update: learns f(PE) without alpha constraint ---
            self.call_module(
                key_module='belief_update_caught',
                key_state='belief_value',
                action_mask=caught_mask,
                inputs=(prediction_error,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='belief_update_missed',
                key_state='belief_value',
                action_mask=1 - caught_mask,
                inputs=(prediction_error,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Composite learning rate: weighted by changepoint probability ---
            alpha_cp = torch.sigmoid(self.state['changepoint_value'])
            alpha_var = torch.sigmoid(self.state['uncertainty_value'])
            alpha = mask_big_pe * alpha_cp + (1 - mask_big_pe) * alpha_var
            
            # --- Gated output: interpolate between shield and belief ---
            # alpha ∈ [0, 1]; 0 = trust current position, 1 = trust internal belief
            spice_signals.logits[trial] = (1 - alpha) * shield[trial] + alpha * self.state['belief_value']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

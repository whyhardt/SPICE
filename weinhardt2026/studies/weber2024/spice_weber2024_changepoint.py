import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Changepoint detection: learns to detect when PE magnitude signals a changepoint
        'changepoint_detection': ('prediction_error',),
        # Changepoint LR: update when CP detected, decay when stable
        'changepoint_lr_update': ('laser_caught',),
        'changepoint_lr_decay': ('laser_caught',),
        # Uncertainty LR: update when stable, decay when CP detected
        'uncertainty_lr_update': ('laser_caught',),
        'uncertainty_lr_decay': ('laser_caught',),
        # Belief update: learns f(PE) without alpha constraint (gated output architecture)
        'belief_update_caught': ('prediction_error',),
        'belief_update_missed': ('prediction_error',),
    },

    memory_state={
        'belief_value': 0,           # internal belief about laser position (sin/cos)
        'changepoint_value': 0,      # changepoint probability logit; sigmoid(0) = 0.5
        'changepoint_lr_value': 3,   # changepoint-driven learning rate (high alpha on changepoints)
        'uncertainty_lr_value': 3,   # uncertainty-driven learning rate (baseline alpha)
    },

    states_in_logit=['belief_value'],

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

            # --- Changepoint detection: learns to map PE magnitude to changepoint probability ---
            self.call_module(
                key_module='changepoint_detection',
                key_state='changepoint_value',
                action_mask=None,  # always updates
                inputs=(prediction_error,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            cp_prob = torch.sigmoid(self.state['changepoint_value'])

            # --- CP mask: binary threshold at 0.5 ---
            mask_cp = (cp_prob > 0.5).float()

            # --- Changepoint LR: update when CP detected (p > 0.5), decay when stable ---
            self.call_module(
                key_module='changepoint_lr_update',
                key_state='changepoint_lr_value',
                action_mask=mask_cp,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='changepoint_lr_decay',
                key_state='changepoint_lr_value',
                action_mask=1 - mask_cp,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Uncertainty LR: update when stable (p <= 0.5), decay when CP detected ---
            self.call_module(
                key_module='uncertainty_lr_update',
                key_state='uncertainty_lr_value',
                action_mask=1 - mask_cp,
                inputs=(caught,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='uncertainty_lr_decay',
                key_state='uncertainty_lr_value',
                action_mask=mask_cp,
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
            changepoint_lr = torch.sigmoid(self.state['changepoint_lr_value'])
            uncertainty_lr = torch.sigmoid(self.state['uncertainty_lr_value'])
            alpha = cp_prob * changepoint_lr + (1 - cp_prob) * uncertainty_lr

            # --- Gated output: interpolate between shield and belief ---
            # alpha ∈ [0, 1]; 0 = trust current position, 1 = trust internal belief
            spice_signals.logits[trial] = (1 - alpha) * shield[trial] + alpha * self.state['belief_value']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

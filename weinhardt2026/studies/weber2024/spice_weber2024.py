import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Belief update: split by catch outcome (externalized gating)
        'belief_update_caught': ('pe',),   # update when shield catches laser
        'belief_update_missed': ('pe',),   # update when shield misses laser
        # Dynamic learning rate: modulates gated output
        'certainty_update_caught': (),    # LR adapts when catching (tracking well)
        'certainty_update_missed':  (),    # LR decays when missing (tracking poorly)
    },

    memory_state={
        'belief_value': 0,    # internal belief about laser position (sin/cos as items 0, 1)
        'certainty_raw': 0,        # dynamic learning rate state; sigmoid(3) = 1.0
    },

    states_in_logit=[
        'belief_value', 
        'certainty_raw',
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

        # self.experiment_embedding = self.setup_embedding(
        #     num_embeddings=self.n_experiments,
        #     embedding_size=2,
        # )

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
        )
        
        self.alpha_raw = self.setup_constant()
    
    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, 2)
        # actions:  shield position (sin, cos) at trial t
        # rewards:  laser position (sin, cos) at trial t
        shield = spice_signals.actions
        laser = spice_signals.feedback

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Initialize belief to first laser observation
        if prev_state is None:
            self.state['belief_value'] = self.state['belief_value'] + laser[0]

        for trial in spice_signals.trials:

            # --- Prediction error: laser minus belief (per sin/cos component) ---
            prediction_error = laser[trial] - self.state['belief_value']

            # --- Catch mask: externalized binary gating ---
            caught = spice_signals.additional_inputs['laser_caught'][trial]
            caught_mask = caught.expand_as(self.state['belief_value'])

            # --- Dynamic learning rate ---
            self.call_module(
                key_module='certainty_update_caught',
                key_state='certainty_raw',
                action_mask=caught_mask,
                # inputs=(prediction_error.detach(),),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='certainty_update_missed',
                key_state='certainty_raw',
                action_mask=1 - caught_mask,
                # inputs=(prediction_error.detach(),),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Gated output: shield_t + alpha * (belief - shield_t) ---
            # alpha ∈ [0, 1] via sigmoid; interpolates between current position and belief
            certainty_value = torch.sigmoid(self.state['certainty_raw'])
            
            # --- Belief update: split by catch outcome ---
            self.call_module(
                key_module='belief_update_caught',
                key_state='belief_value',
                action_mask=caught_mask,
                inputs=(
                    prediction_error,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='belief_update_missed',
                key_state='belief_value',
                action_mask=1 - caught_mask,
                inputs=(
                    prediction_error,
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            
            # E_idx = torch.arange(self.ensemble_size, device=self.device).unsqueeze(1)
            # alpha = torch.sigmoid(
            #     self.alpha_raw[E_idx, spice_signals.participant_ids, spice_signals.experiment_ids]
            # ).expand(-1, -1, self.n_items)
            spice_signals.logits[trial] = (1-certainty_value) * shield[trial] + certainty_value * self.state['belief_value']
            # spice_signals.logits[trial] = self.state['belief_value']
            
        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

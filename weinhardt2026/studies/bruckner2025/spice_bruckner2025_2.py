import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Learning rate: learns adaptive alpha from PE and catch flag
        'lr_update': (
            'prediction_error',
            'catch',
            ),
        'lr_update_catchtrial': (
            # 'prediction_error',
            # 'catch',
            ),
        # Belief update: learns target position from prediction error
        'belief_update_catch': (
            'prediction_error',
            ),
        'belief_update_miss': (
            'prediction_error',
            ),
        # Anchoring bias: learns compensation for bucket displacement
        'anchor_update': (
            'anchor_shift',
            ),
    },

    memory_state={
        'belief_value': 0.5,      # initial belief = center of screen (normalized)
        'lr_value': None,         # learning rate state (sigmoid(0) = 0.5)
        'anchor_value': 0,        # per-trial anchoring correction (reset each trial)
    },

    states_in_logit=[
        'belief_value',
        'lr_value',
        'anchor_value',
    ],
    
    additional_inputs=(
        'z_next',
        'catch',
        'v_t',
        'sigma',
        'r_t',
        'mu_t',
        'c_t',
    ),
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)
        
        # setup customized modules
        # anchor_update: 1 control signal (anchor_shift); no state in library (always reset to 0)
        self.setup_module(key_module='anchor_update', dropout=self.dropout, include_state=False)
        
    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, ...)
        # actions:            b_t / 300   → (T, W, E, B, 1)
        # rewards:            x_t / 300   → (T, W, E, B, 1)
        outcome = spice_signals.feedback                                 # x_t / 300
        bucket = spice_signals.actions                                  # b_t / 300

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:
            
            # --- set belief value to new bucket position ---
            self.state['belief_value'] = bucket[trial]
            
            # --- Prediction error: outcome minus internal belief ---
            prediction_error = outcome[trial] - self.state['belief_value']

            # --- Anchor shift: bucket displacement from push ---
            anchor_shift = spice_signals.additional_inputs['z_next'][trial] - bucket[trial]
            
            # --- Masks ---
            mask_catch = spice_signals.additional_inputs['catch'][trial]
            mask_vt = spice_signals.additional_inputs['v_t'][trial]

            # --- Learning rate: learns adaptive alpha from PE and catch ---
            self.call_module(
                key_module='lr_update',
                key_state='lr_value',
                action_mask=1 - mask_vt,
                inputs=(
                    prediction_error.detach(),
                    spice_signals.additional_inputs['catch'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            
            self.call_module(
                key_module='lr_update_catchtrial',
                key_state='lr_value',
                action_mask=mask_vt,
                inputs=(
                    # prediction_error.detach(),
                    # spice_signals.additional_inputs['catch'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            alpha = torch.sigmoid(self.state['lr_value'])
            
            # --- Belief update: learns where the helicopter is (non-visible trials only) ---
            self.call_module(
                key_module='belief_update_catch',
                key_state='belief_value',
                action_mask=mask_catch * (1 - mask_vt),
                inputs=(
                    prediction_error,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='belief_update_miss',
                key_state='belief_value',
                action_mask=(1 - mask_catch) * (1 - mask_vt),
                inputs=(
                    prediction_error,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # On visible trials (v_t=1), hard-set belief to true helicopter position
            mu_t = spice_signals.additional_inputs['mu_t'][trial]
            self.state['belief_value'] = torch.where(mask_vt > 0.5, mu_t, self.state['belief_value'])
            
            # --- Anchor correction: reset each trial, learns d * anchor_shift ---
            self.state['anchor_value'] = torch.zeros_like(self.state['anchor_value'])
            self.call_module(
                key_module='anchor_update',
                key_state='anchor_value',
                action_mask=None,
                inputs=(anchor_shift,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            
            # --- Gated output + anchor correction ---
            spice_signals.logits[trial] = (
                + (1-alpha) * bucket[trial] 
                + alpha * self.state['belief_value']
                + self.state['anchor_value']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

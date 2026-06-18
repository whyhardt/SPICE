import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Belief update: learns target position from prediction error
        'belief_update': (
            'prediction_error', 
            'catch', 
            'v_t',
            ),
        # Changepoint detection: learns omega (surprise-driven) from PE magnitude
        'changepoint_lr_update': (
            # 'prediction_error', 
            'catch', 
            'v_t',
            ),
        'changepoint_lr_decay': (),
        # Base learning rate: learns tau (condition-dependent baseline)
        'uncertainty_lr_update': (
            # 'prediction_error', 
            'catch', 
            'v_t',
            ),
        'uncertainty_lr_decay': (),
        # Anchoring bias: learns compensation for bucket displacement
        'anchor_update': (
            'anchor_shift',
            ),
    },

    memory_state={
        'belief_value': 0.5,      # initial belief = center of screen (normalized)
        'surprise_value': None,       # changepoint probability state (sigmoid(0) = 0.5)
        'uncertainty_value': None,         # base learning rate state (sigmoid(0) = 0.5)
        'anchor_value': 0,      # per-trial anchoring correction (reset each trial)
    },
    
    states_in_logit=[
        'belief_value',
        'surprise_value',
        'uncertainty_value',
        'anchor_value',
    ],
    
    additional_inputs=(
        'z_next',
        'catch',
        'v_t',
        'sigma',
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

            # --- Prediction error: outcome minus internal belief ---
            prediction_error = outcome[trial] - self.state['belief_value']

            # --- Anchor shift: bucket displacement from push ---
            anchor_shift = spice_signals.additional_inputs['z_next'][trial] - bucket[trial]
            
            # --- PE masks: small PE -> uncertainty; big PE -> CP ---
            # Binary: 1 if |PE| exceeds threshold, 0 otherwise
            # sigma equals bucket width -> therefore no latent task variable but observable
            mask_pe_big = (prediction_error.abs() > 3 * spice_signals.additional_inputs['sigma'][trial]).float()

            # --- Belief update: learns where the helicopter is ---
            self.call_module(
                key_module='belief_update',
                key_state='belief_value',
                action_mask=None,
                inputs=(
                    prediction_error,
                    spice_signals.additional_inputs['catch'][trial],
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Changepoint: learns surprise-driven update rate (omega) ---
            self.call_module(
                key_module='changepoint_lr_update',
                key_state='surprise_value',
                action_mask=mask_pe_big,
                inputs=(
                    # prediction_error.detach(),
                    spice_signals.additional_inputs['catch'][trial],
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            
            self.call_module(
                key_module='changepoint_lr_decay',
                key_state='surprise_value',
                action_mask=1-mask_pe_big,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Learning rate: learns condition-dependent base rate (tau) ---
            self.call_module(
                key_module='uncertainty_lr_update',
                key_state='uncertainty_value',
                action_mask=1-mask_pe_big,
                inputs=(
                    # prediction_error.detach(),
                    spice_signals.additional_inputs['catch'][trial],
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )
            
            self.call_module(
                key_module='uncertainty_lr_decay',
                key_state='uncertainty_value',
                action_mask=mask_pe_big,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

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

            # --- Combined learning rate: alpha = omega + tau - omega*tau ---
            surprise_lr = torch.sigmoid(self.state['surprise_value'])
            uncertainty_lr = torch.sigmoid(self.state['uncertainty_value'])
            # alpha = surprise_lr + uncertainty_lr - surprise_lr * uncertainty_lr  # = 1 - (1-omega)(1-tau), ∈ [0, 1]
            alpha = mask_pe_big * surprise_lr + (1-mask_pe_big) * uncertainty_lr
            
            # --- Gated output + anchor correction ---
            spice_signals.logits[trial] = (
                bucket[trial] + alpha * (self.state['belief_value'] - bucket[trial])
                + self.state['anchor_value']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

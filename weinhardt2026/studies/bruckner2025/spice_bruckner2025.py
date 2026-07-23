import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Changepoint LR: update on big-PE trials, decay on small-PE
        'changepoint_update': ('v_t',),
        'changepoint_decay': ('catch', 'v_t',),
        # Uncertainty LR: update on small-PE trials, decay on big-PE
        'uncertainty_update': ('catch', 'v_t',),
        'uncertainty_decay': ('v_t',),
        # Belief update: learns target position from prediction error
        'belief_update_catch': ('prediction_error',),
        'belief_update_miss': ('prediction_error',),
        # Anchoring bias: learns compensation for bucket displacement
        'anchor_update': ('anchor_shift',),
    },

    memory_state={
        'changepoint_value': 0,    # changepoint LR state (sigmoid(3) ≈ 0.95)
        'uncertainty_value': 0,    # relative uncertainty LR state (sigmoid(3) ≈ 0.95)
        'belief_value': 0.5,      # initial belief = center of screen (normalized)
        'anchor_value': 0,        # per-trial anchoring correction (reset each trial)
    },

    states_in_logit=[
        'changepoint_value',
        'uncertainty_value',
        'belief_value',
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
        self.setup_module(key_module='anchor_update', include_state=False)

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, ...)
        outcome = spice_signals.feedback                                 # x_t / 300
        bucket = spice_signals.actions                                  # b_t / 300

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:
            
            # --- Fix belief value to current position ---
            self.state['belief_value'] = bucket[trial]
            
            # --- Prediction error: outcome minus internal belief ---
            prediction_error = outcome[trial] - self.state['belief_value']

            # --- Anchor shift: bucket displacement from push ---
            anchor_shift = spice_signals.additional_inputs['z_next'][trial] - bucket[trial]

            # --- Masks ---
            mask_catch = spice_signals.additional_inputs['catch'][trial]
            mask_vt = spice_signals.additional_inputs['v_t'][trial]

            # --- PE mask: |PE| > 3 bucket half-widths → changepoint; else → uncertainty ---
            mask_pe_big = (prediction_error.abs() > 3 * spice_signals.additional_inputs['sigma'][trial] / 2).float()

            # --- Changepoint LR: update on big-PE trials, decay on small-PE ---
            self.call_module(
                key_module='changepoint_update',
                key_state='changepoint_value',
                action_mask=mask_pe_big,
                inputs=(
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='changepoint_decay',
                key_state='changepoint_value',
                action_mask=1 - mask_pe_big,
                inputs=(
                    spice_signals.additional_inputs['catch'][trial],
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Uncertainty LR: update on small-PE trials, decay on big-PE ---
            self.call_module(
                key_module='uncertainty_update',
                key_state='uncertainty_value',
                action_mask=1 - mask_pe_big,
                inputs=(
                    spice_signals.additional_inputs['catch'][trial],
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            self.call_module(
                key_module='uncertainty_decay',
                key_state='uncertainty_value',
                action_mask=mask_pe_big,
                inputs=(
                    spice_signals.additional_inputs['v_t'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Hard switch: big PE → use omega, small PE → use tau ---
            alpha_cp = torch.sigmoid(self.state['changepoint_value'])
            alpha_var = torch.sigmoid(self.state['uncertainty_value'])
            alpha = mask_pe_big * alpha_cp + (1 - mask_pe_big) * alpha_var

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
                + (1 - alpha) * bucket[trial]           # how much to trust current position
                + alpha * self.state['belief_value']    # how much to trust internal belief
                + self.state['anchor_value']            # anchor bias
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

"""Delta-rule variant of the weber2024 model (bruckner2025-style): one learned learning rate on the shield itself, no
latent belief and no movement gain.

    PE_t       = wrap(laser_t - shield_t)                                  (degrees)
    shield_t+1 = shield_t + alpha_t * PE_t,     alpha_t = sigmoid(learning_rate_value_t)

As in spice_bruckner2025.py, the learning rate carries all dynamics (surprise relative to the shield, catch and the
block's volatility/stochasticity, recurrent), so its equation is the only place where the gain of the update lives:
the belief-learning and movement gains of spice_weber2024.py, which are not separable from the shield alone, collapse
into one. The physical movement limit stays in the loss (clamped_angular_mse).
"""

import torch

from spice import SpiceConfig, BaseModel

from studies.weber2024.spice_weber2024 import (  # shared data handling, loss and angle helpers
    BINARY_SIGNALS, ESTIMATOR_KWARGS, LOSS_FN, LOSS_FN_KWARGS, NORMALIZE_REWARDS, PE_SCALE,
    degrees_to_sincos, prepare_dataframe, prepare_dataset, sincos_to_degrees, wrap_degrees,
)


CONFIG = SpiceConfig(
    discovered_modules={
        # Learning rate: updated every beam from surprise (|PE| to the shield, scaled), catch and the block's conditions
        'learning_rate_update': ('|prediction_error|', 'caught', 'volatility', 'stochasticity'),
    },

    memory_state={
        'learning_rate_value': 0,   # alpha = sigmoid(value), initially 0.5
    },


    additional_inputs=(
        'laser_caught',             # scored on the beam's onset frame (data_raw/data_processing.py)
        'volatility',
        'stochasticity',
        'trial_duration_frames',
        'trueMean',                 # plotting only (weinhardt2026/studies/weber2024/weber2024.py); never a model input
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

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant for binary indicators.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            binary_signals = [s for s in self.spice_config.discovered_modules[module] if s in BINARY_SIGNALS]
            for index_term, term in enumerate(candidate_terms[module]):
                factors = term.split('*')
                if any(signal + '^2' in factors for signal in binary_signals):
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Positions as angles (degrees) — shapes after init: (T, W, E, B, 1)
        shield = sincos_to_degrees(spice_signals.actions)
        laser = sincos_to_degrees(spice_signals.feedback)

        # Embeddings
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- Prediction error: shortest arc from the shield to the beam ---
            prediction_error = wrap_degrees(laser[trial] - shield[trial])

            # --- Learning rate: updated every beam from surprise |PE| (scaled to [0, 1]) ---
            self.call_module(
                key_module='learning_rate_update',
                key_state='learning_rate_value',
                inputs=(
                    (prediction_error.abs() / PE_SCALE).clamp(max=1.),
                    spice_signals.additional_inputs['laser_caught'][trial],
                    spice_signals.additional_inputs['volatility'][trial],
                    spice_signals.additional_inputs['stochasticity'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            alpha = torch.sigmoid(self.state['learning_rate_value'])

            # --- Next shield position: delta rule on the shield (speed clamp in the loss) ---
            spice_signals.logits[trial] = degrees_to_sincos(shield[trial] + alpha * prediction_error)

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

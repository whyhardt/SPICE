"""Recurrent-movement variant of the weber2024 model: spice_weber2024.py with a movement gain that remembers the
hold/move phase.

    PE_t       = wrap(laser_t - belief_t)                                  (degrees)
    belief_t+1 = belief_t + alpha_t * PE_t,     alpha_t = sigmoid(learning_rate_value_t)
    shield_t+1 = shield_t + g_t * wrap(belief_t+1 - shield_t),  g_t = sigmoid(movement_value_t)

In spice_weber2024.py the movement gain is stateless with the belief-shield distance as its only input. Participants
hold the shield on ~74% of the beams and move in runs; a point prediction scored by squared error then settles on the
mean over holds and moves, a gain that never drops below ~0.1 (the shield always predicted to move a little). Here
the movement gain is recurrent and also sees the shield's last move (as a fraction of the largest possible move,
0 = held, 1 = full speed) and the catch, so it can tell the phases apart: low in a hold, high during a move.
"""

import torch

from spice import SpiceConfig, BaseModel

from studies.weber2024.spice_weber2024 import (  # shared data handling, loss and angle helpers
    BINARY_SIGNALS, DISTANCE_SCALE, ESTIMATOR_KWARGS, LOSS_FN, LOSS_FN_KWARGS, MOVEMENT_SPEED, NORMALIZE_REWARDS,
    PE_SCALE, degrees_to_sincos, prepare_dataframe, prepare_dataset, sincos_to_degrees, wrap_degrees,
)


CONFIG = SpiceConfig(
    library_setup={
        # Learning rate: updated every beam from surprise (|PE|, scaled), catch and the block's conditions
        'learning_rate_update': ('|prediction_error|', 'caught', 'volatility', 'stochasticity'),
        # Movement gain: recurrent; belief-shield distance, the shield's last move and the catch
        'movement_update': ('|belief_shield_distance|', 'move_fraction', 'caught'),
    },

    memory_state={
        'belief_value': 0,          # belief about the laser position (degrees); set to the first beam
        'learning_rate_value': 0,   # alpha = sigmoid(value), initially 0.5
        'movement_value': 0,        # movement gain, g = sigmoid(value), initially 0.5; carried over between beams
        'shield_value': 0,          # previous shield position (degrees), for the last move
        'interval_value': 1,        # previous inter-beam interval (frames), for the largest possible last move
    },

    states_in_logit=[
        'learning_rate_value',
        'movement_value',
    ],

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
            binary_signals = [s for s in self.spice_config.library_setup[module] if s in BINARY_SIGNALS]
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
        interval = spice_signals.additional_inputs['trial_duration_frames']
        caught = spice_signals.additional_inputs['laser_caught']

        # Embeddings
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Initialize the belief to the first beam, the previous shield to the first shield (no last move)
        if prev_state is None:
            self.state['belief_value'] = laser[0].clone()
            self.state['shield_value'] = shield[0].clone()
            self.state['interval_value'] = interval[0].clone()

        for trial in spice_signals.trials:

            # --- Prediction error: shortest arc from belief to beam ---
            prediction_error = wrap_degrees(laser[trial] - self.state['belief_value'])

            # --- Learning rate: updated every beam from surprise |PE| (scaled to [0, 1]) ---
            self.call_module(
                key_module='learning_rate_update',
                key_state='learning_rate_value',
                inputs=(
                    (prediction_error.abs() / PE_SCALE).clamp(max=1.),
                    caught[trial],
                    spice_signals.additional_inputs['volatility'][trial],
                    spice_signals.additional_inputs['stochasticity'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            alpha = torch.sigmoid(self.state['learning_rate_value'])

            # --- Belief update: angular delta rule ---
            self.state['belief_value'] = (self.state['belief_value'] + alpha * prediction_error) % 360.

            # --- Last move of the shield, as a fraction of the largest possible move (0 = held, 1 = full speed) ---
            last_move = wrap_degrees(shield[trial] - self.state['shield_value']).abs()
            move_fraction = (last_move / (MOVEMENT_SPEED * self.state['interval_value'].clamp(min=1.))).clamp(max=1.)
            self.state['shield_value'] = shield[trial].clone()
            self.state['interval_value'] = interval[trial].clone()

            # --- Movement gain: recurrent, learns g(g_prev, |belief - shield|, last move, catch) ---
            distance = wrap_degrees(self.state['belief_value'] - shield[trial])
            self.call_module(
                key_module='movement_update',
                key_state='movement_value',
                inputs=(
                    (distance.abs() / DISTANCE_SCALE).clamp(max=1.),
                    move_fraction,
                    caught[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            movement_gain = torch.sigmoid(self.state['movement_value'])

            # --- Next shield position: move part of the way toward the belief (speed clamp in the loss) ---
            spice_signals.logits[trial] = degrees_to_sincos(shield[trial] + movement_gain * distance)

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

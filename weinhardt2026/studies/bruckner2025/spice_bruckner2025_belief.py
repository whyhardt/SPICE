"""Latent-belief variant of the bruckner2025 model (weber2024-style): a learned learning rate on a latent belief, and
a learned movement gain that carries the bucket from its start position toward the belief.

    PE_t        = target_t - belief_t,           target_t = mu_t if the helicopter is visible, else x_t
    belief_t+1  = belief_t + alpha_t * PE_t,      alpha_t = sigmoid(learning_rate_value_t)
    bucket_t+1  = z_t+1 + g_t * (belief_t+1 - z_t+1),   g_t = sigmoid(movement_value_t)

z_t+1 is the bucket's start position on the next trial: the previous bucket in the no-push blocks (1-2), a displaced
position in the push blocks (3-4). One movement gain therefore covers both a lag toward the belief (no push) and
anchoring-and-adjustment (push: under-adjustment 1 - g from the displaced start), replacing the anchor module of
spice_bruckner2025.py. The model does not contain spice_bruckner2025.py: the belief runs freely from the first bucket
and is never re-anchored on the observed bucket (which the delta rule there uses as the belief on every trial), and
the anchoring is tied to the gain (1 - g) instead of having a coefficient of its own.

As in weber2024 (spice_weber2024.py), the learning rate is recurrent (surprise relative to the belief, catch,
visibility) and the movement gain is stateless with the belief-start distance as its only input. In the no-push blocks
the start is the previous bucket, so the update is exactly weber's shield update; in the push blocks the bucket starts
at the displaced position, from which the participant has to move.
"""

import torch

from spice import SpiceConfig, BaseModel

from studies.bruckner2025.spice_bruckner2025 import PE_SCALE, mse_loss, prepare_dataframe  # shared data handling and loss


# |belief - start| in screen units is of the order of |PE|; scaled the same way and clipped at 1
DISTANCE_SCALE = 0.5

# Binary indicator control signals (x^2 = x)
BINARY_SIGNALS = {'catch', 'v_t'}


CONFIG = SpiceConfig(
    discovered_modules={
        # Learning rate: updated every trial from surprise relative to the belief, catch and helicopter visibility
        'learning_rate_update': ('|prediction_error|', 'catch', 'v_t'),
        # Movement gain: how far the bucket moves from its start toward the belief
        'movement_update': ('|belief_start_distance|',),
    },

    memory_state={
        'belief_value': 0,          # latent belief about the helicopter (screen units / 300); set to the first bucket
        'learning_rate_value': 0,   # alpha = sigmoid(value), initially 0.5
        'movement_value': 0,        # per-trial movement gain (reset each trial), g = sigmoid(value)
    },


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


# Hooks read by weinhardt2026/run.py
NORMALIZE_REWARDS = False       # outcomes are positions, already scaled by prepare_dataframe
LOSS_FN = mse_loss
LOSS_FN_KWARGS = {}
ESTIMATOR_KWARGS = {'n_items': 1}


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)

        # movement_update: stateless (reset each trial), a function of the belief-start distance only
        self.setup_module(key_module='movement_update', include_state=False)

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

        # Extract signals — shapes after init: (T, W, E, B, ...)
        outcome = spice_signals.feedback                                 # x_t / 300
        bucket = spice_signals.actions                                  # b_t / 300

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        experiment_index = spice_signals.experiment_ids if experiment_embedding is not None else None

        # Initialize the belief to the participant's first bucket position
        if prev_state is None:
            self.state['belief_value'] = bucket[0].clone()

        for trial in spice_signals.trials:

            # --- Update target: helicopter position when visible, outcome otherwise ---
            mask_vt = spice_signals.additional_inputs['v_t'][trial]
            target = torch.where(mask_vt > 0.5, spice_signals.additional_inputs['mu_t'][trial], outcome[trial])

            # --- Prediction error relative to the latent belief ---
            prediction_error = target - self.state['belief_value']

            # --- Learning rate: updated every trial from surprise |PE| (scaled to [0, 1]) ---
            self.call_module(
                key_module='learning_rate_update',
                key_state='learning_rate_value',
                inputs=(
                    (prediction_error.abs() / PE_SCALE).clamp(max=1.),
                    spice_signals.additional_inputs['catch'][trial],
                    mask_vt,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=experiment_index,
                experiment_embedding=experiment_embedding,
            )
            alpha = torch.sigmoid(self.state['learning_rate_value'])

            # --- Belief update: delta rule on the latent belief ---
            self.state['belief_value'] = self.state['belief_value'] + alpha * prediction_error

            # --- Movement gain: reset each trial, learns g(|belief - start|) ---
            start = spice_signals.additional_inputs['z_next'][trial]   # = bucket[trial] in the no-push blocks
            distance = self.state['belief_value'] - start
            self.state['movement_value'] = torch.zeros_like(self.state['movement_value'])
            self.call_module(
                key_module='movement_update',
                key_state='movement_value',
                inputs=((distance.abs() / DISTANCE_SCALE).clamp(max=1.),),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=experiment_index,
                experiment_embedding=experiment_embedding,
            )
            movement_gain = torch.sigmoid(self.state['movement_value'])

            # --- Next bucket position: move part of the way from the start toward the belief ---
            spice_signals.logits[trial] = start + movement_gain * distance

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

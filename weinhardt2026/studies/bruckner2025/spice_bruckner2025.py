import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Main belief update: learns from prediction error (x_t - belief)
        'belief_update': ('prediction_error',),
        # Anchoring bias: learns influence of bucket displacement in push conditions
        'anchor_update': ('anchor_shift',),
    },

    memory_state={
        'belief': 0.5,           # initial belief = center of screen (normalized)
        'anchor_value': 0.0,     # initial anchoring bias
    },

    states_in_logit=['belief', 'anchor_value'],
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Experiment embedding first (smaller) so fusion target = participant embedding size
        # self.experiment_embedding = self.setup_embedding(num_embeddings=self.n_experiments, embedding_size=2)
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)

        # belief_update: 1 control signal (prediction_error)
        self.setup_module(key_module='belief_update', input_size=1)
        # anchor_update: 1 control signal (anchor_shift)
        self.setup_module(key_module='anchor_update', input_size=1)

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, ...)
        # actions:            b_t / 300   → (T, W, E, B, 1)
        # rewards:            x_t / 300   → (T, W, E, B, 1)
        # additional_inputs:  [mu_t, c_t, z_{t+1}] → (T, W, E, B, 3)
        outcome = spice_signals.rewards                                 # x_t / 300
        bucket = spice_signals.actions                                  # b_t / 300
        z_next = spice_signals.additional_inputs[..., 2].unsqueeze(-1)  # z_{t+1} / 300 (T, W, E, B, 1)

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- Prediction error: outcome minus internal belief ---
            prediction_error = outcome[trial] - self.state['belief']

            # --- Anchor shift: y_t = z_{t+1} - b_t (push displacement) ---
            anchor_shift = z_next[trial] - bucket[trial]

            # --- Belief update ---
            self.call_module(
                key_module='belief_update',
                key_state='belief',
                action_mask=None,
                inputs=(prediction_error,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Anchor bias update ---
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

            # --- Logit = predicted next position ---
            spice_signals.logits[trial] = (
                + self.state['belief']
                + self.state['anchor_value']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

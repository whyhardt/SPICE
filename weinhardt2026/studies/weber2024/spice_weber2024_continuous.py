import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Belief update: learns target position from prediction error
        'belief_update': ('prediction_error', 'laser_caught'),
    },

    memory_state={
        # Internal belief about target position (sin/cos components stored as items 0, 1)
        'belief_value': 0,
    },

    states_in_logit=['belief_value'],

    additional_inputs=(
        'laser_caught',
        'volatility',
        'stochasticity',
        'trial_duration_frames',
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

        # Belief update: learns to track laser position from prediction error
        # input_size=2 inferred from library_setup (prediction_error, laser_caught)
        self.setup_module(key_module='belief_update')

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, 2)
        # actions:  shield position (sin, cos) at trial t
        # rewards:  laser position (sin, cos) at trial t
        laser = spice_signals.rewards

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # Initialize belief to first laser observation
        if prev_state is None:
            self.state['belief_value'] = self.state['belief_value'] + laser[0]

        for trial in spice_signals.trials:

            # Prediction error: laser minus belief (per sin/cos component)
            prediction_error = laser[trial] - self.state['belief_value']

            # --- Belief update: learns where the laser target is ---
            self.call_module(
                key_module='belief_update',
                key_state='belief_value',
                action_mask=None,
                inputs=(
                    prediction_error,
                    spice_signals.additional_inputs['laser_caught'][trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # Logit = predicted belief position (sin, cos)
            spice_signals.logits[trial] = self.state['belief_value']

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

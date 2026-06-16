import torch

from spice import SpiceConfig, BaseModel


CONFIG = SpiceConfig(
    library_setup={
        # Belief update: learns target position from prediction error
        'belief_update': ('prediction_error', 'catch', 'catch_trial'),
        # Changepoint detection: learns omega (surprise-driven) from PE magnitude
        'changepoint': ('prediction_error', 'catch', 'catch_trial'),
        # Base learning rate: learns tau (condition-dependent baseline)
        'learning_rate': ('catch', 'catch_trial'),
        # Anchoring bias: learns compensation for bucket displacement
        'anchor_update': ('anchor_shift',),
    },

    memory_state={
        'belief_value': 0.5,      # initial belief = center of screen (normalized)
        'omega_value': 0.0,       # changepoint probability state (sigmoid(0) = 0.5)
        'tau_value': 0.0,         # base learning rate state (sigmoid(0) = 0.5)
        'anchor_value': 0.0,      # per-trial anchoring correction (reset each trial)
    },

    states_in_logit=[
        'belief_value',
        'omega_value',
        'tau_value',
        'anchor_value',
    ],
)


class SpiceModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)

        # belief_update: 3 control signals (prediction_error, catch, catch_trial)
        self.setup_module(key_module='belief_update', input_size=3, dropout=self.dropout)
        # changepoint: 3 control signals (prediction_error, catch, catch_trial)
        self.setup_module(key_module='changepoint', input_size=3, dropout=self.dropout)
        # learning_rate: 2 control signals (catch, catch_trial)
        self.setup_module(key_module='learning_rate', input_size=2, dropout=self.dropout)
        # anchor_update: 1 control signal (anchor_shift); no state in library (always reset to 0)
        self.setup_module(key_module='anchor_update', input_size=1, dropout=self.dropout, include_state=False)

    def forward(self, inputs, prev_state=None):

        spice_signals = self.init_forward_pass(inputs, prev_state)

        # Extract signals — shapes after init: (T, W, E, B, ...)
        # actions:            b_t / 300   → (T, W, E, B, 1)
        # rewards:            x_t / 300   → (T, W, E, B, 1)
        # additional_inputs:  [z_{t+1}, catch, v_t, mu_t, c_t] → (T, W, E, B, 5)
        outcome = spice_signals.rewards                                 # x_t / 300
        bucket = spice_signals.actions                                  # b_t / 300
        z_next = spice_signals.additional_inputs[..., 0].unsqueeze(-1)  # z_{t+1} / 300
        catch = spice_signals.additional_inputs[..., 1].unsqueeze(-1).expand_as(spice_signals.actions)
        catch_trial = spice_signals.additional_inputs[..., 2].unsqueeze(-1).expand_as(spice_signals.actions)

        # Embeddings
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids) if hasattr(self, 'experiment_embedding') else None
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- Prediction error: outcome minus internal belief ---
            prediction_error = outcome[trial] - self.state['belief_value']

            # --- Anchor shift: bucket displacement from push ---
            anchor_shift = z_next[trial] - bucket[trial]

            # --- Belief update: learns where the helicopter is ---
            self.call_module(
                key_module='belief_update',
                key_state='belief_value',
                action_mask=None,
                inputs=(
                    prediction_error,
                    catch[trial],
                    catch_trial[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Changepoint: learns surprise-driven update rate (omega) ---
            self.call_module(
                key_module='changepoint',
                key_state='omega_value',
                action_mask=None,
                inputs=(
                    prediction_error,
                    catch[trial],
                    catch_trial[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids if experiment_embedding is not None else None,
                experiment_embedding=experiment_embedding,
            )

            # --- Learning rate: learns condition-dependent base rate (tau) ---
            self.call_module(
                key_module='learning_rate',
                key_state='tau_value',
                action_mask=None,
                inputs=(
                    catch[trial],
                    catch_trial[trial],
                ),
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
            omega = torch.sigmoid(self.state['omega_value'])
            tau = torch.sigmoid(self.state['tau_value'])
            alpha = omega + tau - omega * tau  # = 1 - (1-omega)(1-tau), ∈ [0, 1]

            # --- Gated output + anchor correction ---
            spice_signals.logits[trial] = (
                bucket[trial] + alpha * (self.state['belief_value'] - bucket[trial])
                + self.state['anchor_value']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

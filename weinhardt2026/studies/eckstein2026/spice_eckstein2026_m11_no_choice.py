"""eckstein2026 variant m11_no_choice: m02 without the choice-history modules (value_choice and bias_attention).

See model_variants.md in this directory for the full list of variants.
"""
import torch

from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_env': ['reward[t]'],
        'value_reward_chosen': ['reward_env', 'reward[t]'],
        'value_reward_not_chosen': ['reward_env'],
        'value_exploration_chosen': ['dvalue'],
        'value_exploration_not_chosen': ['dvalue'],
    },
    memory_state={
        'value_reward_env': None,
        'value_reward': None,
        'value_exploration': None,

        # Buffers (excluded from logits)
        'value_reward[t-1]': None,
        'action[t-1]': 0,
    },
    states_in_logit=['value_reward_env', 'value_reward', 'value_exploration'],
)


# Binary indicator control signals (x^2 = x) and mutually exclusive indicator groups (x_i * x_j = 0).
BINARY_SIGNALS = {'action[t]', 'action[t-1]', 'is_adjacent', 'is_opposite'}
EXCLUSIVE_GROUPS = [{'is_adjacent', 'is_opposite'}]
# Modules without state x input terms: these made the exploration equations self-amplifying.
NO_STATE_PRODUCTS = {'value_exploration_chosen', 'value_exploration_not_chosen'}


class SpiceModel(BaseModel):
    """m02 without the choice-history modules (value_choice and bias_attention)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        self.setup_module(key_module='value_reward_env', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_reward_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_not_chosen', input_size=1, dropout=self.dropout)

        self.preprocess_coefficients()

    def preprocess_coefficients(self):
        """Zero out SINDy terms that are structurally redundant or excluded.

        Binary indicators satisfy x^2 = x, so the squared term duplicates the linear one.
        Two indicators of the same mutually exclusive group satisfy x_i * x_j = 0.
        Modules in NO_STATE_PRODUCTS keep no products of their own state with an input.
        """
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            binary_signals = [s for s in control_signals if s in BINARY_SIGNALS]
            for index_term, term in enumerate(candidate_terms[module]):
                factors = term.split('*')
                redundant = any(signal + '^2' in factors for signal in binary_signals)
                for group in EXCLUSIVE_GROUPS:
                    redundant |= sum(signal in factors for signal in group if signal in control_signals) > 1
                if module in NO_STATE_PRODUCTS:
                    redundant |= len(factors) > 1 and module in factors
                if redundant:
                    self.sindy_coefficients_presence[module][..., index_term] = 0
                    self.sindy_coefficients_prior_mask[module][..., index_term] = 0

    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        reward_full = spice_signals.feedback.sum(dim=-1, keepdim=True).expand_as(spice_signals.actions)
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- ENV REWARD ---
            value_reward_env = self.call_module(
                key_module='value_reward_env',
                key_state='value_reward_env',
                inputs=reward_full[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- REWARD VALUE UPDATES ---
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    spice_signals.feedback[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                inputs=value_reward_env,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- EXPLORATION VALUE UPDATES ---
            dvalue = (self.state['value_reward'] - self.state['value_reward[t-1]']).detach()

            self.call_module(
                key_module='value_exploration_chosen',
                key_state='value_exploration',
                action_mask=spice_signals.actions[trial],
                inputs=dvalue,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_exploration_not_chosen',
                key_state='value_exploration',
                action_mask=1 - spice_signals.actions[trial],
                inputs=dvalue,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- BUFFER UPDATES ---
            self.state['value_reward[t-1]'] = self.state['value_reward']
            self.state['action[t-1]'] = spice_signals.actions[trial]

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_reward']
                + self.state['value_exploration']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

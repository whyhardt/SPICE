"""eckstein2026 variant m14_no_exploration_no_env: Step 1 arm: no exploration modules, reward modules without env and mean inputs.

See model_variants.md in this directory for the full list of variants.
"""
import torch

from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward[t]'],
        'value_reward_not_chosen': [],
        'value_choice': ['action[t]', 'action[t-1]'],
        'bias_attention': ['action[t-1]', 'is_adjacent', 'is_opposite'],
    },
    memory_state={
        'value_reward': None,
        'value_choice': None,
        'bias_attention': None,

        # Buffers (excluded from logits)
        'value_reward[t-1]': None,
        'action[t-1]': 0,
    },
    states_in_logit=['value_reward', 'value_choice', 'bias_attention'],
)


# Binary indicator control signals (x^2 = x) and mutually exclusive indicator groups (x_i * x_j = 0).
BINARY_SIGNALS = {'action[t]', 'action[t-1]', 'is_adjacent', 'is_opposite'}
EXCLUSIVE_GROUPS = [{'is_adjacent', 'is_opposite'}]
# Modules without state x input terms: these made the exploration equations self-amplifying.
NO_STATE_PRODUCTS = {}


class SpiceModel(BaseModel):
    """Step 1 arm: no exploration modules, reward modules without env and mean inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        self.setup_module(key_module='value_reward_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0, dropout=self.dropout)
        self.setup_module(key_module='value_choice', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='bias_attention', input_size=3, dropout=self.dropout)

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

        item_indices = torch.arange(self.n_actions, device=self.device)

        for trial in spice_signals.trials:

            # --- REWARD VALUE UPDATES ---
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=spice_signals.feedback[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- CHOICE HISTORY ---
            chosen_idx = spice_signals.actions[trial].argmax(dim=-1, keepdim=True)
            items = item_indices.expand_as(spice_signals.actions[trial])
            raw_dist = torch.abs(items - chosen_idx)
            circ_dist = torch.min(raw_dist, self.n_actions - raw_dist)
            is_adjacent = (circ_dist == 1).float()
            is_opposite = (circ_dist == (self.n_actions // 2)).float()

            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                inputs=(
                    spice_signals.actions[trial],
                    self.state['action[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='bias_attention',
                key_state='bias_attention',
                inputs=(
                    self.state['action[t-1]'],
                    is_adjacent,
                    is_opposite,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- BUFFER UPDATES ---
            self.state['value_reward[t-1]'] = self.state['value_reward']
            self.state['action[t-1]'] = spice_signals.actions[trial]

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                self.state['value_reward']
                + self.state['value_choice']
                + self.state['bias_attention']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

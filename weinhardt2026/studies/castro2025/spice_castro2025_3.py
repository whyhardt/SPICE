import torch

from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_env': [
            'reward[t]',
        ],
        'value_reward_chosen': [
            'reward_env',
            'reward[t]',
            # '(1-reward[t])',
            'value_reward_mean',
        ],
        'value_reward_not_chosen': [
            'reward_env',
            'value_reward_mean',
        ],
        'value_choice_chosen': [
            'action[t-1]',
        ],
        'value_choice_not_chosen': [
            'action[t-1]',
        ],
        'value_exploration_chosen': [
            'dvalue_pos',
            'dvalue_neg',
        ],
        'value_exploration_not_chosen': [
            'dvalue_pos',
            'dvalue_neg',
        ],
        'bias_attention': [
            'action[t-1]',
            'is_adjacent',
            'is_opposite',
        ],
    },
    memory_state={
        'value_reward_env': None,
        'value_reward': None,
        'value_choice': None,
        'value_exploration': None,
        'bias_attention': None,

        # Buffers (excluded from logits)
        'value_reward[t-1]': None,
        'action[t-1]': 0,
    },
    states_in_logit=[
        'value_reward_env',
        'value_reward',
        'value_choice',
        'value_exploration',
        'bias_attention',
        ],
)


class SpiceModel(BaseModel):
    """
    v3: like v2 but with split choice (chosen/not_chosen with action mask).

    Replaces the merged value_choice module with separate choice_chosen and
    choice_not_chosen modules, each with action[t-1] as input. This gives
    each module a smaller SINDy library and allows chosen/unchosen items to
    follow different update rules.

    8 modules, 5 logit states.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        self.setup_module(key_module='value_reward_env', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_reward_chosen', input_size=3, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_choice_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_not_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='bias_attention', input_size=3, dropout=self.dropout)


    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        reward_full = spice_signals.rewards.sum(dim=-1, keepdim=True).expand_as(spice_signals.actions)
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        item_indices = torch.arange(self.n_actions, device=self.device)

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
            mean_value_reward = self.state['value_reward'].mean(
                dim=-1, keepdim=True,
            ).expand_as(self.state['value_reward']).detach()

            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    spice_signals.rewards[trial],
                    # one_minus_reward,
                    mean_value_reward,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    mean_value_reward,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- CHOICE VALUE UPDATES (split) ---
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['action[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    self.state['action[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- EXPLORATION VALUE UPDATES ---
            dvalue = (self.state['value_reward'] - self.state['value_reward[t-1]']).detach()
            dvalue_pos = torch.relu(dvalue)
            dvalue_neg = torch.relu(-dvalue)

            self.call_module(
                key_module='value_exploration_chosen',
                key_state='value_exploration',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    dvalue_pos,
                    dvalue_neg,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_exploration_not_chosen',
                key_state='value_exploration',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    dvalue_pos,
                    dvalue_neg,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- ATTENTION COMPUTATION ---
            chosen_idx = spice_signals.actions[trial].argmax(dim=-1, keepdim=True)
            items = item_indices.expand_as(spice_signals.actions[trial])
            raw_dist = torch.abs(items - chosen_idx)
            circ_dist = torch.min(raw_dist, self.n_actions - raw_dist)

            is_adjacent = (circ_dist == 1).float()
            is_opposite = (circ_dist == (self.n_actions // 2)).float()

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
                + self.state['value_exploration']
                + self.state['bias_attention']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

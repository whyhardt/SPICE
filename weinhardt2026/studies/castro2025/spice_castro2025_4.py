import torch

from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': [
            'reward[t]',
            'value_reward_mean',
        ],
        'value_reward_not_chosen': [
            'value_reward_mean',
        ],
        'value_choice_chosen': [
            'action[t-1]',
        ],
        'value_choice_not_chosen': [
            'action[t-1]',
        ],
        'value_exploration_chosen': [
            'dvalue',
        ],
        'value_exploration_not_chosen': [
            'dvalue',
        ],
    },
    memory_state={
        'value_reward': None,  # learnable per-participant initial value (reward prior)
        'value_choice': 0.,
        'value_exploration': 0.,

        # Buffers (excluded from logits)
        'value_reward[t-1]': 0.,
        'action[t-1]': 0.,
    },
    states_in_logit=['value_reward', 'value_choice', 'value_exploration'],
)


class SpiceModel(BaseModel):
    """
    v4: v3 architecture + learnable initial reward values.

    The only change from v3: value_reward initial value is None in the config,
    which triggers per-participant learnable initial values in BaseModel.

    In the benchmark, Q-values are initialized to a per-participant 'prior'
    parameter (softplus-clipped to [0.01, 0.99]). SPICE v3 uses a fixed 0.,
    which is pessimistic for a [0,1]-reward task. Learning the initial value
    lets each participant start with their own reward expectation, matching
    the benchmark's prior parameter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.participant_embedding = self.setup_embedding(
            num_embeddings=self.n_participants, dropout=self.dropout,
        )

        self.setup_module(key_module='value_reward_chosen', input_size=2, dropout=self.dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_choice_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_chosen', input_size=1, dropout=self.dropout)
        self.setup_module(key_module='value_exploration_not_chosen', input_size=1, dropout=self.dropout)

    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- REWARD VALUE UPDATES ---
            mean_value_reward = self.state['value_reward'].mean(
                dim=-1, keepdim=True,
            ).expand_as(self.state['value_reward']).detach()

            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.rewards[trial],
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
                    mean_value_reward,
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- CHOICE VALUE UPDATES ---
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

            self.call_module(
                key_module='value_exploration_chosen',
                key_state='value_exploration',
                action_mask=spice_signals.actions[trial],
                inputs=(dvalue,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_exploration_not_chosen',
                key_state='value_exploration',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(dvalue,),
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
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

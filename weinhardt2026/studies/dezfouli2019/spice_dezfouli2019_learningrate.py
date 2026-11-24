"""dezfouli2019 SPICE model with discovered learning and forgetting rates and hard-coded belief updates.

Identical to `spice_dezfouli2019.SpiceModel` except for the reward-value updates. Instead of
discovered modules, the belief is updated by two fixed modules:

    chosen:     value_reward[t+1] = value_reward[t] + alpha[t] * (reward[t] - value_reward[t])
    not chosen: value_reward[t+1] = value_reward[t] + phi[t]   * (prior    - value_reward[t])

with alpha = sigmoid(learning_rate_reward) and phi = sigmoid(forgetting_rate_reward). SPICE
discovers the equations of the two rates, memory states split into chosen / not-chosen modules
like every other state. The learning rate may depend on its own past value, the reward and the
current belief (e.g. Pearce-Hall: |reward - value| = reward + value - 2 * reward * value for binary
rewards, which degree-2 polynomials can express); the forgetting rate on its own past value and the
current belief. The prior the unchosen value decays toward is a per-participant constant
(sigmoid(FFN(embedding)) in (0, 1)): ~0 forgetting, ~0.5 resetting to uncertainty, ~1 optimism.

Both updates keep `value_reward` in [0, 1], so its contribution to the logits is scaled by a
per-participant inverse temperature (a constant computed from the participant embedding).
"""

import torch

from spice import SpiceConfig

from spice_dezfouli2019 import SpiceModel as SpiceModelDezfouli2019


def delta_rule(value_reward, learning_rate, reward):
    return value_reward + learning_rate * (reward - value_reward)


def forgetting_rule(value_reward, forgetting_rate, prior):
    return value_reward + forgetting_rate * (prior - value_reward)


CONFIG = SpiceConfig(
    discovered_modules={
        'value_wm_reward_chosen': [
            'reward[t]',
        ],
        'value_wm_reward_not_chosen': [],
        'learning_rate_reward_chosen': [
            'reward[t]',
            'value_reward',
        ],
        'learning_rate_reward_not_chosen': [],
        'forgetting_rate_reward_chosen': [],
        'forgetting_rate_reward_not_chosen': [
            'value_reward',
        ],
        'value_choice': [
            'choice[t]',
        ],
    },
    fixed_modules={
        'value_reward_chosen': (delta_rule, ['value_reward', 'learning_rate_reward', 'reward[t]']),
        'value_reward_not_chosen': (forgetting_rule, ['value_reward', 'forgetting_rate_reward', 'value_reward_prior']),
    },
    constants={
        'inverse_temperature': torch.nn.functional.softplus,
        'value_reward_prior': torch.sigmoid,
    },
    memory_state={
        'value_wm_reward': 0,
        'value_reward': 0,
        'value_choice': 0,
        'learning_rate_reward': 0,
        'forgetting_rate_reward': 0,
    },
)


class SpiceModel(SpiceModelDezfouli2019):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        inverse_temperature = self.call_module(
            key_module='inverse_temperature',
            participant_embedding=participant_embedding,
        )

        value_reward_prior = self.call_module(
            key_module='value_reward_prior',
            participant_embedding=participant_embedding,
        )

        for trial in spice_signals.trials:

            # --- WORKING MEMORY UPDATES ---
            self.call_module(
                key_module='value_wm_reward_chosen',
                key_state='value_wm_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.feedback[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_wm_reward_not_chosen',
                key_state='value_wm_reward',
                action_mask=1 - spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- LEARNING RATE UPDATES ---
            # raw learning rate from the belief before the update
            self.call_module(
                key_module='learning_rate_reward_chosen',
                key_state='learning_rate_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.feedback[trial],
                    self.state['value_reward'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='learning_rate_reward_not_chosen',
                key_state='learning_rate_reward',
                action_mask=1 - spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- FORGETTING RATE UPDATES ---
            # raw forgetting rate from the belief before the update
            self.call_module(
                key_module='forgetting_rate_reward_chosen',
                key_state='forgetting_rate_reward',
                action_mask=spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='forgetting_rate_reward_not_chosen',
                key_state='forgetting_rate_reward',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    self.state['value_reward'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- REWARD VALUE UPDATES ---
            # sigmoids keep the learning and forgetting rates in (0, 1)
            learning_rate_reward = torch.sigmoid(self.state['learning_rate_reward'])
            forgetting_rate_reward = torch.sigmoid(self.state['forgetting_rate_reward'])
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['value_reward'],
                    learning_rate_reward,
                    spice_signals.feedback[trial],
                ),
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1 - spice_signals.actions[trial],
                inputs=(
                    self.state['value_reward'],
                    forgetting_rate_reward,
                    value_reward_prior,
                ),
            )

            # --- CHOICE VALUE UPDATES ---
            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                inputs=spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- LOGITS ---
            spice_signals.logits[trial] = (
                inverse_temperature * self.state['value_reward']
                + self.state['value_wm_reward']
                + self.state['value_choice']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()

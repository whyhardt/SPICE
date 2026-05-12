from ..resources.estimator import SpiceConfig
from ..resources.model import BaseModel

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) BUFFER-BASED REWARD AND CHOICE MEMORY FOR 3 TIMESTEPS (could be easily extended)
# 2) Two time-scales for reward values
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        # Fast value learning from current signals
        'value_reward_chosen': [
            'reward[t]',
        ],
        'value_reward_not_chosen': [
            ],

        # Slow (WM) value learning from recent value history
        'value_wm_reward_chosen': [
            'value_reward[t]',
            'value_reward[t-1]',
            'value_reward[t-2]',
        ],
        'value_wm_reward_not_chosen': [
            'value_reward[t]',
            'value_reward[t-1]',
            'value_reward[t-2]',
            ],

        # Fast choice perseveration from current choice
        'value_choice_chosen': [
            'choice[t]',
        ],
        'value_choice_not_chosen': [
            ],

        # Slow (WM) choice perseveration from recent choice value history
        'value_wm_choice_chosen': [
            'value_choice[t]',
            'value_choice[t-1]',
            'value_choice[t-2]',
        ],
        'value_wm_choice_not_chosen': [
            'value_choice[t]',
            'value_choice[t-1]',
            'value_choice[t-2]',
            ],
    },

    memory_state = {
        'value_reward': 0.,      # reward value (fast learning)
        'value_choice': 0.,      # choice value (fast learning)
        'value_reward[t-1]': 0.,   # t-1 reward value
        'value_reward[t-2]': 0.,   # t-2 reward value
        'value_choice[t-1]': 0.,   # t-1 choice value
        'value_choice[t-2]': 0.,   # t-2 choice value
        'value_wm_reward': 0.,      # reward value (slow learning)
        'value_wm_choice': 0.,      # choice value (slow learning)
    },

    states_in_logit=['value_reward', 'value_wm_reward', 'value_choice', 'value_wm_choice'],
)


class SpiceModel(BaseModel):
    """
    Two-timescale working memory model.

    Fast modules update value_reward and value_choice directly from current signals.
    Slow (WM) modules integrate over the recent history of those fast values,
    stored in explicit buffers (value_reward[t-1], value_reward[t-2], etc.).
    """

    def __init__(self, reward_binary: bool = False, **kwargs):
        super().__init__(**kwargs)

        dropout = 0.1

        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)

        # Fast reward value modules
        self.setup_module(key_module='value_reward_chosen', input_size=1, dropout=dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0, dropout=dropout)

        # Slow (WM) reward value modules — integrate over recent value_reward history
        self.setup_module(key_module='value_wm_reward_chosen', input_size=3, dropout=dropout)
        self.setup_module(key_module='value_wm_reward_not_chosen', input_size=3, dropout=dropout)

        # Fast choice value modules
        self.setup_module(key_module='value_choice_chosen', input_size=1, dropout=dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=0, dropout=dropout)

        # Slow (WM) choice value modules — integrate over recent value_choice history
        self.setup_module(key_module='value_wm_choice_chosen', input_size=3, dropout=dropout)
        self.setup_module(key_module='value_wm_choice_not_chosen', input_size=3, dropout=dropout)

        self.preprocess_coefficients(reward_binary=reward_binary)

    def forward(self, inputs, prev_state=None):
        spice_signals = self.init_forward_pass(inputs, prev_state)

        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:

            # --- FAST REWARD VALUE UPDATES ---
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.rewards[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[trial],
                inputs=(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- SLOW (WM) REWARD VALUE UPDATES ---
            self.call_module(
                key_module='value_wm_reward_chosen',
                key_state='value_wm_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['value_reward'],
                    self.state['value_reward[t-1]'],
                    self.state['value_reward[t-2]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='value_wm_reward_not_chosen',
                key_state='value_wm_reward',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    self.state['value_reward'],
                    self.state['value_reward[t-1]'],
                    self.state['value_reward[t-2]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- FAST CHOICE VALUE UPDATES ---
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.actions[trial],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[trial],
                inputs=(),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- SLOW (WM) CHOICE VALUE UPDATES ---
            self.call_module(
                key_module='value_wm_choice_chosen',
                key_state='value_wm_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['value_choice'],
                    self.state['value_choice[t-1]'],
                    self.state['value_choice[t-2]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='value_wm_choice_not_chosen',
                key_state='value_wm_choice',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    self.state['value_choice'],
                    self.state['value_choice[t-1]'],
                    self.state['value_choice[t-2]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # --- SHIFT VALUE BUFFERS ---
            self.state['value_reward[t-2]'] = self.state['value_reward[t-1]'].clone()
            self.state['value_reward[t-1]'] = self.state['value_reward'].clone()
            self.state['value_choice[t-2]'] = self.state['value_choice[t-1]'].clone()
            self.state['value_choice[t-1]'] = self.state['value_choice'].clone()

            # compute logits for current timestep
            spice_signals.logits[trial] = (
                self.state['value_reward'] + self.state['value_wm_reward']
                + self.state['value_choice'] + self.state['value_wm_choice']
            )

        spice_signals = self.post_forward_pass(spice_signals)

        return spice_signals.logits, self.get_state()

    def preprocess_coefficients(self, reward_binary: bool = True):
        # remove squared terms for binary signals only:
        # reward[t] is binary (if reward_binary), choice[t] is always binary
        # value history signals (value_reward[t], etc.) are continuous — keep their polynomials
        binary_signals = {'choice[t]'}
        if reward_binary:
            binary_signals.add('reward[t]')

        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            control_signals = self.spice_config.library_setup[module]
            for cs in control_signals:
                if cs in binary_signals:
                    for ict, ct in enumerate(candidate_terms[module]):
                        if cs+'^' in ct:
                            self.sindy_coefficients_presence[module][..., ict] = 0

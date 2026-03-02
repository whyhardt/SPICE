from ..resources.estimator import SpiceConfig
from ..resources.rnn import BaseRNN

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) BUFFER-BASED REWARD AND CHOICE MEMORY FOR 3 TIMESTEPS (could be easily extended)
# 2) Two time-scales for reward values
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        # Value learning can depend on recent reward sequence (working memory)
        'value_reward_chosen': [
            'reward[t]',           
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
        ],
        'value_reward_not_chosen': [
            # 'reward_chosen',
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            ],
        'value_choice': [
            'choice[t]',
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            ],
        # 'logits': [
        #     'value_reward',
        #     'value_choice',
        # ]
    },
    
    memory_state = {
        'value_reward': 0.,      # reward value (enables slow learning)
        'value_choice': 0.,      # choice value (enables slow learning)
        # 'logits': 0,             # logits for decision
        'buffer_reward_1': 0.,   # t-1 reward
        'buffer_reward_2': 0.,   # t-2 reward
        'buffer_reward_3': 0.,   # t-3 reward
        'buffer_choice_1': 0.,   # t-1 choice
        'buffer_choice_2': 0.,   # t-2 choice
        'buffer_choice_3': 0.,   # t-3 choice
    },
    
    states_in_logit=['value_reward', 'value_choice'],
)


class SpiceModel(BaseRNN):
    """
    Working memory as explicit buffer of recent rewards.
    
    Key difference from value learning:
    - Stores individual past rewards (not aggregated statistics)
    - Fixed capacity (buffer size)
    - Perfect memory for items in buffer
    - Items fall out of buffer (discrete forgetting)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        dropout = 0.1
        
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)
        
        # Value learning module (slow updates)
        # Can use recent reward history to modulate learning
        self.setup_module(key_module='value_reward_chosen', input_size=4+self.embedding_size, dropout=dropout, include_bias=True)  # -> 21 terms
        self.setup_module(key_module='value_reward_not_chosen', input_size=3+self.embedding_size, dropout=dropout, include_bias=True)  # -> 21 terms
        self.setup_module(key_module='value_choice', input_size=4+self.embedding_size, dropout=dropout, include_bias=True) # -> 21 terms; bias not necessary when module is applied equally to all options
        # self.setup_module(key_module='logits', input_size=2+self.embedding_size, dropout=dropout, include_bias=False, fit_linear=False, include_state=False) # -> 8 terms
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        # perform time-invariant computations
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        # perform time-variant computations (compiled)
        spice_signals.logits = self._trial_loop(
            spice_signals.actions, spice_signals.rewards, spice_signals.logits,
            spice_signals.participant_ids, participant_embedding,
        )

        spice_signals = self.post_forward_pass(spice_signals, batch_first)

        return spice_signals.logits, self.get_state()

    @torch.compile(dynamic=True)
    def _trial_loop(self, actions, rewards, logits, participant_ids, participant_embedding):
        T = actions.shape[0]

        for timestep in range(T):

            actions_t = actions[timestep, 0]   # [E, B, n_actions]
            rewards_t = rewards[timestep, 0]   # [E, B, n_actions]

            # REWARD VALUE UPDATES
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=actions_t,
                inputs=(
                    rewards_t,
                    self.state['buffer_reward_1'],
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                ),
                participant_index=participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-actions_t,
                inputs=(
                    self.state['buffer_reward_1'],
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    ),
                participant_index=participant_ids,
                participant_embedding=participant_embedding,
            )

            # CHOICE VALUE UPDATES
            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                action_mask=None,
                inputs=(
                    actions_t,
                    self.state['buffer_choice_1'],
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                ),
                participant_index=participant_ids,
                participant_embedding=participant_embedding,
            )

            # BUFFER UPDATES:
            # REWARD BUFFER UPDATES: Shift reward buffer for chosen action, keep for not chosen action
            # CHOICE BUFFER UPDATES: Shift all buffer entries according to action
            self.state['buffer_reward_3'] = self.state['buffer_reward_2'] * actions_t + self.state['buffer_reward_3'] * (1-actions_t)
            self.state['buffer_reward_2'] = self.state['buffer_reward_1'] * actions_t + self.state['buffer_reward_2'] * (1-actions_t)
            self.state['buffer_reward_1'] = torch.where(actions_t==1, rewards_t, 0) + torch.where(actions_t==0, self.state['buffer_reward_1'], 0)
            self.state['buffer_choice_3'] = self.state['buffer_choice_2']
            self.state['buffer_choice_2'] = self.state['buffer_choice_1']
            self.state['buffer_choice_1'] = actions_t

            # compute logits for current timestep
            logits[timestep] = self.state['value_reward'] + self.state['value_choice']

        return logits

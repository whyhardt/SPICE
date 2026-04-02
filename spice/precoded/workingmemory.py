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
        # Value learning can depend on recent reward sequence (working memory)
        'value_reward_chosen': [
            'reward[t]',           
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
        ],
        'value_reward_not_chosen': [
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            ],
        
        # 'value_choice': [
        #     'choice[t]',
        #     'choice[t-1]', 
        #     'choice[t-2]',
        #     'choice[t-3]',
        #     ],
        'value_choice_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            ],
        'value_choice_not_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            ],
    },
    
    memory_state = {
        'value_reward': 0.,      # reward value (enables slow learning)
        'value_choice': 0.,      # choice value (enables slow learning)
        'buffer_reward_1': 0.,   # t-1 reward
        'buffer_reward_2': 0.,   # t-2 reward
        'buffer_reward_3': 0.,   # t-3 reward
        'buffer_action_1': 0.,   # t-1 choice
        'buffer_action_2': 0.,   # t-2 choice
        'buffer_action_3': 0.,   # t-3 choice
    },
    
    states_in_logit=['value_reward', 'value_choice'],
)


class SpiceModel(BaseModel):
    """
    Working memory as explicit buffer of recent rewards.
    
    Key difference from value learning:
    - Stores individual past rewards (not aggregated statistics)
    - Fixed capacity (buffer size)
    - Perfect memory for items in buffer
    - Items fall out of buffer (discrete forgetting)
    """
    
    def __init__(self, reward_binary: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        dropout = 0.1
        
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)
        
        # Value learning module (slow updates)
        # Can use recent reward history to modulate learning
        self.setup_module(key_module='value_reward_chosen', input_size=4+self.embedding_size, dropout=dropout)  # -> 21 terms
        self.setup_module(key_module='value_reward_not_chosen', input_size=3+self.embedding_size, dropout=dropout)  # -> 21 terms
        
        # self.setup_module(key_module='value_choice', input_size=4+self.embedding_size, dropout=dropout, include_bias=True) # -> 21 terms; bias not necessary when module is applied equally to all options
        self.setup_module(key_module='value_choice_chosen', input_size=3+self.embedding_size, dropout=dropout) # -> 21 terms; bias not necessary when module is applied equally to all options
        self.setup_module(key_module='value_choice_not_chosen', input_size=3+self.embedding_size, dropout=dropout) # -> 21 terms; bias not necessary when module is applied equally to all options
        
        self.preprocess_coefficients(reward_binary=reward_binary)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)

        # perform time-invariant computations
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:
            
            # REWARD VALUE UPDATES
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    spice_signals.rewards[trial],
                    self.state['buffer_reward_1'],
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    self.state['buffer_reward_1'],
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # CHOICE VALUE UPDATES
            # self.call_module(
            #     key_module='value_choice',
            #     key_state='value_choice',
            #     action_mask=None,
            #     inputs=(
            #         spice_signals.actions[trial],
            #         self.state['buffer_action_1'],
            #         self.state['buffer_action_2'],
            #         self.state['buffer_action_3'],
            #     ),
            #     participant_index=spice_signals.participant_ids,
            #     participant_embedding=participant_embedding,
            # )
            
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['buffer_action_1'],
                    self.state['buffer_action_2'],
                    self.state['buffer_action_3'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    self.state['buffer_action_1'],
                    self.state['buffer_action_2'],
                    self.state['buffer_action_3'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # BUFFER UPDATES:
            # REWARD BUFFER UPDATES: Shift reward buffer for chosen action, keep for not chosen action
            # ACTION BUFFER UPDATES: Shift all buffer entries according to action
            self.state['buffer_reward_3'] = self.state['buffer_reward_2'] * spice_signals.actions[trial] + self.state['buffer_reward_3'] * (1-spice_signals.actions[trial])
            self.state['buffer_reward_2'] = self.state['buffer_reward_1'] * spice_signals.actions[trial] + self.state['buffer_reward_2'] * (1-spice_signals.actions[trial])
            self.state['buffer_reward_1'] = torch.where(spice_signals.actions[trial]==1, spice_signals.rewards[trial], 0) + torch.where(spice_signals.actions[trial]==0, self.state['buffer_reward_1'], 0)
            self.state['buffer_action_3'] = self.state['buffer_action_2']
            self.state['buffer_action_2'] = self.state['buffer_action_1']
            self.state['buffer_action_1'] = spice_signals.actions[trial]

            # compute logits for current timestep
            spice_signals.logits[trial] = self.state['value_reward'] + self.state['value_choice']

        
        spice_signals = self.post_forward_pass(spice_signals, batch_first)

        return spice_signals.logits, self.get_state()
    
    def preprocess_coefficients(self, reward_binary: bool = True):
        # remove unnecessary candidate terms, e.g. polynomials of binary signals
        # if reward_binary: reward[t] = reward[t]^2 -> presence[reward[t]^2] = 0
        # accounts for ALL control signals in workingmemory model if reward is binary; else only choice signals
        
        candidate_terms = self.get_candidate_terms()
        for module in self.get_modules():
            if ('reward' in module and reward_binary) or 'choice' in module:
                control_signals = self.spice_config.library_setup[module]
                for cs in control_signals:
                    for ict, ct in enumerate(candidate_terms[module]):
                        if cs+'^' in ct:
                            self.sindy_coefficients_presence[module][..., ict] = 0
        

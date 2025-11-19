from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) INTERACTIONS BETWEEN REWARD-BASED AND CHOICE-BASED MODULES
# 2) BUFFER-BASED REWARD AND CHOICE MEMORY FOR 3 TIMESTEPS (could be easily extended)
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        # Value learning can depend on recent reward sequence (working memory)
        'value_reward_chosen': [
            'reward[t]',           
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            # 'value_choice',
        ],
        'value_reward_not_chosen': [
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            # 'value_choice',
            ],
        'value_choice_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            # 'value_reward',
            ],
        'value_choice_not_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            # 'value_reward',
            ],
    },
    
    memory_state = {
        'value_reward': 0.5,      # reward value (enables slow learning)
        'value_choice': 0.0,      # choice value (enables slow learning)
        'buffer_reward_1': 0.5,   # t-1 reward
        'buffer_reward_2': 0.5,   # t-2 reward
        'buffer_reward_3': 0.5,   # t-3 reward
        'buffer_choice_1': 0.5,   # t-1 choice
        'buffer_choice_2': 0.5,   # t-2 choice
        'buffer_choice_3': 0.5,   # t-3 choice
    }
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
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        spice_config: SpiceConfig,
        embedding_size: int = 32,
        dropout: float = 0.5,
        sindy_polynomial_degree: int = 2,
        sindy_ensemble_size: int = 10,
        use_sindy: bool = False,
        **kwargs):
        super().__init__(
            n_actions=n_actions,
            n_participants=n_participants,
            embedding_size=embedding_size,
            spice_config=spice_config,
            use_sindy=use_sindy,
            sindy_polynomial_degree = sindy_polynomial_degree,
            sindy_ensemble_size=sindy_ensemble_size,
            )
            
        self.participant_embedding = self.setup_embedding(n_participants, embedding_size, dropout=dropout)

        self.betas['value_reward'] = self.setup_constant(embedding_size)
        self.betas['value_choice'] = self.setup_constant(embedding_size)

        # Value learning module (slow updates)
        # Can use recent reward history to modulate learning
        self.submodules_rnn['value_reward_chosen'] = self.setup_module(input_size=4 + embedding_size, dropout=dropout)  # -> 21 terms
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=3 + embedding_size, dropout=dropout)  # -> 15 terms
        self.submodules_rnn['value_choice_chosen'] = self.setup_module(input_size=3 + embedding_size, dropout=dropout) # -> 15 terms
        self.submodules_rnn['value_choice_not_chosen'] = self.setup_module(input_size=3 + embedding_size, dropout=dropout) # -> 15 terms -> 21+15+15+15 = 66 terms in total

    def forward(self, inputs, prev_state=None, batch_first=False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
        
        # perform time-invariant computations
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        
        # perform time-variant computations
        for timestep in spice_signals.timesteps:
            
            # REWARD VALUE UPDATES
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    spice_signals.rewards[timestep],
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    # self.state['value_choice'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=(
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    # self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # CHOICE VALUE UPDATES
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    # self.state['value_reward'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[timestep],
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    # self.state['value_reward'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            # BUFFER UPDATES: 
            # REWARD BUFFER UPDATES: Shift reward buffer for chosen action, keep for not chosen action (NOTE: deterministic; not learned by SPICE -> Could be made learnable: e.g. decay-rate for not chosen action)
            # CHOICE BUFFER UPDATES: Shift all buffer entries according to action
            self.state['buffer_reward_3'] = self.state['buffer_reward_2'] * spice_signals.actions[timestep] + self.state['buffer_reward_3'] * (1-spice_signals.actions[timestep])
            self.state['buffer_reward_2'] = self.state['buffer_reward_1'] * spice_signals.actions[timestep] + self.state['buffer_reward_2'] * (1-spice_signals.actions[timestep])
            self.state['buffer_reward_1'] = torch.where(spice_signals.actions[timestep]==1, spice_signals.rewards[timestep], 0) + torch.where(spice_signals.actions[timestep]==0, self.state['buffer_reward_1'], 0)  # updating buffer_reward[t-1] with reward for chosen action and keeping values for not-chosen actions
            self.state['buffer_choice_3'] = self.state['buffer_choice_2']
            self.state['buffer_choice_2'] = self.state['buffer_choice_1']
            self.state['buffer_choice_1'] = spice_signals.actions[timestep]
            
            # compute logits for current timestep
            spice_signals.logits[timestep] = self.state['value_reward'] + self.state['value_choice']

        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()

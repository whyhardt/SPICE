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
            'reward[t-1]',
            'reward[t-2]',
            'reward[t-3]',
        ],
        'value_reward_not_chosen': [
            'reward_env',
            'reward[t-1]',
            'reward[t-2]',
            'reward[t-3]',
        ],
        'value_choice_chosen': [
            'action[t-1]',
            'action[t-2]',
            'action[t-3]',
        ],
        'value_choice_not_chosen': [
            'action[t-1]',
            'action[t-2]',
            'action[t-3]',
        ],
    },
    memory_state={
        'value_reward_env': 0.,
        
        'value_reward': 0.,
        'value_choice': 0.,
        
        'reward[t-1]': 0.,
        'reward[t-2]': 0.,
        'reward[t-3]': 0.,
        
        'action[t-1]': 0.,
        'action[t-2]': 0.,
        'action[t-3]': 0.,
    },
    states_in_logit=['value_reward', 'value_choice'],
)


class SpiceModel(BaseModel):
    """
    Value learning with lag-buffer working memory.
    
    Reward buffers store per-item reward history (partial feedback: only the
    chosen item's buffer shifts each trial). Action buffers store the full
    action one-hot history. An environment-level reward tracker provides a
    shared context signal to the reward modules.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        dropout = 0.1
        
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)
        
        self.setup_module(key_module='value_reward_env', input_size=1+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_reward_chosen', input_size=5+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=4+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_choice_chosen', input_size=3+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=3+self.embedding_size, dropout=dropout)
        
    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        reward_full = spice_signals.rewards.sum(dim=-1, keepdim=True).expand_as(spice_signals.actions)
        
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:
            
            # REWARD VALUE UPDATES
            value_reward_env = self.call_module(
                key_module='value_reward_env',
                key_state='value_reward_env',
                inputs=reward_full[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    spice_signals.rewards[trial],
                    self.state['reward[t-1]'],
                    self.state['reward[t-2]'],
                    self.state['reward[t-3]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    self.state['reward[t-1]'],
                    self.state['reward[t-2]'],
                    self.state['reward[t-3]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # CHOICE VALUE UPDATES
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    self.state['action[t-1]'],
                    self.state['action[t-2]'],
                    self.state['action[t-3]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    self.state['action[t-1]'],
                    self.state['action[t-2]'],
                    self.state['action[t-3]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # BUFFER UPDATES
            self.state['action[t-3]'] = self.state['action[t-2]']
            self.state['action[t-2]'] = self.state['action[t-1]']
            self.state['action[t-1]'] = spice_signals.actions[trial]
            self.state['reward[t-3]'] = torch.where(spice_signals.actions[trial]==1, self.state['reward[t-2]'], self.state['reward[t-3]'])
            self.state['reward[t-2]'] = torch.where(spice_signals.actions[trial]==1, self.state['reward[t-1]'], self.state['reward[t-2]'])
            self.state['reward[t-1]'] = torch.where(spice_signals.actions[trial]==1, spice_signals.rewards[trial], self.state['reward[t-1]'])
            
            # Logits
            spice_signals.logits[trial] = (
                self.state['value_reward'] 
                + self.state['value_choice']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
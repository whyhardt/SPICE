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
            # 'reward[t-1]',
        ],
        'value_reward_not_chosen': [
            'reward_env',
            # 'reward[t-1]',
        ],
        'value_choice_chosen': [
            # 'action[t-1]',
        ],
        'value_choice_not_chosen': [
            # 'action[t-1]',
        ],
        'volatility_chosen': [
            'dvalue',
        ],
        'volatility_not_chosen': [],
    },
    memory_state={
        'value_reward_env': 0.,
        
        'value_reward': 0.,
        'value_choice': 0.,
        'volatility': 0.,
        
        'value_reward[t-1]': 0.,
        # 'action[t-1]': 0.,
    },
    states_in_logit=['value_reward', 'value_choice', 'volatility'],
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
        self.setup_module(key_module='value_reward_chosen', input_size=2+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=1+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_choice_chosen', input_size=0+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=0+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='volatility_chosen', input_size=1+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='volatility_not_chosen', input_size=self.embedding_size, dropout=dropout)
        
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
            
            value_reward = self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    spice_signals.rewards[trial],
                    # self.state['reward[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            value_reward += self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    value_reward_env,
                    # self.state['reward[t-1]'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # CHOICE VALUE UPDATES
            value_choice = self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[trial],
                # inputs=(self.state['action[t-1]'],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            value_choice += self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[trial],
                # inputs=(self.state['action[t-1]'],),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # VOLATILITY UPDATES
            dvalue = self.state['value_reward'] - self.state['value_reward[t-1]']

            volatility = self.call_module(
                key_module='volatility_chosen',
                key_state='volatility',
                action_mask=spice_signals.actions[trial],
                inputs=(dvalue,),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            volatility += self.call_module(
                key_module='volatility_not_chosen',
                key_state='volatility',
                action_mask=1-spice_signals.actions[trial],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )

            # BUFFER UPDATES
            # self.state['action[t-1]'] = spice_signals.actions[trial]
            self.state['value_reward[t-1]'] = self.state['value_reward']
            
            # Logits
            spice_signals.logits[trial] = (
                # self.state['value_reward']
                # + self.state['value_choice']
                # + self.state['volatility']
                value_reward
                + value_choice
                + volatility
            )
            
        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
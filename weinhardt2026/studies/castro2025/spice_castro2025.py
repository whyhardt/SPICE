from spice import BaseModel
from spice import SpiceConfig


CONFIG = SpiceConfig(
    library_setup={
        'value_reward_env': [
            'reward',
        ],
        'value_reward_chosen': [
            'reward_env',
            'reward',
            'dvalue',
            'd²value',
        ],
        'value_reward_not_chosen': [
            'reward_env',
            'dvalue',
            'd²value',
        ],
        'value_choice_chosen': [
            'dvalue',
            'd²value',
        ],
        'value_choice_not_chosen': [
            'dvalue',
            'd²value',
        ],
    },
    memory_state={
        'value_reward_env': 0.,
        
        'value_reward': 0.,
        'dvalue_reward': 0.,
        'd²value_reward': 0.,
        'buffer_value_reward': 0,
        'buffer_dvalue_reward': 0.,
        
        'value_choice': 0.,
        'dvalue_choice': 0.,
        'd²value_choice': 0.,
        'buffer_value_choice': 0,
        'buffer_dvalue_choice': 0.,
    },
    states_in_logit=['value_reward', 'value_choice'],
)


class SpiceModel(BaseModel):
    """
    Value learning with delta-value working memory buffers.
    
    Buffers store recent value changes (dV) rather than raw observations or values.
    The current state encodes the level; dV encodes the recent trajectory
    (trend, volatility), providing non-redundant temporal information.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        dropout = 0.1
        
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)
        
        self.setup_module(key_module='value_reward_env', input_size=1+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_reward_chosen', input_size=4+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=3+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_choice_chosen', input_size=2+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='value_choice_not_chosen', input_size=2+self.embedding_size, dropout=dropout)
        
    def forward(self, inputs, state=None):
        spice_signals = self.init_forward_pass(inputs, state)

        reward_full = spice_signals.rewards.sum(dim=-1, keepdim=True).expand_as(spice_signals.actions)
        
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)

        for trial in spice_signals.trials:
            
            # Snapshot pre-update values for delta computation
            self.state['buffer_value_reward'] = self.get_state()['value_reward']
            self.state['buffer_value_choice'] = self.get_state()['value_choice']
            self.state['buffer_dvalue_reward'] = self.get_state()['dvalue_reward']
            self.state['buffer_dvalue_choice'] = self.get_state()['dvalue_choice']
            
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
                    self.state['dvalue_reward'],
                    self.state['d²value_reward'],
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
                    self.state['dvalue_reward'],
                    self.state['d²value_reward'],
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
                    self.state['dvalue_choice'],
                    self.state['d²value_choice'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[trial],
                inputs=(
                    self.state['dvalue_choice'],
                    self.state['d²value_choice'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
            )
            
            # BUFFER UPDATES: compute deltas and shift
            self.state['dvalue_reward'] = self.state['value_reward'] - self.state['buffer_value_reward']
            self.state['dvalue_choice'] = self.state['value_choice'] - self.state['buffer_value_choice']
            self.state['d²value_reward'] = self.state['dvalue_reward'] - self.state['buffer_dvalue_reward']
            self.state['d²value_choice'] = self.state['dvalue_choice'] - self.state['buffer_dvalue_choice']

            # Logits
            spice_signals.logits[trial] = (
                self.state['value_reward'] 
                + self.state['value_choice']
            )

        spice_signals = self.post_forward_pass(spice_signals)
        return spice_signals.logits, self.get_state()
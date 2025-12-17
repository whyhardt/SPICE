from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) PARTICIPANT-EMBEDDING
# 2) FLEXIBLE VALUE UPDATES FOR REWARD-BASED VALUES (CHOSEN + NOT CHOSEN) 
# 3) FLEXIBLE VALUE UPDATES FOR CHOICE-BASED VALUES (CHOSEN + NOT CHOSEN)
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup = {
        'value_reward_chosen': ['reward'],  # --> n_terms = 6
        'value_reward_not_chosen': [],      # --> n_terms = 3
        'value_choice': ['choice'],  # --> n_terms = 6
        },                                  # --> n_terms_total = 15
    memory_state={
            'value_reward': 0.,
            'value_choice': 0.,
            },
)

class SpiceModel(BaseRNN):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        dropout = 0.1
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(self.n_participants, self.embedding_size, dropout=dropout)
        self.experiment_embedding = self.setup_embedding(self.n_experiments, 1)

        # set up the submodules
        self.setup_module(key_module='value_reward_chosen', input_size=1+self.embedding_size+1, dropout=dropout)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0+self.embedding_size+1, dropout=dropout)
        self.setup_module(key_module='value_choice', input_size=1+self.embedding_size+1, dropout=dropout)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
        
        # We compute now the participant embeddings and inverse noise temperatures before the for-loop because they are anyways time-invariant
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        experiment_embedding = self.experiment_embedding(spice_signals.experiment_ids)
        
        for timestep in spice_signals.timesteps:
            
            # updates for value_reward
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.rewards[timestep],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
                # activation_rnn=torch.nn.functional.leaky_relu,
                )
            
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
                # activation_rnn=torch.nn.functional.leaky_relu,
                )
            
            # updates for value_choice
            self.call_module(
                key_module='value_choice',
                key_state='value_choice',
                action_mask=None,
                inputs=spice_signals.actions[timestep],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                experiment_index=spice_signals.experiment_ids,
                experiment_embedding=experiment_embedding,
                # activation_rnn=torch.nn.functional.leaky_relu,
                )
            
            # Now keep track of the logit in the output array
            # spice_signals.logits[timestep] = self.state['value_reward'] * beta_reward + self.state['value_choice'] * beta_choice
            spice_signals.logits[timestep] = self.state['value_reward'] + self.state['value_choice']
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
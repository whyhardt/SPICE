from ..resources.estimator import SpiceConfig
from ..resources.rnn import BaseRNN


# -------------------------------------------------------------------------------
# RESCORLA-WAGNER MODEL WITH VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
    },
    memory_state={
        'value_reward': 0.5,
    }
)

class SpiceModel(BaseRNN):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=0.1)
        
        # set up the submodules
        self.setup_module(key_module='value_reward_chosen', input_size=1+self.embedding_size)
        self.setup_module(key_module='value_reward_not_chosen', input_size=0+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
        
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        
        for timestep in spice_signals.timesteps:
            
            # Let's perform the belief update for the reward-based value of the chosen option
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep, 0],
                inputs=spice_signals.rewards[timestep, 0],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
)

            # Now a RNN-module updates the not-chosen reward-based value instead of keeping it the same
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep, 0],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                )
            
            # Now keep track of this value in the output array
            spice_signals.logits[timestep] = self.state['value_reward']

        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
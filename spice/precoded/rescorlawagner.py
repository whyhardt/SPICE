from ..resources.estimator import SpiceConfig
from ..resources.model import BaseModel


# -------------------------------------------------------------------------------
# SIMPLE RESCORLA-WAGNER MODEL
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
    },
    memory_state={
        'value_reward': 0.5,
    }    
)

class SpiceModel(BaseModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=0.1)
        
        # set up the submodules
        self.setup_module(key_module='value_reward_chosen', input_size=1+self.embedding_size)
        
    def forward(self, inputs, prev_state=None):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        spice_signals = self.init_forward_pass(inputs, prev_state)
        
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        
        for timestep in spice_signals.trials:

            # Let's perform the belief update for the reward-based value of the chosen option
            # since all values are given to the rnn-module (independent of each other), the chosen value is selected by setting the action to the chosen one
            # if we would like to perform a similar update by calling a rnn-module for the non-chosen action, we would set the parameter to action=1-action.
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep, 0],
                inputs=spice_signals.rewards[timestep, 0],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                )

            # Now keep track of this value in the output array
            spice_signals.logits[timestep] = self.state['value_reward']
        
        # post-process the forward pass
        # self.state['value_reward'] = value_reward
        spice_signals = self.post_forward_pass(spice_signals)
        
        return spice_signals.logits, self.get_state()

from ..resources.estimator import SpiceConfig
from ..resources.rnn import BaseRNN

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) PARTICIPANT-EMBEDDING
# 2) FLEXIBLE VALUE UPDATE FOR CHOSEN ACTION 
# 3) VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

# The participant embedding RNN is basically the learning rate RNN with an additional participant embedding layer
CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
    },
    memory_state={
            'value_reward': 0.5,
        }
)


class SpiceModel(BaseRNN):
    
    def __init__(
        self,
        spice_config: SpiceConfig,
        n_actions: int,
        n_participants: int,
        sindy_polynomial_degree: int = 2,
        sindy_ensemble_size: int = 10,
        use_sindy: bool = False,
        **kwargs,
    ):
        super().__init__(
            spice_config=spice_config,
            n_actions=n_actions,
            n_participants=n_participants,
            use_sindy=use_sindy,
            sindy_polynomial_degree=sindy_polynomial_degree,
            sindy_ensemble_size=sindy_ensemble_size,
        )
        
        # specify here the participant embedding
        self.participant_embedding = self.setup_embedding(n_participants, self.embedding_size)
        
        # Add a scaling factor (inverse noise temperature) for each participant by passing the participant embedding
        self.betas['value_reward'] = self.setup_constant(self.embedding_size)
        
        # and here we specify the general module architecture
        # add to the input_size the embedding_size as well because we are going to pass the participant-embedding to the RNN-modules
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

        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        
        for timestep in spice_signals.trials:
            
            # Let's perform the belief update for the reward-based value of the chosen option            
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep, 0],
                inputs=spice_signals.rewards[timestep, 0],
                # add participant-embedding (for RNN-modules) and participant-index (later for SINDy-modules)
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
                )

            # Update of the not-chosen reward-based value
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep, 0],
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
                )
                        
            # Now keep track of the logit in the output array
            spice_signals.logits[timestep] = self.state['value_reward'] * self.betas['value_reward'](participant_embedding)
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
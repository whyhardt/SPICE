from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN

import torch


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) DYNAMIC LEARNING RATE
# 2) HARD-CODED REWARD-PREDICTION-ERROR FOR CHOSEN ACTION
# 3) VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    # Note: the hard-coded module is not listed in library setup because it won't be approximated by SINDy (it's already assumed to be known)
    library_setup={
        'learning_rate_reward': ['reward'],
        'value_reward_not_chosen': [],
    },
    memory_state={
            'value_reward': 0.5,
            'learning_rate_reward': 0,
    }
)


class SpiceModel(BaseRNN):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # set up the submodules
        self.submodules_rnn['learning_rate_reward'] = self.setup_module(input_size=2)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=0)

        # set up hard-coded equations
        # add here a RNN-module in the form of an hard-coded equation to compute the update for the chosen reward-based value
        # the hard-coded equation has to follow the input-output structure of the RNN-modules, i.e. the inputs are represented by the order they are given to call_module
        # self.call_module(..., inputs=(learning_rate, reward), ...)
        self.submodules_eq['value_reward_chosen'] = lambda value, inputs: value + inputs[..., 1] * (inputs[..., 0] - value)

        # add a scaling factor (i.e. inverse noise temperature) for 'value_reward'
        # self.betas = torch.nn.ParameterDict()
        # self.betas['value_reward'] = torch.nn.Parameter(torch.tensor(1.))             
        self.betas['value_reward'] = self.setup_constant()
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
                
        for timestep in spice_signals.timesteps:
            
            # Let's compute the learning rate dynamically
            # Now we have to use a sigmoid activation function on the output learning rate to constrain it to a value range of (0, 1)
            # this is necessary for two reasons:
            #   1. Preventing exploding gradients
            #   2. Remember the found equation for 'value_reward_chosen' from before: 
            #       The learning rate was scaled according to the magnitudes of the reward and the actual value 
            #       e.g. for the reward: alpha*beta -> alpha * beta = 0.3 * 3 = 0.9 and for the reward-based value: 1-alpha = 1 - 0.3 = 0.7
            #       The hard-coded equation for the reward-prediction error does not permit this flexibility. 
            #       But we can circumvein this by applying the sigmoid activation to the learning rate to staying conform with the reward-prediction error
            #       and later applying the inverse noise temperature (i.e. trainable parameter) to the updated value 
            
            learning_rate_reward, sindy_loss_learning_rate = self.call_module(
                key_module='learning_rate_reward',
                key_state='learning_rate_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    spice_signals.rewards[timestep],
                    self.state['value_reward'],
                    ),
                activation_rnn=torch.nn.functional.sigmoid,
            )
            sindy_loss_learning_rate
            
            # Let's perform the belief update for the reward-based value of the chosen option
            # no sindy loss has to be tracked because sindy won't approximate the hard-coded module (the functional structure is already known)            
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    spice_signals.rewards[timestep], 
                    learning_rate_reward,
                    ),
                )
            
            # Update of the not-chosen reward-based value
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                )
            
            # Now keep track of the logit in the output array
            # we are scaling the output logits now with the trainable beta-coefficient (inverse noise temperature)
            spice_signals.logits[timestep] = self.state['value_reward'] * self.betas['value_reward']()
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
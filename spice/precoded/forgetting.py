from spice.estimator import SpiceConfig
from spice.resources.rnn import BaseRNN


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
        
        # set up the submodules
        self.submodules_rnn['value_reward_chosen'] = self.setup_module(input_size=1)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=0)
        
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
            
            # Let's perform the belief update for the reward-based value of the chosen option
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.rewards[timestep],
                )

            # Now a RNN-module updates the not-chosen reward-based value instead of keeping it the same
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                )
            
            # Now keep track of this value in the output array
            spice_signals.logits[timestep] = self.state['value_reward']

        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
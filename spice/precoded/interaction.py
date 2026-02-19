from ..resources.estimator import SpiceConfig
from ..resources.rnn import BaseRNN


# -------------------------------------------------------------------------------
# RL MODEL WITH INTERACTIONS BETWEEN REWARD-BASED AND CHOICE-BASED MODULES
# -------------------------------------------------------------------------------

CONFIG = SpiceConfig(
    # we are enabling interactions between modules by adding the respectively other memory state variable to the library setup of each module
    library_setup = {
        'value_reward_chosen': ['reward', 'value_choice'],
        'value_reward_not_chosen': ['value_choice'],
        'value_choice_chosen': ['value_reward'],
        'value_choice_not_chosen': ['value_reward'],
    },
    memory_state={
            'value_reward': 0.5,
            'value_choice': 0.,
        },
)


class SpiceModel(BaseRNN):
    
    def __init__(
        self,
        spice_config: SpiceConfig,
        n_actions: int,
        n_participants: int,
        embedding_size: int = 32,
        sindy_polynomial_degree: int = 2,
        ensemble_size: int = 1,
        use_sindy: bool = False,
        **kwargs,
    ):
        
        super().__init__(
            n_actions=n_actions,
            n_participants=n_participants,
            embedding_size=embedding_size,
            spice_config=spice_config,
            use_sindy=use_sindy,
            sindy_polynomial_degree = sindy_polynomial_degree,
            ensemble_size=ensemble_size,
        )
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        # self.betas['value_reward'] = self.setup_constant(embedding_size=self.embedding_size)
        # self.betas['value_choice'] = self.setup_constant(embedding_size=self.embedding_size)
        
        # set up the submodules
        self.setup_module(key_module='value_reward_chosen', input_size=2+self.embedding_size)
        self.setup_module(key_module='value_reward_not_chosen', input_size=1+self.embedding_size)
        self.setup_module(key_module='value_choice_chosen', input_size=1+self.embedding_size)
        self.setup_module(key_module='value_choice_not_chosen', input_size=1+self.embedding_size)

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
        # beta_reward = self.betas['value_reward'](participant_embedding)
        # beta_choice = self.betas['value_choice'](participant_embedding)
        
        for timestep in spice_signals.trials: #, rewards_not_chosen
            
            # updates for value_reward
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep, 0],
                inputs=(
                    spice_signals.rewards[timestep, 0],
                    self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep, 0],
                inputs=(
                    # spice_signals.rewards[timestep, 0],   # enable only for Eckstein et al (2022) dataset
                    self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                )

            # updates for value_choice
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[timestep, 0],
                inputs=(self.state['value_reward']),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[timestep, 0],
                inputs=self.state['value_reward'],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # Now keep track of the logit in the output array
            spice_signals.logits[timestep] = self.state['value_reward'] + self.state['value_choice']
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
                
        return spice_signals.logits, self.get_state()

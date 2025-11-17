from spice.resources.rnn import BaseRNN
from spice.estimator import SpiceConfig
import torch

# -------------------------------------------------------------------------------
# SIMPLE RESCORLA-WAGNER MODEL
# -------------------------------------------------------------------------------

RESCORLA_WAGNER_CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
    },
    memory_state={
        'value_reward': 0.5,
    }    
)

class RescorlaWagnerRNN(BaseRNN):

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
            # since all values are given to the rnn-module (independent of each other), the chosen value is selected by setting the action to the chosen one
            # if we would like to perform a similar update by calling a rnn-module for the non-chosen action, we would set the parameter to action=1-action.
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.rewards[timestep],
                participant_index=spice_signals.participant_ids,
                )

            # Now keep track of this value in the output array
            spice_signals.logits[timestep] = self.state['value_reward']
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        # self.state['value_reward'] = value_reward
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()


# -------------------------------------------------------------------------------
# RESCORLA-WAGNER MODEL WITH VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

FORGETTING_RNN_CONFIG = SpiceConfig(
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
    },
    memory_state={
        'value_reward': 0.5,
    }
)

class ForgettingRNN(BaseRNN):
    
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


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) DYNAMIC LEARNING RATE
# 2) HARD-CODED REWARD-PREDICTION-ERROR FOR CHOSEN ACTION
# 3) VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

LEARNING_RATE_RNN_CONFIG = SpiceConfig(
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


class LearningRateRNN(BaseRNN):
    
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
    

# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) PARTICIPANT-EMBEDDING
# 2) FLEXIBLE VALUE UPDATE FOR CHOSEN ACTION 
# 3) VALUE FORGETTING OVER TIME FOR NOT-CHOSEN ACTION
# -------------------------------------------------------------------------------

# The participant embedding RNN is basically the learning rate RNN with an additional participant embedding layer
PARTICIPANT_EMBEDDING_RNN_CONFIG = SpiceConfig(
    library_setup={
        'value_reward_chosen': ['reward'],
        'value_reward_not_chosen': [],
    },
    memory_state={
            'value_reward': 0.5,
        }
)


class ParticipantEmbeddingRNN(BaseRNN):
    
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
        self.submodules_rnn['value_reward_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        
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
        
        for timestep in spice_signals.timesteps:
            
            # Let's perform the belief update for the reward-based value of the chosen option            
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.rewards[timestep],
                # add participant-embedding (for RNN-modules) and participant-index (later for SINDy-modules) 
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # Update of the not-chosen reward-based value
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
                )
                        
            # Now keep track of the logit in the output array
            spice_signals.logits[timestep] = self.state['value_reward'] * self.betas['value_reward'](participant_embedding)
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) PARTICIPANT-EMBEDDING
# 2) FLEXIBLE VALUE UPDATES FOR REWARD-BASED VALUES (CHOSEN + NOT CHOSEN) 
# 3) FLEXIBLE VALUE UPDATES FOR CHOICE-BASED VALUES (CHOSEN + NOT CHOSEN)
# -------------------------------------------------------------------------------

CHOICE_CONFIG = SpiceConfig(
    library_setup = {
        'value_reward_chosen': ['reward'],  # --> n_terms = 6
        'value_reward_not_chosen': [],      # --> n_terms = 3
        'value_choice_chosen': [],          # --> n_terms = 3
        'value_choice_not_chosen': [],      # --> n_terms = 3 
        },                                  # --> n_terms_total = 15
    memory_state={
            'value_reward': 0.5,
            'value_choice': 0.,
            },
)

class ChoiceRNN(BaseRNN):
    
    def __init__(
        self,
        spice_config: SpiceConfig,
        n_actions: int,
        n_participants: int,
        sindy_polynomial_degree: int = 2,
        sindy_ensemble_size: int = 1,
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
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(n_participants, self.embedding_size)
        
        # Inverse noise temperatures for scaling each variable in the memory state for each participant
        # self.betas['value_reward'] = self.setup_constant()#embedding_size=self.embedding_size)
        # self.betas['value_choice'] = self.setup_constant()#embedding_size=self.embedding_size)
        
        # set up the submodules
        self.submodules_rnn['value_reward_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        self.submodules_rnn['value_choice_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        self.submodules_rnn['value_choice_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        
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
        # beta_reward = self.betas['value_reward']()#participant_embedding)
        # beta_choice = self.betas['value_choice']()#participant_embedding)
        
        for timestep in spice_signals.timesteps:
            
            # updates for value_reward
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=spice_signals.rewards[timestep],
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )
            
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,                
                )
            
            # updates for value_choice
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )
            
            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[timestep],
                inputs=None,
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # Now keep track of the logit in the output array
            # spice_signals.logits[timestep] = self.state['value_reward'] * beta_reward + self.state['value_choice'] * beta_choice
            spice_signals.logits[timestep] = self.state['value_reward'] + self.state['value_choice']
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()
    

# -------------------------------------------------------------------------------
# RL MODEL WITH INTERACTIONS BETWEEN REWARD-BASED AND CHOICE-BASED MODULES
# -------------------------------------------------------------------------------

INTERACTION_CONFIG = SpiceConfig(
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


class InteractionRNN(BaseRNN):
    
    def __init__(
        self,
        spice_config: SpiceConfig,
        n_actions: int,
        n_participants: int,
        embedding_size: int = 32,
        sindy_polynomial_degree: int = 2,
        sindy_ensemble_size: int = 1,
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
            sindy_ensemble_size=sindy_ensemble_size,
        )
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        # self.betas['value_reward'] = self.setup_constant(embedding_size=self.embedding_size)
        # self.betas['value_choice'] = self.setup_constant(embedding_size=self.embedding_size)
        
        # set up the submodules
        self.submodules_rnn['value_reward_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['value_choice_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['value_choice_not_chosen'] = self.setup_module(input_size=1+self.embedding_size)

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
        
        for timestep in spice_signals.timesteps: #, rewards_not_chosen
            
            # updates for value_reward
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    spice_signals.rewards[timestep], 
                    self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )
            
            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=(
                    # spice_signals.rewards[timestep],   # enable only for Eckstein et al (2022) dataset
                    self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                )
            
            # updates for value_choice
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[timestep],
                inputs=(self.state['value_reward']),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                # activation_rnn=torch.nn.functional.sigmoid,
                )
            
            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[timestep],
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


# -------------------------------------------------------------------------------
# RL MODEL WITH 
# 1) INTERACTIONS BETWEEN REWARD-BASED AND CHOICE-BASED MODULES
# 2) BUFFER-BASED REWARD AND CHOICE MEMORY FOR 3 TIMESTEPS (could be easily extended)
# -------------------------------------------------------------------------------

BUFFER_WORKING_MEMORY_CONFIG = SpiceConfig(
    library_setup={
        # Value learning can depend on recent reward sequence (working memory)
        'value_reward_chosen': [
            'reward[t]',           
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            'value_choice',
        ],
        'value_reward_not_chosen': [
            'reward[t]',
            'reward[t-1]', 
            'reward[t-2]',
            'reward[t-3]',
            'value_choice',
            ],
        'value_choice_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            'value_reward',
            ],
        'value_choice_not_chosen': [
            'choice[t-1]', 
            'choice[t-2]',
            'choice[t-3]',
            'value_reward',
            ],
    },
    
    memory_state = {
        'value_reward': 0.5,      # reward value (enables slow learning)
        'value_choice': 0.0,      # choice value (enables slow learning)
        'buffer_reward_1': 0.5,   # t-1 reward
        'buffer_reward_2': 0.5,   # t-2 reward
        'buffer_reward_3': 0.5,   # t-3 reward
        'buffer_choice_1': 0.5,   # t-1 choice
        'buffer_choice_2': 0.5,   # t-2 choice
        'buffer_choice_3': 0.5,   # t-3 choice
    }
)


class BufferWorkingMemoryRNN(BaseRNN):
    """
    Working memory as explicit buffer of recent rewards.
    
    Key difference from value learning:
    - Stores individual past rewards (not aggregated statistics)
    - Fixed capacity (buffer size)
    - Perfect memory for items in buffer
    - Items fall out of buffer (discrete forgetting)
    """
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        spice_config: SpiceConfig,
        embedding_size: int = 32,
        dropout: float = 0.5,
        sindy_polynomial_degree: int = 2,
        sindy_ensemble_size: int = 10,
        use_sindy: bool = False,
        **kwargs):
        super().__init__(
            n_actions=n_actions,
            n_participants=n_participants,
            embedding_size=embedding_size,
            spice_config=spice_config,
            use_sindy=use_sindy,
            sindy_polynomial_degree = sindy_polynomial_degree,
            sindy_ensemble_size=sindy_ensemble_size,
            )
            
        self.participant_embedding = self.setup_embedding(n_participants, embedding_size, dropout=dropout)

        self.betas['value_reward'] = self.setup_constant(embedding_size)
        self.betas['value_choice'] = self.setup_constant(embedding_size)

        # Value learning module (slow updates)
        # Can use recent reward history to modulate learning
        self.submodules_rnn['value_reward_chosen'] = self.setup_module(input_size=5 + embedding_size, dropout=dropout)
        self.submodules_rnn['value_reward_not_chosen'] = self.setup_module(input_size=4+1 + embedding_size, dropout=dropout)
        self.submodules_rnn['value_choice_chosen'] = self.setup_module(input_size=4 + embedding_size, dropout=dropout)
        self.submodules_rnn['value_choice_not_chosen'] = self.setup_module(input_size=4 + embedding_size, dropout=dropout)

    def forward(self, inputs, prev_state=None, batch_first=False):
        spice_signals = self.init_forward_pass(inputs, prev_state, batch_first)
        
        # perform time-invariant computations
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        beta_reward = self.betas['value_reward'](participant_embedding)
        beta_choice = self.betas['value_choice'](participant_embedding)
        
        # perform time-variant computations
        for timestep in spice_signals.timesteps:
            
            # REWARD VALUE UPDATES
            self.call_module(
                key_module='value_reward_chosen',
                key_state='value_reward',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    spice_signals.rewards[timestep],
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    self.state['value_choice'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            self.call_module(
                key_module='value_reward_not_chosen',
                key_state='value_reward',
                action_mask=1-spice_signals.actions[timestep],
                inputs=(
                    spice_signals.rewards[timestep],
                    self.state['buffer_reward_1'],  # Recent reward history
                    self.state['buffer_reward_2'],
                    self.state['buffer_reward_3'],
                    self.state['value_choice'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            # CHOICE VALUE UPDATES
            self.call_module(
                key_module='value_choice_chosen',
                key_state='value_choice',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    self.state['value_reward'],
                ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            self.call_module(
                key_module='value_choice_not_chosen',
                key_state='value_choice',
                action_mask=1-spice_signals.actions[timestep],
                inputs=(
                    self.state['buffer_choice_1'],  # Recent choice history
                    self.state['buffer_choice_2'],
                    self.state['buffer_choice_3'],
                    self.state['value_reward'],
                    ),
                participant_index=spice_signals.participant_ids,
                participant_embedding=participant_embedding,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            # BUFFER UPDATES: 
            # REWARD BUFFER UPDATES: Shift reward buffer for chosen action, keep for not chosen action (NOTE: deterministic; not learned by SPICE -> Could be made learnable: e.g. decay-rate for not chosen action)
            # CHOICE BUFFER UPDATES: Shift all buffer entries according to action
            self.state['buffer_reward_3'] = self.state['buffer_reward_2'] * spice_signals.actions[timestep] + self.state['buffer_reward_3'] * (1-spice_signals.actions[timestep])
            self.state['buffer_reward_2'] = self.state['buffer_reward_1'] * spice_signals.actions[timestep] + self.state['buffer_reward_2'] * (1-spice_signals.actions[timestep])
            self.state['buffer_reward_1'] = torch.where(spice_signals.actions[timestep]==1, spice_signals.rewards[timestep], 0) + torch.where(spice_signals.actions[timestep]==0, self.state['buffer_reward_1'], 0)  # updating buffer_reward[t-1] with reward for chosen action and keeping values for not-chosen actions
            self.state['buffer_choice_3'] = self.state['buffer_choice_2']
            self.state['buffer_choice_2'] = self.state['buffer_choice_1']
            self.state['buffer_choice_1'] = spice_signals.actions[timestep]
            
            # compute logits for current timestep
            spice_signals.logits[timestep] = self.state['value_reward'] * beta_reward + self.state['value_choice'] * beta_choice

        spice_signals = self.post_forward_pass(spice_signals, batch_first)
        
        return spice_signals.logits, self.get_state()

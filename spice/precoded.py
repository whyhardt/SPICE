from spice.resources.rnn import BaseRNN
from spice.estimator import SpiceConfig
import torch


RESCOLA_WAGNER_CONFIG = SpiceConfig(
    library_setup={
        'x_value_reward_chosen': ['c_reward'],
    },
    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
    },
    control_parameters=['c_action', 'c_reward'],
    rnn_modules=['x_value_reward_chosen']
)


class RescorlaWagnerRNN(BaseRNN):
    init_values={
        'x_value_reward': 0.5,
    }

    def __init__(
        self,
        n_actions,
        **kwargs,
    ):   
        super(RescorlaWagnerRNN, self).__init__(n_actions=n_actions)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = inputs
        
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            
            # Let's perform the belief update for the reward-based value of the chosen option
            # since all values are given to the rnn-module (independent of each other), the chosen value is selected by setting the action to the chosen one
            # if we would like to perform a similar update by calling a rnn-module for the non-chosen action, we would set the parameter to action=1-action.
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=reward,
                )

            # and keep the value of the not-chosen option unchanged
            next_value_reward_not_chosen = self.state['x_value_reward'] * (1-action)
            
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen  # memory state = (0.8, 0.3) <- next_value = (0.8, 0) + (0, 0.3)
            
            # Now keep track of this value in the output array
            logits[timestep] = self.state['x_value_reward']
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        # self.state['x_value_reward'] = value_reward
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
    

FORGETTING_RNN_CONFIG = SpiceConfig(
    # Add already here the new module and update the library and filter setup.
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen'],
    
    control_parameters=['c_action', 'c_reward'],

    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup={
        'x_value_reward_chosen': ['c_reward'],
        'x_value_reward_not_chosen': [],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
    },
)

class ForgettingRNN(BaseRNN):
    
    init_values = {
        'x_value_reward': 0.5,
    }
    
    def __init__(
        self,
        n_actions,
        **kwargs,
    ):
        super(ForgettingRNN, self).__init__(n_actions=n_actions)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = inputs
                
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            
            # Let's perform the belief update for the reward-based value of the chosen option
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=reward,
                )

            # Now a RNN-module updates the not-chosen reward-based value instead of keeping it the same
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                )
            
            # keep track of the updated value in the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            
            # Now keep track of this value in the output array
            logits[timestep] = self.state['x_value_reward']
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
    

LEARNING_RATE_RNN_CONFIG = SpiceConfig(
    rnn_modules=['x_learning_rate_reward', 'x_value_reward_not_chosen'],
    
    control_parameters=['c_action', 'c_reward'],

    library_setup={
        'x_learning_rate_reward': ['c_reward'],
        'x_value_reward_not_chosen': [],
    },

    filter_setup={
        'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
    },
)


class LearningRateRNN(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_learning_rate_reward': 0,
    }
    
    def __init__(
        self,
        n_actions,
        **kwargs,
    ):
        super(LearningRateRNN, self).__init__(n_actions=n_actions)
        
        # set up the submodules
        self.submodules_rnn['x_learning_rate_reward'] = self.setup_module(input_size=2)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0)

        # set up hard-coded equations
        # add here a RNN-module in the form of an hard-coded equation to compute the update for the chosen reward-based value
        # the hard-coded equation has to follow the input-output structure of the RNN-modules, i.e. the inputs are represented by the order they are given to call_module
        # self.call_module(..., inputs=(learning_rate, reward), ...)
        self.submodules_eq['x_value_reward_chosen'] = lambda value, inputs: value + inputs[..., 1] * (inputs[..., 0] - value)

        # add a scaling factor (i.e. inverse noise temperature) for 'x_value_reward'
        # self.betas = torch.nn.ParameterDict()
        # self.betas['x_value_reward'] = torch.nn.Parameter(torch.tensor(1.))             
        self.betas['x_value_reward'] = self.setup_constant()
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = inputs
                
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('c_value_reward', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            self.record_signal('x_learning_rate_reward', self.state['x_learning_rate_reward'])
            
            # Let's compute the learning rate dynamically
            # Now we have to use a sigmoid activation function on the output learning rate to constrain it to a value range of (0, 1)
            # this is necessary for two reasons:
            #   1. Preventing exploding gradients
            #   2. Remember the found equation for 'x_value_reward_chosen' from before: 
            #       The learning rate was scaled according to the magnitudes of the reward and the actual value 
            #       e.g. for the reward: alpha*beta -> alpha * beta = 0.3 * 3 = 0.9 and for the reward-based value: 1-alpha = 1 - 0.3 = 0.7
            #       The hard-coded equation for the reward-prediction error does not permit this flexibility. 
            #       But we can circumvein this by applying the sigmoid activation to the learning rate to staying conform with the reward-prediction error
            #       and later applying the inverse noise temperature (i.e. trainable parameter) to the updated value 
            learning_rate_reward = self.call_module(
                key_module='x_learning_rate_reward',
                key_state='x_learning_rate_reward',
                action=action,
                inputs=(reward, self.state['x_value_reward']),
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            # Let's perform the belief update for the reward-based value of the chosen option            
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(reward, learning_rate_reward),
                )
            
            # Update of the not-chosen reward-based value
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                )
            
            # updating the memory state
            self.state['x_learning_rate_reward'] = learning_rate_reward
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward']()
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
    

# The participant embedding RNN is basically the learning rate RNN with an additional participant embedding layer
PARTICIPANT_EMBEDDING_RNN_CONFIG = SpiceConfig(
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen'],
    
    control_parameters=['c_action', 'c_reward'],

    library_setup={
        'x_value_reward_chosen': ['c_reward'],
        'x_value_reward_not_chosen': [],
    },

    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
    },
)


class ParticipantEmbeddingRNN(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
        }
    
    def __init__(
        self,
        n_actions,
        # add an additional inputs to set the number of participants in your data
        n_participants,
        embedding_size: int = 8,
        **kwargs,
    ):
        
        super(ParticipantEmbeddingRNN, self).__init__(n_actions=n_actions, embedding_size=embedding_size)
        
        # specify here the participant embedding
        self.participant_embedding = self.setup_embedding(n_participants, embedding_size, dropout=0.)
        
        # Add a scaling factor (inverse noise temperature) for each participant.
        self.betas['x_value_reward'] = self.setup_constant(self.embedding_size)
        
        # and here we specify the general architecture
        # add to the input_size the embedding_size as well because we are going to pass the participant-embedding to the RNN-modules
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = inputs
        participant_id, _ = ids

        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            
            # Let's perform the belief update for the reward-based value of the chosen option            
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(reward),
                # add participant-embedding (for RNN-modules) and participant-index (later for SINDy-modules) 
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # Update of the not-chosen reward-based value
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                )
            
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding)
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
    

CHOICE_CONFIG = SpiceConfig(
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
    control_parameters=['c_action', 'c_reward'],
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup = {
        'x_value_reward_chosen': ['c_reward'],
        'x_value_reward_not_chosen': [],
        'x_value_choice_chosen': [],
        'x_value_choice_not_chosen': [],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup = {
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }    
)


class ChoiceRNN(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
        }
    
    def __init__(
        self,
        n_actions,
        n_participants,
        embedding_size=32,
        dropout=0.5,
        **kwargs,
    ):
        
        super().__init__(n_actions=n_actions, embedding_size=embedding_size)
        
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(n_participants, embedding_size, dropout=dropout)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())
        self.betas['x_value_choice'] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=0+self.embedding_size)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, embedding_variables, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = inputs
        participant_id, _ = embedding_variables
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        beta_reward = self.betas['x_value_reward'](participant_embedding)
        beta_choice = self.betas['x_value_choice'](participant_embedding)
        
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
            self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
            self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])
            
            # updates for x_value_reward
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(reward),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                )
            
            # updates for x_value_choice
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # updating the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
            
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * beta_reward + self.state['x_value_choice'] * beta_choice
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()
    

WEINHARDT_2025_CONFIG = SpiceConfig(
    rnn_modules=['x_value_reward_chosen', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
    control_parameters=['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice'],
    # The new module which handles the not-chosen value, does not need any additional inputs except for the value
    library_setup = {
        'x_value_reward_chosen': ['c_reward_chosen', 'c_value_choice'],
        'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
    },

    # Further, the new module should be applied only to the not-chosen values
    filter_setup = {
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }    
)


class Weinhardt2025RNN(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
        }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        embedding_size = 32,
        dropout = 0.5,
        leaky_relu = 0.01,
        device = torch.device('cpu'),
        **kwargs,
    ):
        
        super().__init__(n_actions=n_actions, device=device, n_participants=n_participants, embedding_size=embedding_size)
                
        # set up the participant-embedding layer
        self.participant_embedding = self.setup_embedding(num_embeddings=self.n_participants, embedding_size=self.embedding_size, dropout=dropout)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas['x_value_reward'] = self.setup_constant(embedding_size=self.embedding_size, leaky_relu=leaky_relu)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size=self.embedding_size, leaky_relu=leaky_relu)
        
        # set up the submodules
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=2+self.embedding_size)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=1+self.embedding_size)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=1+self.embedding_size)
    
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        input_variables, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = input_variables
        participant_id, _ = ids
        
        # derive more observations
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        # rewards_not_chosen = ((1-actions) * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        for timestep, action, reward_chosen in zip(timesteps, actions, rewards_chosen): #, rewards_not_chosen
            
            # record the current memory state and control inputs to the modules for SINDy training
            if not self.training and len(self.submodules_sindy)==0:
                self.record_signal('c_action', action)
                self.record_signal('c_reward_chosen', reward_chosen)
                # self.record_signal('c_reward_not_chosen', reward_not_chosen)
                self.record_signal('c_value_reward', self.state['x_value_reward'])
                self.record_signal('c_value_choice', self.state['x_value_choice'])
                self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
                self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])
            
            # updates for x_value_reward
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    reward_chosen, 
                    self.state['x_value_choice'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=(
                    reward_chosen, 
                    # reward_not_chosen, 
                    self.state['x_value_choice'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                )
            
            # updates for x_value_choice
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=(self.state['x_value_reward']),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=(self.state['x_value_reward']),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )
            
            # updating the memory state
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
             
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice'] * self.betas['x_value_choice'](participant_embedding)
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
                
        return logits, self.get_state()
    
    
BUFFER_WORKING_MEMORY_CONFIG = SpiceConfig(
    rnn_modules=[
        'x_value_reward_chosen',     # Slow value learning (long-term)
        'x_value_reward_not_chosen',
        'x_value_choice_chosen',
        'x_value_choice_not_chosen',
    ],

    control_parameters=[
        'c_action',
        'c_reward',
        'c_reward_t_minus_1',  # Previous reward (from buffer)
        'c_reward_t_minus_2',  # 2 trials ago (from buffer)
        'c_reward_t_minus_3',  # 3 trials ago (from buffer)        
        'c_choice_t_minus_1',  # Previous choice (from buffer)
        'c_choice_t_minus_2',  # 2 trials ago (from buffer)
        'c_choice_t_minus_3',  # 3 trials ago (from buffer)
        'c_value_reward',
        'c_value_choice',
    ],

    library_setup={
        # Value learning can depend on recent reward sequence (working memory)
        'x_value_reward_chosen': [
            'c_reward',           
            'c_reward_t_minus_1', 
            'c_reward_t_minus_2',
            'c_reward_t_minus_3',
            'c_value_choice',
        ],
        'x_value_reward_not_chosen': [
            'c_reward', 
            'c_reward_t_minus_1', 
            'c_reward_t_minus_2',
            'c_reward_t_minus_3',
            'c_value_choice',
            ],
        'x_value_choice_chosen': [
            'c_choice_t_minus_1', 
            'c_choice_t_minus_2',
            'c_choice_t_minus_3',
            'c_value_reward',
            ],
        'x_value_choice_not_chosen': [
            'c_choice_t_minus_1', 
            'c_choice_t_minus_2',
            'c_choice_t_minus_3',
            'c_value_reward',
            ],
    },
    
    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    },
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

    init_values = {
        'x_value_reward': 0.5,      # Long-term value (slow learning)
        'x_value_choice': 0.0,
        'x_reward_buffer_1': 0.5,   # t-1 reward
        'x_reward_buffer_2': 0.5,   # t-2 reward  
        'x_reward_buffer_3': 0.5,   # t-3 reward
        'x_choice_buffer_1': 0.5,   # t-1 choice
        'x_choice_buffer_2': 0.5,   # t-2 choice  
        'x_choice_buffer_3': 0.5,   # t-3 choice
    }

    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        embedding_size: int = 32,
        dropout: float = 0.5,
        enable_sindy_reg: bool = False,
        sindy_config: dict = None,
        sindy_polynomial_degree: int = 2,
        use_sindy: bool = False,
        **kwargs):
        super().__init__(
            n_actions=n_actions,
            n_participants=n_participants,
            embedding_size=embedding_size,
            enable_sindy_reg=enable_sindy_reg,
            sindy_config=sindy_config,
            use_sindy=use_sindy,
            )
            
        self.sindy_polynomial_degree = sindy_polynomial_degree

        self.participant_embedding = self.setup_embedding(n_participants, embedding_size, dropout=dropout)

        self.betas['x_value_reward'] = self.setup_constant(embedding_size)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size)

        # Value learning module (slow updates)
        # Can use recent reward history to modulate learning
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=5 + embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=5 + embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=4 + embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=4 + embedding_size, dropout=dropout)

        # Setup differentiable SINDy coefficients if enabled
        if enable_sindy_reg:
            self.setup_sindy_coefficients(polynomial_degree=sindy_polynomial_degree)

    def forward(self, inputs, prev_state=None, batch_first=False):
        input_variables, ids, logits, timesteps, sindy_loss_timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = input_variables
        participant_id, _ = ids
        
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)

        participant_embedding = self.participant_embedding(participant_id[:, 0].int())

        for timestep, action, reward_chosen in zip(timesteps, actions, rewards_chosen):
            
            # VALUE UPDATE: Uses buffer contents (working memory)
            next_value_reward_chosen, sindy_loss_module = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],  # Recent reward history
                    self.state['x_reward_buffer_2'],
                    self.state['x_reward_buffer_3'],
                    self.state['x_value_choice'],
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            sindy_loss_timesteps[timestep] = sindy_loss_timesteps[timestep] + sindy_loss_module

            next_value_reward_not_chosen, sindy_loss_module = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],  # Recent reward history
                    self.state['x_reward_buffer_2'],
                    self.state['x_reward_buffer_3'],
                    self.state['x_value_choice'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            sindy_loss_timesteps[timestep] = sindy_loss_timesteps[timestep] + sindy_loss_module
            
            # CHOICE UPDATE
            next_value_choice_chosen, sindy_loss_module = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=(
                    self.state['x_choice_buffer_1'],  # Recent choice history
                    self.state['x_choice_buffer_2'],
                    self.state['x_choice_buffer_3'],
                    self.state['x_value_reward'],
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            sindy_loss_timesteps[timestep] = sindy_loss_timesteps[timestep] + sindy_loss_module
            
            next_value_choice_not_chosen, sindy_loss_module = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=(
                    self.state['x_choice_buffer_1'],  # Recent choice history
                    self.state['x_choice_buffer_2'],
                    self.state['x_choice_buffer_3'],
                    self.state['x_value_reward'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            sindy_loss_timesteps[timestep] = sindy_loss_timesteps[timestep] + sindy_loss_module
            
            # STATE UPDATE
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen

            # BUFFER UPDATE: Shift buffer for chosen action (deterministic, not learned by SPICE)
            self.state['x_reward_buffer_3'] = self.state['x_reward_buffer_2'] * action + self.state['x_reward_buffer_3'] * (1-action)
            self.state['x_reward_buffer_2'] = self.state['x_reward_buffer_1'] * action + self.state['x_reward_buffer_2'] * (1-action)
            self.state['x_reward_buffer_1'] = reward_chosen * action + self.state['x_reward_buffer_1'] * (1-action)
            self.state['x_choice_buffer_3'] = self.state['x_choice_buffer_2']
            self.state['x_choice_buffer_2'] = self.state['x_choice_buffer_1']
            self.state['x_choice_buffer_1'] = action
            
            # Decision combines value (long-term) with recent rewards (working memory)
            # Could add direct influence of buffer on choice
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice'] * self.betas['x_value_choice'](participant_embedding)

        logits, sindy_loss_timesteps = self.post_forward_pass(logits, sindy_loss_timesteps, batch_first)
        
        return logits, self.get_state(), sindy_loss_timesteps


BUFFER_WORKING_MEMORY_2_CONFIG = SpiceConfig(
    rnn_modules=[
        'x_value_reward_chosen',
        'x_value_reward_not_chosen',
        'x_value_choice_chosen',
        'x_value_choice_not_chosen',
        'x_update_reward_buffer_chosen',    
        'x_update_reward_buffer_not_chosen',
        'x_update_choice_buffer_chosen',    
        'x_update_choice_buffer_not_chosen',
    ],

    control_parameters=[
        'c_action',
        'c_reward',
        'c_reward_t_minus_1',
        'c_reward_t_minus_2',
        'c_reward_t_minus_3',
        'c_choice_t_minus_1',
        'c_choice_t_minus_2',
        'c_choice_t_minus_3',
        'c_value_reward',
        'c_value_choice',
    ],

    library_setup={
        'x_value_reward_chosen': [
            'c_reward',
            'c_reward_t_minus_1',
            'c_reward_t_minus_2',
            'c_reward_t_minus_3',
            'c_value_choice',
        ],
        'x_value_reward_not_chosen': [
            'c_reward',
            'c_reward_t_minus_1',
            'c_reward_t_minus_2',
            'c_reward_t_minus_3',
            'c_value_choice',
        ],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
        
        # TODO: Will probably make problems when recording signals because of static assignment vs dynamic assignment in forward pass
        # Reward buffer update: receives all buffer values + position encoding
        'x_update_reward_buffer_chosen': [
            'c_reward',
            'c_reward_t_minus_1',
            'c_reward_t_minus_2',
            'c_reward_t_minus_3',
        ],
        'x_update_reward_buffer_not_chosen': [
            'c_reward_t_minus_1',
            'c_reward_t_minus_2',
            'c_reward_t_minus_3',
        ],
        # Choice buffer update: receives all choice buffer values + position encoding
        'x_update_choice_buffer_chosen': [
            'c_choice_t_minus_1',
            'c_choice_t_minus_2',
            'c_choice_t_minus_3',
        ],
        'x_update_choice_buffer_not_chosen': [
            'c_choice_t_minus_1',
            'c_choice_t_minus_2',
            'c_choice_t_minus_3',
        ],
    },

    filter_setup={
        'x_value_reward_chosen': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
        'x_update_reward_buffer_chosen': ['c_action', 1, True],
        'x_update_reward_buffer_not_chosen': ['c_action', 0, True],
        'x_update_choice_buffer_chosen': ['c_action', 1, True],
        'x_update_choice_buffer_not_chosen': ['c_action', 0, True],
    },
)


class BufferWorkingMemoryRNN_2(BaseRNN):
    """
    Working memory as explicit buffer of recent rewards and choices.

    Key difference from value learning:
    - Stores individual past rewards and choices (not aggregated statistics)
    - Fixed capacity (buffer size)
    - Dynamic decay based on buffer contents and position
    - Position-dependent updates via learnable encodings
    """

    init_values = {
        'x_value_reward': 0.5,      # Long-term value (slow learning)
        'x_value_choice': 0.5,
        'x_reward_buffer_1': 0.5,   # t-1 reward
        'x_reward_buffer_2': 0.5,   # t-2 reward
        'x_reward_buffer_3': 0.5,   # t-3 reward
        'x_choice_buffer_1': 0.5,   # t-1 choice
        'x_choice_buffer_2': 0.5,   # t-2 choice
        'x_choice_buffer_3': 0.5,   # t-3 choice
    }

    def __init__(self, n_actions: int, n_participants: int,
                embedding_size: int = 32, **kwargs):
        super().__init__(n_actions=n_actions, n_participants=n_participants, embedding_size=embedding_size)

        self.participant_embedding = self.setup_embedding(n_participants, embedding_size, dropout=0.5)
        
        self.betas['x_value_reward'] = self.setup_constant(embedding_size)
        self.betas['x_value_choice'] = self.setup_constant(embedding_size)

        # Learnable position encodings for buffer positions (participant-specific)
        self.betas['x_reward_buffer_1'] = self.setup_constant(embedding_size, activation=torch.nn.Sigmoid, kwargs_activation={})
        self.betas['x_reward_buffer_2'] = self.setup_constant(embedding_size, activation=torch.nn.Sigmoid, kwargs_activation={})
        self.betas['x_reward_buffer_3'] = self.setup_constant(embedding_size, activation=torch.nn.Sigmoid, kwargs_activation={})
        self.betas['x_choice_buffer_1'] = self.setup_constant(embedding_size, activation=torch.nn.Sigmoid, kwargs_activation={})
        self.betas['x_choice_buffer_2'] = self.setup_constant(embedding_size, activation=torch.nn.Sigmoid, kwargs_activation={})
        self.betas['x_choice_buffer_3'] = self.setup_constant(embedding_size, activation=torch.nn.Sigmoid, kwargs_activation={})
        
        # Value learning modules (use buffer contents)
        self.submodules_rnn['x_value_reward_chosen'] = self.setup_module(input_size=5 + embedding_size)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=5 + embedding_size)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=1 + embedding_size)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=1 + embedding_size)

        # General buffer update modules
        # TODO: Perhaps add values as inputs
        self.submodules_rnn['x_update_reward_buffer_chosen'] = self.setup_module(input_size=4 + embedding_size + 1)
        self.submodules_rnn['x_update_reward_buffer_not_chosen'] = self.setup_module(input_size=4 + embedding_size + 1)
        self.submodules_rnn['x_update_choice_buffer_chosen'] = self.setup_module(input_size=3 + embedding_size + 1)
        self.submodules_rnn['x_update_choice_buffer_not_chosen'] = self.setup_module(input_size=3 + embedding_size + 1)

    def forward(self, inputs, prev_state=None, batch_first=False):
        input_variables, ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, _, _ = input_variables
        participant_id, _ = ids

        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)

        participant_embedding = self.participant_embedding(participant_id[:, 0].int())

        # Compute learnable position encodings from participant embedding
        pos_1_reward = self.betas['x_reward_buffer_1'](participant_embedding).unsqueeze(1).repeat(1, 1, self._n_actions)
        pos_2_reward = self.betas['x_reward_buffer_2'](participant_embedding).unsqueeze(1).repeat(1, 1, self._n_actions)
        pos_3_reward = self.betas['x_reward_buffer_3'](participant_embedding).unsqueeze(1).repeat(1, 1, self._n_actions)
        pos_1_choice = self.betas['x_choice_buffer_1'](participant_embedding).unsqueeze(1).repeat(1, 1, self._n_actions)
        pos_2_choice = self.betas['x_choice_buffer_2'](participant_embedding).unsqueeze(1).repeat(1, 1, self._n_actions)
        pos_3_choice = self.betas['x_choice_buffer_3'](participant_embedding).unsqueeze(1).repeat(1, 1, self._n_actions)

        for timestep, action, reward_chosen in zip(timesteps, actions, rewards_chosen):

            # Record for SPICE (including buffer contents as control signals)
            if not self.training and len(self.submodules_sindy) == 0:
                self.record_signal('c_action', action)
                self.record_signal('c_reward', reward_chosen)
                self.record_signal('c_value_reward', self.state['x_value_reward'])
                self.record_signal('c_value_choice', self.state['x_value_choice'])
                # Buffer contents are control signals
                self.record_signal('c_reward_t_minus_1', self.state['x_reward_buffer_1'])
                self.record_signal('c_reward_t_minus_2', self.state['x_reward_buffer_2'])
                self.record_signal('c_reward_t_minus_3', self.state['x_reward_buffer_3'])
                self.record_signal('x_value_reward_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
                self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])

            # VALUE UPDATE: Uses buffer contents (working memory)
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],
                    self.state['x_reward_buffer_2'],
                    self.state['x_reward_buffer_3'],
                    self.state['x_value_choice'],
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],
                    self.state['x_reward_buffer_2'],
                    self.state['x_reward_buffer_3'],
                    self.state['x_value_choice'],
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # CHOICE UPDATE
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=(self.state['x_value_reward'],),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )

            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=(self.state['x_value_reward'],),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # DYNAMIC BUFFER UPDATE - Chosen action (same module, different positions)
            # Position 1 (t-1): Gets new reward, modulated by position encoding
            next_buffer_1_chosen = self.call_module(
                key_module='x_update_reward_buffer_chosen',
                key_state='x_reward_buffer_1',
                action=action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_2'],
                    self.state['x_reward_buffer_3'],
                    pos_1_reward,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 2 (t-2): Shifts from t-1, modulated by position encoding
            next_buffer_2_chosen = self.call_module(
                key_module='x_update_reward_buffer_chosen',
                key_state='x_reward_buffer_2',
                action=action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],
                    self.state['x_reward_buffer_3'],
                    pos_2_reward,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 3 (t-3): Shifts from t-2, modulated by position encoding
            next_buffer_3_chosen = self.call_module(
                key_module='x_update_reward_buffer_chosen',
                key_state='x_reward_buffer_3',
                action=action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],
                    self.state['x_reward_buffer_2'],
                    pos_3_reward,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # DYNAMIC BUFFER DECAY - Not chosen action (same module, different positions)
            # Position 1: Decay modulated by position encoding
            next_buffer_1_not_chosen = self.call_module(
                key_module='x_update_reward_buffer_not_chosen',
                key_state='x_reward_buffer_1',
                action=1-action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_2'],
                    self.state['x_reward_buffer_3'],
                    pos_1_reward,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 2: Decay modulated by position encoding
            next_buffer_2_not_chosen = self.call_module(
                key_module='x_update_reward_buffer_not_chosen',
                key_state='x_reward_buffer_2',
                action=1-action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],
                    self.state['x_reward_buffer_3'],
                    pos_2_reward,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 3: Decay modulated by position encoding
            next_buffer_3_not_chosen = self.call_module(
                key_module='x_update_reward_buffer_not_chosen',
                key_state='x_reward_buffer_3',
                action=1-action,
                inputs=(
                    reward_chosen,
                    self.state['x_reward_buffer_1'],
                    self.state['x_reward_buffer_2'],
                    pos_3_reward,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )
            
            # DYNAMIC BUFFER UPDATE - Chosen action (same module, different positions)
            # Position 1 (t-1): Gets new reward, modulated by position encoding
            next_choice_buffer_1_chosen = self.call_module(
                key_module='x_update_choice_buffer_chosen',
                key_state='x_choice_buffer_1',
                action=action,
                inputs=(
                    self.state['x_choice_buffer_2'],
                    self.state['x_choice_buffer_3'],
                    pos_1_choice,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 2 (t-2): Shifts from t-1, modulated by position encoding
            next_choice_buffer_2_chosen = self.call_module(
                key_module='x_update_choice_buffer_chosen',
                key_state='x_choice_buffer_2',
                action=action,
                inputs=(
                    self.state['x_choice_buffer_1'],
                    self.state['x_choice_buffer_3'],
                    pos_2_choice,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 3 (t-3): Shifts from t-2, modulated by position encoding
            next_choice_buffer_3_chosen = self.call_module(
                key_module='x_update_choice_buffer_chosen',
                key_state='x_choice_buffer_3',
                action=action,
                inputs=(
                    self.state['x_choice_buffer_1'],
                    self.state['x_choice_buffer_2'],
                    pos_3_choice,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # DYNAMIC BUFFER DECAY - Not chosen action (same module, different positions)
            # Position 1: Decay modulated by position encoding
            next_choice_buffer_1_not_chosen = self.call_module(
                key_module='x_update_choice_buffer_not_chosen',
                key_state='x_choice_buffer_1',
                action=1-action,
                inputs=(
                    self.state['x_choice_buffer_2'],
                    self.state['x_choice_buffer_3'],
                    pos_1_choice,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 2: Decay modulated by position encoding
            next_choice_buffer_2_not_chosen = self.call_module(
                key_module='x_update_choice_buffer_not_chosen',
                key_state='x_choice_buffer_2',
                action=1-action,
                inputs=(
                    self.state['x_choice_buffer_1'],
                    self.state['x_choice_buffer_3'],
                    pos_2_choice,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Position 3: Decay modulated by position encoding
            next_choice_buffer_3_not_chosen = self.call_module(
                key_module='x_update_choice_buffer_not_chosen',
                key_state='x_choice_buffer_3',
                action=1-action,
                inputs=(
                    self.state['x_choice_buffer_1'],
                    self.state['x_choice_buffer_2'],
                    pos_3_choice,
                ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
            )

            # Update all states
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
            
            # Update buffers with dynamic learned updates
            self.state['x_reward_buffer_1'] = next_buffer_1_chosen + next_buffer_1_not_chosen
            self.state['x_reward_buffer_2'] = next_buffer_2_chosen + next_buffer_2_not_chosen
            self.state['x_reward_buffer_3'] = next_buffer_3_chosen + next_buffer_3_not_chosen
            self.state['x_choice_buffer_1'] = next_choice_buffer_1_chosen + next_choice_buffer_1_not_chosen
            self.state['x_choice_buffer_2'] = next_choice_buffer_2_chosen + next_choice_buffer_2_not_chosen
            self.state['x_choice_buffer_3'] = next_choice_buffer_3_chosen + next_choice_buffer_3_not_chosen

            # Decision combines value (long-term) with choice value
            logits[timestep] = self.state['x_value_reward'] * self.betas['x_value_reward'](participant_embedding) + self.state['x_value_choice'] * self.betas['x_value_choice'](participant_embedding)

        logits = self.post_forward_pass(logits, batch_first)
        return logits, self.get_state()
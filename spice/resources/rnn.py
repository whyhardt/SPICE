import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import pysindy as ps
import numpy as np


class GRUModule(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        
        self.gru_in = nn.GRU(input_size, 1)
        # self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(1, 1)
       
    def forward(self, inputs):
        n_actions = inputs.shape[1]
        inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2]).unsqueeze(0)
        next_state = self.gru_in(inputs[..., 1:], inputs[..., :1].contiguous())[1].view(-1, n_actions, 1)
        # next_state = self.dropout(next_state)
        next_state = self.linear_out(next_state)
        return next_state


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs: torch.Tensor):
        return torch.ones_like(inputs, device=inputs.device, dtype=torch.float).view(-1, 1)
    
    
class CustomEmbedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.linear = torch.nn.Linear(num_embeddings, embedding_dim, bias=False)
    
    def forward(self, index: torch.Tensor):
        return self.get_embedding(self.one_hot_encode(index))
    
    def one_hot_encode(self, index: torch.Tensor):
        return torch.eye(self.num_embeddings, dtype=torch.float32, requires_grad=True, device=index.device)[index]
    
    def get_embedding(self, one_hot_encoded: torch.Tensor):
        return self.linear(one_hot_encoded)
    
    
class BaseRNN(nn.Module):
    def __init__(
        self, 
        n_actions, 
        hidden_size: int = 8,
        n_participants: int = 0,
        device=torch.device('cpu'),
        list_signals=['x_V', 'c_a', 'c_r'],
        embedding_size: int = 1,
        ):
        super(BaseRNN, self).__init__()
        
        # define general network parameters
        self.device = device
        self._n_actions = n_actions
        self.hidden_size = hidden_size
        self.embedding_size = 0
        self.n_participants = n_participants
        
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.recording = {key: [] for key in list_signals}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.submodules_sindy = dict()
        
        self.state = self.set_initial_state()
        
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs, prev_state, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        actions = inputs[:, :, :self._n_actions].float()
        rewards = inputs[:, :, self._n_actions:2*self._n_actions].float()
        participant_ids = inputs[0, :, -1:].int()
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)
        
        return (actions, rewards, participant_ids), logits, timesteps
    
    def post_forward_pass(self, logits, batch_first):
        # add model dim again and set state
        # self.set_state(*args)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            
        return logits
    
    def set_initial_state(self, batch_size=1):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
                
        for key in self.recording.keys():
            self.recording[key] = []
        
        # state dimensions: (habit_state, value_state, habit, value)
        # dimensions of states: (batch_size, substate, hidden_size)
        # self.set_state(*[init_value + torch.zeros([batch_size, self._n_actions], dtype=torch.float, device=self.device) for init_value in self.init_values])
        
        state = {key: torch.full(size=[batch_size, self._n_actions], fill_value=self.init_values[key], dtype=torch.float32, device=self.device) for key in self.init_values}
        
        self.set_state(state)
        return self.get_state()
        
    def set_state(self, state_dict):
        """this method sets the latent variables
        
        Args:
            state (Dict[str, torch.Tensor]): hidden state
        """
        
        # self._state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
        self.state = state_dict
      
    def get_state(self, detach=False):
        """this method returns the memory state
        
        Returns:
            Dict[str, torch.Tensor]: Dict of latent variables corresponding to the memory state
        """
        
        state = self.state
        if detach:
            state = {key: state[key].detach() for key in state}

        return state
    
    def to(self, device: torch.device): 
        self.device = device
        super().to(device=device)
        return self
    
    def record_signal(self, key, value: torch.Tensor):
        """appends a new timestep sample to the recording. A timestep sample consists of the value at timestep t-1 and the value at timestep t

        Args:
            key (str): recording key to which append the sample to
            old_value (_type_): value at timestep t-1 of shape (batch_size, feature_dim)
            new_value (_type_): value at timestep t of shape (batch_size, feature_dim)
        """
        
        self.recording[key].append(value.detach().cpu().numpy())
        
    def get_recording(self, key):
        return self.recording[key]
    
    def setup_module(self, input_size: int, hidden_size: int = None, dropout: float = 0., activation: nn.Module = None):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size (_type_): The number of inputs (excluding the memory state)
            hidden_size (_type_): Hidden size after the input layer
            dropout (_type_): Dropout rate before output layer
            activation (nn.Module, optional): Possibility to include an activation function. Defaults to None.

        Returns:
            torch.nn.Module: A torch module which can be called by one line and returns state update
        """
        if hidden_size is None:
            hidden_size = self.hidden_size
            
        # Linear network
        # layers = [
        #     nn.Linear(input_size+1, hidden_size),
        #     nn.Tanh(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 1),
        # ]
        # if activation is not None:
        #     layers.append(activation())
        # module = nn.Sequential(*layers)
        
        # GRU network
        module = GRUModule(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        
        return module 
    
    def call_module(
        self,
        key_module: str,
        key_state: str,
        action: torch.Tensor = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        participant_embedding: torch.Tensor = None, 
        participant_index: torch.Tensor = None,
        activation_rnn: Callable = None,
        scaling: bool = False,
        ):
        """Used to call a submodule of the RNN. Can be either: 
            1. RNN-module (saved in 'self.submodules_rnn')
            2. SINDy-module (saved in 'self.submodules_sindy')
            3. hard-coded equation (saved in 'self.submodules_eq')

        Args:
            key_module (str): _description_
            key_state (str): _description_
            action (torch.Tensor, optional): _description_. Defaults to None.
            inputs (Union[torch.Tensor, Tuple[torch.Tensor]], optional): _description_. Defaults to None.
            participant_embedding (torch.Tensor, optional): _description_. Defaults to None.
            participant_index (torch.Tensor, optional): _description_. Defaults to None.
            activation_rnn (Callable, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        record_signal = False
        
        action = action.unsqueeze(-1)
        value = self.get_state()[key_state].unsqueeze(-1)
        # value = key_state.unsqueeze(-1)
        
        if inputs is None:
            inputs = torch.zeros((*value.shape[:-1], 0), dtype=torch.float32, device=value.device)
            
        if participant_embedding is None:
            participant_embedding = torch.zeros((*value.shape[:-1], 0), dtype=torch.float32, device=value.device)
        elif participant_embedding.ndim == 2:
            participant_embedding = participant_embedding.unsqueeze(1).repeat(1, value.shape[1], 1)
        
        if isinstance(inputs, tuple):
            inputs = torch.concat([inputs_i.unsqueeze(-1) for inputs_i in inputs], dim=-1)
        elif inputs.dim()==2:
            inputs = inputs.unsqueeze(-1)
        
        if key_module in self.submodules_sindy.keys():                
            # sindy module
            
            # convert to numpy
            value = value.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            
            if inputs.shape[-1] == 0:
                # create dummy control inputs
                inputs = torch.zeros((*inputs.shape[:-1], 1))
            
            if participant_index is None:
                participant_index = torch.zeros((1, 1), dtype=torch.int32)
            
            next_value = np.zeros_like(value)
            for index_batch in range(value.shape[0]):
                sindy_model = self.submodules_sindy[key_module][participant_index[index_batch].item()] if isinstance(self.submodules_sindy[key_module], dict) else self.submodules_sindy[key_module]
                next_value[index_batch] = np.concatenate(
                    [sindy_model.predict(value[index_batch, index_action], inputs[index_batch, index_action]) for index_action in range(self._n_actions)], 
                    axis=0,
                    )
            next_value = torch.tensor(next_value, dtype=torch.float32, device=self.device)

        elif key_module in self.submodules_rnn.keys():
            # rnn module
            
            inputs = torch.concat((value, inputs, participant_embedding), dim=-1)
            update_value = self.submodules_rnn[key_module](inputs)
            next_value = value + update_value
            
            if activation_rnn is not None:
                next_value = activation_rnn(next_value)
            
        elif key_module in self.submodules_eq.keys():
            # hard-coded equation
            next_value = self.submodules_eq[key_module](value.squeeze(-1), inputs).unsqueeze(-1)

        else:
            raise ValueError(f'Invalid module key {key_module}.')

        if action is not None:
            # keep only actions necessary for that update and set others to zero
            next_value = next_value * action
        
        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        if scaling:
            # scale by inverse noise temperature
            scaling_factor = self.betas[key_state] if isinstance(self.betas[key_state], nn.Parameter) else self.betas[key_state](participant_embedding)
            next_value = next_value * scaling_factor
        
        return next_value.squeeze(-1)
    
    def integrate_sindy(self, modules: Dict[str, Iterable[ps.SINDy]]):
        # check that all provided modules find a place in the RNN
        checked = 0
        for m in modules:
            if m in self.submodules_rnn.keys():
                checked += 1
        assert checked == len(modules), f'Not all provided SINDy modules {tuple(modules.keys())} found a corresponding RNN module or vice versa.\nSINDy integration aborted.'
        
        # replace rnn modules with sindy modules
        self.submodules_sindy = modules


class RLRNN(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
            'x_learning_rate_reward': 0.,
        }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        hidden_size = 8,
        embedding_size = 8,
        dropout = 0.,
        device = torch.device('cpu'),
        list_signals = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen', 'c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice'],
        **kwargs,
    ):
        
        super(RLRNN, self).__init__(n_actions=n_actions, list_signals=list_signals, hidden_size=hidden_size, device=device, n_participants=n_participants)
        
        # set up the participant-embedding layer
        self.embedding_size = embedding_size
        if embedding_size > 1:
            # self.participant_embedding = torch.nn.Sequential(
            #     torch.nn.Embedding(num_embeddings=n_participants, embedding_dim=self.embedding_size),
            #     # CustomEmbedding(num_embeddings=n_participants, embedding_dim=embedding_size),
            #     torch.nn.ReLU(),
            #     torch.nn.Dropout(p=dropout),
            #     )
            self.participant_embedding = torch.nn.Embedding(num_embeddings=n_participants, embedding_dim=self.embedding_size)
        else:
            self.embedding_size = 1
            self.participant_embedding = DummyModule()
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas = torch.nn.ModuleDict()
        self.betas['x_value_reward'] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())# if embedding_size > 0 else torch.nn.Parameter(torch.tensor(1.0))
        self.betas['x_value_choice'] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())# if embedding_size > 0 else torch.nn.Parameter(torch.tensor(1.0))
        
        # set up the submodules
        self.submodules_rnn['x_learning_rate_reward'] = self.setup_module(input_size=3+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=2+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=1+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=1+self.embedding_size, dropout=dropout)
        
        # set up hard-coded equations
        self.submodules_eq['x_value_reward_chosen'] = self.x_value_reward_chosen
    
    def x_value_reward_chosen(self, value, inputs):
        return value + inputs[..., 1] * (inputs[..., 0] - value)
    
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, participant_id = inputs
        
        # derive more observations
        rewards_chosen = (actions * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        # rewards_not_chosen = ((1-actions) * rewards).sum(dim=-1, keepdim=True).repeat(1, 1, self._n_actions)
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        
        # get scaling factors
        scaling_factors = {}
        for key in self.state:
            if key in self.betas:
                scaling_factors[key] = self.betas[key] if isinstance(self.betas[key], nn.Parameter) else self.betas[key](participant_embedding)
        
        for timestep, action, reward_chosen in zip(timesteps, actions, rewards_chosen): #, rewards_not_chosen
            
            # record the current memory state and control inputs to the modules for SINDy training
            if not self.training and len(self.submodules_sindy)==0:
                self.record_signal('c_action', action)
                self.record_signal('c_reward_chosen', reward_chosen)
                # self.record_signal('c_reward_not_chosen', reward_not_chosen)
                self.record_signal('c_value_reward', self.state['x_value_reward'])
                self.record_signal('c_value_choice', self.state['x_value_choice'])
                self.record_signal('x_learning_rate_reward', self.state['x_learning_rate_reward'])
                self.record_signal('x_value_reward_not_chosen', self.state['x_value_reward'])
                self.record_signal('x_value_choice_chosen', self.state['x_value_choice'])
                self.record_signal('x_value_choice_not_chosen', self.state['x_value_choice'])
            
            # updates for x_value_reward
            learning_rate_reward = self.call_module(
                key_module='x_learning_rate_reward',
                key_state='x_learning_rate_reward',
                action=action,
                inputs=(
                    reward_chosen, 
                    # reward_not_chosen, 
                    self.state['x_value_reward'], 
                    self.state['x_value_choice'],
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(
                    reward_chosen, 
                    learning_rate_reward,
                    ),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
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
                # activation_rnn=torch.nn.functional.sigmoid,
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
            self.state['x_learning_rate_reward'] = learning_rate_reward
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
             
            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] * scaling_factors['x_value_reward'] + self.state['x_value_choice'] * scaling_factors['x_value_choice']
            
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
                
        return logits, self.get_state()

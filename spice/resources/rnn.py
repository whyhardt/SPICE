import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import pysindy as ps
import numpy as np


class GRUModule(nn.Module):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        
        self.gru_in = nn.GRU(input_size, 1)
        self.linear_out = nn.Linear(1, 1)

    def forward(self, inputs):
        n_actions = inputs.shape[1]
        inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2]).unsqueeze(0)
        next_state = self.gru_in(inputs[..., 1:], inputs[..., :1].contiguous())[1].view(-1, n_actions, 1)
        next_state = self.linear_out(next_state)
        return next_state


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs: torch.Tensor):
        return torch.ones_like(inputs, device=inputs.device, dtype=torch.float).view(-1, 1)
    

class SparseLeakyReLU(nn.LeakyReLU):
    
    def __init__(self, negative_slope = 0.01, inplace = False, threshold=0.01):
        """Leaky ReLU which in training-mode behaves like the standard LeakyReLU-module.
        In eval-mode it compares the activation value against a threshold and sets it to 0 if it is below.

        Args:
            negative_slope (float, optional): _description_. Defaults to 0.01.
            inplace (bool, optional): _description_. Defaults to False.
            threshold (float, optional): _description_. Defaults to 0.01.
        """
        super().__init__(negative_slope, inplace)
        self.threshold = threshold
    
    def forward(self, input):
        activation = super().forward(input)
        if self.training:
            return activation
        else:
            if activation < self.threshold:
                return torch.zeros_like(activation)
            else:
                activation
                

class BaseRNN(nn.Module):
    def __init__(
        self, 
        n_actions, 
        n_participants: int = 0,
        device=torch.device('cpu'),
        ):
        super(BaseRNN, self).__init__()
        
        # define general network parameters
        self.device = device
        self._n_actions = n_actions
        self.embedding_size = 0
        self.n_participants = n_participants
        
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.recording = {}
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
        additional_inputs = inputs[:, :, 2*self._n_actions:-3].float()
        blocks = inputs[:, :, -3:-2].int().repeat(1, 1, 2)
        experiment_ids = inputs[0, :, -2:-1].int()
        participant_ids = inputs[0, :, -1:].int()
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)
        
        return (actions, rewards, blocks, additional_inputs), (participant_ids, experiment_ids), logits, timesteps

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
                
        self.recording = {}
                
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
        
        if key not in self.recording:
            self.recording[key] = []
        self.recording[key].append(value.detach().cpu().numpy())
        
    def get_recording(self, key):
        return self.recording[key]
    
    def setup_module(self, input_size: int):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size (_type_): The number of inputs (excluding the memory state)
            dropout (_type_): Dropout rate before output layer
            activation (nn.Module, optional): Possibility to include an activation function. Defaults to None.

        Returns:
            torch.nn.Module: A torch module which can be called by one line and returns state update
        """
        
        # GRU network
        module = GRUModule(input_size=input_size)
        
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
        
        if action is not None:
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

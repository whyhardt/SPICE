import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import pysindy as ps
import numpy as np

from .sindy_differentiable import (
    compute_library_size,
    compute_polynomial_library,
    get_library_feature_names,
)


class GRUModule(nn.Module):
    def __init__(self, input_size, dropout=0., **kwargs):
        super().__init__()
        
        self.linear_in = nn.Linear(input_size, 8+input_size)
        # self.dropout = nn.Dropout(p=dropout)
        # self.relu = nn.LeakyReLU()
        self.gru_in = nn.GRU(8+input_size, 1)
        self.linear_out = nn.Linear(1, 1)
        
        #Simple weight initialization for all parameters
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply Xavier uniform to all parameters"""
        for param in self.parameters():
            if param.dim() > 1:  # Only weight matrices, skip biases
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, inputs):
        n_actions = inputs.shape[1]
        inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2]).unsqueeze(0)
        next_state = self.gru_in(
            # self.dropout(
            #     self.relu(
                    self.linear_in(inputs[..., 1:])
                #     )
                # )
            , inputs[..., :1].contiguous())[1].view(-1, n_actions, 1)
        next_state = self.linear_out(next_state)
        return next_state


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs: torch.Tensor):
        return torch.ones((*inputs.shape[:-1], 1), device=inputs.device, dtype=torch.float)
    
    
class ParameterModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.parameter = nn.Parameter(torch.tensor(1.))
        
    def forward(self, *args, **kwargs):
        return self.parameter
    
    

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
        embedding_size: int = 0,
        device=torch.device('cpu'),
        enable_sindy_reg: bool = False,
        sindy_config: Dict = None,
        use_sindy: bool = False,
        ):
        super(BaseRNN, self).__init__()

        # define general network parameters
        self.device = device
        self._n_actions = n_actions
        self.embedding_size = embedding_size
        self.n_participants = n_participants
        self.use_sindy = use_sindy
        
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c'
        self.recording = {}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.submodules_sindy = dict()
        self.betas = nn.ModuleDict()

        # Differentiable SINDy coefficients (NEW)
        self.enable_sindy_reg = enable_sindy_reg
        self.sindy_config = sindy_config
        self.sindy_coefficients = nn.ParameterDict()
        self.sindy_masks = {}
        self.sindy_library_names = {}
        # self.sindy_ensemble = 20 # TODO: implement ensemble sindy to account for bad initializations

        self.state = self.set_initial_state()
        
    def forward(self, inputs, prev_state, batch_first=False, use_sindy=False):
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
        
        sindy_loss_timesteps = torch.zeros_like(actions[..., 0])
        
        return (actions, rewards, blocks, additional_inputs), (participant_ids, experiment_ids), logits, timesteps, sindy_loss_timesteps


    def post_forward_pass(self, logits, sindy_loss, batch_first):
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            sindy_loss = sindy_loss.permute(1, 0)
        
        return logits, sindy_loss
    
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
        
    def setup_constant(self, embedding_size: int = None, activation: nn.Module = torch.nn.LeakyReLU, kwargs_activation = {'negative_slope': 0.01}):
        if embedding_size is not None:
            return nn.Sequential(nn.Linear(embedding_size, 1), activation(**kwargs_activation))
        else:
            return ParameterModule()
    
    def setup_embedding(self, num_embeddings: int, embedding_size: int, leaky_relu: float = 0.01, dropout: float = 0.5):
        return torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size),
            torch.nn.LeakyReLU(leaky_relu),
            torch.nn.Dropout(p=dropout),
            )
    
    def setup_module(self, input_size: int, dropout: float = 0.):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size (_type_): The number of inputs (excluding the memory state)
            dropout (_type_): Dropout rate before output layer
            activation (nn.Module, optional): Possibility to include an activation function. Defaults to None.

        Returns:
            torch.nn.Module: A torch module which can be called by one line and returns state update
        """
        
        # GRU network
        module = GRUModule(input_size=input_size, dropout=dropout)
        
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
        
        if key_module in self.submodules_rnn.keys():
            if not self.use_sindy:
                # rnn module
                
                inputs_rnn = torch.concat((value, inputs, participant_embedding), dim=-1)
                update_value = self.submodules_rnn[key_module](inputs_rnn)
                next_value = value + update_value
                
                if activation_rnn is not None:
                    next_value = activation_rnn(next_value)
            else:
                next_value = self.forward_sindy(
                    h_current = self.state[key_state], 
                    module_name = key_module, 
                    participant_ids = participant_index.squeeze(), 
                    controls = inputs, 
                    polynomial_degree = self.sindy_polynomial_degree,
                    ).unsqueeze(-1)
            
        elif key_module in self.submodules_eq.keys():
            # hard-coded equation
            next_value = self.submodules_eq[key_module](value.squeeze(-1), inputs).unsqueeze(-1)

        else:
            raise ValueError(f'Invalid module key {key_module}.')

        if action is not None:
            # keep only actions necessary for that update and set others to zero
            next_value = next_value * action
        
        if self.enable_sindy_reg and self.training:
            sindy_loss = self.compute_sindy_loss_for_module(
                    key_module,
                    self.state[key_state],
                    next_value.squeeze(-1),
                    inputs,
                    action.squeeze(-1),
                    participant_index.squeeze(),
                    self.sindy_polynomial_degree,
                )
        else:
            sindy_loss = 0
        
        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        if scaling:
            # scale by inverse noise temperature
            scaling_factor = self.betas[key_state] if isinstance(self.betas[key_state], nn.Parameter) else self.betas[key_state](participant_embedding)
            next_value = next_value * scaling_factor
        
        return next_value.squeeze(-1), sindy_loss
    
    def integrate_sindy(self, modules: Dict[str, Iterable[ps.SINDy]]):
        # check that all provided modules find a place in the RNN
        checked = 0
        for m in modules:
            if m in self.submodules_rnn.keys():
                checked += 1
        assert checked == len(modules), f'Not all provided SINDy modules {tuple(modules.keys())} found a corresponding RNN module or vice versa.\nSINDy integration aborted.'

        # replace rnn modules with sindy modules
        self.submodules_sindy = modules

    def setup_sindy_coefficients(self, polynomial_degree: int = 2):
        """
        Initialize learnable SINDy coefficients for each module.
        Called after submodules are defined in child class __init__.

        Args:
            polynomial_degree: Maximum polynomial degree for library
        """
        if not self.enable_sindy_reg or self.sindy_config is None:
            return

        rnn_modules = tuple(self.submodules_rnn.keys())
        control_parameters = self.sindy_config.get('control_parameters', [])
        library_setup = self.sindy_config.get('library_setup', {})

        for module_name in rnn_modules:
            # Count features for this module: state + relevant controls
            n_state_features = 1  # Current state value
            control_features = library_setup.get(module_name, [])
            n_control_features = len(control_features)
            n_total_features = n_state_features + n_control_features

            # Compute library size
            n_library_terms = compute_library_size(n_total_features, polynomial_degree)

            # Initialize coefficients [n_participants, n_library_terms]
            # Use very small initialization to prevent numerical instability
            init_coeffs = torch.randn(self.n_participants, n_library_terms) * 0.001
            self.sindy_coefficients[module_name] = nn.Parameter(init_coeffs)

            # Initialize masks (all ones initially)
            self.sindy_masks[module_name] = torch.ones(
                self.n_participants, n_library_terms,
                device=self.device
            )

            # Store library feature names
            feature_names = [module_name] + control_features
            self.sindy_library_names[module_name] = get_library_feature_names(
                feature_names, polynomial_degree
            )

    def forward_sindy(self, h_current, module_name, participant_ids, controls, polynomial_degree):
        coeffs = self.sindy_coefficients[module_name][participant_ids]  # [batch, n_library_terms]
        # Move masks to same device as participant_ids before indexing
        masks_device = self.sindy_masks[module_name].to(participant_ids.device)
        masks = masks_device[participant_ids]  # [batch, n_library_terms]
        if len(coeffs.shape) == 1:
            coeffs = coeffs[None]
            masks = masks[None]
        coeffs_sparse = coeffs * masks  # Apply sparsity mask

        # Compute polynomial library
        library = compute_polynomial_library(
            h_current, controls, degree=polynomial_degree, include_bias=True
        )  # [batch, n_actions, n_library_terms]

        # Predict next state via SINDy: x_next = x + library @ coeff
        # library: [batch, n_actions, n_library_terms]
        # coeffs_sparse: [batch, n_library_terms]
        # Need to expand for matrix multiply per action
        delta = torch.einsum('baf,bf->ba', library, coeffs_sparse)  # [batch, n_actions]
        h_next_sindy = h_current + delta
        
        return h_next_sindy     
    
    
    def compute_sindy_loss_for_module(
        self,
        module_name: str,
        h_current: torch.Tensor,
        h_next_rnn: torch.Tensor,
        controls: torch.Tensor,
        action_mask: torch.Tensor,
        participant_ids: torch.Tensor,
        polynomial_degree: int = 2,
    ) -> torch.Tensor:
        """
        Compute differentiable SINDy reconstruction loss for one module.

        Args:
            module_name: Name of the RNN module
            h_current: Current hidden state [batch, n_actions]
            h_next_rnn: RNN's predicted next state [batch, n_actions]
            controls: Control inputs [batch, n_actions, n_controls]
            action_mask: Binary mask for actions [batch, n_actions]
            participant_ids: Participant indices [batch]
            polynomial_degree: Polynomial degree

        Returns:
            Scalar loss value
        """
        if module_name not in self.sindy_coefficients:
            return torch.tensor(0.0, device=self.device)

        # Get coefficients and masks for this batch
        batch_size = h_current.shape[0]
        h_next_sindy = self.forward_sindy(h_current, module_name, participant_ids, controls, polynomial_degree)

        # Reconstruction loss (only for masked actions)
        diff = (h_next_rnn - h_next_sindy) ** 2
        masked_diff = diff * action_mask
        loss = masked_diff.sum(dim=-1)

        # Safety: check for NaN or inf
        loss = torch.where(torch.logical_or(torch.isnan(loss), torch.isinf(loss)), 0, loss)
        
        # Clip loss to prevent explosion
        loss = torch.clamp(loss, max=1000.0)

        return loss
        
    def thresholding(self, threshold, base_threshold=0):
        for module in self.submodules_rnn:
            threshold_tensor = torch.full_like(self.sindy_coefficients[module], threshold)
            # from original sindy-shred implementation - TODO: why this was used
            # for i in range(self.num_replicates):
            #     threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
            self.sindy_masks[module] = torch.abs(self.sindy_coefficients[module]) > threshold_tensor
            self.sindy_coefficients[module].data = self.sindy_masks[module] * self.sindy_coefficients[module].data
            
    def print_spice_model(self, participant_id: int = 0) -> None:
        """
        Get the learned SPICE features and equations.
        
        Returns:
            Dictionary containing features and equations for each agent/model
        """
        
        for module in self.submodules_rnn:
            equation_str = module + "[t+1] = "
            for index_term, term in enumerate(self.sindy_library_names[module]):
                if self.sindy_coefficients[module][participant_id, index_term] != 0:
                    if equation_str[-3:] != " = ":
                        equation_str += "+ "        
                    equation_str += str(np.round(self.sindy_coefficients[module][participant_id, index_term].item(), 4)) + " " + term
                    equation_str += "[t] " if term == module else " "
            print(equation_str)
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
from .spice_utils import SpiceConfig, SpiceSignals


class GRUModule(nn.Module):
    def __init__(self, input_size, dropout=0., **kwargs):
        super().__init__()
        
        self.linear_in = nn.Linear(input_size, 8+input_size)
        self.dropout = nn.Dropout(p=0.)
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
        inputs = inputs.view(inputs.shape[0]*inputs.shape[1], inputs.shape[2])[None, :]
        next_state = self.gru_in(
            # self.dropout(
            #     self.relu(
                    self.linear_in(inputs[..., 1:])
                #     )
                # )
            , inputs[..., :1].contiguous())[1].view(-1, n_actions, 1)
        next_state = self.dropout(self.linear_out(next_state))
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
        spice_config: SpiceConfig,
        n_participants: int = 1,
        n_experiments: int = 1,
        n_items: int = None,
        embedding_size: int = 32,
        use_sindy: bool = False,
        sindy_polynomial_degree: int = 2,
        sindy_ensemble_size: int = 10,
        device=torch.device('cpu'),
        ):
        super(BaseRNN, self).__init__()
        
        # define general network parameters
        self.spice_config = spice_config
        self.device = device
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        self.use_sindy = use_sindy
        self.rnn_training_finished = False
        self.n_items = n_items if n_items is not None else n_actions
        
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c'
        self.recording = {}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.submodules_sindy = dict()
        self.betas = nn.ModuleDict()
        
        # Differentiable SINDy coefficients (NEW)
        self.sindy_polynomial_degree = sindy_polynomial_degree
        self.sindy_coefficients = nn.ParameterDict()
        self.sindy_masks = {}
        self.sindy_library_names = {}        
        
        # Ensemble SINDy for stage 1 training (helps RNN learn better representations)
        self.sindy_ensemble_size = sindy_ensemble_size  # Number of ensemble members (num_replicates in sindy-shred)
        
        # Setup initial values of RNN
        self.sindy_loss = torch.tensor(0, requires_grad=True, device=device, dtype=torch.float32)
        self.state = None
        self.set_initial_state()  # initial memory state
        self.setup_sindy_coefficients()  # differentiable SINDy coefficients
        
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs: torch.Tensor, prev_state: Dict[str, torch.Tensor], batch_first: bool) -> SpiceSignals:
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        self.sindy_loss = torch.tensor(0, requires_grad=True, device=self.device, dtype=torch.float32)
        
        spice_signals = SpiceSignals()
        
        inputs = inputs.nan_to_num(0.)
        
        # create a mask of valid trials
        spice_signals.mask_valid_trials = inputs[:, :, :self.n_actions].sum(dim=-1, keepdim=True) > 0
        
        # item-specific signals
        spice_signals.actions = inputs[:, :, :self.n_actions].float()
        spice_signals.rewards = inputs[:, :, self.n_actions:2*self.n_actions].float()
        
        # additional signals (individual pre processing in forward pass possible)
        spice_signals.additional_inputs = inputs[:, :, 2*self.n_actions:-3].float()
        
        # static identifiers
        spice_signals.blocks = inputs[:, :, -3].int()
        spice_signals.experiment_ids = inputs[0, :, -2].int()
        spice_signals.participant_ids = inputs[0, :, -1].int()
        
        # use previous state or initialize state if not given
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # output signals
        spice_signals.timesteps = torch.arange(spice_signals.actions.shape[0], device=self.device)
        spice_signals.logits = torch.zeros((*spice_signals.actions.shape[:-1], self.n_actions), device=self.device)
        
        return spice_signals

    def post_forward_pass(self, spice_signals: SpiceSignals, batch_first: bool) -> SpiceSignals:
        
        if batch_first:
            spice_signals.logits = spice_signals.logits.permute(1, 0, 2)
        
        return spice_signals
    
    def set_initial_state(self, batch_size=1):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
        
        state = {key: torch.full(size=[batch_size, self.n_items], fill_value=self.spice_config.memory_state[key], dtype=torch.float32, device=self.device) for key in self.spice_config.memory_state}
        
        self.set_state(state)
        return self.get_state()
        
    def set_state(self, state_dict):
        """this method sets the latent variables
        
        Args:
            state (Dict[str, torch.Tensor]): hidden state
        """
        
        # self.state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
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
        # Also move sindy_masks to the device
        for module_name in self.sindy_masks:
            self.sindy_masks[module_name] = self.sindy_masks[module_name].to(device)
        self.sindy_loss = self.sindy_loss.to(device)
        
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
        action_mask: torch.Tensor = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        participant_embedding: torch.Tensor = None, 
        participant_index: torch.Tensor = None,
        activation_rnn: Callable = None,
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
        
        value = self.state[key_state]
        
        if inputs is None:
            inputs = torch.zeros((*value.shape, 0), dtype=torch.float32, device=value.device)
            
        if participant_embedding is None:
            participant_embedding = torch.zeros((*value.shape, 0), dtype=torch.float32, device=value.device)
        else:
            participant_embedding = participant_embedding.unsqueeze(1).repeat(1, value.shape[1], 1)
        
        if isinstance(inputs, tuple):
            inputs = torch.concat([inputs_i.unsqueeze(-1) for inputs_i in inputs], dim=-1)
        elif inputs.dim()==2:
            inputs = inputs.unsqueeze(-1)

        # Replace NaN in inputs with 0 to prevent NaN propagation through the network
        # NaN values should be masked out by action_mask, but we need to clean them before forward pass
        inputs = torch.nan_to_num(inputs, nan=0.0)

        if key_module in self.submodules_rnn.keys():
            if not self.use_sindy:
                # use rnn module

                inputs_rnn = torch.concat((value.unsqueeze(-1), inputs, participant_embedding), dim=-1)
                update_value = self.submodules_rnn[key_module](inputs_rnn)
                next_value = value + update_value.squeeze(-1)

                if activation_rnn is not None:
                    next_value = activation_rnn(next_value)
            else:
                # use sindy coefficients
                if participant_index is not None:
                    next_value = self.forward_sindy(
                        h_current = value, 
                        module_name = key_module, 
                        participant_ids = participant_index.squeeze(), 
                        controls = inputs, 
                        polynomial_degree = self.sindy_polynomial_degree,
                        )
                else:
                    next_value = torch.zeros_like(value)
        
        elif key_module in self.submodules_eq.keys():
            # use hard-coded equation
            next_value = self.submodules_eq[key_module](value, inputs)

        else:
            raise ValueError(f'Invalid module key {key_module}.')
            
        if self.training and not self.rnn_training_finished and participant_index is not None:
            self.sindy_loss = self.sindy_loss + self.compute_sindy_loss_for_module(
                    key_module,
                    self.state[key_state],
                    next_value,
                    inputs,
                    action_mask,
                    participant_index.squeeze(),
                    self.sindy_polynomial_degree,
                )
        
        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        # update memory state
        if action_mask is not None:
            self.state[key_state] = torch.where(action_mask == 1, next_value, self.state[key_state])
        else:
            self.state[key_state] = next_value
            
        return next_value
    
    def integrate_sindy(self, modules: Dict[str, Iterable[ps.SINDy]]):
        # check that all provided modules find a place in the RNN
        checked = 0
        for m in modules:
            if m in self.submodules_rnn.keys():
                checked += 1
        assert checked == len(modules), f'Not all provided SINDy modules {tuple(modules.keys())} found a corresponding RNN module or vice versa.\nSINDy integration aborted.'

        # replace rnn modules with sindy modules
        self.submodules_sindy = modules

    def setup_sindy_coefficients(self, polynomial_degree: int = None, ensemble_size: int = None):
        """
        Initialize learnable SINDy coefficients for each module with ensemble support.
        Called after submodules are defined in child class __init__.

        During stage 1 training: Uses ensemble of SINDy models with different thresholds
        During stage 2 training: Collapses to single model (ensemble_size=1)

        Args:
            polynomial_degree: Maximum polynomial degree for library
        """

        if polynomial_degree == None:
            polynomial_degree = self.sindy_polynomial_degree
        if ensemble_size == None:
            ensemble_size = self.sindy_ensemble_size
            
        rnn_modules = self.spice_config.modules#get('rnn_modules', [])
        library_setup = self.spice_config.library_setup#get('library_setup', {})
        
        for module_name in rnn_modules:
            # Count features for this module: state + relevant controls
            n_state_features = 1  # Current state value
            control_features = library_setup.get(module_name, [])
            n_control_features = len(control_features)
            n_total_features = n_state_features + n_control_features

            # Compute library size
            n_library_terms = compute_library_size(n_total_features, polynomial_degree)

            # Initialize coefficients [n_participants, n_ensemble, n_library_terms]
            # Use very small initialization to prevent numerical instability
            init_coeffs = torch.randn(self.n_participants, ensemble_size, n_library_terms) * 0.001
            self.sindy_coefficients[module_name] = nn.Parameter(init_coeffs)

            # Initialize masks (all ones initially) [n_participants, n_ensemble, n_library_terms]
            self.sindy_masks[module_name] = torch.ones(
                self.n_participants, ensemble_size, n_library_terms,
                device=self.device
            )

            # Store library feature names
            feature_names = tuple([module_name]) + control_features
            self.sindy_library_names[module_name] = get_library_feature_names(
                feature_names, polynomial_degree
            )

    def forward_sindy(self, h_current, module_name, participant_ids, controls, polynomial_degree):
        """
        Forward pass using SINDy model (used during inference/evaluation).
        Uses only the first ensemble member (index 0) for prediction.

        Args:
            h_current: Current hidden state [batch, n_actions]
            module_name: Name of the module
            participant_ids: Participant indices [batch]
            controls: Control inputs [batch, n_actions, n_controls]
           
            h_nex polynomial_degree: Polynomial degree

        Returns:t_sindy: Next hidden state [batch, n_actions]
        """
        # Get coefficients and masks for first ensemble member only
        sindy_coeffs = self.sindy_coefficients[module_name][participant_ids, 0]  # [batch, n_library_terms]
        sindy_masks = self.sindy_masks[module_name][participant_ids, 0]  # [batch, n_library_terms]
        if len(sindy_coeffs.shape) == 1:
            sindy_coeffs = sindy_coeffs[None]
            sindy_masks = sindy_masks[None]
        coeffs_sparse = sindy_coeffs * sindy_masks  # Apply sparsity mask

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
        Compute differentiable SINDy reconstruction loss for one module across all ensemble members.

        During training, computes loss for each ensemble member and returns the mean.
        This encourages the RNN to learn representations that are compatible with
        multiple sparsity patterns.

        Args:
            module_name: Name of the RNN module
            h_current: Current hidden state [batch, n_actions]
            h_next_rnn: RNN's predicted next state [batch, n_actions]
            controls: Control inputs [batch, n_actions, n_controls]
            filter: Binary mask for indicating target positions [batch, n_actions]
            participant_ids: Participant indices [batch]
            polynomial_degree: Polynomial degree

        Returns:
            Loss tensor [batch] (mean across ensemble members)
        """
        if module_name not in self.sindy_coefficients:
            return torch.tensor(0.0, device=self.device)

        # Get coefficients and masks for all ensemble members
        sindy_coeffs = self.sindy_coefficients[module_name][participant_ids]  # [batch, n_ensemble, n_library_terms]
        sindy_masks = self.sindy_masks[module_name][participant_ids]  # [batch, n_ensemble, n_library_terms]

        # Handle single-sample case
        if len(sindy_coeffs.shape) == 2:
            sindy_coeffs = sindy_coeffs[None]  # [1, n_ensemble, n_library_terms]
            sindy_masks = sindy_masks[None]  # [1, n_ensemble, n_library_terms]

        # Apply sparsity masks
        coeffs_sparse = sindy_coeffs * sindy_masks  # [batch, n_ensemble, n_library_terms]

        # Compute polynomial library (shared across ensemble)
        library = compute_polynomial_library(
            h_current, controls, degree=polynomial_degree, include_bias=True
        )  # [batch, n_actions, n_library_terms]

        # Compute predictions for all ensemble members
        # library: [batch, n_actions, n_library_terms]
        # coeffs_sparse: [batch, n_ensemble, n_library_terms]
        # We want: [batch, n_ensemble, n_actions]
        delta = torch.einsum('baf,bef->bea', library, coeffs_sparse)  # [batch, n_ensemble, n_actions]
        h_next_sindy_ensemble = h_current.unsqueeze(1) + delta  # [batch, n_ensemble, n_actions]

        # Compute reconstruction loss for each ensemble member
        # h_next_rnn: [batch, n_actions] -> [batch, 1, n_actions]
        # h_next_sindy_ensemble: [batch, n_ensemble, n_actions]
        diff = (h_next_rnn.unsqueeze(1) - h_next_sindy_ensemble) ** 2  # [batch, n_ensemble, n_actions]

        # Apply action mask: [batch, n_actions] -> [batch, 1, n_actions]
        # masked_diff = diff * action_mask.unsqueeze(1)
        masked_diff = torch.where(action_mask.unsqueeze(1) == 1, diff, 0)  # [batch, n_ensemble, n_actions]

        # 1. Sum over actions (i.e. remove masked out values by action_mask)
        # 2. Mean over ensemble dimension
        # 3. Mean over batch dimension only for finite values to get scalar loss
        sindy_loss = masked_diff.sum(dim=-1).mean() / len(self.submodules_rnn)
        # sindy_loss = sindy_loss[sindy_loss.isfinite()].mean()
        
        # Clip loss to prevent explosion
        sindy_loss = torch.clamp(sindy_loss, max=100.0)
        
        return sindy_loss
        
    def thresholding(self, threshold, n_terms_cutoff: int = None, base_threshold: float = 0.0):
        """
        Apply hard thresholding to SINDy coefficients.

        Args:
            threshold (float): Base threshold value
            n_terms_cutoff (int, optional): Number of smallest terms below threshold to be cut off across all modules
                                        for each participant and ensemble member. If None, all terms below threshold are cut.
            base_threshold (float): Additive base threshold (default: 0.0)
        """
        threshold_e = threshold + base_threshold
        module_list = list(self.submodules_rnn.keys())

        if n_terms_cutoff is None:
            # Simple thresholding: mask all terms below threshold
            for module in module_list:
                abs_coeffs = torch.abs(self.sindy_coefficients[module])
                mask = (abs_coeffs > threshold_e).float()
                self.sindy_masks[module] = mask
                self.sindy_coefficients[module].data *= mask
        else:
            # Collect all coefficients and masks across modules
            all_coeffs = torch.cat([self.sindy_coefficients[m] for m in module_list], dim=-1)
            all_masks = torch.cat([self.sindy_masks[m] for m in module_list], dim=-1)
            # Shape: [n_participants, n_ensemble, total_library_terms]
            
            # Identify candidates: active coefficients below threshold
            # is_candidate = (all_masks > 0) & (torch.abs(all_coeffs) < threshold_e)
            is_candidate = torch.abs(all_coeffs) < threshold_e
            
            # For each participant and ensemble member, find k smallest candidates
            # Set non-candidates to inf so they're ignored by topk
            coeffs_for_selection = torch.abs(all_coeffs).clone()
            coeffs_for_selection[~is_candidate] = float('inf')
            
            # Find k smallest per [participant, ensemble]
            k = min(n_terms_cutoff, is_candidate.sum(dim=-1).min().item())
            if k > 0:
                _, indices = torch.topk(coeffs_for_selection, k, dim=-1, largest=False)
                # Shape: [n_participants, n_ensemble, k]
                
                # Create cutoff mask
                cutoff_mask = torch.zeros_like(all_coeffs, dtype=torch.bool)
                cutoff_mask.scatter_(-1, indices, True)
                cutoff_mask &= is_candidate  # Safety: only cut actual candidates
                
                # Apply cutoff by zeroing masks
                all_masks[cutoff_mask] = 0.0
                
                # Split masks back to modules
                start_idx = 0
                for module in module_list:
                    n_terms = self.sindy_coefficients[module].shape[-1]
                    self.sindy_masks[module] = all_masks[..., start_idx:start_idx + n_terms]
                    start_idx += n_terms
            
            # Apply updated masks to coefficients
            with torch.no_grad():
                for module in module_list:
                    self.sindy_coefficients[module] *= self.sindy_masks[module]
            
    def print_spice_model(self, participant_id: int = 0, ensemble_idx: int = 0) -> None:
        """
        Get the learned SPICE features and equations for each trained module.

        Args:
            participant_id: Participant index to print
            ensemble_idx: Ensemble member index to print (default: 0, the first member)
        """

        for module in self.submodules_rnn:
            equation_str = module + "[t+1] = "
            for index_term, term in enumerate(self.sindy_library_names[module]):
                coeff_value = self.sindy_coefficients[module][participant_id, ensemble_idx, index_term].item()
                if term == module:
                    coeff_value += 1
                if np.abs(coeff_value) > 1e-3:
                    if equation_str[-3:] != " = ":
                        equation_str += "+ "
                    equation_str += str(np.round(coeff_value, 4)) + " " + term
                    equation_str += "[t] " if term == module else " "
            if equation_str[-3:] == " = ":
                equation_str += "0"
            print(equation_str)
            
        if hasattr(self, 'betas') and len(self.betas) > 0:
            for key in self.betas:
                if not isinstance(self.betas[key], ParameterModule):
                    participant_embedding = self.participant_embedding(torch.tensor(participant_id, device=self.device).int())
                else:
                    participant_embedding = None
                print(f"beta({key}) = {self.betas[key](participant_embedding).item():.4f}")
    
    def eval(self, use_sindy=True):
        super().eval()
        self.use_sindy = use_sindy
        return self
        
    def train(self, mode=True):
        super().train(mode)
        # if training mode activate (mode=True) -> do not use sindy for forward pass (self.use_sindy=False)
        self.use_sindy = not mode
        return self
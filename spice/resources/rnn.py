import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import numpy as np

from .sindy_differentiable import compute_library_size, compute_polynomial_library, get_library_feature_names, get_library_term_degrees, solve_sindy_stlsq
from .spice_utils import SpiceConfig, SpiceSignals


class GRUModule(nn.Module):
    def __init__(self, input_size, dropout=0., **kwargs):
        super().__init__()

        self.linear_in = nn.Linear(input_size, 8+input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.gru_in = nn.GRU(8+input_size, 1)

    def forward(self, inputs, state):
        # inputs: [within_ts, batch, n_items, features]
        # state:  [batch, n_items] — GRU hidden state (interpretable value)
        W, B, I, F = inputs.shape

        x = inputs.reshape(W, B * I, F)        # [W, B*I, F]
        h = state[-1].reshape(1, B * I, 1) if state is not None else None          # [1, B*I, 1]

        y = self.dropout(self.linear_in(x))      # [W, B*I, 8+F]
        output, _ = self.gru_in(y, h)            # [W, B*I, 1]
        output = output.reshape(W, B, I, 1)

        return output
    
    
class ParameterModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.parameter = nn.Parameter(torch.tensor(1.))
        
    def forward(self, *args, **kwargs):
        return self.parameter

        
class BaseRNN(nn.Module):
    def __init__(
        self,
        n_actions,
        spice_config: SpiceConfig,
        n_participants: int = 1,
        n_experiments: int = 1,
        n_items: int = None,
        n_reward_features: int = None,
        embedding_size: int = 32,
        use_sindy: bool = False,
        sindy_polynomial_degree: int = 1,
        sindy_ensemble_size: int = 10,
        device=torch.device('cpu'),
        **kwargs,
        ):
        super().__init__()

        # define general network parameters
        self.spice_config = spice_config
        self.device = device
        self.n_actions = n_actions
        self.n_reward_features = n_reward_features if n_reward_features is not None else n_actions
        self.embedding_size = embedding_size
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        self.n_sessions = n_participants * n_experiments
        self.use_sindy = use_sindy
        self.rnn_training_finished = False
        self.n_items = n_items if n_items is not None else n_actions
        
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c'
        self.recording = {}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        
        # Differentiable SINDy coefficients (NEW)
        self.sindy_polynomial_degree = sindy_polynomial_degree
        self.sindy_coefficients = nn.ParameterDict()
        self.sindy_coefficients_presence = {}  # Binary masks to permanently zero out coefficients
        self.sindy_candidate_terms = {}
        self.sindy_degree_weights = {}  # Weights for coefficient penalty based on polynomial degree
        self.sindy_pruning_patience_counters = {}  # Patience counters for thresholding
        self.sindy_specs = {}  # sindy-specific specifications for each module (e.g. include_bias, interaction_only, ...)
        
        # Ensemble SINDy for stage 1 training (helps RNN learn better representations)
        self.sindy_ensemble_size = sindy_ensemble_size  # Number of ensemble members (num_replicates in sindy-shred)
        
        # Data accumulator for direct least-squares SINDy solve (None = disabled)
        self._sindy_solve_data = None

        # Setup initial values of RNN
        self.sindy_loss = torch.tensor(0, requires_grad=True, device=device, dtype=torch.float32)
        self.state = None
        self.init_state()  # initial memory state
                
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs: torch.Tensor, prev_state: Dict[str, torch.Tensor], batch_first: bool) -> SpiceSignals:
        # canonical shape: (outer_ts, within_ts, batch, features)
        if batch_first:
            inputs = inputs.permute(1, 2, 0, 3)

        self.sindy_loss = torch.tensor(0, requires_grad=True, device=self.device, dtype=torch.float32)

        spice_signals = SpiceSignals()

        inputs = inputs.nan_to_num(0.)

        # create a mask of valid trials: [outer_ts, batch, 1]
        spice_signals.mask_valid_trials = inputs[:, :, :, :self.n_actions].sum(dim=(1, 3)).unsqueeze(-1) > 0

        reward_end = self.n_actions + self.n_reward_features

        # item-specific signals: [outer_ts, within_ts, batch, n_actions/n_rewards]
        spice_signals.actions = inputs[:, :, :, :self.n_actions].float()
        spice_signals.rewards = inputs[:, :, :, self.n_actions:reward_end].float()
        
        # additional signals: [outer_ts, within_ts, batch, n_additional]
        spice_signals.additional_inputs = inputs[:, :, :, reward_end:-5].float()

        # static identifiers
        spice_signals.time_trial = inputs[0, :, :, -5].int()       # [inner_ts, batch]
        spice_signals.trials = inputs[:, 0, :, -4].int()           # [outer_ts, batch]
        spice_signals.blocks = inputs[:, 0, :, -3].int()           # [outer_ts, batch]
        spice_signals.experiment_ids = inputs[0, 0, :, -2].int()   # [batch]
        spice_signals.participant_ids = inputs[0, 0, :, -1].int()  # [batch]
        
        # use previous state or initialize state if not given
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.init_state(batch_size=inputs.shape[2], within_ts=inputs.shape[1])

        # output signals
        spice_signals.trials = torch.arange(inputs.shape[0], device=self.device)
        spice_signals.logits = torch.zeros((inputs.shape[0], 1, inputs.shape[2], self.n_actions), device=self.device)

        return spice_signals

    def post_forward_pass(self, spice_signals: SpiceSignals, batch_first: bool) -> SpiceSignals:
        
        if batch_first:
            spice_signals.logits = spice_signals.logits.permute(2, 0, 1, 3)
        
        return spice_signals
    
    def init_state(self, batch_size=1, within_ts=1):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
        
        state = {key: torch.full(size=[within_ts, batch_size, self.n_items], fill_value=self.spice_config.memory_state[key], dtype=torch.float32, device=self.device) for key in self.spice_config.memory_state}
        
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
        self.sindy_loss = self.sindy_loss.to(device)
        # Move masks, weights, and patience counters to the correct device
        for module_name in self.sindy_coefficients_presence:
            self.sindy_coefficients_presence[module_name] = self.sindy_coefficients_presence[module_name].to(device)
            self.sindy_degree_weights[module_name] = self.sindy_degree_weights[module_name].to(device)
            self.sindy_pruning_patience_counters[module_name] = self.sindy_pruning_patience_counters[module_name].to(device)

        return self
        
    def setup_constant(self, embedding_size: int = None, activation: nn.Module = torch.nn.LeakyReLU, kwargs_activation = {'negative_slope': 0.01}):
        if embedding_size is not None:
            return nn.Sequential(nn.Linear(embedding_size, 1), activation(**kwargs_activation))
        else:
            return ParameterModule()
    
    def setup_embedding(self, num_embeddings: int, embedding_size: int, leaky_relu: float = 0.01, dropout: float = 0.):
        return torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size),
            # torch.nn.LeakyReLU(leaky_relu),
            torch.nn.Dropout(p=dropout),
            )
        # return torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size)
    
    def setup_module(
        self, 
        key_module: str, 
        input_size: int, 
        dropout: float = 0., 
        polynomial_degree: int = None, 
        include_bias = True, 
        include_state = True,
        interaction_only = False,
        ):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size (_type_): The number of inputs (excluding the memory state)
            dropout (_type_): Dropout rate before output layer

        Returns:
            torch.nn.Module: A torch module which can be called by one line and returns state update
        """
        
        # GRU network
        if polynomial_degree is None:
            polynomial_degree = self.sindy_polynomial_degree
            
        self.submodules_rnn[key_module] = GRUModule(input_size=input_size, dropout=dropout)
        self.sindy_specs[key_module] = {}
        self.sindy_specs[key_module]['include_bias'] = include_bias
        self.sindy_specs[key_module]['interaction_only'] = interaction_only
        self.sindy_specs[key_module]['include_state'] = include_state
        self.sindy_specs[key_module]['polynomial_degree'] = polynomial_degree
        self.setup_sindy_coefficients(key_module=key_module, polynomial_degree=polynomial_degree)
        
        # set name of each input variable which are then used in the library as features
        input_names = []
        if polynomial_degree > 0:
            start_index = 0
            end_index = np.argmax(np.array([('*' in term) or ('^' in term) for term in self.sindy_candidate_terms[key_module]]))
            if self.sindy_specs[key_module]['include_bias']:
                start_index += 1
            if self.sindy_specs[key_module]['include_state']:
                input_names.append(key_module)
                start_index += 1
            input_names += self.sindy_candidate_terms[key_module][start_index:end_index]
        self.sindy_specs[key_module]['input_names'] = tuple(input_names)
        
    def call_module(
        self,
        key_module: str,
        key_state: Optional[str] = None,
        action_mask: torch.Tensor = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        participant_embedding: torch.Tensor = None,
        participant_index: torch.Tensor = None,
        experiment_embedding: torch.Tensor = None,
        experiment_index: torch.Tensor = None,
        activation_rnn: Callable = None,
        ):
        """Call a submodule (RNN, SINDy, or equation) to compute the next state value.

        Inputs are of shape [W, B, I].
        2D inputs are broadcast to match the within-trial dimension W of any 3D inputs.

        Returns:
            torch.Tensor: [within_ts, batch, n_items] — full within-trial trajectory
        """
        
        value = self.state[key_state] if key_state is not None and key_state in self.state else None  # [W, B, I]

        B = self.state[list(self.state.keys())[0]].shape[1]
        I = self.n_items
        W = 1  # corrected after inputs processing
        
        if inputs is None or (isinstance(inputs, tuple) and len(inputs) == 0):
            if value is None:
                raise ValueError(f"When using BaseRNN.call_module you have to give at least a the state variable or inputs. Both are None.")
            inputs = torch.zeros((W, B, I, 0), dtype=torch.float32, device=self.device)
        elif isinstance(inputs, tuple):
            expanded = []
            for inp in inputs:
                expanded.append(inp.unsqueeze(-1))
            inputs = torch.cat(expanded, dim=-1)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.unsqueeze(-1)
            
        W = inputs.shape[0]
        
        if participant_index is None:
            participant_index = torch.zeros(B, dtype=torch.int, device=self.device)
        if experiment_index is None:
            experiment_index = torch.zeros(B, dtype=torch.int, device=self.device)

        if participant_embedding is None:
            participant_embedding = torch.zeros(B, 0, dtype=torch.float32, device=self.device)
        if experiment_embedding is None:
            experiment_embedding = torch.zeros(B, 0, dtype=torch.float32, device=self.device)

        embedding = torch.cat((experiment_embedding, participant_embedding), dim=-1)  # [B, emb]
        embedding = embedding.view(1, B, 1, -1).expand(W, -1, I, -1)  # [W, B, I, emb]

        # Replace NaN in inputs
        inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if key_module in self.submodules_rnn.keys():
            if not self.use_sindy:
                # RNN module
                inputs_rnn = torch.cat((inputs, embedding), dim=-1)  # [W, B, I, feat+emb]
                next_value = self.submodules_rnn[key_module](inputs_rnn, state=value).squeeze(-1)  # [W, B, I]
                if activation_rnn is not None:
                    next_value = activation_rnn(next_value)
            else:
                # SINDy — operates on last within-trial step
                if participant_index is not None:
                    next_value = torch.zeros((inputs.shape[:-1]), device=inputs.device)
                    if key_state is not None:
                        next_value_t = value[-1]
                    else:
                        next_value_t = torch.zeros(B, I, device=self.device)
                    for timestep in range(inputs.shape[0]):
                        next_value_t = self.forward_sindy(
                            h_current=next_value_t.unsqueeze(0),
                            key_module=key_module,
                            participant_ids=participant_index,
                            experiment_ids=experiment_index,
                            controls=inputs[timestep].unsqueeze(0),  # [B, I, n_controls]
                            polynomial_degree=self.sindy_polynomial_degree,
                            ensemble_idx=0,
                        ).squeeze(0, 2)  # [1, B, I]
                        next_value[timestep] += next_value_t
                else:
                    next_value = torch.zeros((1, *value.shape), device=value.device)

        elif key_module in self.submodules_eq.keys():
            # hard-coded equation — operates on last within-trial step
            next_value = self.submodules_eq[key_module](value, inputs[-1]).unsqueeze(0)  # [1, B, I]

        else:
            raise ValueError(f'Invalid module key {key_module}.')

        # SINDy: compute loss and/or accumulate data for direct solve
        if key_module in self.submodules_rnn and participant_index is not None:
            action_mask_2d = action_mask[-1] if action_mask is not None and action_mask.dim() >= 3 else action_mask
            value_0 = self.state[key_state][-1].unsqueeze(0) if key_state is not None else torch.zeros(W, B, I, device=self.device)
            h_current = torch.concat((value_0, next_value[:-1]))

            # Gradient-based sindy loss for RNN regularization (Stage 1 only)
            if self.training and not self.rnn_training_finished:
                self.sindy_loss = self.sindy_loss + self.compute_sindy_loss_for_module(
                        module_name=key_module,
                        h_current=h_current,
                        h_next_rnn=next_value,
                        controls=inputs,
                        action_mask=action_mask_2d,
                        participant_ids=participant_index.squeeze(),
                        experiment_ids=experiment_index.squeeze(),
                        polynomial_degree=self.sindy_polynomial_degree,
                    )

            # Accumulate data for direct least-squares solve
            if self._sindy_solve_data is not None and key_module in self._sindy_solve_data:
                library = compute_polynomial_library(
                    h_current.detach(), inputs.detach(),
                    degree=self.sindy_polynomial_degree,
                    feature_names=self.sindy_specs[key_module]['input_names'],
                    library=self.sindy_candidate_terms[key_module],
                )
                delta_h = (next_value - h_current).detach()
                self._sindy_solve_data[key_module].append({
                    'library': library,       # [W, B, I, n_terms]
                    'delta_h': delta_h,       # [W, B, I]
                    'action_mask': action_mask_2d.detach() if action_mask_2d is not None else None,
                    'participant_ids': participant_index.squeeze().detach(),
                    'experiment_ids': experiment_index.squeeze().detach(),
                })

        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        if action_mask is not None:
            mask = action_mask[-1] if action_mask.dim() >= 3 else action_mask
            next_value = torch.where(mask == 1, next_value, self.state[key_state])

        if key_state is not None:
            self.state[key_state] = next_value
        
        return next_value  # [W, B, I]
    
    def setup_sindy_coefficients(self, key_module: str, polynomial_degree: int, ensemble_size: int = None):
        """
        Initialize learnable SINDy coefficients for each module with ensemble support.
        Called after submodules are defined in child class __init__.

        During stage 1 training: Uses ensemble of SINDy models with different thresholds
        During stage 2 training: Collapses to single model (ensemble_size=1)

        Args:
            polynomial_degree: Maximum polynomial degree for library
        """

        if ensemble_size == None:
            ensemble_size = self.sindy_ensemble_size
            
        control_features = self.spice_config.library_setup[key_module]
        sindy_specs = self.sindy_specs[key_module]
        
        # Count features for this module: state + relevant controls
        n_state_features = 1  # Current state value
        n_control_features = len(control_features)
        n_total_features = n_state_features + n_control_features

        # Store library feature names
        feature_names = tuple([key_module]) + control_features
        self.sindy_candidate_terms[key_module] = get_library_feature_names(feature_names, polynomial_degree)
        
        # apply sindy_specs
        n_removed = 0
        if not sindy_specs['include_bias']:
            self.sindy_candidate_terms[key_module].remove('1')
            n_removed += 1
        
        if sindy_specs['interaction_only']:
            # remove all 
            for index_term, term in enumerate(self.sindy_candidate_terms[key_module]):
                if '^' in term and not '*' in term:
                    self.sindy_candidate_terms[key_module].remove(term)
                    n_removed += 1
        
        if not sindy_specs['include_state']:
            terms_remove = []
            for index_term, term in enumerate(self.sindy_candidate_terms[key_module]):
                if key_module in term:
                    terms_remove.append(term)
                    n_removed += 1
            for index_term, term in enumerate(terms_remove):
                self.sindy_candidate_terms[key_module].remove(term)
                    
        # Compute library size
        n_library_terms = compute_library_size(n_total_features, polynomial_degree) - n_removed
        
        # Initialize coefficients [n_participants, n_ensemble, n_library_terms]
        # Use very small initialization to prevent numerical instability
        init_coeffs = torch.randn(self.n_participants, self.n_experiments, ensemble_size, n_library_terms) * 0.001
        self.sindy_coefficients[key_module] = nn.Parameter(init_coeffs)
        
        # Initialize mask to all ones (all coefficients active)
        self.sindy_coefficients_presence[key_module] = torch.ones(
            self.n_participants, self.n_experiments, ensemble_size, n_library_terms,
            dtype=torch.bool, device=self.device
        )

        # Compute degree-based weights for coefficient penalty: d=0 -> 1, d=1 -> 2, d=2 -> 3, etc.
        term_degrees = get_library_term_degrees(self.sindy_candidate_terms[key_module])
        degree_weights = torch.tensor([max(1,d*2) for d in term_degrees], dtype=torch.float32, device=self.device)
        self.sindy_degree_weights[key_module] = degree_weights

        # Initialize patience counters to zero
        self.sindy_pruning_patience_counters[key_module] = torch.zeros(
            self.n_participants, self.n_experiments, ensemble_size, n_library_terms,
            dtype=torch.int32, device=self.device
        )
    
    def forward_sindy(self, h_current: torch.Tensor, key_module: str, participant_ids: torch.Tensor, experiment_ids: torch.Tensor, controls: torch.Tensor, polynomial_degree: int, ensemble_idx: int = None):
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
        
        if ensemble_idx is None:
            # Get coefficients and masks for all ensemble members
            sindy_coeffs = self.sindy_coefficients[key_module][participant_ids, experiment_ids]  # [batch, n_ensemble, n_library_terms]
            mask = self.sindy_coefficients_presence[key_module][participant_ids, experiment_ids]  # [batch, n_ensemble, n_library_terms]
        else:
            # Get coefficients and masks for all ensemble members
            sindy_coeffs = self.sindy_coefficients[key_module][participant_ids, experiment_ids, ensemble_idx].unsqueeze(1)  # [batch, n_ensemble, n_library_terms]
            mask = self.sindy_coefficients_presence[key_module][participant_ids, experiment_ids, ensemble_idx].unsqueeze(1)  # [batch, n_ensemble, n_library_terms]

        # custom dropout layer for sindy coefficients
        # if hasattr(self, 'dropout') and self.training:
        #     dropout_mask = (torch.rand_like(sindy_coeffs) > self.dropout).float()
        #     sindy_coeffs = sindy_coeffs * dropout_mask / (1 - self.dropout)
        
        # Apply the mask to enforce sparsity during loss computation
        sindy_coeffs = sindy_coeffs * mask.float()
        
        # Compute polynomial library (shared across ensemble)
        library = compute_polynomial_library(
            h_current, 
            controls, 
            degree=polynomial_degree,
            feature_names=self.sindy_specs[key_module]['input_names'],
            library=self.sindy_candidate_terms[key_module],
            )  # [batch, n_actions, n_library_terms]

        # Compute predictions for all ensemble members
        # library: [batch, actions, library_terms]
        # coeffs_sparse: [batch, ensemble, library_terms]
        # We want: [batch, ensemble, actions]
        h_next_sindy = h_current.unsqueeze(-2) + torch.einsum('wbax,bex->wbea', library, sindy_coeffs)

        return h_next_sindy     

    def compute_sindy_loss_for_module(
        self,
        module_name: str,
        h_current: torch.Tensor,
        h_next_rnn: torch.Tensor,
        controls: torch.Tensor,
        action_mask: torch.Tensor,
        participant_ids: torch.Tensor,
        experiment_ids: torch.Tensor,
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
            experiment_ids: Experiment indices [batch]
            polynomial_degree: Polynomial degree

        Returns:
            Loss tensor [batch] (mean across ensemble members)
        """
            
        if module_name not in self.sindy_coefficients:
            return torch.tensor(0.0, device=self.device)
        
        h_next_sindy_ensemble = self.forward_sindy(
            h_current=h_current, 
            key_module=module_name, 
            participant_ids=participant_ids, 
            experiment_ids=experiment_ids,
            controls=controls, 
            polynomial_degree=polynomial_degree,
            )
        
        # Compute reconstruction loss for each ensemble member
        # h_next_rnn: [batch, n_actions] -> [batch, 1, n_actions]
        # h_next_sindy_ensemble: [batch, n_ensemble, n_actions]
        diff = (h_next_rnn.unsqueeze(-2) - h_next_sindy_ensemble) ** 2  # [batch, n_ensemble, n_actions]

        # Apply action mask: [batch, n_actions] -> [batch, 1, n_actions]
        # masked_diff = diff * action_mask.unsqueeze(1)
        if action_mask is not None:
            masked_diff = torch.where(action_mask.unsqueeze(1) == 1, diff, 0)  # [batch, n_ensemble, n_actions]
            n_masked = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        else:
            masked_diff = diff
            n_masked = diff.shape[-1]  # All actions contribute when no mask
            
        # 1. Sum over actions (i.e. remove masked out values by action_mask)
        # 2. normalize by number of included values (e.g. in 4-armed bandit: non-chosen actions = 3 -> need to normalize; otherwise skewed loss with heavy bias for non-chosen actions)
        # 3. Compute mean
        # 4. Normalize over number of modules to keep SINDy loss in a good range for any SPICE architecture
        sindy_loss = torch.mean(masked_diff.sum(dim=-1) / n_masked) / len(self.submodules_rnn)
        
        # Clip loss to prevent explosion
        sindy_loss = torch.clamp(sindy_loss, max=100.0)
        
        return sindy_loss

    def solve_sindy_coefficients_direct(self, threshold: float = 0.05, max_iter: int = 10, ridge_alpha: float = 1e-4):
        """
        Solve SINDy coefficients via batched STLSQ using accumulated data from forward passes.

        Vectorizes across all (participant, experiment) pairs in a single lstsq call
        per module. Ensemble members sharing the same presence mask are solved once
        and broadcast.

        Args:
            threshold: STLSQ coefficient threshold (0 = plain least squares)
            max_iter: Maximum STLSQ iterations
            ridge_alpha: Ridge regularization strength
        """
        if self._sindy_solve_data is None:
            return

        with torch.no_grad():
            for module_name in self.submodules_rnn:
                if module_name not in self._sindy_solve_data or not self._sindy_solve_data[module_name]:
                    continue

                entries = self._sindy_solve_data[module_name]

                # Concatenate all accumulated entries along the time dimension (dim=0)
                all_library = torch.cat([e['library'] for e in entries], dim=0)   # [T_total, B, I, n_terms]
                all_delta_h = torch.cat([e['delta_h'] for e in entries], dim=0)   # [T_total, B, I]

                # Action masks: broadcast to [T_total, B, I]
                all_masks = []
                for e in entries:
                    T = e['library'].shape[0]
                    if e['action_mask'] is not None:
                        all_masks.append(e['action_mask'].unsqueeze(0).expand(T, -1, -1))
                    else:
                        all_masks.append(torch.ones(T, e['library'].shape[1], e['library'].shape[2],
                                                    dtype=torch.bool, device=self.device))
                all_action_mask = torch.cat(all_masks, dim=0)  # [T_total, B, I]

                all_pids = entries[0]['participant_ids']   # [B]
                all_eids = entries[0]['experiment_ids']    # [B]
                n_terms = all_library.shape[-1]

                # Collect per-(p, e) data into lists for batched stacking
                pe_libs = []
                pe_dhs = []
                pe_indices = []  # (p, e) tuples

                for p in range(self.n_participants):
                    for e in range(self.n_experiments):
                        if all_pids.dim() == 0:
                            batch_mask = (all_pids == p) & (all_eids == e)
                            batch_mask = batch_mask.unsqueeze(0)
                        else:
                            batch_mask = (all_pids == p) & (all_eids == e)

                        if not batch_mask.any():
                            continue

                        lib_pe = all_library[:, batch_mask]
                        dh_pe = all_delta_h[:, batch_mask]
                        mask_pe = all_action_mask[:, batch_mask]

                        lib_flat = lib_pe.reshape(-1, n_terms)
                        dh_flat = dh_pe.reshape(-1)
                        mask_flat = mask_pe.reshape(-1).bool()

                        lib_valid = lib_flat[mask_flat]
                        dh_valid = dh_flat[mask_flat]

                        if lib_valid.shape[0] == 0:
                            continue

                        pe_libs.append(lib_valid)
                        pe_dhs.append(dh_valid)
                        pe_indices.append((p, e))

                if not pe_libs:
                    continue

                # Pad to max N_valid and stack (zero rows don't affect lstsq)
                max_n = max(lib.shape[0] for lib in pe_libs)
                lib_stacked = torch.zeros(len(pe_libs), max_n, n_terms, dtype=pe_libs[0].dtype, device=self.device)
                dh_stacked = torch.zeros(len(pe_libs), max_n, dtype=pe_libs[0].dtype, device=self.device)
                for i, (lib, dh) in enumerate(zip(pe_libs, pe_dhs)):
                    lib_stacked[i, :lib.shape[0]] = lib
                    dh_stacked[i, :dh.shape[0]] = dh
                n_pe = lib_stacked.shape[0]
                K = self.sindy_ensemble_size

                # Gather presence masks: [n_pe, K, n_terms]
                presence_stacked = torch.stack([
                    self.sindy_coefficients_presence[module_name][p, e]
                    for p, e in pe_indices
                ])

                # Check if all ensemble members share the same mask
                all_ens_same = (presence_stacked[:, 0:1] == presence_stacked).all()

                if all_ens_same:
                    # Solve once per (p, e) and broadcast to all ensemble members
                    presence_pe = presence_stacked[:, 0]  # [n_pe, n_terms]
                    coeffs_pe, new_presence_pe = solve_sindy_stlsq(
                        library=lib_stacked,
                        delta_h=dh_stacked,
                        threshold=threshold,
                        max_iter=max_iter,
                        presence_mask=presence_pe,
                        ridge_alpha=ridge_alpha,
                    )  # [n_pe, n_terms] each

                    for i, (p, e) in enumerate(pe_indices):
                        self.sindy_coefficients[module_name].data[p, e] = coeffs_pe[i].unsqueeze(0).expand(K, -1)
                        self.sindy_coefficients_presence[module_name][p, e] = new_presence_pe[i].unsqueeze(0).expand(K, -1)
                else:
                    # Different masks per ensemble: expand and solve all at once
                    lib_expanded = lib_stacked.unsqueeze(1).expand(-1, K, -1, -1).reshape(n_pe * K, -1, n_terms)
                    dh_expanded = dh_stacked.unsqueeze(1).expand(-1, K, -1).reshape(n_pe * K, -1)
                    presence_expanded = presence_stacked.reshape(n_pe * K, n_terms)

                    coeffs_all, presence_all = solve_sindy_stlsq(
                        library=lib_expanded,
                        delta_h=dh_expanded,
                        threshold=threshold,
                        max_iter=max_iter,
                        presence_mask=presence_expanded,
                        ridge_alpha=ridge_alpha,
                    )

                    coeffs_all = coeffs_all.reshape(n_pe, K, n_terms)
                    presence_all = presence_all.reshape(n_pe, K, n_terms)

                    for i, (p, e) in enumerate(pe_indices):
                        self.sindy_coefficients[module_name].data[p, e] = coeffs_all[i]
                        self.sindy_coefficients_presence[module_name][p, e] = presence_all[i]

        self._sindy_solve_data = None

    def sindy_coefficient_pruning(self, threshold, n_terms_pruning: int = None, base_threshold: float = 0.0, patience: int = 1):
        """
        Apply hard thresholding to SINDy coefficients with patience counter.
        A coefficient is only thresholded out if it has been below threshold for 'patience' consecutive calls.

        Args:
            threshold (float): Base threshold value
            n_terms_pruning (int, optional): Number of smallest terms below threshold to be pruned across all modules
                                        for each participant and ensemble member. If None, all terms below threshold are cut.
            base_threshold (float): Additive base threshold (default: 0.0)
            patience (int): Number of consecutive epochs a coefficient must be below threshold before being cut (default: 1)
        """
        threshold_e = threshold + base_threshold
        module_list = list(self.submodules_rnn.keys())

        if n_terms_pruning is None:
            # Simple thresholding: mask all terms below threshold with patience
            for module in module_list:
                abs_coeffs = torch.abs(self.sindy_coefficients[module])
                below_threshold = (abs_coeffs < threshold_e)  # Use < instead of <= to keep coefficients exactly at threshold

                # Update patience counters
                # Increment counter where coefficient is below threshold
                self.sindy_pruning_patience_counters[module] = torch.where(
                    below_threshold,
                    self.sindy_pruning_patience_counters[module] + 1,
                    torch.zeros_like(self.sindy_pruning_patience_counters[module])  # Reset to 0 if above threshold
                )

                # Only threshold if patience exceeded
                mask = (self.sindy_pruning_patience_counters[module] < patience)
                # Update the permanent mask
                self.sindy_coefficients_presence[module] &= mask
                # Zero out the coefficients
                self.sindy_coefficients[module].data *= mask.float()
                # Reset patience counters for thresholded coefficients
                self.sindy_pruning_patience_counters[module] *= mask.int()
        else:
            # Collect all coefficients, masks, and patience counters across modules
            # Shape: [n_participants, n_ensemble, total_library_terms]
            all_coeffs = torch.cat([self.sindy_coefficients[m].abs() for m in module_list], dim=-1)
            all_masks = torch.cat([self.sindy_coefficients_presence[m] for m in module_list], dim=-1)
            all_patience = torch.cat([self.sindy_pruning_patience_counters[m] for m in module_list], dim=-1)


            # Update patience counters for all coefficients (excluding protected terms)
            below_threshold = (all_coeffs < threshold_e) & (all_masks == 1)
            all_patience = torch.where(
                below_threshold,
                all_patience + 1,
                torch.zeros_like(all_patience)  # Reset to 0 if above threshold
            )

            # Identify candidates: active coefficients that have exceeded patience (excluding protected terms)
            is_candidate = (all_patience >= patience) & (all_masks == 1)

            # For n_terms_pruning, only consider coefficients that exceed patience
            temp_coeffs = all_coeffs.clone()
            temp_coeffs[~is_candidate] = torch.inf

            # Find k smallest per [participant, ensemble]
            _, indices = torch.topk(temp_coeffs, min(n_terms_pruning, is_candidate.sum().item()), dim=-1, largest=False)
            # Shape: [n_participants, n_ensemble, k]

            # Create pruning mask
            pruning_mask = torch.zeros_like(all_coeffs, dtype=torch.bool)
            pruning_mask.scatter_(-1, indices, torch.ones_like(indices, dtype=torch.bool))
            pruning_mask &= is_candidate  # Safety: only cut actual candidates

            # Split back to modules and update
            start_idx = 0
            with torch.no_grad():
                for module in module_list:
                    n_terms = self.sindy_coefficients[module].shape[-1]

                    # Update patience counters for this module
                    self.sindy_pruning_patience_counters[module] = all_patience[..., start_idx:start_idx + n_terms]

                    keep_mask = ~pruning_mask[..., start_idx:start_idx + n_terms]
                    # Update the permanent mask
                    self.sindy_coefficients_presence[module] &= keep_mask
                    # Zero out the coefficients
                    self.sindy_coefficients[module] *= keep_mask.float()
                    # Reset patience counters for thresholded coefficients
                    self.sindy_pruning_patience_counters[module] *= keep_mask.int()

                    start_idx += n_terms
            
    def count_sindy_coefficients(self) -> torch.Tensor:
        coefficients = torch.zeros(self.n_participants, self.n_experiments, device=self.device)
        for module in self.submodules_rnn:
            presence = self.sindy_coefficients_presence[module].clone()
            coefficients += (presence != 0).sum(dim=-1).float().mean(dim=-1)
        return coefficients

    def compute_weighted_coefficient_penalty(self, sindy_alpha: float = 0.0, norm: int = 1) -> torch.Tensor:
        """
        Compute weighted coefficient penalty on SINDy coefficients based on polynomial degree.
        Each term is penalized according to its degree: d=0 -> 1*coeff^2, d=1 -> 2*coeff^2, d=2 -> 3*coeff^2, etc.

        Args:
            sindy_alpha: Base coefficient regularization weight
            norm: Norm type (1 for L1, 2 for L2)

        Returns:
            Weighted coefficient penalty (scalar tensor)
        """

        assert norm == 1 or norm == 2, "Only L1-norm or L2-norm are allowed."

        if sindy_alpha == 0.0:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for module_name in self.submodules_rnn:
            if module_name not in self.sindy_coefficients:
                continue

            # Get coefficients: [n_participants, n_ensemble, n_library_terms]
            coeffs = self.sindy_coefficients[module_name]

            # Get degree weights: [n_library_terms]
            degree_weights = self.sindy_degree_weights[module_name].clone()

            # Compute weighted coefficient penalty for each term
            # For each coefficient, penalty = (degree + 1) * |coeff|^norm
            # degree_weights already contains (degree + 1) for each term
            if norm == 2:
                # Sum across coefficient dimension, mean over participants and ensemble
                weighted_penalty = ((coeffs ** 2) * degree_weights).sum(dim=-1).mean()
            else:
                # Sum across coefficient dimension, mean over participants and ensemble
                weighted_penalty = (coeffs.abs() * degree_weights).sum(dim=-1).mean()

            penalty += weighted_penalty

        # Apply base coefficient weight
        penalty = sindy_alpha * penalty

        return penalty
                    
    def get_spice_model_string(self, participant_id: int = 0, experiment_id: int = 0, ensemble_idx: int = 0) -> str:
        """
        Get the learned SPICE features and equations as a string.

        Args:
            participant_id: Participant index
            ensemble_idx: Ensemble member index to print (default: 0, the first member)

        Returns:
            String representation of the SPICE model equations
        """
        lines = []
        max_len_module = max([len(module) for module in self.get_modules()])
        
        for module in self.submodules_rnn:
            sparse_coeffs = (self.sindy_coefficients[module][participant_id, experiment_id, ensemble_idx] * self.sindy_coefficients_presence[module][participant_id, experiment_id, ensemble_idx]).detach().cpu().numpy()
            space_filler = " "+" "*(max_len_module-len(module)) if max_len_module > len(module) else " "
            equation_str = module + "[t+1]" + space_filler + "= "
            for index_term, term in enumerate(self.sindy_candidate_terms[module]):
                if term == module:
                    sparse_coeffs[index_term] += 1
                if np.abs(sparse_coeffs[index_term]) != 0:
                    if equation_str[-3:] != " = ":
                        equation_str += "+ "
                    equation_str += str(np.round(sparse_coeffs[index_term], 3)) + " " + term
                    equation_str += "[t] " if term == module else " "
            if equation_str[-3:] == " = ":
                equation_str += "0"
            lines.append(equation_str)
        return "\n".join(lines)

    def print_spice_model(self, participant_id: int = 0, experiment_id: int = 0, ensemble_idx: int = 0) -> None:
        """
        Print the learned SPICE features and equations for each trained module.

        Args:
            participant_id: Participant index to print
            ensemble_idx: Ensemble member index to print (default: 0, the first member)
        """
        print(self.get_spice_model_string(participant_id, experiment_id, ensemble_idx))

    def get_modules(self):
        return [module for module in self.submodules_rnn]
    
    def get_candidate_terms(self, key_module: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        if key_module is None:
            return self.sindy_candidate_terms
        else:
            return self.sindy_candidate_terms[key_module]
            
    def get_sindy_coefficients(self, key_module: Optional[str] = None):
        if key_module is None:
            key_module = self.get_modules()
        
        if isinstance(key_module, str):
            key_module = [key_module]
        
        sindy_coefficients = {}
        for module in key_module:
            sindy_coefficients[module] = self.sindy_coefficients[module].detach().cpu().numpy() * self.sindy_coefficients_presence[module].detach().cpu().numpy()
            if self.sindy_specs[module].get('include_bias', False):
                sindy_coefficients[module][..., 1] += 1
            else:
                sindy_coefficients[module][..., 0] += 1
        
        return sindy_coefficients
    
    def eval(self, use_sindy=True):
        super().eval()
        self.use_sindy = use_sindy
        return self
        
    def train(self, mode=True):
        super().train(mode)
        # if training mode activate (mode=True) -> do not use sindy for forward pass (self.use_sindy=False)
        self.use_sindy = not mode
        return self
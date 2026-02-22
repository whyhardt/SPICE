import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import numpy as np

from .sindy_differentiable import compute_library_size, compute_polynomial_library, get_library_feature_names, get_library_term_degrees
from .spice_utils import SpiceConfig, SpiceSignals


class EnsembleLinear(nn.Module):
    """Linear layer with independent parameters per ensemble member.

    Parameters have shape (ensemble_size, out_features, in_features).
    Forward uses einsum for vectorized computation across ensemble.
    """
    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(ensemble_size, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight.view(ensemble_size, out_features, in_features))

    def forward(self, x):
        # x: (..., E, *, in_features) -> (..., E, *, out_features)
        # Supports (E, B, F), (W, E, B, F), etc.
        return torch.einsum('eoi,...ei->...eo', self.weight, x) + self.bias


class EnsembleEmbedding(nn.Module):
    """Embedding layer with independent parameters per ensemble member.

    Parameters have shape (ensemble_size, num_embeddings, embedding_dim).
    """
    def __init__(self, ensemble_size, num_embeddings, embedding_dim, dropout=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, num_embeddings, embedding_dim))
        self.dropout = nn.Dropout(p=dropout)
        nn.init.normal_(self.weight)

    def forward(self, indices):
        # indices: (E, B) -> (E, B, D) using per-ensemble advanced indexing
        if indices.dim() == 2:
            E_idx = torch.arange(self.weight.shape[0], device=self.weight.device).unsqueeze(1)  # (E, 1)
            embedded = self.weight[E_idx, indices]  # (E, B, D)
        else:
            # indices: (B,) -> (E, B, D)
            embedded = self.weight[:, indices]
        return self.dropout(embedded)


class EnsembleGRUModule(nn.Module):
    """GRU module with independent parameters per ensemble member.
    
    Uses manual GRU cell implementation with einsum for vectorized
    computation across ensemble members.

    Input:  (within_ts, ensemble, batch, n_items, features)
    Output: (within_ts, ensemble, batch, n_items, 1)
    """
    def __init__(self, ensemble_size, input_size, dropout=0., **kwargs):
        super().__init__()
        proj_size = 8 + input_size
        hidden_size = 1
        input_size = input_size if input_size > 0 else 1
        
        # Linear projection: (E, proj_size, input_size)
        self.weight_linear = nn.Parameter(torch.empty(ensemble_size, proj_size, input_size))
        self.bias_linear = nn.Parameter(torch.zeros(ensemble_size, proj_size))
        nn.init.xavier_uniform_(self.weight_linear.view(ensemble_size, proj_size, input_size))
        
        # GRU cell parameters: 3 gates (reset, update, new) x hidden_size
        self.weight_ih = nn.Parameter(torch.empty(ensemble_size, 3 * hidden_size, proj_size))
        self.weight_hh = nn.Parameter(torch.empty(ensemble_size, 3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(ensemble_size, 3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(ensemble_size, 3 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih.view(ensemble_size, 3 * hidden_size, proj_size))
        nn.init.orthogonal_(self.weight_hh.view(ensemble_size, 3 * hidden_size, hidden_size))

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.ensemble_size = ensemble_size

    @torch.compile
    def forward(self, inputs, state):
        # inputs: (W, E, B, I, F)
        # state:  (W, E, B, I) — last row is current hidden state
        W, E, B, I, F = inputs.shape

        x = inputs.reshape(W, E, B * I, F)                          # (W, E, B*I, F)
        h = state[-1].reshape(E, B * I, 1) if state is not None else torch.zeros(E, B * I, 1, device=inputs.device)

        # Linear projection via einsum
        y = torch.einsum('eoi,webi->webo', self.weight_linear, x) + self.bias_linear.unsqueeze(1)  # (W, E, B*I, proj)
        y = self.dropout(y)

        # Pre-compute all input-to-hidden gates (independent of h)
        gi_all = torch.einsum('ego,webo->webg', self.weight_ih, y) + self.bias_ih.unsqueeze(1)  # (W, E, B*I, 3*H)

        # GRU cell over within-trial timesteps
        H = self.hidden_size
        outputs = []
        for t in range(W):
            gi = gi_all[t] # (E, B*I, 3*H)
            # gi = torch.einsum('ego,ebo->ebg', self.weight_ih, y[t]) + self.bias_ih.unsqueeze(1)  # (W, E, B*I, 3*H)
            gh = torch.einsum('ego,ebo->ebg', self.weight_hh, h) + self.bias_hh.unsqueeze(1)     # (E, B*I, 3*H)
            r = torch.sigmoid(gi[..., :H] + gh[..., :H])           # reset gate
            z = torch.sigmoid(gi[..., H:2*H] + gh[..., H:2*H])     # update gate
            n = torch.tanh(gi[..., 2*H:] + r * gh[..., 2*H:])      # new gate
            h = (1 - z) * n + z * h                                  # (E, B*I, H)
            outputs.append(h)

        output = torch.stack(outputs)              # (W, E, B*I, H)
        return output.reshape(W, E, B, I, 1)


class EnsembleGRUModule2(torch.nn.Module):
    def __init__(self, ensemble_size, input_size, dropout=0, **kwargs):
        super().__init__()

        proj_size = 8 + input_size
        hidden_size = 1
        
        self.ensemble_size = ensemble_size
        
        # Linear projection: (E, proj_size, input_size)
        self.weight_linear = nn.Parameter(torch.empty(ensemble_size, proj_size, input_size))
        self.bias_linear = nn.Parameter(torch.zeros(ensemble_size, proj_size))
        nn.init.xavier_uniform_(self.weight_linear.view(ensemble_size, proj_size, input_size))
        
        self.ensemble_gru = torch.nn.ParameterList()
        
        self.dropout = torch.nn.Dropout(dropout)
        
        for _ in range(ensemble_size):
            self.ensemble_gru.append(torch.nn.GRU(proj_size, hidden_size))
    
    def forward(self, inputs, state):
        # inputs: (W, E, B, I, F)
        # state:  (W, E, B, I) — last row is current hidden state
        W, E, B, I, F = inputs.shape
        
        x = inputs.reshape(W, E, B * I, F)
        h = state[-1].reshape(1, E, B * I, 1) if state is not None else torch.zeros(E, B * I, 1, device=inputs.device)
        outputs = torch.zeros((W, E, B * I, 1), device=inputs.device)
        
        # Linear projection via einsum
        y = torch.einsum('eoi,webi->webo', self.weight_linear, x) + self.bias_linear.unsqueeze(1)  # (W, E, B*I, proj)
        y = self.dropout(y)
        
        for e in range(self.ensemble_size):
            outputs[:, e] = self.ensemble_gru[e](y[:, e], h[:, e])[0]

        return outputs.reshape(W, E, B, I, 1)
            
class ParameterModule(nn.Module):
    def __init__(self, ensemble_size=1):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(ensemble_size))

    def forward(self, *args, **kwargs):
        return self.parameter

        
class BaseRNN(nn.Module):
    def __init__(
        self,
        spice_config: SpiceConfig,
        
        n_actions,
        n_participants: int = 1,
        n_experiments: int = 1,
        n_items: int = None,
        n_reward_features: int = None,
        
        ensemble_size: int = 1,
        embedding_size: int = 32,
        
        use_sindy: bool = False,
        sindy_polynomial_degree: int = 1,
        sindy_alpha: float = 1e-4,
        
        device=torch.device('cpu'),
        
        **kwargs,
        ):
        super().__init__()

        # Dimension Dictionary:
        # T: TRIAL
        # W: WITHIN_TRIAL_TIMESTEPS
        # E: ENSEMBLE MEMBERS
        # B: BATCH
        # F: FEATURES
        # I: ITEMS
        # A: ACTIONS
        # P: PARTICIPANTS
        # X: EXPERIMENTS
        # C: CANDIDATE TERMS
        
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
        self.ensemble_size = ensemble_size

        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c'
        self.recording = {}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()

        # Differentiable SINDy coefficients
        self.sindy_polynomial_degree = sindy_polynomial_degree
        self.sindy_coefficients = nn.ParameterDict()
        self.sindy_coefficients_presence = {}  # Binary masks to permanently zero out coefficients
        self.sindy_candidate_terms = {}
        self.sindy_degree_weights = {}  # Weights for coefficient penalty based on polynomial degree
        self.sindy_pruning_patience_counters = {}  # Patience counters for thresholding
        self.sindy_specs = {}  # sindy-specific specifications for each module (e.g. include_bias, interaction_only, ...)
        self.sindy_alpha = sindy_alpha
        
        # Setup initial values of RNN
        self.sindy_loss = torch.tensor(0, requires_grad=True, device=device, dtype=torch.float32)
        self.state = None
        self.init_state()  # initial memory state
                
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs: torch.Tensor, prev_state: Dict[str, torch.Tensor], batch_first: bool) -> SpiceSignals:
        # Promote 4D -> 5D by adding ensemble dimension
        if inputs.dim() == 4:
            # (B, T, W, F) -> (E, B, T, W, F)
            if batch_first:
                inputs = inputs.unsqueeze(0).expand(self.ensemble_size, -1, -1, -1, -1)
            else:
                # (T, W, B, F) -> (T, W, E, B, F)
                inputs = inputs.unsqueeze(2).expand(-1, -1, self.ensemble_size, -1, -1)

        # canonical shape: (outer_ts, within_ts, ensemble, batch, features)
        if batch_first:
            inputs = inputs.permute(2, 3, 0, 1, 4)  # (E, B, T, W, F) -> (T, W, E, B, F)

        self.sindy_loss = torch.tensor(0, requires_grad=True, device=self.device, dtype=torch.float32)

        spice_signals = SpiceSignals()

        inputs = inputs.nan_to_num(0.)

        # create a mask of valid trials: [outer_ts, ensemble, batch, 1]
        spice_signals.mask_valid_trials = inputs[:, :, :, :, :self.n_actions].sum(dim=(1, 4)).unsqueeze(-1) > 0

        reward_end = self.n_actions + self.n_reward_features

        # item-specific signals: [outer_ts, within_ts, ensemble, batch, n_actions/n_rewards]
        spice_signals.actions = inputs[:, :, :, :, :self.n_actions].float()
        spice_signals.rewards = inputs[:, :, :, :, self.n_actions:reward_end].float()

        # additional signals: [outer_ts, within_ts, ensemble, batch, n_additional]
        spice_signals.additional_inputs = inputs[:, :, :, :, reward_end:-5].float()

        # static identifiers — (E, B) shaped
        spice_signals.time_trial = inputs[0, :, :, :, -5].int()       # [within_ts, ensemble, batch]
        spice_signals.trials = inputs[:, 0, :, :, -4].int()           # [outer_ts, ensemble, batch]
        spice_signals.blocks = inputs[:, 0, :, :, -3].int()           # [outer_ts, ensemble, batch]
        spice_signals.experiment_ids = inputs[0, 0, :, :, -2].int()   # [ensemble, batch]
        spice_signals.participant_ids = inputs[0, 0, :, :, -1].int()  # [ensemble, batch]

        # use previous state or initialize state if not given
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.init_state(batch_size=inputs.shape[3], within_ts=inputs.shape[1])

        # output signals
        spice_signals.trials = torch.arange(inputs.shape[0], device=self.device)
        spice_signals.logits = torch.zeros((inputs.shape[0], 1, self.ensemble_size, inputs.shape[3], self.n_actions), device=self.device)

        return spice_signals

    def post_forward_pass(self, spice_signals: SpiceSignals, batch_first: bool) -> SpiceSignals:

        if batch_first:
            # (T, 1, E, B, A) -> (E, B, T, 1, A)
            spice_signals.logits = spice_signals.logits.permute(2, 3, 0, 1, 4)

        return spice_signals
    
    def init_state(self, batch_size=1, within_ts=1):
        """Initialize the hidden state with shape (within_ts, ensemble, batch, n_items)."""

        state = {key: torch.full(size=[within_ts, self.ensemble_size, batch_size, self.n_items], fill_value=self.spice_config.memory_state[key], dtype=torch.float32, device=self.device) for key in self.spice_config.memory_state}

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
            return nn.Sequential(EnsembleLinear(self.ensemble_size, embedding_size, 1), activation(**kwargs_activation))
        else:
            return ParameterModule(self.ensemble_size)

    def setup_embedding(self, num_embeddings: int, embedding_size: int, leaky_relu: float = 0.01, dropout: float = 0.):
        return EnsembleEmbedding(self.ensemble_size, num_embeddings, embedding_size, dropout=dropout)
    
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
            
        self.submodules_rnn[key_module] = EnsembleGRUModule(ensemble_size=self.ensemble_size, input_size=input_size, dropout=dropout)
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

        Inputs are of shape [W, E, B, I].
        Lower-dim inputs are broadcast to match.

        Returns:
            torch.Tensor: [within_ts, ensemble, batch, n_items] — full within-trial trajectory
        """

        value = self.state[key_state] if key_state is not None and key_state in self.state else None  # [W, E, B, I]

        E = self.ensemble_size
        B = self.state[list(self.state.keys())[0]].shape[2]
        I = self.n_items
        W = 1  # corrected after inputs processing

        if inputs is None or (isinstance(inputs, tuple) and len(inputs) == 0):
            if value is None:
                raise ValueError(f"When using BaseRNN.call_module you have to give at least a the state variable or inputs. Currently both are None.")
            inputs = torch.zeros((W, E, B, I, 0), dtype=torch.float32, device=self.device)
        elif isinstance(inputs, tuple):
            expanded = []
            for inp in inputs:
                expanded.append(inp.unsqueeze(-1))
            inputs = torch.cat(expanded, dim=-1)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.unsqueeze(-1)

        W = inputs.shape[0]

        if participant_index is None:
            participant_index = torch.zeros(E, B, dtype=torch.int, device=self.device)
        if experiment_index is None:
            experiment_index = torch.zeros(E, B, dtype=torch.int, device=self.device)

        if participant_embedding is None:
            participant_embedding = torch.zeros(E, B, 0, dtype=torch.float32, device=self.device)
        if experiment_embedding is None:
            experiment_embedding = torch.zeros(E, B, 0, dtype=torch.float32, device=self.device)

        embedding = torch.cat((experiment_embedding, participant_embedding), dim=-1)  # [E, B, emb]
        embedding = embedding.view(1, E, B, 1, -1).expand(W, -1, -1, I, -1)  # [W, E, B, I, emb]

        # Replace NaN in inputs
        inputs = torch.nan_to_num(inputs, nan=0.0)

        if key_module in self.submodules_rnn.keys():
            if not self.use_sindy or (self.use_sindy and self.training and self.rnn_training_finished):
                # Get RNN module prediction
                inputs_rnn = torch.cat((inputs, embedding), dim=-1)  # [W, E, B, I, feat+emb]
                next_value = self.submodules_rnn[key_module](inputs_rnn, state=value).squeeze(-1)  # [W, E, B, I]
                if activation_rnn is not None:
                    next_value = activation_rnn(next_value)
                if self.use_sindy:
                    # direct lstsq solve for sindy coefficients (usually training stage 3 - sindy finetuning with fixed rnn parameters)
                    self.sindy_ridge_solve(
                        key_module=key_module, 
                        participant_ids=participant_index,
                        experiment_ids=experiment_index,
                        h_next=next_value,
                        h_current=value,
                        controls=inputs,
                    )
                    
            if self.use_sindy:
                # Get SINDy module prediction — operates per within-trial step
                next_value = torch.zeros((inputs.shape[:-1]), device=inputs.device)  # [W, E, B, I]
                if key_state is not None:
                    next_value_t = value[-1]  # [E, B, I]
                else:
                    next_value_t = torch.zeros(E, B, I, device=self.device)
                for timestep in range(inputs.shape[0]):
                    next_value_t = self.forward_sindy(
                        h_current=next_value_t.unsqueeze(0), # [W=1, E, B, I, F=1]
                        key_module=key_module,
                        participant_ids=participant_index,
                        experiment_ids=experiment_index,
                        controls=inputs[timestep].unsqueeze(0),  # [W=1, E, B, I, F=n_controls]
                        polynomial_degree=self.sindy_polynomial_degree,
                    ).squeeze(0)  # [E, B, I]
                    next_value[timestep] += next_value_t

        elif key_module in self.submodules_eq.keys():
            # hard-coded equation — operates on last within-trial step
            next_value = self.submodules_eq[key_module](value, inputs[-1]).unsqueeze(0)  # [1, E, B, I]

        else:
            raise ValueError(f'Invalid module key {key_module}.')

        # SINDy loss (uses unclipped values, last within-trial step)
        if self.training and not self.rnn_training_finished and participant_index is not None:
            action_mask_2d = action_mask[-1] if action_mask is not None and action_mask.dim() >= 4 else action_mask
            value_0 = self.state[key_state][-1].unsqueeze(0) if key_state is not None else torch.zeros(W, E, B, I, device=self.device)
            self.sindy_loss = self.sindy_loss + self.compute_sindy_loss_for_module(
                    module_name=key_module,
                    h_current=torch.concat((value_0, next_value[:-1])),
                    h_next_rnn=next_value,
                    controls=inputs,
                    action_mask=action_mask_2d,
                    participant_ids=participant_index,
                    experiment_ids=experiment_index,
                    polynomial_degree=self.sindy_polynomial_degree,
                )

        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)

        if action_mask is not None:
            mask = action_mask[-1] if action_mask.dim() >= 4 else action_mask
            next_value = torch.where(mask == 1, next_value, self.state[key_state])

        if key_state is not None:
            self.state[key_state] = next_value

        return next_value  # [W, E, B, I]
    
    def setup_sindy_coefficients(self, key_module: str, polynomial_degree: int = None):
        """
        Initialize learnable SINDy coefficients for each module.
        Shape: (ensemble_size, n_participants, n_experiments, n_library_terms)
        """

        if polynomial_degree is None:
            polynomial_degree = self.sindy_polynomial_degree

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

        # Initialize coefficients: (E, P, X, terms)
        init_coeffs = torch.randn(self.ensemble_size, self.n_participants, self.n_experiments, n_library_terms) * 0.001
        self.sindy_coefficients[key_module] = nn.Parameter(init_coeffs)

        # Initialize mask to all ones (all coefficients active)
        self.sindy_coefficients_presence[key_module] = torch.ones(
            self.ensemble_size, self.n_participants, self.n_experiments, n_library_terms,
            dtype=torch.bool, device=self.device
        )

        # Compute degree-based weights for coefficient penalty
        term_degrees = get_library_term_degrees(self.sindy_candidate_terms[key_module])
        degree_weights = torch.tensor([max(1,d*2) for d in term_degrees], dtype=torch.float32, device=self.device)
        self.sindy_degree_weights[key_module] = degree_weights

        # Initialize patience counters to zero
        self.sindy_pruning_patience_counters[key_module] = torch.zeros(
            self.ensemble_size, self.n_participants, self.n_experiments, n_library_terms,
            dtype=torch.int32, device=self.device
        )
    
    def forward_sindy(self, h_current: torch.Tensor, key_module: str, participant_ids: torch.Tensor, experiment_ids: torch.Tensor, controls: torch.Tensor, polynomial_degree: int):
        """
        Forward pass using SINDy model.

        Args:
            h_current: Current hidden state [W, E, B, I]
            key_module: Name of the module
            participant_ids: Participant indices [E, B]
            experiment_ids: Experiment indices [E, B]
            controls: Control inputs [W, E, B, I, n_controls]
            polynomial_degree: Polynomial degree

        Returns:
            h_next_sindy: Next hidden state [W, E, B, I]
        """
        E = self.ensemble_size
        B = participant_ids.shape[-1]

        # Advanced indexing: coefficients (E, P, X, terms) -> (E, B, terms)
        E_idx = torch.arange(E, device=self.device).unsqueeze(1)  # (E, 1)
        sindy_coeffs = self.sindy_coefficients[key_module][E_idx, participant_ids, experiment_ids]  # (E, B, terms)
        mask = self.sindy_coefficients_presence[key_module][E_idx, participant_ids, experiment_ids]  # (E, B, terms)

        # Apply sparsity mask
        sindy_coeffs = sindy_coeffs * mask.float()

        # Compute polynomial library — fold E*B for compatibility with compute_polynomial_library
        W = h_current.shape[0]
        I = h_current.shape[-1]
        h_folded = h_current.reshape(W, E * B, I)
        controls_folded = controls.reshape(W, E * B, I, -1)

        library_folded = compute_polynomial_library(
            h_folded,
            controls_folded,
            degree=polynomial_degree,
            feature_names=self.sindy_specs[key_module]['input_names'],
            library=self.sindy_candidate_terms[key_module],
        )  # (W, E*B, I, terms)

        library = library_folded.reshape(W, E, B, I, -1)  # (W, E, B, I, terms)

        # Compute predictions: library (W, E, B, I, C) @ coeffs (E, B, C) -> (W, E, B, I)
        h_next_sindy = h_current + torch.einsum('webic,ebc->webi', library, sindy_coeffs)
        
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
        Compute differentiable SINDy reconstruction loss for one module.
        Direct comparison per ensemble member (no cross-ensemble averaging).

        Args:
            module_name: Name of the RNN module
            h_current: Current hidden state [W, E, B, I]
            h_next_rnn: RNN's predicted next state [W, E, B, I]
            controls: Control inputs [W, E, B, I, n_controls]
            action_mask: Binary mask [E, B, I] or None
            participant_ids: Participant indices [E, B]
            experiment_ids: Experiment indices [E, B]
            polynomial_degree: Polynomial degree

        Returns:
            Scalar loss tensor
        """

        if module_name not in self.sindy_coefficients:
            return torch.tensor(0.0, device=self.device)

        h_next_sindy = self.forward_sindy(
            h_current=h_current,
            key_module=module_name,
            participant_ids=participant_ids,
            experiment_ids=experiment_ids,
            controls=controls,
            polynomial_degree=polynomial_degree,
        )  # [W, E, B, I]

        # Direct comparison: [W, E, B, I]
        diff = (h_next_rnn - h_next_sindy) ** 2

        if action_mask is not None:
            masked_diff = torch.where(action_mask == 1, diff, 0)
            n_masked = action_mask.sum(dim=-1).clamp(min=1)  # (E, B)
        else:
            masked_diff = diff
            n_masked = diff.shape[-1]

        sindy_loss = torch.mean(masked_diff.sum(dim=-1) / n_masked) / len(self.submodules_rnn)

        # Clip loss to prevent explosion
        sindy_loss = torch.clamp(sindy_loss, max=100.0)

        return sindy_loss
    
    def sindy_ridge_solve(self, key_module: str, participant_ids: torch.Tensor, experiment_ids: torch.Tensor, h_next: torch.Tensor, h_current: torch.Tensor, controls: torch.Tensor):
        W, E, B, I = h_current.shape
        P = self.n_participants
        X = self.n_experiments
        T = self.sindy_coefficients[key_module].shape[-1]

        library = compute_polynomial_library(
            h_current.reshape(W, B*E, I),
            controls.reshape(W, B*E, I, -1),
            degree=self.sindy_specs[key_module]['polynomial_degree'],
            feature_names=self.sindy_specs[key_module]['input_names'],
            library=self.sindy_candidate_terms[key_module],
        )  # (W, E*B, I, T)

        # Reshape to (W, E, B, I, T)
        library = library.reshape(W, E, B, I, T)
        target = h_next - h_current  # (W, E, B, I)

        # Apply presence mask: zero out pruned library columns per (E, P, X) group
        # mask: (E, P, X, T) -> gather per-sample mask via participant/experiment ids
        E_idx = torch.arange(E, device=self.device).unsqueeze(1)  # (E, 1)
        sample_mask = self.sindy_coefficients_presence[key_module][E_idx, participant_ids, experiment_ids]  # (E, B, T)
        library = library * sample_mask.float().unsqueeze(0).unsqueeze(3)  # (1, E, B, 1, T) -> broadcasts to (W, E, B, I, T)

        # Build group index for each (ensemble, batch) sample -> (participant, experiment) pair
        # participant_ids, experiment_ids: (E, B)
        group_ids = (participant_ids * X + experiment_ids).long()  # (E, B)
        n_groups = P * X

        # Flatten W and I into the "samples" dimension per (E, B) pair
        # library: (W, E, B, I, T) -> (E, B, W*I, T)
        library = library.permute(1, 2, 0, 3, 4).reshape(E, B, W * I, T)
        # target: (W, E, B, I) -> (E, B, W*I, 1)
        target = target.permute(1, 2, 0, 3).reshape(E, B, W * I, 1)

        # Compute per-sample outer products and cross terms
        # AtA_samples: (E, B, T, T), Atb_samples: (E, B, T, 1)
        AtA_samples = library.transpose(-2, -1) @ library  # (E, B, T, T)
        Atb_samples = library.transpose(-2, -1) @ target   # (E, B, T, 1)

        # Scatter-accumulate into (E, n_groups, T, T) and (E, n_groups, T, 1)
        # Expand group_ids to broadcast: (E, B) -> (E, B, 1, 1)
        group_idx = group_ids.unsqueeze(-1).unsqueeze(-1)  # (E, B, 1, 1)

        AtA_accum = torch.zeros(E, n_groups, T, T, device=library.device, dtype=library.dtype)
        Atb_accum = torch.zeros(E, n_groups, T, 1, device=library.device, dtype=library.dtype)
        sample_count = torch.zeros(E, n_groups, device=library.device, dtype=library.dtype)

        AtA_accum.scatter_add_(1, group_idx.expand_as(AtA_samples), AtA_samples)
        Atb_accum.scatter_add_(1, group_idx.expand_as(Atb_samples), Atb_samples)
        sample_count.scatter_add_(1, group_ids, torch.ones_like(group_ids, dtype=library.dtype))

        # Reshape to (E, P, X, T, T) and (E, P, X, T, 1)
        AtA_accum = AtA_accum.reshape(E, P, X, T, T)
        Atb_accum = Atb_accum.reshape(E, P, X, T, 1)
        has_data = sample_count.reshape(E, P, X) > 0  # (E, P, X)

        # Add ridge penalty: lambda * diag(degree_weights) + eps*I for numerical stability
        penalty_diag = torch.diag(self.sindy_alpha * self.sindy_degree_weights[key_module])  # (T, T)
        penalty_diag += 1e-2 * torch.eye(T, device=library.device, dtype=library.dtype)
        AtA_accum = AtA_accum + penalty_diag  # broadcasts over (E, P, X)

        # Only solve for groups that have data; keep existing coefficients for empty groups
        if has_data.all():
            coefficients = torch.linalg.solve(AtA_accum, Atb_accum).squeeze(-1)
            self.sindy_coefficients[key_module].data.copy_(coefficients)
        else:
            coefficients = torch.linalg.solve(AtA_accum[has_data], Atb_accum[has_data]).squeeze(-1)
            self.sindy_coefficients[key_module].data[has_data] = coefficients
        
    def sindy_coefficient_pruning(self, threshold, n_terms_pruning: int = None, patience: int = 1):
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
        module_list = list(self.submodules_rnn.keys())
        
        # Collect all coefficients, masks, and patience counters across modules
        # Shape: [ensemble, n_participants, n_experiments, total_library_terms]
        all_coeffs_abs = torch.cat([self.sindy_coefficients[m].abs() for m in module_list], dim=-1)
        all_masks = torch.cat([self.sindy_coefficients_presence[m] for m in module_list], dim=-1)
        all_patience = torch.cat([self.sindy_pruning_patience_counters[m] for m in module_list], dim=-1)

        # Update patience counters for all coefficients (excluding protected terms)
        below_threshold = (all_coeffs_abs < threshold) & (all_masks == 1)
        all_patience = torch.where(
            below_threshold,
            all_patience + 1,  # increase patience counter by one if below threshold 
            torch.zeros_like(all_patience)  # Reset to 0 if above threshold
        )

        # Identify candidates: active coefficients that have exceeded patience (excluding protected terms)
        is_candidate = (all_patience >= patience) & (all_masks == 1)

        # For n_terms_pruning, only consider coefficients that exceed patience
        temp_coeffs = all_coeffs_abs.clone()
        temp_coeffs[~is_candidate] = torch.inf

        # Find k smallest per [participant, ensemble]
        n_terms_pruning = all_coeffs_abs.shape[-1] if n_terms_pruning is None else n_terms_pruning
        _, indices = torch.topk(temp_coeffs, min(n_terms_pruning, is_candidate.sum().item()), dim=-1, largest=False)
        # Shape: [n_participants, n_ensemble, k]

        # Create pruning mask
        pruning_mask = torch.zeros_like(all_coeffs_abs, dtype=torch.bool)
        pruning_mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
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
        """Returns count of active coefficients per (ensemble, participant, experiment)."""
        coefficients = torch.zeros(self.ensemble_size, self.n_participants, self.n_experiments, device=self.device)
        for module in self.submodules_rnn:
            presence = self.sindy_coefficients_presence[module].clone()
            coefficients += (presence != 0).sum(dim=-1).float()
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
                    
    def get_spice_model_string(self, participant_id: int = 0, experiment_id: int = 0, ensemble_idx: int = None) -> str:
        """
        Get the learned SPICE features and equations as a string.

        Args:
            ensemble_idx: Ensemble member index (default: 0)
            participant_id: Participant index
            experiment_id: Experiment index

        Returns:
            String representation of the SPICE model equations
        """
        lines = []
        max_len_module = max([len(module) for module in self.get_modules()])
            
        for module in self.submodules_rnn:
            sparse_coeffs = (self.sindy_coefficients[module][ensemble_idx, participant_id, experiment_id] * self.sindy_coefficients_presence[module][ensemble_idx, participant_id, experiment_id]).detach().cpu().numpy()
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
            coeffs = self.sindy_coefficients[module].detach()  # (E, P, X, T)
            presence = self.sindy_coefficients_presence[module].float()  # (E, P, X, T)

            # Masked median: only aggregate over ensemble members where term is active
            masked_coeffs = coeffs * presence  # zero out inactive
            masked_coeffs[presence == 0] = float('nan')  # mark inactive as NaN so median ignores them
            aggregated = torch.nanmean(masked_coeffs, dim=0, keepdim=True)[0]  # (1, P, X, T)
            aggregated = torch.nan_to_num(aggregated, nan=0.0)  # terms inactive in all members -> 0
            
            # Presence: term is active if majority of ensemble members have it
            aggregated_presence = (presence.sum(dim=0, keepdim=True) > (self.ensemble_size / 2)).float()
            sindy_coefficients[module] = (aggregated * aggregated_presence).cpu().numpy()

            # Add implicit +1 for the identity term (h_next = h_current + library @ coeffs)
            # identity_idx = self.sindy_candidate_terms[module].index(module)
            # sindy_coefficients[module][..., identity_idx] += 1

        return sindy_coefficients
    
    def eval(self, use_sindy=True):
        super().eval()
        self.use_sindy = use_sindy
        return self
        
    def train(self, mode=True, use_sindy=False):
        super().train(mode)
        # if training mode activate (mode=True) -> do not use sindy for forward pass (self.use_sindy=False)
        self.use_sindy = use_sindy
        return self
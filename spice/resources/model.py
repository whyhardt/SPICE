import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import numpy as np

from .sindy_differentiable import compute_library_size, compute_polynomial_library, get_library_feature_names
from .spice_utils import SpiceConfig, SpiceSignals

# EnsembleGRUModule instances with different input_size share one dynamo cache.
# Allow dynamic parameter and input shapes so all instances reuse a single
# shape-generic compiled graph per train/eval mode (~2 cache entries total).
torch._dynamo.config.force_parameter_static_shapes = False


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
    def __init__(self, ensemble_size, num_embeddings, embedding_dim, n_additional_inputs: int = 0, dropout: float = 0.):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, num_embeddings, embedding_dim))
        if n_additional_inputs > 0:
            self.lin_ai = EnsembleLinear(ensemble_size, n_additional_inputs, embedding_dim)
            self.embedding_fusion = EnsembleLinear(ensemble_size, embedding_dim*2, embedding_dim)
        else:
            self.embedding_fusion = None
        self.dropout = nn.Dropout(p=dropout)
        nn.init.normal_(self.weight)

    def forward(self, indices, additional_inputs: list[torch.Tensor] = None):
        # indices: (E, B) -> (E, B, D) using per-ensemble advanced indexing
        if indices.dim() == 2:
            E_idx = torch.arange(self.weight.shape[0], device=self.weight.device).unsqueeze(1)  # (E, 1)
            embedded = self.weight[E_idx, indices]  # (E, B, D)
        else:
            # indices: (B,) -> (E, B, D)
            embedded = self.weight[:, indices]
        
        if additional_inputs is not None and self.embedding_fusion is not None:
            additional_inputs = torch.concat([a[0, 0] for a in additional_inputs], dim=-1)
            embedded_ai = torch.nn.functional.gelu(self.lin_ai(additional_inputs))
            embedded = torch.nn.functional.gelu(embedded)
            embedded = self.embedding_fusion(torch.concat((embedded, embedded_ai), dim=-1))   
            
        return self.dropout(embedded)


class EnsembleEmbeddingFusion(nn.Module):
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

    def forward(self, *args):
        # args: each (E, B, F_i) -> concat along features -> (E, B, sum(F_i))
        # weight: (E, O, F), output: (E, B, O)
        x = torch.cat([torch.nn.functional.gelu(a) for a in args], dim=-1)
        return torch.einsum('eof,ebf->ebo', self.weight, x) + self.bias.unsqueeze(1)
    

class EnsembleRNNModule(nn.Module):
    """GRU module with independent parameters per ensemble member.
    
    Uses manual GRU cell implementation with einsum for vectorized
    computation across ensemble members.

    Input:  (within_ts, ensemble, batch, n_items, features)
    Output: (within_ts, ensemble, batch, n_items, 1)
    """
    def __init__(self, ensemble_size, input_size, embedding_size, dropout=0., compiled_forward=True, dt: float = 1., include_state: bool = True, **kwargs):
        super().__init__()

        proj_size = 8 + input_size + embedding_size

        self._compile = compiled_forward
        self.dropout = nn.Dropout(p=dropout)
        self.dt = dt
        self.include_state = include_state

        # Linear projection: (E, proj_size, input_size)
        self.weight_linear = nn.Parameter(torch.empty(ensemble_size, proj_size, input_size+embedding_size+1))
        self.bias_linear = nn.Parameter(torch.zeros(ensemble_size, proj_size))
        nn.init.xavier_uniform_(self.weight_linear.view(ensemble_size, proj_size, input_size+embedding_size+1))
        
        # GRU cell parameters: 3 gates (reset, update, new) x hidden_size
        self.weight_n = nn.Parameter(torch.empty(ensemble_size, 1, proj_size))
        self.bias_n = nn.Parameter(torch.zeros(ensemble_size, 1))
        nn.init.xavier_uniform_(self.weight_n.view(ensemble_size, 1, proj_size))

        # Output rescaling layer: learns to rescale bounded [-1,1] values
        # Shape: (E, 1, 1) - learnable scale per ensemble member
        self.weight_out_scale = nn.Parameter(torch.ones(ensemble_size, 1, 1))


        if compiled_forward:
            self._compiled_forward = torch.compile(self._uncompiled_forward, dynamic=True)

    def _uncompiled_forward(self, inputs, state):
        # inputs: (W, E, B, I, F)
        # state:  (W, E, B, I) — last row is current hidden state
        W, E, B, I, F = inputs.shape

        x = inputs.reshape(W, E, B * I, F)                          # (W, E, B*I, F)
        h = state[-1].contiguous().reshape(E, B * I, 1) if self.include_state and state is not None else torch.zeros(E, B * I, 1, device=inputs.device)

        # GRU cell over within-trial timesteps
        outputs = []
        for t in range(W):

            # include_state=False: this module's own state is never a valid
            # input to its dynamics -- not just at the first within-trial step
            # but at every step, so a W>1 module doesn't silently become
            # self-referential after step 0 while its output still accumulates.
            h_in = h if self.include_state else torch.zeros_like(h)
            x_t = torch.concat((x[t], h_in), dim=-1)
            gi = torch.einsum('eoi,ebi->ebo', self.weight_linear, x_t) + self.bias_linear.unsqueeze(1)  # (E, B*I, proj)
            gi = self.dropout(torch.nn.functional.gelu(gi))
            
            # New candidate
            n = torch.einsum('ego,ebo->ebg', self.weight_n, gi) + self.bias_n.unsqueeze(1)     # (E, B*I, 1)

            # New hidden state: bounded + learnable rescaling
            h = h + self.dt * n
            # h_bounded = torch.nn.functional.tanh(h + n)     # (E, B*I, 1) in [-1, 1]
            # h = h_bounded * self.weight_out_scale           # (E, B*I, 1) rescaled by (E, 1, 1)
            
            outputs.append(h)
            
        output = torch.stack(outputs)              # (W, E, B*I, H)
        return output.reshape(W, E, B, I, 1)

    def forward(self, inputs, state):
        if self._compile:
            return self._compiled_forward(inputs, state)
        else:
            return self._uncompiled_forward(inputs, state)
           
            
class ParameterModule(nn.Module):
    def __init__(self, n_ensemble, n_participants, n_experiments):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones((n_ensemble, n_participants, n_experiments)))

    def forward(self, *args, **kwargs):
        return self.parameter

        
class BaseModel(nn.Module):
    
    def __init__(
        self,
        spice_config: SpiceConfig,
        
        n_actions,
        n_participants: int = 1,
        n_experiments: int = 1,
        n_items: int = None,
        n_reward_features: int = None,
        
        ensemble_size: int = 1,
        embedding_size: int = 8,
        
        dropout: float = 0.,
        
        use_sindy: bool = False,
        sindy_polynomial_degree: int = 1,
        sindy_alpha: float = 1e-4,
        fit_sindy: bool = True,
        
        device=torch.device('cpu'),
        compiled_forward=True,
        batch_first: bool = True,

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
        self.batch_first = batch_first
        self.n_actions = n_actions
        self.n_reward_features = n_reward_features if n_reward_features is not None else n_actions
        self.embedding_size = embedding_size
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        self.n_sessions = n_participants * n_experiments
        self.use_sindy = use_sindy
        self.ridge_mode = False
        self._ridge_accumulators = {}
        self.n_items = n_items if n_items is not None else n_actions
        self.ensemble_size = ensemble_size
        self.compiled_forward = compiled_forward
        self.fit_sindy = fit_sindy
        self.dropout = dropout

        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c'
        self.recording = {}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.embedding_fusion = lambda *embeddings: torch.cat(embeddings, dim=-1)
        self.total_embedding_size = 0

        # Differentiable SINDy concepts. The per-participant coefficient vector is
        # not a free parameter: it is factorized as A = Z @ V, where V is a
        # population-level dictionary of concept directions (sparse, unit-norm rows)
        # and Z holds each unit's non-negative loading on those concepts. A term is
        # therefore only ever interpretable together with the other terms its
        # concept owns, and a unit's support is a union of whole concepts rather
        # than an arbitrary subset of terms.
        self.sindy_polynomial_degree = sindy_polynomial_degree
        self.sindy_concept_directions = nn.ParameterDict()  # V: (C, T) -- shared across E, P, X
        self.sindy_concept_loadings = nn.ParameterDict()  # Z: (E, P, X, C) -- non-negative
        self.sindy_concept_support = {}  # (C, T) bool: which terms each concept owns
        self.sindy_concept_gates = {}  # (E, P, X, C) bool: which concepts each unit has
        self.sindy_term_prior_mask = {}  # (T,) bool: theory exclusions (e.g. binary^2=0), applied to V's columns
        self.sindy_candidate_terms = {}
        self.sindy_pruning_patience_counters = {}  # (E, P, X, C) patience for concept gates
        self.sindy_support_patience_counters = {}  # (C, T) patience for concept supports
        self.sindy_specs = {}  # sindy-specific specifications for each module (e.g. include_bias, interaction_only, ...)
        self.sindy_alpha = sindy_alpha
        self.sindy_norm = 1
        
        # Learnable initial state values: for memory_state entries set to None,
        # create per-participant learnable parameters instead of fixed scalars.
        self.learnable_initial_values = nn.ParameterDict()
        for key, val in self.spice_config.memory_state.items():
            if val is None:
                self.learnable_initial_values[key] = nn.Parameter(
                    torch.zeros(self.ensemble_size, self.n_participants)
                )

        # Setup initial values of RNN
        self.sindy_loss_reg = torch.tensor(0, requires_grad=True, device=device, dtype=torch.float32)
        self.sindy_loss_fit = torch.tensor(0, requires_grad=True, device=device, dtype=torch.float32)
        self.state = None
        self.init_state()  # initial memory state
        
        # setup default modules from config
        self.setup_modules_from_config()
        
    def forward(self, inputs, state):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs: torch.Tensor, prev_state: Dict[str, torch.Tensor]) -> SpiceSignals:
        # Promote 4D -> 5D by adding ensemble dimension
        if inputs.dim() == 4:
            # (B, T, W, F) -> (E, B, T, W, F)
            if self.batch_first:
                inputs = inputs.unsqueeze(0).expand(self.ensemble_size, -1, -1, -1, -1)
            else:
                # (T, W, B, F) -> (T, W, E, B, F)
                inputs = inputs.unsqueeze(2).expand(-1, -1, self.ensemble_size, -1, -1)

        # canonical shape: (outer_ts, within_ts, ensemble, batch, features)
        if self.batch_first:
            inputs = inputs.permute(2, 3, 0, 1, 4)  # (E, B, T, W, F) -> (T, W, E, B, F)

        self.sindy_loss_reg = torch.tensor(0, requires_grad=True, device=self.device, dtype=torch.float32)
        self.sindy_loss_fit = torch.tensor(0, requires_grad=True, device=self.device, dtype=torch.float32)

        spice_signals = SpiceSignals()

        inputs = inputs.nan_to_num(0.)

        # create a mask of valid trials: [outer_ts, ensemble, batch, 1]
        spice_signals.mask_valid_trials = inputs[:, :, :, :, :self.n_actions].sum(dim=(1, 4)).unsqueeze(-1) > 0

        reward_end = self.n_actions + self.n_reward_features

        # item-specific signals: [outer_ts, within_ts, ensemble, batch, n_actions/n_rewards]
        spice_signals.actions = inputs[:, :, :, :, :self.n_actions].float()
        spice_signals.feedback = inputs[:, :, :, :, self.n_actions:reward_end].float()

        # additional signals: [outer_ts, within_ts, ensemble, batch, n_additional]
        if len(self.spice_config.additional_inputs) > 0:
            spice_signals.additional_inputs = {}
            additional_inputs = inputs[:, :, :, :, reward_end:-5].float()
            if additional_inputs.shape[-1] != len(self.spice_config.additional_inputs):
                raise ValueError(f"The number of additional inputs in the inputs tensor (dim=-1; index={reward_end}:-5) is different from the list of additional inputs given in SpiceConfig.") 
            for index_ai, ai in enumerate(self.spice_config.additional_inputs):
                spice_signals.additional_inputs[ai] = additional_inputs[..., index_ai].unsqueeze(-1)

        # static identifiers — (E, B) shaped
        spice_signals.time_trial = inputs[0, :, :, :, -5].int()       # [within_ts, ensemble, batch]
        spice_signals.trials = inputs[:, 0, :, :, -4].int()           # [outer_ts, ensemble, batch]
        spice_signals.blocks = inputs[0, 0, :, :, -3].int()           # [outer_ts, ensemble, batch]
        spice_signals.experiment_ids = inputs[0, 0, :, :, -2].int()   # [ensemble, batch]
        spice_signals.participant_ids = inputs[0, 0, :, :, -1].int()  # [ensemble, batch]

        # use previous state or initialize state if not given
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.init_state(batch_size=inputs.shape[3], within_ts=inputs.shape[1])

            # Override learnable initial states with per-participant values
            if self.learnable_initial_values:
                W = inputs.shape[1]
                E_idx = torch.arange(self.ensemble_size, device=self.device).unsqueeze(1)
                for key, param in self.learnable_initial_values.items():
                    init_val = param[E_idx, spice_signals.participant_ids]  # [E, B]
                    self.state[key] = init_val.unsqueeze(0).unsqueeze(-1).expand(
                        W, -1, -1, self.n_items,
                    ).clone()

        # output signals
        spice_signals.trials = torch.arange(inputs.shape[0], device=self.device)
        spice_signals.logits = torch.zeros((inputs.shape[0], 1, self.ensemble_size, inputs.shape[3], self.n_actions), device=self.device)

        return spice_signals

    def post_forward_pass(self, spice_signals: SpiceSignals) -> SpiceSignals:

        if self.batch_first:
            # (T, 1, E, B, A) -> (E, B, T, 1, A)
            spice_signals.logits = spice_signals.logits.permute(2, 3, 0, 1, 4)

        return spice_signals
    
    def init_state(self, batch_size=1, within_ts=1):
        """Initialize the hidden state with shape (within_ts, ensemble, batch, n_items).

        States with None initial value in the config use 0. as placeholder;
        they are overridden with learnable per-participant values in init_forward_pass.
        """

        state = {
            key: torch.full(
                size=[within_ts, self.ensemble_size, batch_size, self.n_items],
                fill_value=val if val is not None else 0.,
                dtype=torch.float32, device=self.device,
            )
            for key, val in self.spice_config.memory_state.items()
        }

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
        self.sindy_loss_reg = self.sindy_loss_reg.to(device)
        self.sindy_loss_fit = self.sindy_loss_fit.to(device)
        # Move masks and patience counters to the correct device
        for module_name in self.sindy_concept_gates:
            self.sindy_concept_gates[module_name] = self.sindy_concept_gates[module_name].to(device)
            self.sindy_concept_support[module_name] = self.sindy_concept_support[module_name].to(device)
            self.sindy_term_prior_mask[module_name] = self.sindy_term_prior_mask[module_name].to(device)
            self.sindy_pruning_patience_counters[module_name] = self.sindy_pruning_patience_counters[module_name].to(device)
            self.sindy_support_patience_counters[module_name] = self.sindy_support_patience_counters[module_name].to(device)

        return self
        
    def setup_constant(self, n_ensemble, n_participants, n_experiments):
        # return ParameterModule(n_ensemble, n_participants, n_experiments)
        return nn.Parameter(torch.zeros((n_ensemble, n_participants, n_experiments)))
    
    def setup_embedding(
        self, 
        num_embeddings: int, 
        embedding_size: int = None, 
        dropout: float = None, 
        target_embedding_size_fusion: int = None, 
        n_additional_inputs: int = 0,
        ):
        if embedding_size is None:
            embedding_size = self.embedding_size
        if dropout is None:
            dropout = self.dropout
        self.setup_embedding_fusion(embedding_size=embedding_size, target_embedding_size=embedding_size if target_embedding_size_fusion is None else target_embedding_size_fusion)
        return EnsembleEmbedding(self.ensemble_size, num_embeddings, embedding_size, dropout=dropout, n_additional_inputs=n_additional_inputs)
    
    def setup_embedding_fusion(self, embedding_size: int, target_embedding_size: int):
        self.total_embedding_size += embedding_size
        # For a single embedding the default concat lambda suffices; for multiple
        # embeddings a learned linear layer fuses them in call_module().
        if self.total_embedding_size > embedding_size:
            self.embedding_fusion = EnsembleEmbeddingFusion(self.ensemble_size, self.total_embedding_size, target_embedding_size)

    def setup_modules_from_config(self, dropout: float = None):
        if dropout is None:
            dropout = self.dropout
            
        for module in self.spice_config.library_setup:
            self.setup_module(key_module=module, dropout=dropout)
    
    def setup_module(
        self,
        key_module: str,
        input_size: int = None,
        embedding_size: int = None,
        dropout: float = None,
        polynomial_degree: int = None,
        include_bias = True,
        include_state = True,
        interaction_only = False,
        dt: float = 1.,
        within_trial_timesteps: bool = False,
        ):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size (_type_): The number of inputs (excluding the memory state); Default to None -> takes input_size from SpiceConfig
            dropout (_type_): Dropout rate before output layer
            dt: Physical time step this module's state update represents (default 1. = a unit
                step, matching prior behavior). Both the RNN's residual update and the SINDy
                fit/execution scale their increment by `dt`, so discovered coefficients read as
                per-unit-time rates rather than per-step deltas that shrink as `dt` shrinks.
            within_trial_timesteps: Whether this module's dynamics evolve over within-trial
                timesteps (W axis) rather than across trials (T axis). Default False (a trial
                module: one update per trial). SINDy refit shoots each module along its own
                axis, so a within-trial module (e.g. evidence accumulation) is fit over W and
                a trial module (e.g. an RL value) over T.

        Returns:
            torch.nn.Module: A torch module which can be called by one line and returns state update
        """

        # GRU network
        if polynomial_degree is None:
            polynomial_degree = self.sindy_polynomial_degree

        if embedding_size is None:
            embedding_size = self.embedding_size

        if input_size is None:
            input_size = len(self.spice_config.library_setup[key_module])

        if dropout is None:
            dropout = self.dropout
        
        self.submodules_rnn[key_module] = EnsembleRNNModule(ensemble_size=self.ensemble_size, input_size=input_size, embedding_size=embedding_size, dropout=dropout, compiled_forward=self.compiled_forward, dt=dt, include_state=include_state)
        self.sindy_specs[key_module] = {}
        self.sindy_specs[key_module]['include_bias'] = include_bias
        self.sindy_specs[key_module]['interaction_only'] = interaction_only
        self.sindy_specs[key_module]['include_state'] = include_state
        self.sindy_specs[key_module]['polynomial_degree'] = polynomial_degree
        self.sindy_specs[key_module]['dt'] = dt
        self.sindy_specs[key_module]['within_trial_timesteps'] = within_trial_timesteps
        self.setup_sindy_concepts(key_module=key_module, polynomial_degree=polynomial_degree)
        
        # set name of each input variable which are then used in the library as features
        input_names = []
        if polynomial_degree > 0:
            start_index = 0
            end_index = np.argmax(
                np.array(
                    [('*' in term) or ('^' in term) for term in self.sindy_candidate_terms[key_module]]
                    )
                )
            if end_index == 0:
                end_index = len(self.sindy_candidate_terms[key_module])
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
        
        if key_state is not None:
            if key_state in self.state:
                # If include_state=False, the module is stateless: start from zeros
                # but still write output to state for tracking (e.g. Stage 2 shooting).
                if self.sindy_specs.get(key_module, {}).get('include_state', True):
                    value = self.get_state()[key_state]  # [W, E, B, I]
                else:
                    value = None
            else:
                KeyError(f"key_state {key_state} is not in BaseModel's state.")
        else:
            value = None
            
        E = self.ensemble_size
        B = self.state[list(self.state.keys())[0]].shape[2]
        I = self.n_items
        W = 1  # corrected after inputs processing

        if inputs is None or (isinstance(inputs, tuple) and len(inputs) == 0):
            if value is None:
                raise ValueError(f"When using BaseModel.call_module you have to give at least a the state variable or inputs. Currently both are None.")
            inputs = torch.zeros((W, E, B, I, 0), dtype=torch.float32, device=self.device)
        elif isinstance(inputs, tuple):
            expanded = []
            for inp in inputs:
                if inp.shape[-1] == self.n_items:
                    pass
                elif inp.shape[-1] == 1 and self.n_items > 1:
                    inp = inp.expand(-1, -1, -1, self.n_items)
                expanded.append(inp.unsqueeze(-1))
            inputs = torch.cat(expanded, dim=-1)
        elif isinstance(inputs, torch.Tensor):
            if inputs.shape[-1] == self.n_items:
                    pass
            elif inputs.shape[-1] == 1 and self.n_items > 1:
                inputs = inputs.expand(-1, -1, -1, self.n_items)
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

        # embedding = torch.cat((experiment_embedding, participant_embedding), dim=-1)  # [E, B, emb]
        embedding = self.embedding_fusion(participant_embedding, experiment_embedding)
        embedding = embedding.view(1, E, B, 1, -1).expand(W, -1, -1, I, -1)  # [W, E, B, I, emb]
        
        # Replace NaN in inputs
        inputs = torch.nan_to_num(inputs, nan=0.0)
        
        if key_module in self.submodules_rnn.keys():
            if not self.use_sindy or self.ridge_mode:
                # Get RNN module prediction
                inputs_rnn = torch.cat((inputs, embedding), dim=-1)  # [W, E, B, I, feat+emb]
                next_value = self.submodules_rnn[key_module](inputs_rnn, state=value).squeeze(-1)  # [W, E, B, I]
                if activation_rnn is not None:
                    next_value = activation_rnn(next_value)
                if self.ridge_mode:
                    # Accumulate this chunk's normal-equation contribution for sindy coefficients.
                    # h_current must be the per-step preceding value (value[-1] for w=0,
                    # next_value[w-1] for w>0), matching compute_sindy_loss_for_module below --
                    # `value` alone is the constant state entering this call, wrong
                    # for every w>0 whenever W>1 (within-trial dynamics).
                    value_0 = value[-1].unsqueeze(0) if value is not None else torch.zeros(1, E, B, I, device=self.device)
                    self.sindy_ridge_accumulate(
                        key_module=key_module,
                        participant_ids=participant_index,
                        experiment_ids=experiment_index,
                        h_next=next_value,
                        h_current=torch.concat((value_0, next_value[:-1])),
                        controls=inputs,
                    )
            
            if self.use_sindy:
                # Get SINDy module prediction — operates per within-trial step
                next_value = torch.zeros((inputs.shape[:-1]), device=inputs.device)  # [W, E, B, I]
                if value is not None:
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
        if (self.fit_sindy
            and self.training
            and not self.use_sindy
            and participant_index is not None
            ):
            action_mask_2d = action_mask[-1] if action_mask is not None and action_mask.dim() >= 4 else action_mask
            value_0 = value[-1].unsqueeze(0) if value is not None else torch.zeros(1, E, B, I, device=self.device)
            sindy_loss_reg, sindy_loss_fit = self.compute_sindy_loss_for_module(
                    module_name=key_module,
                    h_current=torch.concat((value_0, next_value[:-1])),
                    h_next_rnn=next_value,
                    controls=inputs,#.detach(),
                    action_mask=action_mask_2d,
                    participant_ids=participant_index,
                    experiment_ids=experiment_index,
                    polynomial_degree=self.sindy_polynomial_degree,
                )
            self.sindy_loss_reg = self.sindy_loss_reg + sindy_loss_reg
            self.sindy_loss_fit = self.sindy_loss_fit + sindy_loss_fit

        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        if action_mask is not None:
            mask = action_mask[-1] if action_mask.dim() >= 4 else action_mask
            next_value = torch.where(mask == 1, next_value,
                                     self.get_state()[key_state] if key_state is not None else torch.zeros_like(next_value))
            
        if key_state is not None:
            self.state[key_state] = next_value

        return next_value  # [W, E, B, I]
    
    def setup_sindy_concepts(self, key_module: str, polynomial_degree: int = None):
        """
        Initialize the concept factorization A = Z @ V for one module.

        V (concept directions): (n_concepts, n_library_terms), shared across ensemble
        members, participants and experiments; rows are unit-norm.
        Z (concept loadings): (ensemble_size, n_participants, n_experiments, n_concepts),
        constrained non-negative. Z is a binary gate mask times a magnitude, and the mask
        starts all-open -- as does the support mask over terms. All sparsity comes from
        the L1 prox and from pruning; none of it is seeded at initialization.

        Concepts start with *full* support -- every concept may draw on every allowed
        term -- because being multi-term is the entire point of a concept. Support only
        ever shrinks, so whatever is unreachable at initialization stays unreachable:
        seeding one term per concept would permanently confine the dictionary to
        single-term concepts and collapse the model back to the per-term parametrization
        it is meant to replace.

        Directions are random unit-norm rather than near-zero. The unit-norm gauge is
        what removes the multiplicative degeneracy Z @ V = (Z D)(D^-1 V), and a near-zero
        V would simply be blown up by the first renormalization. The "start from almost
        no dynamics" behaviour lives in Z, which starts small and strictly positive --
        strictly, because a loading resting exactly at zero gets no gradient under the
        non-negativity constraint and its concept would be dead from the first step.

        Signs live in V, which is free and signed, so a decay-plus-drive mechanism such
        as a Rescorla-Wagner update is a single concept (-1 Q, +1 r) carrying one loading
        per participant, rather than two coefficients that happen to sum to one.

        n_concepts defaults to ceil(n_terms / 2): deliberately undercomplete, since an
        overcomplete dictionary is precisely where the factorization stops being
        identifiable. Concepts can retire but never spawn, so this is a real
        hyperparameter -- too small cannot be recovered from mid-run.
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

        # Concept directions V: random unit-norm rows, (C, T)
        n_concepts = max(1, -(-n_library_terms // 2))
        directions = torch.randn(n_concepts, n_library_terms, device=self.device)
        directions /= directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self.sindy_concept_directions[key_module] = nn.Parameter(directions)

        # Concept loadings Z: (E, P, X, C), strictly positive so no concept starts dead
        self.sindy_concept_loadings[key_module] = nn.Parameter(
            torch.rand(self.ensemble_size, self.n_participants, self.n_experiments, n_concepts) * 1e-3 + 1e-4
        )

        # Concept support: full. Terms leave a concept only by pruning.
        self.sindy_concept_support[key_module] = torch.ones(
            n_concepts, n_library_terms, dtype=torch.bool, device=self.device
        )

        # Concept gates: which concepts each unit has (all open initially)
        self.sindy_concept_gates[key_module] = torch.ones(
            self.ensemble_size, self.n_participants, self.n_experiments, n_concepts,
            dtype=torch.bool, device=self.device
        )

        # Term-level prior mask (theory-driven, never revived by pruning). Uniform
        # across units by construction -- every setter in the codebase writes it per
        # term index -- so it lives in T-space and applies to V's columns.
        # preprocess_coefficients() runs after all setup_module() calls and sets the
        # correct entries afterward.
        self.sindy_term_prior_mask[key_module] = torch.ones(
            n_library_terms, dtype=torch.bool, device=self.device
        )

        # Patience counters: one per gate, one per support entry
        self.sindy_pruning_patience_counters[key_module] = torch.zeros(
            self.ensemble_size, self.n_participants, self.n_experiments, n_concepts,
            dtype=torch.int32, device=self.device
        )
        self.sindy_support_patience_counters[key_module] = torch.zeros(
            n_concepts, n_library_terms, dtype=torch.int32, device=self.device
        )

    @torch.no_grad()
    def reset_concepts(self, key_module: Optional[str] = None) -> None:
        """Re-draw a module's factorization from the initial distribution.

        Used when a stage wants to rediscover structure from scratch (stage 2.1) rather
        than inherit whatever stage 1 converged to. Concept count is never resized --
        retired concepts are zeroed rows, so re-drawing simply refills them.
        """
        modules = self.get_modules() if key_module is None else [key_module]
        for module in modules:
            directions = torch.randn_like(self.sindy_concept_directions[module].data)
            directions /= directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            self.sindy_concept_directions[module].data.copy_(directions)
            self.sindy_concept_support[module] = torch.ones_like(self.sindy_concept_support[module])
            self.sindy_concept_support[module] &= self.sindy_term_prior_mask[module].unsqueeze(0)
            self.sindy_concept_gates[module] = torch.ones_like(self.sindy_concept_gates[module])
            self.sindy_concept_gates[module] &= self.sindy_concept_support[module].any(dim=-1)
            self.sindy_pruning_patience_counters[module].zero_()
            self.sindy_support_patience_counters[module].zero_()
            self.reinit_loadings(module)

    @torch.no_grad()
    def reinit_loadings(self, key_module: Optional[str] = None, scale: float = 1e-3) -> None:
        """Re-draw loadings small and strictly positive, within the open gates.

        Strictly positive matters: a loading sitting exactly at zero receives no
        gradient under the non-negativity constraint, and its concept would be dead for
        the rest of training.
        """
        modules = self.get_modules() if key_module is None else [key_module]
        for module in modules:
            loadings = self.sindy_concept_loadings[module]
            fresh = torch.rand_like(loadings) * scale + 1e-4
            loadings.data.copy_(fresh * self.sindy_concept_gates[module].float())

    def effective_directions(self, key_module: str) -> torch.Tensor:
        """(C, T) concept directions with support and theory exclusions applied."""
        mask = self.sindy_concept_support[key_module] & self.sindy_term_prior_mask[key_module].unsqueeze(0)
        return self.sindy_concept_directions[key_module] * mask.float()

    def effective_loadings(self, key_module: str) -> torch.Tensor:
        """(E, P, X, C) non-negative loadings with closed gates zeroed."""
        loadings = self.sindy_concept_loadings[key_module].clamp(min=0.0)
        return loadings * self.sindy_concept_gates[key_module].float()

    def compose_coefficients(self, key_module: str) -> torch.Tensor:
        """(E, P, X, T) per-unit coefficient vectors implied by the factorization."""
        return torch.einsum(
            'epxc,ct->epxt',
            self.effective_loadings(key_module),
            self.effective_directions(key_module),
        )

    def derived_presence(self, key_module: str) -> torch.Tensor:
        """(E, P, X, T) bool: a term is present for a unit iff some open concept owns it.

        This replaces the free per-unit presence mask. A unit's support is now a union
        of whole concepts, which is what stops the independent per-unit topk from
        manufacturing a distinct support for every participant.
        """
        support = self.sindy_concept_support[key_module] & self.sindy_term_prior_mask[key_module].unsqueeze(0)
        return torch.einsum(
            'epxc,ct->epxt',
            self.sindy_concept_gates[key_module].float(),
            support.float(),
        ) > 0

    @torch.no_grad()
    def normalize_concept_directions(self, key_module: Optional[str] = None) -> None:
        """Fix the multiplicative gauge: unit-norm V rows, inverse scale into Z.

        Z @ V is invariant under (Z D, D^-1 V) for any positive diagonal D, so without
        this the cheapest way to shrink an L1 penalty on Z is to inflate V at no cost to
        the fit, and the sparsity pressure silently evaporates. Call after every
        optimizer step while a penalty is pushing on Z.

        The gauge is fixed on the *masked* row -- the vector effective_directions()
        actually hands to the einsum -- not on the raw parameter. Those differ: nothing
        else re-masks the parameter, so an optimizer step refills coordinates that
        pruning zeroed and they silently regrow. Normalizing the raw row then leaves the
        effective one at norm < 1 and lets magnitude accumulate outside the support,
        where it is invisible to both the norm budget and the L1 prox.
        """
        modules = self.get_modules() if key_module is None else [key_module]
        for module in modules:
            directions = self.sindy_concept_directions[module]
            mask = (self.sindy_concept_support[module]
                    & self.sindy_term_prior_mask[module].unsqueeze(0))
            directions.data *= mask.float()
            norms = directions.data.norm(dim=-1, keepdim=True)  # (C, 1)
            # Dead concepts (all-zero rows) have nothing to normalize; leave them be.
            scale = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
            directions.data /= scale
            self.sindy_concept_loadings[module].data *= scale.squeeze(-1)

    @torch.no_grad()
    def project_loadings(self, key_module: Optional[str] = None) -> None:
        """Re-impose the two hard constraints on Z: non-negativity and closed gates.

        This is a constraint projection, not a penalty. The L1 on the loadings lives in
        the objective (compute_factorization_penalty), so nothing here depends on the
        learning rate and lambda cannot silently change meaning when the LR schedule does.
        """
        modules = self.get_modules() if key_module is None else [key_module]
        for module in modules:
            loadings = self.sindy_concept_loadings[module]
            loadings.data.clamp_(min=0.0)
            loadings.data *= self.sindy_concept_gates[module].float()

    def compute_factorization_penalty(self, sindy_lambda_loading: float = 0.0,
                                      sindy_lambda_concept: float = 0.0,
                                      sindy_lambda_group: float = 0.0) -> torch.Tensor:
        """Sparse group lasso on the factorization:

            lambda_z * ||Z||_1 + lambda_v * ||V||_1 + lambda_g * sum_c ||Z[..., c]||_2

        All three run over the *effective* (gated, support-masked) tensors, so closed
        gates and pruned terms contribute neither value nor gradient. The first two are
        normalized differently on purpose, because their gradients are diluted
        differently:

        V is population-level -- every participant's data contributes to dL/dv -- so its
        fit gradient is O(1) against a mean-reduced behavioural loss and a plain sum is
        already on the right scale. Z is per-participant, so the fit gradient reaching any
        single loading is diluted by ~1/(E*P) while a summed penalty is not; summing there
        makes the same numeric lambda hit Z about P times harder than V, and makes lambda
        depend on the cohort size. Averaging over units fixes both: lambda_z is the price
        of one concept for one participant, in units of the mean behavioural loss.

        The ratio lambda_v : lambda_z sets concept density, and that is the knob worth
        thinking about. With unit-norm V rows a k-dense direction delivers sqrt(k) of
        coefficient mass per unit of loading, so a penalty on Z alone is minimised by
        making every concept as dense as possible -- a density pump. ||v||_1 ranges over
        [1, sqrt(T)] on the unit sphere, so lambda_v prices exactly that density.

        lambda_g is the group-lasso term, and it is the only part of the objective that
        looks along the *participant* axis. The elementwise L1 above cannot see concept
        duplication at all: with Z >= 0, both Z @ V and sum(Z) depend only on the total
        mass along a set of duplicate directions, so splitting a mechanism across four
        near-identical concepts costs exactly the same as concentrating it in one. That
        leaves the population free to fragment across redundant capacity, which shows up
        downstream as spurious individual differences -- participants differing only in
        which copy of a mechanism they were assigned.

        Grouping over (E, P, X) per concept fixes that, because the L2 norm of a column
        grows like sqrt(n) rather than n: the first participant to load on a concept pays
        a lot, the hundredth pays almost nothing. Concentration is cheaper than
        fragmentation, and the per-entry gradient z_p / ||z_c||_2 is largest exactly where
        a column is least populated, so sparsely-used concepts bleed out while well-used
        ones become sticky. Dividing by sqrt(n_units) keeps lambda_g independent of cohort
        size, on the same scale as lambda_z: it is the price of a concept the whole
        population holds.

        Note that lambda_g shrinks a column by a single scalar factor shared by every
        participant in it, so it can retire a concept outright but never distinguishes
        participants *within* a surviving one. That is lambda_z's job, and why both terms
        are needed -- sparse group lasso, not group lasso alone.
        """
        penalty = torch.tensor(0.0, device=self.device)
        for module in self.get_modules():
            if sindy_lambda_loading > 0 or sindy_lambda_group > 0:
                loadings = self.effective_loadings(module)               # (E, P, X, C)
                n_units = loadings.shape[0] * loadings.shape[1] * loadings.shape[2]
            if sindy_lambda_loading > 0:
                penalty = penalty + sindy_lambda_loading * loadings.sum() / n_units
            if sindy_lambda_group > 0:
                # eps keeps the sqrt differentiable at zero: a retired concept's column is
                # all-zero, and d||z||/dz = z/||z|| would be 0/0 there.
                column_norms = loadings.pow(2).sum(dim=(0, 1, 2)).add(1e-12).sqrt()  # (C,)
                penalty = penalty + sindy_lambda_group * column_norms.sum() / n_units ** 0.5
            if sindy_lambda_concept > 0:
                penalty = penalty + sindy_lambda_concept * self.effective_directions(module).abs().sum()
        return penalty

    @torch.no_grad()
    def set_concepts(self, key_module: str, directions: torch.Tensor, loadings: torch.Tensor) -> None:
        """Install an exact concept dictionary, resizing n_concepts to match.

        Unlike set_coefficients_from_dense(), which projects onto whatever dictionary is
        already there, this replaces the dictionary outright. Intended for generative
        ground-truth models, where the mechanisms are known and must be represented
        exactly rather than approximated -- a projection onto a random undercomplete
        dictionary would silently corrupt any parameter-recovery comparison.

        Args:
            directions: (C, T) concept directions. Rows are normalized here.
            loadings: (E, P, X, C) non-negative loadings.
        """
        directions = directions.to(self.device).float()
        loadings = loadings.to(self.device).float()
        if (loadings < 0).any():
            raise ValueError("Concept loadings must be non-negative.")

        norms = directions.norm(dim=-1, keepdim=True)
        scale = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
        directions = directions / scale
        loadings = loadings * scale.squeeze(-1)

        n_concepts, n_terms = directions.shape
        expected = self.sindy_concept_directions[key_module].shape[-1]
        if n_terms != expected:
            raise ValueError(f"{key_module}: directions have {n_terms} terms, expected {expected}.")

        self.sindy_concept_directions[key_module] = nn.Parameter(directions)
        self.sindy_concept_loadings[key_module] = nn.Parameter(loadings)
        self.sindy_concept_support[key_module] = (directions != 0)
        self.sindy_concept_gates[key_module] = (loadings != 0)
        self.sindy_pruning_patience_counters[key_module] = torch.zeros(
            *loadings.shape, dtype=torch.int32, device=self.device)
        self.sindy_support_patience_counters[key_module] = torch.zeros(
            n_concepts, n_terms, dtype=torch.int32, device=self.device)

    @torch.no_grad()
    def concepts_from_dense(self, key_module: str, coefficients: torch.Tensor) -> None:
        """Install an exact dictionary reproducing a dense (E, P, X, T) coefficient tensor.

        Builds one concept per (term, sign) that is actually used, which represents any
        signed coefficient tensor exactly under the non-negativity constraint on the
        loadings. Terms nobody uses get no concept, so C stays as small as the ground
        truth allows.
        """
        coefficients = coefficients.to(self.device).float()
        n_terms = coefficients.shape[-1]

        used_positive = (coefficients > 0).any(dim=0).any(dim=0).any(dim=0)
        used_negative = (coefficients < 0).any(dim=0).any(dim=0).any(dim=0)

        rows, columns = [], []
        for index_term in range(n_terms):
            if used_positive[index_term]:
                rows.append(torch.eye(n_terms, device=self.device)[index_term])
                columns.append(coefficients[..., index_term].clamp(min=0.0))
            if used_negative[index_term]:
                rows.append(-torch.eye(n_terms, device=self.device)[index_term])
                columns.append((-coefficients[..., index_term]).clamp(min=0.0))

        if not rows:  # nothing used: keep a single inert concept
            rows = [torch.zeros(n_terms, device=self.device)]
            columns = [torch.zeros(coefficients.shape[:-1], device=self.device)]

        self.set_concepts(key_module, torch.stack(rows, dim=0), torch.stack(columns, dim=-1))

    @torch.no_grad()
    def set_coefficients_from_dense(self, key_module: str, coefficients: torch.Tensor,
                                    n_iter: int = 200) -> None:
        """Project a dense (E, P, X, T) coefficient tensor onto the concept dictionary.

        The target generally lies outside the span of the concepts -- the dictionary is
        undercomplete and the loadings are non-negative -- so this is a non-negative
        least-squares fit, solved by projected gradient descent. It is a projection, not
        an assignment: the result reproduces the target only to the extent the current
        dictionary can express it.

        Used by the ridge initializer and by anything holding dense coefficients that
        need expressing in concept space.
        """
        coefficients = coefficients.to(self.sindy_concept_loadings[key_module].dtype)

        directions = self.effective_directions(key_module)  # (C, T)
        gram = directions @ directions.T  # (C, C)
        cross = torch.einsum('epxt,ct->epxc', coefficients, directions)
        # Lipschitz constant of the quadratic; guards the fixed step size.
        step = 1.0 / (torch.linalg.matrix_norm(gram, ord=2).clamp(min=1e-12))
        loadings = self.sindy_concept_loadings[key_module].data.clamp(min=0.0)
        for _ in range(n_iter):
            gradient = torch.einsum('epxc,cd->epxd', loadings, gram) - cross
            loadings = (loadings - step * gradient).clamp(min=0.0)
        self.sindy_concept_loadings[key_module].data.copy_(loadings)

        self.project_loadings(key_module=key_module)


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

        # Advanced indexing: loadings (E, P, X, C) -> (E, B, C), then compose against the
        # population-level concept directions. Composing back into term space here keeps
        # the library einsum below -- and every downstream consumer -- unchanged.
        E_idx = torch.arange(E, device=self.device).unsqueeze(1)  # (E, 1)
        loadings = self.effective_loadings(key_module)[E_idx, participant_ids, experiment_ids]  # (E, B, C)
        sindy_coeffs = loadings @ self.effective_directions(key_module)  # (E, B, terms)

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
        dt = self.sindy_specs[key_module].get('dt', 1.)
        h_next_sindy = h_current + dt * torch.einsum('webic,ebc->webi', library, sindy_coeffs)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute differentiable SINDy reconstruction loss for one module.
        Direct comparison per ensemble member (no cross-ensemble averaging).

        Returns two decoupled loss terms:
        - sindy_loss_reg: gradients flow only to RNN parameters (h_next_sindy detached).
          Scaled by sindy_weight in the training loop to control RNN regularization strength.
        - sindy_loss_fit: gradients flow only to SINDy coefficients (h_next_rnn detached).
          Independent of sindy_weight so coefficient fitting is decoupled from RNN regularization.

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
            Tuple of (sindy_loss_reg, sindy_loss_fit) scalar tensors
        """

        if module_name not in self.sindy_concept_directions:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        h_next_sindy = self.forward_sindy(
            h_current=h_current,
            key_module=module_name,
            participant_ids=participant_ids,
            experiment_ids=experiment_ids,
            controls=controls,
            polynomial_degree=polynomial_degree,
        )  # [W, E, B, I]

        # Decoupled losses: detach opposite side so gradients flow to one param set only.
        # Both h_next_rnn and h_next_sindy are h_current + dt*(...), so their raw
        # difference carries a factor of dt that gets squared away to dt^2 -- for
        # dt << 1 this silently attenuates sindy_weight by ~dt^2, well below any
        # value tuned for dt=1 modules. Divide by dt first to compare in rate space,
        # consistent with sindy_ridge_accumulate's target normalization.
        dt = self.sindy_specs[module_name].get('dt', 1.)
        diff_reg = ((h_next_rnn - h_next_sindy) / dt) ** 2  # full gradients
        # diff_reg = (h_next_rnn - h_next_sindy.detach()) ** 2  # gradients → RNN only
        diff_fit = ((h_next_rnn.detach() - h_next_sindy) / dt) ** 2  # gradients → SINDy coefficients only

        if action_mask is not None:
            masked_diff_reg = torch.where(action_mask == 1, diff_reg, 0)
            masked_diff_fit = torch.where(action_mask == 1, diff_fit, 0)
            n_masked = action_mask.sum(dim=-1).clamp(min=1)  # (E, B)
        else:
            masked_diff_reg = diff_reg
            masked_diff_fit = diff_fit
            n_masked = diff_reg.shape[-1]

        n_modules = len(self.submodules_rnn)
        sindy_loss_reg = torch.clamp(torch.mean(masked_diff_reg.sum(dim=-1) / n_masked) / n_modules, max=100.0)
        sindy_loss_fit = torch.clamp(torch.mean(masked_diff_fit.sum(dim=-1) / n_masked) / n_modules, max=100.0)

        return sindy_loss_reg, sindy_loss_fit
    
    def reset_ridge_accumulators(self) -> None:
        """Clear accumulated ridge normal equations for all modules.

        Call once before the first chunk of a new (possibly multi-chunk) ridge solve --
        see sindy_ridge_accumulate/sindy_ridge_finalize.
        """
        self._ridge_accumulators = {}

    def sindy_ridge_accumulate(self, key_module: str, participant_ids: torch.Tensor, experiment_ids: torch.Tensor,
                          h_next: torch.Tensor, h_current: torch.Tensor, controls: torch.Tensor) -> None:
        """Accumulate one chunk's normal-equation contribution toward key_module's ridge solve.

        Builds the polynomial library from (h_current, controls) for this chunk and scatter-adds
        its A^T A / A^T b contributions into self._ridge_accumulators[key_module], per
        (participant, experiment) group. Calling this repeatedly over chunks of a dataset (each
        with reset_ridge_accumulators() called once beforehand) and then sindy_ridge_finalize()
        is mathematically equivalent to solving on the whole dataset at once, but never needs to
        materialize the full dataset's forward pass in memory simultaneously -- necessary for
        architectures with large per-sample state (e.g. SpiceDDM's evidence_pdf grid).

        Args:
            key_module: Module name.
            participant_ids: (E, B) participant indices.
            experiment_ids: (E, B) experiment indices.
            h_next: (W, E, B, I) RNN target states.
            h_current: (W, E, B, I) current states (or None → zeros).
            controls: (W, E, B, I, n_controls) control signals.
        """
        W, E, B, I = h_next.shape
        P = self.n_participants
        X = self.n_experiments
        T = self.sindy_concept_directions[key_module].shape[-1]

        if h_current is None:
            h_current = torch.zeros_like(h_next)

        library = compute_polynomial_library(
            h_current.reshape(W, B*E, I),
            controls.reshape(W, B*E, I, -1),
            degree=self.sindy_specs[key_module]['polynomial_degree'],
            feature_names=self.sindy_specs[key_module]['input_names'],
            library=self.sindy_candidate_terms[key_module],
        )  # (W, E*B, I, T)

        # Reshape to (W, E, B, I, T). float64: condition numbers of 1e8-1e13 are
        # routine here, and float32 can misreport such matrices as singular.
        library = library.reshape(W, E, B, I, T).double()
        dt = self.sindy_specs[key_module].get('dt', 1.)
        target = ((h_next - h_current) / dt).double()  # (W, E, B, I) -- per-unit-time rate, not per-step delta

        # Apply presence mask: zero out pruned library columns per (E, P, X) group.
        # Presence is derived from the factorization -- a term is available to a unit
        # iff one of its open concepts owns it.
        E_idx = torch.arange(E, device=self.device).unsqueeze(1)  # (E, 1)
        sample_mask = self.derived_presence(key_module)[E_idx, participant_ids, experiment_ids]  # (E, B, T)
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

        # Scatter-accumulate into this module's persistent (E, n_groups, T, T) / (E, n_groups, T, 1)
        # accumulators, creating them on first use. Expand group_ids to broadcast: (E, B) -> (E, B, 1, 1)
        group_idx = group_ids.unsqueeze(-1).unsqueeze(-1)  # (E, B, 1, 1)

        # b^T b and the row count complete the sufficient statistics: with them,
        # RSS(c) = c^T A^T A c - 2 c^T A^T b + b^T b is computable in closed form
        # from the accumulators alone, with no forward pass. sindy_ridge_finalize
        # ignores both -- they exist for consumers that need absolute (not just
        # relative) residuals, e.g. the noise-scale estimate a Gaussian BIC needs.
        btb_samples = (target ** 2).sum(dim=2).squeeze(-1)  # (E, B)
        n_rows_samples = torch.full_like(btb_samples, float(library.shape[2]))

        accum = self._ridge_accumulators.get(key_module)
        if accum is None:
            accum = {
                'AtA': torch.zeros(E, n_groups, T, T, device=library.device, dtype=library.dtype),
                'Atb': torch.zeros(E, n_groups, T, 1, device=library.device, dtype=library.dtype),
                'count': torch.zeros(E, n_groups, device=library.device, dtype=library.dtype),
                'btb': torch.zeros(E, n_groups, device=library.device, dtype=library.dtype),
                'n_rows': torch.zeros(E, n_groups, device=library.device, dtype=library.dtype),
            }
            self._ridge_accumulators[key_module] = accum

        accum['AtA'].scatter_add_(1, group_idx.expand_as(AtA_samples), AtA_samples)
        accum['Atb'].scatter_add_(1, group_idx.expand_as(Atb_samples), Atb_samples)
        accum['count'].scatter_add_(1, group_ids, torch.ones_like(group_ids, dtype=library.dtype))
        accum['btb'].scatter_add_(1, group_ids, btb_samples)
        accum['n_rows'].scatter_add_(1, group_ids, n_rows_samples)

    def sindy_ridge_finalize(self, key_module: str, ridge_alpha: float = None) -> bool:
        """Solve the ridge-regularized normal equations accumulated via sindy_ridge_accumulate.

        Adds the ridge penalty to the accumulated A^T A, solves per (participant, experiment)
        group for a dense coefficient vector, then projects that solution into the concept
        factorization via set_coefficients_from_dense(). Call once per module after all
        chunks of a solve have been accumulated.

        The solve itself stays in dense term space because the normal equations are only
        quadratic there; the result is then projected onto the concept dictionary as a
        non-negative least-squares fit, so it initializes the loadings rather than
        reproducing the dense solution exactly.

        Args:
            key_module: Module name.
            ridge_alpha: Ridge penalty strength. Defaults to self.sindy_alpha.

        Returns:
            True if the solve succeeded, False if it failed (e.g. singular matrix). True if no
            data was accumulated for this module (nothing to solve).
        """
        accum = self._ridge_accumulators.get(key_module)
        if accum is None:
            return True

        P = self.n_participants
        X = self.n_experiments
        T = self.sindy_concept_directions[key_module].shape[-1]
        E = accum['AtA'].shape[0]
        alpha = ridge_alpha if ridge_alpha is not None else self.sindy_alpha

        AtA_accum = accum['AtA'].reshape(E, P, X, T, T)
        Atb_accum = accum['Atb'].reshape(E, P, X, T, 1)
        has_data = accum['count'].reshape(E, P, X) > 0  # (E, P, X)

        # Add ridge penalty: alpha*I + eps*I for numerical stability
        penalty_diag = (alpha + 1e-4) * torch.eye(T, device=AtA_accum.device, dtype=AtA_accum.dtype)
        AtA_accum = AtA_accum + penalty_diag  # broadcasts over (E, P, X)

        # Solve; return False on failure (singular matrix, etc.)
        try:
            dense = torch.zeros(E, P, X, T, device=AtA_accum.device, dtype=AtA_accum.dtype)
            if has_data.all():
                dense = torch.linalg.solve(AtA_accum, Atb_accum).squeeze(-1)
            else:
                dense[has_data] = torch.linalg.solve(AtA_accum[has_data], Atb_accum[has_data]).squeeze(-1)
        except torch.linalg.LinAlgError:
            return False

        # Terms excluded by theory never enter the factorization
        dense = dense * self.sindy_term_prior_mask[key_module].to(dense.dtype)
        self.set_coefficients_from_dense(key_module, dense)

        return True


    def concept_gate_patience(self, threshold: float):
        """Advance the patience counter of every open gate whose loading is below threshold."""
        module_list = list(self.submodules_rnn.keys())

        all_loadings = torch.cat([self.effective_loadings(m).detach() for m in module_list], dim=-1)
        all_gates = torch.cat([self.sindy_concept_gates[m] for m in module_list], dim=-1)
        all_patience = torch.cat([self.sindy_pruning_patience_counters[m] for m in module_list], dim=-1)

        below_threshold = (all_loadings < threshold) & all_gates
        all_patience = torch.where(
            below_threshold,
            all_patience + 1,
            torch.zeros_like(all_patience),
        )

        start_idx = 0
        for module in module_list:
            n_concepts = self.sindy_concept_gates[module].shape[-1]
            self.sindy_pruning_patience_counters[module] = all_patience[..., start_idx:start_idx + n_concepts]
            start_idx += n_concepts

    def concept_support_patience(self, threshold: float):
        """Advance the patience counter of every support entry whose |V| is below threshold."""
        for module in self.submodules_rnn:
            directions = self.sindy_concept_directions[module].detach().abs()
            support = self.sindy_concept_support[module]
            counters = self.sindy_support_patience_counters[module]
            below_threshold = (directions < threshold) & support
            self.sindy_support_patience_counters[module] = torch.where(
                below_threshold,
                counters + 1,
                torch.zeros_like(counters),
            )

    @torch.no_grad()
    def prune_concept_gates(self, patience: int = 1, n_concepts_pruning: int = None):
        """Close the weakest concept gates, per unit (member x participant x experiment).

        This is the per-participant half of pruning: it decides only *which concepts a
        unit has*, never which terms exist. Term-level structure is a population decision
        made by prune_concept_support(), so an independent topk here can no longer
        manufacture a distinct term support for every participant.
        """
        module_list = list(self.submodules_rnn.keys())

        all_loadings = torch.cat([self.effective_loadings(m).detach() for m in module_list], dim=-1)
        all_gates = torch.cat([self.sindy_concept_gates[m] for m in module_list], dim=-1)
        all_patience = torch.cat([self.sindy_pruning_patience_counters[m] for m in module_list], dim=-1)

        is_candidate = (all_patience >= patience) & all_gates
        if not is_candidate.any():
            return

        scores = all_loadings.clone()
        scores[~is_candidate] = torch.inf

        # Rate limiting: close at most n_concepts_pruning gates per unit and event, so
        # the budget computed once before training lets the gates reach 0 exactly over
        # the expected number of pruning events.
        k = int(is_candidate.sum(dim=-1).max().item())
        if n_concepts_pruning is not None:
            k = min(k, int(n_concepts_pruning))
        if k == 0:
            return
        _, indices = torch.topk(scores, k, dim=-1, largest=False)

        pruning_mask = torch.zeros_like(all_gates)
        pruning_mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
        pruning_mask &= is_candidate  # Safety: only close actual candidates

        start_idx = 0
        for module in module_list:
            n_concepts = self.sindy_concept_gates[module].shape[-1]
            keep_mask = ~pruning_mask[..., start_idx:start_idx + n_concepts]
            self.sindy_concept_gates[module] &= keep_mask
            self.sindy_concept_loadings[module].data *= keep_mask.float()
            self.sindy_pruning_patience_counters[module] *= keep_mask.int()
            start_idx += n_concepts

        for module in module_list:
            self.retire_dead_concepts(module)

    @torch.no_grad()
    def prune_concept_support(self, patience: int = 1, n_terms_pruning: int = None):
        """Drop the weakest terms out of concept directions -- a population-level decision.

        Runs per module over the (C, T) direction matrix, so a term leaves a concept for
        every participant at once and term support can no longer fragment across units.
        Directions are renormalized afterwards and concepts nobody loads on retired.
        """
        for module in self.submodules_rnn:
            directions = self.sindy_concept_directions[module]
            support = self.sindy_concept_support[module]
            counters = self.sindy_support_patience_counters[module]

            is_candidate = (counters >= patience) & support
            if not is_candidate.any():
                continue

            scores = directions.data.abs().clone()
            scores[~is_candidate] = torch.inf

            flat = scores.reshape(-1)
            # Rate limiting: drop at most n_terms_pruning support entries per module and
            # event, on its own budget independent of the gate budget above.
            k = int(is_candidate.sum().item())
            if n_terms_pruning is not None:
                k = min(k, int(n_terms_pruning))
            if k == 0:
                continue
            _, indices = torch.topk(flat, k, largest=False)

            pruning_mask = torch.zeros_like(flat, dtype=torch.bool)
            pruning_mask.scatter_(0, indices, torch.ones_like(indices, dtype=torch.bool))
            pruning_mask = pruning_mask.reshape(scores.shape) & is_candidate

            keep_mask = ~pruning_mask
            self.sindy_concept_support[module] &= keep_mask
            directions.data *= keep_mask.float()
            self.sindy_support_patience_counters[module] *= keep_mask.int()

            self.normalize_concept_directions(module)
            self.retire_dead_concepts(module)

    @torch.no_grad()
    def retire_dead_concepts(self, key_module: str) -> int:
        """Zero out concepts that no unit loads on, or that own no terms.

        This is the only way C ever falls: a concept no participant loads on carries no
        information, and leaving it in place would let it drift and reappear later.
        """
        support = self.sindy_concept_support[key_module]
        gates = self.sindy_concept_gates[key_module]

        alive = gates.any(dim=0).any(dim=0).any(dim=0) & support.any(dim=-1)  # (C,)
        dead = ~alive
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        self.sindy_concept_support[key_module][dead] = False
        self.sindy_concept_gates[key_module][..., dead] = False
        self.sindy_concept_directions[key_module].data[dead] = 0.0
        self.sindy_concept_loadings[key_module].data[..., dead] = 0.0
        self.sindy_support_patience_counters[key_module][dead] = 0
        self.sindy_pruning_patience_counters[key_module][..., dead] = 0
        return n_dead

    def count_spice_parameters(self) -> Dict[str, torch.Tensor]:
        """Degrees of freedom of the fitted model, split by axis.

        Returns:
            'loadings': (P, X) open concept gates per participant -- the per-participant
                degrees of freedom.
            'directions': scalar -- free values in the shared concept dictionary summed
                over modules. Each live concept costs one fewer than its support size,
                since the unit-norm constraint removes one degree of freedom.

        The two are deliberately never summed. Merging them amortizes population-level
        structure over participants, which makes pooling look nearly free and biases any
        BIC-driven search toward pooling everything -- whereas the claim SPICE makes is
        about the *form* of the equations, not about population variance in a single
        coefficient. Report them as separate numbers.
        """
        loadings = torch.zeros(self.n_participants, self.n_experiments, device=self.device)
        directions = torch.zeros((), device=self.device)

        for module in self.submodules_rnn:
            gates = self.sindy_concept_gates[module]  # (E, P, X, C)
            support = self.sindy_concept_support[module] & self.sindy_term_prior_mask[module].unsqueeze(0)

            # A concept counts for a unit if any ensemble member holds it open
            loadings += gates.any(dim=0).float().sum(dim=-1)

            alive = gates.any(dim=0).any(dim=0).any(dim=0) & support.any(dim=-1)  # (C,)
            support_sizes = support.sum(dim=-1)  # (C,)
            directions += torch.where(alive, (support_sizes - 1).clamp(min=0), torch.zeros_like(support_sizes)).sum()

        return {'loadings': loadings, 'directions': directions}

    def compute_constants_penalty(self, strength: float) -> torch.Tensor:
        """L1/L2 penalty on any learnable constants (e.g. switch biases).

        Concept loadings and directions are *not* penalized here -- their L1 lives in
        compute_factorization_penalty().
        """
        assert self.sindy_norm == 1 or self.sindy_norm == 2, "Only L1-norm or L2-norm are allowed."

        penalty = torch.tensor(0.0, device=self.device)
        if strength == 0:
            return penalty

        if hasattr(self, 'constants') and isinstance(self.constants, torch.nn.ParameterDict):
            for param in self.constants.values():
                if self.sindy_norm == 2:
                    penalty += (param ** 2).mean()
                else:
                    penalty += param.abs().mean()

        return penalty * strength

                    
    def get_spice_model_string(self, participant_id: int = 0, experiment_id: int = 0) -> str:
        """Render each module as its concept dictionary plus this participant\'s loadings.

        Per-participant structure is reported as loadings on population-level concepts,
        not as a free-standing per-term equation: a coefficient is only interpretable
        alongside the other terms its concept owns, and raw per-term supports were never
        comparable across participants in the first place.

        Equations are shown in increment form, which is what the model actually
        parametrizes -- a Rescorla-Wagner update reads as one concept, alpha * (r - Q),
        rather than as two coefficients that happen to sum to one.
        """
        lines = []
        for module in self.submodules_rnn:
            directions = self.effective_directions(module).detach().cpu().numpy()
            loadings = self.effective_loadings(module).detach().mean(dim=0)
            loadings = loadings[participant_id, experiment_id].cpu().numpy()
            terms = self.sindy_candidate_terms[module]

            lines.append(f"d{module}/dt =")
            n_shown = 0
            for index_concept in range(directions.shape[0]):
                if loadings[index_concept] == 0 or not np.any(directions[index_concept] != 0):
                    continue
                body = []
                for index_term, term in enumerate(terms):
                    value = directions[index_concept, index_term]
                    if value == 0:
                        continue
                    rendered = f"{term}[t]" if term == module else term
                    body.append(f"{value:+.3f} {rendered}")
                lines.append(
                    f"    {loadings[index_concept]:8.3f} * [ {' '.join(body)} ]"
                    f"    (concept {index_concept})"
                )
                n_shown += 1
            if n_shown == 0:
                lines.append("    0")

        return "\n".join(lines)

    def print(self, participant_id: int = 0, experiment_id: int = 0) -> None:
        print(self.get_spice_model_string(participant_id=participant_id, experiment_id=experiment_id))

    def get_modules(self):
        return [module for module in self.submodules_rnn]

    def get_candidate_terms(self, key_module: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
        if key_module is None:
            return self.sindy_candidate_terms
        else:
            return self.sindy_candidate_terms[key_module]

    def get_concepts(self, key_module: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Per module, the (C, T) population-level concept dictionary (support applied)."""
        modules = self.get_modules() if key_module is None else (
            [key_module] if isinstance(key_module, str) else key_module
        )
        return {module: self.effective_directions(module).detach() for module in modules}

    def get_concept_loadings(self, key_module: Optional[str] = None, aggregate: bool = False) -> Dict[str, torch.Tensor]:
        """Per module, each unit\'s non-negative loading on every concept.

        Shape (E, P, X, C), or (P, X, C) averaged over ensemble members that hold the
        concept open when aggregate=True. This is the per-participant quantity to report
        and to run individual-differences analyses on.
        """
        modules = self.get_modules() if key_module is None else (
            [key_module] if isinstance(key_module, str) else key_module
        )

        loadings = {}
        for module in modules:
            values = self.effective_loadings(module).detach()
            if aggregate:
                gates = self.sindy_concept_gates[module]
                values = torch.where(gates, values, torch.full_like(values, float('nan')))
                values = torch.nan_to_num(torch.nanmean(values, dim=0), nan=0.0)
            loadings[module] = values
        return loadings

    def get_sindy_coefficients(self, key_module: Optional[str] = None, aggregate: bool = False):
        """Per module, the coefficients implied by the factorization.

        Derived from Z @ V rather than stored, but the shape contract is unchanged --
        (E, P, X, T), or (P, X, T) when aggregate=True -- so downstream consumers that
        just want the fitted equation keep working. Use get_concept_loadings() for
        anything that reports individual differences.
        """
        modules = self.get_modules() if key_module is None else (
            [key_module] if isinstance(key_module, str) else key_module
        )

        sindy_coefficients = {}
        for module in modules:
            coefficients = self.compose_coefficients(module).detach()
            if aggregate:
                presence = self.derived_presence(module)
                masked = torch.where(presence, coefficients, torch.full_like(coefficients, float('nan')))
                sindy_coefficients[module] = torch.nan_to_num(torch.nanmean(masked, dim=0), nan=0.0)
            else:
                sindy_coefficients[module] = coefficients

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
    
    def __call__(self, *args, **kwargs):
        logits, state = super().__call__(*args, **kwargs)
        return logits, state
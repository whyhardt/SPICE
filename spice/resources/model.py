import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import numpy as np

from .sindy_differentiable import compute_library_size, compute_polynomial_library, get_library_feature_names, get_library_term_degrees
from .spice_utils import SpiceConfig, SpiceSignals

# EnsembleGRUModule instances with different input_size share one dynamo cache.
# Allow dynamic parameter and input shapes so all instances reuse a single
# shape-generic compiled graph per train/eval mode (~2 cache entries total).
torch._dynamo.config.force_parameter_static_shapes = False


class ParameterModule(nn.Module):
    def __init__(self, ensemble_size=1):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(ensemble_size))

    def forward(self, *args, **kwargs):
        return self.parameter
    

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

    
class EnsemblePolynomialLayer(nn.Module):
    """Polynomial projection: element-wise product of ``degree`` independent linear projections.
    
    For degree=1: linear (no non-linearity)
    For degree=2: bilinear — (W1 x + b1) * (W2 x + b2), exactly degree-2 polynomial
    For degree=3: trilinear, exactly degree-3 polynomial
    ...and so on.

    The output is an exact degree-N polynomial in the input, which aligns
    perfectly with a SINDy polynomial library of the same degree.

    Parameters per projection: (ensemble_size, output_size, input_size)
    Input:  (E, B, F)
    Output: (E, B, O)
    """
    def __init__(self, ensemble_size: int, input_size: int, output_size: int, degree: int = 2):
        super().__init__()
        assert degree >= 1, f"degree must be >= 1, got {degree}"
        self.degree = degree

        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
            for _ in range(degree)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(ensemble_size, output_size))
            for _ in range(degree)
        ])
        for w in self.weights:
            nn.init.xavier_normal_(w, gain=0.01)# ** 0.5)
            
    def forward(self, x, weight_offsets=None, bias_offsets=None):
        """Forward pass through multilinear projection.

        Args:
            x: (E, B, F) input tensor
            weight_offsets: Optional list of (E, B, O, I) per-batch weight offsets (one per degree).
                When provided, effective weights are group + offset per batch element.
            bias_offsets: Optional list of (E, B, O) per-batch bias offsets (one per degree).
        """
        # x: (E, B, F) -> (E, B, O)
        if weight_offsets is None:
            # Original path: shared weights across batch
            result = torch.einsum('eoi,ebi->ebo', self.weights[0], x) + self.biases[0].unsqueeze(1)
            for w, b in zip(self.weights[1:], self.biases[1:]):
                result = result * (torch.einsum('eoi,ebi->ebo', w, x) + b.unsqueeze(1))
        else:
            # Per-batch path: group weights + per-participant offsets
            w_eff = self.weights[0].unsqueeze(1) + weight_offsets[0]  # (E,1,O,I) + (E,B,O,I)
            b_eff = self.biases[0].unsqueeze(1) + bias_offsets[0]     # (E,1,O) + (E,B,O)
            result = torch.einsum('eboi,ebi->ebo', w_eff, x) + b_eff
            for k in range(1, self.degree):
                w_eff = self.weights[k].unsqueeze(1) + weight_offsets[k]
                b_eff = self.biases[k].unsqueeze(1) + bias_offsets[k]
                result = result * (torch.einsum('eboi,ebi->ebo', w_eff, x) + b_eff)
        if self.degree > 1:
            result = result * (1.0 / self.degree ** 0.5)
        return result


class EnsembleRNNModule(nn.Module):
    """RNN module with independent parameters per ensemble member.

    Uses a residual update with a multilinear projection whose polynomial
    degree can be matched to the SINDy library degree.

    When ``embedding_size > 0``, a hypernetwork generates per-participant
    weight offsets from the embedding, so each participant gets its own
    effective RNN weights: ``W_eff = W_group + W_offset(embedding)``.
    The group-level weights are the structured parameters (bias of the
    hypernetwork conceptually), while the offset generator encodes how
    individual participants deviate from the group.

    Input:  (within_ts, ensemble, batch, n_items, features)
    Output: (within_ts, ensemble, batch, n_items, 1)
    """
    def __init__(self, ensemble_size, input_size, embedding_size=0, dropout=0., compiled_forward=True, polynomial_degree=2, max_offset_ratio=0.5, **kwargs):
        super().__init__()

        self._compile = compiled_forward
        self.embedding_size = embedding_size
        self.max_offset_ratio = max_offset_ratio

        proj_size = input_size * polynomial_degree + 1
        hidden_size = 1
        input_size = input_size if input_size > 0 else 1

        # Multilinear projection: degree-N polynomial non-linearity
        self.projection = EnsemblePolynomialLayer(
            ensemble_size=ensemble_size,
            input_size=input_size + 1,  # +1 for hidden state
            output_size=proj_size,
            degree=polynomial_degree,
        )

        # Output projection
        self.weight_n = nn.Parameter(torch.empty(ensemble_size, 1, proj_size))
        self.bias_n = nn.Parameter(torch.zeros(ensemble_size, 1))
        nn.init.xavier_uniform_(self.weight_n.view(ensemble_size, 1, proj_size))

        # Linear damping (group-level, shared across participants)
        self.damping_coefficient = nn.Parameter(torch.full((ensemble_size, 1, 1), -3.0))

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.ensemble_size = ensemble_size

        # Hypernetwork: embedding -> per-participant weight offsets
        # Two-layer MLP with GELU non-linearity. The final layer is zero-initialized
        # so the model starts as a pure group model (zero offsets).
        if embedding_size > 0:
            # Compute split sizes for each weight/bias matrix
            self._offset_split_sizes = []
            for w in self.projection.weights:
                self._offset_split_sizes.append(w.shape[1] * w.shape[2])  # O * I_in
            for b in self.projection.biases:
                self._offset_split_sizes.append(b.shape[1])  # O
            self._offset_split_sizes.append(self.weight_n.shape[1] * self.weight_n.shape[2])  # 1 * proj_size
            self._offset_split_sizes.append(self.bias_n.shape[1])  # 1
            self._offset_split_sizes.append(1)  # damping coefficient offset

            n_total_params = sum(self._offset_split_sizes)
            hypernet_hidden = embedding_size
            self.hypernet_in = EnsembleLinear(ensemble_size, embedding_size, hypernet_hidden)
            self.hypernet_out = EnsembleLinear(ensemble_size, hypernet_hidden, n_total_params)
            # Zero-init output layer so offsets start at zero
            nn.init.zeros_(self.hypernet_out.weight)
            nn.init.zeros_(self.hypernet_out.bias)

        if compiled_forward:
            self._compiled_forward = torch.compile(self._uncompiled_forward, dynamic=True)

    @staticmethod
    def _constrain_offset(offset, reference, max_ratio):
        """Scale offset so its norm doesn't exceed max_ratio * reference norm.

        Projects offset onto a ball of radius ``max_ratio * ||reference||``.
        Offsets within budget pass through unchanged — no expressivity loss.
        As group weights grow during training, the offset budget grows proportionally.

        Args:
            offset: (E, B, ...) per-participant offset
            reference: (E, ...) group-level parameter (detached for norm computation)
            max_ratio: maximum ratio of offset norm to reference norm
        """
        param_dims = tuple(range(1, reference.dim()))
        offset_dims = tuple(range(2, offset.dim()))

        ref_norm = reference.detach().norm(dim=param_dims) if param_dims else reference.detach().abs().squeeze(-1)
        off_norm = offset.norm(dim=offset_dims) if offset_dims else offset.abs().squeeze(-1)

        # Floor prevents dead offsets when group weights are near zero (e.g. bias init)
        max_norm = (ref_norm.unsqueeze(1) * max_ratio).clamp(min=0.01)
        scale = (max_norm / (off_norm + 1e-8)).clamp(max=1.0)

        for _ in range(len(offset_dims)):
            scale = scale.unsqueeze(-1)

        return offset * scale

    def _generate_weight_offsets(self, embedding):
        """Generate per-participant weight offsets from embedding.

        Args:
            embedding: (E, B, emb_dim) — B already includes item expansion if needed.

        Returns:
            proj_w_offsets: list of (E, B, O, I_in) per degree
            proj_b_offsets: list of (E, B, O) per degree
            out_w_offset: (E, B, 1, proj_size)
            out_b_offset: (E, B, 1)
        """
        E = embedding.shape[0]
        B = embedding.shape[1]

        # EnsembleLinear expects ensemble dim second-to-last: (B, E, emb)
        # Transpose before MLP, transpose back after.
        emb_t = embedding.transpose(0, 1)  # (B, E, emb)
        offsets = self.hypernet_out(self.dropout(torch.nn.functional.gelu(self.hypernet_in(self.dropout(emb_t)))))
        offsets = offsets.transpose(0, 1)  # (E, B, n_total)

        parts = offsets.split(self._offset_split_sizes, dim=-1)

        # Projection weight offsets (constrained relative to group weights)
        idx = 0
        proj_w_offsets = []
        for w in self.projection.weights:
            O, I_in = w.shape[1], w.shape[2]
            offset = self._constrain_offset(parts[idx].reshape(E, B, O, I_in), w, self.max_offset_ratio)
            proj_w_offsets.append(offset)
            idx += 1

        # Projection bias offsets
        proj_b_offsets = []
        for b in self.projection.biases:
            offset = self._constrain_offset(parts[idx], b, self.max_offset_ratio)
            proj_b_offsets.append(offset)
            idx += 1

        # Output weight offset
        out_w_offset = self._constrain_offset(
            parts[idx].reshape(E, B, self.weight_n.shape[1], self.weight_n.shape[2]),
            self.weight_n, self.max_offset_ratio,
        )
        idx += 1

        # Output bias offset
        out_b_offset = self._constrain_offset(parts[idx], self.bias_n, self.max_offset_ratio)
        
        return proj_w_offsets, proj_b_offsets, out_w_offset, out_b_offset

    def precompute_offsets(self, embedding, n_items):
        """Pre-compute per-participant weight offsets from embedding.

        Call once per forward pass; results are reused across trials.

        Args:
            embedding: (E, B, emb_dim)
            n_items: number of items (I) for expansion

        Returns:
            Tuple of (proj_w_offsets, proj_b_offsets, out_w_offset, out_b_offset)
        """
        # Expand embedding for items: (E, B, D) -> (E, B*I, D)
        emb_expanded = embedding.unsqueeze(2).expand(-1, -1, n_items, -1).reshape(
            embedding.shape[0], embedding.shape[1] * n_items, embedding.shape[2]
        )
        return self._generate_weight_offsets(emb_expanded)

    def _uncompiled_forward(self, inputs, state, precomputed_offsets=None):
        # inputs: (W, E, B, I, F) — control signals only (no embeddings)
        # state:  (W, E, B, I) — last row is current hidden state
        # precomputed_offsets: tuple from precompute_offsets() or None
        W, E, B, I, F = inputs.shape

        x = inputs.reshape(W, E, B * I, F)                          # (W, E, B*I, F)
        h = state[-1].contiguous().reshape(E, B * I, 1) if state is not None else torch.zeros(E, B * I, 1, device=inputs.device)

        if precomputed_offsets is not None:
            proj_w_offsets, proj_b_offsets, out_w_offset, out_b_offset = precomputed_offsets
        else:
            proj_w_offsets = proj_b_offsets = None
            out_w_offset = out_b_offset = None

        outputs = []
        for t in range(W):
            x_t = torch.concat((h, x[t]), dim=-1)
            gi = self.projection(x_t, weight_offsets=proj_w_offsets, bias_offsets=proj_b_offsets)

            # Candidate: per-participant output weights if hypernetwork active
            if out_w_offset is not None:
                wn_eff = self.weight_n.unsqueeze(1) + out_w_offset    # (E,1,1,O) + (E,B,1,O) -> (E,B,1,O)
                bn_eff = self.bias_n.unsqueeze(1) + out_b_offset      # (E,1,1) + (E,B,1) -> (E,B,1)
                n = torch.einsum('ebgo,ebo->ebg', wn_eff, gi) + bn_eff
            else:
                n = torch.einsum('ego,ebo->ebg', self.weight_n, gi) + self.bias_n.unsqueeze(1)     # (E, B*I, 1)

            # Residual update with damping
            update_gate = torch.nn.functional.sigmoid(self.damping_coefficient)
            h = (1 - update_gate) * h + update_gate * n                                               # (E, B*I, 1)

            outputs.append(h)

        output = torch.stack(outputs)              # (W, E, B*I, H)
        return output.reshape(W, E, B, I, 1)

    def forward(self, inputs, state, precomputed_offsets=None):
        if self._compile:
            return self._compiled_forward(inputs, state, precomputed_offsets)
        else:
            return self._uncompiled_forward(inputs, state, precomputed_offsets)

        
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
        embedding_size: int = 32,
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
        self.dropout = dropout
        self.n_participants = n_participants
        self.n_experiments = n_experiments
        self.n_sessions = n_participants * n_experiments
        self.use_sindy = use_sindy
        self.rnn_training_finished = False
        self.n_items = n_items if n_items is not None else n_actions
        self.ensemble_size = ensemble_size
        self.compiled_forward = compiled_forward
        self.fit_sindy = fit_sindy

        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c'
        self.recording = {}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.module_output_scales = nn.ParameterDict()  # Learnable output scale per module (for logit computation)

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
        self.aggregate = False
        self.sindy_loss = torch.tensor(0, requires_grad=True, device=device, dtype=torch.float32)
        self.state = None
        self.init_state()  # initial memory state
        
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
        spice_signals.blocks = inputs[0, 0, :, :, -3].int()           # [outer_ts, ensemble, batch]
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

    def post_forward_pass(self, spice_signals: SpiceSignals) -> SpiceSignals:

        if self.batch_first:
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
        embedding_size: int = 0,
        dropout: float = 0.,
        polynomial_degree: int = None,
        max_offset_ratio: float = 0.5,
        include_bias = True,
        include_state = True,
        interaction_only = False,
        rescale_at_output = True,
        ):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size: The number of control signal inputs (excluding the memory state and embeddings)
            embedding_size: Size of participant/experiment embedding for hypernetwork weight generation.
                When > 0, the RNN generates per-participant weights from the embedding instead of
                receiving it as an input. This keeps the bilinear input space clean (controls + state only).
            dropout: Dropout rate before output layer
            max_offset_ratio: Maximum ratio of per-participant offset norm to group weight norm.
                Constrains individual deviations to be at most this fraction of the group weights,
                preventing bilinear amplification from destabilizing training.

        Returns:
            torch.nn.Module: A torch module which can be called by one line and returns state update
        """

        if polynomial_degree is None:
            polynomial_degree = self.sindy_polynomial_degree

        self.submodules_rnn[key_module] = EnsembleRNNModule(
            ensemble_size=self.ensemble_size,
            input_size=input_size,
            embedding_size=embedding_size,
            dropout=dropout,
            compiled_forward=self.compiled_forward,
            polynomial_degree=polynomial_degree,
            max_offset_ratio=max_offset_ratio,
            )
        self.sindy_specs[key_module] = {}
        self.sindy_specs[key_module]['include_bias'] = include_bias
        self.sindy_specs[key_module]['interaction_only'] = interaction_only
        self.sindy_specs[key_module]['include_state'] = include_state
        self.sindy_specs[key_module]['polynomial_degree'] = polynomial_degree
        self.setup_sindy_coefficients(key_module=key_module, polynomial_degree=polynomial_degree)

        # Initialize learnable output scale for this module (one per ensemble member)
        # This scales the output for logit computation while keeping internal state normalized
        # if rescale_at_output:
        #     self.module_output_scales[key_module] = nn.Parameter(torch.ones(self.ensemble_size))            
        
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
                value = self.get_state()[key_state]  # [W, E, B, I]
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

        # Replace NaN in inputs
        inputs = torch.nan_to_num(inputs, nan=0.0)

        # Hypernet offsets: during training, regenerate each call_module() invocation
        # (= each trial) for fresh dropout masks. During eval, cache once per forward pass.
        if self.training or key_module not in self._hypernet_offsets:
            rnn = self.submodules_rnn[key_module] if key_module in self.submodules_rnn else None
            if rnn is not None and rnn.embedding_size > 0 and embedding.shape[-1] > 0:
                self._hypernet_offsets[key_module] = rnn.precompute_offsets(embedding, self.n_items)
            else:
                self._hypernet_offsets[key_module] = None

        if key_module in self.submodules_rnn.keys():
            if (not self.use_sindy  # RNN mode
                # or (self.use_sindy and self.training and self.rnn_training_finished)  # SINDy LSTSQ-solve
                ):
                # Get RNN module prediction — offsets pre-computed from embedding via hypernetwork
                next_value = self.submodules_rnn[key_module](inputs, state=value, precomputed_offsets=self._hypernet_offsets[key_module]).squeeze(-1)  # [W, E, B, I]
                if activation_rnn is not None:
                    next_value = activation_rnn(next_value)
                # if self.use_sindy:
                #     # direct lstsq solve for sindy coefficients
                #     self.sindy_ridge_solve(
                #         key_module=key_module, 
                #         participant_ids=participant_index,
                #         experiment_ids=experiment_index,
                #         h_next=next_value,
                #         h_current=value,
                #         controls=inputs,
                #     )
            
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

        # Buffer states for vectorized SINDy loss (computed in __call__ after forward)
        if (self.fit_sindy
            and self.training
            and not self.rnn_training_finished
            and participant_index is not None
            ):
            action_mask_2d = action_mask[-1] if action_mask is not None and action_mask.dim() >= 4 else action_mask
            value_0 = self.get_state()[key_state][-1].unsqueeze(0) if key_state is not None else torch.zeros(W, E, B, I, device=self.device)

            if key_module not in self._sindy_buffers:
                self._sindy_buffers[key_module] = {
                    'h_current': [],
                    'h_next_rnn': [],
                    'controls': [],
                    'action_mask': [],
                    'participant_ids': participant_index,
                    'experiment_ids': experiment_index,
                }

            self._sindy_buffers[key_module]['h_current'].append(torch.cat((value_0, next_value[:-1])))
            self._sindy_buffers[key_module]['h_next_rnn'].append(next_value)
            self._sindy_buffers[key_module]['controls'].append(inputs)
            self._sindy_buffers[key_module]['action_mask'].append(
                action_mask_2d if action_mask_2d is not None
                else torch.ones(E, B, I, device=self.device)
            )

        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-100, max=100)

        # Apply action mask: store complete state, but return only masked update
        if action_mask is not None:
            state = self.get_state()[key_state]
            mask = action_mask[-1] if action_mask.dim() >= 4 else action_mask
            next_value = next_value * mask  # Masked update (unmasked items → 0)
            old_state = state * (1-mask) if key_state is not None else torch.zeros_like(next_value)
            next_state = next_value + old_state  # Complete state (masked: new, unmasked: old)
        else:
            next_state = next_value

        # Store normalized state (for stability and SINDy fitting)
        if key_state is not None:
            self.state[key_state] = next_state

        # Apply learned output scale for logit computation (compensates for layer norm)
        # Return scaled masked update (only items updated by this module contribute)
        # if key_module in self.module_output_scales:
        #     # Reshape scale from (E,) to (1, E, 1, 1) for broadcasting over (W, E, B, I)
        #     scale = self.module_output_scales[key_module].view(1, -1, 1, 1)
        #     next_value_scaled = next_value * scale
        # else:
        #     # Equation modules don't have scales
        #     next_value_scaled = next_value

        return next_value  # [W, E, B, I] - scaled masked update
    
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
    
    def compute_vectorized_sindy_loss(self):
        """Compute SINDy loss vectorized across all buffered timesteps.

        Called from __call__() after forward() completes. Processes the
        per-module buffers collected during call_module() and computes a
        single batched forward_sindy() + MSE per module.
        """
        n_modules = len(self.submodules_rnn)

        for module_name, buffers in self._sindy_buffers.items():
            if module_name not in self.sindy_coefficients:
                continue

            n_trials = len(buffers['h_current'])

            # Concatenate across trials: (T*W, E, B, I) and (T*W, E, B, I, n_controls)
            h_current = torch.cat(buffers['h_current'], dim=0)
            h_next_rnn = torch.cat(buffers['h_next_rnn'], dim=0)
            controls = torch.cat(buffers['controls'], dim=0)

            # Build action mask: (T*W, E, B, I)
            W_per_trial = buffers['h_current'][0].shape[0]
            masks = torch.stack(buffers['action_mask'], dim=0)  # (T, E, B, I)
            if W_per_trial > 1:
                masks = masks.unsqueeze(1).expand(-1, W_per_trial, -1, -1, -1)
                masks = masks.reshape(-1, *masks.shape[2:])  # (T*W, E, B, I)

            # Single forward_sindy call for all timesteps
            h_next_sindy = self.forward_sindy(
                h_current=h_current,
                key_module=module_name,
                participant_ids=buffers['participant_ids'],
                experiment_ids=buffers['experiment_ids'],
                controls=controls,
                polynomial_degree=self.sindy_polynomial_degree,
            )  # (T*W, E, B, I)

            # Masked Huber loss — MSE for small errors, L1 for large errors
            # Caps gradient at delta, preventing explosion from large RNN-SINDy divergence
            diff = torch.nn.functional.huber_loss(h_next_rnn, h_next_sindy, reduction='none', delta=1.0)
            masked_diff = torch.where(masks == 1, diff, torch.zeros_like(diff))
            n_masked = masks.sum(dim=-1).clamp(min=1)  # (T*W, E, B)
            per_sample_loss = masked_diff.sum(dim=-1) / n_masked  # (T*W, E, B)

            # mean() gives mean over (T*W, E, B); multiply by n_trials to match
            # the original sum-of-means: sum_{t} mean_{W,E,B}(loss_t) = T * mean_{T*W,E,B}
            sindy_loss = torch.clamp(per_sample_loss.mean() * n_trials / n_modules, max=100.0)

            self.sindy_loss = self.sindy_loss + sindy_loss
    
    def sindy_ridge_solve(self, key_module: str, participant_ids: torch.Tensor, experiment_ids: torch.Tensor, h_next: torch.Tensor, h_current: torch.Tensor, controls: torch.Tensor):
        W, E, B, I = h_next.shape
        P = self.n_participants
        X = self.n_experiments
        T = self.sindy_coefficients[key_module].shape[-1]

        if h_current is None:
            h_current = torch.zeros_like(h_next)
        
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
        penalty_diag += 1e-4 * torch.eye(T, device=library.device, dtype=library.dtype)
        AtA_accum = AtA_accum + penalty_diag  # broadcasts over (E, P, X)

        # Only solve for groups that have data; keep existing coefficients for empty groups
        if has_data.all():
            coefficients = torch.linalg.solve(AtA_accum, Atb_accum).squeeze(-1)
            self.sindy_coefficients[key_module].data.copy_(coefficients)
        else:
            coefficients = torch.linalg.solve(AtA_accum[has_data], Atb_accum[has_data]).squeeze(-1)
            self.sindy_coefficients[key_module].data[has_data] = coefficients
        
    def sindy_coefficient_pruning(self, patience: int = 1, n_terms_pruning: int = None):
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
                keep_mask = ~pruning_mask[..., start_idx:start_idx + n_terms]
                # Update the permanent mask
                self.sindy_coefficients_presence[module] &= keep_mask
                # Zero out the coefficients
                self.sindy_coefficients[module] *= keep_mask.float()
                # Reset patience counters for thresholded coefficients
                self.sindy_pruning_patience_counters[module] *= keep_mask.int()

                start_idx += n_terms
                
    def sindy_coefficient_patience(self, threshold: float):
        
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
        
        # Split back to modules and update patience counters for each module
        start_idx = 0
        for module in module_list:
            n_terms = self.sindy_coefficients[module].shape[-1]
            self.sindy_pruning_patience_counters[module] = all_patience[..., start_idx:start_idx + n_terms]
            start_idx += n_terms
    def count_sindy_coefficients(self) -> torch.Tensor:
        """Returns count of active coefficients per (ensemble, participant, experiment)."""
        coefficients = torch.zeros(self.n_participants, self.n_experiments, device=self.device)
        sindy_coefs = self.get_sindy_coefficients(aggregate=True)
        for module in self.submodules_rnn:
            coefficients += (sindy_coefs[module] != 0).sum(dim=-1)
            if self.sindy_specs[module]['include_state']:
                index_state = 1 if self.sindy_specs[module]['include_bias'] else 0
                index_state_not_in_model = torch.where(torch.logical_and(sindy_coefs[module][..., index_state] < -0.95, sindy_coefs[module][..., index_state] > -1.05))[0]
                coefficients[index_state_not_in_model] -= 1
        return coefficients

    def compute_weighted_coefficient_penalty(self, sindy_alpha: float, norm: int = 1) -> torch.Tensor:
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

        penalty = torch.tensor(0.0, device=self.device)
        
        if sindy_alpha == 0:
            return penalty

        for module_name in self.submodules_rnn:
            if module_name not in self.sindy_coefficients:
                continue
                
            # Get coefficients: [n_participants, n_ensemble, n_library_terms]
            coeffs = self.sindy_coefficients[module_name]

            # Get degree weights: [n_library_terms]
            degree_weights = self.sindy_degree_weights[module_name].clone()
            # -------------------------------------------------------
            # TODO: REMOVE IF NOT HELPING TO RECOVER ASYM LEARNING!!!
            # -------------------------------------------------------
            degree_weights = torch.ones_like(degree_weights)
            
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

        return penalty * sindy_alpha
                    
    def get_spice_model_string(self, participant_id: int = 0, experiment_id: int = 0) -> str:
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
        coefs_dict = self.get_sindy_coefficients(aggregate=True)
        for module in self.submodules_rnn:
            sparse_coefs = coefs_dict[module][participant_id, experiment_id].detach().cpu().numpy()
            space_filler = " "+" "*(max_len_module-len(module)) if max_len_module > len(module) else " "
            equation_str = module + "[t+1]" + space_filler + "= "
            for index_term, term in enumerate(self.sindy_candidate_terms[module]):
                if term == module:
                    sparse_coefs[index_term] += 1
                if np.abs(sparse_coefs[index_term]) != 0:
                    if equation_str[-3:] != " = ":
                        equation_str += "+ "
                    equation_str += str(np.round(sparse_coefs[index_term], 3)) + " " + term
                    equation_str += "[t] " if term == module else " "
            if equation_str[-3:] == " = ":
                equation_str += "0"
            lines.append(equation_str)
            
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
    
    def get_sindy_coefficients(self, key_module: Optional[str] = None, aggregate: bool = False):
        if key_module is None:
            key_module = self.get_modules()  
        elif isinstance(key_module, str):
            key_module = [key_module]

        sindy_coefficients = {}
        for module in key_module:
            coeffs = self.sindy_coefficients[module].detach()  # (E, P, X, T)
            presence = self.sindy_coefficients_presence[module].float()  # (E, P, X, T)

            # Masked median: only aggregate over ensemble members where term is active
            masked_coeffs = coeffs * presence  # zero out inactive
            
            # Average across ensemble members without respecting 0
            if aggregate:
                masked_coeffs[presence == 0] = float('nan')  # mark inactive as NaN so median ignores them
                aggregated = torch.nanmean(masked_coeffs, dim=0)  # (E, P, X, C) -> (P, X, C)
                sindy_coefficients[module] = torch.nan_to_num(aggregated, nan=0.0)  # terms inactive in all members -> 0
            else:
                sindy_coefficients[module] = masked_coeffs
            
            # Presence: term is active if majority of ensemble members have it
            # aggregated_presence = (presence.sum(dim=0) > (self.ensemble_size / 2)).float()
            # sindy_coefficients[module] = (aggregated * aggregated_presence).cpu().numpy()

            # Add implicit +1 for the identity term (h_next = h_current + library @ coeffs)
            # identity_idx = self.sindy_candidate_terms[module].index(module)
            # sindy_coefficients[module][..., identity_idx] += 1
            
        return sindy_coefficients
    
    def eval(self, use_sindy=True, aggregate=True):
        super().eval()
        self.use_sindy = use_sindy
        self.aggregate = aggregate
        return self
        
    def train(self, mode=True, use_sindy=False):
        super().train(mode)
        # if training mode activate (mode=True) -> do not use sindy for forward pass (self.use_sindy=False)
        self.use_sindy = use_sindy
        self.aggregate = False
        return self
    
    def __call__(self, *args, **kwargs):
        self._sindy_buffers = {}
        self._hypernet_offsets = {}
        logits, state = super().__call__(*args, **kwargs)
        if self._sindy_buffers:
            self.compute_vectorized_sindy_loss()
        if self.aggregate:
            dim_ensemble = 0 if self.batch_first else 2
            logits = logits.nanmean(dim=dim_ensemble)
        return logits, state
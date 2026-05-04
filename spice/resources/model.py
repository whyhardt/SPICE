import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import numpy as np

from .sindy_differentiable import compute_library_size, compute_polynomial_library, get_library_feature_names, get_library_term_degrees, build_library_structure
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
    """Disentangled polynomial projection: sum of per-degree multilinear products.

    Each degree d has its own independent set of d linear projections (without biases).
    The output is the sum across all degrees:

        y = Σ_{d=1}^{D} Π_{k=1}^{d} (W_{d,k} @ x)

    This keeps degrees disentangled — degree-d terms come exclusively from the
    degree-d group, with no cross-degree contamination. The constant (degree-0)
    term is handled externally by the output bias in ``EnsembleRNNModule``.

    Parameters per degree d: d weight matrices of shape (ensemble_size, output_size, input_size)
    Total weights: D*(D+1)/2 matrices.
    Input:  (E, B, F)
    Output: (E, B, O)
    """
    def __init__(self, ensemble_size: int, input_size: int, output_size: int, degree: int = 2):
        super().__init__()
        assert degree >= 1, f"degree must be >= 1, got {degree}"
        self.degree = degree
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size

        # degree_groups[d_idx] = ParameterList of (d_idx+1) weight matrices for degree (d_idx+1)
        # No biases inside forms — each degree-d product produces exactly degree-d monomials.
        base_gain = 0.01
        self.degree_groups = nn.ModuleList()
        for d in range(1, degree + 1):
            gain_d = base_gain ** (1.0 / d)
            group = nn.ParameterList([
                nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
                for _ in range(d)
            ])
            for w in group:
                nn.init.xavier_normal_(w, gain=gain_d)
            self.degree_groups.append(group)

    def forward(self, x, weight_offsets=None):
        """Forward pass through disentangled polynomial projection.

        Args:
            x: (E, B, F) input tensor
            weight_offsets: Optional nested list of per-batch weight offsets.
                Structure: weight_offsets[d_idx][k] has shape (E, B, O, I) for
                degree group d_idx, form k within that group.
        """
        E, B, F = x.shape
        parts = []

        for d_idx, group in enumerate(self.degree_groups):
            if weight_offsets is None:
                # Group-level: shared weights across batch
                product = torch.einsum('eoi,ebi->ebo', group[0], x)
                for w in group[1:]:
                    product = product * torch.einsum('eoi,ebi->ebo', w, x)
            else:
                # Per-participant: group weights + offsets
                w_eff = group[0].unsqueeze(1) + weight_offsets[d_idx][0]
                product = torch.einsum('eboi,ebi->ebo', w_eff, x)
                for k in range(1, len(group)):
                    w_eff = group[k].unsqueeze(1) + weight_offsets[d_idx][k]
                    product = product * torch.einsum('eboi,ebi->ebo', w_eff, x)
            parts.append(product)

        return torch.cat(parts, dim=-1)  # (E, B, D * output_size)


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

        n_features = input_size + 1  # +1 for hidden state
        hidden_per_degree = max(n_features, 2) * 2
        proj_size = polynomial_degree * hidden_per_degree
        hidden_size = 1

        # Disentangled polynomial projection: sum of per-degree multilinear products
        self.projection = EnsemblePolynomialLayer(
            ensemble_size=ensemble_size,
            input_size=n_features,
            output_size=hidden_per_degree,
            degree=polynomial_degree,
        )

        # Output projection (combines all degree groups via concatenated hidden dims)
        self.weight_n = nn.Parameter(torch.empty(ensemble_size, 1, proj_size))
        self.bias_n = nn.Parameter(torch.zeros(ensemble_size, 1))
        nn.init.xavier_uniform_(self.weight_n.view(ensemble_size, 1, proj_size))

        # Linear damping (scalar, shared across ensemble and participants)
        self.scale_state = nn.Parameter(torch.tensor(-3.0))

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.hidden_per_degree = hidden_per_degree
        self.ensemble_size = ensemble_size

        # Hypernetwork: embedding -> per-participant weight offsets
        # Two-layer MLP with GELU non-linearity. The final layer is zero-initialized
        # so the model starts as a pure group model (zero offsets).
        if embedding_size > 0:
            # Compute split sizes: one entry per weight matrix across all degree groups
            self._offset_split_sizes = []
            for group in self.projection.degree_groups:
                for w in group:
                    self._offset_split_sizes.append(w.shape[1] * w.shape[2])  # O * I_in
            self._offset_split_sizes.append(self.weight_n.shape[1] * self.weight_n.shape[2])  # 1 * proj_size
            self._offset_split_sizes.append(self.bias_n.shape[1])  # 1

            n_total_params = sum(self._offset_split_sizes)
            hypernet_hidden = embedding_size
            self.hypernet_in = EnsembleLinear(ensemble_size, embedding_size, hypernet_hidden)
            self.hypernet_out = EnsembleLinear(ensemble_size, hypernet_hidden, n_total_params)
            # Zero-init output layer so offsets start at zero
            nn.init.zeros_(self.hypernet_out.weight)
            nn.init.zeros_(self.hypernet_out.bias)

        # Precompute library structure for polynomial unfolding
        lib = build_library_structure(n_features, polynomial_degree)
        self._library_terms = lib['terms']
        self._n_library_terms = lib['n_terms']
        self._bias_index = lib['bias_index']
        self._degree_ranges = lib['degree_ranges']
        self.register_buffer('_mult_table', lib['mult_table'])
        self.register_buffer('_linear_indices', lib['linear_indices'])

        if compiled_forward:
            self._compiled_forward = torch.compile(self._forward_impl, dynamic=True)

    @property
    def alpha(self):
        """State damping factor for gated polynomial update.

        Controls how much of the previous state is retained:
        ``h[t+1] = (1-α)*h[t] + α_n * n``
        """
        return torch.sigmoid(self.scale_state)

    @property
    def alpha_n(self):
        """Candidate scaling factor for gated polynomial update.

        Controls how much the polynomial candidate contributes:
        ``h[t+1] = (1-α)*h[t] + α_n * n``
        """
        return 1.0  # torch.sigmoid(self.scale_candidate) for learnable scaling

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
            proj_w_offsets: nested list matching degree_groups structure —
                proj_w_offsets[d_idx][k] has shape (E, B, O, I_in)
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

        # Projection weight offsets organized by degree group
        idx = 0
        proj_w_offsets = []
        for group in self.projection.degree_groups:
            group_offsets = []
            for w in group:
                O, I_in = w.shape[1], w.shape[2]
                offset = self._constrain_offset(parts[idx].reshape(E, B, O, I_in), w, self.max_offset_ratio)
                group_offsets.append(offset)
                idx += 1
            proj_w_offsets.append(group_offsets)

        # Output weight offset
        out_w_offset = self._constrain_offset(
            parts[idx].reshape(E, B, self.weight_n.shape[1], self.weight_n.shape[2]),
            self.weight_n, self.max_offset_ratio,
        )
        idx += 1

        # Output bias offset
        out_b_offset = self._constrain_offset(parts[idx], self.bias_n, self.max_offset_ratio)

        return proj_w_offsets, out_w_offset, out_b_offset

    def precompute_offsets(self, embedding, n_items):
        """Pre-compute per-participant weight offsets from embedding.

        Call once per forward pass; results are reused across trials.

        Args:
            embedding: (E, B, emb_dim)
            n_items: number of items (I) for expansion

        Returns:
            Tuple of (proj_w_offsets, out_w_offset, out_b_offset)
        """
        # Expand embedding for items: (E, B, D) -> (E, B*I, D)
        emb_expanded = embedding.unsqueeze(2).expand(-1, -1, n_items, -1).reshape(
            embedding.shape[0], embedding.shape[1] * n_items, embedding.shape[2]
        )
        return self._generate_weight_offsets(emb_expanded)

    def _get_effective_params(self, individual_offsets=None):
        """Extract effective (group + offset) parameters for polynomial unfolding.

        Returns:
            W_groups: list of lists — W_groups[d_idx][k] is the effective weight
                tensor for degree group d_idx, form k. Shape (..., O, I_in).
            w_out: (..., proj_size) output projection weights (hidden_size dim squeezed)
            b_out: (...,) output projection bias (hidden_size dim squeezed)
        """
        if individual_offsets is not None:
            proj_w_offsets, out_w_offset, out_b_offset = individual_offsets
            W_groups = []
            for d_idx, group in enumerate(self.projection.degree_groups):
                W_groups.append([
                    w.unsqueeze(1) + proj_w_offsets[d_idx][k]
                    for k, w in enumerate(group)
                ])
            w_out = (self.weight_n.unsqueeze(1) + out_w_offset).squeeze(-2)
            b_out = (self.bias_n.unsqueeze(1) + out_b_offset).squeeze(-1)
        else:
            W_groups = [list(group) for group in self.projection.degree_groups]
            w_out = self.weight_n.squeeze(-2)
            b_out = self.bias_n.squeeze(-1)
        return W_groups, w_out, b_out

    def unfold_polynomial_coefficients(self, individual_offsets=None):
        """Unfold weight matrices into polynomial coefficients via per-degree recursive expansion.

        For each degree d, expands the product of d linear projections (without biases)
        ``Π_{k=1}^{d} (W_{d,k} @ x)`` into degree-d monomial coefficients, then
        contracts with the corresponding segment of the output projection ``w_out``.

        Since degrees are disentangled, each degree group produces coefficients
        for exactly its own degree — no cross-degree contamination.

        Works for arbitrary polynomial degree. Fully differentiable.

        Args:
            individual_offsets: Hypernetwork offsets tuple or None.
                Without offsets: returns group-level coefficients (B, n_terms).
                With offsets: returns per-participant coefficients (E, B, n_terms).

        Returns:
            theta: Polynomial coefficients in monomial basis.
        """
        W_groups, w_out, b_out = self._get_effective_params(individual_offsets)
        degree = self.projection.degree
        n_terms = self._n_library_terms
        n_features = self._mult_table.shape[1]
        hidden_per_degree = self.hidden_per_degree

        # Determine leading shape: (E,) for group or (E, B*I) for per-participant
        ref_w = W_groups[0][0]
        leading_shape = ref_w.shape[:-2]  # everything before (O, I_in)
        device = ref_w.device
        dtype = ref_w.dtype

        # Accumulate coefficients across all degree groups
        theta = torch.zeros(*leading_shape, n_terms, device=device, dtype=dtype)

        w_out_offset = 0
        for d_idx, W_list_d in enumerate(W_groups):
            d = d_idx + 1  # degree = 1, 2, 3, ...

            # Initialize with first linear form: only degree-1 (linear) terms
            # coeffs shape: (..., O, n_terms) where O = hidden_per_degree
            coeffs = torch.zeros(*leading_shape, hidden_per_degree, n_terms, device=device, dtype=dtype)
            coeffs[..., self._linear_indices] = W_list_d[0]  # (..., O, n_features)

            # Recursively multiply by remaining forms (k=1 to d-1)
            # No bias propagation — only feature multiplication
            for k in range(1, d):
                new_coeffs = torch.zeros_like(coeffs)
                for f in range(n_features):
                    targets = self._mult_table[:, f]
                    valid = targets >= 0
                    src_idx = torch.where(valid)[0]
                    tgt_idx = targets[src_idx]
                    new_coeffs[..., tgt_idx] = new_coeffs[..., tgt_idx] + coeffs[..., src_idx] * W_list_d[k][..., f:f+1]
                coeffs = new_coeffs

            # Contract with the w_out segment for this degree group
            w_out_segment = w_out[..., w_out_offset:w_out_offset + hidden_per_degree]  # (..., O)
            theta = theta + torch.einsum('...o,...ot->...t', w_out_segment, coeffs)
            w_out_offset += hidden_per_degree

        # Add output bias as the constant (degree-0) term
        theta[..., self._bias_index] = theta[..., self._bias_index] + b_out

        return theta

    def _compute_library(self, features):
        """Compute polynomial library values for all monomial terms.

        Args:
            features: (..., n_features) tensor of [state, controls]

        Returns:
            library: (..., n_terms) tensor of monomial values
        """
        n_terms = self._n_library_terms
        library = torch.ones(*features.shape[:-1], n_terms, device=features.device, dtype=features.dtype)

        # Linear terms: direct gather
        library[..., self._linear_indices] = features

        # Higher-degree terms: products of features
        for t_idx, term in enumerate(self._library_terms):
            if len(term) >= 2:
                val = features[..., term[0]]
                for f_idx in term[1:]:
                    val = val * features[..., f_idx]
                library[..., t_idx] = val

        return library

    def _forward_impl(self, inputs, state, individual_offsets=None, mask=None):
        """Forward pass using unfolded polynomial coefficients with sparsity mask.

        Computes the same function as the standard forward but via the
        monomial-basis representation, enabling direct coefficient masking.

        Args:
            inputs: (W, E, B, I, F) control signals
            state: (W, E, B, I) previous hidden state
            individual_offsets: Hypernetwork offsets tuple or None
            mask: (E, n_terms) or (E, B*I, n_terms) sparsity mask, or None

        Returns:
            output: (W, E, B, I, 1) updated states
        """
        W, E, B, I, F = inputs.shape

        x = inputs.reshape(W, E, B * I, F)
        h = state[-1].contiguous().reshape(E, B * I, 1) if state is not None else torch.zeros(E, B * I, 1, device=inputs.device)

        # Unfold RNN weights into polynomial coefficients (constant across timesteps)
        theta = self.unfold_polynomial_coefficients(individual_offsets)

        if mask is not None:
            if theta.dim() < mask.dim():
                theta = theta.unsqueeze(1)  # (E, 1, n_terms) broadcasts with (E, B*I, n_terms)
            theta = theta * mask.float()

        # Gated update: h = (1-α)*h + α_n*n
        alpha = self.alpha
        alpha_n = self.alpha_n

        outputs = []
        for t in range(W):
            # Build features: [state, controls]
            features = torch.cat([h, x[t]], dim=-1)  # (E, B*I, n_features)

            # Compute library values
            library = self._compute_library(features)  # (E, B*I, n_terms)

            # Compute update: n = library @ theta
            if theta.dim() == 2:
                # Group-level: theta is (E, n_terms)
                n = torch.einsum('ebt,et->eb', library, theta).unsqueeze(-1)
            else:
                # Per-participant: theta is (E, B*I, n_terms)
                n = (library * theta).sum(dim=-1, keepdim=True)

            h = (1 - alpha) * h + alpha_n * n
            outputs.append(h)

        output = torch.stack(outputs)  # (W, E, B*I, 1)
        return output.reshape(W, E, B, I, 1)

    def forward(self, inputs, state, precomputed_offsets=None, mask=None):
        """Forward pass using unfolded polynomial coefficients with sparsity mask.

        Uses the compiled variant when ``compiled_forward=True`` was set at
        construction time, otherwise falls back to the uncompiled path.
        """
        if self._compile:
            return self._compiled_forward(inputs, state, precomputed_offsets, mask)
        else:
            return self._forward_impl(inputs, state, precomputed_offsets, mask)

        
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

        sindy_polynomial_degree: int = 1,

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
        self.n_items = n_items if n_items is not None else n_actions
        self.ensemble_size = ensemble_size
        self.compiled_forward = compiled_forward

        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.module_output_scales = nn.ParameterDict()  # Learnable output scale per module (for logit computation)

        # Polynomial masks and pruning infrastructure
        self.sindy_polynomial_degree = sindy_polynomial_degree
        self.sindy_coefficients_presence = {}  # Binary masks to permanently zero out polynomial terms
        self.sindy_candidate_terms = {}        # Library term names for display
        self.sindy_pruning_patience_counters = {}  # Patience counters for thresholding
        self.sindy_specs = {}  # module-specific specs (include_bias, interaction_only, ...)

        # Setup initial values of RNN
        self.aggregate = False
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
        # Move masks and patience counters to the correct device
        for module_name in self.sindy_coefficients_presence:
            self.sindy_coefficients_presence[module_name] = self.sindy_coefficients_presence[module_name].to(device)
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
        self.setup_polynomial_masks(key_module=key_module, polynomial_degree=polynomial_degree)

        # Input names: state + control signals (matches library structure)
        input_names = [key_module] + list(self.spice_config.library_setup[key_module])
        self.sindy_specs[key_module]['input_names'] = tuple(input_names)
        
    def _get_participant_mask(self, key_module: str, participant_ids: torch.Tensor, experiment_ids: torch.Tensor) -> torch.Tensor:
        """Index presence mask by participant/experiment IDs.

        Args:
            key_module: Module name
            participant_ids: (E, B) participant indices
            experiment_ids: (E, B) experiment indices

        Returns:
            mask: (E, B, n_terms) boolean mask for the given participants/experiments
        """
        E = participant_ids.shape[0]
        E_idx = torch.arange(E, device=participant_ids.device).unsqueeze(1)  # (E, 1)
        return self.sindy_coefficients_presence[key_module][E_idx, participant_ids, experiment_ids]  # (E, B, n_terms)

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
        """Call a submodule (RNN or polynomial equation) to compute the next state value.

        When ``use_sindy=False``: standard RNN forward pass.
        When ``use_sindy=True``: unfolds the RNN weights into polynomial
        coefficients, applies the sparsity mask, and runs the polynomial forward.

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

        # Ensure inputs is 5D: (W, E, B, I, F)
        # Models may index rewards[t, 0] which consumes the W dimension → add it back
        while inputs.dim() < 5:
            inputs = inputs.unsqueeze(0)

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
            rnn = self.submodules_rnn[key_module]
            offsets = self._hypernet_offsets[key_module]

            # Polynomial forward path with sparsity mask
            mask = self._get_participant_mask(key_module, participant_index, experiment_index)  # (E, B, n_terms)
            if I > 1:
                mask = mask.unsqueeze(2).expand(-1, -1, I, -1).reshape(E, B * I, -1)
            next_value = rnn.forward(inputs, state=value, precomputed_offsets=offsets, mask=mask).squeeze(-1)

        elif key_module in self.submodules_eq.keys():
            # hard-coded equation — operates on last within-trial step
            next_value = self.submodules_eq[key_module](value, inputs[-1]).unsqueeze(0)  # [1, E, B, I]

        else:
            raise ValueError(f'Invalid module key {key_module}.')

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

        if key_state is not None:
            self.state[key_state] = next_state

        return next_value  # [W, E, B, I]
    
    def setup_polynomial_masks(self, key_module: str, polynomial_degree: int = None):
        """Initialize polynomial masks and pruning infrastructure for a module.

        Uses the full (unfiltered) polynomial library — term exclusions
        (include_bias, include_state, interaction_only) are encoded in the
        initial presence mask rather than by removing terms from the candidate list.

        The mask shape is ``(E, P, X, n_library_terms)`` matching the unfolded
        coefficient shape from ``EnsembleRNNModule.unfold_polynomial_coefficients()``.
        """

        if polynomial_degree is None:
            polynomial_degree = self.sindy_polynomial_degree

        control_features = self.spice_config.library_setup[key_module]
        sindy_specs = self.sindy_specs[key_module]

        # Full library: state + controls (matches EnsembleRNNModule's library structure)
        feature_names = tuple([key_module]) + control_features
        candidate_terms = get_library_feature_names(feature_names, polynomial_degree)
        n_library_terms = len(candidate_terms)

        self.sindy_candidate_terms[key_module] = candidate_terms

        # Initialize presence mask — all True, then apply exclusions
        mask = torch.ones(
            self.ensemble_size, self.n_participants, self.n_experiments, n_library_terms,
            dtype=torch.bool, device=self.device
        )

        # Mask out excluded terms (instead of removing them from the candidate list)
        for i, term in enumerate(candidate_terms):
            if not sindy_specs['include_bias'] and term == '1':
                mask[..., i] = False
            if not sindy_specs['include_state'] and key_module in term:
                mask[..., i] = False
            if sindy_specs['interaction_only'] and '^' in term and '*' not in term:
                mask[..., i] = False

        self.sindy_coefficients_presence[key_module] = mask

        # Initialize patience counters to zero
        self.sindy_pruning_patience_counters[key_module] = torch.zeros(
            self.ensemble_size, self.n_participants, self.n_experiments, n_library_terms,
            dtype=torch.int32, device=self.device
        )
    
        
    def _get_unfolded_coefficients_for_pruning(self) -> Dict[str, torch.Tensor]:
        """Unfold effective polynomial coefficients for all modules, shaped ``(E, P, X, T)``.

        Returns coefficients scaled by α_n so pruning operates on effective magnitudes.
        The state self-term deliberately excludes ``(1-α)`` — pruning targets only the
        learnable component ``α_n * θ``, not the fixed damping contribution.
        When a hypernetwork is active, unfolds per-participant coefficients so pruning
        sees the actual per-participant magnitudes (averaged over items).
        """
        result = {}
        with torch.no_grad():
            for module in self.submodules_rnn:
                rnn = self.submodules_rnn[module]
                if rnn.embedding_size > 0 and hasattr(self, 'participant_embedding'):
                    # Per-participant unfolding via hypernetwork
                    P = self.n_participants
                    X = self.n_experiments
                    E = self.ensemble_size
                    coeffs_list = []
                    for p_id in range(P):
                        p_idx = torch.full((E, 1), p_id, dtype=torch.int, device=self.device)
                        emb = self.participant_embedding(p_idx)  # (E, 1, emb_dim)
                        offsets = rnn.precompute_offsets(emb, self.n_items)
                        theta = rnn.unfold_polynomial_coefficients(offsets)  # (E, 1*I, n_terms)
                        # Average over items → (E, 1, n_terms)
                        theta = theta.reshape(E, 1, self.n_items, -1).mean(dim=2)
                        coeffs_list.append(theta)
                    # Stack → (E, P, n_terms), expand to (E, P, X, n_terms)
                    theta = torch.cat(coeffs_list, dim=1).unsqueeze(2).expand(-1, -1, X, -1)
                else:
                    # Group-level unfolding, replicate across P and X
                    theta = rnn.unfold_polynomial_coefficients()  # (E, n_terms)
                    theta = theta.unsqueeze(1).unsqueeze(2).expand(
                        -1, self.n_participants, self.n_experiments, -1
                    )
                result[module] = (theta * rnn.alpha_n).clone()
        return result

    def update_pruning_patience(self, failed: dict):
        """Update patience counters: +1 where failed, reset to 0 where passed.

        Args:
            failed: ``{module: (P, X, terms) bool}`` — True where the test failed.
        """
        with torch.no_grad():
            for module in self.submodules_rnn:
                counters = self.sindy_pruning_patience_counters[module]
                failed_e = failed[module].unsqueeze(0).expand_as(counters)
                counters.data = torch.where(failed_e, counters + 1, torch.zeros_like(counters))

    def prune_by_patience(self, patience_threshold: int, n_terms: int = None):
        """Prune terms where patience counter >= threshold.

        When ``n_terms`` is set, only the ``n_terms`` smallest-magnitude
        candidates (across all modules) are pruned per event, preventing
        overly disruptive pruning steps.

        Args:
            patience_threshold: minimum consecutive failures before pruning.
            n_terms: max terms to prune per event (``None`` = prune all eligible).
        """
        with torch.no_grad():
            # Collect candidates across all modules
            unfolded = self._get_unfolded_coefficients_for_pruning()
            module_list = list(self.submodules_rnn.keys())

            all_counters = torch.cat([self.sindy_pruning_patience_counters[m] for m in module_list], dim=-1)
            all_coeffs = torch.cat([unfolded[m].abs() for m in module_list], dim=-1)

            eligible = all_counters[0] >= patience_threshold  # (P, X, total_terms)
            if not eligible.any():
                return

            if n_terms is not None:
                # Rank by magnitude (mean across E), prune smallest n_terms
                mean_coeffs = all_coeffs.mean(dim=0)  # (P, X, total_terms)
                mean_coeffs[~eligible] = torch.inf
                n_to_prune = min(n_terms, eligible.sum().item())
                if n_to_prune == 0:
                    return
                _, indices = torch.topk(mean_coeffs.reshape(-1), n_to_prune, largest=False)
                prune_flat = torch.zeros(mean_coeffs.numel(), dtype=torch.bool, device=mean_coeffs.device)
                prune_flat[indices] = True
                prune = prune_flat.reshape(mean_coeffs.shape) & eligible
            else:
                prune = eligible

            # Apply pruning per module
            start_idx = 0
            for module in module_list:
                n_mod_terms = self.sindy_coefficients_presence[module].shape[-1]
                prune_mod = prune[..., start_idx:start_idx + n_mod_terms]
                if prune_mod.any():
                    prune_e = prune_mod.unsqueeze(0).expand_as(self.sindy_pruning_patience_counters[module])
                    self.sindy_coefficients_presence[module] &= ~prune_e
                    self.sindy_pruning_patience_counters[module].data *= (~prune_e).int()
                start_idx += n_mod_terms
    def count_sindy_coefficients(self) -> torch.Tensor:
        """Returns count of active (non-zero) coefficients per (participant, experiment)."""
        coefficients = torch.zeros(self.n_participants, self.n_experiments, device=self.device)
        sindy_coefs = self.get_sindy_coefficients(aggregate=True)
        for module in self.submodules_rnn:
            coefficients += (sindy_coefs[module] != 0).sum(dim=-1)
        return coefficients

    def get_spice_model_string(self, participant_id: int = 0, experiment_id: int = 0) -> str:
        """Get the learned SPICE features and equations as a string.

        Displays effective polynomial coefficients with the damping gate absorbed:
        ``h[t+1] = c_h*h[t] + c_1*term_1 + c_2*term_2 + ...``

        Args:
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
            space_filler = " " + " " * (max_len_module - len(module)) if max_len_module > len(module) else " "
            equation_str = module + "[t+1]" + space_filler + "= "
            has_terms = False
            for index_term, term in enumerate(self.sindy_candidate_terms[module]):
                coef = sparse_coefs[index_term]
                if np.abs(coef) != 0:
                    if has_terms:
                        equation_str += " + " if coef >= 0 else " - "
                        coef_display = np.abs(coef)
                    else:
                        equation_str += "-" if coef < 0 else ""
                        coef_display = np.abs(coef)
                    term_name = term + "[t]" if term == module else term
                    if term == '1':
                        equation_str += str(np.round(coef_display, 3))
                    else:
                        equation_str += str(np.round(coef_display, 3)) + "*" + term_name
                    has_terms = True
            if not has_terms:
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
        """Return effective polynomial coefficients for each module.

        Effective coefficients absorb the damping gate:
        - All terms: ``θ_eff = α_n * θ`` where ``α_n = sigmoid(damping)``
        - State self-term additionally gets ``+ (1-α)`` for the decay contribution

        Group-level (no hypernetwork): calls ``unfold()`` without offsets → ``(E, n_terms)``,
        then replicates to ``(E, P, X, n_terms)`` so downstream code sees the expected shape.

        Per-participant (hypernetwork): iterates participants and unfolds with their
        hypernetwork offsets → ``(E, P, X, n_terms)``.

        Args:
            key_module: Module name or None for all modules.
            aggregate: If True, average across ensemble members.

        Returns:
            Dict mapping module names to effective coefficient tensors.
        """
        if key_module is None:
            key_module = self.get_modules()
        elif isinstance(key_module, str):
            key_module = [key_module]

        sindy_coefficients = {}
        for module in key_module:
            rnn = self.submodules_rnn[module]
            presence = self.sindy_coefficients_presence[module].float()  # (E, P, X, T)

            # Compute gate values (scalar, shared across ensemble)
            alpha = rnn.alpha.item()
            alpha_n = rnn.alpha_n
            state_linear_idx = rnn._linear_indices[0].item()  # state is feature 0

            with torch.no_grad():
                if rnn.embedding_size > 0 and hasattr(self, 'participant_embedding'):
                    # Per-participant unfolding via hypernetwork
                    P = self.n_participants
                    X = self.n_experiments
                    E = self.ensemble_size
                    coeffs_list = []
                    for p_id in range(P):
                        p_idx = torch.full((E, 1), p_id, dtype=torch.int, device=self.device)
                        emb = self.participant_embedding(p_idx)  # (E, 1, emb_dim)
                        offsets = rnn.precompute_offsets(emb, self.n_items)
                        theta = rnn.unfold_polynomial_coefficients(offsets)  # (E, 1*I, n_terms)
                        # Average over items → (E, 1, n_terms)
                        theta = theta.reshape(E, 1, self.n_items, -1).mean(dim=2)
                        coeffs_list.append(theta)
                    # Stack → (E, P, n_terms), expand to (E, P, X, n_terms)
                    coeffs = torch.cat(coeffs_list, dim=1).unsqueeze(2).expand(-1, -1, X, -1)
                else:
                    # Group-level unfolding
                    theta = rnn.unfold_polynomial_coefficients()  # (E, n_terms)
                    coeffs = theta.unsqueeze(1).unsqueeze(2).expand(-1, self.n_participants, self.n_experiments, -1)

            # Apply presence mask, then compute effective coefficients
            masked_coeffs = coeffs * presence
            masked_coeffs = masked_coeffs * alpha_n
            # Add (1-α) to state self-term (always present regardless of mask)
            masked_coeffs[..., state_linear_idx] = masked_coeffs[..., state_linear_idx] + (1 - alpha)

            if aggregate:
                masked_coeffs_agg = masked_coeffs.clone()
                # For aggregation, mark pruned terms as NaN (but preserve state term's (1-α))
                non_state_mask = torch.ones(masked_coeffs.shape[-1], dtype=torch.bool)
                non_state_mask[state_linear_idx] = False
                masked_coeffs_agg[..., non_state_mask] = torch.where(
                    presence[..., non_state_mask] == 0,
                    torch.tensor(float('nan'), device=masked_coeffs.device),
                    masked_coeffs_agg[..., non_state_mask],
                )
                aggregated = torch.nanmean(masked_coeffs_agg, dim=0)  # (P, X, T)
                sindy_coefficients[module] = torch.nan_to_num(aggregated, nan=0.0)
            else:
                sindy_coefficients[module] = masked_coeffs

        return sindy_coefficients
    
    def eval(self, aggregate=True):
        super().eval()
        self.aggregate = aggregate
        return self

    def train(self, mode=True):
        super().train(mode)
        self.aggregate = False
        return self
    
    def __call__(self, *args, **kwargs):
        self._hypernet_offsets = {}
        logits, state = super().__call__(*args, **kwargs)
        if self.aggregate:
            dim_ensemble = 0 if self.batch_first else 2
            logits = logits.nanmean(dim=dim_ensemble)
        return logits, state
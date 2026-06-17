import torch

from spice import SpiceConfig, BaseModel


"""
Thoughts about missing hypotheses

- Disentangling 
- Action stability via tile idling
- Prediction of partners next position; Could be achieved by:
    - Partner's movement perseverance
    - Tracking which tiles were last visite by partner with time discounting and mark as "partner's responsibility"
- Mark responsibilities: Last visited by self/partner
"""


CONFIG = SpiceConfig(
    library_setup={
        'visited_self': [
            'time_since_visited_by_self',
            'time_progress',
            'time_since_visited_by_partner',
            ],
        'not_visited_self': [
            'time_since_visited_by_self',
            'time_since_visited_by_partner',
            'time_progress',
            ],
        'visited_partner': [
            'time_since_visited_by_self',
            'time_since_visited_by_partner',
            'time_progress',
            ],
        'not_visited_partner': [
            'time_since_visited_by_self',
            'time_since_visited_by_partner',
            'time_progress',
            ],
        'movement_perseverance': [
            'alignment',
            'time_progress',
            ],
    },
    
    memory_state={
        'information_value_self': None,
        'information_value_partner': None,
        'time_since_visited_by_self': 1,
        'time_since_visited_by_partner': 1,
        'time_progress': 0,
        'alignment': None,
        'movement_value': None,
    },
    
    states_in_logit=['information_value_self', 'information_value_partner', 'movement_value'],
    additional_inputs=('partner_tile_index', 'time_point'),
)


class SpiceModel(BaseModel):
    """
    Custom SPICE model for modeling gaze behavior in collaborative memory task.

    Modules:
    - visited_self: update rule for tiles visited by self
    - visited_partner: update rule for tiles visited by partner
    - not_visited: update rule for unvisited tiles
    - perseverance: learned movement perseverance from single-step alignment

    Memory states:
    - information_value: attractiveness/value of each tile
    - time_since_visited_by_self / _by_partner: time since tile was last visited
    - time_progress: normalized time progress (0 -> 1)
    - alignment: per-tile dot product of last movement delta with direction-to-tile
    - perseverance: learned perseverance value, contributes additively to logit
    """

    def __init__(
        self, 
        spice_config, 
        n_actions: int, 
        n_participants: int, 
        grid_size: int = 4,
        time_max: float = 10,
        sindy_polynomial_degree = 2, 
        **kwargs,
        ):
        super().__init__(
            spice_config=spice_config,
            n_actions=n_actions, 
            n_participants=n_participants, 
            embedding_size=16,
            sindy_polynomial_degree=sindy_polynomial_degree,
            **kwargs,
            )
        
        self.grid_size = grid_size
        self.n_tiles = grid_size**2
        self.time_max = time_max
        dropout = 0.1
        
        # Tile coordinates (auto-moves with model.to(device))
        self.register_buffer('tile_rows', (torch.arange(self.n_tiles, device=self.device) // grid_size).float())
        self.register_buffer('tile_cols', (torch.arange(self.n_tiles, device=self.device) % grid_size).float())
        self.register_buffer('adjacency_matrix', self._build_adjacency_matrix())
        
        self.non_adjacency_offset = -4
        self.participant_embedding = self.setup_embedding(n_participants, self.embedding_size, dropout=dropout)
        
        self.setup_module(key_module='visited_self', input_size=3+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='not_visited_self', input_size=3+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='visited_partner', input_size=3+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='not_visited_partner', input_size=3+self.embedding_size, dropout=dropout)
        self.setup_module(key_module='movement_perseverance', input_size=2+self.embedding_size, dropout=dropout)
        
    def forward(self, inputs, prev_state=None):
        # Initialize inputs, outputs, and timesteps
        spice_signals = self.init_forward_pass(inputs, prev_state)
        
        # Get additional inputs
        # trial_progress = spice_signals.blocks.reshape(1, self.ensemble_size, -1, 1).repeat(1, 1, 1, self.n_actions) / spice_signals.blocks.max()
        tiles_visited_partner = spice_signals.additional_inputs['partner_tile_index'].squeeze(-1).long()
        time_point = spice_signals.additional_inputs['time_point'].squeeze(-1) / self.time_max
        
        # One-hot encode partner actions
        actions_partner = torch.nn.functional.one_hot(tiles_visited_partner, num_classes=self.n_actions)
        
        # Tiles visited by neither self nor partner
        actions_not_visited = torch.clamp(1 - (spice_signals.actions + actions_partner), 0, 1)
        
        # Get participant embeddings
        participant_embedding = self.participant_embedding(spice_signals.participant_ids)
        
        # Precompute alignment for all timesteps (no loop needed)
        alignment = self._precompute_alignment(spice_signals.actions)
        
        # All-tiles mask for the perseverance module
        all_tiles_mask = None
        prev_time = 0
                
        # Main loop: process each timestep in a block
        for timestep in spice_signals.trials:
            curr_tile = spice_signals.actions[timestep].argmax(dim=-1)
            
            # --- Update time-since-last-visit states ---
            dt = (time_point[timestep] - prev_time).unsqueeze(-1)
            self.state['time_since_visited_by_self'] = self.state['time_since_visited_by_self'] + dt
            self.state['time_since_visited_by_partner'] = self.state['time_since_visited_by_partner'] + dt
            
            # Update time_progress (same value across all tiles)
            time_progress = time_point[timestep].unsqueeze(-1).expand_as(self.state['information_value_self'])
            
            prev_time = time_point[timestep]
            
            # Update alignment from precomputed values
            self.state['alignment'] = alignment[timestep]
            
            # --- Call information value modules ---
            self.call_module(
                key_module='visited_self',
                key_state='information_value_self',
                action_mask=spice_signals.actions[timestep],
                inputs=(
                    self.state['time_since_visited_by_self'],
                    self.state['time_since_visited_by_partner'],
                    time_progress,
                    ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
            )
            
            self.call_module(
                key_module='not_visited_self',
                key_state='information_value_self',
                action_mask=1-spice_signals.actions[timestep],
                inputs=(
                    self.state['time_since_visited_by_self'],
                    self.state['time_since_visited_by_partner'],
                    time_progress,
                    ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
            )
            
            self.call_module(
                key_module='visited_partner',
                key_state='information_value_partner',
                action_mask=actions_partner[timestep],
                inputs=(
                    self.state['time_since_visited_by_self'],
                    self.state['time_since_visited_by_partner'],
                    time_progress,
                    ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
            )
            
            self.call_module(
                key_module='not_visited_partner',
                key_state='information_value_partner',
                action_mask=1-actions_partner[timestep],
                inputs=(
                    self.state['time_since_visited_by_self'],
                    self.state['time_since_visited_by_partner'],
                    time_progress,
                    ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
            )
            
            # --- Call perseverance module (updates all tiles every timestep) ---
            if all_tiles_mask is None:
                all_tiles_mask = torch.ones_like(spice_signals.actions[timestep])
            
            self.call_module(
                key_module='movement_perseverance',
                key_state='movement_value',
                action_mask=all_tiles_mask,
                inputs=(
                    self.state['alignment'],
                    time_progress,
                    ),
                participant_embedding=participant_embedding,
                participant_index=spice_signals.participant_ids,
            )
            
            # Reset time_since to 0 for tiles visited this timestep
            self.state['time_since_visited_by_self'] = self.state['time_since_visited_by_self'] * (1 - spice_signals.actions[timestep])
            self.state['time_since_visited_by_partner'] = self.state['time_since_visited_by_partner'] * (1 - actions_partner[timestep])
            
            # Apply adjacency mask; logit = information_value + perseverance
            adjacency_mask = self.adjacency_matrix[curr_tile]
            spice_signals.logits[timestep] = self.state['information_value_self'] + self.state['information_value_partner'] + self.state['movement_value']
            spice_signals.logits[timestep, ~adjacency_mask] = self.non_adjacency_offset
            
        # Post-process the forward pass
        spice_signals = self.post_forward_pass(spice_signals)

        return spice_signals.logits, self.get_state()
        
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """
        Build (n_tiles, n_tiles) boolean matrix where [i,j]=True means can move i->j.
        8-way adjacency (cardinal + diagonal) plus stay.
        
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15
        """
        adj = torch.zeros(self.n_tiles, self.n_tiles, dtype=torch.bool, device=self.device)
        
        for tile in range(self.n_tiles):
            row = tile // self.grid_size
            col = tile % self.grid_size
            
            adj[tile, tile] = True
            
            if row > 0:                                          # up
                adj[tile, tile - self.grid_size] = True
            if row < self.grid_size - 1:                          # down
                adj[tile, tile + self.grid_size] = True
            if col > 0:                                           # left
                adj[tile, tile - 1] = True
            if col < self.grid_size - 1:                          # right
                adj[tile, tile + 1] = True
            if row > 0 and col > 0:                               # left-up
                adj[tile, tile - self.grid_size - 1] = True
            if row < self.grid_size - 1 and col > 0:              # left-down
                adj[tile, tile + self.grid_size - 1] = True
            if row > 0 and col < self.grid_size - 1:              # right-up
                adj[tile, tile - self.grid_size + 1] = True
            if row < self.grid_size - 1 and col < self.grid_size - 1:  # right-down
                adj[tile, tile + self.grid_size + 1] = True
        
        return adj
    
    def _precompute_alignment(self, actions):
        """Precompute per-tile alignment with single-step movement delta for all timesteps.
        
        For each timestep t, computes the dot product of the movement direction
        (tile[t] - tile[t-1]) with the vector from tile[t] to every candidate tile.
        Positive = tile lies ahead in movement direction, negative = behind.
        
        Fully vectorized, no loop needed.
        
        Args:
            actions: (T, W, E, B, n_actions) one-hot action tensor
        Returns:
            alignment: (T, W, E, B, n_tiles)
        """
        curr_tiles = actions.argmax(dim=-1)  # (T, W, E, B)
        rows = (curr_tiles // self.grid_size).float()
        cols = (curr_tiles % self.grid_size).float()
        
        # Single-step movement delta
        dr = torch.zeros_like(rows)
        dc = torch.zeros_like(cols)
        dr[1:] = rows[1:] - rows[:-1]
        dc[1:] = cols[1:] - cols[:-1]
        
        # Direction from current position to each tile: (T, W, E, B, n_tiles)
        cand_dr = self.tile_rows - rows.unsqueeze(-1)
        cand_dc = self.tile_cols - cols.unsqueeze(-1)
        
        # Dot product: per-tile alignment with movement direction
        alignment = (dr.unsqueeze(-1) * cand_dr + dc.unsqueeze(-1) * cand_dc) / self.grid_size
        
        return alignment
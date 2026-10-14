import torch


class InformationForagingModel(torch.nn.Module):
    """
    A model for visuospatial information foraging in a collaborative memory tile game between participant A (self) and participant B (partner).
    Each tile are assigned values given the listed hypotheses. 
    The tile values are then passed through a softmax function to compute action probabilities for the next action (next selected tile by self). 
    
    Hypotheses:
    1. Information loss:
    -> Increasing the value of a tile for each timestep a tile was not visited by neither at the current timestep
    2. Information gain:
    -> Decreasing the value of a tile for each timestep a tile was visited by self 
    -> Decreasing the value of a tile for each timestep a tile was visited by partner
    3. Recency:
    -> Tiles visited longer ago by self become more/less attractive (recency_self)
    -> Tiles visited longer ago by partner become more/less attractive (recency_partner)
    4. Choice stickiness:
    -> Bias for/against staying on the current tile
    5. Movement perseverance:
    -> Incrementing the value of a tile if it is in the last movement direction of self
    
    These hypotheses result in a 7-parameter model.
    
    Technical limitations given by task design:
    1. Area-restricted search
    -> A tile can only be visited in the next time step if it is adjacent to the current position.
    Therefore, each non-adjacent tile of self's current position is assigned a temporary negative offset (fixed; non-learnable) in order to mark them with a ~0% next action probability.
    That way softmax-normalization gives higher action probbalities to adjacent tiles.
    """
    
    def __init__(
        self,
        n_participants: int, 
        grid_size: int = 4,
        time_max: float = 10,
        device = torch.device('cpu'),
        *args,
        **kwargs,
        ):
        
        super().__init__()
        
        self.device = device
        self.grid_size = grid_size
        self.n_tiles = grid_size**2
        self.n_actions = grid_size**2
        self.time_max = time_max
        self.non_adjacency_offset = -4
        
        # Tile coordinates (auto-moves with model.to(device))
        self.register_buffer('tile_rows', (torch.arange(self.n_tiles, device=self.device) // grid_size).float())
        self.register_buffer('tile_cols', (torch.arange(self.n_tiles, device=self.device) % grid_size).float())
        self.register_buffer('adjacency_matrix', self._build_adjacency_matrix())
        
        # cognitive parameters
        self.not_visited_self = torch.nn.Parameter(torch.randn(n_participants, 1)*0.001)
        self.not_visited_partner = torch.nn.Parameter(torch.randn(n_participants, 1)*0.001)
        self.visited_by_self = torch.nn.Parameter(torch.randn(n_participants, 1)*0.001)
        self.visited_by_partner = torch.nn.Parameter(torch.randn(n_participants, 1)*0.001)
        # self.choice_stickiness = torch.nn.Parameter(torch.randn(n_participants, 1)*0.001)
        self.movement_perseverance = torch.nn.Parameter(torch.randn(n_participants, 1)*0.001)
        
        # tile value memory
        self.state_keys = (
            'information_value_self',
            'information_value_partner',
            'movement_perseverance',
            'time_since_visited_by_self',
            'time_since_visited_by_partner',
        )
        
        self.init_state()
        
    def forward(self, inputs: torch.Tensor, prev_state=None):
        
        if prev_state is None:
            self.init_state(batch_size=inputs.shape[0])
        else:
            self.state = prev_state
        
        # feature extraction
        inputs = inputs.nan_to_num(0.).permute(1, 2, 0, 3)[:, 0]  # remove within_trail_timestep dimension (always 1)
        visited_by_self = inputs[..., :self.n_tiles].bool()
        visited_by_partner = inputs[..., self.n_tiles].long()
        visited_by_partner = torch.nn.functional.one_hot(visited_by_partner, num_classes=self.n_tiles).bool()
        not_visited = ~visited_by_self * ~visited_by_partner
        time_point = inputs[..., self.n_tiles+1] / self.time_max
        timesteps = torch.arange(inputs.shape[0])
        participant_ids = inputs[..., -1][0].long()
        
        # Precompute alignment for all timesteps (no loop needed)
        alignment = self._precompute_alignment(visited_by_self.long()) > 0  # bool: tile ahead in movement direction
        
        # sequential loop
        logits = torch.zeros_like(visited_by_self, dtype=torch.float)
        prev_time = torch.zeros_like(time_point[0])
        for timestep in timesteps:
            
            # Update time since visited (increment for all tiles)
            dt = (time_point[timestep] - prev_time).unsqueeze(-1)  # (B, 1)
            self.state['time_since_visited_by_self'] = self.state['time_since_visited_by_self'] + dt
            self.state['time_since_visited_by_partner'] = self.state['time_since_visited_by_partner'] + dt
            
            # --- compute state updates ---
            
            # 1. Information loss:
            # -> Increasing the value of a tile for each timestep a tile was not visited by neither at the current timestep
            new_info = self.state['information_value_self'] + self.state['time_since_visited_by_self'] * self.not_visited_self[participant_ids]
            self.state['information_value_self'] = torch.where(not_visited[timestep], new_info, self.state['information_value_self'])
            new_info = self.state['information_value_partner'] + self.state['time_since_visited_by_partner'] * self.not_visited_partner[participant_ids]
            self.state['information_value_partner'] = torch.where(not_visited[timestep], new_info, self.state['information_value_partner'])
            
            # 2. Information gain:
            # -> Decreasing the value of a tile for each timestep a tile was visited by self
            new_info = self.state['information_value_self'] + self.state['time_since_visited_by_self'] * self.visited_by_self[participant_ids]
            self.state['information_value_self'] = torch.where(visited_by_self[timestep], new_info, self.state['information_value_self'])
            
            # -> Decreasing the value of a tile for each timestep a tile was visited by partner
            new_info = self.state['information_value_partner'] + self.state['time_since_visited_by_partner'] * self.visited_by_partner[participant_ids]
            self.state['information_value_partner'] = torch.where(visited_by_partner[timestep], new_info, self.state['information_value_partner'])
            
            # 3. Movement perseverance:
            # -> Incrementing the value of a tile if it is in the last movement direction of self and reset otherwise
            new_pers = self.state['movement_perseverance'] + self.movement_perseverance[participant_ids]
            self.state['movement_perseverance'] = torch.where(alignment[timestep], new_pers, torch.zeros_like(self.state['movement_perseverance']))
            
            # --- update time stamps ---
            
            # Reset time since visited for tiles visited this timestep
            self.state['time_since_visited_by_self'] = self.state['time_since_visited_by_self'] * (1 - visited_by_self[timestep].float())
            self.state['time_since_visited_by_partner'] = self.state['time_since_visited_by_partner'] * (1 - visited_by_partner[timestep].float())
            prev_time = time_point[timestep]
            
            # --- output processing ---
            
            # Compute logits
            tile_logits = self.state['information_value_self'] + self.state['information_value_partner'] + self.state['movement_perseverance']
            
            # Apply adjacency mask
            curr_tile = visited_by_self[timestep].float().argmax(dim=-1)
            adjacency_mask = self.adjacency_matrix[curr_tile]
            tile_logits = torch.where(adjacency_mask, tile_logits, torch.full_like(tile_logits, self.non_adjacency_offset))
            
            # write to output
            logits[timestep] = tile_logits
            
        return logits.permute(1, 0, 2).unsqueeze(2), self.state
        
    def init_state(self, batch_size: int = 1):
        self.state = {}
        for state in self.state_keys:
            self.state[state] = torch.zeros((batch_size, self.n_tiles), dtype=torch.float, device=self.device)
        
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
            actions: (T, B, n_actions) one-hot action tensor
        Returns:
            alignment: (T, B, n_tiles)
        """
        curr_tiles = actions.argmax(dim=-1)
        rows = (curr_tiles // self.grid_size).float()
        cols = (curr_tiles % self.grid_size).float()
        
        # Single-step movement delta
        dr = torch.zeros_like(rows)
        dc = torch.zeros_like(cols)
        dr[1:] = rows[1:] - rows[:-1]
        dc[1:] = cols[1:] - cols[:-1]
        
        # Direction from current position to each tile
        cand_dr = self.tile_rows - rows.unsqueeze(-1)
        cand_dc = self.tile_cols - cols.unsqueeze(-1)
        
        # Dot product: per-tile alignment with movement direction
        alignment = (dr.unsqueeze(-1) * cand_dr + dc.unsqueeze(-1) * cand_dc) / self.grid_size
        
        return alignment
    
    def to(self, device: torch.device):
        self = super().to(device)
        for state in self.state:
            self.state[state] = self.state[state].to(device)
        self.device = device
        return self
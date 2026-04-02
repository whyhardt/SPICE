import torch

from spice.utils.convert_dataset import csv_to_dataset


class GQLModel(torch.nn.Module):
    """
    Generalized Q-Learning (GQL) model from Dezfouli et al. (2019).

    Maintains d-dimensional Q-values and choice histories per action.

    Update rules:
    - Q_chosen   = (1-phi) * Q_chosen   + phi * reward
    - Q_unchosen = (1-phi) * Q_unchosen
    - H_chosen   = (1-chi) * H_chosen   + chi
    - H_unchosen = (1-chi) * H_unchosen

    Action values:
    - V_a = sum_d(beta_d * Q_a_d) + sum_d(kappa_d * H_a_d) + H_a^T C Q_a

    Model parameters (per participant, per dimension):
    - phi:   Learning rate for Q-values          (n_participants, d)
    - chi:   Learning rate for choice history     (n_participants, d)
    - beta:  Q-value weight                       (n_participants, d)
    - kappa: Choice history weight                (n_participants, d)
    - C:     H*Q interaction matrix               (n_participants, d, d)
    """

    init_values = {
        'q_values': 0.5,
        'h_values': 0.0,
    }

    def __init__(self, n_actions: int = 2, n_participants: int = 1, d: int = 2, batch_first: bool = False):
        super().__init__()

        self.n_actions = n_actions
        self.n_participants = n_participants
        self.d = d
        self.batch_first = batch_first

        # Per-participant, per-dimension parameters (raw, transformed via properties)
        self.phi_raw = torch.nn.Parameter(torch.zeros(n_participants, d))
        self.chi_raw = torch.nn.Parameter(torch.zeros(n_participants, d))
        self.beta_raw = torch.nn.Parameter(torch.zeros(n_participants, d))
        self.kappa_raw = torch.nn.Parameter(torch.zeros(n_participants, d))
        self.C_raw = torch.nn.Parameter(torch.zeros(n_participants, d, d))

        self.device = torch.device('cpu')

    @property
    def phi(self):
        return torch.clamp(torch.sigmoid(self.phi_raw), 0.01, 0.99)

    @property
    def chi(self):
        return torch.clamp(torch.sigmoid(self.chi_raw), 0.01, 0.99)

    @property
    def beta(self):
        return torch.clamp(torch.nn.functional.softplus(self.beta_raw), 0.1, 10.0)

    @property
    def kappa(self):
        return torch.clamp(self.kappa_raw, -10.0, 10.0)

    @property
    def C(self):
        return torch.clamp(self.C_raw, -10.0, 10.0)

    def forward(self, inputs, prev_state=None):
        input_variables, participant_ids, logits, timesteps = self.init_forward_pass(
            inputs, prev_state, self.batch_first
        )
        actions, rewards = input_variables

        q_values = self.state['q_values']  # (B, n_actions, d)
        h_values = self.state['h_values']  # (B, n_actions, d)

        for t in timesteps:
            action_t = actions[t].unsqueeze(-1)   # (B, n_actions, 1)
            reward_t = rewards[t].unsqueeze(-1)   # (B, n_actions, 1)

            # Per-participant learning rates: (B, 1, d)
            phi = self.phi[participant_ids].unsqueeze(1)
            chi = self.chi[participant_ids].unsqueeze(1)

            # Q-value update: chosen gets reward, all decay
            q_values = (1 - phi) * q_values + phi * reward_t * action_t

            # History update: chosen gets +1, all decay
            h_values = (1 - chi) * h_values + chi * action_t

            # Action values: sum_d(beta*Q) + sum_d(kappa*H) + H^T C Q
            beta = self.beta[participant_ids].unsqueeze(1)      # (B, 1, d)
            kappa = self.kappa[participant_ids].unsqueeze(1)     # (B, 1, d)
            C = self.C[participant_ids]                          # (B, d, d)

            q_weighted = (beta * q_values).sum(dim=-1)           # (B, n_actions)
            h_weighted = (kappa * h_values).sum(dim=-1)          # (B, n_actions)
            interaction = torch.einsum('bad,bde,bae->ba', h_values, C, q_values)  # (B, n_actions)

            logits[t] = q_weighted + h_weighted + interaction

        self.state['q_values'] = q_values
        self.state['h_values'] = h_values

        logits = logits.unsqueeze(1)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)

        return logits, self.get_state()

    def init_forward_pass(self, inputs, prev_state, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 2, 0, 3)
        inputs = inputs[:, 0]

        inputs = inputs.nan_to_num(0.)

        actions = inputs[:, :, :self.n_actions].float()
        rewards = inputs[:, :, self.n_actions:2 * self.n_actions].float()

        participant_ids = inputs[0, :, -1].long()

        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])

        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)

        return (actions, rewards), participant_ids, logits, timesteps

    def set_initial_state(self, batch_size=1):
        state = {}
        for key, val in self.init_values.items():
            state[key] = torch.full((batch_size, self.n_actions, self.d), val, dtype=torch.float32)
        self.set_state(state)
        return self.get_state()

    def set_state(self, state_dict):
        self.state = state_dict

    def get_state(self, detach=False):
        state = self.state
        if detach:
            state = {key: state[key].detach() for key in state}
        return state
    
    def count_parameters(self):
        return 4 * self.d + 2 * self.d


class EnvironmentDezfouli2019:
    
    def __init__(self):
        self.reward_probs = {
            0: (0.25, 0.05),
            1: (0.12, 0.05),
            2: (0.08, 0.05),
        }
        
    def step(self):
        
        return 
import sys
import torch
from typing import Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_sessiondim

sys.path.append('../../..')
from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior


def get_dataset(path_data: str = None, test_sessions: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:

    # Load your data
    if path_data is None:
        path_data = 'data/dezfouli2019.csv'

    dataset = csv_to_dataset(
        file = path_data,
        )
    dataset.normalize_rewards()

    n_participants = dataset.n_participants
    n_actions = dataset.n_actions

    if verbose:
        print(f"Shape of dataset: {dataset.xs.shape}")
        print(f"Number of participants: {n_participants}")
        print(f"Number of actions in dataset: {n_actions}")

    if test_sessions is None:
        test_sessions = (3, 6, 9)
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, test_sessions)

    info_dataset = {
        'n_participants': n_participants,
        'n_actions': n_actions,
    }

    return dataset_train, dataset_test, info_dataset


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


class EnvironmentDezfouli2019(Env):
    """Two-armed bandit with block-specific Bernoulli reward probabilities.

    Each participant performs 12 blocks. Each block has fixed reward probabilities
    for each arm. Rewards are binary (0 or 1), sampled from Bernoulli distributions.

    Block-reward-probability mapping (from the original experiment data):
        Block 1:  (0.25,  0.05)    Block 7:  (0.05, 0.08)
        Block 2:  (0.05,  0.25)    Block 8:  (0.08, 0.05)
        Block 3:  (0.05,  0.125)   Block 9:  (0.125, 0.05)
        Block 4:  (0.125, 0.05)    Block 10: (0.05, 0.125)
        Block 5:  (0.08,  0.05)    Block 11: (0.05, 0.25)
        Block 6:  (0.05,  0.08)    Block 12: (0.25, 0.05)

    Sessions are processed in parallel: one session per (participant, block) pair.
    """

    # Reward probabilities indexed by block ID (1-indexed, matching dataset metadata).
    # Index 0 is unused padding so that block_id maps directly to tensor index.
    REWARD_PROBS = torch.tensor([
        [0.00, 0.00],   # index 0: unused (blocks are 1-indexed)
        [0.25, 0.05],   # block 1
        [0.05, 0.25],   # block 2
        [0.05, 0.125],  # block 3
        [0.125, 0.05],  # block 4
        [0.08, 0.05],   # block 5
        [0.05, 0.08],   # block 6
        [0.05, 0.08],   # block 7
        [0.08, 0.05],   # block 8
        [0.125, 0.05],  # block 9
        [0.05, 0.125],  # block 10
        [0.05, 0.25],   # block 11
        [0.25, 0.05],   # block 12
    ])

    def __init__(self, n_actions: int, n_participants: int, n_blocks: int = 12):
        super().__init__(n_actions, n_participants, n_blocks)
        self.session_reward_probs = None

    def reset(self, block_ids: torch.Tensor, participant_ids: torch.Tensor = None, **kwargs) -> None:
        """Set up per-session reward probabilities from block IDs.

        Args:
            block_ids: (n_sessions,) block index per session (1-indexed from dataset metadata).
            participant_ids: (n_sessions,) participant index (unused — reward probs are block-specific).
        """
        self.session_reward_probs = self.REWARD_PROBS[block_ids]  # (n_sessions, n_actions)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Bernoulli reward for the chosen arm in each session.

        Args:
            action: (n_sessions,) integer action index (0 or 1) per session.

        Returns:
            reward: (n_sessions,) binary reward (0.0 or 1.0).
            terminated: (n_sessions,) always False (trial count managed externally).
        """
        n_sessions = action.shape[0]
        probs = self.session_reward_probs[torch.arange(n_sessions), action]
        reward = torch.bernoulli(probs)
        reward_cols = torch.full((n_sessions, 2), float('nan'))
        reward_cols[torch.arange(n_sessions), action] = reward
        
        terminated = torch.zeros(n_sessions, dtype=torch.bool)
        return reward_cols, terminated


def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    path_data: str = None,
    dataset: SpiceDataset = None,
    save_dataset: str = None,
    ) -> SpiceDataset:
    """Generate synthetic behavior for the Dezfouli 2019 two-armed bandit task.

    Args:
        model: Fitted model (SpiceEstimator or torch.nn.Module).
        path_data: Path to the original CSV data file.
        dataset: Pre-loaded SpiceDataset (used if provided, otherwise loaded from path_data).
        save_dataset: Optional path to save the generated dataset as CSV.

    Returns:
        SpiceDataset with model-generated behavior.
    """
    if dataset is None:
        dataset, _, _ = get_dataset(path_data=path_data)

    environment = EnvironmentDezfouli2019(
        n_actions=dataset.n_actions,
        n_participants=dataset.n_participants,
        n_blocks=12,
    )

    dataset_gen = _generate_behavior(
        dataset=dataset,
        model=model,
        environment=environment,
        save_dataset=save_dataset,
    )

    return dataset_gen

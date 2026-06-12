import sys

import torch
from typing import Optional, Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

sys.path.append('../../..')
from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior


def get_dataset(path_data: str = None, test_blocks: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:
    
    # Load your data
    if path_data is None:
        path_data = 'data/bustamante2023.csv'
    
    dataset = csv_to_dataset(
        file = path_data,
        df_participant_id='subject_id',
        df_choice='decision',
        df_feedback='reward',
        df_block='overall_round',
        additional_inputs=['harvest_duration', 'travel_duration'],
        )
    dataset.normalize_rewards()

    # structure of dataset:
    # dataset has two main attributes: xs -> inputs; ys -> targets (next action)
    # shape: (n_participants*n_blocks*n_experiments, n_trials, n_timesteps=1, features)
    # features are (n_actions * action, n_actions * reward, n_additional_inputs * additional_input, timestep, trial, block, experiment, participant)

    # in order to set up the participant embedding we have to compute the number of unique participants in our data 
    # to get the number of participants n_participants we do:
    n_participants = dataset.n_participants
    n_actions = dataset.n_actions
    n_additional_inputs = dataset.n_additional_inputs
    
    if verbose:
        print(f"Shape of dataset: {dataset.xs.shape}")
        print(f"Number of participants: {n_participants}")
        print(f"Number of actions in dataset: {n_actions}")
        print(f"Number of additional inputs: {dataset.xs.shape[-1]-2*n_actions-3}")
        
    if test_blocks is None:
        test_blocks = (3, 6)
    if test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train = dataset_test = dataset
    
    info_dataset = {
        'n_participants': n_participants,
        'n_actions': n_actions,
        'n_additional_inputs': n_additional_inputs,
    }
    
    return dataset_train, dataset_test, info_dataset


class MarginalValueTheoremModel(torch.nn.Module):
    """
    Marginal Value Theorem (MVT) model for foraging task (Bustamante et al. 2023).

    The MVT predicts that a forager should leave a patch when the instantaneous
    gain rate in the current patch drops to the average gain rate in the environment.
 
    Key components:
    - Within-patch gain rate tracking (with depletion)
    - Environmental average gain rate estimation
    - Travel time and harvest time from data
    - Individual differences in parameters

    Model parameters (per individual):
    - alpha_env: Learning rate for environmental gain rate
    - beta: Inverse temperature for decision softmax
    - c: Intercept/bias term for decision rule
    - depletion (optional): Rate of within-patch depletion
    - baseline_gain (optional): Initial expected gain in a patch
    """

    init_values = {
        'cumulative_reward': 0.0,  # Cumulative reward in current patch
        'n_harvests': 0,  # Number of harvests in current patch
        'time_in_patch': 0.0,  # Time spent in current patch
        'env_reward_rate': 0.0,  # Estimated environmental gain rate
        'current_tree_state': 0.0,  # Current expected reward from tree (si)
    }

    def __init__(
        self,
        n_actions: int = 2,
        n_participants: int = 1,
        depletion: Optional[float] = None,
        baseline_gain: Optional[float] = None,
        batch_first: Optional[bool] = False,
    ):
        super().__init__()
        
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.batch_first = batch_first

        # Initialize parameters (raw, will be transformed)
        # Learning rate for environmental gain rate (0-1)
        self.alpha_env_raw = torch.nn.Parameter(torch.zeros(n_participants))

        # Inverse temperature for softmax decision (positive)
        self.beta_raw = torch.nn.Parameter(torch.zeros(n_participants))

        # Intercept/bias term for decision rule (unconstrained)
        self.c_raw = torch.nn.Parameter(torch.zeros(n_participants))

        # Within-patch depletion rate (positive)
        if depletion is None:
            self.depletion_raw = torch.nn.Parameter(torch.zeros(n_participants))
            self.learn_depletion = True
        else:
            self.register_buffer('depletion_fixed', torch.full((n_participants), depletion))
            self.learn_depletion = False
            
        # Baseline gain rate expectation (positive)
        if baseline_gain is None:
            self.baseline_gain_raw = torch.nn.Parameter(torch.zeros(n_participants))
            self.learn_baseline_gain = True
        else:
            self.register_buffer('baseline_gain_fixed', torch.full((n_participants), baseline_gain))
            self.learn_baseline_gain = False

        self.device = torch.device('cpu')
        
    def forward(self, inputs, prev_state=None):
        """Forward pass through the model.

        Implements the MVT learning model from Constantino et al. (2015), Table 2:
        - P(harvest) = 1/{1 + exp[-c - β(kappa * si - ro * h)]}
        - delta_i = r_i/tau_i - ro_i
        - ro_{i+1} = ro_i + [1-(1-alpha)^tau_i] · delta_i
        """
        input_variables, participant_ids, logits, timesteps = self.init_forward_pass(inputs, prev_state, self.batch_first)
        actions, rewards, harvest_duration, travel_duration = input_variables

        # Process each timestep
        for t in timesteps:
            # Get current timestep data (shape: [batch_size, ...])
            action_t = actions[t]  # (B, n_actions)
            reward_t = rewards[t]  # (B, n_actions)
            harvest_duration_t = harvest_duration[t].squeeze(-1)  # (B,)
            travel_duration_t = travel_duration[t].squeeze(-1)  # (B,)

            # Compute expected next reward (si) using MVT formulation
            # si is the expected reward if we harvest now
            expected_next_reward = self.state['current_tree_state'] * self.depletion[participant_ids]

            # Compute stay value using MVT: c + β(si - ρh)
            # Stay is better when expected reward exceeds opportunity cost
            value_stay = expected_next_reward - self.state['env_reward_rate'] * harvest_duration_t

            # Compute stay logit for this timestep (for action 0 = harvest); exit value is always 0
            # Include intercept c (bias term) and inverse temperature β
            logits[t, :, 0] = self.c[participant_ids] + self.beta[participant_ids] * value_stay

            # Determine which action was taken
            # action_t[:, 0] = 1 means harvest, action_t[:, 1] = 1 means leave
            did_harvest = action_t[:, 0]  # (B,)
            did_leave = 1.0 - did_harvest

            # Get actual reward received
            actual_reward = reward_t.nansum(dim=-1)

            # Determine time spent on this action
            time_spent = harvest_duration_t * did_harvest + travel_duration_t * did_leave

            # Update environmental reward rate using MVT learning rule (Table 2)
            # delta_i = r_i/tau_i - ro_i
            reward_rate = actual_reward / (time_spent + 1e-8)  # Avoid division by zero
            prediction_error = reward_rate - self.state['env_reward_rate']

            # Effective learning rate: [1 - (1-alpha)^taui]
            # This accounts for the time duration of the action
            effective_lr = 1.0 - torch.pow(1.0 - self.alpha_env[participant_ids], time_spent)
            self.state['env_reward_rate'] = self.state['env_reward_rate'] + effective_lr * prediction_error

            # Update patch state based on action taken
            # If harvested: update cumulative stats for current patch
            # If left: reset patch state for new patch
            self.state['cumulative_reward'] = torch.where(
                did_harvest > 0.5,
                self.state['cumulative_reward'] + actual_reward,
                0
            )
            self.state['n_harvests'] = torch.where(
                did_harvest > 0.5,
                self.state['n_harvests'] + 1,
                0
            )
            self.state['time_in_patch'] = torch.where(
                did_harvest > 0.5,
                self.state['time_in_patch'] + time_spent,
                0
            )
            self.state['current_tree_state'] = torch.where(
                did_harvest > 0.5,
                actual_reward,  # Update state to observed reward
                self.baseline_gain[participant_ids].detach(),
            )

        logits = logits.unsqueeze(1)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)
            
        return logits, self.get_state()

    @property
    def alpha_env(self):
        return torch.clamp(torch.sigmoid(self.alpha_env_raw), 0.01, 0.99)

    @property
    def beta(self):
        return torch.clamp(torch.nn.functional.softplus(self.beta_raw), 0.1, 10.0)

    @property
    def c(self):
        """Intercept/bias term for decision rule (unconstrained, but clamped for stability)"""
        return torch.clamp(self.c_raw, -10.0, 10.0)

    @property
    def depletion(self):
        if self.learn_depletion:
            depletion = torch.clamp(torch.nn.functional.softplus(self.depletion_raw), 0.001, 1.0)
        else:
            depletion = self.depletion_fixed
        return depletion

    @property
    def baseline_gain(self):
        if self.learn_baseline_gain:
            baseline_gain = torch.clamp(torch.nn.functional.softplus(self.baseline_gain_raw), 0.1, 20.0)
        else:
            baseline_gain = self.baseline_gain_fixed
        return baseline_gain

    def init_forward_pass(self, inputs, prev_state, batch_first):
        """Initialize forward pass."""
        if batch_first:
            inputs = inputs.permute(1, 2, 0, 3)
        inputs = inputs[:, 0]
        
        inputs = inputs.nan_to_num(0.)
        
        # Extract actions and rewards
        actions = inputs[:, :, :self.n_actions].float()  # (T, B, n_actions)
        rewards = inputs[:, :, self.n_actions:2*self.n_actions].float()  # (T, B, n_actions)

        # Extract harvest_duration and travel_duration from additional inputs
        # additional_inputs: [harvest_duration, travel_duration]
        harvest_duration = inputs[:, :, 2*self.n_actions:2*self.n_actions+1].float()  # (T, B, 1)
        travel_duration = inputs[:, :, 2*self.n_actions+1:2*self.n_actions+2].float()  # (T, B, 1)

        participant_ids = inputs[0, :, -1].long()
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
            self.state['env_reward_rate'] = self.baseline_gain[participant_ids]
            self.state['current_tree_state'] = self.baseline_gain[participant_ids]

        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)

        return (actions, rewards, harvest_duration, travel_duration), participant_ids, logits, timesteps

    def set_initial_state(self, batch_size=1):
        """Initialize the hidden state for each session."""

        state = {
            'cumulative_reward': torch.zeros(batch_size, dtype=torch.float32),
            'n_harvests': torch.zeros(batch_size, dtype=torch.float32),  # Changed to float for torch.where compatibility
            'time_in_patch': torch.zeros(batch_size, dtype=torch.float32),
            'env_reward_rate': torch.zeros(batch_size, dtype=torch.float32),
            'current_tree_state': torch.zeros(batch_size, dtype=torch.float32),
        }
        self.set_state(state)
        return self.get_state()

    def set_state(self, state_dict):
        """Set the latent variables."""
        self.state = state_dict

    def get_state(self, detach=False):
        """Return the memory state."""
        state = self.state
        if detach:
            state = {key: state[key].detach() for key in state}
        return state
    

class EnvironmentBustamante2023(Env):
    """Patch foraging environment with depleting rewards (Bustamante et al., 2023).

    Each patch starts with an initial reward drawn from a clipped Gaussian.
    Successive harvests deplete the reward via a Beta-distributed multiplicative
    factor.  Exiting a patch yields no reward and resets to a fresh patch.

    Reward generation:
        R_0        ~ N(15, 1),  clipped to [0.5, 20]
        depletion  ~ Beta(14.91, 2.03)            (mean ≈ 0.88)
        R_{t+1}    = max(R_t * depletion, 0.5)

    Actions:
        0 = harvest  (stay in patch, receive reward, deplete)
        1 = exit     (leave patch, no reward, new patch drawn)

    Rewards are normalized to [0, 1] by dividing by the maximum possible
    reward (20).  harvest_duration and travel_duration are fixed task
    parameters (in seconds).

    Sessions are processed in parallel: one session per (participant, block).
    """

    RHO_M = 15.0            # mean of initial-reward Gaussian
    RHO_SD = 1.0            # std  of initial-reward Gaussian
    REWARD_MAX = 20.0       # ceiling (also used as normalization scale)
    REWARD_CUTOFF = 0.5     # floor
    DELTA_A = 14.90873      # Beta α for depletion factor
    DELTA_B = 2.033008      # Beta β for depletion factor

    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        n_blocks: int,
        harvest_duration: float = 2.0,
        travel_duration: float = 8.333333333333334,
    ):
        """
        Args:
            n_actions: Number of actions (2: harvest / exit).
            n_participants: Number of participants.
            n_blocks: Number of blocks per participant.
            harvest_duration: Time (seconds) for a harvest action.
            travel_duration: Time (seconds) for travel to a new patch.
        """
        super().__init__(n_actions, n_participants, n_blocks)
        self.harvest_duration = harvest_duration
        self.travel_duration = travel_duration

    def _draw_initial_reward(self, n: int) -> torch.Tensor:
        """Sample initial patch reward from clipped Gaussian."""
        r = torch.normal(self.RHO_M, self.RHO_SD, size=(n,))
        return torch.clamp(r, self.REWARD_CUTOFF, self.REWARD_MAX)

    def reset(self, block_ids: torch.Tensor = None, participant_ids: torch.Tensor = None) -> None:
        n = block_ids.shape[0] if block_ids is not None else self.n_participants * self.n_blocks
        self.patch_reward = self._draw_initial_reward(n)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one foraging trial for all sessions.

        Args:
            action: (n_sessions,) — 0 = harvest, 1 = exit.

        Returns:
            observation: (n_sessions, 4) — [reward_harvest, reward_exit,
                         harvest_duration, travel_duration].
            terminated:  (n_sessions,) always False.
        """
        n = action.shape[0]

        harvest = (action == 0)

        # Reward: harvesters receive current patch reward; exiters get 0
        reward_raw = torch.where(harvest, self.patch_reward, torch.zeros(n))
        reward = torch.clamp(reward_raw / self.REWARD_MAX, 0.0, 1.0)

        # Deplete patch for harvesters
        depletion = torch.distributions.Beta(self.DELTA_A, self.DELTA_B).sample((n,))
        depleted = torch.clamp(self.patch_reward * depletion, min=self.REWARD_CUTOFF)

        # Exiters get a fresh patch
        new_initial = self._draw_initial_reward(n)
        self.patch_reward = torch.where(harvest, depleted, new_initial)

        # Partial feedback: reward only in chosen action's column
        reward_cols = torch.full((n, self.n_actions), float('nan'))
        reward_cols[torch.arange(n), action] = reward

        # Fixed task durations (constant across all trials)
        h_dur = torch.full((n, 1), self.harvest_duration)
        t_dur = torch.full((n, 1), self.travel_duration)

        observation = torch.cat([reward_cols, h_dur, t_dur], dim=-1)

        terminated = torch.zeros(n, dtype=torch.bool)
        return observation, terminated


def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    path_data: str = None,
    dataset: SpiceDataset = None,
    save_dataset: str = None,
) -> SpiceDataset:
    """Generate synthetic behavior for the Bustamante 2023 foraging task.

    Args:
        model: Fitted model (SpiceEstimator or torch.nn.Module).
        path_data: Path to the original CSV data file.
        dataset: Pre-loaded SpiceDataset (used if provided, otherwise loaded from path_data).
        save_dataset: Optional path to save the generated dataset as CSV.

    Returns:
        SpiceDataset with model-generated behavior.
    """
    if dataset is None:
        dataset, _, _ = get_dataset(path_data=path_data, test_blocks=())

    n_blocks = len(dataset.xs[:, 0, 0, -3].unique())

    environment = EnvironmentBustamante2023(
        n_actions=dataset.n_actions,
        n_participants=dataset.n_participants,
        n_blocks=n_blocks,
    )

    return _generate_behavior(
        dataset=dataset,
        model=model,
        environment=environment,
        save_dataset=save_dataset,
        kwargs_dataset=dict(
            df_participant_id='subject_id',
            df_choice='decision',
            df_feedback='reward',
            df_block='overall_round',
            additional_inputs=['harvest_duration', 'travel_duration'],
        ),
    )

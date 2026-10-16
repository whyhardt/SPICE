import math
import sys

import pandas as pd
import torch
from typing import Optional, Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

sys.path.append('../../..')
from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior
from weinhardt2026.studies.bustamante2023.spice_bustamante2023 import CONFIG, prepare_dataframe


DATA_PATH = 'weinhardt2026/studies/bustamante2023/data/bustamante2023.csv'


def get_dataset(path_data: str = None, test_blocks: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:
    
    # Load your data
    if path_data is None:
        path_data = 'data/bustamante2023.csv'
    
    import pandas as pd
    dataset = csv_to_dataset(
        file=prepare_dataframe(pd.read_csv(path_data, low_memory=False)),     # adds the effort-condition inputs
        df_participant_id='subject_id',
        df_choice='decision',
        df_feedback='reward',
        df_block='overall_round',
        additional_inputs=CONFIG.additional_inputs,
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
    

EFFORT_CONDITIONS = ('button-pressing/low', 'button-pressing/high', 'MSIT/low', 'MSIT/high')


def effort_condition_table(path_data: str) -> torch.Tensor:
    """(n_participants, max_block + 1) effort condition per session, -1 where a session does not exist.

    Condition index = 2 * (travel task is MSIT) + (effort level is high), see EFFORT_CONDITIONS. Participant
    indices follow csv_to_dataset (order of first appearance of subject_id), blocks are the raw overall_round.
    """
    import pandas as pd
    df = pd.read_csv(path_data, low_memory=False)
    participant = df['subject_id'].map({value: index for index, value in enumerate(df['subject_id'].unique())})
    condition = 2 * (df['travelTask'] == 'MSIT').astype(int) + (df['effort_level'] == 'high').astype(int)
    table = torch.full((int(participant.max()) + 1, int(df['overall_round'].max()) + 1), -1, dtype=torch.long)
    sessions = pd.DataFrame({'participant': participant, 'block': df['overall_round'], 'condition': condition}).drop_duplicates()
    if sessions.duplicated(['participant', 'block']).any():
        raise ValueError('A session has more than one effort condition.')
    table[torch.tensor(sessions.participant.values), torch.tensor(sessions.block.values)] = torch.tensor(sessions.condition.values)
    return table


class EffortCostMVTModel(torch.nn.Module):
    """
    Marginal Value Theorem model with effort costs of travel (Bustamante et al. 2023, PNAS, Eqs. 2-5).

    The harvest decision compares the expected reward of the next harvest with a fixed threshold per
    effort condition (travel task: button pressing or MSIT x effort level: low or high):

        P(harvest) = sigmoid(beta * (R_e - rho_condition))
        rho_condition = (sum(r) - n_travel * c_condition) / T

    with R_e = 0.88 * mean(last two rewards), or 0.88 * the last reward if the one before was a travel
    (the rule behind the expectedReward column of the authors' data; 0.88 = mean depletion). On arrival at a
    fresh tree (the decision right after a travel), R_e is the expected yield of a fresh tree (15 apples, a task
    constant). The authors' column averages the travel's 0 into R_e there, which makes the model predict leaving
    on arrival, where participants harvest 95% of the time; fitted that way, the thresholds collapse toward 0 and
    leave decisions are barely discriminated (held-out trial likelihood 0.733 vs. 0.784, leave AUC 0.78 vs. 0.91).

    Fitted by maximum likelihood, the effort costs c enter only through rho, whose remaining terms are
    fixed per participant and condition: the model is fitted with one free threshold per condition,
    from which the costs follow as c = (sum(r) - rho * T) / n_travel. This also avoids the block totals
    in rho, which would be future information for trial-by-trial prediction. The original model was fitted
    hierarchically (Stan); here every participant is fitted individually, like the other benchmarks.

    The decision for trial t+1 is computed after processing trial t, i.e. it uses the reward just received.

    Model parameters (per individual): beta (inverse temperature) and rho for the four effort conditions.
    """

    DEPLETION = 0.88
    FRESH_TREE_MEAN = 15.                                            # apples: mean yield of a fresh tree (N(15, 1))

    def __init__(
        self,
        n_actions: int = 2,
        n_participants: int = 1,
        path_data: str = 'weinhardt2026/studies/bustamante2023/data/bustamante2023.csv',
        batch_first: bool = True,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_participants = n_participants
        self.batch_first = batch_first

        # rewards are normalized to [0, 1], so R_e varies by a few tenths: beta in log space, starting at 10
        # (the paper's prior beta ~ N(0, 0.5) is per apple, rewards 0-20), thresholds near typical R_e
        self.log_beta = torch.nn.Parameter(torch.full((n_participants,), math.log(10.)))
        self.threshold = torch.nn.Parameter(torch.full((n_participants, len(EFFORT_CONDITIONS)), 0.4))
        self.register_buffer('conditions', effort_condition_table(path_data), persistent=False)
        # rewards are min-max normalized over the received rewards (travel = 0 is the minimum)
        import pandas as pd
        rewards = pd.read_csv(path_data, usecols=['reward']).reward
        self.fresh_tree_value = self.FRESH_TREE_MEAN / rewards[rewards > 0].max()

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def count_parameters(self) -> int:
        return 1 + len(EFFORT_CONDITIONS)

    def forward(self, inputs, prev_state=None):
        if self.batch_first:
            inputs = inputs.permute(1, 2, 0, 3)                      # (B, T, W, F) -> (T, W, B, F)
        inputs = inputs[:, 0].nan_to_num(0.)
        actions = inputs[:, :, :self.n_actions]
        rewards = inputs[:, :, self.n_actions:2 * self.n_actions].sum(-1)   # reward of the chosen action (0 for travel)
        participant_ids = inputs[0, :, -1].long()
        block_ids = inputs[0, :, -3].long()

        # -1: no such session -- only rows zeroed for inactive sessions (generate_behavior); their outputs are unused
        condition = self.conditions[participant_ids, block_ids].clamp(min=0)
        threshold = self.threshold[participant_ids, condition]       # (B,)
        beta = self.beta[participant_ids]

        last_reward = prev_state['last_reward'] if prev_state is not None else torch.zeros_like(threshold)
        logits = torch.zeros_like(actions)
        for t in range(actions.shape[0]):
            reward = rewards[t]
            expected = self.DEPLETION * torch.where(last_reward > 0, (reward + last_reward) / 2, reward)
            travelled = actions[t][:, 1] > 0.5                       # the next decision is at a fresh tree
            expected = torch.where(travelled, torch.full_like(expected, self.fresh_tree_value), expected)
            logits[t, :, 0] = beta * (expected - threshold)          # action 0 = harvest; the exit value is 0
            last_reward = reward
        self.state = {'last_reward': last_reward}

        logits = logits.unsqueeze(1)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)
        return logits, self.state


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

    Rewards are scaled like the data (normalize_rewards: divided by the
    largest reward received in the data, `reward_scale`). The non-reward
    inputs (harvest_duration, travel_duration and the block's effort
    condition is_msit, is_high) are task settings and are replayed from the
    original sessions (`inputs`).

    Sessions are processed in parallel: one session per (participant, block).
    """

    RHO_M = 15.0            # mean of initial-reward Gaussian
    RHO_SD = 1.0            # std  of initial-reward Gaussian
    REWARD_MAX = 20.0       # ceiling of the reward process (apples)
    REWARD_CUTOFF = 0.5     # floor
    DELTA_A = 14.90873      # Beta α for depletion factor
    DELTA_B = 2.033008      # Beta β for depletion factor

    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        n_blocks: int,
        inputs: torch.Tensor,
        reward_scale: float,
    ):
        """
        Args:
            n_actions: Number of actions (2: harvest / exit).
            n_participants: Number of participants.
            n_blocks: Number of blocks per participant.
            inputs: (n_sessions, n_trials, n_additional_inputs) replayed task inputs (CONFIG.additional_inputs).
            reward_scale: Divisor that maps apples to the data's reward scale.
        """
        super().__init__(n_actions, n_participants, n_blocks)
        self.inputs = inputs
        self.reward_scale = reward_scale
        self.trial_counter = 0

    def _draw_initial_reward(self, n: int) -> torch.Tensor:
        """Sample initial patch reward from clipped Gaussian."""
        r = torch.normal(self.RHO_M, self.RHO_SD, size=(n,))
        return torch.clamp(r, self.REWARD_CUTOFF, self.REWARD_MAX)

    def reset(self, block_ids: torch.Tensor = None, participant_ids: torch.Tensor = None) -> None:
        n = block_ids.shape[0] if block_ids is not None else self.n_participants * self.n_blocks
        self.patch_reward = self._draw_initial_reward(n)
        self.trial_counter = 0

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one foraging trial for all sessions.

        Args:
            action: (n_sessions,) — 0 = harvest, 1 = exit.

        Returns:
            observation: (n_sessions, 2 + n_additional_inputs) — [reward_harvest, reward_exit,
                         replayed inputs (harvest_duration, travel_duration, is_msit, is_high)].
            terminated:  (n_sessions,) always False.
        """
        n = action.shape[0]

        harvest = (action == 0)

        # Reward: harvesters receive current patch reward; exiters get 0
        reward_raw = torch.where(harvest, self.patch_reward, torch.zeros(n))
        reward = torch.clamp(reward_raw / self.reward_scale, 0.0, 1.0)

        # Deplete patch for harvesters
        depletion = torch.distributions.Beta(self.DELTA_A, self.DELTA_B).sample((n,))
        depleted = torch.clamp(self.patch_reward * depletion, min=self.REWARD_CUTOFF)

        # Exiters get a fresh patch
        new_initial = self._draw_initial_reward(n)
        self.patch_reward = torch.where(harvest, depleted, new_initial)

        # Partial feedback: reward only in chosen action's column
        reward_cols = torch.full((n, self.n_actions), float('nan'))
        reward_cols[torch.arange(n), action] = reward

        observation = torch.cat([reward_cols, self.inputs[:n, self.trial_counter]], dim=-1)
        self.trial_counter += 1

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
    n_actions = dataset.n_actions

    # rewards as normalize_rewards scaled them in get_dataset: by the largest received reward
    rewards = pd.read_csv(path_data or DATA_PATH, usecols=['reward']).reward

    environment = EnvironmentBustamante2023(
        n_actions=n_actions,
        n_participants=dataset.n_participants,
        n_blocks=n_blocks,
        inputs=dataset.xs[:, :, 0, 2 * n_actions:-5].nan_to_num(0.),
        reward_scale=rewards.max(),
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
            additional_inputs=CONFIG.additional_inputs,
        ),
    )

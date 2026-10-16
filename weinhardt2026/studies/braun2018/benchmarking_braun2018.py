import sys

import torch
from typing import Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

sys.path.append('../../..')
from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior
from weinhardt2026.studies.braun2018.spice_braun2018 import CONFIG


def get_dataset(path_data: str = None, test_blocks: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:

    if path_data is None:
        path_data = 'data/braun2018.csv'

    dataset = csv_to_dataset(
        file=path_data,
        df_participant_id='subject',
        df_choice='transcode',
        df_feedback=None,
        df_block='block',
        additional_inputs=CONFIG.additional_inputs,
        timeshift_additional_inputs=(-1, -1, -1),
    )

    n_participants = dataset.n_participants
    n_actions = dataset.n_actions

    if verbose:
        print(f"Shape of dataset: {dataset.xs.shape}")
        print(f"Number of participants: {n_participants}")
        print(f"Number of actions in dataset: {n_actions}")

    if test_blocks is None:
        test_blocks = (3, 6, 9)
    if test_blocks:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train = dataset_test = dataset

    info_dataset = {
        'n_participants': n_participants,
        'n_actions': n_actions,
    }

    return dataset_train, dataset_test, info_dataset


class ExpectedValueControl(torch.nn.Module):
    """
    Expected Value of Control (EVC) model for reward-based voluntary task switching
    (Braun & Arrington, 2018; Shenhav, Botvinick & Cohen, 2013).

    The agent switches tasks when the reward benefit of switching outweighs the control cost of a
    switch, where the control cost grows with accumulated fatigue (block position in the experiment).

    EVC formulation
    ---------------
    Actions are coded relative to the previous task, a = {repeat=0, switch=1}. Only the difference
    EVC(switch) - EVC(repeat) is identifiable under a softmax, so the repeat logit is fixed at 0:

        EVC(repeat) = 0
        EVC(switch) = beta_reward * dV - (beta_cost + beta_fatigue * t_norm)

    where:
        dV     : point value of the other task minus the current task (shown before the choice)
        t_norm : fatigue signal in (0, 1], the (1-indexed) block number / N_BLOCKS

    Parameters (per participant)
    ----------------------------
        beta_reward  (n_participants,) : sensitivity to the value difference; > 0
        beta_cost    (n_participants,) : base control cost of a switch (indifference point); >= 0
        beta_fatigue (n_participants,) : additional switch cost per unit of t_norm; >= 0

    Output
    ------
    logits : (T, 1, B, n_actions) or (B, T, 1, n_actions) if batch_first
        EVC scores for [repeat, switch]
    """

    def __init__(
        self,
        n_actions: int = 2,
        n_participants: int = 1,
        batch_first: bool = True,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.n_participants = n_participants
        self.batch_first = batch_first
        self.n_blocks = 12

        # --- learnable parameters (raw; transformed via properties) ----------
        self.beta_reward_raw  = torch.nn.Parameter(torch.zeros(n_participants))
        self.beta_cost_raw    = torch.nn.Parameter(torch.zeros(n_participants))
        self.beta_fatigue_raw = torch.nn.Parameter(torch.zeros(n_participants))

        self.device = torch.device('cpu')

    # --- parameter transforms ------------------------------------------------

    @property
    def beta_reward(self):
        """Reward sensitivity — strictly positive."""
        return torch.nn.functional.softplus(self.beta_reward_raw).clamp(1e-3, 20.0)

    @property
    def beta_cost(self):
        """Base switch cost — non-negative."""
        return torch.nn.functional.softplus(self.beta_cost_raw).clamp(0.0, 20.0)

    @property
    def beta_fatigue(self):
        """Fatigue coefficient — non-negative (switch cost can only grow over time)."""
        return torch.nn.functional.softplus(self.beta_fatigue_raw).clamp(0.0, 20.0)

    # --- forward pass --------------------------------------------------------

    def forward(self, inputs, *args, **kwargs):
        (value_differences, t_norms), participant_ids, logits = self._init_forward_pass(inputs, self.batch_first)

        # value_differences : (T, B)  dV = V_other - V_current
        # t_norms           : (T, B)  fatigue signal in (0, 1]

        # Gather per-participant parameters for this batch
        beta_r = self.beta_reward[participant_ids]      # (B,)
        beta_c = self.beta_cost[participant_ids]        # (B,)
        beta_f = self.beta_fatigue[participant_ids]     # (B,)

        # EVC(switch) - EVC(repeat): reward benefit minus (base + fatigue) control cost
        logits[..., 1] = beta_r * value_differences - (beta_c + beta_f * t_norms)

        logits = logits.unsqueeze(1)                         # (T, 1, B, n_actions)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)             # (B, T, 1, n_actions)

        return logits, {}   # stateless model: return empty state dict

    def _init_forward_pass(self, inputs, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 2, 0, 3)
        inputs = inputs[:, 0]           # drop layer dim -> (T, B, features)
        inputs = inputs.nan_to_num(0.)

        T, B, _ = inputs.shape

        # first additional input (no reward columns): dV = V_other - V_current
        value_differences = inputs[:, :, self.n_actions].float()     # (T, B)
        block_idx = inputs[:, :, -3].float()                         # (T, B), 1-indexed

        t_norms = block_idx / self.n_blocks

        # Participant IDs are constant within a sequence; read from first timestep
        participant_ids = inputs[0, :, -1].long()   # (B,)

        logits = torch.zeros(T, B, self.n_actions, device=inputs.device)

        return (value_differences, t_norms), participant_ids, logits

    def count_parameters(self):
        """Free parameters per participant: beta_reward, beta_cost, beta_fatigue."""
        return 3


class EnvironmentBraun2018(Env):
    """Reward-based voluntary task switching (rVTS, Braun & Arrington, 2018).

    Two identification tasks, each with a point value (integer, 0-10).
    Both values start at 5 at the beginning of each block.

    After each task selection:
        - Selected task's value:   P=0.5 decreases by 1 (clamped at 0)
        - Unselected task's value: P=0.5 increases by 1 (clamped at 10)

    Actions encode relative choice:
        0 = repeat (perform the same task as the previous trial)
        1 = switch (perform the other task)

    Observation: [difference, current_decreased, other_increased].
    Returns the raw observation from the current step (no buffering).
    The timeshift is applied by get_dataset via timeshift_additional_inputs=(-1,-1,-1)
    when loading the saved CSV.

    Sessions are processed in parallel: one session per (participant, block).
    """

    def __init__(self, n_actions: int, n_participants: int, n_blocks: int):
        super().__init__(n_actions, n_participants, n_blocks)

    def reset(self, block_ids: torch.Tensor, participant_ids: torch.Tensor = None) -> None:
        n = block_ids.shape[0]
        # Absolute task values (A and B are arbitrary labels)
        self.value_A = torch.full((n,), 5.0)
        self.value_B = torch.full((n,), 5.0)
        # Track which task is "current" (was last selected)
        self.current_is_A = torch.ones(n, dtype=torch.bool)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one trial for all sessions.

        Args:
            action: (n_sessions,) — 0 = repeat, 1 = switch.

        Returns:
            observation: (n_sessions, 3) — [difference, current_decreased, other_increased]
                         from the current step.
            terminated:  (n_sessions,) always False.
        """
        n = action.shape[0]

        # Update which task is "current" based on repeat/switch
        switched = (action == 1)
        self.current_is_A = torch.where(switched, ~self.current_is_A, self.current_is_A)

        # Read current and other task values
        current_val = torch.where(self.current_is_A, self.value_A, self.value_B)
        other_val = torch.where(self.current_is_A, self.value_B, self.value_A)

        # Probabilistic value changes (Bernoulli coin flips)
        try_decrease = torch.bernoulli(torch.full((n,), 0.5))
        try_increase = torch.bernoulli(torch.full((n,), 0.5))

        new_current = torch.clamp(current_val - try_decrease, 0, 10)
        new_other = torch.clamp(other_val + try_increase, 0, 10)

        # Record whether a change actually occurred (boundary clamping may block it)
        cur_decreased = (new_current < current_val).float()
        oth_increased = (new_other > other_val).float()

        # Write back to absolute task values
        self.value_A = torch.where(self.current_is_A, new_current, new_other)
        self.value_B = torch.where(self.current_is_A, new_other, new_current)

        difference = new_other - new_current
        observation = torch.stack([difference, cur_decreased, oth_increased], dim=-1)

        terminated = torch.zeros(n, dtype=torch.bool)
        return observation, terminated


def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    path_data: str = None,
    dataset: SpiceDataset = None,
    save_dataset: str = None,
) -> SpiceDataset:
    """Generate synthetic behavior for the Braun 2018 rVTS task.

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

    environment = EnvironmentBraun2018(
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
            df_participant_id='subject',
            df_choice='transcode',
            df_feedback='reward',
            df_block='block',
            additional_inputs=CONFIG.additional_inputs,
        ),
    )

import sys

import torch
from typing import Union

from spice import SpiceEstimator, SpiceDataset, csv_to_dataset, split_data_along_blockdim

sys.path.append('../../..')
from weinhardt2026.utils.task import Env, generate_behavior as _generate_behavior


def get_dataset(path_data: str = None, test_blocks: tuple[int] = None, verbose: bool = False) -> tuple[SpiceDataset, SpiceDataset, dict]:

    if path_data is None:
        path_data = 'data/braun2018.csv'

    dataset = csv_to_dataset(
        file=path_data,
        df_participant_id='subject',
        df_choice='transcode',
        df_feedback=None,
        df_block='block',
        additional_inputs=('difference', 'current', 'other'),
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
    
    The agent selects the task with the highest EVC, computed as the reward
    benefit of a task minus its effort cost, where effort cost grows with
    accumulated fatigue (trial and block position within the experiment).

    EVC formulation
    ---------------
    For a={repeat=0, switch=1}:

        EVC(a) = beta_reward * V_a
               - (beta_cost + beta_fatigue * t_norm) * switch(a, prev_task)
               + bias_a

    where:
        V_a             : observed point value for task a (reward_repeat / reward_switch)
        switch(a, prev) : 1 if a != prev_task, else 0
        t_norm          : fatigue signal in [0, 1], combining trial and block:
                              t_norm = (block * N_TRIALS_PER_BLOCK + trial) / T_max
        bias_a          : per-participant, per-task intercept (captures repetition bias)

    Parameters (per participant)
    ----------------------------
        beta_reward  (n_participants,)           : sensitivity to point values; > 0
        beta_cost    (n_participants,)           : base effort cost of switching; >= 0
        beta_fatigue (n_participants,)           : additional cost per unit of t_norm; >= 0
        bias         (n_participants, n_actions) : per-task intercept; unconstrained

    Output
    ------
    logits : (T, B, 1, n_actions) or (B, T, 1, n_actions) if batch_first
        Raw EVC scores for [repeating, switching]
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
        self.bias_raw         = torch.nn.Parameter(torch.zeros(n_participants, n_actions))

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

    @property
    def bias(self):
        """Per-task intercept — unconstrained."""
        return self.bias_raw.clamp(-10.0, 10.0)

    # --- forward pass --------------------------------------------------------

    def forward(self, inputs, *args, **kwargs):
        (prev_actions, task_values, t_norms), participant_ids, logits, timesteps = self._init_forward_pass(inputs, self.batch_first)

        # prev_actions : (T, B, n_actions)  one-hot of task chosen at t-1
        # task_values  : (T, B, n_actions)  point values [V_repeat, V_switch]
        # t_norms      : (T, B)             fatigue signal in [0, 1]

        B = logits.shape[1]

        # The first trial is always a switch for both options.
        # Initialise prev_action to all-zeros so switch_mask = 1 - 0 = all-ones at t=0.
        prev_action = torch.zeros(B, self.n_actions, device=prev_actions.device)        # (B, n_actions)
        
        # switch_mask[b, a] = 1 if task a was NOT selected on the previous trial.
        # At t=0, prev_action is all-zeros so switch_mask is all-ones (always a switch).
        # switch_mask = 1.0 - prev_action                 # (B, n_actions)
        switch_mask = torch.zeros_like(prev_actions[0])
        switch_mask[..., 1] = 1
        
        # Gather per-participant parameters for this batch
        beta_r = self.beta_reward[participant_ids]      # (B,)
        beta_c = self.beta_cost[participant_ids]        # (B,)
        beta_f = self.beta_fatigue[participant_ids]     # (B,)
        bias_t = self.bias[participant_ids]             # (B, n_actions)
        
        for t in timesteps:
            task_values_t = task_values[t]      # (B, n_actions)
            t_norm_t      = t_norms[t]          # (B,)
            
            # Total effort penalty per action:
            # base cost + fatigue term scaled by normalised position in experiment
            effort_penalty = (
                beta_c + beta_f * t_norm_t
            ).unsqueeze(1)                                   # (B, 1) -> broadcasts over actions

            # EVC for each action
            evc = (
                beta_r.unsqueeze(1) * task_values_t         # reward benefit  (B, n_actions)
                - effort_penalty * switch_mask               # effort cost     (B, n_actions)
                + bias_t                                     # intercept       (B, n_actions)
            )

            logits[t] = evc

            # Advance state: the chosen action at t becomes prev_action for t+1
            prev_action = prev_actions[t]                   # (B, n_actions) one-hot

        logits = logits.unsqueeze(1)                         # (T, B, 1, n_actions)
        if self.batch_first:
            logits = logits.permute(2, 0, 1, 3)             # (B, T, 1, n_actions)

        return logits, {}   # stateless model: return empty state dict

    def _init_forward_pass(self, inputs, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 2, 0, 3)
        inputs = inputs[:, 0]           # drop layer dim -> (T, B, features)
        inputs = inputs.nan_to_num(0.)

        T, B, _ = inputs.shape

        prev_actions = inputs[:, :, 0:self.n_actions].float()
        task_values  = inputs[:, :, self.n_actions].unsqueeze(-1).float()

        # trial_idx = inputs[:, :, self.IDX_TRIAL].float()    # (T, B)
        block_idx = inputs[:, :, -3].float()    # (T, B)

        # Fatigue signal: cumulative position in the experiment, normalised to [0, 1]
        # T_max   = float(self.N_BLOCKS * self.N_TRIALS_PER_BLOCK)
        # t_norms = (block_idx * self.N_TRIALS_PER_BLOCK + trial_idx) / T_max  # (T, B)
        t_norms = block_idx / self.n_blocks

        # Participant IDs are constant within a sequence; read from first timestep
        participant_ids = inputs[0, :, -1].long()   # (B,)

        logits    = torch.zeros(T, B, self.n_actions)
        timesteps = torch.arange(T)

        return (prev_actions, task_values, t_norms), participant_ids, logits, timesteps

    def count_parameters(self):
        """Free parameters per participant: beta_reward, beta_cost, beta_fatigue, bias x n_actions."""
        return 3 + self.n_actions


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
            additional_inputs=['difference', 'current', 'other'],
        ),
    )

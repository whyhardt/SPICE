import torch
import numpy as np
from typing import Union
from tqdm import tqdm

from spice import SpiceEstimator, SpiceDataset, BaseModel, csv_to_dataset, split_data_along_blockdim, dataset_to_csv

from weinhardt2026.studies.weber2024.spice_weber2024 import CONFIG


# --- Constants ---

CATCH_THRESHOLD = 10.0      # degrees — shield catches laser if angular distance <= this
MOVEMENT_SPEED = 1.0        # degrees per frame
INITIAL_SHIELD = 360.0      # degrees (shield starting position each block)
INITIAL_TOTAL_REWARD = 0.997
MISS_PENALTY = -0.003

# Additional input column offsets (relative to start of additional inputs block)
_AI_SHIELD_DIST = 0     # shield_distance_initial
_AI_SHIELD_ROT = 1      # shieldRotation
_AI_LASER_ROT = 2       # laserRotation
_AI_TRIAL_DUR = 3       # trial_duration_frames
_AI_TOTAL_MOVE = 4      # total_movement_degrees
_AI_FRAMES_MOVE = 5     # frames_spent_moving
_AI_BUTTON = 6          # button_press_onsets
_AI_RT = 7              # reaction_time_frames
_AI_CAUGHT = 8          # laser_caught


# --- Helper Functions ---

def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest circular distance in degrees between two angles.

    Handles values outside [0, 360) via modulo.
    """
    diff = (a - b) % 360
    return torch.min(diff, 360 - diff)


def move_toward(shield: torch.Tensor, laser: torch.Tensor, max_degrees: torch.Tensor) -> torch.Tensor:
    """Move shield toward laser by up to max_degrees along shortest arc.

    Returns new shield position (may be outside [0, 360)).
    """
    diff = (laser - shield % 360) % 360
    direction = torch.where(diff <= 180, torch.ones_like(diff), -torch.ones_like(diff))
    dist = angular_distance(shield, laser)
    movement = torch.min(max_degrees, dist)
    return shield + direction * movement


def compute_mean_rts(dataset: SpiceDataset) -> torch.Tensor:
    """Compute per-participant mean RT from move trials where RT > 0.

    Returns:
        Tensor of shape (n_participants,) indexed by remapped participant ID.
    """
    n_actions = dataset.n_actions
    ai_start = n_actions + dataset.n_reward_features
    rt_col = ai_start + _AI_RT

    xs = dataset.xs[:, :, 0, :]  # (sessions, trials, features)
    valid = ~torch.isnan(xs[:, :, 0])
    actions = xs[:, :, :n_actions].nan_to_num(0).argmax(dim=-1)
    rts = xs[:, :, rt_col]

    use_trial = (actions == 1) & valid & (rts > 0) & ~torch.isnan(rts)

    participant_ids = xs[:, 0, -1].long()
    n_participants = participant_ids.max().item() + 1

    mean_rts = torch.zeros(n_participants, dtype=torch.float32)
    for p in range(n_participants):
        p_mask = (participant_ids == p).unsqueeze(1).expand_as(use_trial) & use_trial
        if p_mask.any():
            mean_rts[p] = rts[p_mask].mean()

    return mean_rts


# --- Data Loading ---

def get_dataset(path_data: str = None, test_blocks: tuple[int] = None):

    if path_data is None:
        path_data = 'weinhardt2026/studies/weber2024/data/weber2024.csv'
    if test_blocks is None:
        test_blocks = 2, 5, 9, 14

    dataset = csv_to_dataset(
        file = path_data,
        df_participant_id='participant',
        df_experiment_id='experiment',
        df_choice='action',
        df_feedback=None,
        df_block='block',
        additional_inputs=CONFIG.additional_inputs,
    )
    
    if test_blocks is not None:
        dataset_train, dataset_test = split_data_along_blockdim(dataset, test_blocks)
    else:
        dataset_train, dataset_test = dataset, None

    return dataset_train, dataset_test


# --- Environment (simple data replay) ---

class EnvironmentWeber2024:
    """Replays laser beam trajectories and trial durations from the original dataset.

    No simulation logic — just provides per-trial laser positions and trial
    durations so the agent wrapper can simulate shield movement.
    """

    def __init__(self, dataset: SpiceDataset):
        n_sessions, n_trials = dataset.xs.shape[0], dataset.xs.shape[1]
        ai_start = dataset.n_actions + dataset.n_reward_features

        self.laser_rotations = dataset.xs[:, :, 0, ai_start + _AI_LASER_ROT].clone()
        self.trial_durations = dataset.xs[:, :, 0, ai_start + _AI_TRIAL_DUR].clone()
        self.participant_ids = dataset.xs[:, 0, 0, -1].long()

        self.n_sessions = n_sessions
        self.n_trials = n_trials
        self.trial_counter = 0

    def reset(self):
        """Reset trial counter for a new generation run."""
        self.trial_counter = 0

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return laser position and trial duration for the current trial.

        Returns:
            laser_pos: (n_sessions,) float tensor — laser angle in degrees.
            trial_duration: (n_sessions,) float tensor — frames until next laser.
        """
        t = self.trial_counter
        laser_pos = self.laser_rotations[:, t]
        trial_duration = self.trial_durations[:, t]
        self.trial_counter += 1
        return laser_pos, trial_duration


# --- Agent Wrapper (model + body constraints) ---

class Weber2024Agent:
    """Wraps a SPICE model with the physical constraints of the weber2024 task.

    Handles RT timing, shield movement simulation, forced stays, and
    catch/reward computation.  Works with any model that has standard SPICE I/O
    (forward(inputs, prev_state) → logits, state).
    """

    def __init__(
        self,
        model: Union[SpiceEstimator, torch.nn.Module],
        mean_rts: torch.Tensor,
        participant_ids: torch.Tensor,
        n_actions: int = 2,
        n_rewards: int = 0,
    ):
        if isinstance(model, SpiceEstimator):
            self.model = model.model
        else:
            self.model = model
        self.model.eval()

        self.n_actions = n_actions
        self.ai_start = n_actions + n_rewards  # start of additional inputs block
        self.mean_rts_per_session = mean_rts[participant_ids]  # (n_sessions,)

        # Resolve device
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        elif hasattr(self.model, 'parameters'):
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device('cpu')

        self.mean_rts_per_session = self.mean_rts_per_session.to(self.device)

    def reset(self, n_sessions: int):
        """Initialise agent state for a new generation run."""
        self.shield_positions = torch.full((n_sessions,), INITIAL_SHIELD, device=self.device)
        # rt_debt: NaN means no debt; a positive value is carried over from a forced stay
        self.rt_debt = torch.full((n_sessions,), float('nan'), device=self.device)
        self.prev_executed_move = torch.zeros(n_sessions, dtype=torch.bool, device=self.device)
        self.model_state = None

    @torch.no_grad()
    def step(
        self,
        predicted_action: torch.Tensor,
        laser_pos: torch.Tensor,
        trial_duration: torch.Tensor,
        xs_template_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one trial through the agent.

        Args:
            predicted_action: (n_sessions,) int — 0 = stay, 1 = move.
            laser_pos: (n_sessions,) float — current laser angle in degrees.
            trial_duration: (n_sessions,) float — frames in this trial.
            xs_template_t: (n_sessions, 1, 1, features) — metadata template for
                this trial (preserves time_trial, trials, block, experiment, participant).

        Returns:
            executed_action: (n_sessions,) long tensor.
            xs_t: (n_sessions, 1, 1, features) — full observation for this trial.
            logits_t: (n_sessions, n_actions) — model prediction for next action.
        """
        predicted_action = predicted_action.to(self.device)
        laser_pos = laser_pos.to(self.device)
        trial_duration = trial_duration.to(self.device)
        xs_t = xs_template_t.clone().to(self.device)

        n_sessions = predicted_action.shape[0]
        is_move = (predicted_action == 1)
        was_move = self.prev_executed_move
        has_debt = ~torch.isnan(self.rt_debt)

        # ── Phase 1: RT & movement ──────────────────────────────────────
        # Determine effective RT (either carried debt or fresh participant RT)
        effective_rt = torch.where(has_debt, self.rt_debt, self.mean_rts_per_session)

        time_to_move = torch.zeros(n_sessions, device=self.device)
        new_rt_debt = torch.full((n_sessions,), float('nan'), device=self.device)

        # Case 1: consecutive moves → no new RT, full duration available
        case1 = is_move & was_move
        time_to_move[case1] = trial_duration[case1]

        # Case 2: stay→move, enough time → normal reaction
        case2 = is_move & ~was_move & (trial_duration > effective_rt)
        time_to_move[case2] = trial_duration[case2] - effective_rt[case2]

        # Case 3: stay→move, insufficient time → forced stay, carry debt
        case3 = is_move & ~was_move & (trial_duration <= effective_rt)
        # time_to_move stays 0
        new_rt_debt[case3] = effective_rt[case3] - trial_duration[case3]

        # Case 4 (stay prediction): time_to_move=0, debt=NaN — handled by defaults

        self.rt_debt = new_rt_debt

        # Execute shield movement
        shield_before_move = self.shield_positions.clone()
        executed_move = is_move & (time_to_move > 0)
        max_movement = time_to_move * MOVEMENT_SPEED

        new_shield = move_toward(self.shield_positions, laser_pos, max_movement)
        self.shield_positions = torch.where(executed_move, new_shield, self.shield_positions)

        executed_action = torch.where(
            executed_move,
            torch.ones(n_sessions, dtype=torch.long, device=self.device),
            torch.zeros(n_sessions, dtype=torch.long, device=self.device),
        )

        # ── Phase 2: Catch ────────────────────────────────────────────────
        caught = angular_distance(self.shield_positions, laser_pos) <= CATCH_THRESHOLD

        # ── Phase 3: Build observation & run model ──────────────────────
        ai = self.ai_start
        actual_movement = torch.where(
            executed_move,
            angular_distance(shield_before_move, self.shield_positions),
            torch.zeros(n_sessions, device=self.device),
        )
        gen_rt = torch.zeros(n_sessions, device=self.device)
        gen_rt[case2] = effective_rt[case2]

        # Fill observation into xs_t (metadata in last 5 cols kept from template)
        action_oh = torch.nn.functional.one_hot(executed_action, self.n_actions).float()
        xs_t[:, 0, 0, :self.n_actions] = action_oh
        xs_t[:, 0, 0, ai + _AI_SHIELD_DIST] = angular_distance(shield_before_move, laser_pos)
        xs_t[:, 0, 0, ai + _AI_SHIELD_ROT] = self.shield_positions
        xs_t[:, 0, 0, ai + _AI_LASER_ROT] = laser_pos
        # trial_duration_frames (ai + _AI_TRIAL_DUR) kept from template
        xs_t[:, 0, 0, ai + _AI_TOTAL_MOVE] = actual_movement
        xs_t[:, 0, 0, ai + _AI_FRAMES_MOVE] = time_to_move
        xs_t[:, 0, 0, ai + _AI_BUTTON] = gen_rt
        xs_t[:, 0, 0, ai + _AI_RT] = gen_rt
        xs_t[:, 0, 0, ai + _AI_CAUGHT] = caught.float()

        # Forward pass through inner model
        logits, self.model_state = self.model(xs_t, self.model_state)

        # Normalise logits to (n_sessions, n_actions)
        if logits.dim() == 5:                          # BaseModel: (E, B, T, W, A)
            logits = logits.mean(dim=0)
        if logits.dim() == 4:                          # (B, T, W, A)
            logits_t = logits[:, -1, 0, :self.n_actions]
        else:                                          # (B, T, A)
            logits_t = logits[:, -1, :self.n_actions]

        # ── Phase 4: Update state ───────────────────────────────────────
        self.prev_executed_move = executed_move

        return executed_action, xs_t, logits_t


# --- Behavior Generation ---

@torch.no_grad()
def generate_behavior(
    model: Union[SpiceEstimator, torch.nn.Module],
    dataset: SpiceDataset,
    save_dataset: str = None,
) -> SpiceDataset:
    """Generate synthetic behavioral data by running a model through the weber2024 task.

    Replays the original laser trajectories and simulates shield movement based
    on the model's predictions, with RT timing and forced-stay constraints.

    Args:
        model: Fitted SPICE model (SpiceEstimator or nn.Module).
        dataset: Original dataset (structure + laser trajectories + trial durations).
        save_dataset: Optional path to save generated dataset as CSV.

    Returns:
        SpiceDataset with generated behavior.
    """
    print("Generating behavior...")

    n_sessions, n_trials, _, n_features = dataset.xs.shape
    n_actions = dataset.n_actions
    n_rewards = dataset.n_reward_features

    # Build environment and agent
    env = EnvironmentWeber2024(dataset)
    mean_rts = compute_mean_rts(dataset)
    agent = Weber2024Agent(
        model, mean_rts, env.participant_ids,
        n_actions=n_actions, n_rewards=n_rewards,
    )

    # Initialise
    env.reset()
    agent.reset(n_sessions)

    xs_gen = dataset.xs.clone().to(agent.device)
    ys_gen = torch.zeros_like(dataset.ys, device=agent.device)
    valid_mask = ~torch.isnan(dataset.xs[:, :, 0, 0])  # (n_sessions, n_trials)

    action_idx = torch.zeros(n_sessions, dtype=torch.long, device=agent.device)

    for t in tqdm(range(n_trials)):
        laser_pos, trial_duration = env.step()
        xs_template_t = xs_gen[:, t:t + 1].clone()  # (B, 1, 1, F) — preserves metadata

        executed_action, xs_t, logits_t = agent.step(
            action_idx, laser_pos, trial_duration, xs_template_t,
        )

        # Record observation
        xs_gen[:, t:t + 1] = xs_t

        # Zero out inactive sessions
        inactive = ~valid_mask[:, t]
        xs_gen[inactive, t] = 0

        # Sample next action
        probs = torch.softmax(logits_t, dim=-1)
        action_idx = torch.multinomial(probs, 1).squeeze(-1)

        # Store target (next-action prediction)
        ys_gen[:, t, 0, :] = torch.nn.functional.one_hot(action_idx, n_actions).float()

    # Restore NaN padding for inactive trials
    for t in range(n_trials):
        inactive = ~valid_mask[:, t]
        xs_gen[inactive, t] = float('nan')
        ys_gen[inactive, t] = float('nan')

    xs_gen = xs_gen.cpu()
    ys_gen = ys_gen.cpu()

    dataset_gen = SpiceDataset(xs_gen, ys_gen, n_reward_features=n_rewards)

    if save_dataset is not None:
        dataset_to_csv(
            dataset_gen, save_dataset,
            df_choice='action',
            additional_inputs=[
                'shield_distance_initial', 'shieldRotation', 'laserRotation',
                'trial_duration_frames', 'total_movement_degrees', 'frames_spent_moving',
                'button_press_onsets', 'reaction_time_frames', 'laser_caught',
                'volatility', 'stochasticity',
            ],
        )

    print("Done generating behavior.")
    return dataset_gen

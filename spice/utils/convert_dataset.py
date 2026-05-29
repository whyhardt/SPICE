from typing import List, Optional, Union, Iterable

import numpy as np
import pandas as pd
import torch
import warnings

from ..resources.spice_utils import SpiceDataset


# ============================================================================
# Internal helpers — csv_to_dataset
# ============================================================================

def _normalize_ids(df: pd.DataFrame, column: str) -> None:
    """Map column values to 0-indexed integers in place. Create column = 0 if absent."""
    if column not in df.columns:
        df[column] = 0
    else:
        id_map = {val: idx for idx, val in enumerate(df[column].unique())}
        df[column] = df[column].map(id_map)


def _prepare_choices(df: pd.DataFrame, df_choice: str) -> int:
    """Normalize choice column to 0-based numeric values in place. Returns n_actions."""
    if isinstance(df[df_choice].iloc[0], str):
        choices = df[df_choice].astype('category').cat.codes.values.copy().astype(float)
        choices[choices == -1] = np.nan
        mapping = tuple((i, k) for i, k in enumerate(np.sort(df[df_choice].dropna().unique())))
        warnings.warn(
            f"Choice column '{df_choice}' converted from str to int (alphabetical): {mapping}"
        )
    else:
        choices = df[df_choice].values.astype(float)

    valid = ~np.isnan(choices)
    choices[valid] -= np.nanmin(choices)
    df[df_choice] = choices
    return int(np.nanmax(choices) + 1)


def _validate_additional_inputs(
    df: pd.DataFrame,
    additional_inputs: Optional[Union[str, Iterable[str]]],
) -> tuple[list[str], int]:
    """Validate column existence and convert non-numeric values. Returns (cleaned list, count)."""
    if not additional_inputs:
        return [], 0
    additional_inputs = list(additional_inputs)

    missing = [a for a in additional_inputs if a not in df.columns]
    if missing:
        warnings.warn(f"Additional inputs {missing} not found in dataset. Ignored.")
        for m in missing:
            additional_inputs.remove(m)

    for col in additional_inputs:
        val = df[col].values[0]
        if isinstance(val, str) and not val.isdigit():
            id_map = {v: i for i, v in enumerate(df[col].unique())}
            df[col] = df[col].map(id_map)
            print(f"Non-numerical column '{col}' converted: {id_map}")

    return additional_inputs, len(additional_inputs)


def _build_reward_matrix(
    group_df: pd.DataFrame,
    df_feedback: tuple[str, ...],
    n_actions: int,
    choice_vals: np.ndarray,
) -> np.ndarray:
    """Build (n_rows, n_actions) reward matrix.

    Partial feedback: reward placed in chosen action's column, others NaN.
    Full/counterfactual: each column maps directly to its action index.
    """
    n_rows = len(group_df)
    rewards = np.full((n_rows, n_actions), np.nan)
    for col_idx, col_name in enumerate(df_feedback):
        vals = group_df[col_name].values
        if len(df_feedback) > 1:
            rewards[:, col_idx] = vals
        else:
            for i in range(n_rows):
                rewards[i, int(choice_vals[i])] = vals[i]
    return rewards


def _apply_timeshift(
    xs: np.ndarray, ys: np.ndarray, shifts: Iterable, col_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift additional-input columns along the outer-trial dimension (dim=1)."""
    if isinstance(shifts, int):
        shifts = (shifts,)
    for idx, shift in enumerate(shifts):
        col = col_start + idx
        if shift == -1:
            xs[:, :-1, :, col] = xs[:, 1:, :, col]
        elif shift == 1:
            xs[:, 1:, :, col] = xs[:, :-1, :, col]
            xs[:, 0, :, col] = 0
    if any(s == '-' for s in shifts):
        xs, ys = xs[:, :-1], ys[:, :-1]
    return xs, ys


def _build_arrays_standard(
    df, groupby_kw, df_choice, df_feedback, additional_inputs,
    n_actions, n_reward_cols, n_features, n_groups, max_trials,
    remove_failed_trials,
):
    """Build 4D arrays for trial-level data (within_ts=1)."""
    xs = np.full((n_groups, max_trials, 1, n_features), np.nan)
    ys = np.full((n_groups, max_trials, 1, n_actions), np.nan)

    for idx, (_, group_df) in enumerate(df.groupby(groupby_kw)):
        if remove_failed_trials:
            group_df = group_df[~np.isnan(group_df[df_choice])]

        choice_vals = group_df[df_choice].values.astype(int)
        choice_onehot = np.eye(n_actions)[choice_vals]
        n = len(choice_vals) - 1  # exclude last for next-action shift

        # Data (trials 0..n-1)
        xs[idx, :n, 0, :n_actions] = choice_onehot[:-1]
        if df_feedback is not None:
            rewards = _build_reward_matrix(group_df, df_feedback, n_actions, choice_vals)
            xs[idx, :n, 0, n_actions:n_actions + n_reward_cols] = rewards[:-1]
        for ai, name in enumerate(additional_inputs):
            xs[idx, :n, 0, n_actions + n_reward_cols + ai] = group_df[name].values[:-1]

        # Metadata (all slots including padding)
        xs[idx, :, 0, -5] = 0                                       # time_trial
        xs[idx, :, 0, -4] = np.arange(max_trials)                   # trial index
        xs[idx, :, 0, -3] = group_df[groupby_kw[2]].values[0]      # block
        xs[idx, :, 0, -2] = group_df[groupby_kw[1]].values[0]      # experiment
        xs[idx, :, 0, -1] = group_df[groupby_kw[0]].values[0]      # participant

        ys[idx, :n, 0] = choice_onehot[1:]

    return xs, ys


def _build_arrays_within_trial(
    df, groupby_kw, df_trial, df_choice, df_feedback, df_time,
    additional_inputs, n_actions, n_reward_cols, n_features,
    n_groups, remove_failed_trials,
):
    """Build 4D arrays with within-trial timesteps grouped by df_trial."""
    groupby_trial_kw = groupby_kw + [df_trial]
    max_within_ts = df.groupby(groupby_trial_kw).size().max()
    max_outer_ts = df.groupby(groupby_kw)[df_trial].nunique().max()

    xs = np.full((n_groups, max_outer_ts, max_within_ts, n_features), np.nan)
    ys = np.full((n_groups, max_outer_ts, max_within_ts, n_actions), np.nan)

    for idx, (_, group_df) in enumerate(df.groupby(groupby_kw)):
        trials = [t for _, t in group_df.groupby(df_trial, sort=True)]
        if remove_failed_trials:
            trials = [t for t in trials if not np.isnan(t[df_choice].values.astype(float)).any()]

        # Metadata (all slots)
        xs[idx, :, :, -5] = 0                                         # time_trial default
        xs[idx, :, :, -4] = np.arange(max_outer_ts)[:, None]          # trial index
        xs[idx, :, :, -3] = group_df[groupby_kw[2]].values[0]        # block
        xs[idx, :, :, -2] = group_df[groupby_kw[1]].values[0]        # experiment
        xs[idx, :, :, -1] = group_df[groupby_kw[0]].values[0]        # participant

        # Data (exclude last trial for next-action shift)
        for t_idx in range(len(trials) - 1):
            trial_df = trials[t_idx]
            n_w = len(trial_df)
            choice_vals = trial_df[df_choice].values.astype(int)

            xs[idx, t_idx, :n_w, :n_actions] = np.eye(n_actions)[choice_vals]
            if df_feedback is not None:
                xs[idx, t_idx, :n_w, n_actions:n_actions + n_reward_cols] = \
                    _build_reward_matrix(trial_df, df_feedback, n_actions, choice_vals)
            for ai, name in enumerate(additional_inputs):
                xs[idx, t_idx, :n_w, n_actions + n_reward_cols + ai] = trial_df[name].values

            # Within-trial time
            if df_time is not None:
                xs[idx, t_idx, :n_w, -5] = trial_df[df_time].values.astype(float)
            elif n_w > 1:
                xs[idx, t_idx, :n_w, -5] = np.linspace(0, 1, n_w)

            # Target: next trial's choice (replicated across within-trial timesteps)
            next_choice = int(trials[t_idx + 1][df_choice].values[0])
            ys[idx, t_idx, :n_w] = np.eye(n_actions)[next_choice]

    return xs, ys


# ============================================================================
# Internal helpers — dataset_to_csv
# ============================================================================

def _infer_feature_layout(dataset: SpiceDataset) -> tuple[int, int, int]:
    """Return (n_actions, n_reward_cols, n_additional_inputs) from dataset shape."""
    n_features = dataset.xs.shape[-1]
    n_actions = dataset.ys.shape[-1]
    n_trailing = 5
    if hasattr(dataset, 'n_reward_features') and dataset.n_reward_features is not None:
        n_reward_cols = dataset.n_reward_features
    else:
        remaining = n_features - n_actions - n_trailing
        n_reward_cols = n_actions if remaining >= n_actions else 0
    return n_actions, n_reward_cols, n_features - n_actions - n_reward_cols - n_trailing


def _detect_full_feedback(xs: np.ndarray, n_actions: int, n_reward_cols: int) -> bool:
    """True if unchosen actions have non-NaN feedback (counterfactual / full)."""
    if n_reward_cols == 0:
        return False
    for g in range(xs.shape[0]):
        valid = ~np.isnan(xs[g, :, 0, 0])
        if not np.any(valid):
            continue
        first = int(np.argmax(valid))
        chosen = int(np.argmax(xs[g, first, 0, :n_actions]))
        rewards = xs[g, first, 0, n_actions:n_actions + n_reward_cols]
        for a in range(n_actions):
            if a != chosen and not np.isnan(rewards[a]):
                return True
        break
    return False


def _resolve_additional_input_names(provided: Optional[List[str]], n_expected: int) -> Optional[List[str]]:
    """Validate or auto-generate additional-input column names for CSV export."""
    if n_expected == 0:
        return None
    generic = [f'additional_input_{i}' for i in range(n_expected)]
    if provided is None:
        return generic
    if len(provided) != n_expected:
        warnings.warn(
            f"additional_inputs length ({len(provided)}) != dataset ({n_expected}). Using generic names."
        )
        return generic
    return provided


# ============================================================================
# Public API
# ============================================================================

def csv_to_dataset(
    file: str,
    df_participant_id: str = 'participant',
    df_block: str = 'block',
    df_experiment_id: str = 'experiment',
    df_choice: str = 'choice',
    df_feedback: Optional[Union[str, Iterable[str]]] = 'reward',
    df_time: Optional[str] = None,
    df_trial: Optional[str] = None,
    additional_inputs: Optional[Union[str, Iterable[str]]] = None,
    device=None,
    sequence_length: int = None,
    timeshift_additional_inputs: Optional[Iterable[int]] = None,
    remove_failed_trials: bool = True,
) -> SpiceDataset:
    """Convert a behavioural CSV file to a SpiceDataset.

    Output shape is 4D: ``(sessions, outer_ts, within_ts, features)``.
    Feature layout: ``[actions_onehot, rewards_per_action, *additional_inputs, time_trial,
    trial, block, experiment, participant]``

    Args:
        file: Path to CSV file.
        df_participant_id: Column name for participant IDs.
        df_block: Column name for block IDs.
        df_experiment_id: Column name for experiment IDs.
        df_choice: Column name for choice values.
        df_feedback: Column name(s) for reward/feedback. ``None`` to omit.
        additional_inputs: Column name(s) for extra input signals.
        df_time: Column for within-trial time values. Requires ``df_trial``.
        df_trial: Column grouping rows into outer trials (enables within-trial
            timesteps). Without ``df_time``, linearly spaced values are assigned.
        device: Torch device for the returned dataset.
        sequence_length: Optional BPTT truncation length.
        timeshift_additional_inputs: Per-input shift direction (``-1``, ``0``, or ``1``).
        remove_failed_trials: Drop trials containing NaN choices.

    Returns:
        SpiceDataset with shape ``(sessions, outer_ts, within_ts, features)``.
    """
    if (timeshift_additional_inputs is not None
        and additional_inputs is not None
        and len(timeshift_additional_inputs) != len(additional_inputs)):
        raise ValueError(
            f"timeshift_additional_inputs length ({len(timeshift_additional_inputs)}) "
            f"!= additional_inputs length ({len(additional_inputs)})"
        )

    df = pd.read_csv(file, index_col=None)

    # Normalize ID columns to 0-indexed integers
    _normalize_ids(df, df_participant_id)
    _normalize_ids(df, df_experiment_id)
    if df_block not in df.columns:
        df[df_block] = 0

    groupby_kw = [df_participant_id, df_experiment_id, df_block]
    n_groups = df.groupby(groupby_kw).ngroups
    max_trials = df.groupby(groupby_kw).size().max()

    n_actions = _prepare_choices(df, df_choice)
    additional_inputs, n_additional = _validate_additional_inputs(df, additional_inputs)

    if df_feedback is not None:
        df_feedback = (df_feedback,) if isinstance(df_feedback, str) else tuple(df_feedback)
    n_reward_cols = n_actions if df_feedback is not None else 0
    n_features = n_actions + n_reward_cols + n_additional + 5

    # Build 4D arrays: (sessions, outer_ts, within_ts, features)
    if df_trial is not None:
        xs, ys = _build_arrays_within_trial(
            df, groupby_kw, df_trial, df_choice, df_feedback, df_time,
            additional_inputs, n_actions, n_reward_cols, n_features,
            n_groups, remove_failed_trials,
        )
    else:
        xs, ys = _build_arrays_standard(
            df, groupby_kw, df_choice, df_feedback, additional_inputs,
            n_actions, n_reward_cols, n_features, n_groups, max_trials,
            remove_failed_trials,
        )

    if timeshift_additional_inputs is not None:
        xs, ys = _apply_timeshift(xs, ys, timeshift_additional_inputs, n_actions + n_reward_cols)

    return SpiceDataset(xs, ys, device=device, sequence_length=sequence_length, n_reward_features=n_reward_cols)


def dataset_to_csv(
    dataset: SpiceDataset,
    path: str,
    df_participant_id: str = 'participant',
    df_experiment_id: str = 'experiment',
    df_block: str = 'block',
    df_choice: str = 'choice',
    df_feedback: str = 'reward',
    additional_inputs: Optional[List[str]] = None,
) -> None:
    """Write a SpiceDataset to a CSV file.

    Reconstructs the tabular representation from the 4D tensor layout
    ``(sessions, outer_ts, within_ts, features)``. NaN-padded slots are skipped.
    """
    xs = dataset.xs.cpu().numpy() if hasattr(dataset.xs, 'cpu') else np.array(dataset.xs)
    ys = dataset.ys.cpu().numpy() if hasattr(dataset.ys, 'cpu') else np.array(dataset.ys)
    n_groups, _, n_within_ts, _ = xs.shape

    n_actions, n_reward_cols, n_additional = _infer_feature_layout(dataset)
    additional_inputs = _resolve_additional_input_names(additional_inputs, n_additional)
    is_full = _detect_full_feedback(xs, n_actions, n_reward_cols)

    rows = []
    for g in range(n_groups):
        session_meta = {
            df_participant_id: int(xs[g, 0, 0, -1]),
            df_experiment_id: int(xs[g, 0, 0, -2]),
            df_block: int(xs[g, 0, 0, -3]),
        }
        n_valid = int((~np.isnan(xs[g, :, 0, 0])).sum())

        for t in range(n_valid):
            for w in range(n_within_ts):
                if np.isnan(xs[g, t, w, 0]):
                    continue
                choice = int(np.argmax(xs[g, t, w, :n_actions]))
                row = {
                    **session_meta,
                    'trial': int(xs[g, t, w, -4]),
                    'timestep': float(xs[g, t, w, -5]),
                    df_choice: choice,
                }
                if n_reward_cols > 0:
                    rewards = xs[g, t, w, n_actions:n_actions + n_reward_cols]
                    if is_full:
                        for a in range(n_actions):
                            row[f'{df_feedback}_{a}'] = rewards[a]
                    else:
                        row[df_feedback] = rewards[choice] if not np.isnan(rewards[choice]) else np.nan
                if additional_inputs is not None:
                    for i, name in enumerate(additional_inputs):
                        row[name] = xs[g, t, w, n_actions + n_reward_cols + i]
                rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================================
# Dataset splitting and reshaping utilities
# ============================================================================

def split_data_along_timedim(dataset: SpiceDataset, split_ratio: float, device: torch.device = torch.device('cpu')):
    """Split the data along the time dimension (dim=1).
    Each session (dim=0) can be of individual length and is therefore post-padded with -1.
    To split the data into training and testing samples according to the split_ratio each session's individual length has to be considered.
    E.g.:
    1. session_length = len(session 0) -> 120
    2. split_index = int(split_ratio * session length) -> 96
    3. samples for training data -> 96
    4. samples for testing data -> 24 = 120 - 96

    Args:
        data (torch.Tensor): Data containing all sessions in the shape (session, time, features)
        split_ratio (float): Float number indicating the ratio to be used as training data

    Returns:
        tuple(DatasetRNN, DatasetRNN): Training data and testing data splitted along time dimension (dim=1)
    """

    dim = 1
    xs, ys = dataset.xs, dataset.ys

    # Create a mask of non-zero elements
    non_nan_mask = 1-np.isnan(xs[..., 0]).int()

    # Find cumulative sum along the specified dimension in reverse order
    cumsum = torch.cumsum(non_nan_mask, dim)

    # Find the index where the cumulative sum becomes 1 in the reversed array
    last_nonzero_indices = torch.argmax(cumsum, dim=dim)

    # compute the indeces at which the data is going to be splitted into training and testing data
    split_indices = (last_nonzero_indices * split_ratio).int()

    # initialize training data and testing data storages with NaN padding
    # (model loss masking uses torch.isnan() to detect padding)
    train_xs = torch.full((xs.shape[0], max(split_indices), xs.shape[2], xs.shape[3]), float('nan'), device=dataset.device)
    test_xs = torch.full((xs.shape[0], max(last_nonzero_indices - split_indices), xs.shape[2], xs.shape[3]), float('nan'), device=dataset.device)
    train_ys = torch.full((xs.shape[0], max(split_indices), ys.shape[2], ys.shape[3]), float('nan'), device=dataset.device)
    test_ys = torch.full((xs.shape[0], max(last_nonzero_indices - split_indices), ys.shape[2], ys.shape[3]), float('nan'), device=dataset.device)

    # fill up training and testing data
    for index_session in range(xs.shape[0]):
        train_xs[index_session, :split_indices[index_session]] = xs[index_session, :split_indices[index_session]]
        test_xs[index_session, :last_nonzero_indices[index_session]-split_indices[index_session]] = xs[index_session, split_indices[index_session]:last_nonzero_indices[index_session]]
        train_ys[index_session, :split_indices[index_session]] = ys[index_session, :split_indices[index_session]]
        test_ys[index_session, :last_nonzero_indices[index_session]-split_indices[index_session]] = ys[index_session, split_indices[index_session]:last_nonzero_indices[index_session]]

        # these columns (Timesteps_in_trial, Trials, Blocks, Experiments, Participants) are getting filled up along the trial dimension
        train_xs[index_session, :, :, -5:] = xs[index_session, :train_xs.shape[1], :, -5:]
        test_xs[index_session, :, :, -5:] = xs[index_session, :test_xs.shape[1], :, -5:]

    return SpiceDataset(train_xs, train_ys, device=device), SpiceDataset(test_xs, test_ys, device=device)


def split_data_along_sessiondim(dataset: SpiceDataset, test_sessions: list[int] = None, device: torch.device = torch.device('cpu')):
    """Split the data along the time dimension (dim=1).
    Each session (dim=0) can be of individual length and is therefore post-padded with -1.
    To split the data into training and testing samples according to the split_ratio each session's individual length has to be considered.
    E.g.:
    1. session_length = len(session 0) -> 120
    2. split_index = int(split_ratio * session length) -> 96
    3. samples for training data -> 96
    4. samples for testing data -> 24 = 120 - 96

    Args:
        data (torch.Tensor): Data containing all sessions in the shape (session, time, features)
        split_ratio (float): Float number indicating the ratio to be used as training data

    Returns:
        tuple(DatasetRNN, DatasetRNN): Training data and testing data splitted along time dimension (dim=1)
    """

    if test_sessions is not None:

        dim = 1
        xs, ys = dataset.xs.cpu(), dataset.ys.cpu()

        # get participant ids
        participants_ids = xs[:, 0, 0, -1].unique()

        # get sessions ids
        session_ids = xs[:, 0, 0, -3].unique()

        # set training sessions
        if test_sessions:
            session_ids_test = torch.tensor(test_sessions, dtype=torch.float32)
        else:
            session_ids_test = torch.tensor([])

        if all(session_ids[:-1] < 1):
            session_ids_test /= len(session_ids) - 1

        # build list of training sessions (only includes sessions that exist in data)
        session_ids_train = []
        for sid in session_ids:
            if not sid in session_ids_test:
                session_ids_train.append(sid)
        session_ids_train = torch.tensor(session_ids_train, dtype=torch.float32)

        # setup new variables - use lists to collect only existing participant-session combinations
        train_xs_list, test_xs_list, train_ys_list, test_ys_list = [], [], [], []

        for pid in participants_ids:
            for sid in session_ids:
                mask_ids = torch.logical_and(xs[:, 00, 0, -3] == sid, xs[:, 0, 0, -1] == pid)
                if mask_ids.max():
                    if sid in session_ids_train:
                        train_xs_list.append(xs[mask_ids])
                        train_ys_list.append(ys[mask_ids])
                    elif sid in session_ids_test:
                        test_xs_list.append(xs[mask_ids])
                        test_ys_list.append(ys[mask_ids])
                    else:
                        raise ValueError("session id was not found in training nor test sessions.")

        # stack into tensors - only includes actual data, no empty rows
        train_xs = torch.cat(train_xs_list, dim=0) if train_xs_list else torch.zeros((0, *xs.shape[1:]))
        train_ys = torch.cat(train_ys_list, dim=0) if train_ys_list else torch.zeros((0, *ys.shape[1:]))
        test_xs = torch.cat(test_xs_list, dim=0) if test_xs_list else torch.zeros((0, *xs.shape[1:]))
        test_ys = torch.cat(test_ys_list, dim=0) if test_ys_list else torch.zeros((0, *ys.shape[1:]))

        return SpiceDataset(train_xs, train_ys, device=device, n_reward_features=dataset.n_reward_features), SpiceDataset(test_xs, test_ys, device=device, n_reward_features=dataset.n_reward_features)

    else:
        return dataset, dataset


def reshape_data_along_participantdim(dataset: SpiceDataset, device: torch.device = torch.device('cpu')):
    """Reshape the data along the participant dim.

    Args:
        dataset (DatasetRNN): current dataset of shape (participants*sessions*, trial, features)
        device (torch.device, optional): Defaults to torch.device('cpu').

    Returns:
        DatasetRNN: restructured dataset with shape (participant, session, trial, features)
    """

    xs, ys = dataset.xs.cpu(), dataset.ys.cpu()

    # get participant ids
    participants_ids = xs[:, 0, -1].unique()

    # setup new variables
    xs_new, ys_new = [], []
    n_sessions_max = 0

    # First pass: collect participant-level sessions and find max number of sessions
    for pid in participants_ids:
        mask_ids = xs[:, 0, -1] == pid
        xs_new.append(xs[mask_ids])
        ys_new.append(ys[mask_ids])
        n_sessions_max = max(n_sessions_max, len(xs[mask_ids]))

    # Second pass: pad all participants to have n_sessions_max sessions
    for i in range(len(xs_new)):
        if xs_new[i].shape[0] < n_sessions_max:
            # Create padding for missing sessions
            n_missing = n_sessions_max - xs_new[i].shape[0]
            pad_sessions_xs = torch.zeros((n_missing, *xs_new[i].shape[1:]), device=xs_new[i].device)
            # Set feature dimensions to -1 (except last 3 which are IDs)
            pad_sessions_xs[..., :-3] = -1
            # Set participant ID to match this participant
            pad_sessions_xs[..., -1] = xs_new[i][0, 0, -1]
            # Concatenate padding
            xs_new[i] = torch.cat((xs_new[i], pad_sessions_xs), dim=0)

            # Pad ys with -1
            pad_sessions_ys = torch.full((n_missing, *ys_new[i].shape[1:]), -1.0, device=ys_new[i].device)
            ys_new[i] = torch.cat((ys_new[i], pad_sessions_ys), dim=0)

    # setup new dataset with shape (participant, session, trial, features)
    return SpiceDataset(torch.stack(xs_new), torch.stack(ys_new), device=device)

import sys
import os

from typing import List, Optional

import pandas as pd
import numpy as np
import torch
import warnings
from copy import copy

from ..resources.spice_utils import SpiceDataset


def csv_to_dataset(
    file: str,
    device = None,
    sequence_length: int = None,
    df_participant_id: str = 'participant',
    df_block: str = 'block',
    df_experiment_id: str = 'experiment',
    df_choice: str = 'choice',
    df_feedback: str = 'reward',
    additional_inputs: List[str] = None,
    timeshift_additional_inputs: bool = False,
    remove_failed_trials: bool = True,
    ) -> SpiceDataset:
    df = pd.read_csv(file, index_col=None)
    
    # replace all nan values with -1
    df = df.replace(np.nan, -1)
    
    original_df = copy(df)
    
    groupby_kw = [df_participant_id, df_experiment_id, df_block]
    
    # convert all participant ids to numeric values if necessary
    # if not pd.to_numeric(df[df_participant_id], errors='coerce').notna().all():
    # Map unique participant IDs to numeric values
    id_map = {pid: idx for idx, pid in enumerate(df[df_participant_id].unique())} #this line
    # Replace participant IDs with numeric values
    df[df_participant_id] = df[df_participant_id].map(id_map)    
    
    if not df_experiment_id in df.columns:
        df[df_experiment_id] = 0
    else:
        # convert all experiment ids to numeric values if necessary
        # if not all(pd.to_numeric(df[df_experiment_id], errors='coerce').notna().all()):
        # Map unique participant IDs to numeric values
        id_map = {eid: idx for idx, eid in enumerate(df[df_experiment_id].unique())}
        # Replace experiment IDs with numeric values
        df[df_experiment_id] = df[df_experiment_id].map(id_map)
    
    if not df_block in df.columns:
        df[df_block] = 0
        original_df[df_block] = 0
    # else:
    #     # normalize block numbers
    #     blocks_max = df[df_block].max()
    #     blocks_min = df[df_block].min()
    #     df[df_block] = (df[df_block] - blocks_min) / (blocks_max - blocks_min)
        
    # get maximum number of trials per participant
    n_groups = len(df.groupby(groupby_kw).size())
    max_trials = df.groupby(groupby_kw).size().max()
    
    # let actions begin from 0
    if isinstance(df[df_choice].iloc[0], str):
        choices = df[df_choice].astype('category').cat.codes.values.copy()
        print(f"ValueWarning: Values from choice column ({df_choice}) had to be converted from str to int. The mapping is sorted alphabetically: {tuple([(index, key) for index, key in enumerate(np.sort(df[df_choice].unique()))])}")
    else:
        choices = df[df_choice].values
    choice_min = np.nanmin(choices[choices != -1])
    choices[choices != -1] = choices[choices != -1] - choice_min
    df[df_choice] = choices
    original_df[df_choice] = choices
    
    # number of possible actions
    n_actions = int(df[df_choice].max() + 1)
    
    # get all columns with rewards
    if df_feedback + '_0' in df.columns:
        # counterfactual dataset
        reward_cols = []
        for reward_column in df.columns:
            if df_feedback in reward_column:
                reward_cols.append(reward_column)
    else:
        reward_cols = [df_feedback]
    
    # normalize rewards (exclude -1 which represents missing/NaN values)
    r_min, r_max = [], []
    for reward_column in reward_cols:
        valid_rewards = df[reward_column][df[reward_column] != -1]
        if len(valid_rewards) > 0:
            r_min.append(valid_rewards.min())
            r_max.append(valid_rewards.max())
    r_min = np.min(r_min) if r_min else 0
    r_max = np.max(r_max) if r_max else 1
    # Avoid division by zero if all rewards are the same
    r_range = r_max - r_min if r_max != r_min else 1
    for reward_column in reward_cols:
        # Only normalize valid rewards, keep -1 as-is (will be handled as NaN later)
        mask = df[reward_column] != -1
        df.loc[mask, reward_column] = (df.loc[mask, reward_column] - r_min) / r_range
    
    # check whether the additional inputs are in the dataset
    additional_inputs_xs = 0
    additional_input_not_found = []
    if isinstance(additional_inputs, list) or isinstance(additional_inputs, tuple):
        for additional_input in additional_inputs:
            if additional_input in df.columns:
                additional_inputs_xs += 1
            else:
                additional_input_not_found.append(additional_input)

    if len(additional_input_not_found) > 0:
        warnings.warn(f"Some additional input signals ({additional_input_not_found}) were not found in the dataset. These signals will be ignored.")
        for not_found in additional_input_not_found:
            additional_inputs.remove(not_found)
    
    # convert non-numerical items in additional inputs
    if additional_inputs:
        for index, additional in enumerate(additional_inputs):
            if isinstance(df[additional].values[0], str) and not df[additional].values[0].isdigit():
                # Map unique participant IDs to numeric values
                id_map = {pid: idx for idx, pid in enumerate(df[additional].unique())} #this line
                # Replace participant IDs with numeric values
                df[additional] = df[additional].map(id_map)    
                
                print(f"Values in column {additional} in dataset {file} are not numerical. Converted values are:")
                print(id_map)
            
    n_ids = 0
    if df_participant_id is not None:
        n_ids += 1
    if df_experiment_id is not None:
        n_ids += 1
        
    # Use NaN for padding instead of -1 for cleaner handling in RNN computations
    xs = np.full((n_groups, max_trials, n_actions*2 + n_ids + additional_inputs_xs + 1 if df_block is not None else 0), np.nan)
    ys = np.full((n_groups, max_trials, n_actions), np.nan)

    # for index_group, participant_id in enumerate(participant_ids):
    for index_group, group in enumerate(df.groupby(groupby_kw)):

        # Filter out failed trials (where choice == -1) if requested
        group_df = group[1]
        if remove_failed_trials:
            valid_trials_mask = group_df[df_choice] != -1
            group_df = group_df[valid_trials_mask]

        choice = np.eye(n_actions)[group_df[df_choice].values.astype(int)]
        rewards = np.full((len(choice), n_actions), np.nan)
        for index_column, reward_column in enumerate(reward_cols):
            # add reward into the correct column
            reward = group_df[reward_column].values
            if len(reward_cols) > 1:
                # counterfactual data
                rewards[:, index_column] = reward
            else:
                for index_choice, ch in enumerate(choice):
                    rewards[index_choice, np.argmax(ch)] = reward[index_choice]

        # write arrays for DatasetRNN
        # IMPORTANT: Only include timesteps that have valid targets (exclude last timestep)
        # xs[t] should predict ys[t] = choice[t+1]
        # So we only include choice[:-1] in xs to match ys which contains choice[1:]
        xs[index_group, :len(choice)-1, :n_actions] = choice[:-1]
        xs[index_group, :len(choice)-1, n_actions:n_actions*2] = rewards[:-1]

        # add all grouping variables to the end (use original group for ID values)
        xs[index_group, :, -3] = group[1][df_block].values[0]
        xs[index_group, :, -2] = group[1][df_experiment_id].values[0]
        xs[index_group, :, -1] = group[1][df_participant_id].values[0]
        ys[index_group, :len(choice)-1] = choice[1:]

        # write additional inputs after choices and rewards and before grouping variables
        if additional_inputs is not None and len(additional_inputs) > 0:
            for index, additional_input in enumerate(additional_inputs):
                xs[index_group, :len(choice)-1, n_actions*2+index] = group_df[additional_input].values[:-1]
        
    # move additional inputs one timestep back in order to make the next decision being based on them
    if timeshift_additional_inputs:
        xs[:, :-1, n_actions*2:-3] = xs[:, 1:, n_actions*2:-3]
        xs = xs[:, :-1]
        ys = ys[:, :-1]
    
    dataset = SpiceDataset(xs, ys, device=device, sequence_length=sequence_length)
    
    return dataset


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
    """Writes a SpiceDataset object to a csv file.

    SpiceDataset has the shape (n_participants * n_experiments * n_blocks, n_trials, features).
    The features are in the order (action_0, action_1, ..., action_m, feedback_0, feedback_1, ..., feedback_m, *additional_inputs, block_id, experiment_id, participant_id)
    The action is one-hot-encoded.
    The feedback can be provided fully (feedback observed for all actions including unchosen ones) or partially (feedback observed only for chosen action; feedback values for unchosen values are torch.nan).
    A pandas.DataFrame object has to be created of the shape (n_participants * n_experiments * n_blocks * n_trials, features_dataframe)
    features_dataframe has the shape (participant, experiment, block, action, feedback, *additional_inputs)
    For full feedback each feedback feature from SpiceDataset is written into the pandas.DataFrame by extending df_feedback with f'_{index_action}'
    For partial feedback only the feedback value of the chosen action is written into the pandas.DataFrame; df_feedback is taken as given.

    Note: Due to the time-shifted structure of SpiceDataset (xs[t] predicts ys[t] = choice[t+1]),
    only the trials stored in xs are output. The final choice (in ys) is the prediction target
    for the last xs entry and is reconstructed when loading with csv_to_dataset.
    """

    # Convert to numpy if needed
    xs = dataset.xs.cpu().numpy() if hasattr(dataset.xs, 'cpu') else np.array(dataset.xs)
    ys = dataset.ys.cpu().numpy() if hasattr(dataset.ys, 'cpu') else np.array(dataset.ys)

    n_groups, max_trials, n_features = xs.shape
    n_actions = ys.shape[-1]

    # Calculate number of additional inputs
    # n_features = n_actions * 2 (choice + reward) + n_additional_inputs + 3 (block, experiment, participant)
    n_additional_inputs = n_features - n_actions * 2 - 3

    # Validate additional_inputs list length
    if additional_inputs is not None and len(additional_inputs) != n_additional_inputs:
        warnings.warn(f"Number of additional_inputs names ({len(additional_inputs)}) does not match "
                      f"number of additional input features in dataset ({n_additional_inputs}). "
                      f"Using generic names for additional inputs.")
        additional_inputs = [f'additional_input_{i}' for i in range(n_additional_inputs)]
    elif additional_inputs is None and n_additional_inputs > 0:
        additional_inputs = [f'additional_input_{i}' for i in range(n_additional_inputs)]

    # Determine if feedback is full or partial by checking if unchosen actions have non-NaN feedback
    # Check first valid trial to detect feedback type
    is_full_feedback = False
    for group_idx in range(n_groups):
        valid_mask = ~np.isnan(xs[group_idx, :, 0])
        if np.any(valid_mask):
            first_valid_idx = np.argmax(valid_mask)
            choice_onehot = xs[group_idx, first_valid_idx, :n_actions]
            chosen_action = int(np.argmax(choice_onehot))
            rewards = xs[group_idx, first_valid_idx, n_actions:n_actions*2]
            # Check if any unchosen action has non-NaN feedback
            for action_idx in range(n_actions):
                if action_idx != chosen_action and not np.isnan(rewards[action_idx]):
                    is_full_feedback = True
                    break
            break

    # Prepare list to collect rows
    rows = []

    for group_idx in range(n_groups):
        # Get metadata (constant across trials within a group)
        block_id = xs[group_idx, 0, -3]
        experiment_id = xs[group_idx, 0, -2]
        participant_id = xs[group_idx, 0, -1]

        # Find number of valid timesteps in xs (non-NaN)
        valid_mask = ~np.isnan(xs[group_idx, :, 0])
        n_valid = int(np.sum(valid_mask))

        # Reconstruct trials from SpiceDataset
        # In CSV format: row t has choice_t and reward_t (reward received at trial t)
        # In SpiceDataset: xs[t] = (choice_t, reward_t) - same time alignment
        # The last trial's choice is in ys, but its reward is not stored

        # Trials 0 to n_valid-1: choice and reward from xs[t]
        for trial_idx in range(n_valid):
            choice_onehot = xs[group_idx, trial_idx, :n_actions]
            choice = int(np.argmax(choice_onehot))

            # Get reward from the same timestep
            rewards = xs[group_idx, trial_idx, n_actions:n_actions*2]

            row = {
                df_participant_id: int(participant_id),
                df_experiment_id: int(experiment_id),
                df_block: int(block_id),
                'trial': trial_idx,
                df_choice: choice,
            }

            # Add feedback from the same trial
            if is_full_feedback:
                for action_idx in range(n_actions):
                    row[f'{df_feedback}_{action_idx}'] = rewards[action_idx]
            else:
                row[df_feedback] = rewards[choice] if not np.isnan(rewards[choice]) else np.nan

            # Add additional inputs from current trial
            if additional_inputs is not None:
                for i, name in enumerate(additional_inputs):
                    row[name] = xs[group_idx, trial_idx, n_actions*2 + i]

            rows.append(row)

        # Note: We don't output the final trial (from ys) because:
        # 1. Its choice is already the target for the last xs timestep
        # 2. Its reward is not stored in the dataset
        # 3. Including it causes csv_to_dataset to create NaN entries in ys

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False) 


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
    non_zero_mask = (xs[..., 0] != -1).int()
    
    # Find cumulative sum along the specified dimension in reverse order
    cumsum = torch.cumsum(non_zero_mask, dim)
    
    # Find the index where the cumulative sum becomes 1 in the reversed array
    last_nonzero_indices = torch.argmax(cumsum, dim=dim)
    
    # compute the indeces at which the data is going to be splitted into training and testing data
    split_indices = (last_nonzero_indices * split_ratio).int()
    
    # initialize training data and testing data storages
    train_xs = torch.zeros((xs.shape[0], max(split_indices), xs.shape[2]), device=dataset.device) - 1
    test_xs = torch.zeros((xs.shape[0], max(last_nonzero_indices - split_indices), xs.shape[2]), device=dataset.device) - 1
    train_ys = torch.zeros((xs.shape[0], max(split_indices), ys.shape[2]), device=dataset.device) - 1
    test_ys = torch.zeros((xs.shape[0], max(last_nonzero_indices - split_indices), ys.shape[2]), device=dataset.device) - 1
    
    # get columns which had no -1 values in the first place to fill them up entirely in the training and testing data
    # necessary for e.g. participant-IDs because otherwise -1 will be passed to embedding layer -> Error
    example_session_id = torch.argmax((last_nonzero_indices < xs.shape[1]-1).int()).item()
    full_columns = xs[example_session_id, -1] != -1
    
    # fill up training and testing data
    for index_session in range(xs.shape[0]):
        train_xs[index_session, :split_indices[index_session]] = xs[index_session, :split_indices[index_session]]
        test_xs[index_session, :last_nonzero_indices[index_session]-split_indices[index_session]] = xs[index_session, split_indices[index_session]:last_nonzero_indices[index_session]]
        train_ys[index_session, :split_indices[index_session]] = ys[index_session, :split_indices[index_session]]
        test_ys[index_session, :last_nonzero_indices[index_session]-split_indices[index_session]] = ys[index_session, split_indices[index_session]:last_nonzero_indices[index_session]]

        # fill up non-"-1"-columns (only applicable for xs)
        train_xs[index_session, :, full_columns] = xs[index_session, :train_xs.shape[1], full_columns]
        test_xs[index_session, :, full_columns] = xs[index_session, :test_xs.shape[1], full_columns]
    
    return SpiceDataset(train_xs, train_ys, device=device), SpiceDataset(test_xs, test_ys, device=device)


def split_data_along_sessiondim(dataset: SpiceDataset, list_test_sessions: List[int] = None, device: torch.device = torch.device('cpu')):
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
    
    if list_test_sessions is not None:
        
        dim = 1
        xs, ys = dataset.xs.cpu(), dataset.ys.cpu()
        
        # get participant ids
        participants_ids = xs[:, 0, -1].unique()
        
        # get sessions ids
        session_ids = xs[:, 0, -3].unique()
        
        # set training sessions
        if list_test_sessions:
            session_ids_test = torch.tensor(list_test_sessions, dtype=torch.float32)
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
                mask_ids = torch.logical_and(xs[:, 0, -3] == sid, xs[:, 0, -1] == pid)
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

        return SpiceDataset(train_xs, train_ys, device=device), SpiceDataset(test_xs, test_ys, device=device)

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
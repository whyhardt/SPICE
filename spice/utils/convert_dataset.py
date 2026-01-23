import sys
import os

from typing import List

import pandas as pd
import numpy as np
import torch
import warnings
from copy import copy

from ..resources.spice_utils import SpiceDataset


def convert_dataset(
    file: str,
    device = None,
    sequence_length: int = None,
    df_participant_id: str = 'session',
    df_block: str = 'block',
    df_experiment_id: str = 'experiment',
    df_choice: str = 'choice',
    df_reward: str = 'reward',
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
    if df_reward + '_0' in df.columns:
        # counterfactual dataset
        reward_cols = []
        for reward_column in df.columns:
            if df_reward in reward_column:
                reward_cols.append(reward_column)
    else:
        reward_cols = [df_reward]
    
    # normalize rewards
    r_min, r_max = [], []
    for reward_column in reward_cols:
        r_min.append(df[reward_column].min())
        r_max.append(df[reward_column].max())
    r_min = np.min(r_min)
    r_max = np.max(r_max)
    for reward_column in reward_cols:
        df[reward_column] = (df[reward_column] - r_min) / (r_max - r_min)
    
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
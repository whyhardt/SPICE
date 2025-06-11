import sys
import os

from typing import List

import pandas as pd
import numpy as np
import warnings
from copy import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.rnn_utils import DatasetRNN
from spice.resources.bandits import BanditSession

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
    ):
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
    else:
        # normalize block numbers
        blocks_max = df[df_block].max()
        blocks_min = df[df_block].min()
        df[df_block] = (df[df_block] - blocks_min) / (blocks_max - blocks_min)
        
    # get maximum number of trials per participant
    n_groups = len(df.groupby(groupby_kw).size())
    max_trials = df.groupby(groupby_kw).size().max()
    
    # let actions begin from 0
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
    
    n_ids = 0
    if df_participant_id is not None:
        n_ids += 1
    if df_experiment_id is not None:
        n_ids += 1
        
    xs = np.zeros((n_groups, max_trials, n_actions*2 + n_ids + additional_inputs_xs + 1 if df_block is not None else 0)) - 1
    ys = np.zeros((n_groups, max_trials, n_actions)) - 1
    
    probs_choice = np.zeros((n_groups, max_trials, n_actions)) - 1
    values_action = np.zeros((n_groups, max_trials, n_actions)) - 1
    values_reward = np.zeros((n_groups, max_trials, n_actions)) - 1
    values_choice = np.zeros((n_groups, max_trials, n_actions)) - 1
    
    experiment_list = []
    # for index_group, participant_id in enumerate(participant_ids):
    for index_group, group in enumerate(df.groupby(groupby_kw)):
        
        choice = np.eye(n_actions)[group[1][df_choice].values.astype(int)]
        rewards = np.zeros((len(choice), n_actions)) - 1
        for index_column, reward_column in enumerate(reward_cols):
            # add reward into the correct column
            reward = group[1][reward_column].values
            if len(reward_cols) > 1:
                # counterfactual data
                rewards[:, index_column] = reward
            else:
                for index_choice, ch in enumerate(choice):
                    rewards[index_choice, np.argmax(ch)] = reward[index_choice]

        # write arrays for DatasetRNN
        xs[index_group, :len(choice), :n_actions] = choice
        xs[index_group, :len(choice), n_actions:n_actions*2] = rewards
        
        # add all grouping variables to the end
        xs[index_group, :, -3] = group[1][df_block].values[0]
        xs[index_group, :, -2] = group[1][df_experiment_id].values[0]
        xs[index_group, :, -1] = group[1][df_participant_id].values[0]
        ys[index_group, :len(choice)-1] = choice[1:]
        
        # write additional inputs after choices and rewards and before grouping variables
        if additional_inputs is not None and len(additional_inputs) > 0:
            for index, additional_input in enumerate(additional_inputs):
                xs[index_group, :len(choice), n_actions*2+index] = group[1][additional_input].values
        
        experiment = BanditSession(
            choices=group[1][df_choice].values,
            rewards=rewards,
            session=np.full((*rewards.shape[:-1], 1), index_group),
            reward_probabilities=np.zeros_like(choice)+0.5,
            q=np.zeros_like(choice)+0.5,
            n_trials=len(choice)
        )
        
        # get update dynamics if available - only for generated data with e.g. utils/create_dataset.py
        if 'choice_prob_0' in df.columns:
            for index_choice in range(n_actions):
                probs_choice[index_group, :len(choice), index_choice] = group[1][f'choice_prob_{index_choice}'].values
                values_action[index_group, :len(choice), index_choice] = group[1][f'action_value_{index_choice}'].values
                values_reward[index_group, :len(choice), index_choice] = group[1][f'reward_value_{index_choice}'].values
                values_choice[index_group, :len(choice), index_choice] = group[1][f'choice_value_{index_choice}'].values
        
        experiment_list.append(experiment)
        
    return DatasetRNN(xs, ys, device=device, sequence_length=sequence_length), experiment_list, original_df, (probs_choice, values_action, values_reward, values_choice)
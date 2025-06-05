import sys
import os

from typing import List

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.rnn_utils import DatasetRNN
from spice.resources.bandits import BanditSession

def convert_dataset(
    file: str, 
    device = None, 
    sequence_length: int = None,
    df_participant_id: str = 'session',
    df_choice: str = 'choice',
    df_reward: str = 'reward',
    ):
    df = pd.read_csv(file, index_col=None)
    
    # replace all nan values with -1
    df = df.replace(np.nan, -1)
    
    # get all different sessions
    sessions = df[df_participant_id].unique()
    # get maximum number of trials per session
    max_trials = df.groupby(df_participant_id).size().max()

    # let actions begin from 0
    choices = df[df_choice].values
    choice_min = np.nanmin(choices[choices != -1])
    choices[choices != -1] = choices[choices != -1] - choice_min
    df[df_choice] = choices
    
    # number of possible actions
    n_actions = int(df[df_choice].max() + 1)
    
    # get all columns with rewards
    if df_reward + '_0' in df.columns:
        # counterfactual dataset
        reward_cols = []
        for column in df.columns:
            if df_reward in column:
                reward_cols.append(column)
    else:
        reward_cols = [df_reward]
    
    # normalize rewards
    r_min, r_max = [], []
    for column in reward_cols:
        r_min.append(df[column].min())
        r_max.append(df[column].max())
    r_min = np.min(r_min)
    r_max = np.max(r_max)
    for column in reward_cols:
        df[column] = (df[column] - r_min) / (r_max - r_min)
    
    xs = np.zeros((len(sessions), max_trials, n_actions*2 + 1)) - 1
    ys = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    probs_choice = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_action = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_reward = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_choice = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    experiment_list = []
    for index_session, session in enumerate(sessions):
        choice = np.eye(n_actions)[df[df[df_participant_id] == session][df_choice].values.astype(int)]
        rewards = np.zeros((len(choice), n_actions)) - 1
        for index_column, column in enumerate(reward_cols):
            # add reward into the correct column
            reward = df[df[df_participant_id] == session][column].values
            if len(reward_cols) > 1:
                # counterfactual data
                rewards[:, index_column] = reward
            else:
                for index_choice, ch in enumerate(choice):
                    rewards[index_choice, np.argmax(ch)] = reward[index_choice]
        
        # write arrays for DatasetRNN
        xs[index_session, :len(choice), :n_actions] = choice
        xs[index_session, :len(choice), n_actions:n_actions*2] = rewards
        xs[index_session, :, -1] += index_session+1
        ys[index_session, :len(choice)-1] = choice[1:]

        experiment = BanditSession(
            choices=df[df[df_participant_id] == session][df_choice].values,
            rewards=rewards,
            session=np.full((*rewards.shape[:-1], 1), index_session),
            reward_probabilities=np.zeros_like(choice)+0.5,
            q=np.zeros_like(choice)+0.5,
            n_trials=len(choice)
        )
        
        # get update dynamics if available - only for generated data with e.g. utils/create_dataset.py
        if 'choice_prob_0' in df.columns:
            for index_choice in range(n_actions):
                probs_choice[index_session, :len(choice), index_choice] = df[df[df_participant_id] == session][f'choice_prob_{index_choice}'].values
                values_action[index_session, :len(choice), index_choice] = df[df[df_participant_id] == session][f'action_value_{index_choice}'].values
                values_reward[index_session, :len(choice), index_choice] = df[df[df_participant_id] == session][f'reward_value_{index_choice}'].values
                values_choice[index_session, :len(choice), index_choice] = df[df[df_participant_id] == session][f'choice_value_{index_choice}'].values
        
        experiment_list.append(experiment)
        
    return DatasetRNN(xs, ys, device=device, sequence_length=sequence_length), experiment_list, df, (probs_choice, values_action, values_reward, values_choice)
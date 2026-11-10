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
        path_data = 'data/mvt_deterministic_0p60.csv'
    
    import pandas as pd
    dataset = csv_to_dataset(
        file=pd.read_csv(path_data, low_memory=False),     # adds the effort-condition inputs
        df_participant_id='subject_id',
        df_choice='decision',
        df_feedback='reward',
        df_block='overall_round',
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
    
    info_dataset = {
        'n_participants': n_participants,
        'n_actions': n_actions,
        'n_additional_inputs': n_additional_inputs,
    }
    
    return dataset, dataset, info_dataset
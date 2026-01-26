import numpy as np
import pandas as pd
import torch

from spice.resources.spice_utils import SpiceDataset
from spice.utils.convert_dataset import dataset_to_csv
from spice.utils.agent import Agent, get_update_dynamics

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from weinhardt2025.utils.bandits import create_dataset, BanditsDrift, BanditsSwitch
from weinhardt2025.benchmarking.benchmarking_qlearning import QLearning


list_n_participants = [32]#, 64, 128, 256, 512]
n_trials_per_session = 100
n_blocks_per_session = 4
n_iterations_per_n_sessions = 1
sigma = [0.2]

base_name = 'weinhardt2025/data/synthetic/synthetic_*.csv'

sample_parameters = True
zero_threshold = 0.2
parameter_variance = 0.2
parameters_mean = {
    'beta_reward': 3.0,
    'beta_choice': 1.0,
    'alpha_reward': 0.5,
    'alpha_penalty': 0.5,
    'forget_rate': 0.2,
    'alpha_choice': 0.5,
}


def compute_beta_dist_params(mean, var):
    n = mean * (1-mean) / var**2
    a = mean * n
    b = (1-mean) * n
    return a, b


def sample_parameters() -> dict:
    parameters = {}
    
    # sample scaling parameters (inverse noise temperatures)
    parameters['beta_reward'], parameters['beta_choice'] = np.zeros(n_participants), np.zeros(n_participants)
    
    while np.any((parameters['beta_reward'] == 0) & (parameters['beta_choice'] == 0)):
        mask = (parameters['beta_reward'] == 0) & (parameters['beta_choice']==0)
        parameters['beta_reward'][mask] = np.random.beta(*compute_beta_dist_params(mean=0.5, var=parameter_variance), mask.sum())
        parameters['beta_choice'][mask] = np.random.beta(*compute_beta_dist_params(mean=0.5, var=parameter_variance), (mask).sum())
        # apply zero-threshold if applicable
        parameters['beta_reward'][mask] = parameters['beta_reward'][mask] * 2 * parameters_mean['beta_reward'] * (parameters['beta_reward'][mask] > zero_threshold)
        parameters['beta_choice'][mask] = parameters['beta_choice'][mask] * 2 * parameters_mean['beta_choice'] * (parameters['beta_choice'][mask] > zero_threshold)
        
    # sample auxiliary parameters
    parameters['forget_rate'] = np.random.beta(*compute_beta_dist_params(mean=parameters_mean['forget_rate'], var=parameter_variance), n_participants)
    parameters['forget_rate'] =  parameters['forget_rate'] * (parameters['forget_rate'] > zero_threshold)
    
    parameters['alpha_choice'] = np.random.beta(*compute_beta_dist_params(mean=parameters_mean['alpha_choice'], var=parameter_variance), n_participants)
    parameters['alpha_choice'] = parameters['alpha_choice'] * (parameters['alpha_choice'] > zero_threshold)
    
    # sample learning rate; don't zero out; only check for applicability of asymmetric learning rates
    parameters['alpha_reward'] = np.random.beta(*compute_beta_dist_params(mean=parameters_mean['alpha_reward'], var=parameter_variance), n_participants)
    parameters['alpha_penalty'] = np.random.beta(*compute_beta_dist_params(mean=parameters_mean['alpha_penalty'], var=parameter_variance), n_participants)
    # set to symmetric learning if difference between alpha_reward and alpha_penalty lower than threshold
    index_symmetric_learning = np.abs(parameters['alpha_reward'] - parameters['alpha_penalty']) < zero_threshold
    parameters['alpha_reward'][index_symmetric_learning] = np.mean((parameters['alpha_reward'][index_symmetric_learning], parameters['alpha_penalty'][index_symmetric_learning]))
    parameters['alpha_penalty'][index_symmetric_learning] = np.mean((parameters['alpha_reward'][index_symmetric_learning], parameters['alpha_penalty'][index_symmetric_learning]))
    
    return {key: param.reshape(-1, 1) for key, param in parameters.items()}


for iteration in range(n_iterations_per_n_sessions):
    for n_participants in list_n_participants:
        for experiment_id in range(len(sigma)):
            dataset_name = base_name.replace('*', f'{n_participants}p_{iteration}_{experiment_id}')
            
            parameters = sample_parameters()
            
            # init model
            agent = Agent(QLearning(
                n_actions=2,
                n_participants=n_participants,
                **{key: torch.tensor(param, dtype=torch.float32) for key, param in parameters.items()}
                ), use_sindy=True, deterministic=False)

            environment = BanditsDrift(sigma=sigma[experiment_id])

            xs_session, ys_session = [], []
            for index_block in range(n_blocks_per_session):
                dataset_block = create_dataset(
                            agent=agent,
                            environment=environment,
                            n_trials=n_trials_per_session,
                            n_sessions=n_participants,
                            verbose=False,
                            )[0]
                dataset_block.xs [..., -3] = index_block
                dataset_block.xs[..., -2] = experiment_id
                xs_session.append(dataset_block.xs)
                ys_session.append(dataset_block.ys)
            dataset = SpiceDataset(xs=torch.concat(xs_session), ys=torch.concat(ys_session))
            
            dataset_to_csv(dataset=dataset, path=dataset_name)
            print(f'Data saved to {dataset_name}')
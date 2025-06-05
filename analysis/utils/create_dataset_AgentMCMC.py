import sys, os

import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.bandits import create_dataset, BanditsDrift, get_update_dynamics, AgentQ
from spice.resources.rnn_utils import DatasetRNN
from analysis.utils.setup_agents import setup_agent_mcmc
from analysis.utils.convert_dataset import convert_dataset
from spice.benchmarking.hierarchical_bayes_numpyro import rl_model


def main(path_model, path_data, path_save, n_trials_per_session):
    dataset = convert_dataset(file=path_data)[0]
    n_sessions = len(dataset)

    xs = torch.zeros((n_sessions, n_trials_per_session, dataset.xs.shape[-1]))
    ys = torch.zeros((n_sessions, n_trials_per_session, dataset.ys.shape[-1]))
    
    agent = setup_agent_mcmc(path_model)
    
    for session in tqdm(range(n_sessions)):
        environment = BanditsDrift(sigma=0.2)
        dataset = create_dataset(
                    agent=agent[session],
                    environment=environment,
                    n_trials=n_trials_per_session,
                    n_sessions=1,
                    verbose=False,
                    )[0]
        
        dataset.xs[..., -1] = torch.full_like(dataset.xs[..., -1], session)
        
        xs[session] = dataset.xs[0]
        ys[session] = dataset.ys[0]

    dataset = DatasetRNN(xs, ys)

    # dataset columns
    # general dataset columns
    session, choice, reward = [], [], []
    choice_prob_0, choice_prob_1, action_value_0, action_value_1, reward_value_0, reward_value_1, choice_value_0, choice_value_1 = [], [], [], [], [], [], [], []

    for i in range(len(dataset)):    
        # get update dynamics
        experiment = dataset.xs[i].cpu().numpy()
        qs, choice_probs, _ = get_update_dynamics(experiment, agent[i])
        
        # append behavioral data
        session += list(experiment[:, -1])
        choice += list(np.argmax(experiment[:, :agent[i]._n_actions], axis=-1))
        reward += list(np.max(experiment[:, agent[i]._n_actions:agent[i]._n_actions*2], axis=-1))
        
        # append update dynamics
        choice_prob_0 += list(choice_probs[:, 0])
        choice_prob_1 += list(choice_probs[:, 1])
        action_value_0 += list(qs[0][:, 0])
        action_value_1 += list(qs[0][:, 1])
        reward_value_0 += list(qs[1][:, 0])
        reward_value_1 += list(qs[1][:, 1])
        choice_value_0 += list(qs[2][:, 0])
        choice_value_1 += list(qs[2][:, 1])
        
    columns = ['session', 'choice', 'reward', 'choice_prob_0', 'choice_prob_1', 'action_value_0', 'action_value_1', 'reward_value_0', 'reward_value_1', 'choice_value_0', 'choice_value_1']
    data = np.stack((np.array(session), np.array(choice), np.array(reward), np.array(choice_prob_0), np.array(choice_prob_1), np.array(action_value_0), np.array(action_value_1), np.array(reward_value_0), np.array(reward_value_1), np.array(choice_value_0), np.array(choice_value_1)), axis=-1)
    df = pd.DataFrame(data=data, columns=columns)

    # data_save = path_data.replace('.', '_'+model+'.')
    df.to_csv(path_save, index=False)

    print(f'Data saved to {path_save}')
    

if __name__=='__main__':
    model = 'ApAnBrAcfpAcfnBcfBch'
    path_model = f'params/eckstein2022/mcmc_eckstein2022_{model}.nc'
    path_data = 'data/eckstein2022/eckstein2022.csv'
    n_trials_per_session = 200
    
    main(
        path_model=path_model,
        path_data=path_data,
        path_save=path_data.replace('.', '_training_'+model+'.'),
        n_trials_per_session=n_trials_per_session,
        )
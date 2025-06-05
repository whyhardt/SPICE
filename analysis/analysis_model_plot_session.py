import os
import sys

import numpy as np
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.utils.setup_agents import setup_agent_rnn, setup_agent_spice, setup_agent_q
from analysis.utils.convert_dataset import convert_dataset
from spice.benchmarking.hierarchical_bayes_numpyro import rl_model
from spice.utils.plotting import plot_session

def main(data, model_mcmc, model_rnn, session_id):
    
    # load data
    experiment_list = convert_dataset(data)[1]
    experiment = experiment_list[session_id]
    
    agent_rnn, agent_sindy, agent_mcmc = None, None, None
    n_parameters_rnn, n_parameters_sindy, n_parameters_mcmc = None, None, None
    
    agent_rnn, n_parameters_rnn = [], []
    agent_sindy, n_parameters_sindy = [], []
    
    # setup rnn agent for comparison
    agent_rnn = setup_agent_rnn(model_rnn, len(experiment_list), participant_emb=True, counterfactual=False)
    n_parameters_rnn.append(sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad))

    # setup sindy agent and get number of sindy coefficients which are not 0
    agent_sindy = setup_agent_spice(model_rnn, data, participant_id=session_id)[0]
    n_parameters_sindy.append(agent_sindy._count_sindy_parameters(without_self=True))
    
    # setup mcmc agent
    with open(model_mcmc, 'rb') as file:
        mcmc = pickle.load(file)
    
    parameters = {
        'alpha_pos': np.full(1, 1),
        'alpha_neg': np.full(1, -1),
        'alpha_c': np.full(1, 1),
        'beta_c': np.full(1, 0),
        'beta_r': np.full(1, 1),
    }
    
    n_parameters_mcmc = 0
    # mcmc.print_summary()
    for param in parameters:
        for p in mcmc.get_samples():
            if p == param + f'[{session_id}]':
                parameters[param] = np.mean(mcmc.get_samples()[p], axis=0)
                n_parameters_mcmc += 1
                break
        n_parameters_mcmc = np.full(1, n_parameters_mcmc)

    if np.mean(parameters['alpha_neg']) == -1:
        parameters['alpha_neg'] = parameters['alpha_pos']
    
    agent_mcmc = setup_agent_q(parameters)
    
    fig, axs = plot_session(
        {
            # 'benchmark': agent_mcmc, 
            'rnn': agent_rnn, 
            'sindy': agent_sindy,
            }, 
        experiment,
        )
    
    fig.suptitle('$beta_r=$'+str(np.round(agent_rnn._beta_reward, 2)) + '; $beta_c=$'+str(np.round(agent_rnn._beta_choice, 2)))
    plt.show()

        
if __name__ == "__main__":
    
    data = 'data/2arm/sugawara2021_143_processed.csv'
    model_mcmc = 'benchmarking/params/sugawara2021_143/hierarchical/traces_hbi_ApAcBcBr.nc'
    model_rnn = 'params/benchmarking/rnn_sugawara.pkl'
    session_id = 142 # 140 --> Initial scepticism; 142 --> Initial super-perseverance; 139 --> Normal perseverance
    
    main(data, model_mcmc, model_rnn, session_id)
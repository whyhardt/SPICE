import sys, os

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from spice.resources.bandits import create_dataset, get_update_dynamics, BanditsDrift, BanditsFlip_eckstein2022, Bandits_Standard, Agent, AgentQ
from spice.resources.spice_utils import SpiceDataset
from spice.utils.setup_agents import setup_agent_rnn, setup_agent_spice
from spice.utils.convert_dataset import convert_dataset

# dataset specific SPICE configurations and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from spice.precoded import InteractionRNN, INTERACTION_CONFIG
from weinhardt2025.benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022


# ----------------------- GENERAL CONFIGURATION ----------------------------
agent_type = 'q_agent'  # 'rnn', 'benchmark', 'baseline', 'q_agent'
n_trials_per_session = 200

# ------------------- CONFIGURATION SYNTHETIC --------------------
dataset = 'synthetic'
class_rnn = InteractionRNN
sindy_config = INTERACTION_CONFIG
bandits_environment = BanditsDrift
n_sessions = 1
bandits_kwarg = {'sigma': 0.2}

# ------------------- CONFIGURATION ECKSTEIN2022 --------------------
# dataset = 'eckstein2022'
# benchmark_model = 'ApAnBrBcfBch'
# class_rnn = Weinhardt2025RNN
# sindy_config = WEINHARDT_2025_CONFIG
# bandits_environment = BanditsFlip_eckstein2022
# n_sessions = 1
# bandits_kwargs = {}
# setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
# rl_model = benchmarking_eckstein2022.rl_model
# path_rnn = f'params/{dataset}/rnn_{dataset}_l2_0_0005.pkl'
# path_spice = f'params/{dataset}/spice_{dataset}_l2_0_0005.pkl'
# path_benchmark = f'params/{dataset}/mcmc_{dataset}_{benchmark_model}.nc'

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
# dataset = 'dezfouli2019'
# benchmark_model = 'PhiChiBetaKappaC'
# class_rnn = RLRNN_eckstein2022
# sindy_config = SindyConfig_eckstein2022
# bandits_environment = Bandits_Standard
# n_sessions = 6
# bandits_kwargs = {'reward_prob_0': [0.25, 0.125, 0.08, 0.05, 0.05, 0.05], 'reward_prob_1': [0.05, 0.05, 0.05, 0.25, 0.125, 0.08]}
# setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
# Dezfouli2019GQL = benchmarking_dezfouli2019.Dezfouli2019GQL
# path_rnn = f'params/{dataset}/rnn_{dataset}_l2_0_001.pkl'
# path_spice = f'params/{dataset}/spice_{dataset}_l2_0_001.pkl'
# path_benchmark = f'params/{dataset}/gql_{dataset}_{benchmark_model}.pkl'


# ------------------- PIPELINE ----------------------------

generating_model = agent_type if agent_type != 'benchmark' else benchmark_model
path_data = f'weinhardt2025/data/{dataset}/{dataset}.csv'
path_save = f'weinhardt2025/data/{dataset}/{dataset}_generated_behavior_{generating_model}.csv'

if agent_type == 'rnn':
    path_model = path_rnn
elif agent_type == 'q_agent':
    path_model = None
    path_data = None
    path_save = 'weinhardt2025/data/q_agent_.csv'
else:
    path_model = path_benchmark

if agent_type == 'spice':
    setup_agent = setup_agent_spice
elif agent_type == 'rnn':
    setup_agent = setup_agent_rnn
# elif agent_type == 'benchmark':
else:
    setup_agent = setup_agent_benchmark

if path_data:
    n_participants = len(convert_dataset(path_data).xs[:, 0, -1].unique())
else:
    n_participants = 100
    
dataset_xs, dataset_ys = [], []
for session in range(n_sessions):
    environment = bandits_environment(
        reward_prob_0 = bandits_kwargs['reward_prob_0'][session] if 'reward_prob_0' in bandits_kwargs else None,
        reward_prob_1 = bandits_kwargs['reward_prob_1'][session] if 'reward_prob_1' in bandits_kwargs else None,
        )

    if path_model:
        agent = setup_agent(
            class_rnn=class_rnn,
            path_model=path_model,
            path_rnn=path_rnn,
            path_spice=path_spice,
            deterministic=False,
            model_config=benchmark_model,
            )
    else:
        agent = AgentQ(
            alpha_reward=0.3, 
            beta_reward=3,
            # Different agent-configurations for different experiments
            # forget_rate = 0.2,
            # alpha_penalty=0.5,
            )
        
    if isinstance(agent, tuple):
        # in case of setup_agent_benchmark -> output: agent, n_parameters
        agent = agent[0]

    dataset = create_dataset(
                agent=agent,
                environment=environment,
                n_trials=n_trials_per_session,
                n_sessions=n_participants,
                verbose=False,
                )[0]
    
    dataset_xs.append(dataset.xs)
    dataset_ys.append(dataset.ys)
    
dataset = SpiceDataset(torch.concat(dataset_xs), torch.concat(dataset_ys))

# dataset columns
# general dataset columns
session, choice, reward = [], [], []

print('Saving values...')
n_actions = agent[0]._n_actions if isinstance(agent, list) else agent._n_actions
for i in tqdm(range(len(dataset))):    
    # get update dynamics
    experiment = dataset.xs[i].cpu().numpy()
    # qs, choice_probs, _ = get_update_dynamics(experiment, agent)
    
    # append behavioral data
    session += list(experiment[:, -1])
    choice += list(np.argmax(experiment[:, :n_actions], axis=-1))
    reward += list(np.max(experiment[:, n_actions:n_actions*2], axis=-1))
    
columns = ['session', 'choice', 'reward']
data = np.stack((np.array(session), np.array(choice), np.array(reward)), axis=-1)
df = pd.DataFrame(data=data, columns=columns)

df.to_csv(path_save, index=False)

print(f'Data saved to {path_save}')
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from weinhardt2025.utils.bandits import BanditsDrift, BanditsSwitch, create_dataset
from spice.utils.agent import Agent, get_update_dynamics
from spice.utils.plotting import plot_session
from spice.utils.convert_dataset import csv_to_dataset


path_data = 'weinhardt2025/data/synthetic/synthetic_ApAnBc.csv'
path_model = 'benchmarking/params/mcmc_eckstein2022_ApAnBrBcfBch.nc'

agent1 = Agent(
    beta_reward=3.0,
    alpha_reward=0.5,
    alpha_penalty=0.5,
    beta_choice=0.,
    )

agent2 = Agent(
    beta_reward=3.0,
    alpha_reward=0.5,
    alpha_penalty=0.5,
    beta_choice=1.0,
    )

# agent2 = benchmarking_dezfouli2019.Agent_dezfouli2019(
#     n_actions=2,
#     d=2,
#     phi=np.array([0.145, 0.815]),
#     chi=np.array([0.635, 0.389]),
#     beta=np.array([4.258, -1.002]),
#     kappa=np.array([3.268, -0.974]),
#     C=np.array([[-14.256, 4.243],[17.998, -6.335]]),
#     deterministic=False,
# )

# rl_model = benchmarking_eckstein2022.rl_model
# agent2 = benchmarking_eckstein2022.setup_agent_benchmark(path_model=path_model, model_config='ApBr')

# env = EnvironmentBanditsSwitch(0.05, reward_prob_high=1.0, reward_prob_low=0.5)
# env = BanditsDrift(0.2)
# trajectory = create_dataset(agent1, env, 128, 1)[0]

trajectory = csv_to_dataset(path_data)

agents = {
    'groundtruth': agent1, 
    'benchmark': agent2,
    }
fig, axs = plot_session(agents, trajectory.xs[0])
plt.show()
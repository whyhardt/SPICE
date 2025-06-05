import sys
import os

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.bandits import AgentQ, BanditsDrift, BanditsSwitch, plot_session, create_dataset, get_update_dynamics
from spice.utils.plotting import plot_session

agent1 = AgentQ(
    beta_reward=1.,
    alpha_reward=0.5,
    alpha_penalty=0.5,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=0.,
    alpha_choice=0.,
    alpha_counterfactual=0.,
    )

agent2 = AgentQ(
    beta_reward=1.,
    alpha_reward=0.5,
    alpha_penalty=0.5,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=.5,
    alpha_counterfactual=0.,
    )

# env = EnvironmentBanditsSwitch(0.05, reward_prob_high=1.0, reward_prob_low=0.5)
env = BanditsDrift(0.2)
trajectory = create_dataset(agent1, env, 128, 1)[0]
agents = {'groundtruth': agent1, 'benchmark': agent2}
fig, axs = plot_session(agents, trajectory.xs[0])
plt.show()
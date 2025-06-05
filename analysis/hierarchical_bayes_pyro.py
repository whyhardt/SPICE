import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer as infer
import pyro.optim as optim
import torch
import pandas as pd
import arviz as az


def rl_model(choice: torch.Tensor, reward: torch.Tensor):
    n_sessions = choice.shape[0]
    
    # # Priors for group-level parameters
    # alpha_pos_mean = pyro.sample("alpha_pos_mean", dist.Normal(0, 1))
    # alpha_neg_mean = pyro.sample("alpha_neg_mean", dist.Normal(0, 1))
    # pers_mean = pyro.sample("pers_mean", dist.Normal(0, 1))
    # beta_mean = pyro.sample("beta_mean", dist.HalfNormal(1))
    
    # # Priors for individual-level variation (hierarchical)
    # alpha_pos_std = pyro.sample("alpha_pos_std", dist.HalfNormal(1))
    # alpha_neg_std = pyro.sample("alpha_neg_std", dist.HalfNormal(1))
    # pers_std = pyro.sample("pers_std", dist.HalfNormal(1))
    # beta_std = pyro.sample("beta_std", dist.HalfNormal(1))
    
    # # Individual-level parameters
    # with pyro.plate("participants", 1):  # n_sessions
    #     alpha_pos = pyro.sample("alpha_pos", dist.Normal(alpha_pos_mean, alpha_pos_std))
    #     alpha_neg = pyro.sample("alpha_neg", dist.Normal(alpha_neg_mean, alpha_neg_std))
    #     pers = pyro.sample("pers", dist.Normal(pers_mean, pers_std))
    #     beta = pyro.sample("beta", dist.Normal(beta_mean, beta_std))
    
    # Basic bayesian inference (not hierarchical)
    alpha_pos = pyro.sample("alpha_pos", dist.Uniform(0., 1.))
    # alpha_neg = pyro.sample("alpha_neg", dist.Uniform(0.2, 0.3))
    # pers = pyro.sample("pers", dist.Uniform(0.2, 0.3))
    beta = pyro.sample("beta", dist.Uniform(0., 10.))

    # Define initial Q-values and initialize the previous choice variable
    q_values = torch.tensor((0.5, 0.5))
    prev_choice = torch.zeros(n_sessions)  # stores the previous choice (0 or 1) for each participant

    for t in range(choice.shape[1]-1):
        ch = choice[:, t]
        rw = reward[:, t]
        next_ch = choice[:, t+1, 1]
        # mask = next_ch >= 0
        
        # Compute prediction errors for each outcome
        pe = (rw - q_values) * ch
        # lr = torch.where(rw == 1, alpha_pos, alpha_neg)

        # Update Q-values
        # q_values[torch.arange(n_sessions), ch] = q_values[torch.arange(n_sessions), ch] + alpha_pos * pe
        q_values = q_values + alpha_pos * pe

        # Calculate action probabilities with perseverance effect
        # perseverance_effect = (ch == prev_choice) if t > 0 else torch.zeros(n_sessions) # 1 if repeated choice, 0 otherwise
        logits = beta * (q_values[:, 1] - q_values[:, 0]) #+ pers * perseverance_effect)
        action_prob = torch.nn.functional.sigmoid(logits)

        # Likelihood of observed choices
        # with pyro.handlers.mask(mask=mask):
        #     pyro.sample(f"obs_{t}", dist.Bernoulli(probs=action_prob), obs=mask * next_ch)
        pyro.sample(f"obs_{t}", dist.Bernoulli(probs=action_prob), obs=next_ch)
        
        # Update prev_choice to current choice for the next time step
        # prev_choice = ch


# Prepare the data
data = pd.read_csv("data/data_rnn_a025_b30_ap025.csv")
# get all different sessions
sessions = data['session'].unique()
# get maximum number of trials per session
max_trials = data.groupby('session').size().max()
# sort values into session-grouped arrays
choices = torch.zeros((len(sessions), max_trials, 2)) - 1
rewards = torch.zeros((len(sessions), max_trials, 1)) - 1
for i, s in enumerate(sessions):
    choice = torch.tensor(data[data['session'] == s]['choice'].values, dtype=int)
    reward = torch.tensor(data[data['session'] == s]['reward'].values)
    choices[i, :len(choice)] = torch.eye(2)[choice]
    rewards[i, :len(choice), 0] = reward

# Run the model
kernel = infer.NUTS(rl_model)
mcmc = infer.MCMC(kernel, warmup_steps=500, num_samples=1000)
mcmc.run(choice=choices, reward=rewards)
samples = mcmc.get_samples()
print(samples)

# Convert to ArviZ InferenceData object
inference_data = az.from_pyro(mcmc)

# Plot posterior distributions
az.plot_trace(inference_data)
import matplotlib.pyplot as plt
plt.show()
import pymc as pm
import numpy as np
import pandas as pd
from theano import tensor as T
import arviz as az
import pickle


# Load your data
data = pd.read_csv("data/sugawara2021_143_processed.csv")

# Define the model
with pm.Model() as hierarchical_rl_model:
    # Group-level parameters
    alpha_pos_mean = pm.Normal('alpha_pos_mean', mu=0.5, sigma=0.2)
    alpha_pos_sd = pm.HalfNormal('alpha_pos_sd', sigma=0.1)
    alpha_neg_mean = pm.Normal('alpha_neg_mean', mu=0.5, sigma=0.2)
    alpha_neg_sd = pm.HalfNormal('alpha_neg_sd', sigma=0.1)
    beta_mean = pm.Normal('beta_mean', mu=1.0, sigma=0.5)
    beta_sd = pm.HalfNormal('beta_sd', sigma=0.5)
    perseverance_mean = pm.HalfNormalNormal('perseverance_mean', sigma=0.5)
    perseverance_sd = pm.HalfNormal('perseverance_sd', sigma=0.5)

    # Participant-level parameters (hierarchical)
    unique_participants = data['participant_id'].unique()
    n_participants = len(unique_participants)

    alpha_pos = pm.Normal('alpha_pos', mu=alpha_pos_mean, sigma=alpha_pos_sd, shape=n_participants)
    alpha_neg = pm.Normal('alpha_neg', mu=alpha_neg_mean, sigma=alpha_neg_sd, shape=n_participants)
    beta = pm.Normal('beta', mu=beta_mean, sigma=beta_sd, shape=n_participants)
    # alpha_pos = pm.Deterministic('alpha_pos', pm.math.abs_(alpha))  # Positive learning rate
    # alpha_neg = pm.Deterministic('alpha_neg', pm.math.abs_(alpha))  # Negative learning rate
    perseverance = pm.Normal('perseverance', mu=perseverance_mean, sigma=perseverance_sd, shape=n_participants)

    # Define the likelihood function for each participant
    likelihoods = []
    for i, participant_id in enumerate(unique_participants):
        participant_data = data[data['participant_id'] == participant_id]
        choices = participant_data['choice'].values
        rewards = participant_data['reward'].values

        # Initialize variables for tracking choice probabilities and learning
        q_values = np.zeros(2)  # Assuming a two-action task

        # Define participant-specific parameters
        alpha_pos_p = alpha_pos[i]
        alpha_neg_p = alpha_neg[i]
        beta_p = beta[i]
        perseverance_p = perseverance[i]

        # Calculate likelihood for each trial
        for t in range(len(choices)):
            # Action perseverance term
            perseverance_term = perseverance_p * (choices[t-1] if t > 0 else 0)

            # Softmax decision rule with perseverance and decision noise
            prob_choice_0 = pm.math.sigmoid(beta_p * (q_values[0] - q_values[1] + perseverance_term))
            choice_prob = prob_choice_0 if choices[t] == 0 else 1 - prob_choice_0

            # Observation model (Bernoulli likelihood)
            likelihoods.append(pm.Bernoulli(f'obs_{i}_{t}', p=choice_prob, observed=choices[t]))

            # Update Q-values based on the prediction error
            reward = rewards[t]
            prediction_error = reward - q_values[choices[t]]

            if prediction_error > 0:
                q_values[choices[t]] += alpha_pos[i] * prediction_error
            else:
                q_values[choices[t]] += alpha_neg[i] * prediction_error

    # Combine likelihoods
    pm.Potential("likelihood", T.sum(likelihoods))

# Run the MCMC sampling
with hierarchical_rl_model:
    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)
    
az.plot_trace(trace, var_names=["alpha_mean", "beta_mean", "perseverance_mean"])

# Save the trace
with open('trace_file.pkl', 'wb') as f:
    pickle.dump(trace, f)
""" Task-Agent Interaction: Interaction between reduced Bayesian model and predictive inference task """

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
from al_utilities import compute_persprob, correct_push


def task_agent_int(which_exp, df, agent, agent_vars, sel_coeffs, sim=False):
    """ This function models the interaction between task and agent (RBM)

    :param which_exp: Current experiment: 1: first experiment; 2: follow-up experiment
    :param df: Data frame with relevant data
    :param agent: Agent-object instance
    :param agent_vars: Agent-variables-object instance
    :param sel_coeffs: Free parameters
    :param sim: Indicates whether function is currently used for simulations or not
    :return: llh_mix, df_data: Negative log-likelihoods of mixture model and data frame with simulation results
    """

    # Extract and initialize relevant variables
    # -----------------------------------------
    n_trials = len(df)  # number of trials
    pers = df['a_t'] == 0  # indicates perseveration trials
    high_val = df['r_t'] == 1  # indicates high-value trials
    mu = np.full([n_trials], np.nan)  # inferred mean of the outcome-generating distribution
    mu_bias = np.full([n_trials], np.nan)  # inferred mean with bucket bias
    a_hat = np.full(n_trials, np.nan)  # predicted update according to reduced Bayesian model
    epsilon = np.full(n_trials, np.nan)  # response noise
    omega = np.full(n_trials, np.nan)  # change-point probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    sigma_t_sq = np.full(n_trials, np.nan)  # estimation uncertainty

    # Prediction error
    if not sim:
        delta = df['delta_t']
    else:
        delta = np.full(len(df), np.nan)

    # Log-likelihood
    n_new_block = np.sum(df['new_block'] == 1)
    llh_rbm = np.full([n_trials-n_new_block], np.nan)  # log-likelihood of reduced Bayesian model
    llh_mix = np.full([n_trials-n_new_block], np.nan)  # log-likelihood of mixture model

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_z_t = np.full(n_trials, np.nan)  # simulated initial bucket location
    sim_y_t = np.full(n_trials, np.nan)  # simulated shift of the bucket
    sim_a_t = np.full(n_trials, np.nan)  # simulated update

    # Initialize variables related to estimation
    llh_counter = 0
    corrected_0_p = 1e-10

    # Cycle over trials
    # -----------------

    for t in range(0, n_trials-1):

        # For experiment 2, indicate current block type
        if which_exp == 2:
            no_push_cond = df['cond'][t] == 'main_noPush'

        # Extract noise condition
        agent.sigma = df['sigma'][t]

        # For first trial of new block
        # Futuretodo: create function to re-initialize agent on new block, maybe shared across motor and sampling too
        if df['new_block'][t]:

            # Initialize estimation uncertainty, relative uncertainty and changepoint probability
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            # Record estimation uncertainty
            sigma_t_sq[t] = agent_vars.sigma_0

            if sim:

                # Set initial bucket location, prediction, and push
                sim_z_t[t] = agent_vars.mu_0
                sim_b_t[t] = agent_vars.mu_0
                sim_y_t[t] = 0.0

        # For all other trials
        else:

            if sim and which_exp == 2:

                # For simulations, we take the actual shift in the bucket location
                sim_y_t[t] = df['y_t'][t]

                # Here we check if every trial in the shifting-bucket condition is a push trial.
                # In "edge" trials (see preprocessing), push is absent because either at 0 or 300 (the edges of the
                # screen). If we have such a trial, simulate push at this stage.
                if df['cond'][t] == 'main_push' and df['edge'][t]:
                    sim_y_t[t] = np.random.normal(0, 20, 1)

                # Ensure that push is within screen range [0, 300]
                sim_y_t[t], sim_z_t[t] = correct_push(sim_b_t[t], sim_y_t[t])

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # Record estimation uncertainty of current trial
        sigma_t_sq[t] = agent.sigma_t_sq

        # For all but last trials of a block:
        if not df['new_block'][t+1]:

            # Sequential belief update
            if sim:
                delta[t] = df['x_t'][t] - sim_b_t[t]
                agent.learn(delta[t], sim_b_t[t], df['v_t'][t], df['mu_t'][t], high_val[t])
            else:
                agent.learn(delta[t], df['b_t'][t], df['v_t'][t], df['mu_t'][t], high_val[t])

            # Record updated belief
            mu[t] = agent.mu_t

            # Record predicted update according to reduced Bayesian model
            a_hat[t] = agent.a_t

            # Record change-point probability
            omega[t] = agent.omega_t

            # Record learning rate
            alpha[t] = agent.alpha_t

            # Model bucket bias: hat{a}_t = hat{a}_t + d * y_t
            if which_exp == 2:

                if sim:
                    if df['new_block'][t]:
                        # At the beginning of a block,
                        # we assume that last prediction (from previous block) has no influence on the update
                        bucket_bias = 0.0
                    else:
                        bucket_bias = sel_coeffs[9] * sim_y_t[t]
                else:
                    if df['new_block'][t]:
                        # At the beginning of a block,
                        # we assume that last prediction (from previous block) has no influence on the update
                        bucket_bias = 0.0
                    else:
                        bucket_bias = sel_coeffs[9] * df['y_t'][t]

                # Add bucket bias to prediction update
                a_hat[t] = a_hat[t] + bucket_bias

                # Record updated belief with bucket bias
                mu_bias[t] = agent.mu_t + bucket_bias

            # Compute likelihood of updates according to reduced Bayesian model
            # -----------------------------------------------------------------

            # Compute absolute predicted update
            # |hat{a}_t|
            abs_pred_up = abs(a_hat[t])

            # Compute response noise
            # epsilon := omikron_0 + omikron_1 * |hat{t}_t|
            epsilon[t] = sel_coeffs[0] + sel_coeffs[1] * abs_pred_up  # eq. 18

            # Compute likelihood of predicted update
            # p(a_t) := N(a_t; hat{a}_t, epsilon_t^2)
            p_a_t = norm.pdf(x=df['a_t'][t], loc=a_hat[t], scale=epsilon[t])  # eq. 17

            # Adjust probability of update for numerical stability
            if p_a_t == 0.0:
                p_a_t = corrected_0_p

            # Compute negative log-likelihood of predicted update according to reduced Bayesian model
            llh_rbm[llh_counter] = np.log(p_a_t)

            # In the first experiment, we model perseveration for all trials
            # In the follow-up experiment, we model perseveration only in the no-push condition
            # eq. 20
            if which_exp == 1 or which_exp == 3:
                lambda_t = compute_persprob(sel_coeffs[2], sel_coeffs[3], abs_pred_up)
            else:
                if no_push_cond:
                    lambda_t = compute_persprob(sel_coeffs[2], sel_coeffs[3], abs_pred_up)
                else:
                    lambda_t = 0.0

            # Adjust lambda_t for numerical stability
            if lambda_t == 0.0:
                lambda_t = corrected_0_p
            elif lambda_t == 1:
                lambda_t = 1 - corrected_0_p

            # Adjust delta function for numerical stability during computation of likelihood
            # delta(a_t) := 1 if a_t = 0, 0 else
            # eq. 16
            if pers[t] == 1:
                delta_fun = 1 - corrected_0_p
            elif pers[t] == 0:
                delta_fun = corrected_0_p

            # Compute negative log-likelihood of mixture based on
            # p^{lambda_t,delta}(a_t|x_{1:T}):= lambda_t * delta(a_t) + (1-lambda_t) * p(a_t|x_{1:T})
            # eq. 19
            llh_mix[llh_counter] = logsumexp([np.log(delta_fun) + np.log(lambda_t), np.log((1 - lambda_t)) + llh_rbm[llh_counter]])

            # Simulate perseveration and updates
            if sim:

                # Randomly sample perseveration trials
                rand_pers = np.random.binomial(1, lambda_t)
                if rand_pers == 0:
                    sim_a_t[t] = np.random.normal(a_hat[t], epsilon[t], 1)
                else:
                    sim_a_t[t] = 0.0

                # Updated prediction
                sim_b_t[t+1] = sim_b_t[t] + sim_a_t[t]
                if sim_b_t[t+1] >= 300.0:
                    sim_b_t[t+1] = 300.0
                elif sim_b_t[t+1] <= 0.0:
                    sim_b_t[t+1] = 0.0

            llh_counter += 1

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype='float')
    df_data['a_t_hat'] = a_hat
    df_data['mu_t'] = mu
    df_data['mu_t_bias'] = mu_bias
    df_data['delta_t'] = delta
    df_data['omega_t'] = omega
    df_data['tau_t'] = tau
    df_data['alpha_t'] = alpha
    df_data['sigma_t_sq'] = sigma_t_sq

    if sim:

        # Save simulation-related variables
        df_data['sim_b_t'] = sim_b_t
        df_data['sim_a_t'] = sim_a_t
        df_data['sim_y_t'] = sim_y_t
        df_data['sim_z_t'] = sim_z_t
        df_data['sigma'] = df['sigma'].copy()
        df_data['cond'] = df['cond'].copy()
        df_data['age_group'] = df['age_group'].copy()
        df_data['new_block'] = df['new_block'].copy()
        df_data['x_t'] = df['x_t'].copy()
        df_data['v_t'] = df['v_t'].copy()
        df_data['r_t'] = df['r_t'].copy()

    return llh_mix, df_data

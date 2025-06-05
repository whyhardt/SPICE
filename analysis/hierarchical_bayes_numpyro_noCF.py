import sys, os

import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import argparse
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.utils.convert_dataset import convert_dataset
from spice.resources.rnn_utils import split_data_along_timedim


# @jax.0(static_argnames=['model','hierarchical'])
def rl_model(model, choice, reward, hierarchical):
    """
    A reinforcement learning model with optional hierarchical structure for parameter inference.

    Args:
        model (list of int): A binary vector indicating which parameters to include in the model. 
            - `model[0]`: Include individual positive learning rate (`alpha_pos`).
            - `model[1]`: Include individual negative learning rate (`alpha_neg`).
            - `model[2]`: Include counterfactual positive learning rate (`alpha_cf_pos`).
            - `model[3]`: Include counterfactual negative learning rate (`alpha_cf_neg`).
            - `model[4]`: Include choice perseverance (`alpha_ch`).
            - `model[5]`: Include reward sensitivity scaling (`beta_ch`).
            - `model[6]`: Include inverse temperature for reward sensitivity (`beta_r`).
            - `model[7]`: Include counterfactual sensitivity scaling (`beta_cf`).
        choice (jnp.ndarray): A tensor of shape `(T, N, 2)` where `T` is the number of time steps, `N` 
            is the number of participants, and the last dimension represents binary choices (1 for selected, 0 for not selected).
        reward (jnp.ndarray): A tensor of shape `(T, N)` where `T` is the number of time steps and `N` 
            is the number of participants, containing binary reward outcomes (1 for reward, 0 for no reward).
        hierarchical (int): A flag indicating whether to use hierarchical Bayesian inference. 
            - `1`: Use hierarchical priors (group-level and individual-level parameters).
            - `0`: Use non-hierarchical priors (individual-level parameters only).

    Returns:
        None: The function is designed to be used with NumPyro for sampling model parameters and 
        posterior predictive distributions. Observations are modeled using a Bernoulli likelihood.

    """
    
    def scaled_beta(a, b, low, high):
        return dist.TransformedDistribution(
            dist.Beta(a, b),  # Beta distribution in [0, 1]
            dist.transforms.AffineTransform(0, high - low)  # Scale to e.g. [low=0, high=10]
            )
    beta_scaling = 15
    
    if hierarchical == 1:
        # # Priors for group-level parameters
        # # alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Uniform(low=0.01, high=0.99)) if model[0]==1 else 1
        # # alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Uniform(low=0.01, high=0.99)) if model[1]==1 else -1
        # # alpha_ch_mean = numpyro.sample("alpha_ch_mean", dist.Uniform(low=0.01, high=0.99)) if model[2]==1 else 1
        # # beta_ch_mean = numpyro.sample("beta_ch_mean", dist.Uniform(low=0.01, high=0.99)) if model[3]==1 else 0
        # # beta_r_mean = numpyro.sample("beta_r_mean", dist.Uniform(low=0.01, high=9.99)) if model[4]==1 else 1
        # alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Beta(2, 2)) if model[0]==1 else 1
        # alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Beta(2, 2)) if model[1]==1 else -1
        # alpha_ch_mean = numpyro.sample("alpha_ch_mean", dist.Beta(2, 2)) if model[2]==1 else 1
        # beta_ch_mean = numpyro.sample("beta_ch_mean", scaled_beta(2, 2, 0, 10)) if model[3]==1 else 0
        # beta_r_mean = numpyro.sample("beta_r_mean", scaled_beta(2, 2, 0, 10)) if model[4]==1 else 1
        
        # # Priors for individual-level variation (hierarchical)
        # alpha_pos_std = numpyro.sample("alpha_pos_std", dist.HalfNormal(0.3)) if model[0]==1 else 0
        # alpha_neg_std = numpyro.sample("alpha_neg_std", dist.HalfNormal(0.3)) if model[1]==1 else 0
        # alpha_ch_std = numpyro.sample("alpha_ch_std", dist.HalfNormal(0.3))  if model[2]==1 else 0
        # beta_ch_std = numpyro.sample("beta_ch_std", dist.HalfNormal(1)) if model[3]==1 else 0
        # beta_r_std = numpyro.sample("beta_r_std", dist.HalfNormal(1)) if model[4]==1 else 0
        # # alpha_pos_std = numpyro.sample("alpha_pos_std", dist.Beta(2, 2)) if model[0]==1 else 0
        # # alpha_neg_std = numpyro.sample("alpha_neg_std", dist.Beta(2, 2)) if model[1]==1 else 0
        # # alpha_ch_std = numpyro.sample("alpha_ch_std", dist.Beta(2, 2))  if model[2]==1 else 0
        # # beta_ch_std = numpyro.sample("beta_ch_std", scaled_beta(2, 2, 0, 10)) if model[3]==1 else 0
        # # beta_r_std = numpyro.sample("beta_r_std", scaled_beta(2, 2, 0, 10)) if model[4]==1 else 0

        # # Individual-level parameters
        # alpha_neg = None
        # with numpyro.plate("participants", choice.shape[1]):
        #     alpha_pos = numpyro.sample("alpha_pos", dist.TruncatedNormal(alpha_pos_mean, alpha_pos_std, low=0.01, high=0.99))[:, None] if model[0]==1 else 1
        #     if model[1]==1:
        #         alpha_neg = numpyro.sample("alpha_neg", dist.TruncatedNormal(alpha_neg_mean, alpha_neg_std, low=0.01, high=0.99))[:, None]
        #     alpha_ch = numpyro.sample("alpha_ch", dist.TruncatedNormal(alpha_ch_mean, alpha_ch_std, low=0.01, high=0.99))[:, None] if model[2]==1 else 1
        #     beta_ch = numpyro.sample("beta_ch", dist.TruncatedNormal(beta_ch_mean, beta_ch_std, low=0.01, high=0.99)) if model[3]==1 else 0
        #     beta_r = numpyro.sample("beta_r", dist.TruncatedNormal(beta_r_mean, beta_r_std, low=0.01, high=9.99)) if model[4]==1 else 1
            
        # if model[1]==0:
        #     alpha_neg = alpha_pos
        
        # Priors for group-level parameters
        alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Beta(1, 1)) if model[0]==1 else 1
        alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Beta(1, 1)) if model[1]==1 else -1
        alpha_ch_mean = numpyro.sample("alpha_ch_mean", dist.Beta(1, 1)) if model[2]==1 else 1
        beta_ch_mean = numpyro.sample("beta_ch_mean", dist.Beta(1, 1)) if model[3]==1 else 0
        beta_r_mean = numpyro.sample("beta_r_mean", dist.Beta(1, 1)) if model[4]==1 else 1
        
        # Priors for individual-level variation (hierarchical)
        alpha_pos_kappa = numpyro.sample("alpha_pos_kappa", dist.HalfNormal(1.0)) if model[0]==1 else 0
        alpha_neg_kappa = numpyro.sample("alpha_neg_kappa", dist.HalfNormal(1.0)) if model[1]==1 else 0
        alpha_ch_kappa = numpyro.sample("alpha_ch_kappa", dist.HalfNormal(1.0))  if model[2]==1 else 0
        beta_ch_kappa = numpyro.sample("beta_ch_kappa", dist.HalfNormal(1.0)) if model[3]==1 else 0
        beta_r_kappa = numpyro.sample("beta_r_kappa", dist.HalfNormal(1.0)) if model[4]==1 else 0
        
        # Individual-level parameters
        n_participants = choice.shape[1]
        with numpyro.plate("participants", n_participants):
            # Sample individual-level parameters or assign fixed values
            
            if model[0]:
                alpha_pos = numpyro.sample("alpha_pos", dist.Beta(alpha_pos_mean * alpha_pos_kappa, (1 - alpha_pos_mean) * alpha_pos_kappa))[:, None]
            else:
                alpha_pos = jnp.full((n_participants, 1), 1.0)

            if model[1]:
                alpha_neg = numpyro.sample("alpha_neg", dist.Beta(alpha_neg_mean * alpha_neg_kappa, (1 - alpha_neg_mean) * alpha_neg_kappa))[:, None]
            else:
                alpha_neg = alpha_pos  # share value with alpha_pos
                
            if model[2]:
                alpha_ch = numpyro.sample("alpha_ch", dist.Beta(alpha_ch_mean * alpha_ch_kappa, (1 - alpha_ch_mean) * alpha_ch_kappa))[:, None]
            else:
                alpha_ch = jnp.full((n_participants, 1), 1.0)

            if model[3]:
                # beta_ch = numpyro.sample("beta_ch", scaled_beta(beta_ch_mean * beta_ch_kappa, (1 - beta_ch_mean) * beta_ch_kappa, 0, 15))[:, None]
                beta_ch = numpyro.sample("beta_ch", dist.Beta(beta_ch_mean * beta_ch_kappa, (1 - beta_ch_mean) * beta_ch_kappa))[:, None] * beta_scaling
            else:
                beta_ch = jnp.full((n_participants, 1), 0.0)

            if model[4]:
                # beta_r = numpyro.sample("beta_r", scaled_beta(beta_r_mean * beta_r_kappa, (1 - beta_r_mean) * beta_r_kappa, 0, 15))[:, None]
                beta_r = numpyro.sample("beta_r", dist.Beta(beta_r_mean * beta_r_kappa, (1 - beta_r_mean) * beta_r_kappa))[:, None] * beta_scaling
            else:
                beta_r = jnp.full((n_participants, 1), 1.0)
                            
    else:
        # Basic bayesian inference (not hierarchical)
        alpha_pos = numpyro.sample("alpha_pos", dist.Beta(1, 1)) if model[0]==1 else 1
        alpha_neg = numpyro.sample("alpha_neg", dist.Beta(1, 1)) if model[1]==1 else alpha_pos
        alpha_ch = numpyro.sample("alpha_ch", dist.Beta(1, 1)) if model[2]==1 else 1
        beta_ch = numpyro.sample("beta_ch", scaled_beta(1, 1, 0, 15)) if model[3]==1 else 0
        beta_r = numpyro.sample("beta_r", scaled_beta(1, 1, 0, 15)) if model[4]==1 else 1
        
    def update(carry, x):#, alpha_pos, alpha_neg, alpha_c, beta_r, beta_ch):
        r_values = carry[0]
        c_values = carry[1]
        ch, rw = x[:, :2], x[:, 2][:, None]
        
        # Compute prediction errors for each outcome
        rpe = (rw - r_values) * ch
        cpe = ch - c_values
        
        # Update Q-values
        lr = jnp.where(rw > 0.5, alpha_pos, alpha_neg)
        r_values = r_values + lr * rpe
        c_values = c_values + alpha_ch * cpe
        
        # compute the action probability of option 0
        r_diff = (r_values[:, 0] - r_values[:, 1]).reshape(-1, 1)
        c_diff = (c_values[:, 0] - c_values[:, 1]).reshape(-1, 1)
        action_prob_option_0 = jax.nn.sigmoid(beta_r * r_diff + beta_ch * c_diff).reshape(-1)
        
        return (r_values, c_values), action_prob_option_0
    
    # Define initial Q-values and initialize the previous choice variable
    r_values = jnp.full((choice.shape[1], 2), 0.5)
    c_values = jnp.zeros((choice.shape[1], 2))
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    # _, ys = jax.lax.scan(update, (r_values, c_values), xs)
    
    ys = jnp.zeros((choice.shape[0]-1, r_values.shape[0]))
    carry = (r_values, c_values)
    for i in range(len(choice)-1):
        carry, y = update(carry, xs[i])
        ys = ys.at[i].set(y)
    
    # Use numpyro.plate for sampling
    next_choice_0 = choice[1:, :, 0]  # show whenever option 0 was selected
    valid_mask = (next_choice_0 >= 0) & (next_choice_0 <= 1)
    # Apply the mask to the observations
    if hierarchical == 1:
        with numpyro.handlers.mask(mask=valid_mask):
            with numpyro.plate("participants", choice.shape[1], dim=-1):
                with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
                    numpyro.sample("obs", dist.Bernoulli(probs=ys), obs=next_choice_0)
    else:
        with numpyro.handlers.mask(mask=valid_mask.flatten()):
            numpyro.sample("obs", dist.Bernoulli(probs=ys.flatten()), obs=next_choice_0.flatten())


def encode_model_name(model: str, model_parts: list) -> np.ndarray:
    enc = np.zeros((len(model_parts),))
    for i in range(len(model_parts)):
        if model_parts[i] in model:
            enc[i] = 1
    return enc


def main(file: str, model: str, num_samples: int, num_warmup: int, num_chains: int, hierarchical: bool, output_file: str, checkpoint: bool, train_test_ratio: float = 1.):
    # jax.config.update('jax_disable_jit', True)
    
    # set output file
    output_file = output_file.split('.')[0] + '_' + model + '.nc'
    
    # Check model str
    valid_config = ['Ap', 'An', 'Ach', 'Bch', 'Br']
    model_checked = '' + model
    for c in valid_config:
        model_checked = model_checked.replace(c, '')
    if len(model_checked) > 0:
        raise ValueError(f'The provided model {model} is not supported. At least some part of the configuration ({model_checked}) is not valid. Valid configurations may include {valid_config}.')
    
    # Get and prepare the data
    # data = pd.read_csv(file)
    data = convert_dataset(file)[0]
    data = split_data_along_timedim(dataset=data, split_ratio=train_test_ratio)[0].xs.numpy()
    choices = data[..., :2]
    rewards = np.max(data[..., 2:4], axis=-1, keepdims=True) # !!! that's not applicable to experiments with counterfactual feedback !!!
    
    # # get all different sessions
    # sessions = data['session'].unique()
    # # get maximum number of trials per session
    # max_trials = int(data.groupby('session').size().max() * train_test_ratio)
    # # sort values into session-grouped arrays
    # choices = np.zeros((len(sessions), max_trials, 2)) - 1
    # rewards = np.zeros((len(sessions), max_trials, 1)) - 1
    # for i, s in enumerate(sessions):
    #     choice = data[data['session'] == s]['choice'].values.astype(int)
    #     reward = data[data['session'] == s]['reward'].values
    #     index_train = int(len(choice) * train_test_ratio)
    #     choices[i, :index_train] = np.eye(2)[choice[:index_train]]
    #     rewards[i, :index_train, 0] = reward[:index_train]
    
    # Run the model
    numpyro.set_host_device_count(num_chains)
    print(f'Number of devices: {jax.device_count()}')
    kernel = infer.NUTS(rl_model)
    if checkpoint and num_warmup > 0:
        print(f'Checkpoint was set but num_warmup>0 ({num_warmup}). Setting num_warmup=0.')
        num_warmup = 0
    mcmc = infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    print('Initialized MCMC model.')
    if checkpoint:
        with open(output_file, 'rb') as file:
            checkpoint = pickle.load(file)
        mcmc.post_warmup_state = checkpoint.last_state
        rng_key = mcmc.post_warmup_state.rng_key
        print('Checkpoint loaded.')
    else:
        rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, model=tuple(encode_model_name(model, valid_config)), choice=jnp.array(choices.swapaxes(1, 0)), reward=jnp.array(rewards.swapaxes(1, 0)), hierarchical=hierarchical)

    with open(output_file, 'wb') as file:
        pickle.dump(mcmc, file)
    
    return mcmc

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Performs a hierarchical bayesian parameter inference with numpyro.')
  
    parser.add_argument('--file', type=str, help='Dataset of a 2-armed bandit task with columns (session, choice, reward)')
    parser.add_argument('--model', type=str, default='ApAnAcBcBr', help='Model configuration (Ap: learning rate for positive outcomes, An: learning rate for negative outcomes, Ac: learning rate for choice-based value, Bc: Importance of choice-based values, Br: Importance and inverse noise termperature for reward-based values)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=500, help='Number of warmup samples (additional)')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    parser.add_argument('--hierarchical', action='store_true', help='Whether to do hierarchical inference')
    parser.add_argument('--output_file', type=str, default='benchmarking/params/traces.nc', help='Number of chains')
    parser.add_argument('--checkpoint', action='store_true', help='Whether to load the specified output file as a checkpoint')
    
    args = parser.parse_args()

    # with jax.default_device(jax.devices("cpu")[0]):
    main(args.file, args.model, args.num_samples, args.num_warmup, args.num_chains, args.hierarchical, args.output_file, args.checkpoint)
    
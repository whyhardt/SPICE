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
from typing import List, Callable, Union, Dict
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim
from resources.bandits import Agent, check_in_0_1_range
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session


def rl_update_step(r_values, c_values, choice, reward, params):
    """
    Shared update function for both agent and MCMC model.
    
    Args:
        r_values: Current reward values [..., n_actions]
        c_values: Current choice values [..., n_actions] 
        choice: Choice made (0 or 1) or one-hot encoded choice
        reward: Reward received (scalar or array)
        params: Dictionary containing learning parameters
        
    Returns:
        Updated r_values, c_values, and action probabilities
    """
    n_actions = r_values.shape[-1]
    
    # Handle both scalar choice and one-hot encoded choice
    if isinstance(choice, (int, np.integer)) or choice.ndim == 0:
        choice_onehot = jnp.eye(n_actions)[choice]
    else:
        choice_onehot = choice
    
    # Ensure reward has proper shape for broadcasting
    if reward.ndim == r_values.ndim - 1:
        # Add last dimension to match r_values shape
        reward = jnp.expand_dims(reward, axis=-1)
    elif reward.ndim == r_values.ndim and reward.shape[-1] != n_actions:
        # If reward has same ndim but wrong last dimension, assume it's batched scalar
        reward = jnp.expand_dims(reward[..., 0], axis=-1)
    
    # Broadcast reward to match r_values shape
    reward = jnp.broadcast_to(reward, r_values.shape)
    
    # Expand parameter dimensions to match r_values for broadcasting
    def expand_param(param):
        if jnp.isscalar(param) or param.ndim == 0:
            return param
        else:
            # Add dimensions to match r_values shape
            while param.ndim < r_values.ndim:
                param = jnp.expand_dims(param, axis=-1)
            return jnp.broadcast_to(param, r_values.shape)
    
    # Reward-based updates
    alpha_pos_expanded = expand_param(params['alpha_pos'])
    alpha_neg_expanded = expand_param(params['alpha_neg'])
    alpha_cf_pos_expanded = expand_param(params['alpha_cf_pos'])
    alpha_cf_neg_expanded = expand_param(params['alpha_cf_neg'])
    
    alpha_reward = jnp.where(reward > 0.5, alpha_pos_expanded, alpha_neg_expanded)
    alpha_cf = jnp.where(reward > 0.5, alpha_cf_pos_expanded, alpha_cf_neg_expanded)
    
    # Reward prediction errors
    rpe_factual = reward - r_values
    rpe_counterfactual = (1 - reward) - r_values
    
    # Apply factual and counterfactual updates
    factual_update = alpha_reward * rpe_factual * choice_onehot
    counterfactual_update = alpha_cf * rpe_counterfactual * (1 - choice_onehot) * params['beta_cf']
    
    r_values_new = r_values + factual_update + counterfactual_update
    
    # Choice perseverance update
    alpha_ch_expanded = expand_param(params['alpha_ch'])
    cpe = choice_onehot - c_values
    c_values_new = c_values + alpha_ch_expanded * cpe
    
    # Compute action probabilities
    beta_r = params['beta_r']
    beta_ch = params['beta_ch']
    
    # Ensure proper broadcasting for the final calculation
    r_diff = r_values_new[..., 0] - r_values_new[..., 1]
    c_diff = c_values_new[..., 0] - c_values_new[..., 1]
    
    # Handle parameter broadcasting for final computation
    if jnp.isscalar(beta_r) or beta_r.ndim == 0:
        beta_r_term = beta_r * r_diff
    else:
        beta_r_term = beta_r * r_diff
        
    if jnp.isscalar(beta_ch) or beta_ch.ndim == 0:
        beta_ch_term = beta_ch * c_diff
    else:
        beta_ch_term = beta_ch * c_diff
    
    action_prob_0 = jax.nn.sigmoid(beta_r_term + beta_ch_term)
    
    return r_values_new, c_values_new, action_prob_0


class Agent_eckstein2022(Agent):
    """An agent that runs Q-learning for the two-armed bandit task."""

    def __init__(
        self,
        n_actions: int = 2,
        beta_reward: float = 3.,
        alpha_reward: float = 0.2,
        alpha_penalty: float = -1,
        beta_counterfactual: float = 0.,
        alpha_counterfactual_reward: float = 0.,
        alpha_counterfactual_penalty: float = 0.,
        beta_choice: float = 0.,
        alpha_choice: float = 1.,
        deterministic: bool = True,
    ):
        super().__init__(n_actions=n_actions, deterministic=deterministic)
        
        self._n_actions = n_actions
        self._q_init = 0.5
        
        # Store parameters in format expected by shared update function
        self._params = {
            'alpha_pos': alpha_reward,
            'alpha_neg': alpha_penalty if alpha_penalty >= 0 else alpha_reward,
            'alpha_cf_pos': alpha_counterfactual_reward,
            'alpha_cf_neg': alpha_counterfactual_penalty,
            'alpha_ch': alpha_choice,
            'beta_r': beta_reward,
            'beta_ch': beta_choice,
            'beta_cf': beta_counterfactual,
        }
        
        self.new_sess()
        
        # Validation
        check_in_0_1_range(alpha_reward, 'alpha')
        if alpha_penalty >= 0:
            check_in_0_1_range(alpha_penalty, 'alpha_penalty')
        check_in_0_1_range(alpha_counterfactual_reward, 'alpha_countefactual_reward')
        check_in_0_1_range(alpha_counterfactual_penalty, 'alpha_countefactual_penalty')
        check_in_0_1_range(alpha_choice, 'alpha_choice')

    def update(self, choice: int, reward: np.ndarray, *args, **kwargs):
        """Update the agent after one step using shared update logic."""
        # Extract the reward for the chosen action, or use scalar reward
        if isinstance(reward, np.ndarray) and reward.size > 1:
            reward_value = reward[choice]
        else:
            reward_value = reward.item() if hasattr(reward, 'item') else reward
        
        # Use shared update function
        r_values_new, c_values_new, _ = rl_update_step(
            self._state['x_value_reward'].reshape(-1),
            self._state['x_value_choice'].reshape(-1),
            choice,
            reward_value,
            self._params
        )
        
        # Update state
        self._state['x_value_reward'] = r_values_new
        self._state['x_value_choice'] = c_values_new

    @property
    def q(self):
        return (self._state['x_value_reward'] * self._params['beta_r'] + 
                self._state['x_value_choice'] * self._params['beta_ch']).reshape(-1)

    @property
    def q_choice(self):
        return (self._state['x_value_choice'] * self._params['beta_ch']).reshape(-1)
    

def setup_agent_benchmark(path_model: str, model_config: str, deterministic: bool = True, **kwargs) -> List[Agent_eckstein2022]:
    """Setup MCMC agents using the shared Agent class."""
    
    with open(path_model, 'rb') as file:
        mcmc = pickle.load(file)
    
    n_sessions = mcmc.get_samples()[list(mcmc.get_samples().keys())[0]].shape[-1]
    model_name = path_model.split('_')[-1].split('.')[0]
    
    agents = []
    
    for session in range(n_sessions):
        parameters = {
            'alpha_pos': 1,
            'alpha_neg': -1,
            'alpha_cf_pos': 0,
            'alpha_cf_neg': 0,
            'alpha_ch': 1,
            'beta_ch': 0,
            'beta_r': 1,
        }
        
        for param in parameters:
            if param in mcmc.get_samples():
                samples = mcmc.get_samples()[param]
                if len(samples.shape) == 2:
                    samples = samples[:, session]
                parameters[param] = np.mean(samples, axis=0)
        
        if np.mean(parameters['alpha_neg']) == -1:
            parameters['alpha_neg'] = parameters['alpha_pos']
        
        if np.mean(parameters['alpha_cf_pos']) == 0 and 'Bcf' in model_name:
            parameters['alpha_cf_pos'] = parameters['alpha_pos']
        
        if np.mean(parameters['alpha_cf_neg']) == 0 and 'Acfp' in model_name:
            parameters['alpha_cf_neg'] = parameters['alpha_cf_pos']
        elif np.mean(parameters['alpha_cf_neg']) == 0 and 'Bcf' in model_name:
            parameters['alpha_cf_neg'] = parameters['alpha_neg']
            
        agents.append(Agent_eckstein2022(
            alpha_reward=parameters['alpha_pos'],
            alpha_penalty=parameters['alpha_neg'],
            alpha_counterfactual_reward=parameters['alpha_cf_pos']*(1 if 'Bcf' in model_name else 0),
            alpha_counterfactual_penalty=parameters['alpha_cf_neg']*(1 if 'Bcf' in model_name else 0),
            alpha_choice=parameters['alpha_ch'],
            beta_reward=parameters['beta_r']*15,
            beta_choice=parameters['beta_ch']*15,
            beta_counterfactual=1.0 if 'Bcf' in model_name else 0.0,
            deterministic=deterministic,
        ))
    
    n_parameters = 0
    for letter in model_config:
        if not letter.islower():
            n_parameters += 1
    if 'Bcf' in model_config:
        # remove again one parameter because Bcf is not trained
        n_parameters -= 1
            
    return agents, n_parameters


def rl_model(model, choice, reward):
    """
    A reinforcement learning model using shared update logic for parameter inference.
    """
    def scaled_beta(a, b, low, high):
        return dist.TransformedDistribution(
            dist.Beta(a, b),
            dist.transforms.AffineTransform(0, high - low)
        )
    
    beta_scaling = 15
    
    # Hierarchical priors
    alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Beta(1, 1)) if model[0]==1 else 1
    alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Beta(1, 1)) if model[1]==1 else -1
    alpha_cf_pos_mean = numpyro.sample("alpha_cf_pos_mean", dist.Beta(1, 1)) if model[2]==1 else 0
    alpha_cf_neg_mean = numpyro.sample("alpha_cf_neg_mean", dist.Beta(1, 1)) if model[3]==1 else 0
    alpha_ch_mean = numpyro.sample("alpha_ch_mean", dist.Beta(1, 1)) if model[4]==1 else 1
    beta_ch_mean = numpyro.sample("beta_ch_mean", dist.Beta(1, 1)) if model[5]==1 else 0
    beta_r_mean = numpyro.sample("beta_r_mean", dist.Beta(1, 1)) if model[6]==1 else 1
    
    # Hierarchical variation parameters
    alpha_pos_kappa = numpyro.sample("alpha_pos_kappa", dist.HalfNormal(1.0)) if model[0]==1 else 0
    alpha_neg_kappa = numpyro.sample("alpha_neg_kappa", dist.HalfNormal(1.0)) if model[1]==1 else 0
    alpha_cf_pos_kappa = numpyro.sample("alpha_cf_pos_kappa", dist.HalfNormal(1.0)) if model[2]==1 else 0
    alpha_cf_neg_kappa = numpyro.sample("alpha_cf_neg_kappa", dist.HalfNormal(1.0)) if model[3]==1 else 0
    alpha_ch_kappa = numpyro.sample("alpha_ch_kappa", dist.HalfNormal(1.0))  if model[4]==1 else 0
    beta_ch_kappa = numpyro.sample("beta_ch_kappa", dist.HalfNormal(1.0)) if model[5]==1 else 0
    beta_r_kappa = numpyro.sample("beta_r_kappa", dist.HalfNormal(1.0)) if model[6]==1 else 0
    
    # Individual parameters
    n_participants = choice.shape[1]
    with numpyro.plate("participants", n_participants):
        if model[0]:
            alpha_pos = numpyro.sample("alpha_pos", dist.Beta(alpha_pos_mean * alpha_pos_kappa, (1 - alpha_pos_mean) * alpha_pos_kappa))[:, None]
        else:
            alpha_pos = jnp.full((n_participants, 1), 1.0)

        if model[1]:
            alpha_neg = numpyro.sample("alpha_neg", dist.Beta(alpha_neg_mean * alpha_neg_kappa, (1 - alpha_neg_mean) * alpha_neg_kappa))[:, None]
        else:
            alpha_neg = alpha_pos

        if model[2]:
            alpha_cf_pos = numpyro.sample("alpha_cf_pos", dist.Beta(alpha_cf_pos_mean * alpha_cf_pos_kappa, (1 - alpha_cf_pos_mean) * alpha_cf_pos_kappa))[:, None]
        else:
            alpha_cf_pos = alpha_pos

        if model[3]:
            alpha_cf_neg = numpyro.sample("alpha_cf_neg", dist.Beta(alpha_cf_neg_mean * alpha_cf_neg_kappa, (1 - alpha_cf_neg_mean) * alpha_cf_neg_kappa))[:, None]
        elif not model[3] and model[2]:
            alpha_cf_neg = alpha_cf_pos
        else:
            alpha_cf_neg = alpha_neg

        if model[4]:
            alpha_ch = numpyro.sample("alpha_ch", dist.Beta(alpha_ch_mean * alpha_ch_kappa, (1 - alpha_ch_mean) * alpha_ch_kappa))[:, None]
        else:
            alpha_ch = jnp.full((n_participants, 1), 1.0)

        if model[5]:
            beta_ch = numpyro.sample("beta_ch", dist.Beta(beta_ch_mean * beta_ch_kappa, (1 - beta_ch_mean) * beta_ch_kappa))[:, None] * beta_scaling
        else:
            beta_ch = jnp.full((n_participants, 1), 0.0)
            
        if model[6]:
            beta_r = numpyro.sample("beta_r", dist.Beta(beta_r_mean * beta_r_kappa, (1 - beta_r_mean) * beta_r_kappa))[:, None] * beta_scaling
        else:
            beta_r = jnp.full((n_participants, 1), 1.0)

        beta_cf = 1.0 if model[7] else 0.0
        
    def update(carry, x):
        r_values, c_values = carry
        ch, rw = x[:, :2], x[:, 2][:, None]
        
        # Create parameter dict for shared function
        # Ensure all parameters have compatible shapes
        # Parameters already have shape (n_participants, 1), squeeze the last dim
        params = {
            'alpha_pos': alpha_pos.squeeze(-1),
            'alpha_neg': alpha_neg.squeeze(-1),
            'alpha_cf_pos': alpha_cf_pos.squeeze(-1),
            'alpha_cf_neg': alpha_cf_neg.squeeze(-1),
            'alpha_ch': alpha_ch.squeeze(-1),
            'beta_r': beta_r.squeeze(-1),
            'beta_ch': beta_ch.squeeze(-1),
            'beta_cf': beta_cf,
        }
        
        # Use shared update function
        r_values_new, c_values_new, action_prob_0 = rl_update_step(
            r_values, c_values, ch, rw.squeeze(-1), params
        )
        
        # Ensure action_prob_0 has the right shape (flatten if needed)
        if action_prob_0.ndim > 1:
            action_prob_0 = action_prob_0.reshape(-1)
        
        return (r_values_new, c_values_new), action_prob_0

    # Initialize and run updates
    r_values = jnp.full((choice.shape[1], 2), 0.5)
    c_values = jnp.zeros((choice.shape[1], 2))
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    carry = (r_values, c_values)
    
    # ys = jnp.zeros((choice.shape[0]-1, r_values.shape[0]))
    # for i in range(len(choice)-1):
    #     carry, y = update(carry, xs[i])
    #     ys = ys.at[i].set(y)

    final_carry, ys = jax.lax.scan(update, carry, xs)
    
    # Likelihood
    next_choice_0 = choice[1:, :, 0]
    valid_mask = (next_choice_0 >= 0) & (next_choice_0 <= 1)
    
    
    with numpyro.handlers.mask(mask=valid_mask):
        with numpyro.plate("participants", choice.shape[1], dim=-1):
            with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
                numpyro.sample("obs", dist.Bernoulli(probs=ys), obs=next_choice_0)


def encode_model_name(model: str, model_parts: list) -> np.ndarray:
    enc = np.zeros((len(model_parts),))
    for i in range(len(model_parts)):
        if model_parts[i] in model:
            enc[i] = 1
    return enc


def fit_mcmc(data: str, model: str, num_samples: int, num_warmup: int, num_chains: int, output_dir: str, checkpoint: bool, train_test_ratio: float = 1.):
    # Set output file
    output_file = os.path.join(output_dir, 'mcmc_eckstein2022_'+model+'.nc')
    
    # Check model string
    valid_config = ['Ap', 'An', 'Acfp', 'Acfn', 'Ach', 'Bch', 'Br', 'Bcf']
    model_checked = '' + model
    for c in valid_config:
        model_checked = model_checked.replace(c, '')
    if len(model_checked) > 0:
        raise ValueError(f'The provided model {model} is not supported. At least some part of the configuration ({model_checked}) is not valid. Valid configurations may include {valid_config}.')
    
    # Get and prepare the data
    dataset = convert_dataset(data)[0]
    if isinstance(train_test_ratio, float):
        dataset = split_data_along_timedim(dataset=dataset, split_ratio=train_test_ratio)[0].xs.numpy()
    else:
        dataset = split_data_along_sessiondim(dataset=dataset, list_test_sessions=train_test_ratio)[0].xs.numpy()
    choices = dataset[..., :2]
    rewards = np.max(dataset[..., 2:4], axis=-1, keepdims=True)
    
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
        with open(output_file, 'rb') as data:
            checkpoint = pickle.load(data)
        mcmc.post_warmup_state = checkpoint.last_state
        rng_key = mcmc.post_warmup_state.rng_key
        print('Checkpoint loaded.')
    else:
        rng_key = jax.random.PRNGKey(0)
        
    mcmc.run(rng_key, model=tuple(encode_model_name(model, valid_config)), choice=jnp.array(choices.swapaxes(1, 0)), reward=jnp.array(rewards.swapaxes(1, 0)))

    with open(output_file, 'wb') as data:
        pickle.dump(mcmc, data)
    
    return mcmc


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Performs a hierarchical bayesian parameter inference with numpyro.')

    parser.add_argument('--data', type=str, default='data/eckstein2022/eckstein2022.csv', help='Dataset of a 2-armed bandit task with columns (session, choice, reward)')
    parser.add_argument('--model', type=str, default='ApBr', help='Model configuration (Ap: learning rate for positive outcomes, An: learning rate for negative outcomes, Ac: learning rate for choice-based value, Bc: Importance of choice-based values, Br: Importance and inverse noise termperature for reward-based values)')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=1000, help='Number of warmup samples (additional)')
    parser.add_argument('--num_chains', type=int, default=2, help='Number of chains')
    parser.add_argument('--output_dir', type=str, default='benchmarking/params', help='Output directory')
    parser.add_argument('--checkpoint', action='store_true', help='Whether to load the specified output file as a checkpoint')
    parser.add_argument('--train_test_ratio', type=float, default=1.0, help='Relative training set size of the total number of samples')

    args = parser.parse_args()

    mcmc = fit_mcmc(
        data=args.data, 
        model=args.model, 
        num_samples=args.num_samples, 
        num_warmup=args.num_warmup, 
        num_chains=args.num_chains, 
        output_dir=args.output_dir, 
        checkpoint=args.checkpoint, 
        train_test_ratio=args.train_test_ratio,
        )

    agent_mcmc = setup_agent_benchmark(
        path_model=os.path.join(args.output_dir, 'mcmc_eckstein2022_'+args.model+'.nc'),
        model_config=args.model,
        )
    experiment = convert_dataset(args.file)[0].xs[0]
    fig, axs = plot_session(agents={'benchmark': agent_mcmc[0][0]}, experiment=experiment)
    plt.show()
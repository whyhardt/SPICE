from typing import NamedTuple, Union, Optional, Dict, Tuple, List, Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import torch
from tqdm import tqdm

from .rnn import BaseRNN, ParameterModule
from .spice_utils import SpiceDataset

# Setup so that plots will look nice
small = 15
medium = 18
large = 20
plt.rc('axes', titlesize=large)
plt.rc('axes', labelsize=medium)
plt.rc('xtick', labelsize=small)
plt.rc('ytick', labelsize=small)
plt.rc('legend', fontsize=small)
plt.rc('figure', titlesize=large)
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'

###################################
# CONVENIENCE FUNCTIONS.          #
###################################


def check_in_0_1_range(x, name):
  if not (0 <= x <= 1):
    raise ValueError(
        f'Value of {name} must be in [0, 1] range. Found value of {x}.')


###################################
# GENERATIVE FUNCTIONS FOR AGENTS #
###################################

class Agent:
  """An agent that runs simple Q-learning for the two-armed bandit task.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

  def __init__(
      self,
      n_actions: int = 2,
      beta_reward: float = 3.,
      alpha_reward: float = 0.2,
      parameter_variance: Union[Dict[str, float], float] = 0.,
      deterministic: bool = True,
      ):
    """Update the agent after one step of the task.
    
    Args:
      alpha_reward (float): Baseline learning rate between 0 and 1.
      beta_reward (float): softmax inverse noise temperature. Regulates the noise in the decision-selection.
      n_actions: number of actions (default=2)
      parameter_variance (float): sets a variance around the model parameters' mean values to sample from a normal distribution e.g. at each new session. 0: no variance, -1: std = mean
    """
    
    self._list_params = ['beta_reward', 'alpha_reward']
    
    self._mean_beta_reward = beta_reward
    self._mean_alpha_reward = alpha_reward
    
    self._alpha_reward = alpha_reward
    
    self.betas = {}
    self.betas['value_reward'] = beta_reward
    
    self.n_actions = n_actions
    self._q_init = 0.5
    self._parameter_variance = self.check_parameter_variance(parameter_variance)
    self._deterministic = deterministic
    
    check_in_0_1_range(alpha_reward, 'alpha')
    
  def check_parameter_variance(self, parameter_variance):
    if isinstance(parameter_variance, float):
      par_var_dict = {}
      for key in self._list_params:
        par_var_dict[key] = parameter_variance
      parameter_variance = par_var_dict
    elif isinstance(parameter_variance, dict):
      # check that all keys in parameter_variance are valid
      not_valid_keys = []
      for key in parameter_variance:
        if not key in self._list_params:
          not_valid_keys.append(key)
      if len(not_valid_keys) > 0:
        raise ValueError(f'Some keys in parameter_variance are not valid ({not_valid_keys}). Valid keys are {self._list_params}')
      # check that all parameters are available - set to 0 if a parameter is not available
      for key in self._list_params:
        if not key in parameter_variance:
          parameter_variance[key] = 0.
    return parameter_variance
  
  def new_sess(self, **kwargs):
    """Reset the agent for the beginning of a new session."""
    
    self.state = {
      'value_reward': np.zeros((self.n_actions, 1)),
      'value_choice': np.zeros((self.n_actions, 1)),
      'learning_rate_reward': np.zeros((self.n_actions, 1)),
    }
    
  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    decision_variable = self.q - self.q.min()
    # decision_variable = np.clip(decision_variable, a_min=0, a_max=50)
    decision_variable = np.exp(decision_variable)
    choice_probs = decision_variable / np.sum(decision_variable)
    return choice_probs.reshape(self.n_actions)
    
  def get_choice(self):
    """Sample choice."""
    choice_probs = self.get_choice_probs()
    if self._deterministic:
      return np.argmax(choice_probs)
    else:
      return np.random.choice(self.n_actions, p=choice_probs)

  def update(self, choice: int, reward: np.ndarray, *args, **kwargs):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    
    # adjust learning rates for every received reward
    alpha = np.zeros_like(self.state['learning_rate_reward'])
    rpe = np.zeros_like(self.state['learning_rate_reward'])
    for action in range(self.n_actions):
      if action == choice:
        current_reward = reward[action] if np.min(reward) > -1 else reward[choice]
        alpha[action] = self._alpha_reward
        
        # Reward-prediction-error
        rpe[action] = current_reward - self.state['value_reward'][action]
      
    # (Counterfactual) Reward update
    reward_update = alpha * rpe
    
    # Update memory state
    self.state['value_reward'] += reward_update
    self.state['learning_rate_reward'] = alpha
  
  def get_state_value(self, state: str, multiply_with_beta: bool = True):
    if multiply_with_beta and state in self.betas:
      return (self.state[state] * self.betas[state]).reshape(self.n_actions)
    else:
      return self.state[state].reshape(self.n_actions)

  @property
  def q(self):
    return (self.state['value_reward']*self.betas['value_reward']).reshape(self.n_actions)


class AgentQ(Agent):
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

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
      forget_rate: float = 0.,
      parameter_variance: Union[Dict[str, float], float] = 0.,
      deterministic: bool = False,
      ):
    """Update the agent after one step of the task.
    
    Args:
      alpha (float): Baseline learning rate between 0 and 1.
      beta (float): softmax inverse noise temperature. Regulates the noise in the decision-selection.
      n_actions: number of actions (default=2)
      forget_rate (float): rate at which q values decay toward the initial values (default=0)
      perseverance_bias (float): rate at which q values move toward previous action (default=0)
      alpha_penalty (float): separate learning rate for negative outcomes
      parameter_variance (float): sets a variance around the model parameters' mean values to sample from a normal distribution e.g. at each new session. 0: no variance, -1: std = mean
    """
    
    super().__init__(n_actions=n_actions, parameter_variance=parameter_variance, deterministic=deterministic)
    
    self._list_params = ['beta_reward', 'alpha_reward', 'alpha_penalty', 'alpha_counterfactual', 'beta_choice', 'alpha_choice', 'forget_rate']
    
    self._mean_beta_reward = beta_reward
    self._mean_alpha_reward = alpha_reward
    self._mean_alpha_penalty = alpha_penalty if alpha_penalty >= 0 else alpha_reward
    self._mean_forget_rate = forget_rate
    self._mean_beta_choice = beta_choice
    self._mean_alpha_choice = alpha_choice
    self._mean_alpha_counterfactual_reward = alpha_counterfactual_reward
    self._mean_alpha_counterfactual_penalty = alpha_counterfactual_penalty
    
    self._alpha_reward = alpha_reward
    self._alpha_penalty = alpha_penalty if alpha_penalty >= 0 else alpha_reward
    self._forget_rate = forget_rate
    self._alpha_choice = alpha_choice
    self._alpha_counterfactual_reward = alpha_counterfactual_reward
    self._alpha_counterfactual_penalty = alpha_counterfactual_penalty
    
    self.betas['value_reward'] = beta_reward
    self.betas['value_choice'] = beta_choice
    
    self.n_actions = n_actions
    self._parameter_variance = self.check_parameter_variance(parameter_variance)
    self._q_init = 0.5
    
    self.new_sess()

    check_in_0_1_range(alpha_reward, 'alpha')
    if alpha_penalty >= 0:
      check_in_0_1_range(alpha_penalty, 'alpha_penalty')
    check_in_0_1_range(alpha_counterfactual_reward, 'alpha_countefactual_reward')
    check_in_0_1_range(alpha_counterfactual_penalty, 'alpha_countefactual_penalty')
    check_in_0_1_range(alpha_choice, 'alpha_choice')
    check_in_0_1_range(forget_rate, 'forget_rate')
      
  def new_sess(self, sample_parameters=False, **kwargs):
    """Reset the agent for the beginning of a new session."""
    # self._q = np.full(self._n_actions, self._q_init)
    # self._c = np.zeros(self._n_actions)
    # self._alpha = np.zeros(self._n_actions)
    
    self.state = {
      'value_reward': np.zeros(self.n_actions),
      'value_choice': np.zeros(self.n_actions),
      'learning_rate_reward': np.zeros(self.n_actions),
    }

  def update(self, choice: int, reward: np.ndarray, *args, **kwargs):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    
    # Reward-based updates
    non_chosen_action = np.arange(self.n_actions) != choice
    
    # adjust learning rates for every received reward
    alpha = np.zeros_like(reward)
    rpe = np.zeros_like(reward)
    for action in range(self.n_actions):
      if action == choice:
        # factual case
        alpha_r = self._alpha_reward
        alpha_p = self._alpha_penalty
      else: 
        # counterfactual case
        # counterfactual learning rate applies in case of foregone reward
        # no counterfactual learning in case of foregone penalty
        alpha_r = self._alpha_counterfactual_reward
        alpha_p = self._alpha_counterfactual_penalty
        
      # asymmetric learning rates
      current_reward = reward[action] if not any(np.isnan(reward)) else reward[choice]
      alpha[action] = alpha_p + current_reward * (alpha_r - alpha_p)
      
      # Reward-prediction-error
      r = current_reward if action==choice else 1-current_reward
      rpe[action] = r - self.state['value_reward'][action]
      
    # (Counterfactual) Reward update
    reward_update = alpha * rpe
    
    # Forgetting - restore q-values of non-chosen actions towards the initial value
    forget_update = self._forget_rate * (self._q_init - self.state['value_reward'][non_chosen_action])

    # Choice-Perseverance: Action-based updates
    cpe = np.eye(self.n_actions)[choice] - self.state['value_choice']
    
    # Update memory state
    self.state['value_reward'][non_chosen_action] += reward_update[non_chosen_action] + forget_update
    self.state['value_reward'][choice] += reward_update[choice]
    self.state['value_choice'] += self._alpha_choice * cpe
    self.state['learning_rate_reward'] = alpha

  @property
  def q(self):
    return (self.state['value_reward']*self.betas['value_reward'] + self.state['value_choice']*self.betas['value_choice']).reshape(self.n_actions)


class AgentQ_SampleParams(AgentQ):
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

  def __init__(
      self,
      n_actions: int = 2,
      beta_reward: float = 3.,
      alpha_reward: float = 0.2,
      alpha_penalty: float = -1,
      beta_choice: float = 0.,
      alpha_choice: float = 1.,
      forget_rate: float = 0.,
      parameter_variance: Union[Dict[str, float], float] = 0.,
      zero_threshold: float = 0.2,
      ):
    
    super().__init__(
      n_actions=n_actions,
      beta_reward=beta_reward,
      alpha_reward=alpha_reward,
      alpha_penalty=alpha_penalty,
      beta_choice=beta_choice,
      alpha_choice=alpha_choice,
      forget_rate=forget_rate,
      parameter_variance=parameter_variance,
      )
    
    # self._beta_distribution = beta_distribution
    self._zero_threshold = zero_threshold
  
  def new_sess(self, sample_parameters=False, **kwargs):
    """Reset the agent for the beginning of a new session."""
    
    super().new_sess()
    
    # sample new parameters
    if sample_parameters:
      # sample scaling parameters (inverse noise temperatures)
      self.betas['value_reward'], self.betas['value_choice'] = 0, 0
      while self.betas['value_reward'] <= self._zero_threshold and self.betas['value_choice'] <=  self._zero_threshold:
        
        self.betas['value_reward'] = np.random.beta(*self.compute_beta_dist_params(mean=0.5, var=self._parameter_variance['beta_reward']))
        self.betas['value_choice'] = np.random.beta(*self.compute_beta_dist_params(mean=0.5, var=self._parameter_variance['beta_choice']))
        # apply zero-threshold if applicable
        self.betas['value_reward'] = self.betas['value_reward'] * 2 * self._mean_beta_reward if self.betas['value_reward'] > self._zero_threshold else 0
        self.betas['value_choice'] = self.betas['value_choice'] * 2 * self._mean_beta_choice if self.betas['value_choice'] > self._zero_threshold else 0
      
      # sample auxiliary parameters
      self._forget_rate = np.random.beta(*self.compute_beta_dist_params(mean=self._mean_forget_rate, var=self._parameter_variance['forget_rate']))
      self._forget_rate =  self._forget_rate * (self._forget_rate > self._zero_threshold)
      
      self._alpha_choice = np.random.beta(*self.compute_beta_dist_params(mean=self._mean_alpha_choice, var=self._parameter_variance['alpha_choice']))
      self._alpha_choice = self._alpha_choice * (self._alpha_choice > self._zero_threshold)
      
      # sample learning rate; don't zero out; only check for applicability of asymmetric learning rates
      self._alpha_reward = np.random.beta(*self.compute_beta_dist_params(mean=self._mean_alpha_reward, var=self._parameter_variance['alpha_reward']))
      self._alpha_penalty = np.random.beta(*self.compute_beta_dist_params(mean=self._mean_alpha_penalty, var=self._parameter_variance['alpha_penalty']))
      if np.abs(self._alpha_reward - self._alpha_penalty) < self._zero_threshold:
        alpha_mean = np.mean((self._alpha_reward, self._alpha_penalty))
        self._alpha_reward = alpha_mean
        self._alpha_penalty = alpha_mean
        
  def compute_beta_dist_params(self, mean, var):
    n = mean * (1-mean) / var**2
    a = mean * n
    b = (1-mean) * n
    return a, b


class AgentNetwork(Agent):
  """A class that allows running a pretrained RNN as an agent.

  Attributes:
      model: A PyTorch module representing the RNN architecture
  """

  def __init__(
    self,
    model_rnn: BaseRNN,
    n_actions: int = 2,
    device = torch.device('cpu'),
    deterministic: bool = True,
    use_sindy: bool = False,
    ):
      """Initialize the agent network.

      Args:
          model: A PyTorch module representing the RNN architecture
          n_actions: number of permitted actions (default = 2)
      """
      
      super().__init__(n_actions=n_actions)

      self._deterministic = deterministic
      self._use_sindy = use_sindy
      
      self.model = model_rnn.eval(use_sindy=use_sindy).to(device)
        
  def new_sess(self, participant_id: int = 0, experiment_id: int = 0, additional_embedding_inputs: np.ndarray = torch.zeros(0), **kwargs):
    """Reset the network for the beginning of a new session."""
    if not isinstance(participant_id, torch.Tensor):
      participant_id = torch.tensor(participant_id, dtype=int, device=self.model.device)[None]
    
    self.model.init_state(batch_size=1)
    
    self._meta_data = torch.zeros((1, 2))
    self._meta_data[0, -1] = participant_id
    self._meta_data[0, -2] = experiment_id
    
    self._additional_meta_data = additional_embedding_inputs if isinstance(additional_embedding_inputs, torch.Tensor) else torch.tensor(additional_embedding_inputs, dtype=torch.float32)
    self._additional_meta_data = self._additional_meta_data.view(1, -1)
    
    self.set_state()

  def get_logit(self):
    """Return the value of the agent's current state."""
    # betas = self.get_betas()
    # if betas:
    #   logits = np.sum(
    #     np.concatenate([
    #       (self.state[key] * betas[key]).cpu().numpy() for key in self.state if key in betas 
    #       ]), 
    #     axis=0)
    # else:
    logits = np.sum(
      np.concatenate([
        self.state[key].cpu().numpy() for key in self.state if 'value' in key
        ]),
      axis=0)
    return logits

  def update(self, choice: float, reward: float, block: int = 0, additional_inputs: np.ndarray = torch.zeros(0), **kwargs):
    choice = torch.eye(self.n_actions)[int(choice)]
    xs = torch.concat([choice, torch.tensor(reward), torch.tensor(additional_inputs), torch.tensor(block).view(1), self._meta_data.view(-1)]).view(1, 1, -1).to(device=self.model.device)
    with torch.no_grad():
      self.model(xs, self.model.get_state(detach=True))
    self.set_state()
  
  def set_state(self):
    self.state = self.model.get_state()
    self.betas = self.get_betas()
    
  def get_betas(self):
    if hasattr(self.model, 'betas'):
      betas = {}
      if hasattr(self.model, 'participant_embedding'):
        participant_embedding = self.model.participant_embedding(self._meta_data[..., -1].int().to(device=self.model.device).view(1, 1))
      for key in self.model.betas:
        betas[key] = self.model.betas[key]().item() if isinstance(self.model.betas[key], ParameterModule) else self.model.betas[key](participant_embedding).item()
      return betas
    return None

  def get_participant_ids(self):
    if hasattr(self.model, 'participant_embedding'):
      return tuple(np.arange(self.model.participant_embedding.num_embeddings).tolist())
    
  def get_modules(self):
    return tuple(list(self.model.submodules_rnn.keys()))
  
  @property
  def q(self):
    return self.get_logit()
  
  def count_parameters(self) -> np.ndarray:
    return self.model.count_sindy_coefficients().detach().cpu().numpy()


################
# ENVIRONMENTS #
################


class Bandits:
  
  def __init__(self, *args, **kwargs):
    pass
  
  def step(self, choice):
    pass
  
  def new_sess(self):
    pass


class BanditsGeneral(Bandits):
    """
    A generalized form of the multi-armed bandit which can enable a drifting paradigm, 
    reversal learning, reward/no-reward and reward/penalty schedules, and arm-correlations.
    
    Features:
    - Drift: Reward probabilities change gradually over time
    - Reversals: Sudden swaps in reward probabilities (controlled by hazard rate)
    - Correlated arms: Reward probabilities can move together
    - Flexible reward schedules: Binary rewards (0/1) or penalty schedules (-1/+1)
    """
    
    def __init__(
        self,
        n_arms: int = 2,
        init_reward_prob: Optional[Iterable[float]] = None,
        drift_rate: float = 0.0,
        hazard_rate: float = 0.0,
        reward_prob_correlation: float = 0.0,
        reward_schedule: str = "binary",  # "binary" (0/1) or "penalty" (-1/+1)
        bounds: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = None,
        counterfactual: bool = False,
    ):  
        """
        Args:
            n_arms: Number of arms
            init_reward_prob: Initial reward probabilities for each arm
            drift_rate: Rate of Gaussian random walk drift (std dev per step)
            hazard_rate: Probability of reversal on each step
            reward_prob_correlation: Correlation between arm drifts (-1 to 1)
            reward_schedule: "binary" for 0/1 rewards, "penalty" for -1/+1 rewards
            bounds: Min and max values for reward probabilities
            seed: Random seed for reproducibility
            counterfactual: If True, the reward for all arms is returned
        """

        super().__init__()

        self.n_arms = n_arms
        self.drift_rate = drift_rate
        self.hazard_rate = hazard_rate
        self.reward_prob_correlation = reward_prob_correlation
        self.reward_schedule = reward_schedule
        self.bounds = bounds
        self.counterfactual = counterfactual
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Initialize reward probabilities
        if init_reward_prob is None:
            init_reward_prob = self.rng.uniform(bounds[0], bounds[1], n_arms)
        else:
            init_reward_prob = np.array(init_reward_prob)
            if len(init_reward_prob) != n_arms:
                raise ValueError(f"init_reward_prob must have length {n_arms}")
        
        self.init_reward_prob = np.array(init_reward_prob)
        self.reward_prob = self.init_reward_prob.copy()
        
        # Tracking
        self.t = 0
        self.history = {
            'choices': [],
            'rewards': [],
            'reward_probs': [self.reward_prob.copy()],
            'reversals': []
        }
        
        # For correlated drift (only for 2-armed bandits)
        if self.reward_prob_correlation != 0 and self.n_arms == 2:
            # Construct covariance matrix for bivariate normal
            self.drift_cov = np.array([
                [1, self.reward_prob_correlation],
                [self.reward_prob_correlation, 1]
            ]) * (self.drift_rate ** 2)
        else:
            self.drift_cov = None    

    def step(self, choice: int) -> Tuple[float, dict]:
          """
          Take a step in the environment: apply drift/reversals, then generate reward for the chosen arm.
          
          Args:
              choice: Index of chosen arm (0 to n_arms-1)
              
          Returns:
              reward: The reward received for the chosen arm (if counterfactual, the reward for all arms is returned)
          """
          if choice < 0 or choice >= self.n_arms:
              raise ValueError(f"Choice must be between 0 and {self.n_arms-1}")
          
          # Apply drift and reversals BEFORE reward is generated
          reversal_occurred = self._apply_dynamics()
          
          # Generate reward based on current probabilities
          if self.counterfactual:
              # Generate rewards for all arms
              reward = self._generate_reward(all_arms=True)
              chosen_reward = reward[choice]
          else:
              # Generate reward only for chosen arm
              reward = self._generate_reward(choice=choice, all_arms=False)
              chosen_reward = reward
          
          # Update history (store the actual reward received for the chosen arm)
          self.history['choices'].append(choice)
          self.history['rewards'].append(chosen_reward)
          self.history['reward_probs'].append(self.reward_prob.copy())
          self.history['reversals'].append(reversal_occurred)
          
          self.t += 1
          
          # Return the reward
          if self.counterfactual:
              # Return full reward array for all arms
              return reward
          else:
              # Return formatted reward with -1 for unchosen arm
              choice_onehot = np.eye(self.n_actions)[choice]
              return choice_onehot * reward + (1-choice_onehot)*-1

    @property
    def reward_probs(self) -> np.ndarray:
      return self.history['reward_probs']

    @property
    def n_actions(self) -> int:
      return self.n_arms   
          
    def _apply_dynamics(self) -> bool:
        """Apply drift and check for reversals."""
        reversal_occurred = False
        
        # Check for reversal (sudden swap)
        if self.hazard_rate > 0 and self.rng.random() < self.hazard_rate:
            self._apply_reversal()
            reversal_occurred = True
        
        # Apply gradual drift
        if self.drift_rate > 0:
            self._apply_drift()
        
        return reversal_occurred
    
    def _apply_reversal(self):
        """Apply a reversal: swap the reward probabilities."""
        if self.n_arms == 2:
            # Simple swap for 2 arms
            self.reward_prob = self.reward_prob[::-1]
        else:
            # For >2 arms, randomly permute
            self.reward_prob = self.rng.permutation(self.reward_prob)
    
    def _apply_drift(self):
        """Apply Gaussian random walk drift to reward probabilities."""
        if self.drift_cov is not None and self.n_arms == 2:
            # Correlated drift for 2 arms
            drift = self.rng.multivariate_normal(np.zeros(2), self.drift_cov)
        else:
            # Independent drift
            drift = self.rng.normal(0, self.drift_rate, self.n_arms)
        
        # Apply drift and clip to bounds
        self.reward_prob = np.clip(
            self.reward_prob + drift,
            self.bounds[0],
            self.bounds[1]
        )
    
    def _generate_reward(self, choice: int = None, all_arms: bool = False):
        """Generate reward based on current probabilities.
        
        Args:
            choice: Index of chosen arm (only needed if all_arms=False)
            all_arms: If True, generate rewards for all arms; if False, only for chosen arm
            
        Returns:
            float (if all_arms=False) or np.ndarray (if all_arms=True)
        """
        if all_arms:
            # Generate rewards for all arms
            successes = self.rng.random(self.n_arms) < self.reward_prob
            if self.reward_schedule == "binary":
                return successes.astype(float)
            elif self.reward_schedule == "penalty":
                return np.where(successes, 1.0, -1.0)
            else:
                raise ValueError(f"Unknown reward_schedule: {self.reward_schedule}")
        else:
            # Generate reward for chosen arm only
            success = self.rng.random() < self.reward_prob[choice]
            if self.reward_schedule == "binary":
                return 1.0 if success else 0.0
            elif self.reward_schedule == "penalty":
                return 1.0 if success else -1.0
            else:
                raise ValueError(f"Unknown reward_schedule: {self.reward_schedule}")
    
    def new_sess(self):
        """Reset to initial state for a new session."""
        self.reward_prob = self.init_reward_prob.copy()
        self.t = 0
        self.history = {
            'choices': [],
            'rewards': [],
            'reward_probs': [self.reward_prob.copy()],
            'reversals': []
        }
    
    def get_optimal_arm(self) -> int:
        """Return the index of the arm with highest current reward probability."""
        return int(np.argmax(self.reward_prob))
    
    def get_history_array(self) -> dict:
        """Return history as numpy arrays for analysis."""
        return {
            'choices': np.array(self.history['choices']),
            'rewards': np.array(self.history['rewards']),
            'reward_probs': np.array(self.history['reward_probs']),
            'reversals': np.array(self.history['reversals'])
        }            

class BanditsFlip(Bandits):
  """Env for 2-armed bandit task with reward probs that flip in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.8,
      reward_prob_low: float = 0.2,
      counterfactual: bool = False,
      **kwargs,
  ):
    
    super().__init__()
    
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    self._counterfactual = counterfactual
    # Choose a random block to start in
    self._block = np.random.binomial(1, 0.5)
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self._block = 1 - self._block
    # Set the reward probabilites
    if self._block == 1:
      self.reward_probs = [self._reward_prob_high, self._reward_prob_low]
    else:
      self.reward_probs = [self._reward_prob_low, self._reward_prob_high]

  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    # Sample a reward with this probability
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)
    
    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self._counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def n_actions(self) -> int:
    return 2
  

class BanditsSwitch(Bandits):
  """Env for 2-armed bandit task with fixed sets of reward probs that switch in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.75,
      reward_prob_low: float = 0.25,
      reward_prob_middle: float = 0.5,
      counterfactual: bool = False,
      **kwargs,
  ):
    
    super().__init__()
    
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    self._reward_prob_middle = reward_prob_middle
    self._counterfactual = counterfactual
    self._n_blocks = 7
    
    # Choose a random block to start in
    self._block = np.random.randint(self._n_blocks)
    
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Switch the reward probabilities for a new block."""
    
    # Choose a new random block
    block = np.random.randint(0, self._n_blocks)
    while block == self._block:
      block = np.random.randint(0, self._n_blocks)
    self._block = block
    
    # Set the reward probabilites
    if self._block == 0:
      self._reward_probs = [self._reward_prob_high, self._reward_prob_low]
    elif self._block == 1:
      self._reward_probs = [self._reward_prob_middle, self._reward_prob_middle]
    elif self._block == 2:
      self._reward_probs = [self._reward_prob_low, self._reward_prob_high]
    elif self._block == 3:
      self._reward_probs = [self._reward_prob_low, self._reward_prob_middle]
    elif self._block == 4:
      self._reward_probs = [self._reward_prob_middle, self._reward_prob_high]
    elif self._block == 5:
      self._reward_probs = [self._reward_prob_middle, self._reward_prob_low]
    elif self._block == 6:
      self._reward_probs = [self._reward_prob_high, self._reward_prob_middle]
      
  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    # Sample a reward with this probability
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)

    # Check whether to flip the block
    if np.random.uniform() < self._block_flip_prob:
      self.new_block()

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self._counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return 2


class BanditsDrift(Bandits):
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and 1. On each
  trial, gaussian random noise is added to each.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_actions: number of actions available
  """

  def __init__(
      self,
      sigma: float,
      n_actions: int = 2,
      counterfactual: bool = False,
      **kwargs,
      ):
    """Initialize the environment."""
    
    super().__init__()
    
    # Check inputs
    if sigma < 0:
      msg = f'Argument sigma but must be greater than 0. Found: {sigma}.'
      raise ValueError(msg)

    # Initialize persistent properties
    self._sigma = sigma
    self._n_actions = n_actions
    self._counterfactual = counterfactual

    # Sample new reward probabilities
    self.new_sess()

  def new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = np.random.rand(self._n_actions)

  def step(self, choice: int) -> np.ndarray:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """

    # Sample reward with the probability of the chosen side
    reward = np.array([float(np.random.rand() < self._reward_probs[i]) for i in range(self._n_actions)])
    
    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self._sigma, size=self._n_actions)
    self._reward_probs += drift
    
    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.clip(self._reward_probs, 0, 1)

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self._counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self._n_actions
  
  
class BanditsDrift_eckstein2024(Bandits):
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and 1. On each
  trial, gaussian random noise is added to each. Reward equals the reward probability.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_actions: number of actions available
  """

  def __init__(
      self,
      sigma: float,
      n_actions: int = 2,
      counterfactual: bool = False,
      **kwargs,
      ):
    """Initialize the environment."""
    
    super().__init__()
    
    # Check inputs
    if sigma < 0:
      msg = f'Argument sigma but must be greater than 0. Found: {sigma}.'
      raise ValueError(msg)

    # Initialize persistent properties
    self._sigma = sigma
    self._n_actions = n_actions
    self._counterfactual = counterfactual

    # Sample new reward probabilities
    self.new_sess()

  def new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = np.random.rand(self._n_actions)

  def step(self, choice: int) -> np.ndarray:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """

    # Sample reward with the probability of the chosen side
    # reward = np.array([float(np.random.rand() < self._reward_probs[i]) for i in range(self._n_actions)])
    
    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self._sigma, size=self._n_actions)
    self._reward_probs += drift
    
    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.clip(self._reward_probs, 0, 1)

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self._counterfactual:
      return self._reward_probs
    else:
      return choice_onehot * self._reward_probs[choice] + (1-choice_onehot)*-1

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self._n_actions


class BanditsFlip_eckstein2022(Bandits):
  """Env for 2-armed bandit task with reward probs that flip in blocks as used in Eckstein et al (2022).
  Additional flipping criteria: accumulated reward > threshold with threshold ~ Uniform(7,15)."""

  def __init__(
      self,
      reward_prob_high: float = 0.75,
      reward_prob_low: float = 0.,
      counterfactual: bool = False,
      **kwargs
  ):
    
    super(BanditsFlip_eckstein2022, self).__init__()
    
    # Assign the input parameters as properties
    self._block_flip_criteria = np.random.uniform(7, 15)
    self._accumulated_reward_block = 0
    self._accumulated_steps_block = 0
    
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    
    self._counterfactual = counterfactual
    
    # Choose a random block to start in
    self._block = np.random.binomial(1, 0.5)
    
    # Set up the new block
    self.new_block()

  def new_sess(self):
    # Choose a random block to start in
    self._block = np.random.binomial(1, 0.5)
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self._block = 1 - self._block
    # Set the reward probabilites
    if self._block == 1:
      self.reward_probs = [self._reward_prob_high, self._reward_prob_low]
    else:
      self.reward_probs = [self._reward_prob_low, self._reward_prob_high]
      
    self._block_flip_criteria = np.random.uniform(7, 15)
    self._accumulated_reward_block = 0
    self._accumulated_steps_block = 0

  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    if self._accumulated_steps_block == 0 and choice == np.argmax(self.reward_probs):
      # first trial in a new block -> if the choice is correct -> always reward 
      reward = np.zeros(self.n_actions)
      reward[choice] = 1
    else:
      # Sample a reward with this probability
      reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)
    
    # Check whether to flip the block
    self._accumulated_reward_block += reward[choice]
    self._accumulated_steps_block += 1
    if self._accumulated_reward_block >= self._block_flip_criteria:
      self.new_block()
    
    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self._counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def n_actions(self) -> int:
    return 2
  

class Bandits_Standard(Bandits):
  """Env for 2-armed bandit task with reward probs that flip in blocks."""

  def __init__(
      self,
      reward_prob_0: float = 0.8,
      reward_prob_1: float = 0.2,
      counterfactual: bool = False,
      **kwargs,
  ):
    
    super().__init__()
    
    # Assign the input parameters as properties
    self.reward_probs = [reward_prob_0, reward_prob_1]
    self._counterfactual = counterfactual
    
  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    # Sample a reward with this probability
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self._counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def n_actions(self) -> int:
    return 2


class BanditSession(NamedTuple):
  """Holds data for a single session of a bandit task."""
  choices: np.ndarray
  rewards: np.ndarray
  session: np.ndarray
  # reward_probabilities: np.ndarray
  # q: np.ndarray
  n_trials: int
  
  def set_session(self, session: int):
    return self(
      choices=self.choices, 
      rewards=self.rewards, 
      session=np.full_like(self.session, session), 
      # reward_probabilities=self.reward_probabilities, 
      # q=self.q, 
      n_trials=self.n_trials,
      )
  
  def __getitem__(self, val):
    return self._replace(
      choices=self.choices.__getitem__(val), 
      rewards=self.rewards.__getitem__(val), 
      session=self.session.__getitem__(val), 
      # reward_probabilities=self.reward_probabilities.__getitem__(val), 
      # q=self.q.__getitem__(val), 
      n_trials=self.choices.__getitem__(val).shape[0],
      )


###############
#  EXECUTION  #
###############


def run_experiment(
  agent: Agent,
  environment: Bandits,
  n_trials: int,
  session_id: int = 0,
  ) -> BanditSession:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_trials: The number of steps in the session you'd like to generate

  Returns:
    experiment: A BanditSession holding choices and rewards from the session
  """
  
  choices = np.zeros((n_trials+1)) - 1
  rewards = np.zeros((n_trials+1, agent.n_actions)) - 1
  # qs = np.zeros((n_trials+1, agent._n_actions, agent.state['value_reward'].shape[-1])) - 1
  # reward_probs = np.zeros((n_trials+1, agent._n_actions)) - 1

  for trial in range(n_trials+1):
    # Log environment reward probabilities and Q-Values
    # reward_probs[trial] = environment.reward_probs
    # qs[trial] = agent.q
    # First - agent makes a choice
    choice = agent.get_choice()
    # Second - environment computes a reward
    result = environment.step(choice)
    if isinstance(result, tuple):
      reward, info = result
    else:
      reward = result
      info = {}
    # Log choice and reward
    choices[trial] = choice
    rewards[trial, :len(reward)] = reward
    
    # Third - agent updates its believes based on chosen action and received reward
    agent.update(choice=choice, reward=rewards[trial], participant_id=session_id)
    
  experiment = BanditSession(n_trials=n_trials,
                             choices=choices[:-1].astype(int),
                             rewards=rewards[:-1],
                             session=np.full(rewards[:-1].shape[0], session_id).astype(int),
                            #  reward_probabilities=reward_probs[:-1],
                            #  q=qs[:-1],
                             )
  return experiment, choices.astype(int), rewards


def create_dataset(
  agent: Agent,
  environment: Bandits,
  n_trials: int,
  n_sessions: int,
  sequence_length: int = None,
  stride: int = 1,
  sample_parameters: bool = False,
  device=torch.device('cpu'),
  verbose=False,
  ) -> tuple[SpiceDataset, list[BanditSession], list[dict[str, float]]]:
  """Generates a behavioral dataset from a given agent and environment.

  Args:
    agent: An agent object to generate choices
    environment: An environment object to generate rewards
    n_trials_per_session: The number of trials in each behavioral session to
      be generated
    n_sessions: The number of sessions to generate
    batch_size: The size of the batches to serve from the dataset. If None, 
      batch_size defaults to n_sessions

  Returns:
    A torch.utils.data.Dataset object suitable for training the RNN object.
    An experliment_list with the results of (simulated) experiments
  """
  
  agent_original = agent
  n_actions = agent[0]._n_actions if isinstance(agent_original, list) else agent.n_actions
  xs = np.zeros((n_sessions, n_trials, n_actions*2 + 1))
  ys = np.zeros((n_sessions, n_trials, n_actions))
  experiment_list = []
  parameter_list = []
  
  print('Creating dataset...')
  for session in tqdm(range(n_sessions)):
    if verbose:
      print(f'Running session {session+1}/{n_sessions}...')
    environment.new_sess()
    if isinstance(agent_original, list):
      agent = agent_original[session]
    agent.new_sess(sample_parameters=sample_parameters, participant_id=session)
    experiment, choices, rewards = run_experiment(agent, environment, n_trials, session)
    experiment_list.append(experiment)
    
    # one-hot encoding of choices
    choices = np.eye(agent.n_actions)[choices]
    ys[session] = choices[1:]
    xs[session] = np.concatenate((choices[:-1], rewards[:-1], experiment.session[:, None]), axis=-1)
    
    if isinstance(agent, AgentQ):
      # add current parameters to list
      parameter_list.append(
        {
          'beta_reward': copy(agent.betas['value_reward']),
          'alpha_reward': copy(agent._alpha_reward),
          'alpha_penalty': copy(agent._alpha_penalty),
          'beta_choice': copy(agent.betas['value_choice']),
          'alpha_choice': copy(agent._alpha_choice),
          'forget_rate': copy(agent._forget_rate),
        }
      )

  dataset = SpiceDataset(
    xs=xs, 
    ys=ys,
    sequence_length=sequence_length,
    stride=stride,
    device=device)
  
  return dataset, experiment_list, parameter_list


def process_dataset(dataset, agent, parameter_list, n_trials_per_session):
    # general dataset columns
    session, choice, reward = [], [], []
    choice_prob_0, choice_prob_1 = [], []
    action_value_0, action_value_1 = [], []
    reward_value_0, reward_value_1 = [], []
    choice_value_0, choice_value_1 = [], []

    # parameters
    beta_reward, alpha_reward, alpha_penalty = [], [], []
    beta_choice, alpha_choice = [], []
    forget_rate = []

    # parameter means
    mean_beta_reward, mean_alpha_reward, mean_alpha_penalty = [], [], []
    mean_beta_choice, mean_alpha_choice = [], []
    mean_forget_rate = []

    for i in range(len(dataset)):
        experiment = dataset.xs[i].cpu().numpy()
        qs, choice_probs, _ = get_update_dynamics(experiment, agent)

        # behavioral data
        session.extend(experiment[:, -1].tolist())
        choice.extend(np.argmax(experiment[:, :agent.n_actions], axis=-1).tolist())
        reward.extend(np.max(experiment[:, agent.n_actions : agent.n_actions * 2], axis=-1).tolist())

        # update dynamics
        choice_prob_0.extend(choice_probs[:, 0].tolist())
        choice_prob_1.extend(choice_probs[:, 1].tolist())
        action_value_0.extend(qs[0][:, 0].tolist())
        action_value_1.extend(qs[0][:, 1].tolist())
        reward_value_0.extend(qs[1]["value_reward"][:, 0].tolist())
        reward_value_1.extend(qs[1]["value_reward"][:, 1].tolist())
        choice_value_0.extend(qs[1]["value_choice"][:, 0].tolist())
        choice_value_1.extend(qs[1]["value_choice"][:, 1].tolist())

        params = parameter_list[i]

        # replicate parameters for each trial
        beta_reward.extend([params["beta_reward"]] * n_trials_per_session)
        alpha_reward.extend([params["alpha_reward"]] * n_trials_per_session)
        alpha_penalty.extend([params["alpha_penalty"]] * n_trials_per_session)
        forget_rate.extend([params["forget_rate"]] * n_trials_per_session)
        beta_choice.extend([params["beta_choice"]] * n_trials_per_session)
        alpha_choice.extend([params["alpha_choice"]] * n_trials_per_session)

        # replicate mean parameters for each trial
        mean_beta_reward.extend([agent._mean_beta_reward] * n_trials_per_session)
        mean_alpha_reward.extend([agent._mean_alpha_reward] * n_trials_per_session)
        mean_alpha_penalty.extend([agent._mean_alpha_penalty] * n_trials_per_session)
        mean_forget_rate.extend([agent._mean_forget_rate] * n_trials_per_session)
        mean_beta_choice.extend([agent._mean_beta_choice] * n_trials_per_session)
        mean_alpha_choice.extend([agent._mean_alpha_choice] * n_trials_per_session)

    fields = {
        "session": session,
        "choice": choice,
        "reward": reward,
        "choice_prob_0": choice_prob_0,
        "choice_prob_1": choice_prob_1,
        "action_value_0": action_value_0,
        "action_value_1": action_value_1,
        "reward_value_0": reward_value_0,
        "reward_value_1": reward_value_1,
        "choice_value_0": choice_value_0,
        "choice_value_1": choice_value_1,
        "beta_reward": beta_reward,
        "alpha_reward": alpha_reward,
        "alpha_penalty": alpha_penalty,
        "forget_rate": forget_rate,
        "beta_choice": beta_choice,
        "alpha_choice": alpha_choice,
        "mean_beta_reward": mean_beta_reward,
        "mean_alpha_reward": mean_alpha_reward,
        "mean_alpha_penalty": mean_alpha_penalty,
        "mean_forget_rate": mean_forget_rate,
        "mean_beta_choice": mean_beta_choice,
        "mean_alpha_choice": mean_alpha_choice,
    }

    columns = list(fields.keys())
    data = np.column_stack([np.array(fields[name]) for name in columns])  # .swapaxes(1, 0)

    return pd.DataFrame(data=data, columns=columns)


def get_update_dynamics(experiment: Union[np.ndarray, torch.Tensor], agent: Agent, additional_signals: List[str] = ['value_reward', 'value_choice']):
  """Compute Q-Values of a specific agent for a specific experiment sequence with given actions and rewards.

  Args:
      experiment (BanditSession): _description_
      agent (_type_): _description_

  Returns:
      _type_: _description_
  """

  if isinstance(experiment, np.ndarray) or isinstance(experiment, torch.Tensor):
    if isinstance(experiment, torch.Tensor):
      experiment = experiment.detach().cpu().numpy()
    if len(experiment.shape) == 3:
      experiment = experiment.squeeze(0)
    # get number of actual trials
    n_trials = len(experiment) - np.where(~np.isnan(experiment[::-1][:, 0]))[0][0]
    choices = experiment[:n_trials, :agent.n_actions]
    rewards = experiment[:n_trials, agent.n_actions:2*agent.n_actions]
    # TODO: additional_inputs are currently treated as signals and as meta-information for the embedding
    additional_inputs = experiment[0, 2*agent.n_actions:-3]
    current_block = int(experiment[0, -3])
    experiment_id = int(experiment[0, -2])
    participant_id = int(experiment[0, -1])
  else:
    raise TypeError("experiment is of not of class numpy.ndarray or torch.Tensor")
  
  # reset agent states according to ID
  agent.new_sess(participant_id=participant_id, experiment_id=experiment_id, additional_embedding_inputs=additional_inputs)
  betas_available = hasattr(agent, 'betas') and agent.betas is not None
  
  # initialize storages
  q = np.zeros((n_trials, agent.n_actions))
  values_signal = {signal: np.zeros((n_trials, agent.n_actions)) for signal in additional_signals}
  choice_probs = np.zeros((n_trials, agent.n_actions))
  
  for trial in range(n_trials):
    # track all states
    q[trial] = agent.q
    for signal in additional_signals:
      value = agent.get_state_value(signal, multiply_with_beta=betas_available) if signal in agent.state else np.zeros_like(agent.q)
      if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
      values_signal[signal][trial] = value
    
    choice_probs[trial] = np.clip(agent.get_choice_probs(), 1e-8, 1)
    
    agent.update(
      choice=np.argmax(choices[trial], axis=-1), 
      reward=rewards[trial],  
      block=current_block, 
      additional_inputs=additional_inputs,
      )
  
  return (q, values_signal), choice_probs, agent


###############
# DIAGNOSTICS #
###############


def plot_session(
  choices: np.ndarray,
  rewards: np.ndarray,
  timeseries: Tuple[np.ndarray],
  timeseries_name: str,
  labels: Optional[Tuple[str]] = None,
  title: str = '',
  x_label = 'Trials',
  fig_ax = None,
  compare=False,
  color=None,
  x_axis_info=True,
  y_axis_info=True,
  ):
  """Plot data from a single behavioral session of the bandit task.

  Args:
    choices: The choices made by the agent
    rewards: The rewards received by the agent
    timeseries: The dynamical value of interest on each arm
    timeseries_name: The name of the timeseries
    labels: The labels for the lines in the plot
    fig_ax: A tuple of a matplotlib figure and axis to plot on
    compare: If True, plot multiple timeseries on the same plot
    color: A list of colors to use for the plot; at least as long as the number of timeseries
  """

  if color == None:
    color = [None]*len(timeseries)
  
  # Make the plot
  if fig_ax is None:
    fig, ax = plt.subplots(1, figsize=(10, 3))
  else:
    fig, ax = fig_ax
    
  if compare:
    if timeseries.ndim==2:
      timeseries = np.expand_dims(timeseries, -1)
    if timeseries.ndim!=3 or timeseries.shape[-1]!=1:
      raise ValueError('If compare: timeseries must be of shape (agent, timesteps, 1).')
  else:
    if timeseries.ndim!=2:
      raise ValueError('timeseries must be of shape (timesteps, n_actions).')
                       
  if not compare:
    # choices = np.expand_dims(choices, 0)
    timeseries = np.expand_dims(timeseries, 0)
  
  for i in range(timeseries.shape[0]):
    if labels is not None:
      if timeseries[i].ndim == 1:
        timeseries[i] = timeseries[i, :, None]
      if not compare:
        if len(labels) != timeseries[i].shape[1]:
          raise ValueError('labels length must match timeseries.shape[1].')
      else:
        if timeseries[i].shape[1] != 1:
          raise ValueError('If compare: timeseries.shape[1] must be 1.')
        if len(labels) != timeseries.shape[0]:
          raise ValueError('If compare: labels length must match timeseries.shape[0].')
      for ii in range(timeseries[i].shape[-1]):
          label = labels[ii] if not compare else labels[i]
          ax.plot(timeseries[i, :, ii], label=label, color=color[i])
      ax.legend(bbox_to_anchor=(1, 1))
    else:  # Skip legend.
      ax.plot(timeseries[i], color=color[i])
  
  # Plot ticks relating to whether the option was chosen (factual) or not (counterfactual) and whether it was rewarded
  min_y, max_y = np.min(timeseries), np.max(timeseries)
  # diff_min_max = np.max((5e-2, max_y - min_y))  # make sure the difference is not <= 0
  diff_min_max = np.max((1e-1, max_y - min_y))
  
  x = np.arange(len(choices))
  chosen_y = min_y - 1e-1  # Lower position for chosen (bigger tick)
  not_chosen_y = max_y + 1e-1  # Slightly lower for not chosen (smaller tick)
  # not_chosen_y = chosen_y - 1e-1 * diff_min_max  # Slightly lower for not chosen (smaller tick)

  # if (rewards > 0 and rewards < 1).any():
  # Plot ticks for chosen options
  ax.scatter(x[(choices == 1) & (rewards == 1)], np.full(sum((choices == 1) & (rewards == 1)), chosen_y), color='green', s=120, marker='|')  # Large green tick for chosen reward
  ax.scatter(x[(choices == 1) & (rewards == 0)], np.full(sum((choices == 1) & (rewards == 0)), chosen_y), color='red', s=80, marker='|')  # Large red tick for chosen penalty

  # Plot ticks for not chosen options
  ax.scatter(x[(choices == 0) & (rewards == 1)], np.full(sum((choices == 0) & (rewards == 1)), not_chosen_y), color='green', s=120, marker='|')  # Small green tick
  ax.scatter(x[(choices == 0) & (rewards == 0)], np.full(sum((choices == 0) & (rewards == 0)), not_chosen_y), color='red', s=80, marker='|')  # Small red tick
  # else:
  #   # Plot ticks for chosen options
  #   ax.scatter(x[(choices == 1)], np.full(sum((choices == 1)), chosen_y), color='green', s=100, marker='|')  # Large green tick for chosen reward
  #   ax.scatter(x[(choices == 1)], np.full(sum((choices == 1)), chosen_y), color='red', s=80, marker='|')  # Large red tick for chosen penalty

  #   # Plot ticks for not chosen options
  #   ax.scatter(x[(choices == 0) & (rewards == 1)], np.full(sum((choices == 0)), not_chosen_y), color='green', s=100, marker='|')  # Small green tick
  #   ax.scatter(x[(choices == 0) & (rewards == 0)], np.full(sum((choices == 0)), not_chosen_y), color='red', s=80, marker='|')  # Small red tick
  # ax.set_ylim(not_chosen_y, np.max((-not_chosen_y, np.max(timeseries + 1e-1 * diff_min_max))))
  
  if x_axis_info:
    ax.set_xlabel(x_label)
  else:
    # ax.set_xticks(np.linspace(1, len(timeseries), 5))
    ax.set_xticklabels(['']*5)
    
  if y_axis_info:
    ax.set_ylabel(timeseries_name)
  else:
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(['']*5)
    
  ax.set_title(title)

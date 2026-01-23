from typing import NamedTuple, Union, Optional, Dict, Tuple, List, Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import torch
from tqdm import tqdm

from ...spice.resources.spice_utils import SpiceDataset
from ...spice.utils.agent import Agent
from weinhardt2025.benchmarking.benchmarking_qlearning import QLearning


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
          reversal_occurred = self.apply_dynamics()
          
          # Generate reward based on current probabilities
          if self.counterfactual:
              # Generate rewards for all arms
              reward = self.generate_reward(all_arms=True)
              chosen_reward = reward[choice]
          else:
              # Generate reward only for chosen arm
              reward = self.generate_reward(choice=choice, all_arms=False)
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
          
    def apply_dynamics(self) -> bool:
        """Apply drift and check for reversals."""
        reversal_occurred = False
        
        # Check for reversal (sudden swap)
        if self.hazard_rate > 0 and self.rng.random() < self.hazard_rate:
            self.apply_reversal()
            reversal_occurred = True
        
        # Apply gradual drift
        if self.drift_rate > 0:
            self.apply_drift()
        
        return reversal_occurred
    
    def apply_reversal(self):
        """Apply a reversal: swap the reward probabilities."""
        if self.n_arms == 2:
            # Simple swap for 2 arms
            self.reward_prob = self.reward_prob[::-1]
        else:
            # For >2 arms, randomly permute
            self.reward_prob = self.rng.permutation(self.reward_prob)
    
    def apply_drift(self):
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
    
    def generate_reward(self, choice: int = None, all_arms: bool = False):
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
    self.block_flip_prob = block_flip_prob
    self.reward_prob_high = reward_prob_high
    self.reward_prob_low = reward_prob_low
    self.counterfactual = counterfactual
    # Choose a random block to start in
    self.block = np.random.binomial(1, 0.5)
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self.block = 1 - self.block
    # Set the reward probabilites
    if self.block == 1:
      self.reward_probs = [self.reward_prob_high, self.reward_prob_low]
    else:
      self.reward_probs = [self.reward_prob_low, self.reward_prob_high]

  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    # Sample a reward with this probability
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)
    
    # Check whether to flip the block
    if np.random.binomial(1, self.block_flip_prob):
      self.new_block()

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self.counterfactual:
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
    self.block_flip_prob = block_flip_prob
    self.reward_prob_high = reward_prob_high
    self.reward_prob_low = reward_prob_low
    self.reward_prob_middle = reward_prob_middle
    self.counterfactual = counterfactual
    self.n_blocks = 7
    
    # Choose a random block to start in
    self.block = np.random.randint(self.n_blocks)
    
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Switch the reward probabilities for a new block."""
    
    # Choose a new random block
    block = np.random.randint(0, self.n_blocks)
    while block == self.block:
      block = np.random.randint(0, self.n_blocks)
    self.block = block
    
    # Set the reward probabilites
    if self.block == 0:
      self.reward_probs = [self.reward_prob_high, self.reward_prob_low]
    elif self.block == 1:
      self.reward_probs = [self.reward_prob_middle, self.reward_prob_middle]
    elif self.block == 2:
      self.reward_probs = [self.reward_prob_low, self.reward_prob_high]
    elif self.block == 3:
      self.reward_probs = [self.reward_prob_low, self.reward_prob_middle]
    elif self.block == 4:
      self.reward_probs = [self.reward_prob_middle, self.reward_prob_high]
    elif self.block == 5:
      self.reward_probs = [self.reward_prob_middle, self.reward_prob_low]
    elif self.block == 6:
      self.reward_probs = [self.reward_prob_high, self.reward_prob_middle]
      
  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    # Sample a reward with this probability
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)

    # Check whether to flip the block
    if np.random.uniform() < self.block_flip_prob:
      self.new_block()

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self.counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def reward_probs(self) -> np.ndarray:
    return self.reward_probs.copy()

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
    self.sigma = sigma
    self.n_actions = n_actions
    self.counterfactual = counterfactual

    # Sample new reward probabilities
    self.new_sess()

  def new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self.reward_probs = np.random.rand(self.n_actions)

  def step(self, choice: int) -> np.ndarray:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """

    # Sample reward with the probability of the chosen side
    reward = np.array([float(np.random.rand() < self.reward_probs[i]) for i in range(self.n_actions)])
    
    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self.sigma, size=self.n_actions)
    self.reward_probs += drift
    
    # Fix reward probs that've drifted below 0 or above 1
    self.reward_probs = np.clip(self.reward_probs, 0, 1)

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self.counterfactual:
      return reward
    else:
      return choice_onehot * reward[choice] + (1-choice_onehot)*-1

  @property
  def reward_probs(self) -> np.ndarray:
    return self.reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self.n_actions
  
  
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
    self.sigma = sigma
    self.n_actions = n_actions
    self.counterfactual = counterfactual

    # Sample new reward probabilities
    self.new_sess()

  def new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self.reward_probs = np.random.rand(self.n_actions)

  def step(self, choice: int) -> np.ndarray:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """

    # Sample reward with the probability of the chosen side
    # reward = np.array([float(np.random.rand() < self.reward_probs[i]) for i in range(self.n_actions)])
    
    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self.sigma, size=self.n_actions)
    self.reward_probs += drift
    
    # Fix reward probs that've drifted below 0 or above 1
    self.reward_probs = np.clip(self.reward_probs, 0, 1)

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self.counterfactual:
      return self.reward_probs
    else:
      return choice_onehot * self.reward_probs[choice] + (1-choice_onehot)*-1

  @property
  def reward_probs(self) -> np.ndarray:
    return self.reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self.n_actions


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
    self.block_flip_criteria = np.random.uniform(7, 15)
    self.accumulated_reward_block = 0
    self.accumulated_steps_block = 0
    
    self.reward_prob_high = reward_prob_high
    self.reward_prob_low = reward_prob_low
    
    self.counterfactual = counterfactual
    
    # Choose a random block to start in
    self.block = np.random.binomial(1, 0.5)
    
    # Set up the new block
    self.new_block()

  def new_sess(self):
    # Choose a random block to start in
    self.block = np.random.binomial(1, 0.5)
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self.block = 1 - self.block
    # Set the reward probabilites
    if self.block == 1:
      self.reward_probs = [self.reward_prob_high, self.reward_prob_low]
    else:
      self.reward_probs = [self.reward_prob_low, self.reward_prob_high]
      
    self.block_flip_criteria = np.random.uniform(7, 15)
    self.accumulated_reward_block = 0
    self.accumulated_steps_block = 0

  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    if self.accumulated_steps_block == 0 and choice == np.argmax(self.reward_probs):
      # first trial in a new block -> if the choice is correct -> always reward 
      reward = np.zeros(self.n_actions)
      reward[choice] = 1
    else:
      # Sample a reward with this probability
      reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)
    
    # Check whether to flip the block
    self.accumulated_reward_block += reward[choice]
    self.accumulated_steps_block += 1
    if self.accumulated_reward_block >= self.block_flip_criteria:
      self.new_block()
    
    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self.counterfactual:
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
    self.counterfactual = counterfactual
    
  def step(self, choice: int = None):
    """Step the model forward given chosen action."""

    # Sample a reward with this probability
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)

    # Return the reward
    choice_onehot = np.eye(self.n_actions)[choice]
    if self.counterfactual:
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
    return self.replace(
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
  # qs = np.zeros((n_trials+1, agent.n_actions, agent.state['value_reward'].shape[-1])) - 1
  # reward_probs = np.zeros((n_trials+1, agent.n_actions)) - 1

  for trial in range(n_trials+1):
    # Log environment reward probabilities and Q-Values
    # reward_probs[trial] = environment.reward_probs
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
  n_actions = agent[0].n_actions if isinstance(agent_original, list) else agent.n_actions
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
    
    if isinstance(agent.model, QLearning):
      # add current parameters to list
      parameter_list.append(
        {
          'beta_reward': copy(agent.betas['value_reward']),
          'alpha_reward': copy(agent.alpha_reward),
          'alpha_penalty': copy(agent.alpha_penalty),
          'beta_choice': copy(agent.betas['value_choice']),
          'alpha_choice': copy(agent.alpha_choice),
          'forget_rate': copy(agent.forget_rate),
        }
      )

  dataset = SpiceDataset(
    xs=xs, 
    ys=ys,
    sequence_length=sequence_length,
    stride=stride,
    device=device)
  
  return dataset, experiment_list, parameter_list
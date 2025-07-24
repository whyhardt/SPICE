import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from typing import Iterable, List, Dict, Tuple, Callable, Union
import itertools
import random
from tqdm import tqdm
import dill
import torch

from .bandits import AgentNetwork, AgentSpice, get_update_dynamics, BanditSession, BanditsDrift
from .rnn_utils import DatasetRNN
from .model_evaluation import log_likelihood


def make_sindy_data(
    dataset,
    agent,
    sessions=-1,
    ):

  # Get training data for SINDy
  # put all relevant signals in x_train

  if not isinstance(sessions, Iterable) and sessions == -1:
    # use all sessions
    sessions = np.arange(len(dataset))
  else:
    # use only the specified sessions
    sessions = np.array(sessions)
    
  n_control = 2
  
  choices = np.stack([dataset[i].choices for i in sessions], axis=0)
  rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
  qs = np.stack([dataset[i].q for i in sessions], axis=0)
  
  choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
  for sess in sessions:
    # one-hot encode choices
    choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
    
  # concatenate all qs values of one sessions along the trial dimension
  qs_all = np.concatenate([np.stack([np.expand_dims(qs_sess[:, i], axis=-1) for i in range(agent._n_actions)], axis=0) for qs_sess in qs], axis=0)
  c_all = np.concatenate([np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in choices_oh], axis=0)
  r_all = np.concatenate([np.stack([r_sess for _ in range(agent._n_actions)], axis=0) for r_sess in rewards], axis=0)
  
  # get observed dynamics
  x_train = qs_all
  feature_names = ['q']

  # get control
  control_names = []
  control = np.zeros((*x_train.shape[:-1], n_control))
  control[:, :, 0] = c_all
  control_names += ['c']
  control[:, :, n_control-1] = r_all
  control_names += ['r']
  
  feature_names += control_names
  
  print(f'Shape of Q-Values is: {x_train.shape}')
  print(f'Shape of control parameters is: {control.shape}')
  print(f'Feature names are: {feature_names}')
  
  # make x_train and control sequences instead of arrays
  x_train = [x_train_sess for x_train_sess in x_train]
  control = [control_sess for control_sess in control]
 
  return x_train, control, feature_names


def create_dataset(
  agent: AgentNetwork,
  data: DatasetRNN,
  rnn_modules: List,
  control_signals: List,
  dataprocessing: Dict[str, List] = None,
  shuffle: bool = False,
  groupby: str = 'c_action',
  ):
  
  n_trials = data.xs.shape[1]
  keys_x = rnn_modules
  
  keys_c = control_signals
  
  x_train = {key: [] for key in keys_x}
  control = {key: [] for key in keys_c}
  
  # determine whether trimming is specified in any of the variables of the dataprocessing-setup
  if dataprocessing is not None and any([int(dataprocessing[key][0]) for key in dataprocessing]):
    trimming = int(0.25*n_trials)
  else:
    trimming = 0
  
  for session in data.xs:
    # perform agent updates to record values over trials
    agent = get_update_dynamics(session, agent)[-1]
    
    # sort the data of one session into the corresponding signals
    for key in keys_x+keys_c:
      if len(agent._model.get_recording(key)) > 0:
        for index_action in range(agent._n_actions):
          # get all recorded values for the current session of one specific key 
          recording = agent._model.get_recording(key)
          # create tensor from list of tensors 
          values = np.concatenate(recording)[trimming:, index_action]
          # remove insignificant updates with a high-pass filter: check if dv/dt > threshold; otherwise set v(t=1) = v(t=0)
          # dvdt = np.abs(np.diff(values))
          # for index_time in range(1, dvdt.shape[0]):
          #   values[index_time] = np.where(dvdt[index_time-1] > highpass_threshold_dt, values[index_time], values[index_time-1])
          
          if key in keys_x:
            x_train[key].append(values)
          elif key in keys_c:
            control[key].append(values)
          
  feature_names = None
  
  # group recorded values into sequences by grouping them such that the groupby values do not change for one sequence
  # -> e.g. only c_action = 1.0 in one sequence; only c_action = 0.0 in next sequence (useful to split into chosen and not chosen)
  # Use itertools.groupby to find the indices of groups in the target sequence
  # groups = []
  grouped_x_train = []
  grouped_control = []
  for index_session, groupby_session in enumerate(control[groupby]):
    for key_group, group in itertools.groupby(enumerate(groupby_session), key=lambda x: x[1]):
      indices_group = [g[0] for g in list(group)]
      grouped_x_train_array = np.zeros((len(indices_group), len(keys_x)))
      grouped_control_array = np.zeros((len(indices_group), len(keys_c)))
      for index_key, key in enumerate(keys_x):
        grouped_x_train_array[:, index_key] = x_train[key][index_session][indices_group]
      for index_key, key in enumerate(keys_c):
        grouped_control_array[:, index_key] = control[key][index_session][indices_group]
      grouped_x_train.append(grouped_x_train_array)
      grouped_control.append(grouped_control_array)
    
  # data processing  
  if dataprocessing is not None:
    for index_key, key in enumerate(keys_x):
      # Offset-clearing
      if key in dataprocessing.keys() and int(dataprocessing[key][1]):
        x_min = np.min(np.concatenate(grouped_x_train)[:, index_key])
        for index_group in range(len(grouped_x_train)):
          grouped_x_train[index_group][:, index_key] -= x_min
      # Normalization
      if key in dataprocessing.keys() and int(dataprocessing[key][2]):
        x_min = np.min(np.concatenate(grouped_x_train)[:, index_key])
        x_max = np.max(np.concatenate(grouped_x_train)[:, index_key])
        for index_group in range(len(grouped_x_train)):
          grouped_x_train[index_group][:, index_key] = (grouped_x_train[index_group][:, index_key] - x_min) / (x_max - x_min)
          
    for index_key, key in enumerate(keys_c):
      # Offset-clearing
      if key in dataprocessing.keys() and int(dataprocessing[key][1]):
        x_min = np.min(np.concatenate(grouped_control)[:, index_key])
        for index_group in range(len(grouped_control)):
          grouped_control[index_group][:, index_key] -= x_min
      # Normalization
      if key in dataprocessing.keys() and int(dataprocessing[key][2]):
        x_min = np.min(np.concatenate(grouped_control)[:, index_key])
        x_max = np.max(np.concatenate(grouped_control)[:, index_key])
        for index_group in range(len(grouped_control)):
          grouped_control[index_group][:, index_key] = (grouped_control[index_group][:, index_key] - x_min) / (x_max - x_min)
  
    # compute scaling factor for beta
  #   beta_scaling = np.abs(x_maxs - x_mins).reshape(-1)
  # else:
  beta_scaling = None#np.ones((x_train_array.shape[-1]))
  
  if shuffle:
    # shuffle_idx = np.random.permutation(len(grouped_x_train))
    # grouped_x_train = grouped_x_train[shuffle_idx]
    # grouped_control = grouped_control[shuffle_idx]
    zipped = list(zip(grouped_x_train, grouped_control))
    random.shuffle(zipped)
    grouped_x_train, grouped_control = zip(*zipped)
    grouped_x_train, grouped_control = list(grouped_x_train), list(grouped_control)
  
  return grouped_x_train, grouped_control, feature_names, beta_scaling


def check_library_setup(library_setup: Dict[str, List[str]], feature_names: List[str], verbose=False) -> bool:
  msg = '\n'
  for key in library_setup.keys():
    if key not in feature_names:
      msg += f'Key {key} not in feature_names.\n'
    else:
      for feature in library_setup[key]:
        if feature not in feature_names:
          msg += f'Key {key}: Feature {feature} not in feature_names.\n'
  if msg != '\n':
    msg += f'Valid feature names are {feature_names}.\n'
    print(msg)
    return False
  else:
    if verbose:
      print('Library setup is valid. All keys and features appear in the provided list of features.')
    return True
  

def remove_control_features(control_variables: List[np.ndarray], feature_names: List[str], target_feature_names: List[str]) -> List[np.ndarray]:
  if len(target_feature_names) > 0:
    index_target_features = []
    for index_f, f_name in enumerate(feature_names):
      if f_name in target_feature_names:
        index_target_features.append(index_f)
    index_target_features = np.array(index_target_features)
    
    for index_group in range(len(control_variables)):
      control_variables[index_group] = control_variables[index_group][:, index_target_features]
  else:
    # for index_group in range(len(control_variables)):
    #   control_variables[index_group] = np.zeros_like(control_variables[index_group][:, :1])
    return None
  return control_variables

def conditional_filtering(x_train: List[np.ndarray], control: List[np.ndarray], feature_names: List[str], feature_filter: str, condition: float, remove_feature_filter=True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  x_filtered = []
  control_filtered = []
  index_filter = feature_names.index(feature_filter)-1
  for index_group in range(len(x_train)):
    if control[index_group][0, index_filter] == condition:
      x_filtered.append(x_train[index_group])
      control_filtered.append(control[index_group])

  if remove_feature_filter:
    feature_names.pop(index_filter+1)
    for index_group in range(len(x_filtered)):
      control_filtered[index_group] = control_filtered[index_group][:, np.arange(len(feature_names))!=index_filter]
    
    
  return x_filtered, control_filtered, feature_names


def sindy_loss_x(agent: Union[AgentSpice, AgentNetwork], data: List[BanditSession], loss_fn: Callable = log_loss):
  """Compute the loss of the SINDy model directly on the data in x-coordinates i.e. predicting behavior.
  This loss is not used for SINDy-Training, but for analysis purposes only.

  Args:
      model (ps.SINDy): _description_
      x_data (DatasetRNN): _description_
      loss_fn (Callable, optional): _description_. Defaults to log_loss.
  """
  
  loss_total = 0
  for experiment in data:
    agent.new_sess()
    choices = experiment.choices
    rewards = experiment.rewards
    loss_session = 0
    for t in range(len(choices)-1):
      beta = agent._beta if hasattr(agent, "_beta") else 1
      y_pred = np.exp(agent.q * beta)/np.sum(np.exp(agent.q * beta))
      agent.update(choices[t], rewards[t])
      loss_session += loss_fn(np.eye(agent._n_actions)[choices[t+1]], y_pred)
    loss_total += loss_session/(t+1)
  return loss_total/len(data)


def bandit_loss(agent: Union[AgentSpice, AgentNetwork], data: List[BanditSession], coordinates: str = "x"):
  """Compute the loss of the SINDy model directly on the data in z-coordinates i.e. predicting q-values.
  This loss is also used for SINDy-Training.

  Args:
      model (ps.SINDy): _description_
      x_data (DatasetRNN): _description_
      loss_fn (Callable): _description_. Defaults to log_loss.
      coordinates (str): Defines the coordinate system in which to compute the loss. Can be either "x" (predicting behavior) or "z" (comparing choice probabilities). Defaults to "x".
      """
  
  loss_total = 0
  for experiment in data:
    agent.new_sess()
    choices = experiment.choices
    rewards = experiment.rewards
    qs = np.exp(experiment.q)/np.sum(np.exp(experiment.q))
    loss_session = 0
    for t in range(len(choices)-1):
      beta = agent._beta if hasattr(agent, "_beta") else 1
      y_pred = np.exp(agent.q * beta)/np.sum(np.exp(agent.q * beta))
      agent.update(choices[t], rewards[t])
      if coordinates == 'x':
        y_target = np.eye(agent._n_actions)[choices[t+1]]
        loss_session = log_loss(y_target, y_pred)
      elif coordinates == 'z':
        y_target = qs[t]
        loss_session = mean_squared_error(y_target, y_pred)
    loss_total += loss_session/(t+1)
  return loss_total/len(data)


def remove_bad_participants(agent_spice: AgentSpice, agent_rnn: AgentNetwork, dataset: DatasetRNN, participant_ids: Iterable[int], trial_likelihood_difference_threshold: float=0.05, verbose: bool = False):
  """Check for badly fitted participants in the SPICE models w.r.t. the SPICE-RNN and return only the IDs of the well-fitted participants.

  Args:
      agent_spice (AgentSpice): _description_
      agent_rnn (AgentNetwork): _description_
      dataset_test (DatasetRNN): _description_
      participant_ids (Iterable[int]): _description_
      verbose (bool, optional): _description_. Defaults to False.

  Returns:
      AgentSpice: SPICE agent without badly fitted participants
      Iterable[int]: List of well-fitted participants
  """
  # if verbose:
  print("\nFiltering badly fitted SPICE models...")
  
  filtered_participant_ids = []
  
  # Create a copy of the valid participant IDs
  valid_participant_ids = list(participant_ids)
  
  removed_pids = []
  for pid in tqdm(participant_ids):
      # Skip if participant is not in the SPICE model
      if pid not in agent_spice._model.submodules_sindy[list(agent_spice._model.submodules_sindy.keys())[0]]:
          continue
         
      # Calculate normalized log likelihood for SPICE and RNN
      mask_participant_id = dataset.xs[:, 0, -1] == pid
      if not mask_participant_id.any():
          continue
          
      participant_data = DatasetRNN(*dataset[mask_participant_id])
      
      # Get probabilities from SPICE and RNN models
      agent_spice.new_sess(participant_id=pid)
      agent_rnn.new_sess(participant_id=pid)
      
      # Calculate NLL for both models
      probs_spice = get_update_dynamics(experiment=participant_data.xs, agent=agent_spice)[1]
      probs_rnn = get_update_dynamics(experiment=participant_data.xs, agent=agent_rnn)[1]
      n_trials_test = len(probs_spice)
      
      # Calculate scores (negative log likelihood)
      ll_spice = log_likelihood(data=participant_data.ys[0, :n_trials_test].cpu().numpy(), probs=probs_spice)
      ll_rnn = log_likelihood(data=participant_data.ys[0, :n_trials_test].cpu().numpy(), probs=probs_rnn)
      
      # Check if SPICE model is badly fitted
      spice_per_action_likelihood = np.exp(ll_spice/(n_trials_test*agent_rnn._n_actions))
      rnn_per_action_likelihood = np.exp(ll_rnn/(n_trials_test*agent_rnn._n_actions))
      
      # Idea for filter criteria:
      # If accuracy is very low for SPICE (near chance) but not so low for RNN then bad SPICE fitting (at least a bit higher than chance)
      # TODO: Check for better filter criteria
      if rnn_per_action_likelihood - spice_per_action_likelihood > trial_likelihood_difference_threshold:
          if verbose:
              print(f'SPICE trial likelihood ({spice_per_action_likelihood:.2f}) is unplausibly low compared to RNN trial likelihood ({rnn_per_action_likelihood:.2f}).')
              print(f'SPICE optimizer may be badly parameterized. Skipping participant {pid}.')
          
          # Remove this participant from the SPICE model
          for module in agent_spice._model.submodules_sindy:
              if pid in agent_spice._model.submodules_sindy[module]:
                  del agent_spice._model.submodules_sindy[module][pid]
          
          # Remove from valid participant IDs
          if pid in valid_participant_ids:
              valid_participant_ids.remove(pid)
              removed_pids.append(pid)
      else:
          # Keep track of filtered (good) participants
          filtered_participant_ids.append(pid)
  
  if verbose:
      print(f"\nAfter filtering: {len(valid_participant_ids)} of {len(participant_ids)} participants have well-fitted SPICE models.")
      print(f"Removed participants: {removed_pids}")
  
  return agent_spice, np.array(valid_participant_ids)


def save_spice(agent_spice: AgentSpice, file: str):
  """Saves the SINDy models (coefficients) in a pickle file
  
  Args:
      file (str): file where to store the SINDy coefficients
  """  
    
  with open(file, 'wb') as f:
    dill.dump(agent_spice.get_modules(), f)
    
    
def load_spice(file) -> Dict:
  with open(file, 'rb') as f:
    spice_modules = dill.load(f)
  return spice_modules


def generate_off_policy_data(participant_id: int, block: int, experiment_id: int, n_trials_off_policy: int, n_trials_same_action_off_policy: int, n_sessions_off_policy: int = 1, n_actions: int = 2, additional_inputs: np.ndarray = np.zeros(0), sigma_drift: float = 0.2) -> DatasetRNN:
  """Generates a simple off-policy dataset where each action is repeated n_trials_same_action_off_policy times and then switched.

  Args:
      participant_id (int): _description_
      n_trials_off_policy (int): _description_
      n_trials_same_action_off_policy (int): _description_
      n_sessions_off_policy (int, optional): _description_. Defaults to 1.
      n_actions (int, optional): _description_. Defaults to 2.

  Returns:
      DatasetRNN: Off-policy dataset
  """
  
  # set up environment to create an off-policy dataset (w.r.t to trained RNN) of arbitrary length
  # The trained RNN will then perform value updates to get off-policy data
  environment = BanditsDrift(sigma=sigma_drift, n_actions=n_actions)
  
  # create a dummy dataset where each choice is chosen for n times and then an action switch occures
  xs = torch.zeros((n_sessions_off_policy, n_trials_off_policy, 2*n_actions+3+additional_inputs.shape[-1])) - 1
  for session in range(n_sessions_off_policy):
      # initialize first action
      current_action = torch.zeros(n_actions)
      current_action[0] = 1
      for trial in range(n_trials_off_policy):
          current_action_index = torch.argmax(current_action).int().item()
          reward = torch.tensor(environment.step(current_action_index))
          xs[session, trial, :2*n_actions] = torch.concat((current_action, reward))
          # action switch - go to next possible action item and if final go to first one
          if trial >= n_trials_same_action_off_policy and trial % n_trials_same_action_off_policy == 0:
              current_action[current_action_index] = 0
              current_action[current_action_index+1 if current_action_index+1 < len(current_action) else 0] = 1
              
  # setup of dataset
  ys = xs[:, 1:, :n_actions]
  xs = xs[:, :-1]
  xs[..., 2*n_actions:-3] = additional_inputs
  xs[..., -3] = block
  xs[..., -2] = experiment_id
  xs[..., -1] = participant_id
  
  # dataset_fit = DatasetRNN(xs_fit, ys_fit)
  # # repeat the off-policy data for every participant and add the corresponding participant ID
  # xs_fit = dataset_fit.xs.repeat(len(participant_ids), 1, 1)
  # ys_fit = dataset_fit.ys.repeat(len(participant_ids), 1, 1)
  # set participant ids correctly
  # for index_pid in range(0, len(participant_ids)):
  #     xs[n_sessions_off_policy*index_pid:n_sessions_off_policy*(index_pid+1):, :, -1] = participant_ids[index_pid]
  
  return DatasetRNN(xs=xs, ys=ys)


SindyConfig = {
  
  # tracked variables and control signals in the RNN
  'rnn_modules': ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
  'control_parameters': ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice'],
  
  # library setup: 
  # which terms are allowed as control inputs in each SINDy model
  # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
  'library_setup': {
      'x_learning_rate_reward': ['c_reward', 'c_value_reward', 'c_value_choice'],
      'x_value_reward_not_chosen': ['c_value_choice'],
      'x_value_choice_chosen': ['c_value_reward'],
      'x_value_choice_not_chosen': ['c_value_reward'],
  },
  
  # data-filter setup: 
  # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
  # key is the SINDy model name, value is a list with a triplet of values:
  #   1. str: feature name to be used as a filter
  #   2. numeric: the numeric filter condition
  #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
  # Multiple conditions can also be given as a list of triplets.
  # Example:
  #   'x_value_choice_not_chosen': ['c_action', 0, True] means that for the SINDy model 'x_value_choice_not_chosen', only samples where the feature 'c_action' == 0 are used for training the SINDy model. 
  #   The control parameter 'c_action' is removed afterwards from the list of control signals for training of the model
  'filter_setup': {
      'x_learning_rate_reward': ['c_action', 1, True],
      'x_value_reward_not_chosen': ['c_action', 0, True],
      'x_value_choice_chosen': ['c_action', 1, True],
      'x_value_choice_not_chosen': ['c_action', 0, True],
  },
  
  # data pre-processing setup:
  # define the processing steps for each variable and control signal.
  # possible processing steps are: 
  #   1. Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
  #   2. Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
  #   3. Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
  # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
  'dataprocessing_setup': {
      'x_learning_rate_reward': [0, 0, 0],
      'x_value_reward_not_chosen': [0, 0, 0],
      'x_value_choice_chosen': [1, 1, 0],
      'x_value_choice_not_chosen': [1, 1, 0],
      'c_value_reward': [0, 0, 0],
      'c_value_choice': [1, 1, 0],
  },
  
}


SindyConfig_eckstein2022 = {
  
  # tracked variables and control signals in the RNN
  'rnn_modules': ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
  'control_parameters': ['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice'],
  
  # library setup: 
  # which terms are allowed as control inputs in each SINDy model
  # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
  'library_setup': {
      'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
      'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
      'x_value_choice_chosen': ['c_value_reward'],
      'x_value_choice_not_chosen': ['c_value_reward'],
  },
  
  # data-filter setup: 
  # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
  # key is the SINDy model name, value is a list with a triplet of values:
  #   1. str: feature name to be used as a filter
  #   2. numeric: the numeric filter condition
  #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
  # Multiple conditions can also be given as a list of triplets.
  # Example:
  #   'x_value_choice_not_chosen': ['c_action', 0, True] means that for the SINDy model 'x_value_choice_not_chosen', only samples where the feature 'c_action' == 0 are used for training the SINDy model. 
  #   The control parameter 'c_action' is removed afterwards from the list of control signals for training of the model
  'filter_setup': {
      'x_learning_rate_reward': ['c_action', 1, True],
      'x_value_reward_not_chosen': ['c_action', 0, True],
      'x_value_choice_chosen': ['c_action', 1, True],
      'x_value_choice_not_chosen': ['c_action', 0, True],
  },
  
  # data pre-processing setup:
  # define the processing steps for each variable and control signal.
  # possible processing steps are: 
  #   1. Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
  #   2. Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
  #   3. Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
  # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
  'dataprocessing_setup': {
      'x_learning_rate_reward': [0, 0, 0],
      'x_value_reward_not_chosen': [0, 0, 0],
      'x_value_choice_chosen': [0, 0, 0],
      'x_value_choice_not_chosen': [0, 0, 0],
      'c_value_reward': [0, 0, 0],
      'c_value_choice': [0, 0, 0],
  },
  
}


SindyConfig_dezfouli2019 = {
  
  # tracked variables and control signals in the RNN
  'rnn_modules': ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
  'control_parameters': ['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice'],
  
  # library setup: 
  # which terms are allowed as control inputs in each SINDy model
  # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
  'library_setup': {
      'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
      'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
      'x_value_choice_chosen': ['c_value_reward'],
      'x_value_choice_not_chosen': ['c_value_reward'],
  },
  
  # data-filter setup: 
  # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
  # key is the SINDy model name, value is a list with a triplet of values:
  #   1. str: feature name to be used as a filter
  #   2. numeric: the numeric filter condition
  #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
  # Multiple conditions can also be given as a list of triplets.
  # Example:
  #   'x_value_choice_not_chosen': ['c_action', 0, True] means that for the SINDy model 'x_value_choice_not_chosen', only samples where the feature 'c_action' == 0 are used for training the SINDy model. 
  #   The control parameter 'c_action' is removed afterwards from the list of control signals for training of the model
  'filter_setup': {
      'x_learning_rate_reward': ['c_action', 1, True],
      'x_value_reward_not_chosen': ['c_action', 0, True],
      'x_value_choice_chosen': ['c_action', 1, True],
      'x_value_choice_not_chosen': ['c_action', 0, True],
  },
  
  # data pre-processing setup:
  # define the processing steps for each variable and control signal.
  # possible processing steps are: 
  #   1. Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
  #   2. Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
  #   3. Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
  # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
  'dataprocessing_setup': {
      'x_learning_rate_reward': [0, 0, 0],
      'x_value_reward_not_chosen': [0, 0, 0],
      'x_value_choice_chosen': [0, 0, 0],
      'x_value_choice_not_chosen': [0, 0, 0],
      'c_value_reward': [0, 0, 0],
      'c_value_choice': [0, 0, 0],
  },
  
}


SindyConfig_dezfouli2019_blocks = {
  
  # tracked variables and control signals in the RNN
  'rnn_modules': ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen', 'x_value_block', 'x_value_trial'],
  'control_parameters': ['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice', 'c_value_block', 'c_value_trial'],
  
  # library setup: 
  # which terms are allowed as control inputs in each SINDy model
  # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
  'library_setup': {
      'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice', 'c_value_block', 'c_value_trial'],
      'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice', 'c_value_block', 'c_value_trial'],
      'x_value_choice_chosen': ['c_value_reward', 'c_value_block', 'c_value_trial'],
      'x_value_choice_not_chosen': ['c_value_reward', 'c_value_block', 'c_value_trial'],
  },
  
  # data-filter setup: 
  # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
  # key is the SINDy model name, value is a list with a triplet of values:
  #   1. str: feature name to be used as a filter
  #   2. numeric: the numeric filter condition
  #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
  # Multiple conditions can also be given as a list of triplets.
  # Example:
  #   'x_value_choice_not_chosen': ['c_action', 0, True] means that for the SINDy model 'x_value_choice_not_chosen', only samples where the feature 'c_action' == 0 are used for training the SINDy model. 
  #   The control parameter 'c_action' is removed afterwards from the list of control signals for training of the model
  'filter_setup': {
      'x_learning_rate_reward': ['c_action', 1, True],
      'x_value_reward_not_chosen': ['c_action', 0, True],
      'x_value_choice_chosen': ['c_action', 1, True],
      'x_value_choice_not_chosen': ['c_action', 0, True],
  },
  
  # data pre-processing setup:
  # define the processing steps for each variable and control signal.
  # possible processing steps are: 
  #   1. Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
  #   2. Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
  #   3. Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
  # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
  'dataprocessing_setup': {
      'x_learning_rate_reward': [0, 0, 0],
      'x_value_reward_not_chosen': [0, 0, 0],
      'x_value_choice_chosen': [0, 0, 0],
      'x_value_choice_not_chosen': [0, 0, 0],
      'c_value_reward': [0, 0, 0],
      'c_value_choice': [0, 0, 0],
  },
  
}
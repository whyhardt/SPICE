from typing import Tuple, Union
from torch import device
import numpy as np
import torch

from ..resources.rnn import BaseRNN
from ..estimator import SpiceEstimator


class Agent:
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

      self.deterministic = deterministic
      self.use_sindy = use_sindy
      self.device = device
      self.state = {}
      self.model = model_rnn.eval(use_sindy=use_sindy).to(device) if isinstance(model_rnn, BaseRNN) else model_rnn.eval().to(device)
      self.new_sess()
      
  def new_sess(self, participant_id: int = 0, experiment_id: int = 0, additional_embedding_inputs: np.ndarray = torch.zeros(0), **kwargs):
    """Reset the network for the beginning of a new session."""
    if not isinstance(participant_id, torch.Tensor):
      participant_id = torch.tensor(participant_id, dtype=int, device=self.device)[None]
    
    if isinstance(self.model, BaseRNN):
      self.model.init_state(batch_size=1)
      state = self.model.get_state()
    else:
      state = None
      
    self.logits = torch.zeros((1, self.n_actions), device=self.device, dtype=torch.float32)
    
    self.meta_data = torch.zeros((1, 2), dtype=torch.float32)
    self.meta_data[0, -1] = participant_id
    self.meta_data[0, -2] = experiment_id
    
    self.additional_meta_data = additional_embedding_inputs if isinstance(additional_embedding_inputs, torch.Tensor) else torch.tensor(additional_embedding_inputs, dtype=torch.float32)
    self.additional_meta_data = self.additional_meta_data.view(1, -1)
    
    self.set_state(self.logits, state)

  def update(self, choice: float, reward: float, block: int = 0, additional_inputs: np.ndarray = torch.zeros(0), **kwargs):
    if not isinstance(reward, np.ndarray):
      reward_array = np.zeros(self.n_actions) + np.nan
      reward_array[int(choice)] = reward
      reward = reward_array
      
    choice = torch.eye(self.n_actions, dtype=torch.float32)[int(choice)]
    
    xs = torch.concat([choice, torch.tensor(reward, dtype=torch.float32), torch.tensor(additional_inputs, dtype=torch.float32), torch.tensor(block, dtype=torch.float32).view(1), self.meta_data.view(-1)]).view(1, 1, -1).to(device=self.device)
    
    with torch.no_grad():
      logits, state = self.model(xs, self.get_state()[1] if isinstance(self.model, BaseRNN) else self.get_state()[1]['hidden'])
    
    self.set_state(logits, state)
  
  def set_state(self, logits, state):
    self.logits = logits
    
    if isinstance(self.model, BaseRNN):
      self.state = self.model.get_state()
    else:
      self.state['hidden'] = state
      
  def get_state(self, numpy: bool = False, **kwargs):    
    if isinstance(self.model, BaseRNN):
      state = self.model.get_state(detach=True)
    else:
      state = self.state
      
    if numpy:
      state_numpy = {}
      logits = self.logits.detach().cpu().numpy()
      for key in state:
        if isinstance(state[key], torch.Tensor):
          state_numpy[key] = state[key].detach().cpu().numpy()
      return logits, state_numpy
    else:
      return self.logits, state
      
  def get_participant_ids(self):
    if hasattr(self.model, 'participant_embedding'):
      return tuple(np.arange(self.model.participant_embedding.num_embeddings).tolist())
    
  def get_modules(self):
    if isinstance(self.model, BaseRNN):
      return tuple(list(self.model.submodules_rnn.keys()))
    else:
      raise TypeError(f"Agent model is not a SPICE model. This function is only executable for SPICE models.")
  
  def count_parameters(self) -> np.ndarray:
    if isinstance(self.model, BaseRNN):
      return self.model.count_sindy_coefficients().detach().cpu().numpy()
    else:
      raise TypeError(f"Agent model is not a SPICE model. This function is only executable for SPICE models.")

  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    logits = self.logits
    if isinstance(logits, torch.Tensor):
      logits = logits.detach().cpu().numpy()
      
    decision_variable = logits - logits.min()
    decision_variable = np.exp(decision_variable)
    choice_probs = decision_variable / np.sum(decision_variable)
    return choice_probs.reshape(self.n_actions)
  
  def get_choice(self):
    """Sample choice."""
    choice_probs = self.get_choice_probs()
    if self.deterministic:
      return np.argmax(choice_probs)
    else:
      return np.random.choice(self.n_actions, p=choice_probs)
      
  
def get_update_dynamics(experiment: Union[np.ndarray, torch.Tensor], agent: Agent):
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
    choices = np.nan_to_num(experiment[:n_trials, :agent.n_actions])
    rewards = np.nan_to_num(experiment[:n_trials, agent.n_actions:2*agent.n_actions])
    # TODO: additional_inputs are currently treated as signals and as meta-information for the embedding
    additional_inputs = np.nan_to_num(experiment[0, 2*agent.n_actions:-3])
    current_block = np.nan_to_num(int(experiment[0, -3]))
    experiment_id = np.nan_to_num(int(experiment[0, -2]))
    participant_id = np.nan_to_num(int(experiment[0, -1]))
  else:
    raise TypeError("experiment is of not of class numpy.ndarray or torch.Tensor")
  
  # reset agent states according to ID
  agent.new_sess(participant_id=participant_id, experiment_id=experiment_id, additional_embedding_inputs=additional_inputs)
  
  # initialize storages
  logits = np.zeros((n_trials, agent.n_actions))
  additional_signals = [state for state in agent.state if 'value' in state]
  n_values = agent.n_actions
  if hasattr(agent, 'model'):
    if hasattr(agent.model, 'n_items'):
      n_values = agent.model.n_items
  state_values = {signal: np.zeros((n_trials, n_values)) for signal in additional_signals}
  choice_probs = np.zeros((n_trials, agent.n_actions))
  
  for trial in range(n_trials):
    # track all states
    current_logits, state = agent.get_state(numpy=True)      
    logits[trial] = current_logits[0]
    for signal in additional_signals:
      if isinstance(agent.state, dict):
        if signal in state:
          value = state[signal]
        else: 
          value = np.zeros_like(agent.logits)
      else:
        value = np.zeros(agent.n_actions)
      if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
      state_values[signal][trial] = value
    
    choice_probs[trial] = np.clip(agent.get_choice_probs(), 1e-8, 1)
    
    agent.update(
      choice=np.argmax(choices[trial], axis=-1), 
      reward=rewards[trial],  
      block=current_block, 
      additional_inputs=additional_inputs,
      )
  
  return (logits, state_values), choice_probs, agent
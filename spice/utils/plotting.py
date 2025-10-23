import sys, os
from typing import List, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import torch
from typing import Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import AgentQ, AgentNetwork, BanditSession, get_update_dynamics, plot_session as plt_session
from resources.spice_utils import SpiceDataset


def plot_dynamics(
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
  reward_range=(0, 1),  # New parameter for reward range
  ):
  """Plot data from a single behavioral session of the bandit task.

  Args:
    choices: The choices made by the agent
    rewards: The rewards received by the agent (can be binary or float)
    timeseries: The dynamical value of interest on each arm
    timeseries_name: The name of the timeseries
    labels: The labels for the lines in the plot
    fig_ax: A tuple of a matplotlib figure and axis to plot on
    compare: If True, plot multiple timeseries on the same plot
    color: A list of colors to use for the plot; at least as long as the number of timeseries
    reward_range: Tuple specifying the min and max possible reward values
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
  
  # Plot ticks relating to whether the option was chosen (factual) or not (counterfactual) and reward value
  min_y, max_y = np.min(timeseries), np.max(timeseries)
  diff_min_max = np.max((1e-1, max_y - min_y))
  
  x = np.arange(len(choices))
  chosen_y = min_y - 1e-1  # Lower position for chosen (bigger tick)
  not_chosen_y = max_y + 1e-1  # Upper position for not chosen (smaller tick)

  # Set up color mapping for rewards
#   reward_min, reward_max = reward_range
  reward_min, reward_max = reward_range[0], reward_range[1]
  norm = Normalize(vmin=reward_min, vmax=reward_max)
  
  # Use RdYlGn colormap (Red-Yellow-Green) reversed so red is negative, green is positive
  cmap = cm.RdYlGn
  
  # Check if rewards are binary or continuous
  is_binary = reward_min == 0 and reward_max == 1 and np.all(np.isin(rewards, [-1, 0, 1]))
  
  if is_binary:
    # Handle binary rewards (backward compatibility)
    # Plot ticks for chosen options
    ax.scatter(x[(choices == 1) & (rewards == 1)], np.full(sum((choices == 1) & (rewards == 1)), chosen_y), 
               color='green', s=120, marker='|')  # Large green tick for chosen reward
    ax.scatter(x[(choices == 1) & (rewards == 0)], np.full(sum((choices == 1) & (rewards == 0)), chosen_y), 
               color='red', s=80, marker='|')  # Large red tick for chosen penalty

    # Plot ticks for not chosen options
    # ax.scatter(x[(choices == 0) & (rewards == 1)], np.full(sum((choices == 0) & (rewards == 1)), not_chosen_y), 
    #            color='green', s=120, marker='|')  # Small green tick
    # ax.scatter(x[(choices == 0) & (rewards == 0)], np.full(sum((choices == 0) & (rewards == 0)), not_chosen_y), 
    #            color='red', s=80, marker='|')  # Small red tick
  else:
    # Handle continuous rewards
    # Plot ticks for chosen options
    chosen_mask = (choices == 1)
    if np.any(chosen_mask):
      chosen_rewards = rewards[chosen_mask]
      chosen_colors = cmap(norm(chosen_rewards))
      # Size based on absolute reward value (larger for more extreme rewards)
      chosen_sizes = 60 + 60 * np.abs(chosen_rewards) / max(abs(reward_min), abs(reward_max))
      ax.scatter(x[chosen_mask], np.full(sum(chosen_mask), chosen_y), 
                 c=chosen_colors, s=chosen_sizes, marker='|', alpha=0.8)

    # Plot ticks for not chosen options
    # not_chosen_mask = (choices == 0)
    # if np.any(not_chosen_mask):
    #   not_chosen_rewards = rewards[not_chosen_mask]
    #   not_chosen_colors = cmap(norm(not_chosen_rewards))
    #   # Smaller size for not chosen options
    #   not_chosen_sizes = 40 + 40 * np.abs(not_chosen_rewards) / max(abs(reward_min), abs(reward_max))
    #   ax.scatter(x[not_chosen_mask], np.full(sum(not_chosen_mask), not_chosen_y), 
    #              c=not_chosen_colors, s=not_chosen_sizes, marker='|', alpha=0.6)
    
    # Add a colorbar to show the reward scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Reward Value')
    
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
  
  return fig, ax


def plot_session(
    agents: Dict[str, Union[AgentNetwork, AgentQ]], 
    experiment: Union[BanditSession, np.ndarray], 
    labels: List[str] = None, 
    save: str = None, 
    display_choice: int = 0,
    reward_range: List[float] = [0, 1],
    signals_to_plot: List[str] = ['value_reward', 'value_choice'],
    ):    
    # plot the dynamcis associated with the first arm
    
    y_axis = True
    x_axis = True
    
    # valid keys in agent dictionary
    valid_keys_color_pairs = {'groundtruth': 'tab:blue', 'rnn': 'tab:orange', 'sindy': 'tab:pink', 'spice': 'tab:pink', 'benchmark':'tab:grey'}
    
    n_actions = agents[list(agents.keys())[0]]._n_actions
    if isinstance(experiment, BanditSession):
        choices = np.eye(n_actions)[experiment.choices.astype(int)][:, display_choice]
        rewards = experiment.rewards[:, display_choice]
    elif isinstance(experiment, np.ndarray) or isinstance(experiment, torch.Tensor):
        if isinstance(experiment, torch.Tensor):
            experiment = experiment.detach().cpu().numpy()
        assert experiment.ndim == 2, 'Experiment data should have only two dimensions -> (timesteps, features)'
        # choices = experiment[:, :n_actions].argmax(axis=-1)
        # rewards = np.array([exp[choices[i]] for i, exp in enumerate(experiment[:, n_actions:2*n_actions])])  
        choices = experiment[:, display_choice]
        rewards = experiment[:, n_actions]  
    
    list_probs = []
    list_q_value = []
    list_signals = {signal: [] for signal in signals_to_plot}
    values_signals = {signal: [] for signal in signals_to_plot}

    colors = []
    if labels is None:
        labels = []
        
    for valid_key in valid_keys_color_pairs:        
        # get q-values from agent
        if valid_key in [key.lower() for key in agents]:
            qs, probs, _ = get_update_dynamics(experiment, agents[valid_key], additional_signals=signals_to_plot)
            list_probs.append(np.expand_dims(probs, 0))
            list_q_value.append(np.expand_dims(qs[0], 0))
            for signal in signals_to_plot:
              list_signals[signal].append(np.expand_dims(qs[1][signal], 0))
            
            # get color from current agent
            colors.append(valid_keys_color_pairs[valid_key])
            
            if len(labels) < len(agents):
                # get labels from current agent
                labels.append(valid_key)

    # concatenate all choice probs and q-values
    probs = np.concatenate(list_probs, axis=0)
    q_value = np.concatenate(list_q_value, axis=0)
    for signal in signals_to_plot:
      values_signals[signal] = np.concatenate(list_signals[signal], axis=0)

    fig, axs = plt.subplots(2 + len(signals_to_plot), 1, figsize=(10, 10))
    axs_row = 0
    fig_col = None
    
    plot_dynamics(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=probs[:, :, display_choice],
        timeseries_name='$P(action)$',
        color=colors,
        labels=labels,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        x_axis_info=x_axis,
        y_axis_info=y_axis,
        reward_range=reward_range,
        )
    axs_row += 1
    
    plot_dynamics(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=q_value[:, :, display_choice],
        timeseries_name='$q$',
        color=colors,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        y_axis_info=x_axis,
        x_axis_info=y_axis,
        reward_range=reward_range,
        )
    axs_row += 1
    
    for signal in signals_to_plot:
      plot_dynamics(
          compare=True,
          choices=choices,
          rewards=rewards,
          timeseries=values_signals[signal][:, :, display_choice],
          timeseries_name=signal,
          color=colors,
          fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
          x_axis_info=x_axis,
          y_axis_info=y_axis,
          reward_range=reward_range,
          )
      axs_row += 1
    
    if save is not None:
        plt.savefig(save, dpi=300)
    
    return fig, axs
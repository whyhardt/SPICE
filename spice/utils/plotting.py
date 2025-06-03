import sys, os
from typing import List, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.bandits import AgentQ, AgentNetwork, AgentSpice, BanditSession, get_update_dynamics, plot_session as plt_session
from spice.resources.rnn_utils import DatasetRNN

def plot_session(agents: Dict[str, Union[AgentSpice, AgentNetwork, AgentQ]], experiment: Union[BanditSession, np.ndarray], labels: List[str] = None, save: str = None):    
    # plot the dynamcis associated with the first arm
    
    # valid keys in agent dictionary
    valid_keys_color_pairs = {'groundtruth': 'tab:blue', 'rnn': 'tab:orange', 'spice': 'tab:pink', 'benchmark':'tab:grey'}    
    
    n_actions = agents[list(agents.keys())[0]]._n_actions
    if isinstance(experiment, BanditSession):
        choices = np.eye(n_actions)[experiment.choices.astype(int)][:, 0]
        rewards = experiment.rewards[:, 0]
    elif isinstance(experiment, np.ndarray) or isinstance(experiment, torch.Tensor):
        if isinstance(experiment, torch.Tensor):
            experiment = experiment.detach().cpu().numpy()
        assert experiment.ndim == 2, 'Experiment data should have only two dimensions -> (timesteps, features)'
        # choices = experiment[:, :n_actions].argmax(axis=-1)
        # rewards = np.array([exp[choices[i]] for i, exp in enumerate(experiment[:, n_actions:2*n_actions])])  
        choices = experiment[:, 0]
        rewards = experiment[:, n_actions]  
    
    list_probs = []
    list_Qs = []
    list_qs = []
    list_cs = []
    list_alphas = []

    colors = []
    if labels is None:
        labels = []
        
    for valid_key in valid_keys_color_pairs:        
        # get q-values from agent
        if valid_key in [key.lower() for key in agents]:
            qs, probs, _ = get_update_dynamics(experiment, agents[valid_key])
            list_probs.append(np.expand_dims(probs, 0))
            list_Qs.append(np.expand_dims(qs[0], 0))
            list_qs.append(np.expand_dims(qs[1], 0))
            list_cs.append(np.expand_dims(qs[2], 0))
            list_alphas.append(np.expand_dims(qs[3], 0))
            
            # get color from current agent
            colors.append(valid_keys_color_pairs[valid_key])
            
            if len(labels) < len(agents):
                # get labels from current agent
                labels.append(valid_key)

    # concatenate all choice probs and q-values
    probs = np.concatenate(list_probs, axis=0)
    Qs = np.concatenate(list_Qs, axis=0)
    qs = np.concatenate(list_qs, axis=0)
    cs = np.concatenate(list_cs, axis=0)
    alphas = np.concatenate(list_alphas, axis=0)

    # normalize q-values
    # def normalize(qs):
    #     return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

    # qs = normalize(qs)

    fig, axs = plt.subplots(5, 1, figsize=(20, 10))
    axs_row = 0
    fig_col = None
    
    plt_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=probs[:, :, 0],
        timeseries_name='$P(action)$',
        color=colors,
        labels=labels,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        x_axis_info=False,
        y_axis_info=True,
        )
    axs_row += 1
    
    plt_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=Qs[:, :, 0],
        timeseries_name='$q$',
        color=colors,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        y_axis_info=True,
        x_axis_info=False,
        )
    axs_row += 1
    
    plt_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=qs[:, :, 0],
        timeseries_name='$q_{reward}$',
        color=colors,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        x_axis_info=False,
        y_axis_info=True,
        )
    axs_row += 1
    
    plt_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=alphas[:, :, 0],
        timeseries_name=r'$\alpha$',
        color=colors,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        x_axis_info=False,
        y_axis_info=True,
        )
    axs_row += 1
    
    plt_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=cs[:, :, 0],
        timeseries_name='$q_{choice}$',
        color=colors,
        fig_ax=(fig, axs[axs_row, fig_col]) if fig_col is not None else (fig, axs[axs_row]),
        x_axis_info=True,
        y_axis_info=True,
        )
    axs_row += 1
    
    if save is not None:
        plt.savefig(save, dpi=300)
    
    return fig, axs
    
def pca(agents: Dict[str, Union[AgentSpice, AgentNetwork, AgentQ]], experiments: List[BanditSession], labels=None, save=None):
    
    # valid keys in agent dictionary
    valid_keys_color_pairs = {'groundtruth': 'tab:blue', 'rnn': 'tab:orange', 'sindy': 'tab:pink', 'benchmark':'tab:grey'}    
    
    choices = experiment.choices
    rewards = experiment.rewards

    list_probs = []
    list_Qs = []
    list_qs = []
    list_hs = []

    colors = []
    if labels is None:
        labels = []
        
    for valid_key in valid_keys_color_pairs:        
        # get q-values from agent
        if valid_key in [key.lower() for key in agents]:
            qs_test, probs_test, _ = get_update_dynamics(experiment, agents[valid_key])
            list_probs.append(np.expand_dims(probs_test, 0))
            list_Qs.append(np.expand_dims(qs_test[0], 0))
            list_qs.append(np.expand_dims(qs_test[1], 0))
            list_hs.append(np.expand_dims(qs_test[2], 0))
            
            # get color from current agent
            colors.append(valid_keys_color_pairs[valid_key])
            
            if len(labels) < len(agents):
                # get labels from current agent
                labels.append(valid_key)

    # concatenate all choice probs and q-values
    probs = np.concatenate(list_probs, axis=0)
    Qs = np.concatenate(list_Qs, axis=0)
    qs = np.concatenate(list_qs, axis=0)
    hs = np.concatenate(list_hs, axis=0)
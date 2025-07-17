import sys
import os

from typing import List, Dict
from torch import device, load
import numpy as np
import torch
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn import BaseRNN
from resources.bandits import AgentSpice, AgentNetwork, AgentQ
from resources.sindy_training import fit_spice
from resources.sindy_utils import load_spice
from utils.convert_dataset import convert_dataset


def setup_rnn(
    class_rnn,
    path_model,
    n_actions=2,
    counterfactual=False,
    device=device('cpu'),
    **kwargs,
) -> BaseRNN:
    
    # get n_participants and hidden_size from state dict
    state_dict = torch.load(path_model, map_location=torch.device('cpu'))
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    participant_embedding_index = [i for i, s in enumerate(list(state_dict.keys())) if 'participant_embedding' in s]
    participant_embedding_bool = True if len(participant_embedding_index) > 0 else False
    n_participants = 0 if not participant_embedding_bool else state_dict[list(state_dict.keys())[participant_embedding_index[0]]].shape[0]
    
    key_hidden_size = [key for key in state_dict if 'x' in key.lower()][0]  # first key that contains the hidden_size
    hidden_size = state_dict[key_hidden_size].shape[0]
    
    key_embedding_size = [key for key in state_dict if 'embedding' in key.lower()]
    if len(key_embedding_size) > 0:
        embedding_size = state_dict[key_embedding_size[0]].shape[1]
    else:
        embedding_size = 0
        
    rnn = class_rnn(
        n_actions=n_actions, 
        hidden_size=hidden_size, 
        embedding_size=embedding_size,
        n_participants=n_participants, 
        device=device, 
        counterfactual=counterfactual,
        )
    rnn.load_state_dict(state_dict)
        
    return rnn


def setup_agent_rnn(
    class_rnn,
    path_model,
    n_actions=2,
    counterfactual=False,
    deterministic=True,
    device=device('cpu'),
    **kwargs,
    ) -> AgentNetwork:
    
    rnn = setup_rnn(class_rnn=class_rnn, path_model=path_model, device=device, n_actions=n_actions, counterfactual=counterfactual)
    agent = AgentNetwork(model_rnn=rnn, n_actions=n_actions, deterministic=deterministic)
    
    return agent


def setup_agent_spice(
    class_rnn: type,
    path_rnn: str,
    path_spice: str = None,
    path_data: str = None,
    rnn_modules: List[str] = None,
    control_parameters: List[str] = None,
    sindy_library_polynomial_degree: int = None,
    sindy_library_setup: Dict[str, List] = None,
    sindy_filter_setup: Dict[str, List] = None,
    sindy_dataprocessing: Dict[str, List] = None,
    threshold: float = 0.05,
    regularization: float = 0.1,
    participant_id: int = None,
    deterministic: bool = True,
    filter_bad_participants: bool = False,
    use_optuna: bool = False,
    verbose: bool = False,
    **kwargs,
) -> AgentSpice:
    
    agent_rnn = setup_agent_rnn(class_rnn=class_rnn, path_model=path_rnn)
    
    if path_spice is None or path_spice == '':
        dataset = convert_dataset(file=path_data)[0]

        # fit SPICE model to RNN
        agent_spice, _ = fit_spice(
            agent_rnn=agent_rnn,
            data=dataset,
            rnn_modules=rnn_modules,
            control_signals=control_parameters,
            polynomial_degree=sindy_library_polynomial_degree,
            library_setup=sindy_library_setup,
            filter_setup=sindy_filter_setup,
            dataprocessing=sindy_dataprocessing,
            participant_id=participant_id,
            n_sessions_off_policy=1,
            n_trials_off_policy=1000,
            optimizer_type='SR3_weighted_l1',
            optimizer_alpha=regularization,
            optimizer_threshold=threshold,
            deterministic=deterministic,
            filter_bad_participants=filter_bad_participants,
            use_optuna=use_optuna,
            verbose=verbose,
        )
    else:
        # load SPICE model from a file
        spice_modules = load_spice(path_spice)
        agent_spice = AgentSpice(model_rnn=agent_rnn._model, sindy_modules=spice_modules, n_actions=agent_rnn._n_actions)
        
    return agent_spice

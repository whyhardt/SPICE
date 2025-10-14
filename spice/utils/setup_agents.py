from typing import Tuple
from torch import device
import torch

from spice.resources.bandits import AgentNetwork
from spice.estimator import SpiceEstimator


def setup_agent(
    class_rnn,
    path_model,
    sindy_config,
    n_actions=2,
    sindy_polynomial_degree=2,
    counterfactual=False,
    deterministic=True,
    device=device('cpu'),
    **kwargs,
    ) -> Tuple[AgentNetwork, AgentNetwork]:
    
    # get n_participants and hidden_size from state dict
    state_dict = torch.load(path_model, map_location=torch.device('cpu'))
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    participant_embedding_index = [i for i, s in enumerate(list(state_dict.keys())) if 'participant_embedding' in s]
    participant_embedding_bool = True if len(participant_embedding_index) > 0 else False
    n_participants = 0 if not participant_embedding_bool else state_dict[list(state_dict.keys())[participant_embedding_index[0]]].shape[0]
    
    estimator = SpiceEstimator(
        rnn_class=class_rnn,
        spice_config=sindy_config,
        n_actions=n_actions,
        spice_library_polynomial_degree=sindy_polynomial_degree,
        n_participants=n_participants,
    )
    estimator.load_spice(path_model)
    
    return estimator.rnn_agent, estimator.spice_agent
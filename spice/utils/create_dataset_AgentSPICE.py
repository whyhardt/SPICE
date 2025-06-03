import sys, os

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spice.resources.bandits import create_dataset, BanditsDrift, get_update_dynamics
from spice.utils.setup_agents import setup_agent_spice


def main(path_rnn, path_spice, path_data, path_save, n_trials_per_session):
    # sindy configuration
    rnn_modules = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen']
    control_parameters = ['c_action', 'c_reward_chosen', 'c_value_reward', 'c_value_choice']
    sindy_library_setup = {
        'x_learning_rate_reward': ['c_reward_chosen', 'c_value_reward', 'c_value_choice'],
        'x_value_reward_not_chosen': ['c_reward_chosen', 'c_value_choice'],
        'x_value_choice_chosen': ['c_value_reward'],
        'x_value_choice_not_chosen': ['c_value_reward'],
    }
    sindy_filter_setup = {
        'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    }
    sindy_dataprocessing = None

    agent = setup_agent_spice(
        path_data=path_data, 
        path_rnn=path_rnn,
        path_spice=path_spice,
        rnn_modules=rnn_modules,
        control_parameters=control_parameters,
        sindy_library_setup=sindy_library_setup,
        sindy_filter_setup=sindy_filter_setup,
        sindy_dataprocessing=sindy_dataprocessing,
        sindy_library_polynomial_degree=1,
        deterministic=False,
        verbose=True,
        # participant_id=0,
        )

    environment = BanditsDrift(sigma=0.2)

    dataset, _, _ = create_dataset(
                agent=agent,
                environment=environment,
                n_trials=n_trials_per_session,
                n_sessions=agent._model.n_participants,
                verbose=False,
                )

    # dataset columns
    # general dataset columns
    session, choice, reward = [], [], []
    choice_prob_0, choice_prob_1, action_value_0, action_value_1, reward_value_0, reward_value_1, choice_value_0, choice_value_1 = [], [], [], [], [], [], [], []

    for i in tqdm(range(len(dataset))):    
        # get update dynamics
        experiment = dataset.xs[i].cpu().numpy()
        qs, choice_probs, _ = get_update_dynamics(experiment, agent)
        
        # append behavioral data
        session += list(experiment[:, -1])
        choice += list(np.argmax(experiment[:, :agent._n_actions], axis=-1))
        reward += list(np.max(experiment[:, agent._n_actions:agent._n_actions*2], axis=-1))
        
        # append update dynamics
        choice_prob_0 += list(choice_probs[:, 0])
        choice_prob_1 += list(choice_probs[:, 1])
        action_value_0 += list(qs[0][:, 0])
        action_value_1 += list(qs[0][:, 1])
        reward_value_0 += list(qs[1][:, 0])
        reward_value_1 += list(qs[1][:, 1])
        choice_value_0 += list(qs[2][:, 0])
        choice_value_1 += list(qs[2][:, 1])
        
    columns = ['session', 'choice', 'reward', 'choice_prob_0', 'choice_prob_1', 'action_value_0', 'action_value_1', 'reward_value_0', 'reward_value_1', 'choice_value_0', 'choice_value_1']
    data = np.stack((np.array(session), np.array(choice), np.array(reward), np.array(choice_prob_0), np.array(choice_prob_1), np.array(action_value_0), np.array(action_value_1), np.array(reward_value_0), np.array(reward_value_1), np.array(choice_value_0), np.array(choice_value_1)), axis=-1)
    df = pd.DataFrame(data=data, columns=columns)

    # data_save = path_data.replace('.', '_spice.')
    df.to_csv(path_save, index=False)

    print(f'Data saved to {path_save}')
    

if __name__=='__main__':
    path_rnn = 'params/eckstein2022/rnn_eckstein2022_reward.pkl'
    path_spice = 'params/eckstein2022/spice_eckstein2022_reward.pkl'
    path_data = 'data/eckstein2022/eckstein2022.csv'
    n_trials_per_session = 200
    
    main(
        path_rnn=path_rnn,
        path_spice=path_spice,
        path_data=path_data,
        path_save=path_data.replace('.', '_test_spice.'),
        n_trials_per_session=n_trials_per_session,
    )
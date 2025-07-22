import sys, os

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn_utils import parameter_file_naming
from resources.bandits import create_dataset, AgentQ, AgentQ_SampleZeros, BanditsDrift, BanditsSwitch, get_update_dynamics


n_sessions = [128]#[16, 32, 64, 128, 256]
n_trials_per_session = 100
n_iterations_per_n_sessions = 1
sigma = 0.2
base_name = 'data/data_*.csv'
sample_parameters = False

for iteration in range(n_iterations_per_n_sessions):
    for n_sess in n_sessions:
        dataset_name = base_name.replace('*', f'{n_sess}p_{iteration}')

        # agent = AgentQ_SampleZeros(
        #     beta_reward=3.,
        #     beta_choice=.5,
        #     zero_threshold=0.2,
        #     )
        
        agent = AgentQ(
            alpha_reward=0.25,
            beta_reward=3.,
            beta_choice=1,
            )

        environment = BanditsDrift(sigma=sigma)

        dataset, _, parameter_list = create_dataset(
                    agent=agent,
                    environment=environment,
                    n_trials=n_trials_per_session,
                    n_sessions=n_sess,
                    sample_parameters=sample_parameters,
                    verbose=False,
                    )

        # dataset columns
        # general dataset columns
        session, choice, reward = [], [], []
        choice_prob_0, choice_prob_1, action_value_0, action_value_1, reward_value_0, reward_value_1, choice_value_0, choice_value_1 = [], [], [], [], [], [], [], []
        # parameters
        beta_reward, alpha_reward, alpha_penalty = [], [], []
        beta_choice, alpha_choice = [], []
        confirmation_bias, forget_rate = [], []
        # parameter means
        mean_beta_reward, mean_alpha_reward, mean_alpha_penalty = [], [], []
        mean_beta_choice, mean_alpha_choice = [], []
        mean_confirmation_bias, mean_forget_rate = [], []
        
        for i in range(len(dataset)):    
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
            
            # append all model parameters for each trial
            beta_reward += [parameter_list[i]['beta_reward']]  * n_trials_per_session
            alpha_reward += [parameter_list[i]['alpha_reward']] * n_trials_per_session
            alpha_penalty += [parameter_list[i]['alpha_penalty']] * n_trials_per_session
            confirmation_bias += [parameter_list[i]['confirmation_bias']] * n_trials_per_session
            forget_rate += [parameter_list[i]['forget_rate']] * n_trials_per_session
            beta_choice += [parameter_list[i]['beta_choice']] * n_trials_per_session
            alpha_choice += [parameter_list[i]['alpha_choice']] * n_trials_per_session
            
            # append all mean model parameters for each trial
            mean_beta_reward += [agent._mean_beta_reward] * n_trials_per_session
            mean_alpha_reward += [agent._mean_alpha_reward] * n_trials_per_session
            mean_alpha_penalty += [agent._mean_alpha_penalty] * n_trials_per_session
            mean_confirmation_bias += [agent._mean_confirmation_bias] * n_trials_per_session
            mean_forget_rate += [agent._mean_forget_rate] * n_trials_per_session
            mean_beta_choice += [agent._mean_beta_choice] * n_trials_per_session
            mean_alpha_choice += [agent._mean_alpha_choice] * n_trials_per_session

        columns = ['session', 'choice', 'reward', 'choice_prob_0', 'choice_prob_1', 'action_value_0', 'action_value_1', 'reward_value_0', 'reward_value_1', 'choice_value_0', 'choice_value_1', 'beta_reward', 'alpha_reward', 'alpha_penalty', 'confirmation_bias', 'forget_rate', 'beta_choice', 'alpha_choice', 'mean_beta_reward', 'mean_alpha_reward', 'mean_alpha_penalty', 'mean_confirmation_bias', 'mean_forget_rate', 'mean_beta_choice', 'mean_alpha_choice']
        data = np.stack((np.array(session), np.array(choice), np.array(reward), np.array(choice_prob_0), np.array(choice_prob_1), np.array(action_value_0), np.array(action_value_1), np.array(reward_value_0), np.array(reward_value_1), np.array(choice_value_0), np.array(choice_value_1), np.array(beta_reward), np.array(alpha_reward), np.array(alpha_penalty), np.array(confirmation_bias), np.array(forget_rate), np.array(beta_choice), np.array(alpha_choice), np.array(mean_beta_reward), np.array(mean_alpha_reward), np.array(mean_alpha_penalty), np.array(mean_confirmation_bias), np.array(mean_forget_rate), np.array(mean_beta_choice), np.array(mean_alpha_choice)), axis=-1)#.swapaxes(1, 0)
        df = pd.DataFrame(data=data, columns=columns)
        
        if dataset_name is None:
            dataset_name = parameter_file_naming(
                'data/data', 
                alpha_reward=np.round(agent._mean_alpha_reward, 2), 
                beta_reward=np.round(agent._mean_beta_reward, 2), 
                forget_rate=np.round(agent._mean_forget_rate, 2), 
                alpha_penalty=np.round(agent._mean_alpha_penalty, 2), 
                confirmation_bias=np.round(agent._mean_confirmation_bias, 2), 
                alpha_choice=np.round(agent._mean_alpha_choice, 2), 
                beta_choice=np.round(agent._mean_beta_choice, 2),
                alpha_counterfactual=0.00,
                ).replace('.pkl','.csv')
        
        df.to_csv(dataset_name, index=False)
        
        print(f'Data saved to {dataset_name}')
import sys
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weinhardt2025.utils.model_evaluation import bayesian_information_criterion, log_likelihood
from spice.utils.plotting import plot_session
from weinhardt2025.utils.bandits import BanditsDrift
from spice.resources.rnn import RLRNN
from spice.utils.agent import Agent, get_update_dynamics

np.random.seed(42)
torch.manual_seed(42)

n_actions = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("EXPERIMENT CONFIG")
logger.info("=" * 80)
logger.info(f"Number of actions: {n_actions}")

data_path = 'data/parameter_recovery_participants/data_16p_0.csv'
df = pd.read_csv(data_path)

unique_participants = df['session'].unique()
n_participants = len(unique_participants)
logger.info(f"Number of participants: {n_participants}")

all_xs = []
all_ys = []

participant_index_to_id = {}
participant_params = {}

logger.info("=" * 80)
logger.info("PROCESSING PARTICIPANT DATA")
logger.info("=" * 80)

for i, participant_id in enumerate(unique_participants):
    participant_df = df[df['session'] == participant_id]
    n_trials = len(participant_df)
    
    alpha_reward = participant_df['alpha_reward'].iloc[0]
    alpha_penalty = participant_df['alpha_penalty'].iloc[0]
    beta_reward = participant_df['beta_reward'].iloc[0]
    beta_choice = participant_df['beta_choice'].iloc[0]
    forget_rate = participant_df['forget_rate'].iloc[0]
    
    participant_index_to_id[i] = participant_id
    
    participant_params[participant_id] = {
        'alpha_reward': alpha_reward,
        'alpha_penalty': alpha_penalty,
        'beta_reward': beta_reward,
        'beta_choice': beta_choice,
        'forget_rate': forget_rate
    }
    
    xs = torch.zeros((1, n_trials, 5))
    for t in range(1, n_trials):
        prev_choice = participant_df['choice'].iloc[t-1]
        xs[0, t, int(prev_choice)] = 1.0
        #  -1 for the unchosen reward column
        if int(prev_choice) == 0:
            xs[0, t, 2] = participant_df['reward'].iloc[t-1]
            xs[0, t, 3] = -1
        else:
            xs[0, t, 2] = -1
            xs[0, t, 3] = participant_df['reward'].iloc[t-1]
    xs[0, :, 4] = i
    
    ys = torch.zeros((1, n_trials, n_actions))
    for t in range(n_trials):
        choice = participant_df['choice'].iloc[t]
        ys[0, t, int(choice)] = 1.0
    
    all_xs.append(xs)
    all_ys.append(ys)


combined_xs_full = torch.cat(all_xs)
combined_ys_full = torch.cat(all_ys)
combined_dataset_full = DatasetRNN(combined_xs_full, combined_ys_full)


list_rnn_modules = [
    'x_learning_rate_reward',
    'x_value_reward_not_chosen',
    'x_value_choice_chosen',
    'x_value_choice_not_chosen'
]

list_control_parameters = ['c_action', 'c_reward', 'c_value_reward']

library_setup = {
    'x_learning_rate_reward': ['c_reward', 'c_value_reward'],
    'x_value_reward_not_chosen': [],
    'x_value_choice_chosen': [],
    'x_value_choice_not_chosen': [],
}

filter_setup = {
    'x_learning_rate_reward': ['c_action', 1, True],
    'x_value_reward_not_chosen': ['c_action', 0, True],
    'x_value_choice_chosen': ['c_action', 1, True],
    'x_value_choice_not_chosen': ['c_action', 0, True],
}


def visualize_random_participant(dataset, agent_rnn, agent_sindy, subset_length):
    """Create visualization plots for a specific participant using all key parameters."""
    participant_indices = torch.unique(dataset.xs[:, 0, -1]).tolist()
    
    selected_idx = 0
    
    participant_id = participant_index_to_id[selected_idx]
    
    mask = (dataset.xs[:, 0, -1] == selected_idx)
    xs_participant = dataset.xs[mask]
    
    xs_participant = xs_participant[0]
    
    logger.info(f"Visualizing for participant index {selected_idx}, ID {participant_id}")
    logger.info(f"Data shape for this participant: {xs_participant.shape}")
    
    params = participant_params[participant_id]
    alpha_reward = params['alpha_reward']
    alpha_penalty = params['alpha_penalty']
    beta_reward = params['beta_reward']
    beta_choice = params['beta_choice']  
    forget_rate = params['forget_rate']
    
    agent = AgentQ(
        n_actions=n_actions,
        alpha_reward=alpha_reward,
        alpha_penalty=alpha_penalty,
        beta_reward=beta_reward,
        beta_choice=beta_choice,
        forget_rate=forget_rate,
        alpha_choice = 1.0
    )
    
    agent.new_sess()
    agent_rnn.new_sess(participant_id=selected_idx)
    agent_sindy.new_sess(participant_id=selected_idx)
    
    agents_dict = {
        'groundtruth': agent,
        'rnn': agent_rnn,
        'sindy': agent_sindy
    }
    
    experiment = xs_participant.cpu().numpy()
    
    fig, axs = plot_session(
        agents=agents_dict,
        experiment=experiment
    )
    
    plt.suptitle(f'Participant ID: {participant_id} (#{selected_idx}) - Model Comparison\n' + 
                 f'α_reward: {alpha_reward:.3f}, α_penalty: {alpha_penalty:.3f}, ' + 
                 f'β_reward: {beta_reward:.3f}, β_choice: {beta_choice:.3f}, ' + 
                 f'forget_rate: {forget_rate:.3f}',
                 fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'updated_participant_{participant_id}_full_params_trial_{subset_length}.png', dpi=300)
    

def run_training_and_evaluation(dataset, label):
    logger.info(f"\n==== Running pipeline for dataset: {label} ====")
    logger.info("\nTraining RNN...")
    model_rnn = RLRNN(
        n_actions=n_actions, 
        n_participants=n_participants,
        list_signals=list_rnn_modules + list_control_parameters
    )
    rnn_params = sum(p.numel() for p in model_rnn.parameters() if p.requires_grad)
    logger.info(f"RNN model trainable parameters: {rnn_params}")
    
    optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=5e-3)
    
    model_rnn, optimizer_rnn, final_train_loss = fit_model(
        model=model_rnn,
        optimizer=optimizer_rnn,
        dataset_train=dataset,
        epochs = 128,
        n_steps=16,
        scheduler=True,
        convergence_threshold=0,
    )
    
    logger.info(f"Final training loss: {final_train_loss:.7f}")
    
    agent_rnn = Agent(model_rnn=model_rnn, n_actions=n_actions)
    
    logger.info("\nFitting SINDy...")
    agent_sindy, _ = fit_spice(
        rnn_modules=list_rnn_modules,
        control_signals=list_control_parameters,
        agent_rnn=agent_rnn,
        data_off_policy=dataset,
        off_policy=True, 
        polynomial_degree=2,
        library_setup=library_setup,
        filter_setup=filter_setup,
        optimizer_threshold=0.05,
        optimizer_alpha=1,
        verbose=True,
    )
    sindy_params = agent_sindy.count_parameters()
    
    logger.info("\nEvaluating SINDy models...")
    available_participant_ids = set()
    for module_name in list_rnn_modules:
        if hasattr(agent_sindy.model, 'submodules_sindy') and module_name in agent_sindy.model.submodules_sindy:
            available_participant_ids.update(agent_sindy.model.submodules_sindy[module_name].keys())
    logger.info(f"SINDy models are available for participants: {available_participant_ids}")
    
    bic_values = []
    ll_values = []  

    for pid in sorted(available_participant_ids):
        logger.info(f"Evaluating participant {pid} using its own SINDy model...")
        mask = (dataset.xs[:, 0, -1] == pid)
        xs_participant = dataset.xs[mask]
        ys_participant = dataset.ys[mask]
        xs_modified = xs_participant.clone()
        xs_modified[:, :, -1] = pid
        agent_sindy.new_sess(participant_id=pid)
        logger.info(f"SINDy model parameters for participant {pid}: {sindy_params[pid]}")
        values, probs, agent_sindy = get_update_dynamics(xs_participant[0], agent_sindy)
        values_q, values_reward, values_choice, learning_rates = values
        choices_np = xs_participant[..., :agent_sindy._n_actions].cpu().numpy().squeeze(0)
        ll = log_likelihood(data=choices_np, probs=probs)
        n_trials = choices_np.shape[0]
        normalized_ll = ll / n_trials
        ll_values.append(normalized_ll)
        bic = bayesian_information_criterion(data=choices_np, probs=probs, n_parameters=sindy_params[pid], ll=ll)
        normalized_bic = bic / n_trials
        logger.info(f"Participant {pid}: Log-likelihood: {ll:.4f}, Normalized LL: {normalized_ll:.4f}, Raw BIC: {bic:.4f}, Normalized BIC: {normalized_bic:.4f}")
        bic_values.append(normalized_bic)
    
    avg_bic = np.mean(bic_values)
    logger.info(f"\nAverage SINDy BIC for {label}: {avg_bic:.4f}")
    
    avg_ll = np.mean(ll_values)
    logger.info(f"\nAverage Normalized Log Likelihood for {label}: {avg_ll:.4f}")
    
    logger.info("\nIdentified SINDy equations:")
    for module_name in list_rnn_modules:
        logger.info(f"Module: {module_name}")
        for pid in sorted(available_participant_ids):
            if (hasattr(agent_sindy.model, 'submodules_sindy') and 
                module_name in agent_sindy.model.submodules_sindy and 
                pid in agent_sindy.model.submodules_sindy[module_name]):
                sindy_model = agent_sindy.model.submodules_sindy[module_name][pid]
                logger.info(f"  Participant {pid} Equation: {sindy_model}")
                coeffs = sindy_model.coefficients()
                if coeffs is not None and len(coeffs) > 0:
                    logger.info(f"  Participant {pid} Coefficients: {coeffs[0]}")
                    logger.info(f"  Feature names: {sindy_model.feature_names}")
    
    visualize_random_participant(dataset, agent_rnn, agent_sindy, int(label.split()[0]))
    
    return avg_bic, avg_ll, agent_rnn, agent_sindy

logger.info("=" * 80)
logger.info("RUNNING PIPELINE FOR DIFFERENT TRIAL SUBSET SIZES")
logger.info("=" * 80)

max_trials = combined_xs_full.shape[1]
trial_step = 100

bic_results = []
ll_results = [] 

for subset_length in range(trial_step, max_trials + 1, trial_step):
    logger.info(f"Creating dataset subset with {subset_length} trials per participant")
    subset_xs = []
    subset_ys = []
    for participant_id in range(n_participants):
        participant_mask = (combined_xs_full[:, 0, -1] == participant_id)
        participant_xs = combined_xs_full[participant_mask]
        participant_ys = combined_ys_full[participant_mask]
        trials_to_use = min(subset_length, participant_xs.shape[1])
        subset_participant_xs = participant_xs[:, :trials_to_use, :]
        subset_participant_ys = participant_ys[:, :trials_to_use, :]
        subset_xs.append(subset_participant_xs)
        subset_ys.append(subset_participant_ys)
    
    combined_xs_subset = torch.cat(subset_xs)
    combined_ys_subset = torch.cat(subset_ys)
    combined_dataset_subset = DatasetRNN(combined_xs_subset, combined_ys_subset)
    
    logger.info(f"Subset with {subset_length} trials per participant:")
    logger.info(f"  Dataset shape: {combined_dataset_subset.xs.shape}")
    
    label = f"{subset_length} trials per participant"
    print(f"\n{'='*40}\nStarting run: {label}")
    avg_bic, avg_ll, agent_rnn, agent_sindy = run_training_and_evaluation(combined_dataset_subset, label)
    bic_results.append((subset_length, avg_bic))
    ll_results.append((subset_length, avg_ll))

trial_lengths, avg_bic_values = zip(*bic_results)
plt.figure(figsize=(8, 5))
plt.plot(trial_lengths, avg_bic_values, marker='o')
plt.xlabel('Trials per Participant')
plt.ylabel('Average Normalized BIC')
plt.title('Average SINDy BIC vs. Trials per Participant')
plt.grid(True)
plt.savefig('sindy_normalized_bic_vs_trials.png')
plt.show()

trial_lengths, avg_ll_values = zip(*ll_results)
plt.figure(figsize=(8, 5))
plt.plot(trial_lengths, avg_ll_values, marker='o')
plt.xlabel('Trials per Participant')
plt.ylabel('Average Normalized Log Likelihood')
plt.title('Average Normalized Log Likelihood vs. Trials per Participant')
plt.grid(True)
plt.savefig('normalized_log_likelihood_vs_trials.png')
plt.show()
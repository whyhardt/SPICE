import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import os
import torch
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from utils.convert_dataset import convert_dataset
from resources.rnn_utils import DatasetRNN
from resources.model_evaluation import log_likelihood, bayesian_information_criterion
from resources.bandits import get_update_dynamics, AgentSpice
from resources.sindy_utils import load_spice, SindyConfig_dezfouli2019
from utils.setup_agents import setup_agent_rnn
from resources.rnn import RLRNN_dezfouli2019, ExtendedEmbedding

# ─── BEHAVIORAL METRICS ─────────────────────────────────────────────────────────────────────────────────
additional_inputs = ['diag']  # include diagnosis as an additional input
#or None if not diagnosis

data_path = 'data/dezfouli2019/dezfouli2019.csv' 

# (1) Read raw CSV and cast 'session' → int
dataset, _, original_df, _ = convert_dataset(file=data_path, additional_inputs=additional_inputs)
#original_df['session'] = original_df['session'].astype(int)

# Get a list of unique session IDs from original_df
unique_sessions = original_df['session'].unique().tolist()

behavior_metrics = []
for pid in tqdm(unique_sessions, desc="Computing behavior metrics"):
    participant_df = original_df[original_df['session'] == pid]
    if participant_df.empty:
        continue

    choices = participant_df['choice'].values
    rewards = participant_df['reward'].values

    # — Stay after reward rate
    stay_after_reward_count = 0
    stay_after_reward_total = 0
    for i in range(len(choices) - 1):
        if rewards[i] > 0:
            stay_after_reward_total += 1
            if choices[i + 1] == choices[i]:
                stay_after_reward_count += 1
    stay_after_reward_rate = (
        stay_after_reward_count / stay_after_reward_total
        if stay_after_reward_total > 0
        else 0
    )

    # — Overall switch rate
    switch_count = 0
    switch_total = 0
    for i in range(len(choices) - 1):
        switch_total += 1
        if choices[i + 1] != choices[i]:
            switch_count += 1
    switch_rate = switch_count / switch_total if switch_total > 0 else 0

    # — Perseveration: stay after 3 consecutive unrewarded trials
    perseveration_count = 0
    perseveration_total = 0
    for i in range(3, len(choices)):
        prev_3_rewards = rewards[i - 3 : i]
        prev_3_choices = choices[i - 3 : i]
        if (
            np.all(prev_3_rewards == 0)
            and np.all(prev_3_choices == prev_3_choices[0])
            and len(np.unique(prev_3_choices)) == 1
        ):
            perseveration_total += 1
            if choices[i] == prev_3_choices[0]:
                perseveration_count += 1
    perseveration = (
        perseveration_count / perseveration_total if perseveration_total > 0 else 0
    )

    # — Stay based on last two outcomes ( ++, +−, −+, −− )
    stay_pp = stay_pm = stay_mp = stay_mm = 0
    total_pp = total_pm = total_mp = total_mm = 0
    for i in range(2, len(choices) - 1):
        prev1 = rewards[i - 2]
        prev2 = rewards[i - 1]
        curr = choices[i]
        nxt = choices[i + 1]

        if prev1 > 0 and prev2 > 0:
            total_pp += 1
            if nxt == curr:
                stay_pp += 1
        elif prev1 > 0 and prev2 == 0:
            total_pm += 1
            if nxt == curr:
                stay_pm += 1
        elif prev1 == 0 and prev2 > 0:
            total_mp += 1
            if nxt == curr:
                stay_mp += 1
        elif prev1 == 0 and prev2 == 0:
            total_mm += 1
            if nxt == curr:
                stay_mm += 1

    stay_pp_rate = stay_pp / total_pp if total_pp > 0 else np.nan
    stay_pm_rate = stay_pm / total_pm if total_pm > 0 else np.nan
    stay_mp_rate = stay_mp / total_mp if total_mp > 0 else np.nan
    stay_mm_rate = stay_mm / total_mm if total_mm > 0 else np.nan

    # — Average reward (removed avg_rt)
    avg_reward = participant_df['reward'].mean()

    diagnosis = participant_df['diag'].iloc[0] if 'diag' in participant_df.columns else np.nan

    participant_data = {
        'participant_id': pid,
        'stay_after_reward': stay_after_reward_rate,
        'switch_rate': switch_rate,
        'perseveration': perseveration,
        'stay_after_plus_plus': stay_pp_rate,
        'stay_after_plus_minus': stay_pm_rate,
        'stay_after_minus_plus': stay_mp_rate,
        'stay_after_minus_minus': stay_mm_rate,
        'avg_reward': avg_reward,
        'n_trials': len(participant_df),
        'Diagnosis': diagnosis
    }

    behavior_metrics.append(participant_data)

behavior_df = pd.DataFrame(behavior_metrics)
print(f"Behavioral metrics computed for {len(behavior_df)} participants.")


# ─── SINDy AND RNN MODELS ──────────────────────────────────────────────────────────────────────────────

model_rnn_path = '/Users/martynaplomecka/closedloop_rl/data/dezfouli2019/rnn_dezfouli2019_no_l1_l2_0.pkl'
model_spice_path = '/Users/martynaplomecka/closedloop_rl/data/dezfouli2019/spice_dezfouli2019_no_l1_l2_0.pkl'

#class_rnn = RLRNN_meta_dezfouli2019
class_rnn = RLRNN_dezfouli2019

sindy_config = SindyConfig_dezfouli2019


agent_rnn = setup_agent_rnn(
    path_model=model_rnn_path,
    class_rnn=class_rnn,
    list_sindy_signals=sindy_config["rnn_modules"] + sindy_config['control_parameters']
)

spice_modules = load_spice(file=model_spice_path)
agent_spice = AgentSpice(
    model_rnn=agent_rnn.model,
    sindy_modules=spice_modules,
    n_actions=agent_rnn._n_actions
)

list_rnn_modules = SindyConfig_dezfouli2019["rnn_modules"]

# Build mappings between "real session ID" and SPICE's internal index 0..(N-1)
session_to_index = {pid: i for i, pid in enumerate(unique_sessions)}
index_to_session = {i: pid for i, pid in enumerate(unique_sessions)}

n_participants = len(unique_sessions)
# Now `participant_ids` is the list of real session IDs, not 0..n_participants-1
participant_ids = unique_sessions

print(f"Total unique sessions: {len(unique_sessions)}")
print(f"SPICE knows about {n_participants} participants")
print(f"Mapped {len(index_to_session)} indices to actual PIDs")

# Collect all SINDy feature names
all_feature_names = set()
for module in list_rnn_modules:
    for idx_internal in agent_spice.model.submodules_sindy[module]:
        sindy_model = agent_spice.model.submodules_sindy[module][idx_internal]
        for name in sindy_model.get_feature_names():
            all_feature_names.add(f"{module}_{name}")

# Extract embedding matrix from the SPICE model
if isinstance(agent_spice.model.participant_embedding, torch.nn.Embedding):
    embedding_matrix = agent_spice.model.participant_embedding.weight.detach().cpu().numpy()
elif isinstance(agent_spice.model.participant_embedding, ExtendedEmbedding):
    embedding_matrix = agent_spice.model.participant_embedding.embedding.weight.detach().cpu().numpy()
else:
    raise RuntimeError("Unknown embedding type in SPICE model.")

embedding_size = embedding_matrix.shape[1]

# ─── PRECOMPUTE BETAS (fixed) ─────────────────────────────────────────────────────────────────────────

features = {'beta_reward': {}, 'beta_choice': {}}

for index_pid, pid in enumerate(participant_ids):
    # Filter original_df on the real session ID (pid)
    subset = original_df[original_df["session"] == pid]
    if subset.empty:
        print(f"Warning: no rows in original_df where session == {pid}")
        continue

    internal_idx = session_to_index[pid]
    mask_pid = dataset.xs[:, 0, -1] == index_pid
    additional_embedding_inputs = dataset.xs[mask_pid, 0, 2*agent_spice._n_actions:-3]
    # Initialize SPICE for this internal index with the participant's diagnosis
    agent_spice.new_sess(participant_id=internal_idx, additional_embedding_inputs=additional_embedding_inputs)
    betas = agent_spice.get_betas()
    features['beta_reward'][internal_idx] = betas.get('x_value_reward', 0.0)
    features['beta_choice'][internal_idx] = betas.get('x_value_choice', 0.0)

# Build the SINDy‐params DataFrame
sindy_params = []
for internal_idx in tqdm(range(n_participants), desc="Extracting SINDy/RNN params"):
    pid = index_to_session[internal_idx]
    param_dict = {'participant_id': pid}

    # Initialize all feature columns to 0.0
    for feat in all_feature_names:
        param_dict[feat] = 0.0

    # Insert beta_reward and beta_choice (if they exist)
    param_dict['beta_reward'] = features['beta_reward'].get(internal_idx, 0.0)
    param_dict['beta_choice'] = features['beta_choice'].get(internal_idx, 0.0)

    # Fill in each submodule's coefficients
    for module in list_rnn_modules:
        if internal_idx in agent_spice.model.submodules_sindy[module]:
            model = agent_spice.model.submodules_sindy[module][internal_idx]
            coefs = model.model.steps[-1][1].coef_.flatten()
            for i, name in enumerate(model.get_feature_names()):
                param_dict[f"{module}_{name}"] = coefs[i]
            param_dict[f"params_{module}"] = np.sum(np.abs(coefs) > 1e-10)
        else:
            param_dict[f"params_{module}"] = 0

    # Total nonzero SINDy parameters across all submodules
    param_dict['total_params'] = sum(param_dict[f"params_{m}"] for m in list_rnn_modules)

    # Append embedding entries
    if internal_idx < embedding_matrix.shape[0]:
        for j in range(embedding_size):
            param_dict[f'embedding_{j}'] = embedding_matrix[internal_idx, j]
    else:
        for j in range(embedding_size):
            param_dict[f'embedding_{j}'] = np.nan

    sindy_params.append(param_dict)

sindy_df = pd.DataFrame(sindy_params)
print(f"Number of participants in SINDY DataFrame: {len(sindy_df)}")
print(f"Number of participants in behavior DataFrame: {len(behavior_df)}")

# ─── MERGE 1: ONLY KEEP INTERSECTION OF SINDY & BEHAVIOR ───────────────────────────────────────────────

merged_df = pd.merge(sindy_df, behavior_df, on='participant_id', how='inner')
print(f"After inner‐join of SINDy + behavior: {len(merged_df)} participants")

# ─── SECTION 3: CALCULATE MODEL EVALUATION METRICS ─────────────────────────────────────────────────────

# Reload or reuse the same dataset object for testing
dataset_test, _, _, _ = convert_dataset(file=data_path, additional_inputs=additional_inputs)

# Collect metrics for each session, then average per participant
session_metrics = []

for internal_idx in tqdm(range(n_participants), desc="Computing model metrics"):
    pid = index_to_session[internal_idx]
    
    if pid not in set(merged_df['participant_id']):
        continue
    
    # Find all sessions for this participant
    participant_sessions = dataset_test.xs[dataset_test.xs[:, 0, -1] == internal_idx, 0, -3].unique()
    
    for sid in participant_sessions:
        # Mask for selecting trials belonging to this participant and session
        mask = torch.logical_and(
            dataset_test.xs[:, 0, -1] == internal_idx, 
            dataset_test.xs[:, 0, -3] == sid
        )
        
        if not mask.any():
            continue

        participant_data = DatasetRNN(*dataset_test[mask])

        # Reset agents before computing predictions
        # Get additional embedding inputs for this participant
        mask_pid = dataset.xs[:, 0, -1] == internal_idx
        additional_embedding_inputs = dataset.xs[mask_pid, 0, 2*agent_spice._n_actions:-3]
        
        agent_spice.new_sess(participant_id=internal_idx, additional_embedding_inputs=additional_embedding_inputs)
        agent_rnn.new_sess(participant_id=internal_idx)

        # Get predicted probabilities from both models
        _, probs_spice, _ = get_update_dynamics(
            experiment=participant_data.xs, agent=agent_spice
        )
        _, probs_rnn, _ = get_update_dynamics(
            experiment=participant_data.xs, agent=agent_rnn
        )

        n_trials_test = len(probs_spice)
        if n_trials_test == 0:
            continue

        true_actions = participant_data.ys[0, :n_trials_test].cpu().numpy()
        ll_spice = log_likelihood(data=true_actions, probs=probs_spice)
        ll_rnn = log_likelihood(data=true_actions, probs=probs_rnn)

        spice_per_trial_like = np.exp(ll_spice / (n_trials_test * agent_rnn._n_actions))
        rnn_per_trial_like = np.exp(ll_rnn / (n_trials_test * agent_rnn._n_actions))

        n_params_dict = agent_spice.count_parameters()
        if internal_idx not in n_params_dict:
            continue
        n_parameters_spice = n_params_dict[internal_idx]

        bic_spice = bayesian_information_criterion(
            data=true_actions,
            probs=probs_spice,
            n_parameters=n_parameters_spice
        )
        aic_spice = 2 * n_parameters_spice - 2 * ll_spice

        session_metrics.append({
            'participant_id': pid,
            'session_id': sid.item(),  # Convert tensor to scalar
            'nll_spice': -ll_spice,
            'nll_rnn': -ll_rnn,
            'trial_likelihood_spice': spice_per_trial_like,
            'trial_likelihood_rnn': rnn_per_trial_like,
            'bic_spice': bic_spice,
            'aic_spice': aic_spice,
            'n_parameters_spice': n_parameters_spice,
            'metric_n_trials': n_trials_test
        })

session_metrics_df = pd.DataFrame(session_metrics)

# Now average metrics per participant
metrics_data = []
for pid in session_metrics_df['participant_id'].unique():
    participant_sessions = session_metrics_df[session_metrics_df['participant_id'] == pid]
    
    # Calculate weighted averages (weighted by number of trials in each session)
    total_trials = participant_sessions['metric_n_trials'].sum()
    weights = participant_sessions['metric_n_trials'] / total_trials
    
    # For likelihood-based metrics, we want to sum log-likelihoods and then compute final metrics
    total_nll_spice = participant_sessions['nll_spice'].sum()
    total_nll_rnn = participant_sessions['nll_rnn'].sum()
    
    # Weighted averages for per-trial metrics
    avg_trial_likelihood_spice = (participant_sessions['trial_likelihood_spice'] * weights).sum()
    avg_trial_likelihood_rnn = (participant_sessions['trial_likelihood_rnn'] * weights).sum()
    
    # For BIC/AIC, sum the log-likelihoods and use total trials
    total_ll_spice = -total_nll_spice
    avg_bic_spice = -2 * total_ll_spice + participant_sessions['n_parameters_spice'].iloc[0] * np.log(total_trials)
    avg_aic_spice = 2 * participant_sessions['n_parameters_spice'].iloc[0] - 2 * total_ll_spice
    
    metrics_data.append({
        'participant_id': pid,
        'nll_spice': total_nll_spice,
        'nll_rnn': total_nll_rnn,
        'trial_likelihood_spice': avg_trial_likelihood_spice,
        'trial_likelihood_rnn': avg_trial_likelihood_rnn,
        'bic_spice': avg_bic_spice,
        'aic_spice': avg_aic_spice,
        'n_parameters_spice': participant_sessions['n_parameters_spice'].iloc[0],  # Same for all sessions
        'metric_n_trials': total_trials,
        'n_sessions': len(participant_sessions)
    })

metrics_df = pd.DataFrame(metrics_data)
print(f"Number of participants with model metrics: {len(metrics_df)}")
print(f"Average number of sessions per participant: {session_metrics_df.groupby('participant_id').size().mean():.2f}")

# ─── MERGE 2: KEEP ONLY PARTICIPANTS WHO ALSO HAVE METRICS ───────────────────────────────────────────────
final_df = pd.merge(merged_df, metrics_df, on='participant_id', how='inner')
print(f"After inner‐join with metrics: {len(final_df)} participants")


final_df.to_csv('dezfouli_final_df_sindy_analysis_with_metrics.csv', index=False)
behavior_df.to_csv('dezfouli_behavior_metrics_fixed.csv', index=False)
sindy_df.to_csv('dezfouli_sindy_parameters.csv', index=False)
metrics_df.to_csv('dezfouli_model_evaluation_metrics.csv', index=False)
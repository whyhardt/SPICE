import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
import os
import torch
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from spice import csv_to_dataset, SpiceDataset, SpiceEstimator
from weinhardt2025.utils.model_evaluation import log_likelihood, bayesian_information_criterion
from spice.utils.agent import get_update_dynamics
from spice.precoded import workingmemory_rewardbinary

# ─── BEHAVIORAL METRICS ─────────────────────────────────────────────────────────────────────────────────
additional_inputs = None#['age']  # If you want to include age as an additional input
#or None if not age

data_path = 'weinhardt2025/data/eckstein2022/eckstein2022.csv'
#data_path = 'data/eckstein2022/eckstein2022.csv'

slcn_path = 'weinhardt2025/data/eckstein2022/SLCN.csv'
# (1a) Read raw CSV and cast 'participant' → int
dataset = csv_to_dataset(file=data_path, additional_inputs=additional_inputs)
original_df = pd.read_csv(data_path)
original_df['participant'] = original_df['participant'].astype(int)

n_actions = dataset.ys.shape[-1]
n_participants = len(dataset.xs[..., -1].unique())

# (1b) Read SLCN metadata and cast 'ID' → int
slcn_df = pd.read_csv(slcn_path)
slcn_df['ID'] = slcn_df['ID'].astype(int)

# Keep only the columns we care about from SLCN
columns_to_keep = ['ID', 'age - years', 'age_maria', 'Category']
available_columns = [col for col in columns_to_keep if col in slcn_df.columns]
slcn_df = slcn_df[available_columns]

# Build a mapping { participant_id → { 'ID', 'age - years', 'Category' } }
slcn_mapping = {}
for _, row in slcn_df.iterrows():
    slcn_mapping[row['ID']] = row.to_dict()

# Get a list of unique session IDs from original_df
unique_sessions = original_df['participant'].unique().tolist()

behavior_metrics = []
for pid in tqdm(unique_sessions, desc="Computing behavior metrics"):
    if pid not in slcn_mapping:
        continue

    participant_df = original_df[original_df['participant'] == pid]
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

    # — Average reward and reaction time
    avg_reward = participant_df['reward'].mean()
    avg_rt = participant_df['rt'].mean()

    sl = slcn_mapping[pid]
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
        'avg_rt': avg_rt,
        'n_trials': len(participant_df),
        'Age': sl.get('age - years', np.nan),
        'Age_Category': sl.get('Category', np.nan),
        'age_maria': sl.get('age_maria', np.nan) 

    }

    behavior_metrics.append(participant_data)

behavior_df = pd.DataFrame(behavior_metrics)
print(f"Behavioral metrics computed for {len(behavior_df)} participants.")


# ─── SINDy AND RNN MODELS ──────────────────────────────────────────────────────────────────────────────

model_spice_path = 'weinhardt2025/params/eckstein2022/spice_eckstein2022.pkl'

class_rnn = workingmemory_rewardbinary.SpiceModel

sindy_config = workingmemory_rewardbinary.CONFIG

estimator = SpiceEstimator(
    rnn_class=class_rnn,
    spice_config=sindy_config,
    n_actions=n_actions,
    n_participants=n_participants,
    sindy_library_polynomial_degree=2,
    use_sindy=True,
)

estimator.load_spice(model_spice_path)

agent_rnn = estimator.rnn_agent
agent_spice = estimator.spice_agent

list_rnn_modules = estimator.get_modules()

# Build mappings between “real session ID” and SPICE’s internal index 0..(N-1)
session_to_index = {pid: i for i, pid in enumerate(unique_sessions)}
index_to_session = {i: pid for i, pid in enumerate(unique_sessions)}

# Now `participant_ids` is the list of real session IDs, not 0..n_participants-1
participant_ids = unique_sessions

print(f"Total unique sessions: {len(unique_sessions)}")
print(f"SPICE knows about {n_participants} participants")
print(f"Mapped {len(index_to_session)} indices to actual PIDs")

# Collect all SINDy feature names
all_feature_names = set()
for module in list_rnn_modules:
    for name in agent_spice.model.sindy_candidate_terms[module]:
        all_feature_names.add(f"{module}_{name}")

# Extract embedding matrix from the SPICE model
embedding_matrix = agent_spice.model.participant_embedding[0].weight.detach().cpu().numpy()
embedding_size = embedding_matrix.shape[1]


# Build the SINDy‐params DataFrame
sindy_params = []
for internal_idx in tqdm(range(n_participants), desc="Extracting SINDy/RNN params"):
    pid = index_to_session[internal_idx]
    param_dict = {'participant_id': pid}

    # Initialize all feature columns to 0.0
    for feat in all_feature_names:
        param_dict[feat] = 0.0

    # Fill in each submodule’s coefficients
    for module in list_rnn_modules:
        if internal_idx in range(agent_spice.model.sindy_coefficients[module].shape[1]):
            coefs = agent_spice.model.sindy_coefficients[module][0, internal_idx].flatten().detach().cpu().numpy()
            for i, name in enumerate(agent_spice.model.sindy_candidate_terms[module]):
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
# dataset_test = convert_dataset(file=data_path, additional_inputs=additional_inputs)
dataset_test = dataset  # ensure we use the original dataset

metrics_data = []
for internal_idx in tqdm(range(n_participants), desc="Computing model metrics"):
    pid = index_to_session[internal_idx]

    # Only compute metrics if this pid is in the merged_df
    if pid not in set(merged_df['participant_id']):
        continue

    # Mask for selecting trials belonging to this SPICE index
    mask = (dataset_test.xs[:, 0, -1] == internal_idx)
    if not mask.any():
        continue

    participant_data = SpiceDataset(*dataset_test[mask])

    # Reset agents before computing predictions
    agent_spice.new_sess(participant_id=internal_idx)
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

    spice_per_trial_like = np.exp(ll_spice / (n_trials_test * agent_rnn.n_actions))
    rnn_per_trial_like = np.exp(ll_rnn / (n_trials_test * agent_rnn.n_actions))

    n_params_dict = agent_spice.count_parameters()
    n_parameters_spice = int(n_params_dict[internal_idx])

    bic_spice = bayesian_information_criterion(
        data=true_actions,
        probs=probs_spice,
        n_parameters=n_parameters_spice
    )
    aic_spice = 2 * n_parameters_spice - 2 * ll_spice

    metrics_data.append({
        'participant_id': pid,
        'nll_spice': -ll_spice,
        'nll_rnn': -ll_rnn,
        'trial_likelihood_spice': spice_per_trial_like,
        'trial_likelihood_rnn': rnn_per_trial_like,
        'bic_spice': bic_spice,
        'aic_spice': aic_spice,
        'n_parameters_spice': n_parameters_spice,
        'metric_n_trials': n_trials_test
    })

metrics_df = pd.DataFrame(metrics_data)
print(f"Number of participants with model metrics: {len(metrics_df)}")

# ─── MERGE 2: KEEP ONLY PARTICIPANTS WHO ALSO HAVE METRICS ───────────────────────────────────────────────
final_df = pd.merge(merged_df, metrics_df, on='participant_id', how='inner')
print(f"After inner‐join with metrics: {len(final_df)} participants")

results_path = 'weinhardt2025/analysis/participants_analysis_eckstein2022/results'

final_df.to_csv(os.path.join(results_path, 'final_df_sindy_analysis_with_metrics.csv'), index=False)
behavior_df.to_csv(os.path.join(results_path, 'behavior_metrics_fixed.csv'), index=False)
sindy_df.to_csv(os.path.join(results_path, 'sindy_parameters.csv'), index=False)
metrics_df.to_csv(os.path.join(results_path, 'model_evaluation_metrics.csv'), index=False)

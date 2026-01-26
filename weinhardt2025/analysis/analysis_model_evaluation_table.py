import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy

from weinhardt2025.benchmarking import benchmarking_gru

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# standard methods and classes used for every model evaluation
from benchmarking import benchmarking_dezfouli2019
from weinhardt2025.utils.model_evaluation import get_scores
from spice.utils.agent import get_update_dynamics, Agent
from spice.utils.agent import setup_agent_rnn, setup_agent_spice
from spice.utils.convert_dataset import csv_to_dataset, split_data_along_timedim, split_data_along_sessiondim

# dataset specific SPICE models
from spice.resources import rnn, sindy_utils

# dataset specific benchmarking models
from benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022
from benchmarking.benchmarking_dezfouli2019 import Dezfouli2019GQL

l2_values = ['0.0001']#['0', '0.00001', '0.00005', '0.0001', '0.0005', '0.001']

# -------------------------------------------------------------------------------
# AGENT CONFIGURATIONS
# -------------------------------------------------------------------------------

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
# study = 'eckstein2022'
# train_test_ratio = 0.8
# sindy_config = sindy_utils.SindyConfig_eckstein2022
# rnn_class = rnn.RLRNN_eckstein2022
# additional_inputs = None
# setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
# rl_model = benchmarking_eckstein2022.rl_model
# model_config_baseline = 'ApBr'
# model_config_benchmark = 'ApAnBrBcfBch'
# benchmark_file = f'mcmc_eckstein2022_ApAnBrBcfBch.nc'
# baseline_file = f'mcmc_eckstein2022_ApBr.nc'

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
study = 'dezfouli2019'
train_test_ratio = [3, 6, 9]
sindy_config = sindy_utils.SindyConfig_eckstein2022
rnn_class = rnn.RLRNN_eckstein2022
additional_inputs = None
setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
gql_model = benchmarking_dezfouli2019.Dezfouli2019GQL
model_config_baseline = 'PhiBeta'
model_config_benchmark = 'PhiChiBetaKappaC'
benchmark_file = f'gql_dezfouli2019_PhiChiBetaKappaC.pkl'
baseline_file = f'gql_dezfouli2019_PhiBeta.pkl'

# ------------------------- CONFIGURATION FILE PATHS ------------------------
path_data = f'data/{study}/{study}.csv'
path_model_baseline = None#os.path.join(f'params/{study}/', baseline_file)
path_model_benchmark = None#os.path.join(f'params/{study}', benchmark_file)
path_model_benchmark_lstm = None#f'params/{study}/lstm_{study}.pkl'

path_model_rnn = f'params/{study}/rnn_{study}_l2_L2VALUE_v2_ep4096.pkl'
path_model_spice = f'params/{study}/spice_{study}_l2_L2VALUE_v2_ep4096.pkl'

# -------------------------------------------------------------------------------
# MODEL COMPARISON PIPELINE
# -------------------------------------------------------------------------------

dataset = csv_to_dataset(path_data, additional_inputs=additional_inputs)
participant_ids = dataset.xs[:, 0, -1].unique().cpu().numpy()

# ------------------------------------------------------------
# Setup of agents
# ------------------------------------------------------------

# setup baseline model
if path_model_baseline:
    print("Setting up baseline agent from file", path_model_baseline)
    agent_baseline = setup_agent_benchmark(path_model=path_model_baseline, model_config=model_config_baseline)
else:
    print("Setting up dummy baseline agent")
    agent_baseline = [[AgentQ(alpha_reward=0.2, beta_reward=1, beta_choice=3) for _ in range(len(dataset))], 2]
n_parameters_baseline = 2

# setup benchmark model
if path_model_benchmark:
    print("Setting up benchmark agent from file", path_model_benchmark)
    agent_benchmark = setup_agent_benchmark(path_model=path_model_benchmark, model_config=model_config_benchmark)
    n_parameters_benchmark = agent_benchmark[1]
else:
    n_parameters_benchmark = 0

if path_model_benchmark_lstm:
    print("Setting up LSTM agent from file", path_model_benchmark_lstm)
    agent_lstm = benchmarking_gru.setup_agent_gru(path_model=path_model_benchmark_lstm)
    n_parameters_lstm = sum(p.numel() for p in agent_lstm.model.parameters() if p.requires_grad)
else:
    n_parameters_lstm = 0
    
# setup rnn agent
if path_model_rnn is not None:
    agent_rnn = {}
    for value in l2_values:
        current_rnn = path_model_rnn.replace('L2VALUE', value.replace('.', '_'))
        print("Setting up RNN agent from file", current_rnn)
        agent_rnn[value] = setup_agent_rnn(
            class_rnn=rnn_class,
            path_model=current_rnn,
            )
    n_parameters_rnn = sum(p.numel() for p in agent_rnn[value].model.parameters() if p.requires_grad)
else:
    n_parameters_rnn = 0
    
# setup spice agent
if path_model_spice is not None:
    agent_spice = {}
    for value in l2_values:
        current_rnn = path_model_rnn.replace('L2VALUE', value.replace('.', '_'))
        current_spice = path_model_spice.replace('L2VALUE', value.replace('.', '_'))
        print("Setting up SPICE agent from file", current_spice)
        agent_spice[value] = setup_agent_spice(
            class_rnn=rnn_class,
            path_rnn=current_rnn,
            path_spice=current_spice,
        )
n_parameters_spice = 0

# ------------------------------------------------------------
# Dataset splitting
# ------------------------------------------------------------

# split data into train and test data according to train_test_ratio
if isinstance(train_test_ratio, float):
    dataset_train, dataset_test = split_data_along_timedim(dataset, split_ratio=train_test_ratio)
    
elif isinstance(train_test_ratio, list) or isinstance(train_test_ratio, tuple):
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, list_test_sessions=train_test_ratio)
    
else:
    raise TypeError("train_test_raio must be either a float number or a list of integers containing the session/block ids which should be used as test sessions/blocks")

# ------------------------------------------------------------
# Computation of metrics
# ------------------------------------------------------------

print('Running model evaluation...')
table_values_raw = np.zeros((3 + 3*len(l2_values), 7, len(dataset_test)))
considered_trials = np.zeros((table_values_raw.shape[0], 1, len(dataset_test)))

failed_attempts = 0

# running on test data for every model except for RNN on training data

for index_data in tqdm(range(len(dataset_test))):
    try:
        # use whole session to include warm-up phase; make sure to exclude warm-up phase when computing metrics
        pid = dataset_test.xs[index_data, 0, -1].int().item()
        
        if not pid in participant_ids:
            print(f"Skipping participant {index_data} because they could not be found in the SPICE participants. Probably due to prior filtering of badly fitted participants.")
            continue
        
        data_input = dataset.xs if isinstance(train_test_ratio, float) else dataset_test.xs
        
        # Baseline model
        probs_baseline = get_update_dynamics(experiment=data_input[index_data], agent=agent_baseline[0][pid])[1]
        
        # get number of actual trials
        n_trials = len(probs_baseline)
        
        if isinstance(train_test_ratio, float):
            data_ys = dataset.xs[index_data, :n_trials, :agent_baseline[0][0].n_actions].cpu().numpy()
        else:
            data_ys = dataset_test.xs[index_data, :n_trials, :agent_baseline[0][0].n_actions].cpu().numpy()
            
        n_trials_test = int(n_trials * (1-train_test_ratio)) if isinstance(train_test_ratio, float) else n_trials
        index_start = n_trials - n_trials_test
        index_end = n_trials
        
        scores_baseline = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_baseline[index_start:index_end], n_parameters=agent_baseline[1]))
        table_values_raw[0, -3:, index_data] = scores_baseline
        table_values_raw[0, 0, index_data] = n_parameters_baseline
        considered_trials[0, 0, index_data] += index_end - index_start
        
        # get scores of all mcmc benchmark models but keep only the best one for each session
        if path_model_benchmark:
            n_parameters_model = agent_benchmark[1]
            probs_benchmark = get_update_dynamics(experiment=data_input[index_data], agent=agent_benchmark[0][pid])[1]
            scores_benchmark = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_benchmark[index_start:index_end], n_parameters=n_parameters_model))
            table_values_raw[1, -3:, index_data] = scores_benchmark
            table_values_raw[1, 0, index_data] = n_parameters_benchmark
            considered_trials[1, 0, index_data] += index_end - index_start

        # Benchmark LSTM
        if path_model_benchmark_lstm:
            probs_lstm = get_update_dynamics(experiment=data_input[index_data], agent=agent_lstm)[1]
            scores_lstm = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_lstm[index_start:index_end], n_parameters=n_parameters_lstm))
            table_values_raw[2, -3:, index_data] = scores_lstm
            table_values_raw[2, 0, index_data] = n_parameters_lstm
            considered_trials[2, 0, index_data] += index_end - index_start
            
        for index_value, value in enumerate(l2_values):
            
            # SPICE-RNN
            if path_model_rnn is not None:
                probs_rnn = get_update_dynamics(experiment=data_input[index_data], agent=agent_rnn[value])[1]
                    
                if isinstance(train_test_ratio, float):
                    scores_rnn = np.array(get_scores(data=data_ys[:index_start], probs=probs_rnn[:index_start], n_parameters=n_parameters_rnn))
                    table_values_raw[3+3*index_value, -3:, index_data] = scores_rnn
                    table_values_raw[3+3*index_value, 0, index_data] = n_parameters_rnn
                    considered_trials[3+3*index_value, 0, index_data] += index_start
                
                scores_rnn = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_rnn[index_start:index_end], n_parameters=n_parameters_rnn))
                table_values_raw[3+3*index_value+1, -3:, index_data] = scores_rnn
                table_values_raw[3+3*index_value+1, 0, index_data] = n_parameters_rnn
                considered_trials[3+3*index_value+1, 0, index_data] += index_end - index_start
                
            # SPICE
            if path_model_spice is not None:
                additional_inputs_embedding = data_input[0, agent_spice[value].n_actions*2:-3]
                agent_spice[value].new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
                n_params_spice = agent_spice[value].count_parameters()[pid]
                
                probs_spice = get_update_dynamics(experiment=data_input[index_data], agent=agent_spice[value])[1]
                scores_spice = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_spice[index_start:index_end], n_parameters=n_params_spice))
                table_values_raw[3+3*index_value+2, -3:, index_data] = scores_spice
                table_values_raw[3+3*index_value+2, 0, index_data] = n_params_spice
                considered_trials[3+3*index_value+2, 0, index_data] += index_end - index_start
        
    except Exception as e:  
        print(e)
        failed_attempts += 1

# running on training data only for RNN (if test data are a different set of sessions than training data)

table_values_raw_training = np.zeros((3 + 3*len(l2_values), 7, len(dataset_train)))

failed_attempts_training = 0
considered_trials_training = 0

# metric_participant = np.zeros((len(table_values), len(dataset_test)))
# spice_parameters_participant = np.zeros((len(l2_values), len(dataset_test)))
considered_trials_participant_training = np.zeros(len(dataset_train))

if not isinstance(train_test_ratio, float) and path_model_rnn is not None:
    for index_data in tqdm(range(len(dataset_train))):
        try:
            # use whole session to include warm-up phase; make sure to exclude warm-up phase when computing metrics
            pid = dataset_train.xs[index_data, 0, -1].int().item()
            
            if not pid in participant_ids:
                print(f"Skipping participant {index_data} because they could not be found in the SPICE participants. Probably due to prior filtering of badly fitted participants.")
                continue
            
            data_input = dataset.xs if isinstance(train_test_ratio, float) else dataset_train.xs
            
            # Baseline model
            probs_baseline = get_update_dynamics(experiment=data_input[index_data], agent=agent_baseline[0][pid])[1]
            
            # get number of actual trials
            n_trials = len(probs_baseline)
            
            if isinstance(train_test_ratio, float):
                data_ys = dataset.xs[index_data, :n_trials, :agent_baseline[0][0].n_actions].cpu().numpy()
            else:
                data_ys = dataset_train.xs[index_data, :n_trials, :agent_baseline[0][0].n_actions].cpu().numpy()
            
            n_trials_test = int(n_trials * train_test_ratio) if isinstance(train_test_ratio, float) else 0
            index_end = n_trials - n_trials_test
            
            for index_value, value in enumerate(l2_values):
                
                # SPICE-RNN
                if path_model_rnn is not None:
                    # on training data
                    probs_rnn = get_update_dynamics(experiment=data_input[index_data], agent=agent_rnn[value])[1]
                    scores_rnn = np.array(get_scores(data=data_ys[:index_end], probs=probs_rnn[:index_end], n_parameters=n_parameters_rnn))
                    # metric_participant[3+index_value*3, index_data] = scores_rnn[0]
                    table_values_raw_training[3+3*index_value, -3:, index_data] = scores_rnn
                    table_values_raw_training[3+3*index_value, 0, index_data] = n_parameters_rnn
            
            considered_trials_participant_training[index_data] += index_end
            considered_trials_training += index_end
            
        except Exception as e:  
            print(e)
            failed_attempts_training += 1

# ------------------------------------------------------------
# Post processing
# ------------------------------------------------------------

table_values = np.zeros(table_values_raw.shape[:-1])

# post-processing of runs of test data

mask_non_zero = table_values_raw[:, -3, 0] != 0

# compute averaged scores and replace with summed scores in table_values (table_values[..., -3:])
table_values[mask_non_zero, -3:] += table_values_raw[mask_non_zero, -3:].sum(axis=-1) / considered_trials[mask_non_zero, :1].sum(axis=-1)
# compute average trial likelihood and std from NLL
table_values[mask_non_zero, 2] += np.exp(-table_values_raw[mask_non_zero, -3].sum(axis=-1) / considered_trials[mask_non_zero, 0].sum(axis=-1))
table_values[mask_non_zero, 3] += np.exp(-table_values_raw[mask_non_zero, -3] / considered_trials[mask_non_zero, 0]).std(axis=-1)

# compute mean and std for n_parameters
table_values[mask_non_zero, 0] += table_values_raw[mask_non_zero, 0].mean(axis=-1)
table_values[mask_non_zero, 1] += table_values_raw[mask_non_zero, 0].std(axis=-1)

# post-processing of runs of training data

mask_non_zero = table_values_raw_training[:, -3, 0] != 0

# compute averaged scores and replace with summed scores in table_values (table_values[..., -3:])
table_values[mask_non_zero, -3:] += table_values_raw_training[mask_non_zero, -3:].sum(axis=-1) / considered_trials_training
# compute average trial likelihood and std from NLL
table_values[mask_non_zero, 2] += np.exp(-table_values_raw_training[mask_non_zero, -3].sum(axis=-1) / considered_trials_training)
table_values[mask_non_zero, 3] += np.exp(-table_values_raw_training[mask_non_zero, -3] / considered_trials_participant_training.reshape(1, -1)).std(axis=-1)

# compute mean and std for n_parameters
table_values[mask_non_zero, 0] += table_values_raw_training[mask_non_zero, 0].mean(axis=-1)
table_values[mask_non_zero, 1] += table_values_raw_training[mask_non_zero, 0].std(axis=-1)

# ------------------------------------------------------------
# Latex table
# ------------------------------------------------------------

# convert np.ndarray into latex table with the given header and index labels   
headers = ['$n_\\text{parameters}$', '$(\sigma)$', '$\\bar{\mathcal{L}}$', '($\sigma$)', 'NLL', 'AIC', 'BIC']
indexes = ['Baseline', 'Benchmark', 'LSTM']
for value in l2_values:
    indexes.append(value)
    indexes.append('RNN (train)')
    indexes.append('RNN')
    indexes.append('SPICE')
    
    
try:
    df = pd.DataFrame(table_values, columns=headers)
except Exception:
    df = pd.DataFrame(table_values.T, columns=headers)
df.to_csv(os.path.join('analysis/analysis_model_evaluation_tables', 'model_evaluation'+study+'.csv'), index=False)


# print latex table content into terminal    
str_headers = ""
for header in headers:
    str_headers += "&" + header
str_headers += "\\\\"
print("\\toprule")
print(str_headers)
print("\\midrule")

n_l2_values_printed = 0
for index_index, index in enumerate(indexes):
    if index in l2_values:
        print("\\midrule")
    str_content = index
    if index in indexes[:3]:
        for index_header, header in enumerate(headers):
            if index_header == 0 or index_header == 1:
                str_content += "&" + f"{int(table_values[index_index, index_header])}"
            else:
                str_content += "&" + f"{table_values[index_index, index_header]:.5f}"
        str_content += "\\\\"
        print(str_content)
    elif index in l2_values:
        str_content = f"$l_2={index}$"
        str_content += "&"*len(headers)
        str_content += "\\\\"
        print(str_content)
        n_l2_values_printed += 1
    else:
        for index_header, header in enumerate(headers):
            if index_header == 0 or index_header == 1: 
                if index == 'SPICE':
                    str_content += "&" + f"{table_values[index_index-n_l2_values_printed, index_header]:.2f}"
                else:
                    str_content += "&" + f"{int(table_values[index_index-n_l2_values_printed, index_header])}"
            else:
                str_content += "&" + f"{table_values[index_index-n_l2_values_printed, index_header]:.5f}"
        str_content += "\\\\"
        print(str_content)
    
        
print("\\bottomrule")
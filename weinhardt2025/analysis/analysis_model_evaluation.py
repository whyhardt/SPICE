import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# standard methods and classes used for every model evaluation
from spice.resources.model_evaluation import get_scores
from spice.resources.bandits import get_update_dynamics, AgentQ
from spice.resources.rnn_utils import split_data_along_timedim, split_data_along_sessiondim
from spice.utils.setup_agents import setup_agent_rnn, setup_agent_spice
from spice.utils.convert_dataset import convert_dataset

# dataset specific SPICE models
from spice.resources import rnn, sindy_utils

# dataset specific benchmarking models
from weinhardt2025.benchmarking import benchmarking_dezfouli2019
from weinhardt2025.benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022
from weinhardt2025.benchmarking.benchmarking_dezfouli2019 import Dezfouli2019GQL
from weinhardt2025.benchmarking import benchmarking_lstm
from ganesh2024a.spice_config import RNN_ContrDiff, SpiceConfig


# -------------------------------------------------------------------------------
# AGENT CONFIGURATIONS
# -------------------------------------------------------------------------------

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
study = 'ganesh2024a'
train_test_ratio = [2,4,6]
sindy_config = SpiceConfig
rnn_class = RNN_ContrDiff
additional_inputs = ['contrast_difference']
path_data = f'{study}/data/{study}_merged_rewards.csv'
path_model_benchmark_lstm = f'{study}/params/lstm_{study}.pkl'
kw_dataset_convert = {'df_participant_id': 'subjID', 'df_choice': 'chose_right', 'df_block': 'blocks'}

# ------------------- CONFIGURATION ECKSTEIN2022 w/o AGE --------------------
# study = 'eckstein2022'
# models_benchmark = ['ApAnBrBcfBch']#['ApBr', 'ApBrAcfpBcf', 'ApBrAcfpBcfBch', 'ApAnBrBch', 'ApAnBrAcfpAcfnBcfBch', 'ApAnBrBcfBch']
# train_test_ratio = 0.8
# sindy_config = sindy_utils.SindyConfig_eckstein2022
# rnn_class = rnn.RLRNN_eckstein2022
# additional_inputs = None
# setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
# rl_model = benchmarking_eckstein2022.rl_model
# benchmark_file = f'mcmc_{study}_MODEL.nc'
# model_config_baseline = 'ApBr'
# baseline_file = f'mcmc_{study}_ApBr.nc'
# kw_dataset_convert = {}

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
# study = 'dezfouli2019'
# train_test_ratio = [3, 6, 9]
# models_benchmark = ['PhiChiBetaKappaC']
# sindy_config = sindy_utils.SindyConfig_eckstein2022
# rnn_class = rnn.RLRNN_eckstein2022
# additional_inputs = []
# # setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_benchmark
# # gql_model = benchmarking_dezfouli2019.gql_model
# setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
# gql_model = benchmarking_dezfouli2019.Dezfouli2019GQL
# benchmark_file = f'gql_{study}_MODEL.pkl'
# model_config_baseline = 'PhiBeta'
# baseline_file = f'gql_{study}_PhiBeta.pkl'
# kw_dataset_convert = {}

# ------------------------ CONFIGURATION GERSHMAN2018 -----------------------
# study = 'gershmanB2018'
# train_test_ratio = [4, 8, 12, 16]
# models_benchmark = ['PhiBeta']
# sindy_config = sindy_utils.SindyConfig_eckstein2022
# rnn_class = rnn.RLRNN_eckstein2022
# additional_inputs = []
# # setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_benchmark
# # gql_model = benchmarking_dezfouli2019.gql_model
# setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
# gql_model = benchmarking_dezfouli2019.Dezfouli2019GQL
# benchmark_file = f'ql_{study}_MODEL.pkl'
# model_config_baseline = 'PhiBeta'
# baseline_file = f'ql_{study}_PhiBeta.pkl'
# kw_dataset_convert = {}

# ------------------------- CONFIGURATION FILE PATHS ------------------------
use_test = True

# path_data = f'data/{study}/{study}.csv'
path_model_rnn = None#f'params/{study}/rnn_{study}_l2_0_00001.pkl'
path_model_spice = None#f'params/{study}/spice_{study}_l2_0_00001.pkl'
path_model_baseline = None#os.path.join(f'params/{study}/', baseline_file)
path_model_benchmark = None#os.path.join(f'params/{study}', benchmark_file) if len(models_benchmark) > 0 else None
# path_model_benchmark_lstm = f'params/{study}/lstm_{study}.pkl'

# -------------------------------------------------------------------------------
# MODEL COMPARISON PIPELINE
# -------------------------------------------------------------------------------

dataset = convert_dataset(path_data, additional_inputs=additional_inputs, **kw_dataset_convert)[0]
# use these participant_ids if not defined later
participant_ids = dataset.xs[:, 0, -1].unique().cpu().numpy()

# ------------------------------------------------------------
# Setup of agents
# ------------------------------------------------------------

print("Computing metrics on", 'test' if use_test else 'training', "data...")

# setup baseline model
# old: win-stay-lose-shift -> very bad fit; does not bring the point that SPICE models are by far better than original ones
# new: Fitted ApBr model -> Tells the "true" story of how much better SPICE models can actually be by setting a good relative baseline
print("Setting up baseline agent from file", path_model_baseline)
if path_model_baseline:
    agent_baseline = setup_agent_benchmark(path_model=path_model_baseline, model_config=model_config_baseline)
else:
    # agent_baseline = [AgentQ(alpha_reward=0.3, beta_reward=1) for _ in range(len(dataset))]
    agent_baseline = [[AgentQ(alpha_reward=0., beta_reward=1, beta_choice=3) for _ in range(len(dataset))], 2]

n_parameters_baseline = 2

# setup benchmark models
if path_model_benchmark:
    print("Setting up benchmark agent from file", path_model_benchmark)
    agent_benchmark = {}
    for model in models_benchmark:
        agent_benchmark[model] = setup_agent_benchmark(path_model=path_model_benchmark.replace('MODEL', model), model_config=model)
else:
    models_benchmark = []
n_parameters_benchmark = 0

if path_model_benchmark_lstm:
    print("Setting up LSTM agent from file", path_model_benchmark_lstm)
    agent_lstm = benchmarking_lstm.setup_agent_lstm(path_model=path_model_benchmark_lstm)
    n_parameters_lstm = sum(p.numel() for p in agent_lstm._model.parameters() if p.requires_grad)
else:
    n_parameters_lstm = 0
    
# setup rnn agent
if path_model_rnn is not None:
    print("Setting up RNN agent from file", path_model_rnn)
    agent_rnn = setup_agent_rnn(
        class_rnn=rnn_class,
        path_model=path_model_rnn, 
        )
    n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)
else:
    n_parameters_rnn = 0
    
# setup spice agent
if path_model_spice is not None:
    print("Setting up SPICE agent from file", path_model_spice)
    
    # get SPICE agent
    agent_spice = setup_agent_spice(
        class_rnn=rnn_class,
        path_rnn=path_model_rnn,
        path_spice=path_model_spice,
    )

n_parameters_spice = 0
    
# ------------------------------------------------------------
# Dataset splitting
# ------------------------------------------------------------

# split data into according to train_test_ratio
if isinstance(train_test_ratio, float):
    dataset_train, dataset_test = split_data_along_timedim(dataset, split_ratio=train_test_ratio)
    data_input = dataset.xs
    data_test = dataset.xs[..., :agent_baseline[0][0]._n_actions]
    # n_trials_test = dataset_test.xs.shape[1]
    
elif isinstance(train_test_ratio, list) or isinstance(train_test_ratio, tuple):
    dataset_train, dataset_test = split_data_along_sessiondim(dataset, list_test_sessions=train_test_ratio)
    if not use_test:
        dataset_test = dataset_train
    data_input = dataset_test.xs
    data_test = dataset_test.xs[..., :agent_baseline[0][0]._n_actions]
    
else:
    raise TypeError("train_test_raio must be either a float number or a list of integers containing the session/block ids which should be used as test sessions/blocks")

# ------------------------------------------------------------
# Computation of metrics
# ------------------------------------------------------------

print('Running model evaluation...')
scores = np.zeros((5+len(models_benchmark), 3))

failed_attempts = 0
considered_trials = 0

metric_participant = np.zeros((len(scores), len(dataset_test)))
parameters_participant = np.zeros((1, len(dataset_test)))
best_benchmarks_participant, considered_trials_participant = np.array(['' for _ in range(len(dataset_test))]), np.zeros(len(dataset_test))

# from resources.rnn_utils import DatasetRNN
# mask_dataset_test = dataset_test.xs[:, 0, -1] == 45
# dataset_test = DatasetRNN(dataset_test.xs[mask_dataset_test], dataset_test.ys[mask_dataset_test])

for index_data in tqdm(range(len(dataset_test))):
    try:
        # use whole session to include warm-up phase; make sure to exclude warm-up phase when computing metrics
        pid = dataset_test.xs[index_data, 0, -1].int().item()
        
        if not pid in participant_ids:
            print(f"Skipping participant {index_data} because they could not be found in the SPICE participants. Probably due to prior filtering of badly fitted participants.")
            continue
        
        # Baseline model
        probs_baseline = get_update_dynamics(experiment=data_input[index_data], agent=agent_baseline[0][pid])[1]
        
        # get number of actual trials
        n_trials = len(probs_baseline)
        data_ys = data_test[index_data, :n_trials].cpu().numpy()
        
        if isinstance(train_test_ratio, float):
            n_trials_test = int(n_trials*(1-train_test_ratio))
            if use_test:
                index_start = n_trials - n_trials_test
                index_end = n_trials
            else:
                index_start = 0
                index_end = n_trials - n_trials_test
        else:
            index_start = 0
            index_end = n_trials
        
        scores_baseline = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_baseline[index_start:index_end], n_parameters=agent_baseline[1]))
        metric_participant[0, index_data] += scores_baseline[0]
        
        # get scores of all mcmc benchmark models but keep only the best one for each session
        if path_model_benchmark:
            scores_benchmark = np.zeros((len(models_benchmark), 3))
            for index_model, model in enumerate(models_benchmark):
                n_parameters_model = agent_benchmark[model][1]
                probs_benchmark = get_update_dynamics(experiment=data_input[index_data], agent=agent_benchmark[model][0][pid])[1]
                scores_benchmark[index_model] += np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_benchmark[index_start:index_end], n_parameters=n_parameters_model))
            index_best_benchmark = np.argmin(scores_benchmark, axis=0)[1] # index 0 -> NLL is indicating metric
            n_parameters_benchmark += agent_benchmark[models_benchmark[index_best_benchmark]][1]
            best_benchmarks_participant[index_data] += models_benchmark[index_best_benchmark]
            metric_participant[1, index_data] += scores_benchmark[index_best_benchmark, 0]
            metric_participant[5:, index_data] += scores_benchmark[:, 0]
        
        # Benchmark LSTM
        if path_model_benchmark_lstm:
            probs_lstm = get_update_dynamics(experiment=data_input[index_data], agent=agent_lstm)[1]
            scores_lstm = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_lstm[index_start:index_end], n_parameters=n_parameters_lstm))
            metric_participant[2, index_data] += scores_lstm[0]
            
        # SPICE-RNN
        if path_model_rnn is not None:
            probs_rnn = get_update_dynamics(experiment=data_input[index_data], agent=agent_rnn)[1]
            scores_rnn = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_rnn[index_start:index_end], n_parameters=n_parameters_rnn))
            metric_participant[3, index_data] = scores_rnn[0]
            
        # SPICE
        if path_model_spice is not None:
            additional_inputs_embedding = data_input[0, agent_spice._n_actions*2:-3]
            agent_spice.new_sess(participant_id=pid, additional_embedding_inputs=additional_inputs_embedding)
            n_params_spice = agent_spice.count_parameters()[pid]
            
            probs_spice = get_update_dynamics(experiment=data_input[index_data], agent=agent_spice)[1]
            scores_spice = np.array(get_scores(data=data_ys[index_start:index_end], probs=probs_spice[index_start:index_end], n_parameters=n_params_spice))
            n_parameters_spice += n_params_spice
            parameters_participant[0, index_data] = n_params_spice
            metric_participant[4, index_data] = scores_spice[0]        
        
        considered_trials_participant[index_data] += index_end - index_start
        considered_trials += index_end - index_start
        
        # track scores
        scores[0] += scores_baseline
        if path_model_benchmark:
            scores[1] += scores_benchmark[index_best_benchmark]
            scores[5:] += scores_benchmark
        if path_model_benchmark_lstm:
            scores[2] += scores_lstm
        if path_model_rnn is not None:
            scores[3] += scores_rnn
        if path_model_spice is not None:
            scores[4] += scores_spice
        
    except Exception as e:  
        # print(e)
        raise e
        failed_attempts += 1

# ------------------------------------------------------------
# Post processing
# ------------------------------------------------------------

if path_model_benchmark:
    # print how often each benchmark model was the best one
    from collections import Counter
    occurrences = Counter(best_benchmarks_participant)
    print("Counter for each benchmark model being the best one:")
    print(occurrences)

# compute trial-level metrics (and NLL -> Likelihood)
scores = scores / (considered_trials)# * agent_baseline[0]._n_actions)
avg_trial_likelihood = np.exp(- scores[:, 0])
# avg_trial_likelihood = np.exp(- scores_participant.T / (considered_trials_participant.reshape(-1, 1) * agent_baseline[0]._n_actions))
# scores[:, 0] = scores[:, 0] / len(dataset)

metric_participant_std = (metric_participant/considered_trials_participant).std(axis=1)
avg_trial_likelihood_participant = np.exp(- metric_participant / considered_trials_participant)
avg_trial_likelihood_participant_std = avg_trial_likelihood_participant.std(axis=1)
parameter_participant_std = parameters_participant.std(axis=1)

# index_sorted = np.argsort(avg_trial_likelihood_participant[3] - avg_trial_likelihood_participant[4])[::-1]
# index_sorted = np.argsort(avg_trial_likelihood_participant[3])
# pd.DataFrame({
#     'INDEX': np.arange(0, 303)[index_sorted],
#     'SESSION': (np.zeros((101, 3),dtype=int)+np.array((3,6,9),dtype=int).reshape(1, 3)).reshape(-1,)[index_sorted],
#     'PID':dataset_test.xs[:, 0, -1].numpy()[index_sorted],
#     'RNN':avg_trial_likelihood_participant[3][index_sorted],
#     'SPICE':avg_trial_likelihood_participant[4][index_sorted]
#     }).to_csv('avg_trial_likelihood_participants.csv')

# pd.DataFrame(data=np.concatenate((np.array(best_benchmarks_participant).reshape(-1, 1), avg_trial_likelihood), axis=1), columns=['benchmark model', 'baseline','benchmark', 'lstm', 'rnn', 'spice']+models_benchmark).to_csv('best_scores_benchmark.csv')

# compute average number of parameters
n_parameters_benchmark_single_models = [agent_benchmark[model][1] for model in models_benchmark] if path_model_benchmark else []
n_parameters = np.array([
    n_parameters_baseline,
    n_parameters_benchmark/(len(dataset_test)-failed_attempts) if path_model_benchmark else 0, 
    n_parameters_lstm,
    n_parameters_rnn, 
    n_parameters_spice/(len(dataset_test)-failed_attempts),
    ]+n_parameters_benchmark_single_models)

scores = np.concatenate((avg_trial_likelihood.reshape(-1, 1), avg_trial_likelihood_participant_std.reshape(-1, 1), scores[:, :1], metric_participant_std.reshape(-1, 1), scores[:, 1:], n_parameters.reshape(-1, 1)), axis=1)


# ------------------------------------------------------------
# Printing model performance table
# ------------------------------------------------------------

print(f'Failed attempts: {failed_attempts}')

df = pd.DataFrame(
    data=scores,
    index=['Baseline', 'Benchmark', 'LSTM', 'RNN', 'SPICE']+models_benchmark,
    columns = ('Trial Lik.', '(std)', 'NLL', '(std)', 'AIC', 'BIC', 'n_parameters'),
    )
print(df)
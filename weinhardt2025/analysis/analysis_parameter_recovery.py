import os, sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from copy import copy
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.setup_agents import setup_agent_spice
from resources.bandits import AgentSpice
from resources.rnn import RLRNN_eckstein2022 as RLRNN
from resources.sindy_utils import SindyConfig_eckstein2022 as SindyConfig


save_plots = False

# -----------------------------------------------------------------------------------------------
# create a mapping of ground truth parameters to library parameters
# -----------------------------------------------------------------------------------------------

mapping_x_learning_rate_reward = {
    '1': lambda alpha_reward, alpha_penalty: alpha_penalty,
     
    'x_learning_rate_reward': lambda alpha_reward, alpha_penalty: 0,
    
    'c_reward_chosen': lambda alpha_reward, alpha_penalty: alpha_reward,
    
    'c_value_reward': lambda alpha_reward, alpha_penalty: 0,

    'c_value_choice': lambda alpha_reward, alpha_penalty: 0,
    
    # 'x_learning_rate_reward c_value_reward': lambda alpha_reward, alpha_penalty: 0,
    # 'c_value_reward x_learning_rate_reward': lambda alpha_reward, alpha_penalty: 0,
    
    # 'x_learning_rate_reward c_reward': lambda alpha_reward, alpha_penalty: 0,
    # 'c_reward x_learning_rate_reward': lambda alpha_reward, alpha_penalty: 0,
    
    # 'x_learning_rate_reward^2': lambda alpha_reward, alpha_penalty: 0,
    
    # 'c_value_reward c_reward': lambda alpha_reward, alpha_penalty: 0,
    # 'c_reward c_value_reward': lambda alpha_reward, alpha_penalty: 0,
    
    # 'c_value_reward^2': lambda alpha_reward, alpha_penalty: 0,
    
    # 'c_reward^2': lambda alpha_reward, alpha_penalty: 0,
}

mapping_x_value_reward_not_chosen = {
    '1': lambda forget_rate: 0.5*forget_rate,
    
    'x_value_reward_not_chosen': lambda forget_rate: 1-forget_rate,
    
    'c_reward_chosen': lambda forget_rate: 0,
    
    'c_value_choice': lambda forget_rate: 0,
    
    # 'x_value_reward_not_chosen c_reward': lambda forget_rate: 0,
    # 'c_reward x_value_reward_not_chosen': lambda forget_rate: 0,
    
    # 'x_value_reward_not_chosen^2': lambda forget_rate: 0,
    
    # 'c_reward^2': lambda forget_rate: 0,
}

mapping_x_value_choice_chosen = {
    '1': lambda alpha_choice: 1, #alpha_choice,
    
    'x_value_choice_chosen': lambda alpha_choice: 0,
    
    'c_value_reward': lambda alpha_choice: 0,
    
    # 'x_value_choice_chosen^2': lambda alpha_choice: 0,
}

mapping_x_value_choice_not_chosen = {
    '1': lambda alpha_choice: 0,
    
    'x_value_choice_not_chosen': lambda alpha_choice: 0,
    
    'c_value_reward': lambda alpha_choice: 0,    
    
    # 'x_value_choice_not_chosen^2': lambda alpha_choice: 0,
}

mapping_betas = {
    'x_learning_rate_reward': lambda agent_or_data: agent_or_data.get_betas()['x_value_reward'] if isinstance(agent_or_data, AgentSpice) else agent_or_data['beta_reward'], # Question: no scaling of learning rate?
    'x_value_reward_not_chosen': lambda agent_or_data: agent_or_data.get_betas()['x_value_reward'] if isinstance(agent_or_data, AgentSpice) else agent_or_data['beta_reward'],
    'x_value_choice_chosen': lambda agent_or_data: agent_or_data.get_betas()['x_value_choice'] if isinstance(agent_or_data, AgentSpice) else agent_or_data['beta_choice'],
    'x_value_choice_not_chosen': lambda agent_or_data: agent_or_data.get_betas()['x_value_choice'] if isinstance(agent_or_data, AgentSpice) else agent_or_data['beta_choice'],
}

mapping_libraries = {
    'x_learning_rate_reward': mapping_x_learning_rate_reward,
    'x_value_reward_not_chosen': mapping_x_value_reward_not_chosen,
    'x_value_choice_chosen': mapping_x_value_choice_chosen,
    'x_value_choice_not_chosen': mapping_x_value_choice_not_chosen,
}

mapping_variable_names = {
    '1': '1',
    'x_learning_rate_reward': r'$\alpha_{t}$',
    'x_value_reward_not_chosen': r'$q_{r,t}$',
    'x_value_choice_chosen': r'$q_{c,t}$',
    'x_value_choice_not_chosen': r'$q_{c,t}$',
    'c_reward_chosen': r'$r$',
    'c_value_reward': r'$q_{r}$',
    'c_value_choice': r'$q_{c}$',
}


# -----------------------------------------------------------------------------------------------
# analysis auxiliary functions
# -----------------------------------------------------------------------------------------------

# special-cases-handles
# necessary because some sindy notations in the mappings interpet the parameters differently than AgentQ
def handle_asymmetric_learning_rates(alpha_reward, alpha_penalty):
    # in AgentQ: alpha = alpha_reward if reward > 0.5 else alpha_penalty
    # in SINDy: alpha = alpha_penalty 1 - alpha_reward r 
    if alpha_reward == alpha_penalty:
        # same learning rates -> 'c_reward' in SINDy is 0 in simplest model
        alpha_reward = 0
    elif alpha_reward > alpha_penalty:
        alpha_reward = alpha_reward - alpha_penalty
    elif alpha_reward < alpha_penalty:
        alpha_reward = -1 * (alpha_penalty - alpha_reward)
    return alpha_reward, alpha_penalty

def argument_extractor(data, library: str):
    if library == 'x_learning_rate_reward':
        return handle_asymmetric_learning_rates(data['alpha_reward'], data['alpha_penalty'])
    elif library == 'x_value_reward_not_chosen':
        return tuple([data['forget_rate']])
    elif library == 'x_value_choice_chosen':
        return tuple([data['alpha_choice']])
    elif library == 'x_value_choice_not_chosen':
        return tuple([data['alpha_choice']])
    else:
        raise ValueError(f'The argument extractor for the library {library} is not implemented.')
    
def identified_params(true_coefs: np.ndarray, recovered_coefs: np.ndarray):
    # check which true coefficients are zeros and non-zeros
    non_zero_features = true_coefs != 0
    zero_features = true_coefs == 0
    
    # count all correctly and non-correctly identified parameters
    true_pos = np.sum(recovered_coefs[non_zero_features] != 0)
    false_neg = np.sum(recovered_coefs[non_zero_features] == 0)
    true_neg = np.sum(recovered_coefs[zero_features] == 0)
    false_pos = np.sum(recovered_coefs[zero_features] != 0)
    
    # get identification rates of correctly and non-correctly identified parameters
    true_pos_rel = true_pos / np.sum(non_zero_features) if np.sum(non_zero_features) > 0 else np.nan
    false_neg_rel = false_neg / np.sum(non_zero_features) if np.sum(non_zero_features) > 0 else np.nan
    true_neg_rel = true_neg / np.sum(zero_features) if np.sum(zero_features) > 0 else np.nan
    false_pos_rel = false_pos / np.sum(zero_features) if np.sum(zero_features) > 0 else np.nan
    
    return (true_pos, true_neg, false_pos, false_neg), (true_pos_rel, true_neg_rel, false_pos_rel, false_neg_rel)

def n_true_params(true_coefs):
    # Count number of non-zero coefficients in AgentQ-parameters
    # Filter for parameter groups
    # group parameters by being either reward- or choice-based
    # if beta of one group is 0, then the other parameters can be considered also as 0s because they won't have any influence on the result
    # same goes for alphas.
    # Caution: the values will also change for true_coefs outside this function. In this scenario it's fine because these values should be modelled as 0s anyways 
    
    # reward-based parameter group
    if true_coefs['beta_reward'] == 0 or (true_coefs['alpha_reward'] == 0 and true_coefs['alpha_penalty'] == 0):
        true_coefs['alpha_reward'] = 0
        true_coefs['alpha_penalty'] = 0
        true_coefs['forget_rate'] = 0
        true_coefs['beta_reward'] = 0
    
    # set alpha_penalty to 0 if alpha_penalty == alpha_reward
    if true_coefs['alpha_reward'] == true_coefs['alpha_penalty']:
        true_coefs['alpha_penalty'] = 0
    
    # choice-based parameter group
    # if true_coefs['beta_choice'] == 0 or true_coefs['alpha_choice'] == 0:
    #     true_coefs['beta_choice'] = 0
    
    return np.sum([
        true_coefs['beta_reward'] != 0,
        true_coefs['alpha_reward'] != 0,
        true_coefs['alpha_penalty'] != 0,
        true_coefs['forget_rate'] != 0,
        true_coefs['beta_choice'] != 0,
        ]).astype(int)


# -----------------------------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------------------------

random_sampling = [0.25, 0.5, 0.75]
n_sessions = [128]#[16, 32, 64, 128, 256]
iterations = 1

base_name_data = 'data/parameter_recovery/data_SESSp_IT.csv'
base_name_params = 'params'#/parameter_recovery'#/rnn_SESSp_IT.pkl'
kw_participant_id = 'session'
path_plots = 'analysis/plots_parameter_recovery'

# meta parameters
# mapping_lens = {'x_V_LR': 10, 'x_V_nc': 3, 'x_C': 3, 'x_C_nc': 3}
mapping_lens = {'x_V_LR': 5, 'x_V_nc': 4, 'x_C': 3, 'x_C_nc': 3}
n_candidate_terms = np.sum([mapping_lens[key] for key in mapping_lens])
n_params_q = 5

# -----------------------------------------------------------------------------------------------
# Initialization of storages
# -----------------------------------------------------------------------------------------------

# parameter correlation coefficients
true_params = []#[np.zeros((sess*iterations, n_candidate_terms)) for sess in n_sessions]
recovered_params = []#[np.zeros((sess*iterations, n_candidate_terms)) for sess in n_sessions]

# parameter identification matrices
true_pos_count = np.zeros((len(n_sessions)+len(random_sampling), n_params_q))
true_neg_count = np.zeros((len(n_sessions)+len(random_sampling),  n_params_q))
false_pos_count = np.zeros((len(n_sessions)+len(random_sampling),  n_params_q))
false_neg_count = np.zeros((len(n_sessions)+len(random_sampling),  n_params_q))

true_pos_rates = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q))
true_neg_rates = np.zeros((len(n_sessions)+len(random_sampling), iterations,  n_params_q))
false_pos_rates = np.zeros((len(n_sessions)+len(random_sampling), iterations,  n_params_q))
false_neg_rates = np.zeros((len(n_sessions)+len(random_sampling), iterations,  n_params_q))

true_pos_rates_count = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q))
true_neg_rates_count = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q))
false_pos_rates_count = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q))
false_neg_rates_count = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q))


# -----------------------------------------------------------------------------------------------
# Start analysis
# -----------------------------------------------------------------------------------------------

random_sampling, n_sessions = tuple(random_sampling), tuple(n_sessions)
for index_sess, sess in enumerate(n_sessions):
    for it in range(iterations):
        path_data = base_name_data.replace('SESS', str(sess)).replace('IT', str(it))
        # path_rnn = base_name_params.replace('SESS', str(sess)).replace('IT', str(it))
        path_rnn = os.path.join(base_name_params, path_data.split(os.path.sep)[-1].replace('.csv', '.pkl').replace('data', 'rnn'))
        path_spice = path_rnn.replace('rnn', 'spice')
        
        if not os.path.isfile(path_rnn):
            continue
            
        # setup of ground truth model from current dataset
        data = pd.read_csv(path_data)
        participant_ids = np.unique(data[kw_participant_id].values)
        
        # setup of sindy agent for current dataset
        sindy_agent = setup_agent_spice(
            class_rnn=RLRNN,
            path_rnn=path_rnn, 
            path_data=path_data,
            path_spice=path_spice,
            rnn_modules=SindyConfig['rnn_modules'], 
            control_parameters=SindyConfig['control_parameters'], 
            sindy_library_setup=SindyConfig['library_setup'],
            sindy_filter_setup=SindyConfig['filter_setup'],
            sindy_dataprocessing=SindyConfig['dataprocessing_setup'],
            sindy_library_polynomial_degree=1,
            use_optuna=True,
            )
        sindy_models = sindy_agent.get_modules()
        
        for index_participant, participant in enumerate(participant_ids):
            sindy_agent.new_sess(participant_id=participant)
            
            # get all true parameters of current participant from dataset
            data_coefs_all = data.loc[data[kw_participant_id]==participant].iloc[-1]
            index_params = n_true_params(copy(data_coefs_all)) - 1
            
            sindy_coefs_array = []
            data_coefs_array = []
            feature_names = []
            
            index_all_candidate_terms = 0
            for index_library, library in enumerate(mapping_libraries):
                # get sindy coefficients
                sindy_coefs_library = sindy_models[library][participant].coefficients()[0]
                # drop every entry feature that contains a u-feature (i.e. dummy-feature)
                feature_names_library = sindy_models[library][participant].get_feature_names()
                index_keep = ['dummy' not in feature for feature in feature_names_library]
                sindy_coefs_array += (sindy_coefs_library[index_keep] * mapping_betas[library](sindy_agent)).tolist()
                feature_names_library = np.array(feature_names_library)[index_keep]
                feature_names += feature_names_library.tolist()
                
                # translate data coefficient to sindy coefficients
                data_coefs_library = np.zeros(len(feature_names_library))
                for index_feature, feature in enumerate(feature_names_library):
                    # data_coefs[feature] = mapping_libraries[library][feature](*argument_extractor(data_coefs_all, library)) 
                    data_coefs_library[index_feature] = mapping_libraries[library][feature](*argument_extractor(data_coefs_all, library)) * mapping_betas[library](data_coefs_all)
                data_coefs_array += data_coefs_library.tolist()
            
            sindy_coefs_array = np.array(sindy_coefs_array)
            data_coefs_array = np.array(data_coefs_array)
            
            # initialize param storages if empty
            if len(true_params) == 0:
                for s in n_sessions:
                    true_params.append(np.zeros((s*iterations, len(feature_names))))
                    recovered_params.append(np.zeros((s*iterations, len(feature_names))))
            
            # add true and recovered parameters for later parameter correlation
            true_params[index_sess][sess*it+index_participant] = data_coefs_array
            recovered_params[index_sess][sess*it+index_participant] = sindy_coefs_array
            
            # compute number of correctly and non-correctly identified parameters
            identification_count, identification_rates = identified_params(data_coefs_array, sindy_coefs_array)
            true_pos, true_neg, false_pos, false_neg = identification_count
            true_pos_rel, true_neg_rel, false_pos_rel, false_neg_rel = identification_rates
            
            # add identification counts
            true_pos_count[index_sess, index_params] += true_pos
            true_neg_count[index_sess, index_params] += true_neg
            false_pos_count[index_sess, index_params] += false_pos
            false_neg_count[index_sess, index_params] += false_neg
            
            # add identification rates
            true_pos_rates[index_sess, it, index_params] += true_pos_rel if not np.isnan(true_pos_rel) else 0
            true_neg_rates[index_sess, it, index_params] += true_neg_rel if not np.isnan(true_neg_rel) else 0
            false_pos_rates[index_sess, it, index_params] += false_pos_rel if not np.isnan(false_pos_rel) else 0
            false_neg_rates[index_sess, it, index_params] += false_neg_rel if not np.isnan(false_neg_rel) else 0
            
            # add identification rate counter
            true_pos_rates_count[index_sess, it, index_params] += 1 if not np.isnan(true_pos_rel) else 0
            true_neg_rates_count[index_sess, it, index_params] += 1 if not np.isnan(true_neg_rel) else 0
            false_pos_rates_count[index_sess, it, index_params] += 1 if not np.isnan(false_pos_rel) else 0
            false_neg_rates_count[index_sess, it, index_params] += 1 if not np.isnan(false_neg_rel) else 0
            
            # sample random coefficients for biggest dataset
            if sess == max(n_sessions):
                # sample random coefficients for each random sampling strategy
                for index_rnd, rnd in enumerate(random_sampling):
                    rnd_coefs = np.random.choice((1, 0), p=(rnd, 1-rnd), size=len(sindy_coefs_array))
                    rnd_betas = np.random.choice((1, 0), p=(rnd, 1-rnd), size=1)
                    
                    # do same stuff as with sindy coefs
                    # compute number of correctly and non-correctly identified parameters
                    identification_count, identification_rates = identified_params(data_coefs_array, rnd_coefs*rnd_betas)
                    true_pos, true_neg, false_pos, false_neg = identification_count
                    true_pos_rel, true_neg_rel, false_pos_rel, false_neg_rel = identification_rates
                    
                    # add identification counts
                    true_pos_count[len(n_sessions)+index_rnd, index_params] += true_pos
                    true_neg_count[len(n_sessions)+index_rnd, index_params] += true_neg
                    false_pos_count[len(n_sessions)+index_rnd, index_params] += false_pos
                    false_neg_count[len(n_sessions)+index_rnd, index_params] += false_neg

                    # add identification rates
                    true_pos_rates[len(n_sessions)+index_rnd, it, index_params] += true_pos_rel if not np.isnan(true_pos_rel) else 0
                    true_neg_rates[len(n_sessions)+index_rnd, it, index_params] += true_neg_rel if not np.isnan(true_neg_rel) else 0
                    false_pos_rates[len(n_sessions)+index_rnd, it, index_params] += false_pos_rel if not np.isnan(false_pos_rel) else 0
                    false_neg_rates[len(n_sessions)+index_rnd, it, index_params] += false_neg_rel if not np.isnan(false_neg_rel) else 0
                    
                    # add identification rate counter
                    true_pos_rates_count[len(n_sessions)+index_rnd, it, index_params] += 1 if not np.isnan(true_pos_rel) else 0
                    true_neg_rates_count[len(n_sessions)+index_rnd, it, index_params] += 1 if not np.isnan(true_neg_rel) else 0
                    false_pos_rates_count[len(n_sessions)+index_rnd, it, index_params] += 1 if not np.isnan(false_pos_rel) else 0
                    false_neg_rates_count[len(n_sessions)+index_rnd, it, index_params] += 1 if not np.isnan(false_neg_rel) else 0


# ------------------------------------------------
# post-processing coefficient correlation
# ------------------------------------------------

# remove outliers in both true and recovered coefs where recovered coefficients are bigger than a big threshold (e.g. abs(recovered_coeff) > 1e1)
threshold = 1e2
removed_params = 0
for index_sess in range(len(n_sessions)):
    mask_keep = np.all(recovered_params[index_sess] <= threshold, axis=1)

    # Count the number of removed rows
    removed_params += np.sum(~mask_keep)
    
    # Filter out the rows exceeding the threshold
    true_params[index_sess] = true_params[index_sess][mask_keep]
    recovered_params[index_sess] = recovered_params[index_sess][mask_keep]
    
print(f'excluded parameters because of high values: {removed_params}')

# get all features across all libraries; remove dummy features with are named 'dummy'
feature_names = []
for library in mapping_libraries:
    feature_names += sindy_models[library][participant].get_feature_names()
index_keep = ['dummy' not in feature for feature in feature_names]
feature_names = np.array(feature_names)[index_keep]

correlation_matrix, recovery_errors, recovery_errors_median, recovery_errors_std = [], [], [], []
for index_sess in range(len(n_sessions)):
#     correlation_matrix.append(
#         pd.DataFrame(
#             np.corrcoef(
#                 true_params[index_sess], 
#                 recovered_params[index_sess], 
#                 rowvar=False,
#                 )[len(feature_names):, :len(feature_names)],
#             columns=feature_names,
#             index=feature_names,
#             )
#         )

    # normalizing params
    v_max = np.nanmax(true_params[index_sess], axis=0)
    v_min = np.nanmin(true_params[index_sess], axis=0)
    index_normalize = v_max-v_min != 0
    
    true_params[index_sess] = (true_params[index_sess] - v_min)
    true_params[index_sess][:, index_normalize] = true_params[index_sess][:, index_normalize] / (v_max - v_min)[index_normalize]
    
    recovered_params[index_sess] = (recovered_params[index_sess] - v_min) 
    recovered_params[index_sess][:, index_normalize] = recovered_params[index_sess][:, index_normalize]/ (v_max - v_min)[index_normalize]
    
    recovery_errors.append(
        true_params[index_sess] - recovered_params[index_sess]
        )


# ------------------------------------------------
# post-processing identification rates
# ------------------------------------------------

# average across counts
true_pos_rates[true_pos_rates_count > 0] /= true_pos_rates_count[true_pos_rates_count > 0]
true_neg_rates[true_neg_rates_count > 0] /= true_neg_rates_count[true_neg_rates_count > 0]
false_pos_rates[false_pos_rates_count > 0] /= false_pos_rates_count[false_pos_rates_count > 0]
false_neg_rates[false_neg_rates_count > 0] /= false_neg_rates_count[false_neg_rates_count > 0]
true_pos_rates[true_pos_rates_count == 0] = np.nan
true_neg_rates[true_neg_rates_count == 0] = np.nan
false_pos_rates[false_pos_rates_count == 0] = np.nan
false_neg_rates[false_neg_rates_count == 0] = np.nan

true_pos_sessions_mean = np.nanmean(true_pos_rates, axis=1)
true_neg_sessions_mean = np.nanmean(true_neg_rates, axis=1)
false_pos_sessions_mean = np.nanmean(false_pos_rates, axis=1)
false_neg_sessions_mean = np.nanmean(false_neg_rates, axis=1)

true_pos_sessions_std = np.nanstd(true_pos_rates, axis=1)
true_neg_sessions_std = np.nanstd(true_neg_rates, axis=1)
false_pos_sessions_std = np.nanstd(false_pos_rates, axis=1)
false_neg_sessions_std = np.nanstd(false_neg_rates, axis=1)


# ------------------------------------------------
# post-processing identification counts
# ------------------------------------------------

accuracy = (true_pos_count + true_neg_count) / (true_pos_count + true_neg_count + false_pos_count + false_neg_count)
precision = true_pos_count / (true_pos_count + false_pos_count)
recall = true_pos_count / (true_pos_count + false_neg_count)
false_pos_rate = false_pos_count / (false_pos_count + true_neg_count)
f1_score = 2 * (precision * recall) / (precision + recall)

# ------------------------------------------------
# configuration identification plots
# ------------------------------------------------

v_min = 0
v_max = 1#np.nanmax(np.stack((true_pos_sessions_mean, false_pos_sessions_mean, true_neg_sessions_mean, false_neg_sessions_mean)), axis=(-1, -2, -3))

identification_matrix_mean = [
    [true_pos_sessions_mean , false_pos_sessions_mean], 
    [false_neg_sessions_mean , true_neg_sessions_mean],
]

identification_matrix_std = [
    [true_pos_sessions_std, false_pos_sessions_std], 
    [false_neg_sessions_std, true_neg_sessions_std],
    ]

identification_metrics_matrix = [
    [accuracy, precision],
    [recall, f1_score],
]

identification_xlabels = [
    ['', ''],
    ['Number of model parameters', 'Number of model parameters'],
]

identification_ylabels = [
    ['Number of participants', ''],
    ['Number of participants', ''],
]

identification_headers = [
    ['true positive', 'false pos'],
    ['false negative', 'true negative'],
]

identification_metrics_headers = [
    ['Accuracy', 'Precision'],
    ['Recall', 'F1 Score'],
]

bin_edges_params = np.arange(1, n_params_q+1)
identification_x_axis_ticks = [
    [bin_edges_params, bin_edges_params],
    [bin_edges_params, bin_edges_params],
]

y_tick_labels = n_sessions + random_sampling

linestyles = ['-'] * len(n_sessions) + ['--'] * len(random_sampling)
alphas = [0.3] * len(n_sessions) + [0] * len(random_sampling)

for index_feature, feature in enumerate(feature_names):
    for index_symbol, symbol in enumerate(mapping_variable_names):
        if symbol in feature:
            feature = feature.replace(symbol, mapping_variable_names[symbol])
    feature_names[index_feature] = feature

# ------------------------------------------------
# parameter identification rates
# ------------------------------------------------

# heatmap of relative true positives, false positives, false negatives, true negatives

fig, axs = plt.subplots(
    nrows=len(identification_matrix_mean), 
    ncols=len(identification_matrix_mean[0])+1,
    gridspec_kw={
        'width_ratios': [10]*len(identification_matrix_mean[0]) + [1],
        },
    )

for index_row, row in enumerate(identification_matrix_mean):
    for index_col, col in enumerate(row):
        if col is not None:
            sns.heatmap(
                col, 
                annot=True, 
                cmap='viridis',
                center=0,
                ax=axs[index_row, index_col],
                cbar=True if index_col == len(identification_matrix_mean[0])-1 else False, 
                cbar_ax=axs[index_row, len(identification_matrix_mean[0])],
                xticklabels=np.arange(1, n_params_q+1) if index_row == len(identification_matrix_mean)-1 else ['']*n_params_q, 
                yticklabels=y_tick_labels if index_col == 0 else ['']*len(n_sessions), 
                vmin=v_min,
                vmax=v_max,
                )
        axs[index_row, index_col].set_title(identification_headers[index_row][index_col], fontsize=10)
        axs[index_row, index_col].tick_params(labelsize=10)
        axs[index_row, index_col].set_xlabel(identification_xlabels[index_row][index_col], fontsize=10)
        axs[index_row, index_col].set_ylabel(identification_ylabels[index_row][index_col], fontsize=10)

if save_plots:
    plt.savefig(os.path.join(path_plots, 'ident_rates'), dpi=500)
else:
    plt.show()

# line plots

fig, axs = plt.subplots(
    nrows=len(identification_matrix_mean), 
    ncols=len(identification_matrix_mean[0]),
    sharey=True,
    )

for index_row, row in enumerate(identification_matrix_mean):
    for index_col, col in enumerate(row):
        if col is not None:
            ax = axs[index_row, index_col]
            for index_sample_size, sample_size in enumerate(y_tick_labels):
                std = np.std(col[index_sample_size])
                ax.fill_between(
                    x=np.arange(0, n_params_q),
                    y1=col[index_sample_size] + identification_matrix_std[index_row][index_col][index_sample_size],
                    y2=col[index_sample_size] - identification_matrix_std[index_row][index_col][index_sample_size],
                    alpha=alphas[index_sample_size]
                    )
                ax.plot(
                    np.arange(0, n_params_q),
                    col[index_sample_size],
                    marker='.',
                    linestyle=linestyles[index_sample_size],
                    label=sample_size,
                    markersize=10,
                    )
                # Set titles for individual subplots
                ax.set_title(identification_headers[index_row][index_col])
                
                # Configure x-axis labels and ticks only for the lowest row
                if index_row == len(identification_matrix_mean) - 1:
                    ax.set_xlabel('$n_{parameters}$')
                    ax.set_xticks(bin_edges_params)
                    ax.tick_params(axis='x', labelsize=8)
                else:
                    ax.set_xticklabels([])

                ax.set_ylim([0, 1])
axs[0, 0].legend()
plt.tight_layout()

if save_plots:
    plt.savefig(os.path.join(path_plots, 'ident_rates_lines'), dpi=500)
else:
    plt.show()

# heatmap of accuracy, precision, true positive rate, false positive rate

fig, axs = plt.subplots(
    nrows=len(identification_metrics_matrix), 
    ncols=len(identification_metrics_matrix[0])+1,
    gridspec_kw={
        'width_ratios': [10]*len(identification_metrics_matrix[0]) + [1],
        },
    )

for index_row, row in enumerate(identification_metrics_matrix):
    for index_col, col in enumerate(row):
        if col is not None:
            sns.heatmap(
                col, 
                annot=True, 
                cmap='viridis',
                center=0,
                ax=axs[index_row, index_col],
                cbar=True if index_col == len(identification_metrics_matrix[0])-1 else False, 
                cbar_ax=axs[index_row, len(identification_metrics_matrix[0])],
                xticklabels=np.arange(1, n_params_q+1) if index_row == len(identification_metrics_matrix)-1 else ['']*n_params_q, 
                yticklabels=y_tick_labels if index_col == 0 else ['']*len(n_sessions), 
                vmin=v_min,
                vmax=v_max,
                )
        axs[index_row, index_col].set_title(identification_metrics_headers[index_row][index_col], fontsize=10)
        axs[index_row, index_col].tick_params(labelsize=10)
        axs[index_row, index_col].set_xlabel(identification_xlabels[index_row][index_col], fontsize=10)
        axs[index_row, index_col].set_ylabel(identification_ylabels[index_row][index_col], fontsize=10)

if save_plots:
    plt.savefig(os.path.join(path_plots, 'ident_metrics'), dpi=500)
else:
    plt.show()

# ------------------------------------------------
# parameter correlation coefficients
# ------------------------------------------------

# heatmaps per dataset size

# v_min, v_max = -1, 1
# fig, axs = plt.subplots(
#     nrows=1, 
#     ncols=len(n_sessions)+1,
#     gridspec_kw={
#         'width_ratios': [10]*len(n_sessions) + [1],
#         },
#     )

# for index_sess in range(len(n_sessions)):
#     sns.heatmap(
#         correlation_matrix[index_sess],
#         annot=True,
#         cmap='viridis',
#         center=0,
#         ax=axs[index_sess],
#         cbar=True if index_sess == len(n_sessions)-1 else False,
#         cbar_ax=axs[index_sess+1],
#         xticklabels=feature_names,
#         yticklabels=feature_names if index_sess == 0 else ['']*len(feature_names),
#         vmin=v_min,
#         vmax=v_max,
#         )
# plt.show()

# box plot

# fig, axs = plt.subplots(
#     nrows=1, 
#     ncols=max((len(n_sessions), 2)),
#     sharey=True,
#     )

# for index_sess in range(len(n_sessions)):
#     ax = axs[index_sess]
#     sns.boxplot(
#         pd.DataFrame(
#             recovery_errors[index_sess],
#             columns=feature_names,
#         ),
#         ax=ax,
#         showfliers=False,
#     )
#     ax.plot(feature_names[-1], 0, '--', color='tab:gray', linewidth=0.5)
#     ax.tick_params(axis='x', labelrotation=45, labelsize=8)
#     ax.tick_params(axis='y', labelsize=8)
#     ax.set_title(n_sessions[index_sess])
# plt.show()

# #  scatter plot

# fig, axs = plt.subplots(
#     nrows=max((len(n_sessions), 2)),
#     ncols=len(feature_names),
#     )

# for index_sess in range(len(n_sessions)):
#     for index_feature in range(len(feature_names)):
#         ax = axs[index_sess, index_feature]
        
#         ax.scatter(
#             true_params[index_sess][:, index_feature], 
#             recovered_params[index_sess][:, index_feature], 
#             marker='o', 
#             color='tab:red', 
#             alpha=0.2,
#             )
#         ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color='tab:gray')
        
#         # Axes settings
#         ax.set_ylim(0, 1)
#         ax.set_xlim(0, 1)
        
#         if index_sess != len(n_sessions) - 1:
#             ax.tick_params(axis='x', which='both', labelbottom=False)  # No x-axis ticks
#         else:
#             ax.set_xlabel('True', fontsize=10)
#             ax.tick_params(axis='x', labelsize=8)
        
#         if index_feature != 0:
#             ax.tick_params(axis='y', which='both', labelleft=False)  # No y-axis ticks
#         else:
#             ax.set_ylabel(f'{n_sessions[index_sess]} participants\nRecovered', fontsize=10)
#             ax.tick_params(axis='y', labelsize=8)
        
#         if index_sess == 0:
#             ax.set_title(feature_names[index_feature], fontsize=10)
        
# plt.show()

# scatter plot

# Adjust the width of each column
col_width_ratios = []
for index_feature in range(len(feature_names)):
    feature_ranges = [
        np.max(true_params[index_sess][:, index_feature]) - np.min(true_params[index_sess][:, index_feature])
        for index_sess in range(len(n_sessions))
    ]
    # Use a fallback small value for zero ranges
    col_width_ratios.append(max(max(feature_ranges), 0.1))  # Ensure non-zero width

# Normalize width ratios
col_width_ratios = [r / max(col_width_ratios) for r in col_width_ratios]

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(len(n_sessions), len(feature_names), width_ratios=col_width_ratios)

for index_sess in range(len(n_sessions)):
    for index_feature in range(len(feature_names)):
        # ax = axs[index_sess, index_feature]
        ax = fig.add_subplot(gs[index_sess, index_feature])
        
        # Scatter plot
        ax.scatter(
            true_params[index_sess][:, index_feature], 
            recovered_params[index_sess][:, index_feature], 
            marker='o', 
            color='cadetblue', 
            alpha=0.2,
            # markersize=0.1,
        )

        # Calculate linear regression for trend line
        index_not_nan = (1-(np.isnan(true_params[index_sess][:, index_feature]) + np.isnan(recovered_params[index_sess][:, index_feature]))).astype(bool)
        model = LinearRegression()
        model.fit(true_params[index_sess][:, index_feature].reshape(-1, 1)[index_not_nan], recovered_params[index_sess][:, index_feature][index_not_nan])
        trend_line = model.predict(np.linspace(0, 1, 100).reshape(-1, 1))
        
        # Reference line
        if true_params[index_sess][:, index_feature].max() > 0:
            ax.plot([0, 1], [0, 1], '--', color='tab:gray', linewidth=1)
            # Plot dashed trend line
            ax.plot(np.linspace(0, 1, 100), trend_line, '--', color='tab:blue', label='Trend line')
            # Axes settings
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
        else:
            # Axes settings
            ax.set_ylim(-.1, .1)
            ax.set_xlim(-.1, .1)
            # Plot dashed trend line
            ax.plot(np.linspace(-1, 1, 100), trend_line, '--', color='tab:blue', label='Trend line')
            
        if index_sess != len(n_sessions) - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)  # No x-axis ticks
        else:
            ax.set_xlabel('True', fontsize=10)
            ax.tick_params(axis='x', labelsize=8)

        if index_feature != 0:
            ax.tick_params(axis='y', which='both', labelleft=False)  # No y-axis ticks
        else:
            ax.set_ylabel(f'{n_sessions[index_sess]} participants\nRecovered', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

        if index_sess == 0:
            ax.set_title(feature_names[index_feature], fontsize=10)

if save_plots:
    plt.savefig(os.path.join(path_plots, 'param_recovery_scatter'), dpi=500)
else:
    plt.show()


# box plot
# like scatter plot before but with box-whisker per true-parameter range (e.g. 0.1) instead of single dots

# Parameters for binning
num_bins = 10  # Number of bins for 5% width (0.05 * 100 = 20 bins)
bin_edges = np.linspace(0, 1, num_bins + 1)
color_box = 'cadetblue'
trend_line_color = 'black'

# fig, axs = plt.subplots(
#     nrows=max(len(n_sessions), 2),
#     ncols=len(feature_names),
#     figsize=(12, 6)
# )

# Adjust the width of each column
col_width_ratios = []
for index_feature in range(len(feature_names)):
    feature_ranges = [
        np.max(true_params[index_sess][:, index_feature]) - np.min(true_params[index_sess][:, index_feature])
        for index_sess in range(len(n_sessions))
    ]
    # Use a fallback small value for zero ranges
    col_width_ratios.append(max(max(feature_ranges), 0.1))  # Ensure non-zero width

# Normalize width ratios
col_width_ratios = [r / max(col_width_ratios) for r in col_width_ratios]

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(len(n_sessions), len(feature_names), width_ratios=col_width_ratios)

for index_sess in range(len(n_sessions)):
    for index_feature in range(len(feature_names)):
        # ax = axs[index_sess, index_feature]
        ax = fig.add_subplot(gs[index_sess, index_feature])
        
        # Prepare the data
        true_vals = true_params[index_sess][:, index_feature]
        recovered_vals = recovered_params[index_sess][:, index_feature]
        
        # Bin the data based on true parameters
        bins = np.digitize(true_vals, bin_edges) - 1  # Binning index
        bins = np.clip(bins, 0, num_bins - 1)  # Ensure bin indices are within range
        
        # Compute box-plot statistics for each bin
        box_data = [recovered_vals[bins == i] for i in range(num_bins)]
        median = [np.mean(data) if len(data) > 0 else np.nan for data in box_data]

        # Create box-whisker plot
        ax.boxplot(
            box_data, 
            positions=bin_edges[:-1] + 0.025, 
            widths=0.04, 
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color_box, color=color_box, alpha=0.8),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color=color_box, linestyle="--", alpha=0.5),
            capprops=dict(color=color_box, alpha=0.5)
            )
        
        # Plot dashed trend line
        index_not_nan = (1-(np.isnan(true_params[index_sess][:, index_feature]) + np.isnan(recovered_params[index_sess][:, index_feature]))).astype(bool)
        model = LinearRegression()
        model.fit(true_params[index_sess][:, index_feature].reshape(-1, 1)[index_not_nan], recovered_params[index_sess][:, index_feature][index_not_nan])
        trend_line = model.predict(np.linspace(0, 1, 100).reshape(-1, 1))
        
        # Reference line
        if true_params[index_sess][:, index_feature].max() > 0:
            ax.plot([0, 1], [0, 1], ':', color='tab:gray', linewidth=1)
            # Plot dashed trend line
            ax.plot(np.linspace(0, 1, 100), trend_line, '--', color=trend_line_color, label='Trend line')
            # Axes settings
            ax.set_ylim(-.1, 1.1)
            ax.set_xlim(-.1, 1.1)
        else:
            # Axes settings
            ax.set_ylim(-.1, .1)
            ax.set_xlim(-.1, .1)
            # Plot dashed trend line
            ax.plot(np.linspace(-1, 1, 100), trend_line, '--', color=trend_line_color, label='Trend line')
        
        if index_sess != len(n_sessions) - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)  # No x-axis ticks
        else:
            ax.set_xlabel('True', fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
        
        if index_feature != 0:
            ax.tick_params(axis='y', which='both', labelleft=False)  # No y-axis ticks
        else:
            ax.set_ylabel(f'{n_sessions[index_sess]} participants\nRecovered', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
        
        if index_sess == 0:
            ax.set_title(feature_names[index_feature], fontsize=10)

plt.tight_layout()

if save_plots:
    plt.savefig(os.path.join(path_plots, 'param_recovery_box'), dpi=500)
else:
    plt.show()
import os, sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
from copy import copy
from sklearn.linear_model import LinearRegression
from scipy import stats

from spice import Agent, SpiceEstimator, csv_to_dataset
from spice.precoded import workingmemory_rewardbinary, workingmemory, choice

sys.path.append(os.path.abspath(os.path.join(os.path.__file__, '..', '..')))
from weinhardt2025.benchmarking.benchmarking_qlearning import QLearning


save_plots = False
USE_STRUCTURAL_FILTERING = False  # Structural filtering enabled

# spice_model = choice
# base_name_params = 'weinhardt2025/params/synthetic/spice_synthetic_choice_SESSp_IT.pkl'

spice_class = workingmemory_rewardbinary
base_name_params = 'weinhardt2025/params/synthetic/spice_synthetic_SESSp_IT_0.pkl'

random_sampling = [0.25, 0.5, 0.75]
n_sessions = [256]#[16, 32, 64, 128, 256]
iterations = 1
n_runs = 4  # Number of runs per dataset (0, 1, 2, 3)
coef_threshold = 0.05

# -----------------------------------------------------------------------------------------------
# create a mapping of ground truth parameters to library parameters
# -----------------------------------------------------------------------------------------------

parameter_names = ['beta_reward', 'beta_choice', 'alpha_reward', 'alpha_penalty', 'alpha_choice', 'forget_rate']

mapping_variable_names = {
    '1': '1',
    'value_reward_chosen': r'$q_{t}$',
    'value_reward_not_chosen': r'$q_{t}$',
    'value_choice': r'$c_{t}$',
    'reward': r'$r$',
    'reward_chosen_success': r'$r_{c}$',
    'reward_chosen_fail': r'$(1-r_{c})$',
    'choice': r'$c$',
}


# -----------------------------------------------------------------------------------------------
# analysis auxiliary functions
# -----------------------------------------------------------------------------------------------

def identified_params(true_coefs: np.ndarray, recovered_coefs: np.ndarray):
    zero_division_offset = 1e-9
    
    # check which true coefficients are zeros and non-zeros
    non_zero_features = true_coefs != 0
    zero_features = true_coefs == 0
    
    # count all correctly and non-correctly identified parameters
    true_pos = np.sum(recovered_coefs[non_zero_features] != 0)
    false_neg = np.sum(recovered_coefs[non_zero_features] == 0)
    true_neg = np.sum(recovered_coefs[zero_features] == 0)
    false_pos = np.sum(recovered_coefs[zero_features] != 0)
    
    # get identification rates of correctly and non-correctly identified parameters
    true_pos_rel = (true_pos+zero_division_offset) / (np.sum(non_zero_features)+zero_division_offset)
    false_neg_rel = (false_neg+zero_division_offset) / (np.sum(non_zero_features)+zero_division_offset)
    true_neg_rel = (true_neg+zero_division_offset) / (np.sum(zero_features)+zero_division_offset)
    false_pos_rel = (false_pos+zero_division_offset) / (np.sum(zero_features)+zero_division_offset)
    
    return (true_pos, true_neg, false_pos, false_neg), (true_pos_rel, true_neg_rel, false_pos_rel, false_neg_rel)

def n_true_params(true_coefs):
    # Count number of non-zero coefficients in AgentQ-parameters
    # Filter for parameter groups
    # group parameters by being either reward- or choice-based
    # if beta of one group is 0, then the other parameters can be considered also as 0s because they won't have any influence on the result
    # same goes for alphas.
    # Caution: the values will also change for true_coefs outside this function. In this scenario it's fine because these values should be modelled as 0s anyways

    n_params_active = len(parameter_names)
    
    # reward-based parameter group
    if true_coefs['beta_reward'] == 0 or (true_coefs['alpha_reward'] == 0 and true_coefs['alpha_penalty'] == 0):
        n_params_active -= 4  # no beta_reward, alpha_reward, alpha_penalty, forget_rate

    # set alpha_penalty to 0 if alpha_penalty == alpha_reward
    elif true_coefs['alpha_reward'] == true_coefs['alpha_penalty']:
        n_params_active -= 1  # no extra alpha_penalty; only applicable if not first case

    # choice-based parameter group
    if true_coefs['beta_choice'] == 0 or true_coefs['alpha_choice'] == 0:
        n_params_active -= 2  # no beta_choice, alpha_choice
        
    return n_params_active

def compute_structural_mask(spice_agents, n_participants, mapping_libraries):
    """
    Compute structural coefficient mask across multiple runs.

    Based on playground.py filtering logic:
    - Population frequency > 0.5 (present in >50% of participants)
    - Consistency across runs (std < 0.25)
    - Statistical significance (p < 0.05)
    - Effect size (Cohen's d > 0.3)

    Args:
        spice_agents: List of SPICE agents from different runs
        n_participants: Number of participants
        mapping_libraries: Dictionary mapping library names

    Returns:
        Dictionary mapping library names to structural masks
    """
    n_runs = len(spice_agents)

    # Collect coefficients from all runs
    coefs_all_libs = {}

    for library in mapping_libraries:
        coefs_runs = []
        for spice_agent in spice_agents:
            # Get coefficients for this library across all participants
            coefs_library = spice_agent.model.sindy_coefficients[library][:, 0].detach().cpu().numpy()
            presence_library = spice_agent.model.sindy_coefficients_presence[library][:, 0].detach().cpu().numpy()
            coefs_runs.append(coefs_library * presence_library)

        coefs_all_libs[library] = np.stack(coefs_runs)  # Shape: (n_runs, n_participants, n_terms)

    # Compute structural mask for each library
    structural_masks = {}

    for library, coefs_all in coefs_all_libs.items():
        # 1. Population-level frequency (within each run)
        freq_within_run = np.mean(coefs_all != 0, axis=1)  # (n_runs, n_terms)
        mean_pop_freq = np.mean(freq_within_run, axis=0)   # (n_terms,)
        std_pop_freq = np.std(freq_within_run, axis=0)

        # 2. Population-level significance test
        n_terms = coefs_all.shape[2]
        p_values_pop = np.zeros(n_terms)
        for c in range(n_terms):
            coef_values = coefs_all[:, :, c].flatten()
            t_stat, p_val = stats.ttest_1samp(coef_values, 0)
            p_values_pop[c] = p_val

        # 3. Effect size at population level
        mean_magnitude = np.mean(np.abs(coefs_all), axis=(0, 1))
        cohen_d_pop = mean_magnitude / (np.std(coefs_all, axis=(0, 1)) + 1e-9)

        # 4. Identify structural coefficients
        structural_mask = (
            (mean_pop_freq > 0.5) &        # Present in >50% of participants
            (std_pop_freq < 0.25) &         # Consistent across runs
            (p_values_pop < 0.05) &         # Significantly non-zero
            (cohen_d_pop > 0.3)             # Medium to large effect size
        )

        structural_masks[library] = structural_mask

        print(f"\nLibrary: {library}")
        print(f"  Total coefficient positions: {len(structural_mask)}")
        print(f"  Structural coefficients: {np.sum(structural_mask)}")
        print(f"  Structural coefficient indices: {np.where(structural_mask)[0]}")

    return structural_masks


# -----------------------------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------------------------

base_name_data = 'weinhardt2025/data/synthetic/synthetic_SESSp_IT.csv'
kw_participant_id = 'session'
path_plots = 'analysis/plots_parameter_recovery'
n_params_q = len(parameter_names)

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

        # setup of ground truth model from current dataset
        dataset = csv_to_dataset(path_data, additional_inputs=parameter_names)
        n_actions = dataset.ys.shape[-1]
        participant_ids = dataset.xs[:, 0, -1].unique()
        parameters = {key: dataset.xs[:, 0, 2*n_actions+index_key].unique(sorted=False) for index_key, key in enumerate(parameter_names)}
        qlearning = QLearning(
            n_actions=n_actions,
            n_participants=len(participant_ids),
            **parameters,
        )

        # load trained spice model
        path_spice = base_name_params.replace('SESS', str(sess)).replace('IT', str(it)).replace('RUN', str(run))
        spice_esimator = SpiceEstimator(
            rnn_class=spice_class.SpiceModel, 
            spice_config=spice_class.CONFIG, 
            n_actions=n_actions,
            n_participants=len(participant_ids),
            sindy_library_polynomial_degree=2,
            )
        spice_esimator.load_spice(path_spice)
        spice_model = spice_esimator.spice_agent.model

        for index_participant, participant in enumerate(participant_ids):
            
            # get all true parameters of current participant from dataset
            data_coefs_all = data.loc[data[kw_participant_id]==participant].iloc[-1]
            index_params = n_true_params(copy(data_coefs_all)) - 1

            sindy_coefs_array = []
            data_coefs_array = []
            feature_names = []
            
            index_all_candidate_terms = 0
            for index_library, library in enumerate(mapping_libraries):
                # get sindy coefficients
                sindy_coefs_library = spice_agent.model.sindy_coefficients[library][index_participant, 0, 0].detach().cpu().numpy()
                # sindy_coefs_presence_library = spice_agent.model.sindy_coefficients_presence[library][index_participant][0].detach().cpu().numpy()
                # correct for delta-update rule (x[t+1] = x[t] + input*sindy_coefs) in sindy's next state computation (see: spice.resources.rnn.BaseRNN.forward_sindy)
                sindy_coefs_library[1] += 1
                # sindy_coefs_presence_library = np.abs(sindy_coefs_library) > coef_threshold
                sindy_coefs_library *= np.abs(sindy_coefs_library) > coef_threshold
                
                feature_names_library = spice_agent.model.sindy_candidate_terms[library]
                # sindy_coefs_array += (sindy_coefs_library * sindy_coefs_presence_library).tolist()
                sindy_coefs_array += sindy_coefs_library.tolist()
                feature_names += feature_names_library
                
                # translate data coefficient to sindy coefficients
                data_coefs_library = np.zeros_like(sindy_coefs_library)
                for index_feature, feature in enumerate(feature_names_library):
                    data_coefs_library[index_feature] = mapping_libraries[library].get(feature, lambda *args: 0)(*argument_extractor(data_coefs_all, library))
                data_coefs_array += data_coefs_library.tolist()

            sindy_coefs_array = np.array(sindy_coefs_array)
            data_coefs_array = np.array(data_coefs_array)
            data_coefs_array *= np.abs(data_coefs_array) > coef_threshold
            
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
            true_pos_rates[index_sess, it, index_params] += true_pos_rel
            true_neg_rates[index_sess, it, index_params] += true_neg_rel
            false_pos_rates[index_sess, it, index_params] += false_pos_rel
            false_neg_rates[index_sess, it, index_params] += false_neg_rel
            
            # add identification rate counter
            true_pos_rates_count[index_sess, it, index_params] += 1
            true_neg_rates_count[index_sess, it, index_params] += 1
            false_pos_rates_count[index_sess, it, index_params] += 1
            false_neg_rates_count[index_sess, it, index_params] += 1
            
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
# coef_threshold = 1e2
# removed_params = 0
# for index_sess in range(len(n_sessions)):
#     mask_keep = np.all(recovered_params[index_sess] <= coef_threshold, axis=1)

#     # Count the number of removed rows
#     removed_params += np.sum(~mask_keep)
    
#     # Filter out the rows exceeding the threshold
#     true_params[index_sess] = true_params[index_sess][mask_keep]
#     recovered_params[index_sess] = recovered_params[index_sess][mask_keep]

# if removed_params > 0:
#     print(f'excluded parameters because of high values: {removed_params}')

# zero_division_offset = 1e-9
# for index_sess in range(len(n_sessions)):

#     # normalizing params
#     v_max = np.max(true_params[index_sess], axis=0, keepdims=True) + zero_division_offset
#     v_min = np.min(true_params[index_sess], axis=0, keepdims=True)
    
#     true_params[index_sess] = (true_params[index_sess] - v_min) / (v_max - v_min)
#     recovered_params[index_sess] = (recovered_params[index_sess] - v_min) / (v_max - v_min)


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

true_pos_rates_sessions_mean = np.nanmean(true_pos_rates, axis=1)
true_neg_rates_sessions_mean = np.nanmean(true_neg_rates, axis=1)
false_pos_rates_sessions_mean = np.nanmean(false_pos_rates, axis=1)
false_neg_rates_sessions_mean = np.nanmean(false_neg_rates, axis=1)

true_pos_rates_sessions_std = np.nanstd(true_pos_rates, axis=1)
true_neg_rates_sessions_std = np.nanstd(true_neg_rates, axis=1)
false_pos_rates_sessions_std = np.nanstd(false_pos_rates, axis=1)
false_neg_rates_sessions_std = np.nanstd(false_neg_rates, axis=1)

# true_pos_count_sessions_mean = np.nanmean(true_pos_count, axis=1)
# true_neg_count_sessions_mean = np.nanmean(true_neg_count, axis=1)
# false_pos_count_sessions_mean = np.nanmean(false_pos_count, axis=1)
# false_neg_count_sessions_mean = np.nanmean(false_neg_count, axis=1)

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
    [true_pos_rates_sessions_mean , false_pos_rates_sessions_mean], 
    [false_neg_rates_sessions_mean , true_neg_rates_sessions_mean],
]

identification_matrix_std = [
    [true_pos_rates_sessions_std, false_pos_rates_sessions_std], 
    [false_neg_rates_sessions_std, true_neg_rates_sessions_std],
    ]

# identification_matrix_mean = [
#     [true_pos_count, false_pos_count], 
#     [false_neg_count, true_neg_count],
# ]

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

# Print identification rates instead of showing plots
print(f"\n{'='*80}")
print(f"IDENTIFICATION RATES (n_sessions={n_sessions[0]})")
print(f"{'='*80}")
print(f"\nTrue Positive Rates (Recall):")
print(true_pos_rates_sessions_mean)
print(f"\nFalse Positive Rates:")
print(false_pos_rates_sessions_mean)
print(f"\nTrue Negative Rates:")
print(true_neg_rates_sessions_mean)
print(f"\nFalse Negative Rates:")
print(false_neg_rates_sessions_mean)

if save_plots:
    plt.savefig(os.path.join(path_plots, 'ident_rates'), dpi=500)
    plt.close()
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

# Print identification metrics
print(f"\n{'='*80}")
print(f"IDENTIFICATION METRICS (n_sessions={n_sessions[0]})")
print(f"{'='*80}")
print(f"\nAccuracy:")
print(accuracy)
print(f"\nPrecision:")
print(precision)
print(f"\nRecall:")
print(recall)
print(f"\nF1 Score:")
print(f1_score)
print(f"\n{'='*80}\n")

if save_plots:
    plt.savefig(os.path.join(path_plots, 'ident_metrics'), dpi=500)
    plt.close()
else:
    plt.show()

# ------------------------------------------------
# parameter correlation coefficients
# ------------------------------------------------

# box plot

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
    plt.close()
else:
    plt.show()
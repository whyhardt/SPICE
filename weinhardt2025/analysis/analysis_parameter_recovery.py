import numpy as np
import matplotlib.pyplot as plt
import torch

from spice import SpiceEstimator, csv_to_dataset
from spice.precoded import choice, workingmemory, workingmemory_rewardbinary, workingmemory_rewardflags

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from weinhardt2025.benchmarking.benchmarking_qlearning import QLearning


path_data = 'weinhardt2025/data/synthetic/synthetic_PARp_IT_0.csv'
path_model = path_data.replace('data', 'params').replace('/synthetic_', '/spice_synthetic_').replace('.csv', '.pkl')
spice_model = workingmemory_rewardbinary
rl_parameters = ['beta_reward', 'beta_choice', 'alpha_reward', 'alpha_penalty', 'alpha_choice', 'forget_rate']
participants = [32, 64, 128, 256, 512]
iterations = 8
coefficient_threshold = 0.05

# Dynamically determine n_coefficients from a sample model
sample_dataset = csv_to_dataset(
    file=path_data.replace('PAR', str(participants[0])).replace('IT', '0'),
    additional_inputs=rl_parameters
)
sample_model = SpiceEstimator(
    rnn_class=spice_model.SpiceModel,
    spice_config=spice_model.CONFIG,
    n_actions=sample_dataset.ys.shape[-1],
    n_participants=1,
    sindy_library_polynomial_degree=2,
)
n_coefficients_fitted_model = sum(
    sample_model.rnn_model.sindy_coefficients[m].shape[-1]
    for m in sample_model.rnn_model.get_modules()
)
print(f"Detected n_coefficients_fitted_model = {n_coefficients_fitted_model}")

# get coefficients storage
true_coefs = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
fitted_coefs = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
active_params = np.zeros((len(participants), participants[-1]*iterations, 1))

def count_active_params(dict_rl_parameters):
    
    n_participants = dict_rl_parameters[rl_parameters[0]].shape[0]
    n_active_params = torch.zeros((n_participants, 1)) + len(rl_parameters)
    
    n_active_params = torch.where(torch.logical_or(dict_rl_parameters['beta_reward'] == 0, dict_rl_parameters['alpha_reward'] == 0), n_active_params-4, n_active_params)  # if beta_reward == 0: no alpha_reward, alpha_penalty, forget_rate as well
    n_active_params = torch.where(torch.logical_or(dict_rl_parameters['beta_choice'] == 0, dict_rl_parameters['alpha_choice'] == 0), n_active_params-2, n_active_params)  # if beta_choice == 0: no alpha_choice as well
    
    n_active_params = torch.where(dict_rl_parameters['alpha_penalty'] == 0, n_active_params-1, n_active_params)
    n_active_params = torch.where(dict_rl_parameters['forget_rate'] == 0, n_active_params-1, n_active_params)
    
    return n_active_params.reshape(n_participants).numpy()

# -------------------------------------------------------------------------------
# Get true and fitted SINDy coefficients
# -------------------------------------------------------------------------------

for index_par, par in enumerate(participants):
    for it in range(iterations):

        # load dataset and collect true rl parameters
        dataset = csv_to_dataset(file=path_data.replace('PAR', str(par)).replace('IT', str(it)), additional_inputs=rl_parameters)
        n_actions = dataset.ys.shape[-1]
        mask = dataset.xs[:, 0, -3] == 0  # block -> 0; each participant only once
        rl_parameters_dataset = {param: dataset.xs[mask, 0, n_actions*2+index_param].unsqueeze(-1) for index_param, param in enumerate(rl_parameters)}

        # load true model
        true_model = QLearning(
            n_actions=n_actions,
            n_participants=par,
            **rl_parameters_dataset,
        )
        
        # load fitted model
        fitted_model = SpiceEstimator(
            rnn_class=spice_model.SpiceModel,
            spice_config=spice_model.CONFIG,
            n_actions=n_actions,
            n_participants=par,
            sindy_library_polynomial_degree=2,
        )
        fitted_model.load_spice(path_model=path_model.replace('PAR', str(par)).replace('IT', str(it)))
        fitted_model = fitted_model.rnn_model

        # put all coefs into storage
        index_coefs_all = 0
        for module in true_model.get_modules():
            n_terms_module = fitted_model.sindy_coefficients[module].shape[-1]

            # get candidate terms from true model to map into fitted model coef positions
            candidate_terms_fitted_model = fitted_model.sindy_candidate_terms[module]
            candidate_terms_true_model = true_model.sindy_candidate_terms[module]
            for term in candidate_terms_true_model:
                if term not in candidate_terms_fitted_model:
                    raise ValueError(f"Candidate term {term} of the true model was not found among the candidate terms of the fitted model ({candidate_terms_fitted_model}).")

                index_coef = candidate_terms_fitted_model.index(term)
                # Extract coefficient values: shape is (n_participants, n_experiments, n_ensemble, n_terms)
                true_coef_vals = true_model.sindy_coefficients[module][:, 0, 0, candidate_terms_true_model.index(term)].detach().cpu().numpy()
                if module == term:
                    true_coef_vals += 1
                true_coefs[index_par, par*it:par*(it+1), index_coefs_all+index_coef] = true_coef_vals

            # place fitted module coefs in storage (apply presence mask)
            # fitted_coef_vals = (
            #     fitted_model.sindy_coefficients[module][:, 0, 0, :] *
            #     fitted_model.sindy_coefficients_presence[module][:, 0, 0, :]
            # ).detach().cpu().numpy()
            fitted_coef_vals = fitted_model.sindy_coefficients[module][:, 0, 0, :].detach().cpu().numpy()
            fitted_coef_vals[:, 1] += 1
            fitted_coefs[index_par, par*it:par*(it+1), index_coefs_all:index_coefs_all+n_terms_module] = fitted_coef_vals

            index_coefs_all += n_terms_module

        # store number of active params per participant
        active_params[index_par, par*it:par*(it+1), 0] = count_active_params(rl_parameters_dataset)
        
# -------------------------------------------------------------------------------
# POST-PROCESSING: Compute classification metrics
# -------------------------------------------------------------------------------

# Apply threshold to determine active coefficients (use >= for boundary consistency with training)
true_active = np.abs(true_coefs) >= coefficient_threshold
fitted_active = np.abs(fitted_coefs) >= coefficient_threshold

# Get unique active param counts for x-axis
max_active_params = int(np.nanmax(active_params)) + 1
n_param_bins = max_active_params

# Initialize metric storage: (n_participant_sizes, n_active_param_bins)
true_pos_count = np.zeros((len(participants), n_param_bins))
true_neg_count = np.zeros((len(participants), n_param_bins))
false_pos_count = np.zeros((len(participants), n_param_bins))
false_neg_count = np.zeros((len(participants), n_param_bins))
sample_count = np.zeros((len(participants), n_param_bins))

# Compute confusion matrix counts
for index_par in range(len(participants)):
    par = participants[index_par]
    n_samples = par * iterations

    for i in range(n_samples):
        n_active = int(active_params[index_par, i, 0])
        if n_active == 0:
            continue

        true_act = true_active[index_par, i]
        fitted_act = fitted_active[index_par, i]

        # Compute TP, TN, FP, FN for this sample
        tp = np.sum(true_act & fitted_act)
        tn = np.sum(~true_act & ~fitted_act)
        fp = np.sum(~true_act & fitted_act)
        fn = np.sum(true_act & ~fitted_act)

        true_pos_count[index_par, n_active] += tp
        true_neg_count[index_par, n_active] += tn
        false_pos_count[index_par, n_active] += fp
        false_neg_count[index_par, n_active] += fn
        sample_count[index_par, n_active] += 1

# Compute rates (avoid division by zero)
eps = 1e-9
total_count = true_pos_count + true_neg_count + false_pos_count + false_neg_count + eps

true_pos_rate = true_pos_count / (true_pos_count + false_neg_count + eps)
true_neg_rate = true_neg_count / (true_neg_count + false_pos_count + eps)
false_pos_rate = false_pos_count / (false_pos_count + true_neg_count + eps)
false_neg_rate = false_neg_count / (false_neg_count + true_pos_count + eps)

# Compute classification metrics
accuracy = (true_pos_count + true_neg_count) / total_count
precision = true_pos_count / (true_pos_count + false_pos_count + eps)
recall = true_pos_count / (true_pos_count + false_neg_count + eps)
f1_score = 2 * (precision * recall) / (precision + recall + eps)
f2_score = 5 * (precision * recall) / (4 * precision + recall + eps)

# Mask out bins with no samples
mask_no_samples = sample_count == 0
true_pos_rate[mask_no_samples] = np.nan
true_neg_rate[mask_no_samples] = np.nan
false_pos_rate[mask_no_samples] = np.nan
false_neg_rate[mask_no_samples] = np.nan
accuracy[mask_no_samples] = np.nan
precision[mask_no_samples] = np.nan
recall[mask_no_samples] = np.nan
f1_score[mask_no_samples] = np.nan
f2_score[mask_no_samples] = np.nan

print(precision)

# -------------------------------------------------------------------------------
# PLOTTING: 2x2 Confusion Matrix Rates
# -------------------------------------------------------------------------------

import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8),
                        gridspec_kw={'width_ratios': [10, 10, 1]})

confusion_matrices = [
    [true_pos_rate, false_pos_rate],
    [false_neg_rate, true_neg_rate],
]
confusion_titles = [
    ['True Positive Rate', 'False Positive Rate'],
    ['False Negative Rate', 'True Negative Rate'],
]

x_labels = list(range(n_param_bins))
y_labels = participants

for row in range(2):
    for col in range(2):
        sns.heatmap(
            confusion_matrices[row][col],
            annot=True,
            fmt='.2f',
            cmap='viridis',
            ax=axs[row, col],
            cbar=(col == 1),
            cbar_ax=axs[row, 2] if col == 1 else None,
            xticklabels=x_labels if row == 1 else [''] * n_param_bins,
            yticklabels=y_labels if col == 0 else [''] * len(participants),
            vmin=0,
            vmax=1,
            mask=np.isnan(confusion_matrices[row][col]),
        )
        axs[row, col].set_title(confusion_titles[row][col], fontsize=12)
        if row == 1:
            axs[row, col].set_xlabel('Number of Active Parameters', fontsize=10)
        if col == 0:
            axs[row, col].set_ylabel('Number of Participants', fontsize=10)

plt.suptitle('Confusion Matrix Rates', fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------
# PLOTTING: 2x2 Classification Metrics
# -------------------------------------------------------------------------------

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8),
                        gridspec_kw={'width_ratios': [10, 10, 1]})

metrics_matrices = [
    [accuracy, precision],
    [recall, f2_score],
]
metrics_titles = [
    ['Accuracy', 'Precision'],
    ['Recall', 'F2 Score'],
]

for row in range(2):
    for col in range(2):
        sns.heatmap(
            metrics_matrices[row][col],
            annot=True,
            fmt='.2f',
            cmap='viridis',
            ax=axs[row, col],
            cbar=(col == 1),
            cbar_ax=axs[row, 2] if col == 1 else None,
            xticklabels=x_labels if row == 1 else [''] * n_param_bins,
            yticklabels=y_labels if col == 0 else [''] * len(participants),
            vmin=0,
            vmax=1,
            mask=np.isnan(metrics_matrices[row][col]),
        )
        axs[row, col].set_title(metrics_titles[row][col], fontsize=12)
        if row == 1:
            axs[row, col].set_xlabel('Number of Active Parameters', fontsize=10)
        if col == 0:
            axs[row, col].set_ylabel('Number of Participants', fontsize=10)

plt.suptitle('Classification Metrics', fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------
# PLOTTING: Parameter Recovery Box Plots
# -------------------------------------------------------------------------------

# Get candidate term names from a fitted model for labeling
# For now, use generic term indices
n_terms = true_coefs.shape[-1]

# Find which terms have any non-zero true coefficients (active terms)
active_term_mask = np.any(np.abs(true_coefs) > coefficient_threshold, axis=(0, 1))
active_term_indices = np.where(active_term_mask)[0]

if len(active_term_indices) == 0:
    print("No active terms found in true coefficients.")
else:
    # Create one figure per participant size
    for index_par, par in enumerate(participants):
        n_samples = par * iterations

        # Determine grid size for subplots
        n_active_terms = len(active_term_indices)
        n_cols = min(6, n_active_terms)
        n_rows = int(np.ceil(n_active_terms / n_cols))

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))
        if n_active_terms == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs.reshape(1, -1)
        elif n_cols == 1:
            axs = axs.reshape(-1, 1)

        for idx, term_idx in enumerate(active_term_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = axs[row, col]

            # Get true and fitted coefficients for this term
            true_vals = true_coefs[index_par, :n_samples, term_idx]
            fitted_vals = fitted_coefs[index_par, :n_samples, term_idx]

            # Remove NaN values
            valid_mask = ~(np.isnan(true_vals) | np.isnan(fitted_vals))
            true_vals_valid = true_vals[valid_mask]
            fitted_vals_valid = fitted_vals[valid_mask]

            if len(true_vals_valid) == 0:
                ax.set_visible(False)
                continue

            # Bin the data for box plot
            num_bins = 10
            val_min, val_max = true_vals_valid.min(), true_vals_valid.max()
            if val_max - val_min < 1e-6:
                # All values are the same, just scatter plot
                ax.scatter(true_vals_valid, fitted_vals_valid, alpha=0.5, s=10)
            else:
                bin_edges = np.linspace(val_min, val_max, num_bins + 1)
                bins = np.digitize(true_vals_valid, bin_edges) - 1
                bins = np.clip(bins, 0, num_bins - 1)

                box_data = [fitted_vals_valid[bins == i] for i in range(num_bins)]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = (val_max - val_min) / num_bins * 0.8

                # Create box plot
                bp = ax.boxplot(
                    box_data,
                    positions=bin_centers,
                    widths=bin_width,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor='cadetblue', alpha=0.7),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='cadetblue', linestyle='--', alpha=0.5),
                    capprops=dict(color='cadetblue', alpha=0.5),
                )

            # Plot identity line
            plot_min = min(val_min, fitted_vals_valid.min())
            plot_max = max(val_max, fitted_vals_valid.max())
            ax.plot([plot_min, plot_max], [plot_min, plot_max], ':', color='gray', linewidth=1, label='Identity')

            # Linear regression fit
            if len(true_vals_valid) > 1:
                model = LinearRegression()
                model.fit(true_vals_valid.reshape(-1, 1), fitted_vals_valid)
                x_line = np.linspace(plot_min, plot_max, 100)
                y_line = model.predict(x_line.reshape(-1, 1))
                ax.plot(x_line, y_line, '--', color='black', linewidth=1, label='Linear fit')

            ax.set_xlabel('True coefficient', fontsize=8)
            ax.set_ylabel('Fitted coefficient', fontsize=8)
            ax.set_title(f'Term {term_idx}', fontsize=10)
            ax.tick_params(labelsize=7)

        # Hide unused subplots
        for idx in range(n_active_terms, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axs[row, col].set_visible(False)

        plt.suptitle(f'Parameter Recovery (N={par} participants)', fontsize=14)
        plt.tight_layout()
        plt.show()

print("Analysis complete.")

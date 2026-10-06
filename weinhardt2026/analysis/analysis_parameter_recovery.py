import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import torch

from spice import SpiceEstimator, csv_to_dataset
from weinhardt2026.studies.synthetic.benchmarking_qlearning import QLearning

from spice.precoded.choice import SpiceModel, CONFIG
# from spice.precoded.workingmemory import SpiceModel, CONFIG
# from weinhardt2026.studies.dezfouli2019.spice_dezfouli2019 import SpiceModel, CONFIG


path_data = 'weinhardt2026/studies/synthetic/data/synthetic_balanced_PARp_IT_0.csv'
path_model = 'weinhardt2026/studies/synthetic/params/spice_synthetic_balanced_256p_0_0_al0.001_gp0.001_fp0.0001_hardsigmoid.pkl'

rl_parameters = ['beta_reward', 'beta_choice', 'alpha_reward', 'alpha_penalty', 'alpha_choice', 'forget_rate']
participants = [256]#[32, 64, 128, 256, 512]
iterations = 1
coefficient_threshold = 0.01
ensemble_size = 10

# Compare the choice trace in the minimum-norm gauge: the ground-truth choice values in [0, beta_choice] are
# shifted to [-beta_choice/2, beta_choice/2]. A common shift of both options' values is behaviorally invisible
# (equal update rates in the chosen and not-chosen module), and the L2 penalty on the RNN weights prefers the
# centered representation: constant +alpha*beta/2 in the chosen module and -alpha*beta/2 in the not-chosen
# module instead of +alpha*beta and 0. Reward values stay in their native gauge (anchored by the reward input).
center_choice_gauge = True

# term collapsing in fitted model from term tuple[1] -> tuple[0]; used for binary signals where signal^1=signal^2, e.g. binary reward or choice
term_collapsing = (
    ('reward', 'reward^2'),
    ('reward[t]', 'reward[t]^2'),
    ('reward[t-1]', 'reward[t-1]^2'),
    ('reward[t-2]', 'reward[t-2]^2'),
    ('reward[t-3]', 'reward[t-3]^2'),
    ('choice[t]', 'choice[t]^2'),
    ('choice[t-1]', 'choice[t-1]^2'),
    ('choice[t-2]', 'choice[t-2]^2'),
    ('choice[t-3]', 'choice[t-3]^2'),
)

# signal names that differ between the ground-truth (QLearning) and the fitted library: ground truth -> fitted
signal_aliases = {'reward[t]': 'reward'}

def translate_term(term, fitted_terms):
    """Name of a ground-truth term in the fitted library (identity if it exists there)."""
    if term in fitted_terms:
        return term
    for name_true, name_fitted in signal_aliases.items():
        term = term.replace(name_true, name_fitted)
    return term

# Dynamically determine n_coefficients from a sample model
sample_dataset = csv_to_dataset(
    file=path_data.replace('PAR', str(participants[0])).replace('IT', '0'),
    additional_inputs=rl_parameters
)
sample_dataset.normalize_rewards()
sample_model = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=sample_dataset.ys.shape[-1],
    n_participants=1,
    sindy_library_polynomial_degree=2,
)
n_coefficients_fitted_model = sum(
    sample_model.model.sindy_coefficients[m].shape[-1]
    for m in sample_model.model.get_modules()
)
print(f"Detected n_coefficients_fitted_model = {n_coefficients_fitted_model}")

# get coefficients storage
true_coefs = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
fitted_coefs = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
active_params = np.zeros((len(participants), participants[-1]*iterations), dtype=int)
active_mechanisms = np.full((len(participants), participants[-1]*iterations), None, dtype=object)

# number of parameters each mechanism adds to the ground-truth model
mechanism_n_params = {'Reward': 2, 'Asymmetry': 1, 'Forgetting': 1, 'Choice': 2}

# ground-truth mechanism a module requires; without it the module has no (identifiable) dynamics:
# value_reward never enters the logits without Reward, value_reward_not_chosen only changes through Forgetting,
# value_choice never enters the logits (or stays 0) without Choice
module_mechanism = {
    'value_reward_chosen': 'Reward',
    'value_reward_not_chosen': 'Forgetting',
    'value_choice_chosen': 'Choice',
    'value_choice_not_chosen': 'Choice',
}

def center_choice_trace(true_model, chosen='value_choice_chosen', not_chosen='value_choice_not_chosen'):
    """Shift the ground-truth choice trace by half its value range (in place, delta-form coefficients).

    Chosen module: dv = c - a*v with value range [0, c/a]; shifting all values by b = c/(2a) gives
    dv' = (c - a*b) - a*v' in the chosen and dv' = -a_nc*b - a_nc*v' in the not-chosen module.
    """
    coefs = true_model.sindy_coefficients
    terms_ch, terms_nc = true_model.sindy_candidate_terms[chosen], true_model.sindy_candidate_terms[not_chosen]
    with torch.no_grad():
        c = coefs[chosen][..., terms_ch.index('1')]
        a = -coefs[chosen][..., terms_ch.index(chosen)]
        a_nc = -coefs[not_chosen][..., terms_nc.index(not_chosen)]
        b = torch.where(a > 0, c / (2 * a.clamp(min=1e-12)), torch.zeros_like(c))
        coefs[chosen][..., terms_ch.index('1')] = c - a * b
        coefs[not_chosen][..., terms_nc.index('1')] = coefs[not_chosen][..., terms_nc.index('1')] - a_nc * b

def get_mechanism_masks(dict_rl_parameters):
    """Boolean array (n_participants,) per mechanism: is it active in the ground-truth model?

    alpha_penalty is never zero in the synthetic data; asymmetry means alpha_penalty != alpha_reward.
    """

    reward = (dict_rl_parameters['beta_reward'] != 0) & (dict_rl_parameters['alpha_reward'] != 0)
    choice = (dict_rl_parameters['beta_choice'] != 0) & (dict_rl_parameters['alpha_choice'] != 0)
    masks = {
        'Reward': reward,
        'Asymmetry': reward & (dict_rl_parameters['alpha_penalty'] != dict_rl_parameters['alpha_reward']),
        'Forgetting': reward & (dict_rl_parameters['forget_rate'] != 0),
        'Choice': choice,
    }
    return {name: mask.reshape(-1).cpu().numpy() for name, mask in masks.items()}

def get_active_mechanisms(mechanism_masks):
    """Label each participant's ground-truth model by its active mechanisms.

    Returns a list of labels and an array of parameter counts, one entry per participant.
    """

    labels, n_params_total = [], []
    for index_participant in range(len(mechanism_masks['Reward'])):
        active = [(name, mechanism_n_params[name]) for name, is_active in mechanism_masks.items() if is_active[index_participant]]
        labels.append('\n'.join(name for name, _ in active) if active else 'None')
        n_params_total.append(sum(n_params for _, n_params in active))
    return labels, np.array(n_params_total)

# -------------------------------------------------------------------------------
# Get true and fitted SINDy coefficients
# -------------------------------------------------------------------------------

for index_par, par in enumerate(participants):
    for it in range(iterations):
        
        # load dataset and collect true rl parameters
        dataset = csv_to_dataset(file=path_data.replace('PAR', str(par)).replace('IT', str(it)), additional_inputs=rl_parameters)
        dataset.normalize_rewards()
        n_actions = dataset.ys.shape[-1]
        mask = dataset.xs[:, 0, 0, -3] == 0  # block -> 0; each participant only once
        rl_parameters_dataset = {param: dataset.xs[mask, 0, 0, n_actions*2+index_param].unsqueeze(-1) for index_param, param in enumerate(rl_parameters)}
        mechanism_masks = get_mechanism_masks(rl_parameters_dataset)

        # load true model
        true_model = QLearning(
            n_actions=n_actions,
            n_participants=par,
            **rl_parameters_dataset,
        )
        if center_choice_gauge:
            center_choice_trace(true_model)
        
        # load fitted model
        fitted_model = SpiceEstimator(
            spice_class=SpiceModel,
            spice_config=CONFIG,
            n_actions=n_actions,
            n_participants=par,
            sindy_library_polynomial_degree=2,
            ensemble_size=ensemble_size,
        )
        fitted_model.load_spice(path_model=path_model.replace('PAR', str(par)).replace('IT', str(it)))
        fitted_model = fitted_model.model
        fitted_coef_vals = fitted_model.get_sindy_coefficients(aggregate=True)

        # put all coefs into storage
        index_coefs_all = 0
        for module in fitted_model.get_modules():
            n_terms_module = fitted_model.sindy_coefficients[module].shape[-1]

            # get candidate terms from true model to map into fitted model coef positions
            candidate_terms_fitted_model = fitted_model.sindy_candidate_terms[module]
            # modules absent from the true model (e.g. working memory) keep true coefficients at 0
            module_in_true_model = module in true_model.sindy_candidate_terms
            candidate_terms_true_model = true_model.sindy_candidate_terms[module] if module_in_true_model else []
            # participants whose ground truth contains the module (requires its mechanism); others are treated like absent modules
            # all coefficients are compared in delta form (v[t+1] = v[t] + delta): 0 = term inactive
            module_present = mechanism_masks[module_mechanism[module]] if module in module_mechanism else np.full(par, module_in_true_model)
            for term in candidate_terms_true_model:
                # Extract coefficient values: shape is (n_ensemble, n_participants, n_experiments, n_terms)
                true_coef_vals = true_model.sindy_coefficients[module][0, :, 0, candidate_terms_true_model.index(term)].detach().cpu().numpy()
                term_fitted = translate_term(term, candidate_terms_fitted_model)
                if term_fitted not in candidate_terms_fitted_model:
                    if np.any(true_coef_vals != 0):
                        raise ValueError(f"Candidate term {term} of the true model was not found among the candidate terms of the fitted model ({candidate_terms_fitted_model}).")
                    continue  # term inactive for everyone in the true model and absent from the fitted library

                index_coef = candidate_terms_fitted_model.index(term_fitted)
                true_coefs[index_par, par*it:par*(it+1), index_coefs_all+index_coef] = true_coef_vals * module_present

            # term collapsing
            for term in term_collapsing:
                index_target_term = candidate_terms_fitted_model.index(term[0]) if term[0] in candidate_terms_fitted_model else None
                index_source_term = candidate_terms_fitted_model.index(term[1]) if term[1] in candidate_terms_fitted_model else None
                if index_source_term is not None and index_target_term is not None:
                    fitted_coef_vals[module][..., index_target_term] += fitted_coef_vals[module][..., index_source_term]
                    fitted_coef_vals[module][..., index_source_term] = 0
            
            # place fitted module coefs in storage (apply presence mask)
            # fitted_coef_vals = (
            #     fitted_model.sindy_coefficients[module][0, :, 0, :] *
            #     fitted_model.sindy_coefficients_presence[module][0, :, 0, :]
            # ).detach().cpu().numpy()
            # fitted_coef_vals = (fitted_model.sindy_coefficients[module][:, :, 0, :].median(dim=0)[0] * fitted_model.sindy_coefficients_presence[module][:, :, 0, :].float().median(dim=0)[0]).detach().cpu().numpy()
            fitted_coefs[index_par, par*it:par*(it+1), index_coefs_all:index_coefs_all+n_terms_module] = fitted_coef_vals[module][:, 0]
            
            index_coefs_all += n_terms_module

        # store active mechanisms and number of active params per participant
        labels_mechanisms, n_params_mechanisms = get_active_mechanisms(mechanism_masks)
        active_mechanisms[index_par, par*it:par*(it+1)] = labels_mechanisms
        active_params[index_par, par*it:par*(it+1)] = n_params_mechanisms

# -------------------------------------------------------------------------------
# POST-PROCESSING: Compute classification metrics
# -------------------------------------------------------------------------------

# Apply threshold to determine active coefficients (use >= for boundary consistency with training)
true_active = np.abs(true_coefs) >= coefficient_threshold
fitted_active = np.abs(fitted_coefs) >= coefficient_threshold

# Get unique active param counts for x-axis
max_active_params = int(np.max(active_params)) + 1
n_param_bins = max_active_params

def compute_confusion_counts(group_index, n_groups):
    """Sum TP/TN/FP/FN over all coefficients per (participant size, group).

    group_index: int array (n_participant_sizes, n_samples) assigning each sample to a group; -1 = skip.
    """

    counts = {key: np.zeros((len(participants), n_groups)) for key in ('tp', 'tn', 'fp', 'fn', 'samples')}
    for index_par, par in enumerate(participants):
        for i in range(par * iterations):
            group = group_index[index_par, i]
            if group < 0:
                continue

            true_act = true_active[index_par, i]
            fitted_act = fitted_active[index_par, i]

            counts['tp'][index_par, group] += np.sum(true_act & fitted_act)
            counts['tn'][index_par, group] += np.sum(~true_act & ~fitted_act)
            counts['fp'][index_par, group] += np.sum(~true_act & fitted_act)
            counts['fn'][index_par, group] += np.sum(true_act & ~fitted_act)
            counts['samples'][index_par, group] += 1
    return counts

def compute_classification_metrics(counts, eps=1e-9):
    """Confusion rates and classification metrics from counts; groups without samples are NaN."""

    tp, tn, fp, fn = counts['tp'], counts['tn'], counts['fp'], counts['fn']
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    metrics = {
        'true_pos_rate': recall,
        'true_neg_rate': tn / (tn + fp + eps),
        'false_pos_rate': fp / (fp + tn + eps),
        'false_neg_rate': fn / (fn + tp + eps),
        'accuracy': (tp + tn) / (tp + tn + fp + fn + eps),
        'precision': precision,
        'recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall + eps),
        'f2_score': 5 * (precision * recall) / (4 * precision + recall + eps),
    }
    for metric in metrics.values():
        metric[counts['samples'] == 0] = np.nan
    return metrics

# Metrics binned by number of active parameters (0 active parameters are skipped)
group_index_params = active_params.copy()
group_index_params[group_index_params == 0] = -1
metrics_params = compute_classification_metrics(compute_confusion_counts(group_index_params, n_param_bins))

true_pos_rate = metrics_params['true_pos_rate']
true_neg_rate = metrics_params['true_neg_rate']
false_pos_rate = metrics_params['false_pos_rate']
false_neg_rate = metrics_params['false_neg_rate']
accuracy = metrics_params['accuracy']
precision = metrics_params['precision']
recall = metrics_params['recall']
f1_score = metrics_params['f1_score']
f2_score = metrics_params['f2_score']

# Metrics binned by ground-truth model type (active mechanisms), sorted by number of parameters
mechanism_types = sorted(
    {(n_params, label) for n_params, label in zip(active_params.ravel(), active_mechanisms.ravel()) if label is not None}
)
mechanism_labels = [label for _, label in mechanism_types]
mechanism_label_to_index = {label: index for index, label in enumerate(mechanism_labels)}
group_index_mechanisms = np.vectorize(lambda label: mechanism_label_to_index.get(label, -1), otypes=[int])(active_mechanisms)
counts_mechanisms = compute_confusion_counts(group_index_mechanisms, len(mechanism_labels))
metrics_mechanisms = compute_classification_metrics(counts_mechanisms)

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
# PLOTTING: 2x2 Classification Metrics per ground-truth model type
# -------------------------------------------------------------------------------

n_mechanism_types = len(mechanism_labels)
mechanism_n_samples = counts_mechanisms['samples'].sum(axis=0).astype(int)
x_labels_mechanisms = [
    f"{label}\n(k={n_params}, n={n_samples})"
    for (n_params, label), n_samples in zip(mechanism_types, mechanism_n_samples)
]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(max(12, 1.6 * n_mechanism_types), 9),
                        gridspec_kw={'width_ratios': [10, 10, 1]})

metrics_matrices = [
    [metrics_mechanisms['accuracy'], metrics_mechanisms['precision']],
    [metrics_mechanisms['recall'], metrics_mechanisms['f1_score']],
]
metrics_titles = [
    ['Accuracy', 'Precision'],
    ['Recall', 'F1 Score'],
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
            xticklabels=x_labels_mechanisms if row == 1 else [''] * n_mechanism_types,
            yticklabels=y_labels if col == 0 else [''] * len(participants),
            vmin=0,
            vmax=1,
            mask=np.isnan(metrics_matrices[row][col]),
        )
        axs[row, col].set_title(metrics_titles[row][col], fontsize=12)
        if row == 1:
            axs[row, col].set_xlabel('Ground-Truth Mechanisms (k = parameters, n = participants)', fontsize=10)
            axs[row, col].tick_params(axis='x', labelsize=7, rotation=0)
        if col == 0:
            axs[row, col].set_ylabel('Number of Participants', fontsize=10)

plt.suptitle('Classification Metrics by Ground-Truth Model', fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------
# PLOTTING: Per-term recovery (ground-truth terms) and inclusion (non-ground-truth terms)
# -------------------------------------------------------------------------------

# Build flat list of term names matching coefficient storage layout (module by module)
term_names = []
term_collapsed = []  # source terms of term collapsing are zero by construction
collapsed_source_terms = {source for _, source in term_collapsing}
for module in sample_model.model.get_modules():
    for term in sample_model.model.sindy_candidate_terms[module]:
        term_names.append(f"{module}: {term}")
        term_collapsed.append(term in collapsed_source_terms)
n_terms = true_coefs.shape[-1]
term_indices_plot = [index for index in range(n_terms) if not term_collapsed[index]]

# per-term outcome counts over participants: (n_participant_sizes, n_terms_plot)
term_counts = {key: np.zeros((len(participants), len(term_indices_plot)), dtype=int) for key in ('tp', 'fn', 'fp', 'tn')}
for index_par, par in enumerate(participants):
    n_samples = par * iterations
    valid_samples = active_mechanisms[index_par, :n_samples] != None
    true_act = true_active[index_par, :n_samples][valid_samples][:, term_indices_plot]
    fitted_act = fitted_active[index_par, :n_samples][valid_samples][:, term_indices_plot]

    term_counts['tp'][index_par] = (true_act & fitted_act).sum(axis=0)
    term_counts['fn'][index_par] = (true_act & ~fitted_act).sum(axis=0)
    term_counts['fp'][index_par] = (~true_act & fitted_act).sum(axis=0)
    term_counts['tn'][index_par] = (~true_act & ~fitted_act).sum(axis=0)

# recovery rate: P(fitted active | term in ground truth); inclusion rate: P(fitted active | term not in ground truth)
n_term_true = term_counts['tp'] + term_counts['fn']
n_term_false = term_counts['fp'] + term_counts['tn']
with np.errstate(invalid='ignore', divide='ignore'):
    recovery_rate = np.where(n_term_true > 0, term_counts['tp'] / n_term_true, np.nan)
    inclusion_rate = np.where(n_term_false > 0, term_counts['fp'] / n_term_false, np.nan)

# short tick labels (module's own state written as 'v') grouped under module headers
term_modules_plot, term_labels_plot = [], []
for index in term_indices_plot:
    module, term = term_names[index].split(': ')
    term_modules_plot.append(module)
    term_labels_plot.append(term.replace(module, 'v'))
module_groups = []  # (module, first index, last index + 1) in plotting order
for index, module in enumerate(term_modules_plot):
    if module_groups and module_groups[-1][0] == module:
        module_groups[-1] = (module, module_groups[-1][1], index + 1)
    else:
        module_groups.append((module, index, index + 1))

def draw_module_groups(axs_column, offset):
    """Separate module groups by vertical lines and label them on top of the first axis.

    offset: x position of the left edge of term 0 (heatmap: 0, bars centered on integers: -0.5).
    """
    for ax in axs_column:
        for _, start, _ in module_groups[1:]:
            ax.axvline(start + offset, color='dimgray', linewidth=1)
    top_axis = axs_column[0].secondary_xaxis('top')
    top_axis.set_xticks([offset + (start + end) / 2 for _, start, end in module_groups])
    top_axis.set_xticklabels([module for module, _, _ in module_groups], fontsize=8)
    top_axis.tick_params(length=0, pad=6)
    top_axis.spines['top'].set_visible(False)

fig, axs = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(max(10, 0.6 * len(term_indices_plot) + 3), 2 + 1.6 * len(participants)),
                        gridspec_kw={'width_ratios': [30, 1]})

term_panels = (
    ('Recovered | term in ground truth (grey: never in ground truth)', recovery_rate),
    ('Included | term not in ground truth (grey: always in ground truth)', inclusion_rate),
)

for row, (title, rates) in enumerate(term_panels):
    axs[row, 0].set_facecolor('lightgrey')  # masked (NaN) tiles show the background
    sns.heatmap(
        rates,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        ax=axs[row, 0],
        cbar_ax=axs[row, 1],
        xticklabels=term_labels_plot,
        yticklabels=participants,
        vmin=0,
        vmax=1,
        mask=np.isnan(rates),
    )
    axs[row, 0].set_title(title, fontsize=11, pad=22 if row == 0 else 6)
fig.supylabel('Number of Participants', fontsize=10)
axs[0, 0].tick_params(axis='x', bottom=False, labelbottom=False)
axs[1, 0].tick_params(axis='x', labelsize=8, rotation=0)
axs[1, 0].set_xlabel("Candidate Term (v = module's own state)", fontsize=10)
draw_module_groups(axs[:, 0], offset=0)

plt.suptitle('Per-Term Recovery', fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------
# PLOTTING: Parameter Recovery Box Plots
# -------------------------------------------------------------------------------

# Find which terms have any non-zero true coefficients (active terms)
active_term_mask = np.any(np.abs(true_coefs) > coefficient_threshold, axis=(0, 1))
active_term_indices = np.where(active_term_mask)[0]

if len(active_term_indices) == 0:
    print("No active terms found in true coefficients.")
else:
    for index_par, par in enumerate(participants):
        n_samples = par * iterations
        n_active_terms = len(active_term_indices)
        n_cols = min(6, n_active_terms)
        n_rows = int(np.ceil(n_active_terms / n_cols))

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
        if n_active_terms == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs.reshape(1, -1)
        elif n_cols == 1:
            axs = axs.reshape(-1, 1)

        for idx, term_idx in enumerate(active_term_indices):
            ax = axs[idx // n_cols, idx % n_cols]

            true_vals = true_coefs[index_par, :n_samples, term_idx]
            fitted_vals = fitted_coefs[index_par, :n_samples, term_idx]
            valid_mask = ~(np.isnan(true_vals) | np.isnan(fitted_vals))
            true_v = true_vals[valid_mask]
            fitted_v = fitted_vals[valid_mask]

            if len(true_v) == 0:
                ax.set_visible(False)
                continue

            # Shared axis range
            axis_min = min(true_v.min(), fitted_v.min())
            axis_max = max(true_v.max(), fitted_v.max())
            axis_pad = (axis_max - axis_min) * 0.05
            axis_min -= axis_pad
            axis_max += axis_pad

            # Identity line (behind everything)
            ax.plot([axis_min, axis_max], [axis_min, axis_max], '-', color='#cccccc', linewidth=1, zorder=1)

            # Box plot binned by true values
            num_bins = 8
            true_range = true_v.max() - true_v.min()
            if true_range < 1e-6:
                ax.scatter(true_v, fitted_v, alpha=0.4, s=8, color='cadetblue', zorder=3)
            else:
                bin_edges = np.linspace(true_v.min(), true_v.max(), num_bins + 1)
                bins = np.clip(np.digitize(true_v, bin_edges) - 1, 0, num_bins - 1)
                box_data = [fitted_v[bins == i] for i in range(num_bins)]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = true_range / num_bins * 0.75

                ax.boxplot(
                    box_data,
                    positions=bin_centers,
                    widths=bin_width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    boxprops=dict(facecolor='cadetblue', alpha=0.6, linewidth=0.5),
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(color='gray', linewidth=0.7),
                    capprops=dict(color='gray', linewidth=0.7),
                    zorder=2,
                )

            # Linear regression
            if len(true_v) > 1 and true_range > 1e-6:
                reg = LinearRegression().fit(true_v.reshape(-1, 1), fitted_v)
                x_fit = np.array([axis_min, axis_max])
                ax.plot(x_fit, reg.predict(x_fit.reshape(-1, 1)), '--', color='black', linewidth=1, zorder=4)

            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(term_names[term_idx], fontsize=9)
            ax.tick_params(labelsize=7)

            # Only label outer axes
            if idx // n_cols == n_rows - 1 or idx == n_active_terms - 1:
                ax.set_xlabel('True', fontsize=8)
            if idx % n_cols == 0:
                ax.set_ylabel('Fitted', fontsize=8)

        # Hide unused subplots
        for idx in range(n_active_terms, n_rows * n_cols):
            axs[idx // n_cols, idx % n_cols].set_visible(False)

        plt.suptitle(f'Parameter Recovery (N={par})', fontsize=13)
        plt.tight_layout()
        plt.show()

print("Analysis complete.")

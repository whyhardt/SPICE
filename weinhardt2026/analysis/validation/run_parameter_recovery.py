import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from spice import SpiceEstimator, csv_to_dataset
from weinhardt2026.studies.synthetic.benchmarking_qlearning import QLearning
from weinhardt2026.analysis.validation.run_spice_recovery import term_library_rms

from spice.precoded.choice import SpiceModel, CONFIG
# from spice.precoded.workingmemory import SpiceModel, CONFIG
# from weinhardt2026.studies.dezfouli2019.spice_dezfouli2019 import SpiceModel, CONFIG

import matplotlib as mpl
from spice.utils.plotting import EMPTY_COLOR, MODEL_COLORS, rate_cmap
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['svg.fonttype'] = 'none'  # keep text as editable text in vector exports
mpl.rcParams['pdf.fonttype'] = 42

path_data = 'weinhardt2026/studies/synthetic/data/synthetic_balanced_PARp_IT_0.csv'
path_model = 'weinhardt2026/studies/synthetic/params/spice_synthetic_balanced_PARp_IT_0_probs_th0.05_sw0_al0.0005_er0.5_gp0.01_fp0.0001_rsSEED.pkl'

rl_parameters = ['beta_reward', 'beta_choice', 'alpha_reward', 'alpha_penalty', 'alpha_choice', 'forget_rate']
participants = [32, 64, 128, 256, 512]
iterations = 8
ensemble_size = 10
# training seeds per dataset (SEED in path_model); a term counts as recovered only if it is active in more than half of them.
# Within a seed a term is active if the majority of the ensemble members carry it (presence masks, no coefficient cutoff),
# as in the SPICE -> SPICE recovery (run_spice_recovery.py)
seeds = list(range(10))

# figures are saved twice: annotated and as '_clean' copy without annotations (titles, axis labels, tick labels, numbers inside heatmaps)
path_results = 'weinhardt2026/results/parameter_recovery'
figure_prefix = os.path.splitext(os.path.basename(path_model))[0].replace('_rsSEED', f'_{len(seeds)}seeds')

# term collapsing in fitted model from term tuple[1] -> tuple[0]; used for binary signals where signal^1=signal^2, e.g. binary reward or choice
term_collapsing = (
    ('reward', 'reward^2'),
    ('choice', 'choice^2'),
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
signal_aliases = {'reward[t]': 'reward', 'choice[t]': 'choice'}

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

# module and term of every storage index; source terms of term collapsing are zero by construction
term_modules_storage, term_labels_storage, term_collapsed_storage = [], [], []
for module in sample_model.model.get_modules():
    for term in sample_model.model.sindy_candidate_terms[module]:
        term_modules_storage.append(module)
        term_labels_storage.append(term)
        term_collapsed_storage.append(term in {source for _, source in term_collapsing})

# get coefficients storage
true_effects = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
cells_datasets = []  # signed participant x term cells per dataset (written to <prefix>_cells.csv)
true_coefs = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
fitted_coefs = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model))
fitted_active = np.zeros((len(participants), participants[-1]*iterations, n_coefficients_fitted_model), dtype=bool)
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
    'value_choice': 'Choice',
}

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

def collapse_terms(values_module, candidate_terms_fitted_model, combine):
    """Merge each source term of term_collapsing into its target term (combine(target, source)); the source becomes empty."""

    for term in term_collapsing:
        index_target_term = candidate_terms_fitted_model.index(term[0]) if term[0] in candidate_terms_fitted_model else None
        index_source_term = candidate_terms_fitted_model.index(term[1]) if term[1] in candidate_terms_fitted_model else None
        if index_source_term is not None and index_target_term is not None:
            values_module[..., index_target_term] = combine(values_module[..., index_target_term], values_module[..., index_source_term])
            values_module[..., index_source_term] = 0
    return values_module

def get_fitted_coefficients(fitted_model):
    """Ensemble-aggregated, term-collapsed fitted coefficients (n_participants, n_coefficients) in storage layout."""

    fitted_coef_vals = fitted_model.get_sindy_coefficients(aggregate=True)
    coefs_modules = []
    for module in fitted_model.get_modules():
        coefs_module = fitted_coef_vals[module][:, 0].detach().cpu().numpy().copy()
        coefs_modules.append(collapse_terms(coefs_module, fitted_model.sindy_candidate_terms[module], np.add))
    return np.concatenate(coefs_modules, axis=-1)

def get_fitted_presence(fitted_model):
    """Term-collapsed presence (n_participants, n_coefficients): carried by the majority of the ensemble members."""

    presence_modules = []
    for module in fitted_model.get_modules():
        presence_module = (fitted_model.sindy_coefficients_presence[module].float().mean(dim=0)[:, 0] > 0.5).cpu().numpy()
        presence_modules.append(collapse_terms(presence_module, fitted_model.sindy_candidate_terms[module], np.logical_or))
    return np.concatenate(presence_modules, axis=-1)

def get_fitted_rms(fitted_model, inputs):
    """Ensemble-mean, term-collapsed RMS of the library values along the fitted SPICE-EQ trajectories (n_participants, n_coefficients).
    Collapsed source terms are binary duplicates of their target, so the target keeps its own RMS."""

    rms = term_library_rms(fitted_model, inputs)
    rms_modules = []
    for module in fitted_model.get_modules():
        n_terms_module = fitted_model.sindy_coefficients[module].shape[-1]
        rms_module = rms[module].mean(axis=0) if module in rms else np.zeros((fitted_model.n_participants, n_terms_module))
        rms_modules.append(collapse_terms(rms_module.copy(), fitted_model.sindy_candidate_terms[module], lambda target, source: target))
    return np.concatenate(rms_modules, axis=-1)

def signed_cells(true_block, true_effect_block, coefs_seeds, presence_seeds, effects_seeds):
    """Participant x signed term cells of one dataset that the ground truth has or the seeds' vote recovers
    (`term (+)` and `term (-)` are different terms, as in run_spice_recovery.py)."""

    rows = []
    for sign, label in ((1, '+'), (-1, '-')):
        true = np.sign(true_block) == sign
        present = presence_seeds & (np.sign(coefs_seeds) == sign)  # (S, P, C)
        n_seeds = present.sum(axis=0)
        found = n_seeds > len(coefs_seeds) / 2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            coefficient_found = np.nan_to_num(np.nanmedian(np.where(present, coefs_seeds, np.nan), axis=0))
        effect_found = (effects_seeds * present).sum(axis=0) / np.maximum(n_seeds, 1)
        participants_cells, terms_cells = np.nonzero((true | found) & ~np.array(term_collapsed_storage))
        rows.append(pd.DataFrame(dict(
            module=[term_modules_storage[t] for t in terms_cells],
            term=[f'{term_labels_storage[t]} ({label})' for t in terms_cells], participant=participants_cells,
            true=true[participants_cells, terms_cells], found=found[participants_cells, terms_cells],
            coefficient_true=true_block[participants_cells, terms_cells] * true[participants_cells, terms_cells],
            coefficient_found=coefficient_found[participants_cells, terms_cells] * found[participants_cells, terms_cells],
            effect_true=true_effect_block[participants_cells, terms_cells], effect_found=effect_found[participants_cells, terms_cells],
        )))
    return pd.concat(rows, ignore_index=True)

def aggregate_seeds(coefs_seeds, presence_seeds):
    """Majority vote over seeds (n_seeds, ...): a term is active if active in more than half the seeds;
    its value is the median over the seeds in which it is active (0 otherwise). Returns (values, active)."""

    active_majority = presence_seeds.sum(axis=0) > len(coefs_seeds) / 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN slices: term inactive in every seed
        median_active = np.nanmedian(np.where(presence_seeds, coefs_seeds, np.nan), axis=0)
    return np.where(active_majority, np.nan_to_num(median_active), 0.), active_majority

for index_par, par in enumerate(participants):
    for it in range(iterations):

        path_data_replaced = path_data.replace('PAR', str(par)).replace('IT', str(it))
        paths_model_replaced = [path_model.replace('PAR', str(par)).replace('IT', str(it)).replace('SEED', str(seed)) for seed in seeds]

        missing = [path for path in paths_model_replaced if not os.path.isfile(path)]
        if missing:
            print("FileNotFoundWarning: No such file(s):", *missing, sep='\n  ')
            print("Skipping dataset and continuing with next one.")
            continue

        # load dataset and collect true rl parameters
        dataset = csv_to_dataset(file=path_data_replaced, additional_inputs=rl_parameters)
        dataset.normalize_rewards()
        n_actions = dataset.ys.shape[-1]
        mask = dataset.xs[:, 0, 0, -3] == 0  # block -> 0; each participant only once
        rl_parameters_dataset = {param: dataset.xs[mask, 0, 0, n_actions*2+index_param].unsqueeze(-1) for index_param, param in enumerate(rl_parameters)}
        mechanism_masks = get_mechanism_masks(rl_parameters_dataset)

        # inputs without the generating parameters, for the trajectories behind the effect sizes
        dataset_inputs = csv_to_dataset(file=path_data_replaced)
        dataset_inputs.normalize_rewards()
        inputs = dataset_inputs.xs
        n_trials_dataset = int((~inputs[:, :, :, 0].isnan().all(dim=2)).sum())

        # load true model
        true_model = QLearning(
            n_actions=n_actions,
            n_participants=par,
            **rl_parameters_dataset,
        )

        rms_true = term_library_rms(true_model, inputs)  # effect size = |coefficient| x RMS of the term's library value

        # load fitted models (one per seed)
        fitted_coefs_seeds, fitted_presence_seeds, fitted_effects_seeds = [], [], []
        for path_model_replaced in paths_model_replaced:
            fitted_model = SpiceEstimator(
                spice_class=SpiceModel,
                spice_config=CONFIG,
                n_actions=n_actions,
                n_participants=par,
                sindy_library_polynomial_degree=2,
                ensemble_size=ensemble_size,
            )
            fitted_model.load_spice(path_model=path_model_replaced)
            fitted_model = fitted_model.model
            fitted_coefs_seeds.append(get_fitted_coefficients(fitted_model))
            fitted_presence_seeds.append(get_fitted_presence(fitted_model))
            fitted_effects_seeds.append(np.abs(fitted_coefs_seeds[-1]) * get_fitted_rms(fitted_model, inputs))
        fitted_coefs[index_par, par*it:par*(it+1)], fitted_active[index_par, par*it:par*(it+1)] = aggregate_seeds(
            np.stack(fitted_coefs_seeds), np.stack(fitted_presence_seeds))

        # put true coefs into storage (candidate terms are identical across seeds)
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
                rms_term = rms_true[module][0, :, candidate_terms_true_model.index(term)] if module in rms_true else 0.
                true_effects[index_par, par*it:par*(it+1), index_coefs_all+index_coef] = np.abs(true_coef_vals) * rms_term * module_present

            index_coefs_all += n_terms_module

        cells_datasets.append(signed_cells(
            true_coefs[index_par, par*it:par*(it+1)], true_effects[index_par, par*it:par*(it+1)],
            np.stack(fitted_coefs_seeds), np.stack(fitted_presence_seeds), np.stack(fitted_effects_seeds),
        ).assign(dataset=f'{par}p_{it}', n_participants=par, n_trials=n_trials_dataset))

        # store active mechanisms and number of active params per participant
        labels_mechanisms, n_params_mechanisms = get_active_mechanisms(mechanism_masks)
        active_mechanisms[index_par, par*it:par*(it+1)] = labels_mechanisms
        active_params[index_par, par*it:par*(it+1)] = n_params_mechanisms

# -------------------------------------------------------------------------------
# POST-PROCESSING: Compute classification metrics
# -------------------------------------------------------------------------------

# Active coefficients: nonzero in the ground truth; fitted_active comes from the presence vote (aggregate_seeds)
true_active = true_coefs != 0

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
precision = metrics_params['precision']
recall = metrics_params['recall']

# Metrics binned by ground-truth model type (active mechanisms), sorted by number of parameters
mechanism_types = sorted(
    {(n_params, label) for n_params, label in zip(active_params.ravel(), active_mechanisms.ravel()) if label is not None}
)
mechanism_labels = [label for _, label in mechanism_types]
mechanism_label_to_index = {label: index for index, label in enumerate(mechanism_labels)}
group_index_mechanisms = np.vectorize(lambda label: mechanism_label_to_index.get(label, -1), otypes=[int])(active_mechanisms)
counts_mechanisms = compute_confusion_counts(group_index_mechanisms, len(mechanism_labels))
metrics_mechanisms = compute_classification_metrics(counts_mechanisms)

os.makedirs(path_results, exist_ok=True)
np.savez(
    os.path.join(path_results, f'{figure_prefix}_metrics.npz'),
    participants=np.array(participants),
    mechanism_labels=np.array(mechanism_labels),
    **{f'params_{key}': value for key, value in metrics_params.items()},
    **{f'mechanisms_{key}': value for key, value in metrics_mechanisms.items()},
    true_coefs=true_coefs,
    fitted_coefs=fitted_coefs,
    active_mechanisms=active_mechanisms.astype(str),
)

# Signed participant x term cells per dataset (true / found, coefficients, effect sizes), read by run_recovery_by_target.py
cells = pd.concat(cells_datasets, ignore_index=True)
cells['term_id'] = cells['module'] + '|' + cells['term']
cells.to_csv(os.path.join(path_results, f'{figure_prefix}_cells.csv'), index=False)

# Classification metrics per dataset size and ground-truth model (one row each), the numbers behind the figures
mechanism_rows = []
for index_par, par in enumerate(participants):
    for index_mechanism, (n_params, label) in enumerate(mechanism_types):
        mechanism_rows.append({
            'n_participants': par,
            'ground_truth_model': label.replace('\n', ' + '),
            'n_parameters': n_params,
            'n_fits': int(counts_mechanisms['samples'][index_par, index_mechanism]),
            **{key: value[index_par, index_mechanism] for key, value in metrics_mechanisms.items()},
        })
pd.DataFrame(mechanism_rows).to_csv(os.path.join(path_results, f'{figure_prefix}_metrics_by_mechanism.csv'), index=False)

# -------------------------------------------------------------------------------
# PLOTTING: 2x2 Confusion Matrix Rates
# -------------------------------------------------------------------------------

import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

def strip_annotations(fig):
    """Remove titles, axis labels, tick labels and text inside the axes (heatmap numbers)."""
    for figure_label in (fig._suptitle, fig._supxlabel, fig._supylabel):
        if figure_label is not None:
            figure_label.set_text('')
    for ax in fig.axes:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        for text in list(ax.texts):
            text.remove()
        # secondary axes (e.g. module group headers) only carry tick labels
        for child_axis in ax.child_axes:
            child_axis.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

def finalize_figure(fig, name):
    """Save the annotated figure and a '_clean' copy without annotations to path_results, then show it."""
    fig.tight_layout()
    os.makedirs(path_results, exist_ok=True)
    fig.savefig(os.path.join(path_results, f'{figure_prefix}_{name}.png'), bbox_inches='tight')
    strip_annotations(fig)
    fig.savefig(os.path.join(path_results, f'{figure_prefix}_{name}_clean.png'), bbox_inches='tight')
    plt.show()

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
        axs[row, col].set_facecolor(EMPTY_COLOR)  # masked (NaN) tiles show the background
        sns.heatmap(
            confusion_matrices[row][col],
            annot=True,
            fmt='.2f',
            cmap=rate_cmap(),
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
finalize_figure(fig, 'confusion_rates')

# -------------------------------------------------------------------------------
# PLOTTING: Classification Metrics (precision, recall)
# -------------------------------------------------------------------------------

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.5),
                        gridspec_kw={'width_ratios': [10, 10, 1]})

for col, (title, metric) in enumerate((('Precision', precision), ('Recall', recall))):
    axs[col].set_facecolor(EMPTY_COLOR)  # masked (NaN) tiles show the background
    sns.heatmap(
        metric,
        annot=True,
        fmt='.2f',
        cmap=rate_cmap(),
        ax=axs[col],
        cbar=(col == 1),
        cbar_ax=axs[2] if col == 1 else None,
        xticklabels=x_labels,
        yticklabels=y_labels if col == 0 else [''] * len(participants),
        vmin=0,
        vmax=1,
        mask=np.isnan(metric),
    )
    axs[col].set_title(title, fontsize=12)
    axs[col].set_xlabel('Number of Active Parameters', fontsize=10)
    if col == 0:
        axs[col].set_ylabel('Number of Participants', fontsize=10)

plt.suptitle('Classification Metrics', fontsize=14)
finalize_figure(fig, 'classification_metrics')

# -------------------------------------------------------------------------------
# PLOTTING: Precision and recall per ground-truth model type (mechanisms on y-axis)
# -------------------------------------------------------------------------------

n_mechanism_types = len(mechanism_labels)
mechanism_n_samples = counts_mechanisms['samples'].sum(axis=0).astype(int)
y_labels_mechanisms = [
    f"{label.replace(chr(10), ' + ')} (k={n_params}, n={n_samples})"
    for (n_params, label), n_samples in zip(mechanism_types, mechanism_n_samples)
]

metrics_panels = (
    ('Precision', metrics_mechanisms['precision']),
    ('Recall', metrics_mechanisms['recall']),
)

fig, axs = plt.subplots(nrows=1, ncols=len(metrics_panels) + 1, figsize=(10, 1.5 + 0.45 * n_mechanism_types),
                        gridspec_kw={'width_ratios': [10] * len(metrics_panels) + [1]})

for col, (title, metric) in enumerate(metrics_panels):
    # (participant sizes, mechanisms) -> (mechanisms, participant sizes)
    metric = metric.T
    axs[col].set_facecolor(EMPTY_COLOR)  # masked (NaN) tiles show the background
    sns.heatmap(
        metric,
        annot=True,
        fmt='.2f',
        cmap=rate_cmap(),
        ax=axs[col],
        cbar=(col == len(metrics_panels) - 1),
        cbar_ax=axs[-1] if col == len(metrics_panels) - 1 else None,
        xticklabels=y_labels,
        yticklabels=y_labels_mechanisms if col == 0 else [''] * n_mechanism_types,
        vmin=0,
        vmax=1,
        mask=np.isnan(metric),
    )
    axs[col].set_title(title, fontsize=12)
    axs[col].set_xlabel('Number of Participants', fontsize=10)
    if col == 0:
        axs[col].set_ylabel('Ground-Truth Mechanisms', fontsize=10)
        axs[col].tick_params(axis='y', labelsize=7, rotation=0)

plt.suptitle('Classification Metrics by Ground-Truth Model', fontsize=14)
finalize_figure(fig, 'classification_metrics_by_mechanism')

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
    axs[row, 0].set_facecolor(EMPTY_COLOR)  # masked (NaN) tiles show the background
    sns.heatmap(
        rates,
        annot=True,
        fmt='.2f',
        cmap=rate_cmap(),
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
finalize_figure(fig, 'per_term_recovery')

# -------------------------------------------------------------------------------
# PLOTTING: Parameter Recovery Box Plots
# -------------------------------------------------------------------------------

# Find which terms have any non-zero true coefficients (active terms)
active_term_mask = np.any(true_active, axis=(0, 1))
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
                ax.scatter(true_v, fitted_v, alpha=0.4, s=8, color=MODEL_COLORS['SPICE-EQ'], zorder=3)
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
                    boxprops=dict(facecolor=MODEL_COLORS['SPICE-EQ'], alpha=0.6, linewidth=0.5),
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
        finalize_figure(fig, f'coefficients_{par}p')

print("Analysis complete.")

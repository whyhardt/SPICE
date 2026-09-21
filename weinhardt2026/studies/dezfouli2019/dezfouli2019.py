import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from spice import SpiceEstimator

# from spice.precoded.workingmemory import SpiceModel, CONFIG
# from spice.precoded.choice import SpiceModel, CONFIG
from spice_dezfouli2019 import SpiceModel, CONFIG

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import (
    GQLModel, EnvironmentDezfouli2019, get_dataset, generate_behavior,
)
from weinhardt2026.studies.dezfouli2019.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions
from weinhardt2026.analysis.analysis_coefficients_individuals import analysis_coefficients_individuals
from weinhardt2026.analysis.analysis_coefficient_dendrogram import analysis_coefficient_dendrogram
from weinhardt2026.analysis.analysis_coefficient_betas import analysis_coefficient_betas
from weinhardt2026.analysis.analysis_sparsity_hpscan import analysis_sparsity_hpscan, plot_hpscan_heatmaps
from weinhardt2026.analysis.analysis_morphing import analysis_morphing
from weinhardt2026.analysis.analysis_bestbic import (
    analysis_embedding_dynamics,
    analysis_metric_fingerprint,
    analysis_group_betas,
    analysis_metric_betas,
)
from weinhardt2026.utils.checkpoints import (
    load_estimator,
    select_lowest_bic_checkpoint,
    participant_label_map,
)
from weinhardt2026.utils.generation import generate_repeated


train_spice = True
train_benchmark = False
train_gru = False

run_hpscan = False
run_morphing = False
run_bestbic = False

generate_data = False
N_REPEATS = 100

path_data = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'
data_dir = 'weinhardt2026/studies/dezfouli2019/data'
output_dir = 'weinhardt2026/studies/dezfouli2019/results'
path_spice = 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019_new.pkl'
path_benchmark = 'weinhardt2026/studies/dezfouli2019/params/benchmark_dezfouli2019.pkl'
path_gru = 'weinhardt2026/studies/dezfouli2019/params/gru_dezfouli2019.pkl'
params_array_dir = 'weinhardt2026/studies/dezfouli2019/params_array'
figures_dir = 'weinhardt2026/studies/dezfouli2019/figures'
path_metrics = 'weinhardt2026/studies/dezfouli2019/results/behavioral_metrics_real.csv'

# LaTeX symbols for the workingmemory submodules, used when printing equations
LATEX_MODULE_SYMBOLS = {
    'value_reward_chosen': r'V^{r}_{\mathrm{ch}}',
    'value_reward_not_chosen': r'V^{r}_{\mathrm{un}}',
    'value_choice_chosen': r'V^{c}_{\mathrm{ch}}',
    'value_choice_not_chosen': r'V^{c}_{\mathrm{un}}',
}

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {info_dataset['n_participants']}")
print(f"Number of actions in dataset: {info_dataset['n_actions']}")

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=info_dataset['n_actions'],
    n_participants=info_dataset['n_participants'],
    kwargs_spice_class={'reward_binary': True},

    epochs=1000,
    warmup_steps=500,
    
    sindy_threshold_pruning=0.1,
    sindy_alpha=1e-4,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys)#, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(path_spice)

# -------------------------------------------------------------------------------------------
# GQL BENCHMARK MODEL (Dezfouli 2019)
# -------------------------------------------------------------------------------------------

benchmark = GQLModel(
    n_participants=info_dataset['n_participants'],
    batch_first=True,
)

if train_benchmark:
    optimizer = torch.optim.Adam(params=benchmark.parameters(), lr=0.01)
    benchmark = training(
        model=benchmark, optimizer=optimizer,
        dataset_train=dataset_train, dataset_test=dataset_test,
        epochs=1000, device=torch.device('cpu'),
    )
    torch.save(benchmark.state_dict(), path_benchmark)
else:
    benchmark.load_state_dict(torch.load(path_benchmark, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# GRU BENCHMARK MODEL
# -------------------------------------------------------------------------------------------


gru = GRUModel(
    n_actions=info_dataset['n_actions'],
    n_participants=info_dataset['n_participants'],
    additional_inputs=2,
    dropout=0.25,
    embedding_size=8,
    hidden_size=8,
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)
    gru = training(
        model=gru, optimizer=optimizer,
        dataset_train=dataset_train, dataset_test=dataset_test,
        epochs=1000,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
benchmark.eval()
gru.eval()

for pid in range(3):
    print(f"\nExample SPICE model (participant {pid}):")
    estimator.print_spice_model(participant_id=pid)

print("\n--- Model evaluation (train) ---")
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=benchmark.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
))

print("\n--- Model evaluation (test) ---")
print(analysis_model_evaluation(
    dataset=dataset_test,
    held_out=True,
    spice_model=estimator,
    benchmark_model=benchmark.to(torch.device('cpu')),
    gru_model=gru.to(torch.device('cpu')),
    output_dir=output_dir,
))

# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

if generate_data:

    estimator.use_sindy(False)
    ds_spice_rnn = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=estimator,
        dataset=dataset_train,
    )

    estimator.use_sindy(True)
    ds_spice = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=estimator,
        dataset=dataset_train,
    )

    ds_benchmark = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=benchmark,
        dataset=dataset_train,
    )

    ds_gru = generate_repeated(
        generate_behavior,
        n_repeats=N_REPEATS,
        model=gru,
        dataset=dataset_train,
    )

    # -------------------------------------------------------------------------------------------
    # ANALYSIS: GENERATIVE BEHAVIOR
    # -------------------------------------------------------------------------------------------

    analysis_generative_behavior(
        path_data_real=path_data,
        path_data_gru=ds_gru,
        path_data_benchmark=ds_benchmark,
        path_data_spice=ds_spice,
        path_data_spice_rnn=ds_spice_rnn,
        output_dir=output_dir,
    )

# -------------------------------------------------------------------------------------------
# ANALYSIS: INDIVIDUAL DIFFERENCES
# -------------------------------------------------------------------------------------------

analysis_coefficients_distributions(
    spice_model=estimator,
    output_dir=output_dir,
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: STRUCTURAL GROUP DIFFERENCES
# -------------------------------------------------------------------------------------------

analysis_coefficients_individuals(
    spice_model=estimator,
    path_data=path_data,
    analysis='disc',
    criterion='diag',
    reference='Control',
    output_dir=output_dir,
)


# -------------------------------------------------------------------------------------------
# ANALYSIS: COEFFICIENT CLUSTERING AND BETA EFFECTS
# -------------------------------------------------------------------------------------------

# Hierarchical clustering of the candidate terms of each submodule, so ties that
# act as one mechanism become visible; correlations run over co-present
# participants only, which keeps the parametric axis separate from the
# structural (presence) one.
analysis_coefficient_dendrogram(
    spice_model=estimator,
    output_dir=output_dir,
    prefix='dezfouli2019',
    n_bootstrap=500,
)

# Structural counterpart: which terms switch on together across participants
# (phi on presence). Merges are tested by permutation (FWER per module); the
# significant ones are the groups, and they are the leaves of the beta analysis.
analysis_coefficient_dendrogram(
    spice_model=estimator,
    output_dir=output_dir,
    prefix='dezfouli2019',
    mode='presence',
    n_permutations=10000,
)

# Structural beta effects: does the presence of a term — or of a group of terms
# that switch on together — relate to the participant's reward rate? The metric
# must not scale with trial count; the analysis checks and warns.
analysis_coefficient_betas(
    spice_model=estimator,
    data_path=path_data,
    output_dir=output_dir,
    criterion_col='reward',
    prefix='dezfouli2019',
)

# Same structural question against diagnosis, for every pair of groups.
analysis_coefficient_betas(
    spice_model=estimator,
    data_path=path_data,
    output_dir=output_dir,
    criterion_col='diag',
    criterion_type='discrete',
    comparisons=[('Depression', 'Control'), ('Bipolar', 'Control'), ('Bipolar', 'Depression')],
    prefix='dezfouli2019',
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: SPARSITY HYPERPARAMETER SCAN
# -------------------------------------------------------------------------------------------

if run_hpscan:
    df_hpscan = analysis_sparsity_hpscan(
        pkl_pattern=os.path.join(params_array_dir, 'spice_dezfouli2019_*.pkl'),
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=info_dataset['n_actions'],
        data_path=path_data,
        test_blocks=test_blocks,
        polynomial_degree=2,
        model_kwargs={'reward_binary': True},
    )
    df_hpscan.to_csv(os.path.join(output_dir, 'hpscan_results.csv'), index=False)
    plot_hpscan_heatmaps(df_hpscan, os.path.join(output_dir, 'hpscan_heatmaps.png'))


# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL MORPHING
# -------------------------------------------------------------------------------------------

# Morphs participant embeddings along the avg_reward direction, refits SINDy per
# ensemble member, and validates that generated behavior follows the axis.
if run_morphing:
    analysis_morphing(
        estimator=estimator,
        dataset=dataset_train,
        metric_values=pd.read_csv(path_metrics)['avg_reward'].values,
        output_dir=output_dir,
        environment=EnvironmentDezfouli2019(n_actions=info_dataset['n_actions'],
                                            n_participants=info_dataset['n_participants'] * 20,
                                            n_blocks=12),
        n_steps=20,
        save_dir=os.path.join('weinhardt2026/studies/dezfouli2019/params', 'morphing'),
        prefix='morphing',
    )

# -------------------------------------------------------------------------------------------
# ANALYSIS: DEEPER LOOK AT THE LOWEST-BIC CHECKPOINT
# -------------------------------------------------------------------------------------------

if run_bestbic:
    path_bestbic, row_bestbic = select_lowest_bic_checkpoint(
        hpscan_csv=os.path.join(output_dir, 'hpscan_results.csv'),
        params_dir=params_array_dir,
    )
    print(f"Lowest-BIC checkpoint: {os.path.basename(path_bestbic)} (BIC={row_bestbic['BIC']:.1f})")

    estimator_bestbic = load_estimator(
        model_path=path_bestbic,
        spice_class=SpiceModel,
        spice_config=CONFIG,
        n_actions=info_dataset['n_actions'],
        model_kwargs={'reward_binary': True},
    )
    label_map = participant_label_map(path_data, label_col='diag')

    figures_bestbic = os.path.join(figures_dir, 'bestbic')
    results_bestbic = os.path.join(output_dir, 'bestbic')
    os.makedirs(figures_bestbic, exist_ok=True)
    os.makedirs(results_bestbic, exist_ok=True)

    analysis_embedding_dynamics(
        estimator_bestbic, dataset_train, figures_bestbic, results_bestbic,
        label_map=label_map, module_symbols=LATEX_MODULE_SYMBOLS, n_select=3,
    )
    analysis_metric_fingerprint(
        estimator_bestbic, figures_bestbic, results_bestbic,
        metrics_csv=path_metrics, data_path=path_data, metric='avg_reward',
        label_map=label_map,
    )
    analysis_group_betas(
        estimator_bestbic, figures_bestbic, results_bestbic,
        data_path=path_data, criterion='diag', reference='Control',
    )
    analysis_metric_betas(
        estimator_bestbic, figures_bestbic, results_bestbic,
        data_path=path_data, metrics_csv=path_metrics, metric='avg_reward',
    )

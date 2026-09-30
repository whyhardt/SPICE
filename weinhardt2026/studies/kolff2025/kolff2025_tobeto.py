import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd

from spice import SpiceEstimator
from weinhardt2026.studies.kolff2025.spice_kolff2025_tobeto import CONFIG, SpiceModel, filter_own_acts
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025_tobeto import (
    get_dataset, ConditionalTobetoModel, PERSPECTIVE_DATA_PATH,
)
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions
from weinhardt2026.analysis.analysis_coefficients_individuals import analysis_coefficients_individuals
from weinhardt2026.analysis.analysis_coefficient_dendrogram import analysis_coefficient_dendrogram
from weinhardt2026.analysis.analysis_coefficient_betas import analysis_coefficient_betas
from weinhardt2026.figures.figure5 import plot_figure5


train_spice = True
train_gru = True
train_benchmark = True

# -------------------------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/kolff2025/data/kolff2025_categories.csv'
path_spice = 'weinhardt2026/studies/kolff2025/params/spice_kolff2025_tobeto.pkl'
path_gru = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025_tobeto.pkl'
output_dir = 'weinhardt2026/studies/kolff2025/results/tobeto'

Path(path_spice).parent.mkdir(parents=True, exist_ok=True)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

# test set = ~20% of dyads, held out entirely
dataset_train, dataset_test, info = get_dataset(path_data=path_data, test_fraction=0.2)
print(info)

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    n_reward_features=dataset_train.n_reward_features,

    embedding_size=8,
    epochs=1000,
    warmup_steps=500,
    ensemble_size=10,

    sindy_library_polynomial_degree=1,
    sindy_weight=1e-1,
    sindy_alpha=1e-3,
    sindy_threshold_pruning=5e-2,

    device=device,
    verbose=True,
    compiled_forward=True,
)

if train_spice:
    estimator.fit(
        data=dataset_train.xs,
        targets=dataset_train.ys,
        data_test=dataset_test.xs,
        target_test=dataset_test.ys,
    )
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(path_spice)

estimator.print_spice_model()

# -------------------------------------------------------------------------------------------
# BENCHMARKS
# -------------------------------------------------------------------------------------------

# lag-1 conditional-frequency table: P(own act next | own act, partner act, focal ape)
benchmark = ConditionalTobetoModel(n_participants=dataset_train.n_participants)
if train_benchmark:
    benchmark.fit(dataset_train)

gru = GRUModel(
    n_actions=dataset_train.n_actions,
    n_participants=dataset_train.n_participants,
    additional_inputs=dataset_train.n_additional_inputs,
    n_reward_features=0,
    embedding_size=4,
    hidden_size=8,
)

if train_gru:
    gru = training(
        model=gru,
        optimizer=torch.optim.Adam(gru.parameters(), lr=0.01),
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
gru.eval().to(torch.device('cpu'))
benchmark.eval().to(torch.device('cpu'))

print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
))

# held-out dyads
print(analysis_model_evaluation(
    dataset=dataset_test,
    held_out=True,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    output_dir=output_dir,
))

# restricted to events where the focal ape itself acts next
print(analysis_model_evaluation(
    dataset=dataset_test,
    held_out=True,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    trial_filter=filter_own_acts,
))

# Per-ape dominance rank, in the order csv_to_dataset assigns participant indices
# (order of first appearance in the CSV), used to colour the ensemble error-bar plots.
_df_rank = pd.read_csv(PERSPECTIVE_DATA_PATH)
rank_per_participant = (_df_rank.groupby('focal').rank_own.first()
                        .reindex(_df_rank['focal'].unique()).values)

analysis_coefficients_distributions(
    spice_model=estimator,
    output_dir=output_dir,
    participant_values=rank_per_participant,
    participant_label='rank_own (0 = dominant)',
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: BETA EFFECTS OF OWN DOMINANCE RANK
# -------------------------------------------------------------------------------------------
# Regresses every per-participant SINDy coefficient on the focal ape's own dominance rank.
# With the centred rank_diff in the library, the two rank questions separate: the bias
# terms (resting drive with a typical partner) carry "do dominant apes differ", the
# rank_diff terms carry "does an ape adjust to this partner's relative rank".

analysis_coefficients_individuals(
    spice_model=estimator,
    path_data=PERSPECTIVE_DATA_PATH,
    analysis='cont',
    criterion='rank_own',
    output_dir=output_dir,
    dataset_kwargs={
        'df_participant_id': 'focal',
        'df_block': 'interaction_id',
        'df_choice': 'choice',
        'df_feedback': None,
        'additional_inputs': CONFIG.additional_inputs,
    },
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: STABILITY ACROSS ENSEMBLE MEMBERS (figure 5 approach)
# -------------------------------------------------------------------------------------------
# Until separate stability runs exist, each of the 10 ensemble members of this fit is
# treated as a "run": per term, how consistently the members agree on its presence
# (averaged over apes), and each member's coefficient mean +/- SD across apes.

plot_figure5(
    stability_pkl_paths=[path_spice],
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_actions=dataset_train.n_actions,
    output_dir=str(Path(output_dir) / 'figure5_members'),
    polynomial_degree=1,
    per_member=True,
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: COEFFICIENT DENDROGRAMS
# -------------------------------------------------------------------------------------------
# Per module, clusters the terms by how their per-ape coefficients co-vary (coefficients)
# and by which terms are switched on together (presence), with permutation-tested merges.
# min_overlap is lowered from the default 30 because there are only 41 apes.

for mode in ('coefficients', 'presence'):
    analysis_coefficient_dendrogram(
        spice_model=estimator,
        output_dir=str(Path(output_dir) / 'dendrogram'),
        prefix='spice_kolff2025_tobeto',
        mode=mode,
        min_overlap=15,
        n_permutations=10000,
    )

# -------------------------------------------------------------------------------------------
# ANALYSIS: STRUCTURAL BETA EFFECTS OF OWN RANK ON PRESENCE-DENDROGRAM GROUPS
# -------------------------------------------------------------------------------------------
# Terms split by sign; groups = significantly co-occurring terms (permutation-tested);
# presence ~ z(rank_own), BH-corrected over all nodes. Unlike the per-term analysis above,
# this does not adjust for the number of events per ape.

analysis_coefficient_betas(
    data_path=PERSPECTIVE_DATA_PATH,
    output_dir=str(Path(output_dir) / 'betas'),
    spice_model=estimator,
    criterion_col='rank_own',
    participant_col='focal',
    prefix='spice_kolff2025_tobeto',
)

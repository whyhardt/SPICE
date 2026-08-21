import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd

from spice import SpiceEstimator
from weinhardt2026.studies.kolff2025.spice_kolff2025_groom import CONFIG, SpiceModel, filter_grooming
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025_groom import (
    get_dataset, ConditionalGroomingModel, PERSPECTIVE_DATA_PATH,
)
from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions
from weinhardt2026.analysis.analysis_coefficients_individuals import analysis_coefficients_individuals


train_spice = False
train_gru = True
train_benchmark = True

# -------------------------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/kolff2025/data/kolff2025.csv'
path_spice = 'weinhardt2026/studies/kolff2025/params/spice_kolff2025_groom.pkl'
path_gru = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025_groom.pkl'
output_dir = 'weinhardt2026/studies/kolff2025/results/groom'

Path(path_spice).parent.mkdir(parents=True, exist_ok=True)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

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

    embedding_size=4,
    epochs=1000,
    warmup_steps=500,
    ensemble_size=10,
    
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

# memoryless conditional-frequency table: P(outcome next | own act, received act, focal ape)
benchmark = ConditionalGroomingModel(n_participants=dataset_train.n_participants)
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

print(analysis_model_evaluation(
    dataset=dataset_test,
    held_out=True,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    output_dir=output_dir,
))

# restricted to events where grooming actually happens next
print(analysis_model_evaluation(
    dataset=dataset_test,
    held_out=True,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    trial_filter=filter_grooming,
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
# Regresses every per-participant SINDy coefficient on the focal ape's own dominance rank,
# yielding a beta weight per equation term: which parts of the grooming-expectation
# mechanism differ between low- and high-ranking apes.
#
# `rank_own` is normalized within community (the two hierarchies have different depth), and
# is constant per ape, so the per-participant coefficients merge against it unambiguously.
# Reads the perspective-encoded CSV that get_dataset() wrote, not the raw kolff2025.csv --
# the participant column there is `focal` and the target is the 3-way `outcome`.

analysis_coefficients_individuals(
    spice_model=estimator,
    path_data=PERSPECTIVE_DATA_PATH,
    analysis='cont',
    criterion='rank_own',
    output_dir=output_dir,
    dataset_kwargs={
        'df_participant_id': 'focal',
        'df_block': 'interaction_id',
        'df_choice': 'outcome',
        'df_feedback': None,
        'additional_inputs': CONFIG.additional_inputs,
    },
)

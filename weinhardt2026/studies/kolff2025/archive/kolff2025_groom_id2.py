import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd

from spice import SpiceEstimator
from weinhardt2026.studies.kolff2025.spice_kolff2025_groom_id2 import (
    CONFIG, SpiceModel, LOSS_FN, LOSS_FN_KWARGS, ESTIMATOR_KWARGS,
)
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025_groom_id2 import (
    get_dataset, evaluation_dataset, has_target, ConditionalGroomModel, ModelInputGRU, ID2_DATA_PATH,
)
from weinhardt2026.utils.benchmarking_gru import training
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions


train_spice = True
train_gru = True

path_data = 'weinhardt2026/studies/kolff2025/data/kolff2025_categories.csv'
path_spice = 'weinhardt2026/studies/kolff2025/params/spice_kolff2025_groom_id2.pkl'
path_gru = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025_groom_id2.pkl'
output_dir = 'weinhardt2026/studies/kolff2025/results/groom_id2'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

# writes the ID2-perspective csv used by run.py / slurm_jobs/grid_search.sh
dataset_train, dataset_test, info = get_dataset(path_data=path_data, test_fraction=0.2)
print({k: v for k, v in info.items() if k != 'test_blocks'})

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR (settings of the kolff2025_groom_id2 grid entry)
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_participants=dataset_train.n_participants,
    n_reward_features=0,
    loss_fn=LOSS_FN,
    loss_fn_kwargs=LOSS_FN_KWARGS,
    **ESTIMATOR_KWARGS,
    embedding_size=8,
    epochs=1000,
    warmup_steps=500,
    ensemble_size=10,
    sindy_library_polynomial_degree=1,
    sindy_weight=0.,
    sindy_alpha=5e-4,
    sindy_threshold_pruning=5e-2,
    gate_penalty=1e-2,
    feature_penalty=1e-4,
    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(path_spice)

for participant_id in range(3):
    print(f"\nExample SPICE model (participant {participant_id}):")
    estimator.print_spice_model(participant_id=participant_id)

# -------------------------------------------------------------------------------------------
# BENCHMARK (lag-1 lookup table on ID1's act, shrunk toward the pooled table)
# -------------------------------------------------------------------------------------------

benchmark = ConditionalGroomModel(n_participants=dataset_train.n_participants).fit(dataset_train)

# -------------------------------------------------------------------------------------------
# GRU (no participant embedding, same inputs as SPICE)
# -------------------------------------------------------------------------------------------

gru = ModelInputGRU()

if train_gru:
    gru = training(
        model=gru,
        optimizer=torch.optim.Adam(gru.parameters(), lr=0.01),
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=1000,
        loss_fn=LOSS_FN,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL EVALUATION (scored only where ID2 acts next)
# -------------------------------------------------------------------------------------------

estimator.eval()
gru.eval().to(torch.device('cpu'))
benchmark.eval()

print("\n--- Model evaluation (train) ---")
print(analysis_model_evaluation(
    dataset=evaluation_dataset(dataset_train),
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    trial_filter=has_target,
))

print("\n--- Model evaluation (test: held-out dyads) ---")
print(analysis_model_evaluation(
    dataset=evaluation_dataset(dataset_test),
    held_out=True,
    spice_model=estimator,
    benchmark_model=benchmark,
    gru_model=gru,
    trial_filter=has_target,
    output_dir=output_dir,
))

# -------------------------------------------------------------------------------------------
# ANALYSIS: INDIVIDUAL DIFFERENCES (ID2 apes, coloured by own rank)
# -------------------------------------------------------------------------------------------

_df = pd.read_csv(ID2_DATA_PATH)
rank_per_participant = (_df.groupby('participant').rank_own.first()
                        .reindex(_df['participant'].unique()).values)
analysis_coefficients_distributions(
    spice_model=estimator,
    output_dir=output_dir,
    participant_values=rank_per_participant,
    participant_label='rank_own (0 = dominant)',
)

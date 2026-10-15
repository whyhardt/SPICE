import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator

from weinhardt2026.studies.kolff2025.spice_kolff2025 import (
    SpiceModel, CONFIG, LOSS_FN, LOSS_FN_KWARGS, ESTIMATOR_KWARGS,
)
from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import (
    get_dataset, evaluation_dataset, has_target, LogisticRegressionModel, ModelInputGRU,
)
from weinhardt2026.utils.benchmarking_gru import training
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_generative_comparison import run_generative_analysis


train_spice = False
train_gru = False

generate_data = False

path_spice = 'weinhardt2026/studies/kolff2025/params/spice_kolff2025.pkl'
path_gru = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025.pkl'
output_dir = 'weinhardt2026/studies/kolff2025/results'
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

# Both perspectives of every interaction (benchmarking_kolff2025.build_dataframe); the test set
# is fold 0 of 5 folds of whole dyads, i.e. pairings the models were never fit on.
dataset_train, dataset_test, info_dataset = get_dataset(test_fold=0, verbose=True)

# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR (one group model for all apes)
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_class=SpiceModel,
    spice_config=CONFIG,
    n_participants=1,
    n_reward_features=0,
    loss_fn=LOSS_FN,
    loss_fn_kwargs=LOSS_FN_KWARGS,
    **ESTIMATOR_KWARGS,

    epochs=1000,
    warmup_steps=500,
    ensemble_size=10,
    embedding_size=8,
    dropout=0.1,
    # group model: gates are global constants, so their penalties are off; linear library
    gate_penalty=0.,
    feature_penalty=0.,
    sindy_weight=0.01,
    sindy_library_polynomial_degree=1,

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(path_spice)

print("\nSPICE group model:")
estimator.print_spice_model(participant_id=0)

# -------------------------------------------------------------------------------------------
# BENCHMARK: LAG-1 LOGISTIC REGRESSION ON THE CURRENT EVENT (group model)
# -------------------------------------------------------------------------------------------

benchmark = LogisticRegressionModel().fit(dataset_train)

# -------------------------------------------------------------------------------------------
# GRU (no participant embedding, same inputs as SPICE)
# -------------------------------------------------------------------------------------------

gru = ModelInputGRU()

if train_gru or not os.path.exists(path_gru):
    torch.manual_seed(0)
    gru = training(
        model=gru,
        optimizer=torch.optim.Adam(gru.parameters(), lr=0.01),
        dataset_train=dataset_train,
        epochs=1000,
        loss_fn=LOSS_FN,
        scheduler=True,
    )
    torch.save(gru.state_dict(), path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))

# -------------------------------------------------------------------------------------------
# ANALYSIS: MODEL EVALUATION (scored only where the participant acts next)
# -------------------------------------------------------------------------------------------

estimator.eval()
gru.eval().to(torch.device('cpu'))
benchmark.eval()

for split, dataset in (('train', dataset_train), ('test', dataset_test)):
    print(f"\n--- Model evaluation ({split}{': held-out dyads' if split == 'test' else ''}) ---")
    df_evaluation = analysis_model_evaluation(
        dataset=evaluation_dataset(dataset),
        spice_model=estimator,
        benchmark_model=benchmark,
        gru_model=gru,
        trial_filter=has_target,
        held_out=split == 'test',
        pooled=True,
    )
    print(df_evaluation)
    df_evaluation.to_csv(os.path.join(output_dir, f'model_evaluation_{split}.csv'))

# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------
# The models predict only whether the participant's next act is grooming, so the simulation is
# closed-loop on that decision; the partner's acts and the timing of the participant's acts are
# replayed (benchmarking_kolff2025.generate_behavior). Outputs in results/; also runnable on its own:
# python weinhardt2026/analysis/analysis_generative_comparison.py kolff2025

if generate_data:
    run_generative_analysis('kolff2025')   # runs per model: N_REPEATS in analysis_generative.py

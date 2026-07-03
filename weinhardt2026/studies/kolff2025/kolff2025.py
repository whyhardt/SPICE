import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator, csv_to_dataset, split_data_along_blockdim
from spice_kolff2025 import CONFIG, SpiceModel, cross_entropy_loss_mask_waiting

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation


# Set to False to reuse saved params instead of retraining.
train_spice = False
train_gru = False

# Practical full-pipeline settings. Increase SPICE_ENSEMBLE_SIZE for final runs.
SPICE_EPOCHS = 1000
GRU_EPOCHS = 1000
SPICE_ENSEMBLE_SIZE = 1
SINDY_WEIGHT = 0.
N_GENERATED_TRIALS = 20

# -------------------------------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------------------------------

path_data = 'weinhardt2026/studies/kolff2025/data/kolff2025_processed.csv'
path_spice = 'weinhardt2026/studies/kolff2025/params/spice_kolff2025.pkl'
path_gru = 'weinhardt2026/studies/kolff2025/params/gru_kolff2025.pkl'
data_dir = 'weinhardt2026/studies/kolff2025/data'
output_dir = 'weinhardt2026/studies/kolff2025/results'

for path in (path_spice, path_gru):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(output_dir).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

dataset = csv_to_dataset(
    file=path_data,
    df_participant_id='interaction_id',
    df_choice='SigAct_ID1',
    df_feedback=None,
    additional_inputs=['SigAct_ID2', 'ID1', 'ID2'],
)

# Replace participant id and experiment id columns in dataset with ID1 and ID2 columns.
# participant id -> ID1
# experiment id -> ID2
dataset.xs[..., -1] = dataset.xs[..., dataset.n_actions + 1].nan_to_num(0)
dataset.xs[..., -2] = dataset.xs[..., dataset.n_actions + 2].nan_to_num(0)
n_participants = dataset.xs[..., [-1, -2]].nan_to_num(0).max().int().item() + 1

test_blocks = (1,)
dataset_train, dataset_test = split_data_along_blockdim(dataset=dataset, test_blocks=test_blocks)


# -------------------------------------------------------------------------------------------
# SPICE ESTIMATOR
# -------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

estimator = SpiceEstimator(
    spice_config=CONFIG,
    spice_class=SpiceModel,
    n_actions=dataset_train.n_actions,
    n_participants=n_participants,
    n_experiments=n_participants,
    n_reward_features=dataset_train.n_reward_features,
    
    embedding_size=4,
    loss_fn=cross_entropy_loss_mask_waiting,
    epochs=SPICE_EPOCHS,
    warmup_steps=500,
    ensemble_size=SPICE_ENSEMBLE_SIZE,
    sindy_weight=SINDY_WEIGHT,
    
    device=device,
    verbose=True,
    compiled_forward=False,
)

if train_spice:
    estimator.fit(
        data=dataset_train.xs,
        targets=dataset_train.ys,
        data_test=dataset_test.xs if dataset_test is not None else None,
        target_test=dataset_test.ys if dataset_test is not None else None,
    )
    estimator.save_spice(path_spice)
else:
    estimator.load_spice(path_spice)


# -------------------------------------------------------------------------------------------
# GRU FOR BENCHMARKING
# -------------------------------------------------------------------------------------------

gru = GRUModel(
    n_actions=dataset_train.n_actions,
    additional_inputs=dataset_train.n_additional_inputs,
    n_reward_features=0,
    # for using vanilla GRU -> set both to 1 (not using participant embeddings)
    n_participants=n_participants,  # to set sender
    n_experiments=n_participants,   # to set receiver; each receiver is basically an experimental condition
)

if train_gru:
    optimizer = torch.optim.Adam(gru.parameters(), lr=0.01)

    gru = training(
        model=gru,
        optimizer=optimizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        epochs=GRU_EPOCHS,
        loss_fn=cross_entropy_loss_mask_waiting,
        scheduler=True,
    )

    torch.save(gru.state_dict(), path_gru)
    print("Trained GRU parameters saved to " + path_gru)
else:
    gru.load_state_dict(torch.load(path_gru, map_location='cpu'))


# -------------------------------------------------------------------------------------------
# ANALYSIS
# -------------------------------------------------------------------------------------------

estimator.eval()
gru.eval().to(torch.device('cpu'))

# General analysis: model evaluation (average trial likelihood)
print(analysis_model_evaluation(
    dataset=dataset_train,
    spice_model=estimator,
    gru_model=gru,
))

print(analysis_model_evaluation(
    dataset=dataset_test,
    spice_model=estimator,
    gru_model=gru,
    output_dir=output_dir,
))


# -------------------------------------------------------------------------------------------
# GENERATIVE BENCHMARKING
# -------------------------------------------------------------------------------------------

from weinhardt2026.studies.kolff2025.benchmarking_kolff2025 import generate_behavior
from weinhardt2026.studies.kolff2025.analysis_generative import analysis_generative_behavior

# Generate behavior for GRU
gru.eval().to(torch.device('cpu'))
dataset_gen_gru = generate_behavior(
    model=gru,
    dataset=dataset,
    n_trials=N_GENERATED_TRIALS,
    save_csv=f'{data_dir}/gen_gru.csv',
)

# Generate behavior for SPICE-RNN
estimator.eval()
estimator.model.to(torch.device('cpu'))
estimator.use_sindy(False)
dataset_gen_spice_rnn = generate_behavior(
    model=estimator,
    dataset=dataset,
    n_trials=N_GENERATED_TRIALS,
    save_csv=f'{data_dir}/gen_spice_rnn.csv',
)

# Generate behavior for SPICE-EQ
estimator.use_sindy(True)
dataset_gen_spice = generate_behavior(
    model=estimator,
    dataset=dataset,
    n_trials=N_GENERATED_TRIALS,
    save_csv=f'{data_dir}/gen_spice.csv',
)

# Generative analysis: compare real vs generated behavior
analysis_generative_behavior(
    dataset_real=dataset,
    dataset_gru=dataset_gen_gru,
    dataset_spice_rnn=dataset_gen_spice_rnn,
    dataset_spice=dataset_gen_spice,
    output_dir=output_dir,
)

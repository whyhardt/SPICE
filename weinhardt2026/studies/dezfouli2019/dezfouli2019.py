import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator

from spice.precoded.workingmemory import SpiceModel, CONFIG
# from spice.precoded.choice import SpiceModel, CONFIG
# from spice_dezfouli2019 import SpiceModel, CONFIG

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import GQLModel, get_dataset, generate_behavior
from weinhardt2026.studies.dezfouli2019.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions
from weinhardt2026.analysis.analysis_coefficients_individuals import analysis_coefficients_individuals
from weinhardt2026.utils.generation import generate_repeated

sindy_lambda_group=0.0001
sindy_lambda_loading=0.0001
sindy_lambda_concept=0.00001

train_spice = True
train_benchmark = False
train_gru = False

generate_data = False
N_REPEATS = 100

path_data = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'
data_dir = 'weinhardt2026/studies/dezfouli2019/data'
output_dir = 'weinhardt2026/studies/dezfouli2019/results'
path_spice = f'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019_choice_z{sindy_lambda_loading}_v{sindy_lambda_concept}_g{sindy_lambda_group}.pkl'
path_spice_compressed = 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019_compressed.pkl'
path_benchmark = 'weinhardt2026/studies/dezfouli2019/params/benchmark_dezfouli2019.pkl'
path_gru = 'weinhardt2026/studies/dezfouli2019/params/gru_dezfouli2019.pkl'

# -------------------------------------------------------------------------------------------
# DATALOADER
# -------------------------------------------------------------------------------------------

test_blocks = (3, 6, 9)

dataset_train, dataset_test, info_dataset = get_dataset(path_data=path_data, test_blocks=test_blocks, verbose=True)

print(f"Shape of dataset: {dataset_train.xs.shape}")
print(f"Number of participants: {info_dataset['n_participants']}")
print(f"Number of actions in dataset: {info_dataset['n_actions']}")

# remove long tail of trials to make training faster
from spice import SpiceDataset
dataset_train = SpiceDataset(dataset_train.xs[:, :100], dataset_train.ys[:, :100])

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
    
    sindy_weight=0.1,
    sindy_refit=True,
    ensemble_size=10,
    sindy_threshold_pruning=0.1,
    sindy_lambda_loading=sindy_lambda_loading,
    sindy_lambda_concept=sindy_lambda_concept,
    sindy_lambda_group=sindy_lambda_group,
    
    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    if estimator.epochs==0:
        estimator.load_spice(path_spice)    
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
# ANALYSIS: STRUCTURAL + MAGNITUDE GROUP DIFFERENCES (on the concept loadings)
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
# NOTE: the post-hoc coefficient-compression analysis that used to live here has been
# removed. Structure is now discovered during training as a concept factorization, so the
# quantities it produced come straight off the fitted model:
#   estimator.get_concepts()          -- the population-level concept dictionary
#   estimator.get_concept_loadings()  -- each participant's loadings on it
#   estimator.count_spice_parameters()-- loadings/participant and shared direction values
# The group-difference analysis above now runs on those loadings: structural differences are
# which concepts a participant's gates hold open, magnitude differences are how strongly
# they load on the concepts they have.
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# The three participants furthest apart in embedding space -- maximizing the
# minimum pairwise distance among all C(n_participants, 3) triplets -- as a
# genuinely diverse example set (rather than the arbitrary first 3 by index),
# saved in the compact mechanism-loading format.
# -------------------------------------------------------------------------------------------

embeddings = estimator.get_participant_embeddings()
emb_matrix = torch.stack([embeddings[pid] for pid in range(info_dataset['n_participants'])]).float()
pairwise_dist = torch.cdist(emb_matrix, emb_matrix)

best_triplet, best_min_dist = None, -1.0
n_p = info_dataset['n_participants']
for i in range(n_p):
    for j in range(i + 1, n_p):
        for k in range(j + 1, n_p):
            d = min(pairwise_dist[i, j].item(), pairwise_dist[i, k].item(), pairwise_dist[j, k].item())
            if d > best_min_dist:
                best_min_dist = d
                best_triplet = (i, j, k)

print(f"\nMost mutually distant participants (by embedding, min pairwise dist={best_min_dist:.3f}): {best_triplet}")

os.makedirs(os.path.join(output_dir, 'concepts'), exist_ok=True)
with open(os.path.join(output_dir, 'concepts', 'participant_equations_most_distinct.txt'), 'w') as f:
    f.write(
        "Concept loadings for the three participants furthest apart in\n"
        "embedding space (maximizing the minimum pairwise distance among all\n"
        f"triplets). Participants: {best_triplet}, min pairwise distance={best_min_dist:.3f}.\n\n"
    )
    for pid in best_triplet:
        f.write(f"--- participant {pid} ---\n")
        f.write(estimator.model.get_spice_model_string(participant_id=pid, experiment_id=0))
        f.write("\n\n")

print(f"Most-distinct-participant equations saved to: "
      f"{os.path.join(output_dir, 'concepts', 'participant_equations_most_distinct.txt')}")
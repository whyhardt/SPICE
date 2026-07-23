import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from spice import SpiceEstimator

# from spice.precoded.workingmemory import SpiceModel, CONFIG
from spice.precoded.choice import SpiceModel, CONFIG

from weinhardt2026.utils.benchmarking_gru import GRUModel, training
from weinhardt2026.studies.dezfouli2019.benchmarking_dezfouli2019 import GQLModel, get_dataset, generate_behavior
from weinhardt2026.studies.dezfouli2019.analysis_generative import analysis_generative_behavior
from weinhardt2026.analysis.analysis_model_evaluation import analysis_model_evaluation
from weinhardt2026.analysis.analysis_coefficients_distributions import analysis_coefficients_distributions
from weinhardt2026.analysis.analysis_coefficients_individuals import analysis_coefficients_individuals
from weinhardt2026.analysis.analysis_coefficient_compression import analysis_coefficient_compression
from weinhardt2026.analysis.analysis_mechanism_individuals import analysis_mechanism_individuals
from weinhardt2026.utils.generation import generate_repeated


train_spice = False
train_benchmark = False
train_gru = False

generate_data = False
N_REPEATS = 100

path_data = 'weinhardt2026/studies/dezfouli2019/data/dezfouli2019.csv'
data_dir = 'weinhardt2026/studies/dezfouli2019/data'
output_dir = 'weinhardt2026/studies/dezfouli2019/results'
path_spice = 'weinhardt2026/studies/dezfouli2019/params/spice_dezfouli2019_choice.pkl'
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

    device=device,
    verbose=True,
    save_path_spice=path_spice,
)

if train_spice:
    estimator.fit(dataset_train.xs, dataset_train.ys, dataset_test.xs, dataset_test.ys)
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
# ANALYSIS: COEFFICIENT COMPRESSION
#
# MODEL = U @ H (no population-mean offset held outside the loading-based
# parameter count -- see conversation with Daniel, 2026-07-21). Concise
# mechanism names below were assigned by hand after inspecting
# mechanisms.txt for K_per_module=8, alpha_W=0.002, alpha_H=0.0, center=False
# -- if the hyperparameter search picks a different setting on a rerun
# (different data, different sklearn version, etc.), this list will no
# longer line up and analysis_coefficient_compression will raise rather than
# silently mislabel; rerun the search, re-inspect mechanisms.txt, and
# reassign names in that case.
# -------------------------------------------------------------------------------------------

CONCISE_MECHANISM_NAMES = [
    # value_reward_chosen
    'RewC: immediate vs. recent reward',
    'RewC: baseline & nonlinear decay',
    'RewC: distant reward consistency',
    'RewC: immediate vs. recent reward (2)',
    'RewC: reward suppression + gating',
    'RewC: reward consistency (t, t-1)',
    'RewC: baseline + reward consistency',
    'RewC: reward history + consistency',
    # value_reward_not_chosen
    'RewNC: persistence + recent suppression',
    'RewNC: baseline & persistence',
    'RewNC: recent-reward gating',
    'RewNC: distant-reward gating',
    'RewNC: nonlinear self-suppression',
    'RewNC: recent-reward gating (2)',
    'RewNC: mid-lag reward gating',
    'RewNC: nonlinear mid-lag gating',
    # value_choice_chosen
    'ChC: persistence + t-2 suppression',
    'ChC: baseline + t-2 suppression',
    'ChC: strong persistence',
    'ChC: immediate perseveration (t-1)',
    'ChC: baseline level',
    'ChC: persistence + gated t-1',
    'ChC: choice consistency (t-1, t-3)',
    'ChC: baseline + t-2 suppression (2)',
    # value_choice_not_chosen
    'ChNC: choice[t-2] suppression',
    'ChNC: immediate choice-history (t-1)',
    'ChNC: baseline + consistency (t-1, t-2)',
    'ChNC: baseline + consistency (t-2, t-3)',
    'ChNC: choice contrast (t-1 vs t-2)',
    'ChNC: strong persistence',
    'ChNC: choice consistency (t-1, t-3)',
    'ChNC: baseline & persistence',
]

search_df, loadings_df, compressed_model = analysis_coefficient_compression(
    spice_model=estimator,
    dataset_train=dataset_train,
    dataset_test=dataset_test,
    output_dir=os.path.join(output_dir, 'compression'),
    mechanism_names_override=CONCISE_MECHANISM_NAMES,
)

# -------------------------------------------------------------------------------------------
# ANALYSIS: STRUCTURAL GROUP DIFFERENCES IN COMPRESSED MECHANISMS
# -------------------------------------------------------------------------------------------

analysis_mechanism_individuals(
    loadings_df=loadings_df,
    path_data=path_data,
    reference='Control',
    criterion='diag',
    output_dir=os.path.join(output_dir, 'compression'),
)

# -------------------------------------------------------------------------------------------
# FINAL MODEL: commit the compressed (denoised) coefficients as the model's
# actual SINDy coefficients and save as the paper's final SPICE-EQ model.
#
# Caveat: under MODEL = U @ H (center=False), report model complexity using
# the active mechanism count from the compression search
# (search_df.loc[chosen].n_active_mean, ~17/participant here), not
# `estimator.count_sindy_coefficients()` on the committed model -- that
# counts nonzero *terms* in the reconstructed (loadings @ dictionary) view,
# which is a different quantity than the true degrees of freedom (the
# loadings themselves) and need not match it.
# -------------------------------------------------------------------------------------------

compressed_model.commit(estimator)
estimator.save_spice(path_spice_compressed)

with open(os.path.join(output_dir, 'compression', 'final_model_equations_all_participants.txt'), 'w') as f:
    f.write(
        "Final (NMF-compressed) SPICE-EQ model -- per-participant equations in the\n"
        "standard SPICE format, using the denoised/compressed coefficients.\n"
        "True free parameters per participant: active mechanism count (see\n"
        "compression/hyperparameter_search.csv chosen row's n_active_mean), NOT the\n"
        "term count shown below.\n\n"
    )
    for pid in range(info_dataset['n_participants']):
        f.write(f"--- participant {pid} ---\n")
        f.write(estimator.model.get_spice_model_string(participant_id=pid, experiment_id=0))
        f.write("\n\n")

print(f"\nFinal compressed model saved to: {path_spice_compressed}")
print(f"Full per-participant equations saved to: "
      f"{os.path.join(output_dir, 'compression', 'final_model_equations_all_participants.txt')}")

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

with open(os.path.join(output_dir, 'compression', 'participant_equations_most_distinct.txt'), 'w') as f:
    f.write(
        "Compressed equations for the three participants furthest apart in\n"
        "embedding space (maximizing the minimum pairwise distance among all\n"
        f"triplets). Participants: {best_triplet}, min pairwise distance={best_min_dist:.3f}.\n\n"
    )
    for pid in best_triplet:
        f.write(f"--- participant {pid} ---\n")
        f.write(compressed_model.participant_string(pid))
        f.write("\n\n")

print(f"Most-distinct-participant equations saved to: "
      f"{os.path.join(output_dir, 'compression', 'participant_equations_most_distinct.txt')}")
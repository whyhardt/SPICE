import numpy as np
import torch
from sklearn.decomposition import NMF

from spice import csv_to_dataset, SpiceEstimator
from spice.precoded import workingmemory, workingmemory_rewardbinary

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from weinhardt2025.benchmarking.benchmarking_qlearning import QLearning


n_concepts = 6  # Start with fewer concepts (≈ n_features / 2)
use_real = True  # if True: load trained sindy coefs; else: load groundtruth RL parameters from data file and convert into sindy coefs (baseline parameter recovery)

path_data = 'weinhardt2025/data/synthetic/synthetic_256p_0_0.csv'
path_model = 'weinhardt2025/params/synthetic/spice_synthetic_256p_0_0.pkl'
rl_parameters = ['beta_reward', 'beta_choice', 'alpha_reward', 'alpha_penalty', 'alpha_choice', 'forget_rate']
dataset = csv_to_dataset(
    file=path_data,
    additional_inputs=rl_parameters,
)
n_participants = len(dataset.xs[:, 0, -1].unique())
n_experiments = len(dataset.xs[:, 0, -2].unique())
n_actions = dataset.ys.shape[-1]

if use_real:
    estimator = SpiceEstimator(
        rnn_class=workingmemory_rewardbinary.SpiceModel,
        spice_config=workingmemory_rewardbinary.CONFIG,
        n_actions=n_actions,
        n_participants=n_participants,
        n_experiments=n_experiments,
        sindy_library_polynomial_degree=2,
    )
    estimator.load_spice(path_model=path_model)
    model = estimator.rnn_model
else:
    # Use a synthetic Q-Learning model
    parameters = dataset.xs[:, 0, n_actions*2:-3]
    parameters_dict = {param: torch.zeros((n_participants, 1)) for param in rl_parameters}
    for participant in range(n_participants):
        participant_mask = dataset.xs[:, 0, -1] == participant
        participant_parameters = dataset.xs[participant_mask][0, 0, n_actions*2:-3]
        for index_param, param in enumerate(parameters_dict):
            parameters_dict[param][participant, 0] = participant_parameters[index_param]
            
    model = QLearning(
        n_actions=n_actions,
        n_participants=n_participants,
        **parameters_dict,
    )

# Collect only present terms across all modules (binary presence with sign)
present_terms = []       # list of (module, term_name, term_idx)
presence_pos = []        # list of binary arrays: term present AND positive
presence_neg = []        # list of binary arrays: term present AND negative

for module in model.get_modules():
    # most important handlers for SPICE
    term_presence = model.sindy_coefficients_presence[module]  # (n_participants, n_experiments, 1, n_terms)
    coefficients = model.sindy_coefficients[module]            # (n_participants, n_experiments, 1, n_terms)
    terms = model.sindy_candidate_terms[module]

    # Check which terms are present in at least one participant (first experiment, first ensemble)
    presence = term_presence[:, 0, 0, :]  # (n_participants, n_terms)
    coefs = coefficients[:, 0, 0, :]      # (n_participants, n_terms)
    presence_any = presence.any(dim=0)    # (n_terms,)

    for term_idx, term_name in enumerate(terms):
        if presence_any[term_idx]:
            present_terms.append((module, term_name, term_idx))
            # Binary: present AND positive
            pos = (presence[:, term_idx] & (coefs[:, term_idx] > 0)).float().numpy()
            # Binary: present AND negative
            neg = (presence[:, term_idx] & (coefs[:, term_idx] < 0)).float().numpy()
            presence_pos.append(pos)
            presence_neg.append(neg)

n_terms_present = len(present_terms)
print(f"Present terms: {n_terms_present} / {sum(len(model.sindy_candidate_terms[m]) for m in model.get_modules())}")

# Build binary presence matrix: (n_participants, n_terms_present * 2)
C_pos = np.stack(presence_pos, axis=1)  # (n_participants, n_terms)
C_neg = np.stack(presence_neg, axis=1)  # (n_participants, n_terms)
C_extended = np.hstack([C_pos, C_neg])

# Build candidate term names with module prefix
candidate_terms = [f"{module}:{term}" for module, term, _ in present_terms]
candidate_terms_extended = [f"+{t}" for t in candidate_terms] + [f"-{t}" for t in candidate_terms]

print(f"Unique presence patterns: {len(np.unique(C_extended, axis=0))}")

print(f"Extended coefficient matrix shape: {C_extended.shape}")

# Fit sparse NMF (adjust n_concepts if needed for initialization)
n_features = C_extended.shape[1]
n_components_actual = min(n_concepts, n_features, n_participants)
use_overcomplete = n_concepts > n_features

# Use fewer concepts if we have few features
if not use_overcomplete:
    n_concepts = n_components_actual

nmf = NMF(
    n_components=n_concepts,
    init='random',    # random init to avoid degenerate solutions
    alpha_W=0.0,      # no sparsity penalty initially - let data speak
    l1_ratio=1.0,
    max_iter=1000,
    random_state=42,
)

print(f"NMF: {n_concepts} concepts from {n_features} features, init={'random' if use_overcomplete else 'nndsvd'}")

W = nmf.fit_transform(C_extended)  # (n_participants, n_concepts)
H = nmf.components_                 # (n_concepts, n_terms_extended)

# Reconstruction error
reconstruction = W @ H
recon_error = np.linalg.norm(C_extended - reconstruction, 'fro')
print(f"Reconstruction error: {recon_error:.4f}")

# Sparsity statistics (use relative threshold based on W magnitude)
w_threshold = max(0.01, W.max() * 0.05)  # 5% of max or 0.01, whichever is larger
active_per_participant = (W > w_threshold).sum(axis=1)
print(f"W range: [{W.min():.4f}, {W.max():.4f}], threshold: {w_threshold:.4f}")
print(f"Active concepts per participant: {active_per_participant.mean():.1f} ± {active_per_participant.std():.1f}")


def interpret_concepts(H, candidate_terms, top_k=5, threshold=0.05):
    """Extract interpretable concept definitions from NMF components."""
    concepts = []

    for concept_idx in range(H.shape[0]):
        weights = H[concept_idx]

        # Skip near-empty concepts
        if weights.max() < threshold:
            continue

        # Get top contributing terms (by weight)
        top_indices = np.argsort(weights)[::-1][:top_k]

        terms = []
        for idx in top_indices:
            w = weights[idx]
            if w > threshold * weights.max():
                terms.append((candidate_terms[idx], w))

        if terms:
            concepts.append({
                'index': concept_idx,
                'terms': terms,
                'definition': " + ".join([f"{w:.2f}*{t}" for t, w in terms]),
            })

    return concepts


def get_participant_concepts(W, concepts, threshold=None):
    """Get dominant concepts for each participant."""
    if threshold is None:
        threshold = max(0.01, W.max() * 0.05)

    participant_concepts = []

    for p in range(W.shape[0]):
        active = []
        for c in concepts:
            usage = W[p, c['index']]
            if usage > threshold:
                active.append((c['index'], c['definition'], usage))

        # Sort by usage
        active.sort(key=lambda x: -x[2])
        participant_concepts.append(active)

    return participant_concepts


# Debug: show H matrix stats
print(f"\nH matrix shape: {H.shape}")
print(f"H row maxes: {H.max(axis=1)}")

# Interpret discovered concepts
print("\n" + "="*60)
print("DISCOVERED CONCEPTS")
print("="*60)

concepts = interpret_concepts(H, candidate_terms_extended, top_k=5, threshold=0.001)

for c in concepts:
    print(f"\nConcept {c['index']}:")
    print(f"  {c['definition']}")

# Show participant concept usage
print("\n" + "="*60)
print("PARTICIPANT CONCEPT USAGE (first 10)")
print("="*60)

participant_concepts = get_participant_concepts(W, concepts)

for p in range(min(10, n_participants)):
    print(f"\nParticipant {p}:")

    # Show ground truth params only for synthetic model
    if not use_real:
        true_params = {param: parameters_dict[param][p, 0].item() for param in rl_parameters}
        print(f"  True params: α_r={true_params['alpha_reward']:.2f}, "
              f"α_p={true_params['alpha_penalty']:.2f}, "
              f"γ={true_params['forget_rate']:.2f}, "
              f"β_c={true_params['beta_choice']:.2f}")

    if participant_concepts[p]:
        for idx, definition, usage in participant_concepts[p][:3]:
            print(f"  Concept {idx} ({usage:.2f}): {definition[:70]}...")
    else:
        print("  No active concepts")
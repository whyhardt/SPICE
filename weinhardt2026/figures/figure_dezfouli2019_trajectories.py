"""
Value trajectories + observations of three maximally distinct dezfouli2019 participants.

Participants: two of the triplet whose closest pair is furthest apart in embedding space
(SPICE indices 11 and 78; see select_max_min_triplet) around the most average participant
(closest to the embedding centroid; see select_most_average).
Distances are averaged over every seed and ensemble member of the grid-search runs
(each member's embedding space has its own orientation, so the embeddings themselves
cannot be averaged; their distance matrices can).

Model shown: the seed with the lowest training AIC (SPICE-EQ), and within it the
ensemble member with the highest training likelihood (SPICE-EQ). Both SPICE-RNN
(dashed) and SPICE-EQ (solid) trajectories are drawn.

Rows: reward, choice, P(choice), and one row per latent state (value of item 1,
i.e. the option whose choice is plotted). Each participant's longest training block is shown. Values and probabilities at trial t are
those the model holds *before* the choice at t.

Usage:
    python weinhardt2026/figures/figure_dezfouli2019_trajectories.py [--participants 11 19 78] [--n_trials 60]
"""

import argparse
import itertools
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'weinhardt2026', 'figures'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import figure_model_evaluation as evaluation

STUDY = 'dezfouli2019'
OUTPUT_DIR = os.path.join(ROOT, 'weinhardt2026', 'figures', 'dezfouli2019_trajectories')
ITEM = 1  # plotted option

STATE_LABELS = {
    'value_wm_reward': 'Working-memory\nreward value',
    'value_reward': 'Reward value',
    'value_choice': 'Choice value',
}
COLOR_OBSERVED = {'reward': '#e0a830', 'choice': '#c44e52'}
COLOR_EQ, COLOR_RNN = '#d9868a', '#a83a40'


# ── Selection ───────────────────────────────────────────────────────

def _seed(path):
    return int(path.rsplit('_rs', 1)[1].split('.')[0])


def select_max_min_triplet(checkpoints):
    """Triplet maximizing its smallest pairwise distance in the seed/member-averaged distance matrix."""
    distances = []
    for path in checkpoints:
        embedding = torch.load(path, map_location='cpu')['model']['participant_embedding.weight'].numpy()  # (E, P, D)
        distance = np.mean([squareform(pdist(member)) for member in embedding], axis=0)
        distances.append(distance / distance[np.triu_indices_from(distance, 1)].mean())
    distance = np.mean(distances, axis=0)

    def min_pairwise(triplet):
        return min(distance[a, b] for a, b in itertools.combinations(triplet, 2))

    return list(max(itertools.combinations(range(distance.shape[0]), 3), key=min_pairwise))


def select_most_average(checkpoints):
    """Participant closest to the embedding centroid, averaged over every seed and ensemble member."""
    distances = []
    for path in checkpoints:
        embedding = torch.load(path, map_location='cpu')['model']['participant_embedding.weight'].numpy()  # (E, P, D)
        for member in embedding:
            distance = np.linalg.norm(member - member.mean(axis=0), axis=1)
            distances.append(distance / distance.mean())
    return int(np.argmin(np.mean(distances, axis=0)))


def select_seed_by_train_aic(checkpoints):
    cache = os.path.join(evaluation.OUTPUT_DIR, f'model_evaluation_{STUDY}.csv')
    results = pd.read_csv(cache)
    spice_eq = results[results['model'] == 'SPICE-EQ'].set_index('seed')['aic']
    best_seed = int(spice_eq.idxmin())
    path = next(p for p in checkpoints if _seed(p) == best_seed)
    return path, spice_eq


@torch.no_grad()
def select_member_by_train_likelihood(estimator, dataset_train):
    estimator.eval(use_sindy=True)
    estimator.model.init_state(batch_size=dataset_train.xs.shape[0])
    logits, _ = estimator(dataset_train.xs.to(estimator.device))  # (E, B, T, W, A)
    likelihoods = []
    for member_logits in logits.cpu():
        nll, n = evaluation.discrete_session_nll(member_logits, dataset_train)
        likelihoods.append(torch.exp(-nll.sum() / n.sum()).item())
    return int(np.argmax(likelihoods)), likelihoods


# ── Trajectories ────────────────────────────────────────────────────

def longest_train_session(dataset, participant):
    sessions = torch.where(dataset.xs[:, 0, 0, -1].long() == participant)[0]
    lengths = (~torch.isnan(dataset.xs[sessions, :, 0, 0])).sum(dim=1)
    session = sessions[lengths.argmax()]
    return dataset.xs[session:session + 1]


@torch.no_grad()
def run_trajectories(estimator, xs_session, member, use_sindy):
    """Per-trial P(item), latent states and observations of one ensemble member."""
    model = estimator.model
    estimator.eval(use_sindy=use_sindy)
    n_actions = model.n_actions
    n_trials = int((~torch.isnan(xs_session[0, :, 0, 0])).sum())
    xs_session = xs_session.to(estimator.device)

    state = model.init_state(batch_size=1)
    states = {key: [] for key in STATE_LABELS}
    probability = []
    for trial in range(n_trials):
        # quantities before the choice at this trial
        logits = sum(state[key] for key in model.spice_config.states_in_logit)[0, member, 0]
        probability.append(torch.softmax(logits, dim=-1)[ITEM].item())
        for key in STATE_LABELS:
            states[key].append(state[key][0, member, 0, ITEM].item())
        _, state = model(xs_session[:, trial:trial + 1], {key: value.clone() for key, value in state.items()})

    observed = xs_session[0, :n_trials, 0].cpu()
    choice = observed[:, :n_actions].argmax(dim=-1).numpy()
    reward = observed[torch.arange(n_trials), n_actions + torch.as_tensor(choice)].numpy()
    return dict(probability=np.array(probability), choice=(choice == ITEM).astype(float),
                reward=reward, **{key: np.array(values) for key, values in states.items()})


# ── Equations ───────────────────────────────────────────────────────

def format_equations(estimator, participant, member):
    model = estimator.model
    coefficients = model.get_sindy_coefficients()
    lines = []
    for module in model.get_modules():
        values = coefficients[module][member, participant, 0]
        values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
        terms = model.get_candidate_terms(key_module=module)
        parts = []
        for term, value in zip(terms, values):
            value = value + 1 if term == module else value  # state term is stored as deviation from identity
            if value == 0:
                continue
            name = f'{module}[t]' if term == module else term
            parts.append(f'{value:+.3f}' if term == '1' else f'{value:+.3f}·{name}')
        lines.append(f'{module}[t+1] = ' + (' '.join(parts) if parts else '0'))
    return lines


# ── Plot ────────────────────────────────────────────────────────────

def plot_trajectories(trajectories, participant_labels, n_trials, output_path):
    rows = [('reward', 'Reward $r$'), ('choice', 'Choice $c$'), ('probability', 'Choice\nprobability $p(c)$')] \
        + list(STATE_LABELS.items())
    height_ratios = [0.45, 0.45] + [1] * (len(rows) - 2)
    n_cols = len(trajectories)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.3 * n_cols + 1, 1.0 * sum(height_ratios) + 1.2),
                             sharex='col', squeeze=False, gridspec_kw={'height_ratios': height_ratios})

    for col, (trajectory, label) in enumerate(zip(trajectories, participant_labels)):
        trials = np.arange(1, min(n_trials, len(trajectory['eq']['choice'])) + 1)
        T = len(trials)
        for row, (key, ylabel) in enumerate(rows):
            ax = axes[row, col]
            if key in ('reward', 'choice'):
                ax.step(trials, trajectory['eq'][key][:T], where='mid', color=COLOR_OBSERVED[key], linewidth=1.4)
                ax.set_ylim(-0.2, 1.2)
            else:
                ax.plot(trials, trajectory['eq'][key][:T], color=COLOR_EQ, linewidth=2.6, label='SPICE-EQ')
                ax.plot(trials, trajectory['rnn'][key][:T], color=COLOR_RNN, linewidth=1.2, linestyle='--', label='SPICE-RNN')
                if key == 'probability':
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_yticks([0, 1])
                else:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.tick_params(axis='y', labelsize=7)
            if key in ('reward', 'choice'):
                ax.set_yticks([])
            ax.grid(axis='y', color='#eeeeee', linewidth=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10, rotation=0, ha='right', va='center', labelpad=8)
        axes[-1, col].set_xlabel('Trial $t$', fontsize=10)

    handles, labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(f'{output_path}.pdf', bbox_inches='tight')
    fig.savefig(f'{output_path}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {output_path}.pdf/.png')


# ── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--participants', type=int, nargs=3, default=None, help='SPICE participant indices (default: 11, most average participant, 78)')
    parser.add_argument('--n_trials', type=int, default=60)
    args = parser.parse_args()

    spice_module, dataset_train, _ = evaluation.load_study(STUDY)
    checkpoints = evaluation.STUDIES[STUDY]['checkpoints']()

    if args.participants is None:
        # the two most distinct participants of the max-min triplet (11, 78) around the most average one
        extremes = [11, 78]
        participants = [extremes[0], select_most_average(checkpoints), extremes[1]]
    else:
        participants = args.participants
    path, aic_per_seed = select_seed_by_train_aic(checkpoints)
    print(f'Participants: {participants}')
    print(f'Seed by lowest train AIC: {os.path.basename(path)} (AIC {aic_per_seed.min():.1f})')

    estimator = evaluation.load_spice(path, spice_module, dataset_train)
    member, likelihoods = select_member_by_train_likelihood(estimator, dataset_train)
    print(f'Ensemble member by highest train likelihood: {member} '
          f'({likelihoods[member]:.4f}; members {min(likelihoods):.4f}–{max(likelihoods):.4f})')

    raw = pd.read_csv(evaluation._study_path(STUDY, 'data', f'{STUDY}.csv'))
    participant_ids = list(raw['participant'].unique())

    trajectories, labels, report = [], [], []
    for index, participant in enumerate(participants):
        xs_session = longest_train_session(dataset_train, participant)
        trajectories.append({
            'rnn': run_trajectories(estimator, xs_session, member, use_sindy=False),
            'eq': run_trajectories(estimator, xs_session, member, use_sindy=True),
        })
        labels.append(f'Participant {index + 1}')
        report.append(f'Participant {index + 1}: SPICE index {participant}, ID {participant_ids[participant]}')
        report.extend('  ' + line for line in format_equations(estimator, participant, member))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    header = [f'checkpoint: {os.path.basename(path)}', f'ensemble member: {member}', '']
    with open(os.path.join(OUTPUT_DIR, 'equations.txt'), 'w') as file:
        file.write('\n'.join(header + report) + '\n')
    print('\n'.join(report))

    plot_trajectories(trajectories, labels, args.n_trials, os.path.join(OUTPUT_DIR, 'figure_dezfouli2019_trajectories'))
